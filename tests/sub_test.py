# Copyright 2024 Daniel Bershatsky
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Type

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from yax import (
    Equation, Mox, eval_mox, make_mox, query, sub, update_in_trees,
    update_var_trees)


@dataclass(slots=True)
class ModelState:
    mox: Mox
    model: nn.Module  # Modified model.
    params: dict[str, Any]
    batch: jax.Array


class Identity(nn.Module):
    features: int = 10

    def __call__(self, xs):
        # TODO(@daskol): Fix tracing or evaluation: duplicated tracer/variable
        # results in exception on `write` to `env during evaluation.
        return jnp.array(xs)


class ResModule(nn.Module):
    features: int = 10
    inner_ty: Type[nn.Module] = nn.Dense

    @nn.compact
    def __call__(self, xs):
        return xs + self.inner_ty(self.features)(xs)


@pytest.fixture
def res_block():
    model = ResModule(inner_ty=Identity)
    batch = jnp.ones((1, 10))
    params = jax.jit(model.init)(jax.random.PRNGKey(42), batch)
    mox = make_mox(model.apply)(params, batch)
    modified_model = ResModule(inner_ty=nn.Dense)
    del model
    yield ModelState(mox, modified_model, params, batch)


@pytest.fixture
def mlp():
    model = nn.Sequential([nn.Dense(4), nn.relu, nn.Dense(2)])
    batch = jnp.ones((1, 4))
    params = jax.jit(model.init)(jax.random.PRNGKey(42), batch)
    mox = make_mox(model.apply)(params, batch)
    modified_model = nn.Sequential([nn.Dense(4), nn.gelu, nn.Dense(2)])
    del model
    yield ModelState(mox, modified_model, params, batch)


class NestedModule(nn.Module):
    @nn.compact
    def __call__(self, xs):
        ys = nn.Dense(4)(xs)
        return nn.relu(ys)


class ParentModule(nn.Module):
    @nn.compact
    def __call__(self, xs):
        return xs + NestedModule()(xs)


@pytest.fixture
def dummy():
    model = ParentModule()
    batch = jnp.ones((1, 4))
    params = jax.jit(model.init)(jax.random.PRNGKey(42), batch)
    mox = make_mox(model.apply)(params, batch)
    del model, batch, params
    yield mox


def test_update_in_trees(dummy: Mox):
    """Test substitution with new `Equation`."""
    relu_expr, *_ = query('//[@primitive="pjit"][@name="relu"]', dummy)
    assert isinstance(relu_expr, Equation), 'Equation for relu is expected.'

    def act(xs, ys):
        return nn.relu(xs + ys)

    # Build module expression for test activation function `act`. Then rewire
    # inputs and outputs (the first inputs and all outputs are the same).
    act_mox = make_mox(jax.jit(act))(jnp.ones((1, 4)), jnp.ones((1, 4)))
    act_expr = act_mox.children[0]
    act_expr.inputs = relu_expr.inputs[:1] + act_expr.inputs[1:]
    act_expr.outputs = act_expr.outputs

    # Back up all `in_tree` and input symbols.
    parents: tuple[Mox] = (dummy, dummy.children[0],
                           dummy.children[0].children[0])
    in_trees = [p.in_tree for p in parents]
    inputs = [p.inputs for p in parents]

    # Update `in_tree` and `inputs` of all parents: one extra input is added.
    update_in_trees(parents, 1, act_expr)

    assert all(p.is_ephemeral for p in parents), \
        'All parents must be epheneral.'

    for lhs, rhs in zip((p.inputs for p in parents), inputs):
        assert len(lhs) == len(rhs) + 1, 'Missing auxiliary input.'
        assert lhs[:-1] == rhs, 'No original symbols.'

    for lhs, rhs in zip((p.in_tree for p in parents), in_trees):
        lhs_args, lhs_kwargs = lhs.children()
        rhs_args, rhs_kwargs = rhs.children()
        assert lhs_kwargs == rhs_kwargs, 'Keyword arguments have been changed.'
        lhs_args = lhs_args.children()
        rhs_args = rhs_args.children()
        assert len(lhs_args) == len(rhs_args) + 1, 'Missing auxiliary input.'
        assert lhs_args[:-1] == rhs_args, 'Original subtree is not preserved.'


def test_update_var_trees(dummy: Mox):
    """Test recursive tree update due to new `Mox`."""
    dense_mox, *_ = query('//[@primitive="module_call"][@type="Dense"]', dummy)
    assert isinstance(dense_mox, Mox), 'Module expression is expected.'

    # Build module expression for replacement module and substitute some inputs
    # and all outputs.
    module = ResModule(4)
    params = jax.jit(module.init)(jax.random.PRNGKey(3705), jnp.ones((1, 4)))
    expr = make_mox(module.apply)(params, jnp.ones((1, 4)))
    mox, *_ = expr.children
    mox.inputs = mox.inputs[:-1] + dense_mox.inputs[-1:]
    mox.outputs = dense_mox.outputs
    _, empty_treedef = jax.tree.flatten(())
    assert mox.var_tree != empty_treedef, 'Param tree is empty.'

    # Back up all `in_tree` and input symbols.
    parents: tuple[Mox] = (dummy, dummy.children[0],
                           dummy.children[0].children[0])
    inputs = [p.inputs for p in parents]
    in_trees = [p.in_tree for p in parents]
    var_trees = [p.var_tree for p in parents]

    # Update all `in_tree` and `var_tree` recursively.
    update_var_trees(parents, 0, mox)

    assert all(p.is_ephemeral for p in parents), \
        'All parents must be epheneral.'

    # TODO(@daskol): Check in-trees and var-trees.
    # for i, pair in enumerate(zip((p.var_tree for p in parents), var_trees)):
    for lhs, rhs in zip((p.inputs for p in parents), inputs):
        assert len(lhs) == len(rhs), 'Number of inputs must be the same.'
        assert lhs[-1:] == rhs[-1:], 'Module argument inputs must be the same.'

    def remove_collection(col: dict[str, Any], path: tuple[str, ...]):
        col = col['params']
        for key in path[1:-1]:
            col = col[key]
        col.pop(path[-1])

    # Verify all non-root internal nodes (parents).
    lhs_path = ('params', 'NestedModule_0', 'ResModule_0')
    rhs_path = ('params', 'NestedModule_0', 'Dense_0')
    for i, (parent, treedef) in enumerate(zip(parents[1:], var_trees[1:])):
        num_params = parent.var_tree.num_leaves
        flat_params = parent.inputs[:num_params]
        lhs_params = parent.var_tree.unflatten(flat_params)
        rhs_params = treedef.unflatten(flat_params)

        remove_collection(lhs_params, ('params', ) + lhs_path[i + 1:])
        remove_collection(rhs_params, ('params', ) + rhs_path[i + 1:])
        assert lhs_params == rhs_params, 'The rest of the trees are the same.'

        _, lhs_tree = jax.tree.flatten(lhs_params)
        _, rhs_tree = jax.tree.flatten(rhs_params)
        assert lhs_tree == rhs_tree, 'Params tree are mismatched'

    # Verify consistency of the root node.
    root, *_ = parents
    (lhs_params, *_), _ = root.in_tree.unflatten(root.inputs)
    syms, var_tree = jax.tree_flatten(params)
    assert var_tree != var_trees[0], 'Params tree must differs.'

    args_tree, _ = in_trees[0].children()
    params_tree, *_ = args_tree.children()
    rhs_params = params_tree.unflatten(syms)

    remove_collection(lhs_params, lhs_path)
    remove_collection(rhs_params, rhs_path)
    _, lhs_tree = jax.tree.flatten(lhs_params)
    _, rhs_tree = jax.tree.flatten(rhs_params)
    assert lhs_tree == rhs_tree, 'Params tree are mismatched'


def test_sub_equation(mlp: ModelState):
    gelu_mox = make_mox(jax.jit(jax.nn.gelu))(mlp.batch)
    gelu_expr = gelu_mox.children[0]
    assert isinstance(gelu_expr, Equation)
    mox = sub('//[@primitive="pjit"][@name="relu"]', gelu_expr, mlp.mox)

    actual = eval_mox(mox, mlp.params, mlp.batch)
    desired = mlp.model.apply(mlp.params, mlp.batch)
    assert_allclose(actual, desired)


def test_sub_mox(res_block: ModelState):
    # TODO(@daskol): Bug in multiple predicates evaluation.
    # nodes = query('//[@primitive="module_call"][@name="layers_0"]', mlp)
    # assert len(nodes) == 1
    module = nn.Dense(10)
    # Firstly, param dictionary must be updated.
    params = jax.jit(module.init)(jax.random.PRNGKey(3705), res_block.batch)
    variables = {**res_block.params}
    variables['params'] = {'Dense_0': params['params']}
    # Secondly, module expression for substitution must be built.
    inner_expr = make_mox(module.apply)(params, res_block.batch)
    inner_mox = inner_expr.children[0]
    assert isinstance(inner_mox, Mox)
    # Thirdly, original MoX can be updated.
    mox = sub('//[@primitive="module_call"][@name="Identity_0"]', inner_mox,
              res_block.mox)

    actual = eval_mox(mox, variables, res_block.batch)
    desired = res_block.model.apply(variables, res_block.batch)
    assert_allclose(actual, desired)
