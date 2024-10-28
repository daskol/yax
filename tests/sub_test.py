from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from yax import (
    Equation, Mox, mox as make_mox, mtree_query as query, mtree_sub as sub,
    update_in_trees, update_var_trees)


class ResModule(nn.Module):
    features: int = 10

    @nn.compact
    def __call__(self, xs):
        return xs + nn.Dense(self.features)(xs)


@pytest.fixture
def res_block():
    model = ResModule()
    batch = jnp.ones((1, 10))
    params = jax.jit(model.init)(jax.random.PRNGKey(42), batch)
    mox = make_mox(model.apply)(params, batch)
    del model, batch, params
    yield mox


@pytest.fixture
def mlp():
    model = nn.Sequential([nn.Dense(4), nn.relu, nn.Dense(2)])
    batch = jnp.ones((1, 4))
    params = jax.jit(model.init)(jax.random.PRNGKey(42), batch)
    mox = make_mox(model.apply)(params, batch)
    del model, batch, params
    yield mox


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


def test_sub_gelu(mlp: Mox):
    gelu_mox = make_mox(jax.jit(jax.nn.gelu))(jnp.ones((1, 4)))
    gelu_expr = gelu_mox.children[0]
    assert isinstance(gelu_expr, Equation)
    mod = sub('//[@primitive="pjit"][@name="relu"]', gelu_expr, mlp)
    print('MODIFIED MOX')
    print(mod)


def test_sub_mlp(res_block: Mox):
    # TODO(@daskol): Bug in multiple predicates evaluation.
    # nodes = query('//[@primitive="module_call"][@name="layers_0"]', mlp)
    # assert len(nodes) == 1
    module = nn.Dense(4)
    params = jax.jit(module.init)(jax.random.PRNGKey(3705), jnp.ones((1, 4)))

    mox = make_mox(module.apply)(params, jnp.ones((1, 4)))
    assert isinstance(mox, Mox)

    mod = sub('//[@primitive="module_call"][@name="layers_0"]', mox, mlp)
    print('MODIFIED MOX')
    print(mod)
