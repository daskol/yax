import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from yax import (
    Equation, Mox, mox as make_mox, mtree_query as query, mtree_sub as sub,
    update_in_trees_leaf)


class ResBlock(nn.Module):
    @nn.compact
    def __call__(self, xs):
        return xs + nn.Dense(10)(xs)


@pytest.fixture
def res_block():
    model = ResBlock()
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


def test_update_in_tree_leaf(dummy: Mox):
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
    update_in_trees_leaf(parents, 1, act_expr)

    assert all(p.is_ephemeral for p in parents), \
        'All parents must be epheneral.'

    for lhs, rhs in zip((p.inputs for p in parents), inputs):
        assert len(lhs) == len(rhs) + 1, 'Missing auxiliary input.'
        assert lhs[:-1] == rhs, 'No original symbols.'

    for lhs, rhs in zip((p.in_tree for p in parents), in_trees):
        lhs_args, lhs_kwargs = lhs.children()
        rhs_args, rhs_kwargs = rhs.children()
        assert lhs_kwargs == rhs_kwargs, 'Keyword arguments has been changed.'
        lhs_args = lhs_args.children()
        rhs_args = rhs_args.children()
        assert len(lhs_args) == len(rhs_args) + 1, 'Missing auxiliary input.'
        assert lhs_args[:-1] == rhs_args, 'Original subtree is not preserved.'


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
