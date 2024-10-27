import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from yax import Equation, Mox, mox as make_mox, mtree_sub as sub


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
