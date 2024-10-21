from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from yax import Mox, mox as make_mox, mtree_eval as eval_mox

broken_lifting = pytest.mark.xfail(reason='broken lifting')


class Model(nn.Module):
    @nn.compact
    def __call__(self, xs):
        ys = nn.Dense(2)(xs)
        zs = nn.relu(ys)
        return nn.Dense(2)(zs)


class TestProcessCustomJVPCall:

    @pytest.mark.parametrize('apply', [
        pytest.param(lambda x: x, id='no grad'),
        pytest.param(jax.grad, id='grad', marks=broken_lifting),
    ])
    def test_relu(self, apply: Callable[..., Any]):
        xs = jnp.array(1.0)
        fn = apply(jax.nn.relu)
        mox = make_mox(fn)(xs)
        actual = eval_mox(mox, xs)
        desired = fn(xs)
        assert_allclose(actual, desired)

    @pytest.mark.parametrize('fn', [
        pytest.param(lambda x: x, id='no grad'),
        pytest.param(jax.grad, id='grad'),
    ])
    def test_model_mlp(self, fn: Callable[..., Any]):
        batch = jnp.ones((1, 3))
        key = jax.random.PRNGKey(42)

        # 1. Build and intialize model.
        model = Model()
        key, subkey = jax.random.split(key)
        params = jax.jit(model.init)(subkey, batch)

        # 2. Build a module expression.
        mox: Mox = make_mox(model.apply)(params, batch)
        actual = eval_mox(mox, params, batch)
        desired = model.apply(params, batch)
        assert_allclose(actual, desired)
