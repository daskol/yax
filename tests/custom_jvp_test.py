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

from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from yax import Mox, mox as make_mox, mtree_eval as eval_mox


class Model(nn.Module):
    @nn.compact
    def __call__(self, xs):
        ys = nn.Dense(2)(xs)
        zs = nn.relu(ys)
        return nn.Dense(2)(zs)


class TestProcessCustomJVPCall:

    @pytest.mark.parametrize('apply', [
        pytest.param(lambda x: x, id='no grad'),
        pytest.param(jax.grad, id='grad'),
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
