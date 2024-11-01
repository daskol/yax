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

from typing import Callable, TypeAlias

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array
from numpy.testing import assert_allclose

from yax import Mox, eval_mox, make_mox, sub

ActivationFn: TypeAlias = Callable[[Array], Array]


class Model(nn.Module):
    act_fn: ActivationFn = jax.nn.relu

    @nn.compact
    def __call__(self, xs):
        ys = nn.Dense(2)(xs)
        zs = self.act_fn(ys)
        return nn.Dense(2)(zs)


def test_nonlinearity():
    """A use case of substituting an activation function."""
    batch = jnp.empty((1, 2))
    key = jax.random.PRNGKey(42)

    # 1. Build and intialize model.
    def make_model(act_fn: ActivationFn):
        model = Model()
        _, subkey = jax.random.split(key)
        params = jax.jit(model.init)(subkey, batch)
        return model, params

    model_relu, params = make_model(jax.nn.relu)

    # 2. Build a module expression.
    mox_relu: Mox = make_mox(model_relu.apply)(params, batch)

    # 3. Substitute all `relu` activation functions with `gelu`.
    mox = sub('//pjit[@name="relu"]', mox_relu, jax.nn.gelu)
    print(mox)

    # 4. Compare inference of modified `mox` against `mox_gelu` initialized
    #    with GeLU activation function.
    model_gelu, params_gelu = make_model(jax.nn.gelu)
    assert params == params_gelu, \
        f'Model weights differ: {params} vs {params_gelu}.'
    desired = model_gelu.apply(params, batch)
    actual = eval_mox(mox, params, batch)
    assert_allclose(actual, desired)
