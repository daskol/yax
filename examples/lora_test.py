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

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from flax.linen.initializers import lecun_normal, zeros_init
from flax.typing import Dtype, Initializer, PrecisionLike

from yax import Mox, make_mox, sub


class LoRA(nn.Module):

    features: int
    rank: int
    alpha: float = 1.0

    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    lhs_init: Initializer = lecun_normal()
    rhs_init: Initializer = zeros_init()

    @nn.compact
    def __call__(self, xs):
        out_features = jnp.shape(xs)[-1]

        lhs_shape = (self.features, self.rank)
        lhs = self.param('lhs', self.lhs_init, lhs_shape, self.param_dtype)

        rhs_shape = (out_features, self.rank)
        rhs = self.param('rhs', self.rhs_init, rhs_shape, self.param_dtype)

        # NOTE Contraction order is important. See also
        # https://arxiv.org/abs/2312.03415
        return nn.Dense(self.features)(xs) + \
               (self.alpha / self.rank) * (xs @ lhs) @ rhs.T


class Model(nn.Module):
    @nn.compact
    def __call__(self, xs):
        ys = nn.Conv(10, 3)(xs)
        ys = nn.relu(ys)
        ys = nn.Dense(10)(ys)
        return ys


@pytest.mark.xfail(reason='no jvp support')
def test_lora():
    """A use case of substituting a affine layer with LoRA-adapter."""
    batch = jnp.empty((1, 10))
    key = jax.random.PRNGKey(42)

    # 1. Build and intialize model.
    model = Model()
    key, subkey = jax.random.split(key)
    params = jax.jit(model.init)(subkey, batch)

    # 2. Build a module expression.
    mtree: Mox = make_mox(model.apply)(params, batch)

    # 3. Initialize LoRA-adapter.
    adapter = LoRA(features=10, rank=2)
    _, subkey = jax.random.split(key)
    adapter_params = jax.jit(adapter.init)(subkey, batch)

    # 4. Substitute dense weight tree with LoRA-adapter weights.
    # TODO(@daskol): What is the easiest? Flatten/unflatten?
    dense_params = params['params'].pop('Dense_0')
    params['params']['LoRA_0'] = {**adapter_params, 'Dense_0': dense_params}

    # 5. Substitute (single) dense layer with LoRA-adapter.
    mtree_lora = sub('//[type="Dense"]', mtree, adapter)

    # TODO(@daskol): Present new module and weight tree. What about optimizer
    # state?
    print(mtree_lora)
