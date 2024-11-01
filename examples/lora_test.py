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

from copy import deepcopy
from itertools import count

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import lecun_normal, zeros_init
from flax.typing import Dtype, Initializer, PrecisionLike
from numpy.testing import assert_allclose

from yax import Expr, Mox, eval_mox, make_mox, sub


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
        in_features = jnp.shape(xs)[-1]

        lhs_shape = (in_features, self.rank)
        lhs = self.param('lhs', self.lhs_init, lhs_shape, self.param_dtype)

        rhs_shape = (self.features, self.rank)
        rhs = self.param('rhs', self.rhs_init, rhs_shape, self.param_dtype)

        # NOTE Contraction order is important. See also
        # https://arxiv.org/abs/2312.03415
        return nn.Dense(self.features)(xs) + \
               (self.alpha / self.rank) * (xs @ lhs) @ rhs.T


class Model(nn.Module):
    @nn.compact
    def __call__(self, xs):
        ys = nn.Dense(10)(xs)
        zs = nn.relu(ys)
        return nn.Dense(2)(zs)


def test_lora():
    """A use case of substituting an affine layers with LoRA-adapters."""
    batch = jnp.ones((1, 10))
    key = jax.random.PRNGKey(42)
    lora_ix = (f'LoRA_{ix}' for ix in count())  # Name LoRA-adapters.

    # 1. Build and intialize model.
    model = Model()
    key, subkey = jax.random.split(key)
    model_params = jax.jit(model.init)(subkey, batch)

    # 2. Build a module expression of the original model.
    mox: Mox = make_mox(model.apply)(model_params, batch)
    xs = eval_mox(mox, model_params, batch)

    def sub_fn(path: tuple[str, ...], node: Expr) -> Expr:
        if not isinstance(node, Mox):
            return node

        features: int
        if (features := node.params.get('features')) is None:
            raise RuntimeError(f'No `features` attribute: {node}.')
        inputs = node.inputs[node.var_tree.num_leaves:]
        in_shape = inputs[0].value.shape
        *_, name = path

        # 3. Initialize LoRA-adapter.
        nonlocal key
        key, subkey = jax.random.split(key)
        batch = jnp.ones(in_shape)
        adapter = LoRA(features=features, rank=2)
        adapter_name = next(lora_ix)
        adapter_params = jax.jit(adapter.init)(subkey, batch)

        # 4. Substitute dense weight tree with LoRA-adapter weights.
        # TODO(@daskol): What is the easiest? Flatten/unflatten?
        lora_params['params'][adapter_name] = {
            **adapter_params['params'],
            'Dense_0': lora_params['params'].pop(name),
        }

        # 5. Build a module expression for new LoRA-adapter.
        mox = make_mox(adapter.apply)(adapter_params, batch)
        mox = mox.children[0]
        mox.params['name'] = adapter_name  # Must be unique in node.
        return mox

    # 5. Substitute (single) dense layer with LoRA-adapter.
    lora_params = deepcopy(model_params)
    lora_mox = sub('//[@type="Dense"]', sub_fn, mox)

    # 6. Apply LoRA-adapted model without and with JIT.
    ys = eval_mox(lora_mox, lora_params, batch)
    assert_allclose(ys, xs)

    def apply(params, batch):
        return eval_mox(lora_mox, lora_params, batch)

    zs = jax.jit(apply)(lora_params, batch)
    assert_allclose(zs, xs, atol=1e-6)

    # TODO(@daskol): What about optimizer state?
