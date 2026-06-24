# Copyright 2026 Daniel Bershatsky
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
from numpy.testing import assert_allclose

from yax import eval_mox, make_mox


class DropoutModule(nn.Module):

    @nn.compact
    def __call__(self, xs, train: bool = True):
        return nn.Dropout(0.5)(xs, deterministic=not train)


@pytest.mark.parametrize('kwargs', [{}, {'train': True}])
def test_static_bool_kwargs(kwargs):
    xs = jnp.ones((2, 3))
    rngs = {
        'params': jax.random.key(0),
        'dropout': jax.random.key(1),
    }
    model = DropoutModule()
    params = model.init(rngs, xs, **kwargs)
    apply_rngs = {'dropout': jax.random.key(2)}

    mox = make_mox(model.apply)(params, xs, rngs=apply_rngs, **kwargs)

    actual = eval_mox(mox, params, xs, rngs=apply_rngs, **kwargs)
    desired = model.apply(params, xs, rngs=apply_rngs, **kwargs)
    assert_allclose(actual, desired)


def test_numeric_scalar_inputs_stay_dynamic():
    def fn(xs, scale: float):
        return xs * scale

    xs = jnp.ones((2, 3))
    mox = make_mox(fn)(xs, 2.0)

    assert_allclose(eval_mox(mox, xs, 3.0), fn(xs, 3.0))
