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

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
import scipy.special
from numpy.testing import assert_allclose

from fewbit import BOUNDARIES, LEVELS, gelu, gelu_vjp


class TestGELU:

    def test_abstract_eval(self):
        _ = jax.make_jaxpr(gelu)(jnp.empty((2, 3)), BOUNDARIES)

    def test_lowering(self):
        xs = jnp.arange(2 * 3, dtype=jnp.float32).reshape(2, 3)
        ys = gelu(xs, BOUNDARIES)

        def fn(xs):
            ys = np.asarray(xs, dtype=np.float64)
            zs = 0.5 * ys * (1 + sp.special.erf(ys / np.sqrt(2)))
            return zs.astype(xs.dtype)

        zs = fn(xs)
        assert_allclose(ys, zs)

    def test_custom_vjp(self):
        xs = jnp.float32(1.5)
        ys = gelu_vjp(xs, BOUNDARIES, LEVELS)
        zs, _ = jax.value_and_grad(gelu_vjp)(xs, BOUNDARIES, LEVELS)
        assert_allclose(ys, zs)
