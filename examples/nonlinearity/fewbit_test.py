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
from numpy.testing import assert_allclose, assert_equal

from fewbit import BOUNDARIES, LEVELS, gelu, gelu_fwd, grandient_quantized_bwd


class TestGrandientQuantizedBwd:

    def test_abstract_eval(self):
        xs = jnp.array([89, 109, 255], dtype=jnp.uint8)
        cotangents = jnp.ones((2, 4))
        _ = jax.make_jaxpr(grandient_quantized_bwd)(xs, LEVELS, cotangents)

    def test_lowering(self):
        ix = jnp.array([1, 3, 5, 6, 6, 6, 7, 7]).reshape(2, 4)
        xs = jnp.array([89, 109, 255], dtype=jnp.uint8)
        cotangents = jnp.ones((2, 4))
        actual = grandient_quantized_bwd(xs, LEVELS, cotangents)
        desired = LEVELS[ix]
        assert_allclose(actual, desired)


class TestGELUForward:

    def test_abstract_eval(self):
        _ = jax.make_jaxpr(gelu_fwd)(jnp.empty((2, 3)), BOUNDARIES)

    def test_lowering(self):
        xs = jnp.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 200]).reshape(2, 4)
        ys, qs = gelu_fwd(xs, BOUNDARIES)
        assert_equal(qs, np.array([89, 109, 255], dtype=np.uint8))

        def fn(xs):
            ys = np.asarray(xs, dtype=np.float64)
            zs = 0.5 * ys * (1 + sp.special.erf(ys / np.sqrt(2)))
            return zs.astype(xs.dtype)

        zs = fn(xs)
        assert_allclose(ys, zs)


class TestGELU:

    def test_abstract_eval(self):
        _ = jax.make_jaxpr(gelu)(jnp.empty((2, 3)), BOUNDARIES, LEVELS)

    def test_lowering(self):
        xs = jnp.arange(2 * 3, dtype=jnp.float32).reshape(2, 3)
        ys = gelu(xs, BOUNDARIES, LEVELS)

        def fn(xs):
            ys = np.asarray(xs, dtype=np.float64)
            zs = 0.5 * ys * (1 + sp.special.erf(ys / np.sqrt(2)))
            return zs.astype(xs.dtype)

        zs = fn(xs)
        assert_allclose(ys, zs)

    def test_custom_vjp(self):
        xs = jnp.float32(1.5)
        gelu_vjp = jax.value_and_grad(gelu)

        # Trace gradient-quantized GELU first.
        _ = jax.make_jaxpr(gelu_vjp)(xs, BOUNDARIES, LEVELS)

        # Verify correctness numerically.
        desired = gelu(xs, BOUNDARIES, LEVELS)
        actual, _ = jax.jit(gelu_vjp)(xs, BOUNDARIES, LEVELS)
        assert_allclose(actual, desired)  # TODO(@daskol): Assert grads.
