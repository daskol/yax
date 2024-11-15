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

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.interpreters import mlir, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

import _fewbit

for name, fn in _fewbit.registrations().items():
    xla_client.register_custom_call_target(name, fn)

BOUNDARIES = jnp.array([
    -1.00000000e+02, -2.41658117e+00, -7.10008030e-01, -3.25840561e-01,
    1.06942188e-04, 3.26057167e-01, 7.10240866e-01, 2.41447889e+00,
    1.00000000e+02
])

LEVELS = jnp.array([
    -1.93991237e-04, -8.82791291e-02, 1.25683834e-01, 3.72314415e-01,
    6.27851368e-01, 8.74450518e-01, 1.08834805e+00, 1.00019494e+00
])


gelu_p = Primitive('fewbit_gelu')
gelu_p.multiple_results = False
gelu_p.def_impl(partial(xla.apply_primitive, gelu_p))


def gelu(xs: Array, boundaries: Array) -> Array:
    return gelu_p.bind(xs, boundaries)


@gelu_p.def_abstract_eval
def gelu_abstract_eval(xs: Array, boundaries: Array) -> ShapedArray:
    return ShapedArray(xs.shape, xs.dtype)


@partial(mlir.register_lowering, gelu_p)
def gelu_lowering(ctx: mlir.LoweringRuleContext, xs: Array, bs: Array):
    # Describe input buffers.
    dtype = mlir.ir.RankedTensorType(xs.type)
    size = np.prod(dtype.shape).astype(np.int64)
    layout = tuple(range(len(dtype.shape) - 1, -1, -1))

    # Describe output buffers.
    bits = np.int64(bs.type.shape[0].bit_length() - 1)
    elem_dtype = mlir.ir.IntegerType.get_unsigned(8)
    accum_dtype = mlir.ir.RankedTensorType.get([size], elem_dtype)

    op = custom_call(
        'cpu_gelu_f32',
        operands=[mlir.ir_constant(size), xs,
                  mlir.ir_constant(bits), bs],
        operand_layouts=[(), layout, (), (0, )],
        result_types=[dtype, accum_dtype],
        result_layouts=[layout, (0, )],
    )
    return op.results[:1]


@partial(jax.custom_vjp, nondiff_argnums=())
def gelu_vjp(xs: Array, boundaries: Array, levels: Array) -> Array:
    return gelu(xs, boundaries)


def gelu_fwd(xs: Array, boundaries: Array, levels: Array):
    return gelu(xs, boundaries), (jnp.ones_like(xs) * 2, levels)


def gelu_bwd(residue: tuple[Array, Array], cotangent: Array):
    return (cotangent, None, None)


gelu_vjp.defvjp(gelu_fwd, gelu_bwd)
gelu_p.def_effectful_abstract_eval
