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

__all__ = ('gelu', 'gelu_fwd', 'grandient_quantized_bwd')

for name, fn in _fewbit.registrations().items():
    xla_client.register_custom_call_target(name, fn)

BOUNDARIES = jnp.array([
    -1.00000000e+02, -2.41658117e+00, -7.10008030e-01, -3.25840561e-01,
    1.06942188e-04, 3.26057167e-01, 7.10240866e-01, 2.41447889e+00,
    1.00000000e+02
])[1:-1]

LEVELS = jnp.array([
    -1.93991237e-04, -8.82791291e-02, 1.25683834e-01, 3.72314415e-01,
    6.27851368e-01, 8.74450518e-01, 1.08834805e+00, 1.00019494e+00
])


grandient_quantized_bwd_p = Primitive('fewbit::grandient_quantized_bwd')
grandient_quantized_bwd_p.multiple_results = False


def grandient_quantized_bwd(xs: Array, levels: Array,
                            cotangents: Array) -> Array:
    """Compute VJP of gradient-quantized activation function."""
    return grandient_quantized_bwd_p.bind(xs, levels, cotangents)


@grandient_quantized_bwd_p.def_impl
def grandient_quantized_bwd_impl(xs: Array, levels: Array,
                                 cotangents: Array) -> Array:
    return xla.apply_primitive(grandient_quantized_bwd_p, xs, levels,
                               cotangents)


@grandient_quantized_bwd_p.def_abstract_eval
def grandient_quantized_bwd_abstract_eval(xs: Array, levels: Array,
                                          cotangents: Array) -> ShapedArray:
    return ShapedArray(cotangents.shape, cotangents.dtype)


@partial(mlir.register_lowering, grandient_quantized_bwd_p)
def grandient_quantized_bwd_lowering(ctx: mlir.LoweringRuleContext, xs, levels,
                                     cotangents):
    # Describe input and output buffers.
    bits = np.int64(levels.type.shape[0].bit_length() - 1)
    dtype = mlir.ir.RankedTensorType(cotangents.type)
    size = np.prod(dtype.shape).astype(np.int64)
    layout = tuple(range(len(dtype.shape) - 1, -1, -1))

    op = custom_call(
        'cpu_gradient_quantized_bwd_f32',
        operands=[
            mlir.ir_constant(bits), xs, levels,
            mlir.ir_constant(size), cotangents
        ],
        operand_layouts=[(), (0, ), (0, ), (), layout],
        result_types=[dtype],
        result_layouts=[layout],
    )
    return op.results


gelu_fwd_p = Primitive('fewbit::gelu_fwd')
gelu_fwd_p.multiple_results = True
gelu_fwd_p.def_impl(partial(xla.apply_primitive, gelu_fwd_p))


def gelu_fwd(xs: Array, boundaries: Array) -> Array:
    """Compute VJP-forward rule of gradient-quantized GELU."""
    return gelu_fwd_p.bind(xs, boundaries)


@gelu_fwd_p.def_abstract_eval
def gelu_fwd_abstract_eval(xs: Array, boundaries: Array):
    bits = boundaries.size.bit_length()
    total_bytes = (xs.size * bits) // 8
    if (xs.size * bits) % 8 > 0:
        total_bytes += 1
    return ShapedArray(xs.shape, xs.dtype), ShapedArray(total_bytes, jnp.uint8)


@partial(mlir.register_lowering, gelu_fwd_p)
def gelu_fwd_lowering(ctx: mlir.LoweringRuleContext, xs: Array, bs: Array):
    # Describe input buffers.
    dtype = mlir.ir.RankedTensorType(xs.type)
    size = np.prod(dtype.shape).astype(np.int64)
    layout = tuple(range(len(dtype.shape) - 1, -1, -1))

    # Describe output buffers.
    bits = np.int64(bs.type.shape[0].bit_length())
    length = (size * bits) // 8
    if (size * bits) % 8 > 0:
        length += 1
    elem_dtype = mlir.ir.IntegerType.get_unsigned(8)
    accum_dtype = mlir.ir.RankedTensorType.get([length], elem_dtype)

    op = custom_call(
        'cpu_gelu_fwd_f32',
        operands=[mlir.ir_constant(size), xs,
                  mlir.ir_constant(bits), bs],
        operand_layouts=[(), layout, (), (0, )],
        result_types=[dtype, accum_dtype],
        result_layouts=[layout, (0, )],
    )
    return op.results


gelu_p = Primitive('fewbit_gelu')
gelu_p.multiple_results = False
gelu_p.def_impl(partial(xla.apply_primitive, gelu_p))


@jax.custom_vjp
def gelu(xs: Array, boundaries: Array, levels: Array) -> Array:
    """Calculate activations of gradient-quantized GELU.

    Args:
      xs: Inputs.
      boundaries: Gradient quantization boundaries.
      levels: Gradient quantization levels.

    Returns:
      GELU activations.
    """
    return gelu_p.bind(xs, boundaries, levels)


@gelu_p.def_abstract_eval
def gelu_abstract_eval(xs: Array, boundaries: Array,
                       levels: Array) -> ShapedArray:
    return ShapedArray(xs.shape, xs.dtype)


@partial(mlir.register_lowering, gelu_p)
def gelu_lowering(ctx: mlir.LoweringRuleContext, xs: Array, bs: Array,
                  ls: Array):
    # Describe input buffers.
    dtype = mlir.ir.RankedTensorType(xs.type)
    size = np.prod(dtype.shape).astype(np.int64)
    layout = tuple(range(len(dtype.shape) - 1, -1, -1))

    op = custom_call(
        'cpu_gelu_f32',
        operands=[mlir.ir_constant(size), xs],
        operand_layouts=[(), layout],
        result_types=[dtype],
        result_layouts=[layout],
    )
    return op.results


def gelu_vjp_fwd(inputs: Array, boundaries: Array, levels: Array):
    outputs, primals = gelu_fwd(inputs, boundaries)
    return outputs, (primals, levels)


def gelu_vjp_bwd(residue: tuple[Array, Array], cotangents: Array):
    primals, levels = residue
    results = grandient_quantized_bwd(primals, levels, cotangents)
    return (results, None, None)


gelu.defvjp(gelu_vjp_fwd, gelu_vjp_bwd)
