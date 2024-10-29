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

from typing import Type

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from jax.core import ConcreteArray, ShapedArray
from numpy.testing import assert_allclose

from yax import Literal, Mox, Symbol, Var, eval_mox, make_mox


def assert_empty(mox):
    __tracebackhide__ = True  # Hide traceback for py.test
    assert isinstance(mox, Mox)
    assert mox.is_ephemeral
    assert mox.children == []


def assert_output_types(mox: Mox, types: Type[Symbol]):
    __tracebackhide__ = True  # Hide traceback for py.test
    assert len(mox.outputs) == len(types)
    for output, type_ in zip(mox.outputs, types):
        assert isinstance(output, type_)
        if type_ is Literal:
            assert isinstance(output.value, ConcreteArray)
        elif type_ is Var:
            assert isinstance(output.value, ShapedArray)
        else:
            raise RuntimeError(f'Unknown symbol type: {type_}.')


def test_pure_inputs():
    def fn(xs) -> jax.Array:
        return xs

    mox = make_mox(fn)(jnp.ones((2, 3)))
    assert_empty(mox)
    assert len(mox.inputs) == 1
    assert_output_types(mox, (Var, ))

    value = jnp.ones((2, 3))
    assert_allclose(eval_mox(mox, value), value)


def test_pure_consts():
    def fn() -> jax.Array:
        return jnp.ones((3, 2))

    mox = make_mox(fn)()
    assert_empty(mox)
    assert len(mox.inputs) == 0
    assert_output_types(mox, (Literal, ))

    const = jnp.ones((3, 2))
    assert_allclose(eval_mox(mox), const)


def test_pure_mixed():
    def fn(xs) -> jax.Array:
        return xs, jnp.ones((3, 2))

    mox = make_mox(fn)(jnp.ones((2, 3)))
    assert len(mox.inputs) == 1
    assert_output_types(mox, (Var, Literal))

    value = jnp.ones((2, 3))
    const = jnp.ones((3, 2))
    actual = eval_mox(mox, value)
    assert len(actual) == 2
    assert_allclose(actual[0], value)
    assert_allclose(actual[1], const)


class InputLiteralModel(nn.Module):
    @nn.compact
    def __call__(self, xs):
        ys = jnp.ones_like(xs)
        return nn.Dense(2)(ys)


class OutputLiteralModel(nn.Module):
    @nn.compact
    def __call__(self, xs):
        ys = jnp.ones((2, 3))
        return nn.Dense(2)(xs), ys


@pytest.mark.parametrize('model_ty', [
    pytest.param(InputLiteralModel, id='input'),
    pytest.param(OutputLiteralModel, id='output'),
])
def test_pure_dense(model_ty: Type[nn.Module]):
    batch = jnp.ones((1, 2))
    model = model_ty()
    params = jax.jit(model.init)(jax.random.PRNGKey(42), batch)
    mox = make_mox(model.apply)(params, batch)

    actual = eval_mox(mox, params, batch)
    desired = model.apply(params, batch)
    for lhs, rhs in zip(actual, desired):
        assert_allclose(lhs, rhs)
