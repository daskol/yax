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

from collections.abc import Mapping

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from yax import Static, branch, make_mox, map_param_tree, reconstruct


def absolute(xs):
    return jnp.abs(xs)


def negate(xs):
    return -xs


def make_eq(fn=negate, shape=(2,)):
    return make_mox(fn)(jnp.ones(shape)).children[0]


class DuplicateCases(Mapping):

    def __init__(self, items):
        self._items = tuple(items)

    def __iter__(self):
        return (k for k, v in self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, key):
        for k, v in self._items:
            if k is key:
                return v
        raise KeyError

    def items(self):
        return self._items


class Scale(nn.Module):

    @nn.compact
    def __call__(self, xs):
        scale = self.param('scale', nn.initializers.ones, xs.shape)
        return xs * scale


def test_reconstruct_dict_dict():
    res = reconstruct({('a', 'a'): 1, ('a', 'b'): 2, ('b', 'a'): 3})
    assert res['a']['a'] == 1
    assert res['a']['b'] == 2
    assert res['b']['a'] == 3


def test_reconstruct_dict_list():
    res = reconstruct({('a', 0): 1, ('a', 1): 2, ('b', 'a'): 3})
    assert res['a'] == [1, 2]
    assert res['b']['a'] == 3


def test_reconstruct_list_dict():
    res = reconstruct({(0, 'a'): 1, (0, 'b'): 2, (1, 'a'): 3})
    assert res[0]['a'] == 1
    assert res[0]['b'] == 2
    assert res[1]['a'] == 3


def test_reconstruct_interleaved():
    with pytest.raises(RuntimeError, match='interleave together'):
        reconstruct({('a', 'a'): 1, ('a', 'b'): 2, ('a', ): 3})


def test_map_param_tree():
    xs = {'params': {'kernel': jnp.eye(2)}}
    ys = {'params': {'bias': jnp.zeros(2)}}

    def fn(leaves):
        for leaf in (x for x in leaves if x is not None):
            return leaf
        else:
            raise RuntimeError(f'No defined leaves: {leaves}.')

    res = map_param_tree(fn, (xs, ys))
    assert_array_equal(res['params']['kernel'], xs['params']['kernel'])
    assert_array_equal(res['params']['bias'], ys['params']['bias'])


def test_branch_key_normalization():
    expr = make_eq()
    node = branch('mode', {np.bool_(True): expr, False: expr})

    assert tuple(node.cases) == (True, False)
    assert isinstance(node.inputs[node.var_tree.num_leaves], Static)

    with pytest.raises(ValueError, match='Duplicate static branch key'):
        branch('mode', DuplicateCases([(True, expr), (np.bool_(True), expr)]))
    with pytest.raises(TypeError, match='Unsupported static branch key'):
        branch('mode', {1: expr})
    with pytest.raises(ValueError, match='at least one case'):
        branch('mode', {})


def test_branch_boundary_validation():
    with pytest.raises(RuntimeError, match='input symbol shapes differ'):
        branch('mode', {'a': make_eq(shape=(2,)), 'b': make_eq(shape=(3,))})

    def first(xs):
        return xs[:1]

    with pytest.raises(RuntimeError, match='output symbol shapes differ'):
        branch('mode', {'a': make_eq(), 'b': make_eq(first)})
