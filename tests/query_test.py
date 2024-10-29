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

from yax import Equation, Expr, Mox, XPath, make_mox, query, tokenize_xpath

with_xpaths = pytest.mark.parametrize('value', [
    '/',
    '//',
    '//pjit',
    '//[@primitive="module_call"]',
    '//[@primitive="pjit"][@name="relu"]',
    '//[@name="Dense_0"][@features=10]',
    '//[@name="ResBlock"]//[@features=10]',
    './module_call/..//[@type="Dense"][@features=10]',
])


class ResBlock(nn.Module):
    @nn.compact
    def __call__(self, xs):
        return xs + nn.Dense(10)(xs)


@pytest.fixture
def mox():
    model = ResBlock()
    batch = jnp.ones((1, 10))
    params = jax.jit(model.init)(jax.random.PRNGKey(42), batch)
    mox = make_mox(model.apply)(params, batch)
    del model, batch, params
    yield mox


@with_xpaths
def test_tokenize_xpath(value: str):
    tokens = tokenize_xpath(value)
    assert ''.join(str(x) for x in tokens) == value


@with_xpaths
def test_parse_xpath(value: str):
    xpath = XPath(value)
    assert str(xpath)


class TestQuery:

    def test_query_root(self, mox: Mox):
        xpath = XPath('/')
        nodes = query(xpath, mox)
        assert len(nodes) == 1
        assert nodes[0] is mox

    def test_query_all_decendants(self, mox: Mox):
        xpath = XPath('//')
        nodes = query(xpath, mox)
        assert len(nodes) == 7
        assert nodes[0] is mox

    @pytest.mark.parametrize('value,type_', [
        pytest.param('//pjit', Equation, id='jaxpr'),
        pytest.param('//module_call', Mox, id='mox'),
    ])
    def test_query_by_name(self, mox: Mox, value: str, type_: Type[Expr]):
        xpath = XPath(value)
        nodes = query(xpath, mox)
        assert 1 <= len(nodes) <= 3
        assert isinstance(nodes[0], type_)

    def test_query_by_type(self, mox: Mox):
        xpath = XPath('//[@type="Dense"]')
        nodes = query(xpath, mox)
        assert len(nodes) == 1
        assert isinstance(nodes[0], Mox)
        module: Mox = nodes[0]
        assert not module.is_ephemeral
        assert module.module_ty.__name__

    def test_query_by_attr(self, mox: Mox):
        xpath = XPath('//[@features=10]')
        nodes = query(xpath, mox)
        assert len(nodes) == 1
        assert isinstance(nodes[0], Mox)
        module: Mox = nodes[0]
        assert not module.is_ephemeral
        assert module.module_ty.__name__
