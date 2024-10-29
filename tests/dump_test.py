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

from io import BytesIO, StringIO
from xml.etree import ElementTree

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from yax import Mox, dump_xml, dump_yson, make_mox


class ResBlock(nn.Module):
    features: int = 10

    @nn.compact
    def __call__(self, xs):
        return xs + nn.Dense(self.features)(xs)


@pytest.fixture
def mox():
    batch = jnp.ones((2, 10))
    model = ResBlock()
    params = jax.jit(model.init)(jax.random.PRNGKey(42), batch)
    mox = make_mox(model.apply)(params, batch)
    del batch, model, params
    yield mox


def test_dump_xml(mox: Mox):
    buf = StringIO()
    dump_xml(mox, buf)
    assert buf.tell() > 0, 'Output buffer is empty.'
    buf.seek(0)

    etree = ElementTree.parse(buf)
    children = etree.findall('module_call')
    assert len(children) == 1
    children = children[0].findall('module_call')
    assert len(children) == 1


def test_dump_yson(mox: Mox):
    _ = pytest.importorskip('yt.yson')
    import yt.yson

    buf = BytesIO()
    dump_yson(mox, buf)
    assert buf.tell() > 0, 'Output buffer is empty.'
    buf.seek(0)

    obj = yt.yson.load(buf)
    assert obj, 'Deserialized list is empty.'
    assert obj.attributes['primitive'] == 'module_call'
    assert obj.attributes['ephemeral']
