from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from yax import Mox, XPath, mox as make_mox, tokenize_xpath

curr_dir = Path(__file__).parent

with_xpaths = pytest.mark.parametrize('value', [
    '/',
    '//',
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
    yield None
    return
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
def test_parse_xpath(mox: Mox, value: str):
    xpath = XPath(value)
    assert str(xpath)
