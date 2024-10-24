import pytest

from yax import tokenize_xpath


@pytest.mark.parametrize('value', [
    '/',
    '//',
    '//[@primitive="module_call"]',
    '//[@primitive="pjit"][@name="relu"]',
    '//[@name="Dense_0"][@features=10]',
    '//[@name="ResBlock"]//[@features=10]',
])
def test_tokenize_xpath(value: str):
    tokens = tokenize_xpath(value)
    assert ''.join(str(x) for x in tokens) == value
