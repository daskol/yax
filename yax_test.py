import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from jax import Array

from yax import mox


def test_mox_add():
    xs = jnp.ones(3)
    ys = jnp.ones(3)
    mtree = mox(lambda x, y: x + y)(xs, ys)
    print(mtree)
    assert mtree.children == []


class AddModule(nn.Module):

    @nn.compact
    def __call__(self, xs: Array, ys: Array) -> Array:
        return xs + ys


class ContainerModule(nn.Module):

    @nn.compact
    def __call__(self, xs: Array, ys: Array) -> Array:
        return xs * AddModule()(xs, ys)


def test_mox_trivial():
    xs = jnp.ones(3)
    ys = jnp.ones(3)
    key = jax.random.PRNGKey(42)
    model = ContainerModule()
    params = jax.jit(model.init)(key, xs, ys)

    mtree = mox(model.apply)(params, xs, ys)
    print(mtree)


class ResBlock(nn.Module):

    features: int = 4

    @nn.compact
    def __call__(self, xs: Array) -> Array:
        assert self.features == xs.shape[-1], \
            f'Mismatched numbers of features: {self.features} vs {xs.shape}.'
        return xs + nn.Dense(self.features)(xs)


def test_residual():
    xs = jnp.ones(4)
    key = jax.random.PRNGKey(42)
    model = ResBlock()
    params = jax.jit(model.init)(key, xs)

    mtree = mox(model.apply)(params, xs)
    print(mtree)


@pytest.mark.xfail(reason='no dynamic trace')
@pytest.mark.slow
class TestHFModels:

    def test_roberta(self):
        from os import environ
        environ['USE_FLAX'] = '1'
        environ['USE_TORCH'] = '0'

        from transformers import FlaxRobertaForSequenceClassification
        model = FlaxRobertaForSequenceClassification \
            .from_pretrained('roberta-base')

        input_ids = jnp.ones((1, 3), dtype=jnp.int32)
        mtree = mox(model)(input_ids)
        print(mtree)
