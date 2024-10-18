import operator
from dataclasses import astuple, dataclass
from typing import Any, Iterator

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jax.extend.core import ClosedJaxpr, primitives
from numpy.testing import assert_allclose

from yax import Equation, Mox, mox, mtree_eval


@pytest.mark.parametrize('fn', [
    pytest.param(operator.add, id='add'),
    pytest.param(operator.mul, id='mul'),
    pytest.param(operator.sub, id='sub'),
])
class TestBinaryFunc:

    def test_make(self, fn):
        xs = jnp.ones(3)
        ys = jnp.ones(3)
        mtree = mox(fn)(xs, ys)

        assert isinstance(mtree, Mox)
        assert mtree.is_ephemeral
        assert len(mtree.children) == 1
        assert isinstance(mtree.children[0], Equation)

        eq: Equation = mtree.children[0]
        assert len(eq.inputs) == 2
        assert len(eq.outputs) == 1
        assert eq.prim.name == 'pjit'

        assert 'jaxpr' in eq.params
        jaxpr: ClosedJaxpr = eq.params['jaxpr']
        assert len(jaxpr.eqns) == 1
        jaxpr_eq, *_ = jaxpr.eqns
        desired_primitive = getattr(primitives, f'{fn.__name__}_p')
        assert jaxpr_eq.primitive == desired_primitive

    def test_eval(self, fn):
        xs = jnp.ones(3)
        ys = jnp.ones(3)
        mtree = mox(fn)(xs, ys)
        actual = mtree_eval(mtree, xs, ys)
        desired = fn(xs, ys)
        assert_allclose(actual, desired)


@dataclass
class ModelState:
    model: nn.Module
    params: dict[str, Any]
    batch: jax.Array

    def to_tuple(self) -> tuple[nn.Module, dict[str, Any], jax.Array]:
        return astuple(self)


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


class TestResBlock:

    @pytest.fixture
    @staticmethod
    def state() -> Iterator[ModelState]:
        batch = jnp.ones(4)
        key = jax.random.PRNGKey(42)
        model = ResBlock()
        params = jax.jit(model.init)(key, batch)
        yield ModelState(model, params, batch)

    def test_make(self, state: ModelState):
        mtree = mox(state.model.apply)(state.params, state.batch)
        print(mtree)

    def test_eval(self, state: ModelState):
        mtree = mox(state.model.apply)(state.params, state.batch)
        actual = mtree_eval(mtree)
        print(actual)

        desired = (state.model.apply)(state.params, state.batch)
        print(desired)

        # assert_allclose(actual, desired)


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
