import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array

from yax import find_modules, trace


class AddModule(nn.Module):

    @nn.compact
    def __call__(self, xs: Array, ys: Array) -> Array:
        print(self.__class__.__name__)
        print(xs)
        print(ys)
        find_modules()
        return xs + ys


class ContainerModule(nn.Module):

    @nn.compact
    def __call__(self, xs: Array, ys: Array) -> Array:
        print(self.__class__.__name__)
        find_modules()
        zs = AddModule()(xs, ys)
        print('zs =', zs)
        return ys * zs


def test_trace_add():
    xs = jnp.ones(3)
    ys = jnp.ones(3)
    key = jax.random.PRNGKey(42)
    model = ContainerModule()
    params = jax.jit(model.init)(key, xs, ys)

    res = trace(model.apply)(params, xs, ys)
    print(res)


if __name__ == '__main__':
    test_trace_add()
