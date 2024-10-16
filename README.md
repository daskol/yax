# YAX

*Yet Another X*

## Overview

**YAX** is a library in JAX/FLAX ecosystem to build, evaluate, and modify
Module eXpressions (MoX) which is similar to Jaxpr by its meaning.

## Proposal

Deep learning frameworks like PyTorch, Keras, or evan Flax usually provides a
"module-level" API, i.e. we operate with modules. However, this sometimes is
not very convenient to work with programmatically. Namely, it is tricky to
change model architecture on-fly. Interestingly, change weight structure is not
a big deal. So why can't we work with modules in the same way?

```python
import flax.linen as nn
from jax import Array

class SuperMod(nn.Module):
    @nn.compact
    def __call__(self, xs: Array) -> Array:
        return xs + nn.Dense(4)(xs)

mod = SuperMod()
params = jax.jit(mod.init)(jax.random.PRNGKey(42), jnp.empty(1, 4))
tape = yax.trace(mod.apply)(params, jnp.empty(1, 4))
print(tape)
```

### Module Tree

Complex module and its internal code blocks are represented as a (non-binary)
module tree.

```python
mtree = {SuperMod(): [nn.Dense(4), lambda xs: env['xs'] + xs]}
```

We can apply a transformation to this module tree. For example, we can
enumerate all leaf modules.

```python
def fn(mpath, mod: nn.Module):
    print(mpath, mod)

_ = mtree_map(fn, mtree)
# /0: nn.Dense(4)
# /1: jaxpr
```

Another example is module substitution, e.g. substitution of affine layer with
LoRA-adapter.

```python
mtree = mtree_sub('/0', nn.Identity())
assert mtree == {SuperMod(): [nn.Identity(), lambda xs: env['xs'] + xs]}сЖц
```

In-order graph traversing is equivalent to an abstract avaluation.
