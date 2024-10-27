![Linting and testing][1]
![Nightly][2]

[1]: https://github.com/daskol/yax/actions/workflows/on-push.yml/badge.svg
[2]: https://github.com/daskol/yax/actions/workflows/on-schedule.yml/badge.svg

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

### Substitution

Substitution requires the preservation of some invariants.

- Inputs and outputs are reused.
- New outputs are prohibited for now.
- New inputs are propagated to root node. There is a difference between Jaxpr
  (leaf) and Mox (inode).

  - \[Jaxpr\] New inputs are append to all parents.
  - \[MoX\] Inernal node have two kind of input parameters: plain inputs and
    weight params. FLAX requires weight params to be the first input parameter.
    Thus old subtree should be updated with the new one.

  In order to update input parameters, we should update `in_tree` as well.
  Similarly, update to weight params requires update to `var_tree`. Note that
  inputs/params handling for root node differs since params are passed
  explicitely while for all internal expressions params comprises closure
  context. Surely, any modification of `in_tree` or `var_tree` requires update
  of input symbols.

  Note, the all parent should be marks as ephemeral. Also, inputs and outputs
  of a replacement should be type checked agains its predcessors and successors
  respectively.

  ```python
  def substitute(parents, expr):
    for parent in reversed(parents):
      update_param_tree(parent, expr)
  ```
