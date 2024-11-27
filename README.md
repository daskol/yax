![Linting and testing][1]
![Nightly][2]

[1]: https://github.com/daskol/yax/actions/workflows/on-push.yml/badge.svg
[2]: https://github.com/daskol/yax/actions/workflows/on-schedule.yml/badge.svg

# YAX

*Yet Another X: JAX/FLAX module tracing, modification, and evaluation.*

## Overview

Deep learning frameworks like PyTorch, Keras, and JAX/FLAX usually provide a
"module-level" API, which abstracts a layer—an architectural unit in a neural
network. While modules are descriptive and easy to use, they can sometimes be
inconvenient to work with programmatically. Specifically, it is challenging to
modify model architecture on the fly, though changing weight structures
dynamically is not as difficult. So, why can't we work with modules in the same
flexible way?

YAX is a library within the JAX/FLAX ecosystem for building, evaluating, and
modifying the intermediate representation of a neural network's modular
structure. Modular structures are represented with the help of MoX, a Module
eXpression, which is an extension of JAX expressions (Jaxpr). MoX is pronounced
as ∗[mokh]∗ and means "moss" in Russian.

```bash
pip install git+https://github.com/daskol/yax.git
```

## Usage

Module expressions (MoX) are extremely useful in certain situations. For
example, they enable the application of custom LoRA-like adapters or model
performance optimizations, such as quantized gradient activation functions (see
[fewbit][4]). We've briefly discussed what YAX/MoX can accomplish, and we’ll
use the ResBlock below for further demonstrations.

```python
import flax.linen as nn
import yax

class ResBlock(nn.Module):
    @nn.compact
    def __call__(self, xs):
        return xs + nn.Dense(10)(xs)

mod = ResBlock()
batch = jnp.empty(1, 10)
params = jax.jit(mod.init)(jax.random.PRNGKey(42), batch)
```

**Tracing** First, we need to build a module representation (also known as
MoX). This can be done in a similar way to creating a Jaxpr (see
`jax.make_jaxpr`).

```python
mox = yax.make_mox(mod.apply)(params, batch)
print(mox)
```

Pretty printing is is not very pretty for MoX at the moment but it will look
like the following. Also, we have implemented serialization to XML and YSON
(see Serialization section).

```jaxpr
{ lambda ; a:f32[10] b:f32[10,10] c:f32[1,10]. let
  d:f32[1,10] = module_call {
    lambda ; a:f32[10] b:f32[10,10] c:f32[1,10]. let
      d:f32[1,10] = dot_general[dimension_numbers=(([1], [0]), ([], []))] c b
      e:f32[1,10] = reshape[dimensions=None new_sizes=(1, 10)] a
      f:f32[1,10] = add d e
    in (f,) }
  e:f32[1,10] = add d a
  in (e,)}
```

**Evaluation** MoX can be evaluated similarly to Jaxpr, but the most important
feature is that `yax.eval_mox` can be composed with common JAX transformations,
as shown below.

```python
def apply(params, batch):
    return yax.eval_mox(mox, params, input_batch)

_ = apply(params, batch)  # Greedy evaluation.
_ = jax.jit(apply)(params, batch)  # JIT-compiled execution.
```

**Querying** MoX provides tools for model exploration and examination.
Specifically, MoX can help answer questions like: "What `nn.Dense` modules have
10 features?"

```python
modules: Sequence[yax.Mox] = yax.query('//module_call[@features=10]', mox)
```

We use XPath (the familiar XML Path expression language) for writing queries.
XPath is a concise and convenient way to express search conditions. In fact,
the module tree can be represented similarly to a DOM structure, which
effectively models the nested structure of a neural network as well as the
module attributes in its internal nodes.

**Modification** With such an expressive query language, modifying an original
model on the fly becomes easy. For example, one can replace all ReLU activation
functions with GELU or substitute all `nn.Dense` layers with LoRA adapters.

```python
# Replace ReLU with GELU
gelu_mox = yax.make_mox(nn.gelu)(inputs)
modified_mox = yax.sub('//pjit[@name="relu"]', gelu_mox, mox)

# Apply LoRA-adapters to all fully-connected layers.
lora_mox = yax.make_mox(lora.apply)(params, inputs)
modified_mox = yax.sub('//module_call[@type="Dense"]', lora_mox, mox)
```

[4]: https://proceedings.mlr.press/v202/novikov23a.html

## Module Expression (MoX)

### XML

The funniest part about MoX is that it can be serialized to XML. Hardly anyone
uses XML nowadays outside the Java ecosystem and some legacy projects. However,
XML is actually a good and even appropriate serialization format.

```xml
<module_call type="flax.nn.Dense" name="Dense_0" features="10">
  <input type="fp32[10]">a</input>
  <input type="fp32[10,10]">b</input>
  <input type="fp32[10]">c</input>
  <dot_general dimension_numbers="(([0], [0]), ([], []))">
    <input type="fp32[10,10]">b</input>
    <input type="fp32[10]">c</input>
    <output type="fp32[10,10]">d</output>
  </dot_general>
  <pjit
    jaxpr="{ lambda ; a:f32[10], b:f32[10]. let c:f32[10] = add a b in (c,) }">
    <input type="fp32[10]">d</input>
    <input type="fp32[10]">a</input>
    <output type="fp32[10]">e</output>
  </pjit>
  <outputs type="fp32[10]">e</outputs>
</module_call>
```

### YSON

[YSON][1] stands for Yandex Serialization Object Notation. It is a
serialization format similar to JSON due to its compact notation but is more
expressive. In terms of representational expressiveness, YSON is comparable to
XML.

```yson
<primitive="module_call";
 type="flax.nn.Dense"; name="Dense_0"; features=10;
 inputs={a="fp32[10]"; b="fp32[10,10]"; c="fp32[10]};
 outputs={e="fp32[10]"}>[
  <primitive="dot_general";
   dimension_numbers="[[[0], [0]], [[], []]]";
   inputs={с="fp32[10]"; b="fp32[10,10]};
   outputs={d="fp32[10]"}>#;
  <primitive="pjit";
   inputs={d="fp32[10]"; a="fp32[10]"};
   outputs={e="fp32[10]"};
   jaxpr="{ lambda ; a:f32[10], b:f32[10]. let c:f32[10] = add a b in (c,) }";
  >#;
]
```

[1]: https://ytsaurus.tech/docs/en/user-guide/storage/yson
[2]: https://msgpack.org/
[3]: https://protobuf.dev/

### Limitations

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
- Compositionality with `jax.scan`, `jax.vmap`, and `jax.pmap` is not verified.
- Pretty printing of module expressions is not available for now.

# Container

```shell
docker pull ghcr.io/daskol/yax
```
