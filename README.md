![Linting and testing][1]
![Nightly][2]

[1]: https://github.com/daskol/yax/actions/workflows/on-push.yml/badge.svg
[2]: https://github.com/daskol/yax/actions/workflows/on-schedule.yml/badge.svg

# YAX

*Yet Another X: JAX/FLAX module tracing, modification, and evaluation.*

## Overview

Deep learning frameworks like PyTorch, Keras, or even JAX/FLAX usually provides
a "module-level" API, i.e. an abstraction of layer, an architecture unit in a
neural network. Modules are descriptive and easy to use but sometimes they are
quite inconvenient to work with programmatically. Specifically, it is tricky to
change model architecture on-fly while changing weight structure on-fly is not
a big deal. So why can't we work with modules in the same way?

**YAX** is a library in JAX/FLAX ecosystem to build, evaluate, and modify
intermediate representation of a modular structure of neural network. Modular
structure are represented with help of MoX, a Module eXpression, which is is an
extension of JAX expressions (Jaxpr). MoX is pronounced as \[*mokh*\] and means
moss in Russian.

## Usage

Module expressions (MoX) are extremely usefull in some situations. For example,
applying a custom LoRA-like adapter or model performance optimization like
quantized gradient activation functions (see [fewbit][4]). However, we briefly
discuss what YAX/MoX are capable of. We will use the `ResBlock` below for
further demonstrations.

```python
import flax.linen as nn
import yax

class ResBlock(nn.Module):
    @nn.compact
    def __call__(self, xs):
        return xs + nn.Dense(4)(xs)

mod = ResBlock()
batch = jnp.empty(1, 4)
params = jax.jit(mod.init)(jax.random.PRNGKey(42), batch)
```

**Tracing** First of all, we should build a module representation (aka MoX). It
can be done similarly to making Jaxpr (see `jax.make_jaxpr`).

```python
mox = yax.make_mox(mod.apply)(params, batch)
print(mox)
```

Pretty printing is not available for MoX at the moment but it will look like
the following.

```jaxpr
{ lambda ; a:f32[4] b:f32[4,4] c:f32[1,4]. let
  d:f32[1,4] = module_call {
    lambda ; a:f32[4] b:f32[4,4] c:f32[1,4]. let
      d:f32[1,4] = dot_general[dimension_numbers=(([1], [0]), ([], []))] c b
      e:f32[1,4] = reshape[dimensions=None new_sizes=(1, 4)] a
      f:f32[1,4] = add d e
    in (f,) }
  e:f32[1,4] = add d a
  in (e,)}
```

**Evaluation** MoX can be evaluated similarly to Jaxpr but the most important
thing is that `yax.eval_mox` is composable with common JAX transformation in
the following way.

```python
def apply(params, batch):
    return yax.eval_mox(mox, params, input_batch)

_ = apply(params, batch)  # Greedy evaluation.
_ = jax.jit(apply)(params, batch)  # JIT-compiled execution.
```

**Querying** MoX provides facility for model exploration and examination.
Specifically, MoX helps to answer for the following kind of questions: What
does `nn.Dense` module have `10` features?

```python
modules: Sequence[yax.Mox] = yax.query('//module_call[@features=10]', mox)
```

We use XPath (same old XML Path expression language) for writing requests.
XPath is a brief and convient way to expression search conditions. In fact,
module tree can be represented with respect to DOM which excelently represents
nested structure of a neural network as well as module attributes in internal
nodes.

**Modification** With such expressive query language, it is easy to change an
original model on-the-fly. For example, one can change all ReLU activation
functions with GELU or replace all `nn.Dense` layers with LoRA-adapters.

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

The most funny part about MoX is that MoX can be serialized to XML. Barely,
nobody uses XML nowdays outside Java ecosystem and some projects with long
history. However, XML is actually a good and even right serialization format.

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
serialization format similar to JSON due to its compact notation but more
expressive. Regarding to expressivity of representation, YSON is comparable to
XML. Besides textual wire representation, YSON has a binary one as well. This
that makes it something comparable to [MessagePack][2] or [Protobuf][3].

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
