# YAX: LoRA

This example demonstrates how one can exploit a module expression (MoX)
approach to modify a base model for [parameter efficient fine-tuning][1] (PEFT)
with [LoRA-adapters][2]. Real-world example of preparation RoBERTa model can be
found in test [`lora_test.py::test_roberta`][3].

Assume we have already build a module expression `mox` for a FLAX model with
weights `params`. And we want to tweak a model in a way that all
fully-connected layers are replaced with their corresponding LoRA-adapters.
This can be easily achieved with `yax` as follows.

```python
from lora import lora
from yax import eval_mox

mox, params, merge = lora(key, '//@[type="Dense"]', mox, params, rank, alpha)

@jax.jit
def apply(params, inputs):
    return eval_mox(mox, params, inputs)
```

After model training, we can transform parameter tree of LoRAfied model back to
the original one with `merge` routine.

```python
params = merge(params)
outputs = model.apply(params, inputs)  # Apply original model.
```

That's it!!1!

[1]: https://paperswithcode.com/task/parameter-efficient-fine-tuning
[2]: https://openreview.net/forum?id=nZeVKeeFYf9
[3]: lora_test.py
