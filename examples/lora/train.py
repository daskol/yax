#!/usr/bin/env python3
# Copyright 2024 Daniel Bershatsky
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example of training script to fine-tuning a RoBERTa model with
LoRA-adapters.
"""

from argparse import ArgumentParser, Namespace
from functools import partial
from operator import or_

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import Array
from transformers import FlaxRobertaForSequenceClassification, RobertaConfig

import yax
from lora import Mask, Params, lora, mask_by_prefix, merge, split

parser = ArgumentParser(description=__doc__)

g_opt = parser.add_argument_group('optimizer options')
g_opt.add_argument('--lr', type=float, default=1e-1)

g_lora = parser.add_argument_group('lora options')
g_lora.add_argument('--lora-rank', type=int, default=2)
g_lora.add_argument('--lora-alpha', type=float, default=10.0)


def count_params(params: Params) -> int:
    return jax.tree.reduce(lambda x, y: x + np.prod(y.shape), params, 0)


def count_bytes(params: Params) -> int:
    return jax.tree.reduce(lambda x, y: x + y.nbytes, params, 0)


def train(lr: float = 1e-1, rank: int = 2, alpha: float = 10):
    # 0. Configure and instantiate model.
    config = RobertaConfig(
        vocab_size=4096,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=384)
    model = FlaxRobertaForSequenceClassification(config)

    def apply(params, input_ids, rngs):
        return model(params=params['params'], input_ids=input_ids,
                     dropout_rng=rngs['dropout'], train=True)

    # 1. Initialize weights.
    params = {}
    params['params'] = model.init_weights(jax.random.key(42), (1, 64))

    num_params = count_params(params)
    num_mbytes = count_bytes(params) / 1024**2
    print(f'(total) num params:     {num_params:_d}')
    print(f'(total) num bytes:      {num_mbytes:_.3f} Mb')

    # 2. Build module expression for the original model.
    input_ids = jnp.arange(10)[None]
    labels = jnp.ones((1, ), dtype=jnp.int32)
    rngs = {'dropout': jax.random.key(42)}
    mox = yax.make_mox(apply)(params, input_ids, rngs)

    # 3. Decide what modules (layers) should be LoRAfied.
    xpath = '//[@name="attention"]//[@type="Dense"]'
    nodes = yax.query(xpath, mox)
    assert len(nodes) == 4 * 2

    # 4. Apply LoRA transformation and build an application function for
    #    LoRAfied model.
    key = jax.random.key(42)
    mox_lora, lora_params, lora_mask, _ = \
        lora(key, xpath, mox, params, rank=2, alpha=10)

    def apply_lora(params: Params, input_ids: Array, rngs):
        return yax.eval_mox(mox_lora, params, input_ids, rngs)

    # 5. Mask classification head as trainable too.
    head_mask = mask_by_prefix(['params', 'classifier'], lora_params)
    mask = jax.tree.map(or_, lora_mask, head_mask)
    trainable_params, frozen_params = split(mask, lora_params)

    num_lora_params = count_params(lora_params)
    num_train_params = count_params(trainable_params)
    num_frozen_params = count_params(frozen_params)
    print(f'(lora) num params:      {num_lora_params:_d}')
    print(f'(trainable) num params: {num_train_params:_d}')
    print(f'(frozen) num params:    {num_frozen_params:_d}')

    # 6. Instatiate optimize and learning rate schedule.
    opt_schedule = optax.constant_schedule(lr)
    opt = optax.adamw(opt_schedule)

    # 7. Bring alltogether and run gradient descent.
    # TODO(@daskol): Default `TrainState` does not distinguish between
    # trainable and frozen weights. Also, it does not accept `mask` and `rngs`.
    state = TrainState.create(apply_fn=apply_lora,
                              params=trainable_params,
                              tx=opt)
    state = fit(state, mask, frozen_params, rngs, input_ids, labels)


def fit(state: TrainState, mask: Mask, frozen_params: Params,
        rngs: dict[str, Array], inputs: Array, outputs: Array) -> TrainState:
    for it in range(10):
        state, loss = \
            fit_batch(state, mask, frozen_params, rngs, inputs, outputs)
        print(f'[{it: 2d}] loss={loss:e}')
    return state


def fit_batch(state: TrainState, mask: Mask, frozen_params, rngs, inputs,
              outputs) -> tuple[TrainState, Array]:
    @partial(jax.value_and_grad, has_aux=True)
    def objective(params):
        all_params = merge(params, frozen_params)
        logits = state.apply_fn(all_params, inputs, rngs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, outputs)
        return loss.mean(), {}

    (loss, _), grads = objective(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def main():
    ns: Namespace = parser.parse_args()
    train(ns.lr, ns.lora_rank, ns.lora_alpha)


if __name__ == '__main__':
    main()
