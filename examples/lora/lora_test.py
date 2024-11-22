from operator import add

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose
from transformers import FlaxRobertaForSequenceClassification as RoBERTa

from lora import Params, lora, mask_by_prefix
from yax import eval_mox, make_mox, query


class Model(nn.Module):
    @nn.compact
    def __call__(self, xs):
        return xs + nn.Dense(4)(xs)


def test_simple():
    # 0. Create model and initialize weights.
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    model = Model()
    params = model.init(subkey, jnp.empty(4))

    # 1. Make MoX from application function on probing input.
    mox = make_mox(model.apply)(params, jnp.empty(4))

    # 2. Decide what modules (layers) should be LoRAfied.
    key, subkey = jax.random.split(key)
    xpath = '//[@type="Dense"]'
    nodes = query(xpath, mox)
    assert len(nodes) == 1

    # 3. Apply LoRA transformation and build an application function for
    #    LoRAfied model.
    lora_mox, lora_params, mask, merge = \
        lora(subkey, xpath, mox, params, rank=2, alpha=10)
    assert 'LoRA_0' in lora_params['params']
    dense = params['params']['Dense_0']
    dense_frozen = lora_params['params']['LoRA_0'].get('Dense_0')
    assert dense == dense_frozen
    assert jax.tree.reduce(lambda x, y: x + y, mask) == 2

    @jax.jit
    def apply_lora(params: Params, *inputs):
        return eval_mox(lora_mox, params, *inputs)

    actual = apply_lora(lora_params, jnp.ones(4))
    desired = model.apply(params, jnp.ones(4))
    assert_allclose(actual, desired)

    # 4. Merge LoRA-adapter to original fully-connected layer.
    merged_params = merge(lora_params)
    flat_merged_params, merged_tree = jax.tree_flatten(merged_params)
    flat_params, tree = jax.tree_flatten(params)
    assert merged_tree == tree
    for actual, desired in zip(flat_merged_params, flat_params):
        assert_allclose(actual, desired)
    _ = model.apply(params, jnp.ones(4))


@pytest.mark.slow
def test_roberta():
    # 0. Load model and create application function (HuggingFace stores weights
    #    as a part of the model).
    model = RoBERTa.from_pretrained('roberta-base')
    params = {'params': model.params}

    def apply_fn(params: Params, input_ids: jax.Array, dropout_rng=None,
                 train: bool = True):
        return model(params=params['params'], input_ids=input_ids,
                     dropout_rng=dropout_rng, train=train)

    # 1. Make MoX from application function on probing input.
    input_ids = jnp.ones((1, 3), dtype=jnp.int32)
    key = jax.random.PRNGKey(42)
    mox = make_mox(apply_fn)(params, input_ids, key)

    def apply(params: Params, input_ids: jax.Array, dropout_rng):
        return eval_mox(mox, params, input_ids, dropout_rng)

    desired = apply(params, input_ids, key)

    # 2. Decide what modules (layers) should be LoRAfied.
    xpath = '//[@name="attention"]//[@type="Dense"]'
    nodes = query(xpath, mox)
    assert len(nodes) == 4 * 12

    # 3. Apply LoRA transformation and build an application function for
    #    LoRAfied model.
    key = jax.random.PRNGKey(42)
    mox_lora, params_lora, mask, merge = \
        lora(key, xpath, mox, params, rank=2, alpha=10)
    assert jax.tree.reduce(lambda x, y: x + y, mask) == 2 * len(nodes)

    def apply_lora(params: Params, input_ids: jax.Array, dropout_rng):
        return eval_mox(mox_lora, params, input_ids, dropout_rng)

    actual = apply_lora(params_lora, input_ids, key)
    assert_allclose(actual, desired)

    # 4. Merge LoRA-adapter to original fully-connected layer.
    params_merged = merge(params_lora)
    flat_params_merged, merged_tree = jax.tree_flatten(params_merged)
    flat_params, tree = jax.tree_flatten(params)
    assert merged_tree == tree
    for actual, desired in zip(flat_params_merged, flat_params):
        assert_allclose(actual, desired)
    _ = apply_fn(params, input_ids, key)


def test_mask_dummy():
    leaf = jnp.empty(())
    mask = mask_by_prefix(('a', ), {'a': {'x': leaf}, 'b': {'y': leaf}})
    assert mask == {'a': {'x': True}, 'b': {'y': False}}
    assert jax.tree.reduce(add, mask) == 1


def test_mask_simple():
    params = {
        'params': {
            'roberta': {
                'Dense_0': {
                    'kernel': jnp.empty((2, 2)),
                    'bias': jnp.empty((2, )),
                }
            },
            'classifier': {
                'Dense_1': {
                    'kernel': jnp.empty((2, 2)),
                    'bias': jnp.empty((2, )),
                }
            }
        }
    }
    mask = mask_by_prefix(['params', 'classifier'], params)
    assert jax.tree.reduce(add, mask) == 2
