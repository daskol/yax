import flax.linen as nn
import jax
import jax.numpy as jnp
from lora import Params, lora
from numpy.testing import assert_allclose

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
    mox = make_mox(model.apply)(params, jnp.empty(4))

    # 1. Apply LoRA-transformation to all dense modules.
    key, subkey = jax.random.split(key)
    xpath = '//[@type="Dense"]'
    nodes = query(xpath, mox)
    assert len(nodes) == 1
    lora_mox, lora_params, merge = lora(subkey, xpath, mox, params, rank=2,
                                        alpha=10)
    assert 'LoRA_0' in lora_params['params']
    dense = params['params']['Dense_0']
    dense_frozen = lora_params['params']['LoRA_0'].get('Dense_0')
    assert dense == dense_frozen

    # 2. Evaluate LoRAfied model to sample input.
    @jax.jit
    def apply_lora(params: Params, *inputs):
        return eval_mox(lora_mox, params, *inputs)

    actual = apply_lora(lora_params, jnp.ones(4))
    desired = model.apply(params, jnp.ones(4))
    assert_allclose(actual, desired)

    # 3. Merge LoRA-adapter to original fully-connected layer.
    merged_params = merge(lora_params)
    flat_merged_params, merged_tree = jax.tree_flatten(merged_params)
    flat_params, tree = jax.tree_flatten(params)
    assert merged_tree == tree
    for actual, desired in zip(flat_merged_params, flat_params):
        assert_allclose(actual, desired)
    _ = model.apply(params, jnp.ones(4))
