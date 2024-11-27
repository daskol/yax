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
"""Module `lora` implements LoRA-adapter itself and routines to patch model
with LoRA-adapters and roll them back.
"""

from functools import reduce, wraps
from itertools import count
from time import monotonic
from typing import Any, Callable, Sequence, TypeAlias, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import lecun_normal, zeros_init
from flax.typing import Dtype, Initializer, PrecisionLike
from jax.tree_util import DictKey, tree_map_with_path

from yax import Expr, Mox, XPath, make_mox, sub

__all__ = ('LoRA', 'Mask', 'MergeFn', 'Params', 'lora', 'mask_by_prefix',
           'merge', 'split')

KeyArray: TypeAlias = jax.Array

# TODO(@daskol): `type` statement.
Mask: TypeAlias = dict[str, Union['Mask', bool]]

Params: TypeAlias = dict[str, Any]

MergeFn = Callable[[Params], Params]

XPathLike: TypeAlias = XPath | str


class LoRA(nn.Module):
    """Low-Rank Adaptation (LoRA) module that applies a low-rank correction to
    the original :py:class:`flax.linen.Dense` module.

    [1]: https://openreview.net/forum?id=nZeVKeeFYf9
    """

    features: int
    rank: int
    alpha: float = 1.0
    original_name: str | None = None
    lhs_init: Initializer = lecun_normal()
    rhs_init: Initializer = zeros_init()

    # The rest of parameters corresponds to `flax.linen.Dense`.
    use_bias: bool = True
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Initializer = lecun_normal()
    bias_init: Initializer = zeros_init()

    @nn.compact
    def __call__(self, xs):
        in_features = jnp.shape(xs)[-1]

        lhs_shape = (in_features, self.rank)
        lhs = self.param('lhs', self.lhs_init, lhs_shape, self.param_dtype)

        rhs_shape = (self.rank, self.features)
        rhs = self.param('rhs', self.rhs_init, rhs_shape, self.param_dtype)

        kwargs = dict(use_bias=self.use_bias, dtype=self.dtype,
                      param_dtype=self.dtype, precision=self.precision,
                      kernel_init=self.kernel_init, bias_init=self.bias_init)
        if self.original_name:
            kwargs['name'] = self.original_name

        # TODO(@daskol): Account for `precision` and `dtype` for computations.
        return nn.Dense(self.features, **kwargs)(xs) + \
               (self.alpha / self.rank) * (xs @ lhs) @ rhs


def lora(key: KeyArray, xpath: XPathLike, mox: Mox, params: Params,
         rank: int = 2, alpha: float = 1.0, instrument: bool = False,
         **kwargs) -> tuple[Mox, Params, Mask, MergeFn]:
    """Apply LoRA-adapter modules specified by `xpath` in `mox` with `rank`."""
    lora_ix = (f'LoRA_{ix}' for ix in count())  # Name LoRA-adapters.
    param_bl = ('dot_general', 'dot_general_cls', 'features', 'name')
    lora_params = jax.tree.map(lambda x: x, params)
    lora_paths = []

    def sub_fn(path: tuple[str, ...], node: Expr) -> Expr:
        if not isinstance(node, Mox):
            return node

        # Validate node (briefly) and get input/output shapes.
        features: int
        if (features := node.params.get('features')) is None:
            raise RuntimeError(f'No `features` attribute in node: {node}.')
        inputs = node.inputs[node.var_tree.num_leaves:]
        in_shape = inputs[0].value.shape

        nonlocal key
        key, subkey = jax.random.split(key)

        # Name of the original module if exists.
        original_name = 'Dense_0'
        if len(path) > 1:
            *_, original_name = path

        # Initialize LoRA-adapter.
        adapter_params = {
            k: v
            for k, v in node.params.items() if k not in param_bl
        }
        adapter = LoRA(features, rank, alpha, original_name, **adapter_params)
        adapter_name = next(lora_ix)
        adapter_params = adapter.init(subkey, jnp.empty(in_shape))

        # Update weight tree in place.
        nonlocal lora_params
        dense_params = get_subcol(lora_params, path[1:])
        adapter_params['params'].pop(original_name)
        if len(path) > 1:
            set_subcol(adapter_params, (original_name, ), dense_params)
        else:
            set_subcol(adapter_params, (original_name, ), dense_params)
        update_subcol(lora_params, path[1:], adapter_name, adapter_params)

        # Record path to the current adapter to merge it back later.
        nonlocal lora_paths
        lora_path = (adapter_name, original_name)
        if len(path) > 1:
            lora_path = path[1:-1] + lora_path
        lora_paths += [lora_path]

        # Build a module expression for new LoRA-adapter.
        mox = make_mox(adapter.apply)(adapter_params, jnp.empty(in_shape))
        mox = mox.children[0]
        mox.params['name'] = adapter_name  # Must be unique in node.
        return mox

    if instrument:
        sub_fn = instrument(sub_fn)
    mox = sub(xpath, sub_fn, mox)

    def predicate(key_path: tuple[DictKey, ...], _) -> bool:
        # TODO(@daskol): Inefficient: too many iterations.
        key = tuple([x.key for x in key_path[1:]])
        if key[-1] not in ('lhs', 'rhs'):
            return False
        for path in lora_paths:
            if key[:-1] == path[:-1]:
                return True
        else:
            return False

    mask = tree_map_with_path(predicate, lora_params)

    def merge(params: Params) -> Params:
        """Merge LoRA factors back to frozen weights and restore original
        weight tree.

        Args:
          `params`: Weight tree of LoRA-fied model.

        Return:
          Merged parameter tree.
        """
        scale = alpha / rank
        for path in reversed(lora_paths):
            adapter_params = get_subcol(params, path[:-1])
            lhs = adapter_params['lhs']
            rhs = adapter_params['rhs']
            dense_params = get_subcol(params, path)
            dense_params['kernel'] += scale * (lhs @ rhs)
            set_subcol(params, path[:-2] + path[-1:], dense_params)
            del_subcol(params, path[:-1])
        return params

    return mox, lora_params, mask, merge


def elapsed(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        elapsed = -monotonic()
        res = fn(*args, **kwargs)
        elapsed += monotonic()
        print(f'elapsed in {elapsed:.3f} ({args[0]})')
        return res
    return inner


def del_subcol(where: Params, what: Sequence[str], collection: str = 'params'):
    if what:
        parent = get_subcol(where, what[:-1], collection=collection)
        parent.pop(what[-1])
    elif collection in where:
        where[collection] = {}


def get_subcol(where: Params, what: Sequence[str],
               collection: str = 'params') -> Params:
    return reduce(lambda x, y: x[y], (collection, ) + what, where)


def set_subcol(where: Params, what: Sequence[str], subcol: Params,
               collection: str = 'params'):
    if what:
        parent = get_subcol(where, what[:-1], collection=collection)
        parent[what[-1]] = subcol
    else:
        where[collection] = subcol


def update_subcol(where: Params, what: Sequence[str], name: str, repl: Params,
                  collection: str = 'params') -> Params:
    if what:
        parent = get_subcol(where, what[:-1], collection=collection)
        parent.pop(what[-1])
        parent[name] = repl[collection]
    else:
        where[collection] = repl[collection]


def mask_by_prefix(prefix: Sequence[str], params: Params) -> Mask:
    """Make a mask tree for a weights tree `params` by prefix `prefix`.

    >>> leaf = jnp.empty(())
    >>> mask_by_prefix(('a', ), {'a': {'x': leaf}, 'b': {'y': leaf}})
    {'a': {'x': True}, 'b': {'y': False}}
    """
    prefix = tuple(prefix)

    def fn(key_path: tuple[DictKey, ...], _):
        key = tuple([x.key for x in key_path])
        return key[:len(prefix)] == prefix

    return tree_map_with_path(fn, params)


def split(pred: Mask | Callable[..., Any],
          params: Params) -> tuple[Params, Params]:
    """Split a tree `params` on two according to mask `pred`.

    >>> tree = {'a': {'x': 0}, 'b': {'y': 0}}
    >>> mask = {'a': {'x': True}, 'b': {'y': False}}
    >>> lhs, rhs = split(mask, tree)
    >>> lhs
    {'a': {'x': 0}, 'b': {'y': None}}
    >>> rhs
    {'a': {'x': None}, 'b': {'y': 1}}
    """
    if callable(pred):
        mask: Mask = jax.tree.map(pred, params)
        return split(mask, params)
    lhs = jax.tree.map(lambda m, p: p if m else None, pred, params)
    rhs = jax.tree.map(lambda m, p: None if m else p, pred, params)
    return lhs, rhs


def merge(ps: Params, qs: Params):
    """Merge two trees of the same structure.

    >>> tree = {'a': {'x': 0}, 'b': {'y': 0}}
    >>> mask = {'a': {'x': True}, 'b': {'y': False}}
    >>> tree == merge(*split(mask, tree))
    """
    return jax.tree.map(lambda x, y: x if y is None else y, ps, qs,
                        is_leaf=lambda x: x is None)
