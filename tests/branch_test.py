# Copyright 2026 Daniel Bershatsky
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

from collections.abc import Mapping
from io import BytesIO, StringIO
from typing import Type, cast
from xml.etree import ElementTree

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from yax import (
    Branch, Mox, ShapedArray, Static, Symbol, Var, branch, dump, dump_xml,
    dump_yson, eval_mox, make_mox, map_param_tree, merge_branch_params, query,
    reconstruct, sub)


def absolute(xs):
    return jnp.abs(xs)


def negate(xs):
    return -xs


def make_eq(fn=negate, shape=(2,)):
    return make_mox(fn)(jnp.ones(shape)).children[0]


class DuplicateCases(Mapping):

    def __init__(self, items):
        self._items = tuple(items)

    def __iter__(self):
        return (k for k, v in self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, key):
        for k, v in self._items:
            if k is key:
                return v
        raise KeyError

    def items(self):
        return self._items


class Pass(nn.Module):

    @nn.compact
    def __call__(self, xs):
        return xs + 0.0


class SinglePass(nn.Module):

    @nn.compact
    def __call__(self, xs):
        return Pass()(xs)


class TwoPass(nn.Module):

    @nn.compact
    def __call__(self, xs):
        return Pass(name='left')(xs) + Pass(name='right')(xs)


class Scale(nn.Module):

    @nn.compact
    def __call__(self, xs):
        *_, features = xs.shape
        kernel = self.param('scale', nn.initializers.ones, (features, ) * 2)
        return xs @ kernel


class ScalePlus(nn.Module):

    @nn.compact
    def __call__(self, xs):
        *_, features = xs.shape
        kernel = self.param('scale', nn.initializers.ones, (features, ) * 2)
        return xs @ kernel + 1.0


class Shift(nn.Module):

    @nn.compact
    def __call__(self, xs):
        *_, features = xs.shape
        shift = self.param('shift', nn.initializers.ones, (features, ))
        return xs + shift


def module_case(module: nn.Module, xs):
    params = module.init(jax.random.key(0), xs)
    return make_mox(module.apply)(params, xs).children[0]


def pass_mox(xs):
    model = SinglePass()
    params = model.init(jax.random.key(1), xs)
    return make_mox(model.apply)(params, xs)


def test_reconstruct_empty():
    res = reconstruct({})
    assert res == {}


def test_reconstruct_dict_dict():
    res = reconstruct({('a', 'a'): 1, ('a', 'b'): 2, ('b', 'a'): 3})
    assert res['a']['a'] == 1
    assert res['a']['b'] == 2
    assert res['b']['a'] == 3


def test_reconstruct_dict_list():
    res = reconstruct({('a', 0): 1, ('a', 1): 2, ('b', 'a'): 3})
    assert res['a'] == [1, 2]
    assert res['b']['a'] == 3


def test_reconstruct_list_dict():
    res = reconstruct({(0, 'a'): 1, (0, 'b'): 2, (1, 'a'): 3})
    assert res[0]['a'] == 1
    assert res[0]['b'] == 2
    assert res[1]['a'] == 3


def test_reconstruct_interleaved():
    with pytest.raises(RuntimeError, match='interleave together'):
        reconstruct({('a', 'a'): 1, ('a', 'b'): 2, ('a', ): 3})


def test_map_param_tree():
    xs = {'params': {'kernel': jnp.eye(2)}}
    ys = {'params': {'bias': jnp.zeros(2)}}

    def fn(leaves):
        for leaf in (x for x in leaves if x is not None):
            return leaf
        else:
            raise RuntimeError(f'No defined leaves: {leaves}.')

    res = map_param_tree(fn, (xs, ys))
    assert_array_equal(res['params']['kernel'], xs['params']['kernel'])
    assert_array_equal(res['params']['bias'], ys['params']['bias'])


def test_merge_branch_params():
    xs = jnp.empty((2, ))

    def make_expr(module_ty: Type[nn.Module]):
        module = module_ty()
        params = module.init(jax.random.key(42), xs)
        mox = make_mox(module.apply)(params, xs)
        return mox.children[0]

    scale = make_expr(Scale)
    shift = make_expr(Shift)

    leaves, treedef, mappings = \
        merge_branch_params({np.bool_(True): scale, False: shift})
    variables = jax.tree.unflatten(treedef, leaves)

    symbol: Symbol = variables['params']['scale']
    value: ShapedArray = cast(ShapedArray, symbol.value)
    assert value.shape == (2, 2)
    assert value.dtype == jnp.float32

    symbol: Symbol = variables['params']['shift']
    value: ShapedArray = cast(ShapedArray, symbol.value)
    assert value.shape == (2, )
    assert value.dtype == jnp.float32


def test_branch_key_normalization():
    expr = make_eq()
    node = branch('mode', {np.bool_(True): expr, False: expr})

    assert tuple(node.cases) == (True, False)
    assert isinstance(node.inputs[node.var_tree.num_leaves], Static)

    with pytest.raises(ValueError, match='Duplicate static branch key'):
        branch('mode', DuplicateCases([(True, expr), (np.bool_(True), expr)]))
    with pytest.raises(TypeError, match='Unsupported static branch key'):
        branch('mode', {1: expr})
    with pytest.raises(ValueError, match='at least one case'):
        branch('mode', {})


def test_branch_boundary_validation():
    with pytest.raises(RuntimeError, match='input symbol shapes differ'):
        branch('mode', {'a': make_eq(shape=(2,)), 'b': make_eq(shape=(3,))})

    def first(xs):
        return xs[:1]

    with pytest.raises(RuntimeError, match='output symbol shapes differ'):
        branch('mode', {'a': make_eq(), 'b': make_eq(first)})


def test_dump():
    expr = make_eq()
    node = branch('mode', {np.bool_(True): expr, False: expr})
    buf = StringIO()
    dump(node, buf)
    output = buf.getvalue()

    # inputs =[Static(value='mode'), ...]
    # outputs=[...]
    # branch None : static_branch(selector=mode, cases=(True, False)) { # 0
    #   case True { ... }
    #   case False { ... }
    # }
    assert 'Static' in output
    assert '\'mode\'' in output
    assert 'branch' in output
    assert 'static_branch' in output
    assert 'case False {' in output
    assert 'case True {' in output


def test_dump_xml():
    expr = make_eq()
    node = branch('mode', {np.bool_(True): expr, False: expr})
    buf = StringIO()
    dump_xml(node, buf)
    root = ElementTree.fromstring(buf.getvalue())
    assert root.tag == 'static_branch'
    assert root.attrib['selector'] == 'mode'


def test_dump_yson():
    yt_yson = pytest.importorskip('yt.yson')

    expr = make_eq()
    node = branch('mode', {np.bool_(True): expr, False: expr})
    buf = BytesIO()

    dump_yson(node, buf)
    obj = yt_yson.loads(buf.getvalue())
    assert obj.attributes['primitive'] == 'static_branch'
    assert obj.attributes['selector'] == 'mode'
    assert len(obj) == 2


def test_replace_module_with_branch_and_eval_modes():
    xs = jnp.ones((2,))
    mox = pass_mox(xs)
    replacement = branch('mode', {
        'scale': module_case(Scale(), xs),
        'shift': module_case(Shift(), xs),
    })

    sub('//[@type="Pass"]', replacement, mox)
    variables = {
        'params': {
            'Pass_0': {
                'scale': jnp.eye(xs.shape[-1]) * 3.0,
                'shift': jnp.full(xs.shape, 5.0),
            },
        },
    }

    assert_allclose(eval_mox(mox, variables, xs, mode='scale'), xs * 3.0)
    assert_allclose(eval_mox(mox, variables, xs, mode='shift'), xs + 5.0)


def test_only_selected_case_runs_with_shared_outputs():
    xs = jnp.array([-1.0, 2.0])
    mox = pass_mox(xs)
    replacement = branch('mode', {
        'neg': make_eq(negate),
        'abs': make_eq(absolute),
    })

    sub('//[@type="Pass"]', replacement, mox)

    # Both cases share output symbols after branch construction. Evaluating
    # both cases would attempt to write the same symbol twice.
    assert_allclose(eval_mox(mox, {}, xs, mode='neg'), negate(xs))
    assert_allclose(eval_mox(mox, {}, xs, mode='abs'), absolute(xs))


def test_shared_parameter_path_uses_one_runtime_leaf():
    xs = jnp.ones((2,))
    replacement = branch('mode', {
        'scale': module_case(Scale(), xs),
        'plus': module_case(ScalePlus(), xs),
    })

    assert replacement.var_tree.num_leaves == 1
    [shared] = replacement.inputs[:replacement.var_tree.num_leaves]
    for case in replacement.cases.values():
        assert case.inputs[0] is shared

    mox = pass_mox(xs)
    sub('//[@type="Pass"]', replacement, mox)

    variables = {'params': {'Pass_0': {'scale': jnp.eye(xs.shape[-1]) * 2.0}}}
    assert_allclose(eval_mox(mox, variables, xs, mode='scale'), xs * 2.0)
    assert_allclose(eval_mox(mox, variables, xs, mode='plus'), xs * 2.0 + 1.0)

    variables = {'params': {'Pass_0': {'scale': jnp.eye(xs.shape[-1]) * 4.0}}}
    assert_allclose(eval_mox(mox, variables, xs, mode='scale'), xs * 4.0)
    assert_allclose(eval_mox(mox, variables, xs, mode='plus'), xs * 4.0 + 1.0)


def test_same_symbol_alias_in_one_tree_is_rejected():
    param = Var(ShapedArray((2,), jnp.float32))
    xs = Var(ShapedArray((2,), jnp.float32))
    ys = Var(ShapedArray((2,), jnp.float32))
    params, var_tree = jax.tree.flatten({'params': {'a': param, 'b': param}})
    case = Mox(params + [xs], [ys], {}, var_tree=var_tree)

    with pytest.raises(RuntimeError, match='multiple variable paths'):
        branch('mode', {'a': case, 'b': case})


def test_same_symbol_alias_across_case_paths_is_rejected():
    param = Var(ShapedArray((2,), jnp.float32))
    xs = Var(ShapedArray((2,), jnp.float32))
    ys = Var(ShapedArray((2,), jnp.float32))
    lhs_params, lhs_tree = jax.tree.flatten({'params': {'a': param}})
    rhs_params, rhs_tree = jax.tree.flatten({'params': {'b': param}})
    lhs = Mox(lhs_params + [xs], [ys], {}, var_tree=lhs_tree)
    rhs = Mox(rhs_params + [xs], [ys], {}, var_tree=rhs_tree)

    with pytest.raises(RuntimeError, match='multiple variable paths'):
        branch('mode', {'lhs': lhs, 'rhs': rhs})


def test_incompatible_repeated_parameter_path_is_rejected():
    xs = Var(ShapedArray((2,), jnp.float32))
    ys = Var(ShapedArray((2,), jnp.float32))
    lhs_param = Var(ShapedArray((2,), jnp.float32))
    rhs_param = Var(ShapedArray((3,), jnp.float32))
    lhs_params, lhs_tree = jax.tree.flatten({'params': {'scale': lhs_param}})
    rhs_params, rhs_tree = jax.tree.flatten({'params': {'scale': rhs_param}})
    lhs = Mox(lhs_params + [xs], [ys], {}, var_tree=lhs_tree)
    rhs = Mox(rhs_params + [xs], [ys], {}, var_tree=rhs_tree)

    with pytest.raises(RuntimeError, match='Parameter symbol shapes differ'):
        branch('mode', {'lhs': lhs, 'rhs': rhs})


def test_incompatible_repeated_parameter_path_dtype_is_rejected():
    xs = Var(ShapedArray((2,), jnp.float32))
    ys = Var(ShapedArray((2,), jnp.float32))
    lhs_param = Var(ShapedArray((2,), jnp.float32))
    rhs_param = Var(ShapedArray((2,), jnp.int32))
    lhs_params, lhs_tree = jax.tree.flatten({'params': {'scale': lhs_param}})
    rhs_params, rhs_tree = jax.tree.flatten({'params': {'scale': rhs_param}})
    lhs = Mox(lhs_params + [xs], [ys], {}, var_tree=lhs_tree)
    rhs = Mox(rhs_params + [xs], [ys], {}, var_tree=rhs_tree)

    with pytest.raises(RuntimeError, match='Parameter symbol dtypes differ'):
        branch('mode', {'lhs': lhs, 'rhs': rhs})


def test_multiple_branches_share_one_root_selector():
    xs = jnp.array([-1.0, 2.0])
    model = TwoPass()
    params = model.init(jax.random.key(0), xs)
    mox = make_mox(model.apply)(params, xs)
    replacement = branch('mode', {
        'neg': make_eq(negate),
        'abs': make_eq(absolute),
    })

    sub('//[@type="Pass"]', replacement, mox)

    selectors = [sym for sym in mox.inputs if isinstance(sym, Static)]
    assert len(selectors) == 1
    assert selectors[0].name == 'mode'
    assert_allclose(eval_mox(mox, {}, xs, mode='neg'), negate(xs) * 2.0)
    assert_allclose(eval_mox(mox, {}, xs, mode='abs'), absolute(xs) * 2.0)


def test_bool_selector_and_eval_errors():
    xs = jnp.ones((2,))
    mox = pass_mox(xs)
    sub('//[@type="Pass"]', branch('flag', {
        True: make_eq(negate),
        False: make_eq(absolute),
    }), mox)

    assert_allclose(eval_mox(mox, {}, xs, flag=True), negate(xs))
    assert_allclose(eval_mox(mox, {}, xs, flag=np.bool_(False)), absolute(xs))
    with pytest.raises(KeyError, match='Missing static branch selector'):
        eval_mox(mox, {}, xs)
    with pytest.raises(ValueError, match='Unsupported value'):
        eval_mox(mox, {}, xs, flag='bad')
    with pytest.raises(TypeError, match='must be a Python str or bool'):
        eval_mox(mox, {}, xs, flag=jnp.array(True))


def test_selector_conflict_with_existing_non_static_kwarg():
    def fn(xs, mode: bool = False):
        if mode:
            return jnp.abs(xs)
        return -xs

    xs = jnp.array([-1.0, 2.0])
    mox = make_mox(fn)(xs, mode=False)
    replacement = branch('mode', {
        'neg': make_eq(negate),
        'abs': make_eq(absolute),
    })

    with pytest.raises(RuntimeError, match='already exists'):
        sub('//jit', replacement, mox)


def test_parameterized_branch_cannot_replace_primitive():
    xs = jnp.ones((2,))
    mox = make_mox(negate)(xs)
    replacement = branch('mode', {
        'scale': module_case(Scale(), xs),
        'plus': module_case(ScalePlus(), xs),
    })

    with pytest.raises(NotImplementedError, match='primitive Equation'):
        sub('//jit', replacement, mox)


def test_branch_query_and_dumps():
    xs = jnp.ones((2,))
    mox = pass_mox(xs)
    sub('//[@type="Pass"]', branch('mode', {
        'neg': make_eq(negate),
        'abs': make_eq(absolute),
    }), mox)

    [node] = query('//static_branch', mox)
    assert isinstance(node, Branch)
    assert len(query('//jit', mox)) == 2
    assert node.to_dict(False)['primitive'] == 'static_branch'

    buf = StringIO()
    dump(mox, buf)
    assert 'static_branch' in buf.getvalue()

    buf = StringIO()
    dump_xml(mox, buf)
    assert ElementTree.fromstring(buf.getvalue()).find('.//static_branch') \
        is not None

    yt_yson = pytest.importorskip('yt.yson')
    buf = BytesIO()
    dump_yson(mox, buf)
    obj = yt_yson.loads(buf.getvalue())
    assert b'static_branch' in yt_yson.dumps(obj)


def test_sub_inside_existing_branch_case_is_unsupported():
    xs = jnp.ones((2,))
    mox = pass_mox(xs)
    sub('//[@type="Pass"]', branch('mode', {
        'neg': make_eq(negate),
        'abs': make_eq(absolute),
    }), mox)

    with pytest.raises(NotImplementedError, match='inside static branch'):
        sub('//jit', make_eq(absolute), mox)
