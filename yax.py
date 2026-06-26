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

"""Module `yax` implements module tracing routines, module expression (MoX)
building, querying, and mutation.
"""

import re
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field, fields
from functools import partial, wraps
from io import StringIO
from json import dumps
from typing import (
    IO, Any, ClassVar, Generic, Mapping, ParamSpec, Self, Sequence, Type,
    TypeAlias, TypeVar, cast)
from xml.etree.ElementTree import (
    Element, ElementTree, SubElement, indent as indent_etree)

import flax
import flax.linen as nn
import jax
import jax.extend as jex
import jax.extend.linear_util as lu
import jax.numpy as jnp
import numpy as np
from flax.core.scope import LazyRng
from flax.linen.module import Callable, InterceptorContext, intercept_methods
from flax.typing import RNGSequences
from jax._src.core import set_current_trace  # noqa: PLC2701
from jax.core import AbstractValue, ShapedArray, Trace, Tracer, find_top_trace
from jax.extend.core import ClosedJaxpr as Jaxpr
from jax.tree_util import DictKey, SequenceKey

# TODO(@daskol): Make PR on reexporting PyTreeDef.
try:
    from jax.tree import PyTreeDef  # type: ignore[attr-defined]
except ImportError:
    from jax.tree_util import PyTreeDef

__all__ = ('Branch', 'Equation', 'Expr', 'Literal', 'Mox', 'Static',
           'Symbol', 'Var', 'branch', 'dump_yson', 'dump_xml', 'eval_mox',
           'make_mox', 'map_mox', 'query', 'sub')

# TODO(@daskol): Python 3.12 introduced new type parameter syntax (PEP-0695)
# but some code quality tools (e.g. yapf) do not support this syntax.
Args = ParamSpec('Args')

ArrayTree: TypeAlias = Mapping[str, Any]


class ModuleTracer(Tracer):

    __slots__ = 'value'

    def __init__(self, trace: Trace, value):
        super().__init__(trace)
        self.value = value

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other) -> bool:
        return self is other

    @property
    def aval(self) -> AbstractValue:
        # TODO(@daskol): Some abstract values should be less abstract. There is
        # cases when we should preserve original concrete values. For example,
        # arguments of binary operation between tracer and numerical literal
        # like `x > 0`. In this case `x` should be evaluated as `ShapedArray`
        # and zero as a literal. Otherwise, this constant values get missing
        # in evaluation time.
        if isinstance(self.value, AbstractValue):
            return self.value
        elif isinstance(self.value, Tracer):
            return self.value.aval
        else:
            return jax.typeof(self.value)

    def full_lower(self):
        # TODO(@daskol): How to properly implement lowering?
        #
        # The idea is that we need only tracers for abstract avaluation. Thus,
        # we should not lower tracer to anything other than a Tracer (actually,
        # a ModuleTracer). It seems that we should `full_lower` everywhere
        # instead of `full_raise`.
        match self.value:
            case ModuleTracer():
                return self.value.full_lower()
            case Tracer():
                raise RuntimeError(
                    'Unreachable executation path: full lowering expects '
                    'value to be either ModuleTracer or an Array '
                    f'implementation but actual type is {type(self.value)}.')
            case _:
                return self


class ModuleTrace(Trace[ModuleTracer]):

    def __init__(self, *, builder: 'MoxBuilder', **kwargs) -> None:
        super().__init__()
        self.builder: MoxBuilder = builder

    def pure(self, val) -> ModuleTracer:
        """Wrap value to monadic/functorial context.

        Our `pure` differ from original implementation in a way to work not
        only with Arrays but with AbstractValues. The idea is to interpret
        concrete arrays as literals while shaped arrays are just variables.
        Thus, we abstractify input arrays (arguments) manually.
        """
        if isinstance(val, AbstractValue):
            aval = val
        else:
            aval = jax.typeof(val)
        return ModuleTracer(self, aval)

    def literal(self, val) -> ModuleTracer:
        """Wrap a concrete value so it is preserved as a MoX literal."""
        return ModuleTracer(self, val)

    def lift(self, tracer: Tracer) -> ModuleTracer:
        return ModuleTracer(self, tracer)

    def sublift(self, tracer: Tracer) -> ModuleTracer:
        return ModuleTracer(self, tracer)

    def full_raise(self, val) -> ModuleTracer:
        if isinstance(val, ModuleTracer):
            if val._trace is self:
                return val
            return ModuleTracer(self, val)
        if isinstance(val, Tracer):
            return ModuleTracer(self, val)
        return self.literal(val)

    def process_primitive(self, prim: jex.core.Primitive, tracers, params):
        tracers = [self.full_raise(x) for x in tracers]
        result, _ = prim.abstract_eval(*[x.aval for x in tracers], **params)

        outs, out_tree = jax.tree.flatten(result)
        assert all(isinstance(x, AbstractValue) for x in outs if x), \
            'Assumption on type of result is violated: {outs}.'

        out_flat_tracers = [ModuleTracer(self, x) for x in outs]
        out_tracers = jax.tree.unflatten(out_tree, out_flat_tracers)
        self.builder.append(prim, params, tracers, out_flat_tracers)
        return out_tracers

    def process_custom_jvp_call(self, primitive, fun, jvp, tracers, **kwargs):
        del primitive, jvp, kwargs
        # TODO(@daskol): Why reference implementations in JAX create a
        # subtracer? Is it just purely debugging think? It seems that sublevels
        # helps to find leaking tracers. This can be highly important for
        # partial evauation but here we just forward tracers to our
        # `process_primitive` method.
        #
        # We would enforce subtracers here. Then one should uncomment the
        # following. The only issue is that its unclear how to process
        # `out_tracers`. Should we `full_lower` them and wrap up them again
        # with outer tracer?
        #
        #   with jax.core.new_sublevel():
        #       trace: ModuleTrace = self.main.with_cur_sublevel()
        #       in_tracers = [trace.sublift(t) for t in tracers]
        #       out_tracers = fun.call_wrapped(*in_tracers)
        #       return [trace.full_raise(t) for t in out_tracers]
        tracers = [self.full_raise(x) for x in tracers]
        with set_current_trace(self):
            return fun.call_wrapped(*tracers)


def _is_static_input(val) -> bool:
    return (
        not isinstance(val, AbstractValue | Tracer)
        and isinstance(val, bool | np.bool_)
    )


@lu.transformation
def _trace(builder: 'MoxBuilder', *ins):
    trace = ModuleTrace(builder=builder)

    with set_current_trace(trace, True):
        # NOTE We manually abstractify up to ShapedArray all input arguments.
        in_tracers, call_args = [], []
        for x in ins:
            if _is_static_input(x):
                in_tracers.append(trace.literal(x))
                call_args.append(x)
            else:
                in_tracers.append(
                    trace.pure(jax.api_util.shaped_abstractify(x)))
                call_args.append(in_tracers[-1])
        builder.set_inputs(in_tracers)
        outs = yield call_args, {}

        # TODO(@daskol): We do not use output tracers. Should we remove them?
        # This issue is related to the proper implementation of (sub)lifting
        # and unboxing that still have quite vague semantics.
        out_tracers = [trace.full_raise(t) for t in outs]
        builder.set_outputs(out_tracers)
        yield out_tracers


def trace_modules(wf: lu.WrappedFun, builder: 'MoxBuilder') -> lu.WrappedFun:
    return _trace(wf, builder)


SymbolValueType = TypeVar('SymbolValueType')


@dataclass(slots=True)
class Symbol(Generic[SymbolValueType]):
    """A value placeholder in evaluation time.

    Derived dataclasses should not generate `__eq__` to preserve `__hash__`.
    """

    value: SymbolValueType

    def __eq__(self, other) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    @property
    def aval(self) -> AbstractValue:
        if isinstance(self.value, AbstractValue):
            return self.value
        return jax.typeof(self.value)


@dataclass(slots=True, eq=False)
class Var(Symbol[ShapedArray]):
    pass


@dataclass(slots=True, eq=False)
class Literal(Symbol[Any]):
    aval: AbstractValue

    @property
    def const(self):
        return self.value


@dataclass(slots=True, eq=False)
class Static(Symbol[str]):
    """A named Python-static input selected at evaluation time."""

    @property
    def name(self) -> str:
        return self.value


# Check hashability of Symbol hierarchy.
_ = {
    Static('kwarg'),
    Symbol(ShapedArray((), jnp.float32)),
    Var(ShapedArray((), jnp.float32)),
    Literal(jnp.empty(()), ShapedArray((), jnp.float32)),
}


def default_treedef(default=()) -> PyTreeDef:
    _, treedef = jax.tree.flatten(default)
    return treedef


@dataclass(slots=True)
class Expr:
    """A base type that represents a module tree structure."""

    inputs: list[Symbol]
    outputs: list[Symbol]
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(slots=True)
class Equation(Expr):
    """A leaf of module tree that refers to a JAX
    :class:`jax.extend.core.Primitive`.
    """

    prim: jex.core.Primitive

    def to_dict(self, recursively=True) -> dict[str, Any]:
        return {**self.params, 'primitive': self.prim.name}


BranchKey: TypeAlias = str | bool


@dataclass(slots=True)
class Branch(Expr):
    """Static branching expression."""

    selector_name: str
    selector: Static
    cases: dict[BranchKey, Expr]
    var_tree: PyTreeDef = field(default_factory=default_treedef)

    def to_dict(self, recursively=True) -> dict[str, Any]:
        res = {
            **self.params,
            'primitive': 'static_branch',
            'selector': self.selector_name,
            'cases': tuple(self.cases.keys()),
        }
        if recursively:
            res['children'] = {
                key: val.to_dict() if hasattr(val, 'to_dict') else {}
                for key, val in self.cases.items()
            }
        return res


@dataclass(slots=True)
class Mox(Expr):
    children: list[Expr] = field(default_factory=list)
    module_ty: Type[nn.Module] | None = None
    entrypoint: str | None = None
    in_tree: PyTreeDef = field(default_factory=default_treedef)
    out_tree: PyTreeDef = field(default_factory=default_treedef)
    var_tree: PyTreeDef = field(default_factory=default_treedef)
    rngs: RNGSequences = field(default_factory=dict)

    @property
    def is_ephemeral(self) -> bool:
        """Ephemeral module expression does not reflect any real
        :class:`flax.linen.Module`.
        """
        return self.module_ty is None or self.entrypoint is None

    def __repr__(self) -> str:
        buf = StringIO()
        dump(self, buf)
        return buf.getvalue()

    def to_dict(self, recursively=True) -> dict[str, Any]:
        res = {
            **self.params,
            'primitive': 'module_call',
            'entrypoint': self.entrypoint,
            'ephemeral': self.is_ephemeral,
        }
        if self.module_ty:
            res['type'] = self.module_ty.__name__
        if recursively:
            children = []
            for child in self.children:
                children.append(child.to_dict())
            res['children'] = children
        return res

    def to_json(self, indent: int | None = None) -> str:
        def default(obj):
            if isinstance(obj, Jaxpr):
                return obj.pretty_print(use_color=False)
            return getattr(obj, '__name__', str(obj))
        return dumps(self.to_dict(), ensure_ascii=False, indent=indent,
                     default=default)


def param_count(expr: Expr) -> int:
    if isinstance(expr, Mox | Branch):
        return expr.var_tree.num_leaves
    return 0


def external_inputs(expr: Expr) -> list[Symbol]:
    num_params = param_count(expr)
    if isinstance(expr, Branch):
        return expr.inputs[num_params + 1:]
    return expr.inputs[num_params:]


def external_params(expr: Expr) -> list[Symbol]:
    num_params = param_count(expr)
    if isinstance(expr, Branch):
        return expr.inputs[:num_params + 1]
    return expr.inputs[:num_params]


def external_param_tree(expr: Expr) -> ArrayTree:
    leaves = external_params(expr)
    match expr:
        case Equation():
            return {}
        case Mox():
            node = cast(Mox, expr)
        case Branch():
            node = cast(Branch, expr)
        case _:
            raise RuntimeError(
                f'No treedef for an object of type {type(expr).__name__}.')
    return jax.tree.unflatten(node.var_tree, leaves)


def set_external_inputs(expr: Expr, inputs: Sequence[Symbol]) -> None:
    num_params = param_count(expr)
    if isinstance(expr, Branch):
        expr.inputs = expr.inputs[:num_params + 1] + [*inputs]
    else:
        expr.inputs = expr.inputs[:num_params] + [*inputs]


def clone_symbol(sym: Symbol) -> Symbol:
    if isinstance(sym, Static):
        return Static(sym.value)
    if isinstance(sym, Literal):
        return Literal(sym.value, sym.aval)
    if isinstance(sym, Var):
        return Var(sym.value)
    return Symbol(sym.value)


def clone_expr(expr: Expr, memo: dict[Symbol, Symbol] | None = None) -> Expr:
    if memo is None:
        memo = {}

    def clone(sym: Symbol) -> Symbol:
        if sym not in memo:
            memo[sym] = clone_symbol(sym)
        return memo[sym]

    obj = copy(expr)
    obj.inputs = [clone(s) for s in expr.inputs]
    obj.outputs = [clone(s) for s in expr.outputs]
    obj.params = {**expr.params}
    if isinstance(expr, Mox):
        obj.children = [clone_expr(child, memo) for child in expr.children]
        obj.rngs = {
            name: LazyRng(clone(rng.rng), rng.suffix)
            for name, rng in expr.rngs.items()
        }
    elif isinstance(expr, Branch):
        obj.selector = clone(expr.selector)
        obj.cases = {
            key: clone_expr(case, memo)
            for key, case in expr.cases.items()
        }
    return obj


def replace_symbols(expr: Expr, mapping: Mapping[Symbol, Symbol]) -> None:
    def replace(sym: Symbol) -> Symbol:
        return mapping.get(sym, sym)

    expr.inputs = [replace(s) for s in expr.inputs]
    expr.outputs = [replace(s) for s in expr.outputs]
    if isinstance(expr, Mox):
        expr.children = [child for child in expr.children]
        expr.rngs = {
            name: LazyRng(replace(rng.rng), rng.suffix)
            for name, rng in expr.rngs.items()
        }
        for child in expr.children:
            replace_symbols(child, mapping)
    elif isinstance(expr, Branch):
        expr.selector = replace(expr.selector)
        for case in expr.cases.values():
            replace_symbols(case, mapping)


def normalize_branch_key(key: Any, *, selector_name='selector') -> BranchKey:
    if isinstance(key, np.bool_):
        return bool(key)
    if isinstance(key, bool):
        return key
    if isinstance(key, str):
        return key
    if isinstance(key, Tracer) or hasattr(key, 'aval'):
        raise TypeError(
            f'Static branch selector {selector_name!r} must be a Python '
            f'str or bool, not {type(key).__name__}.')
    raise TypeError(
        f'Unsupported static branch key {key!r} of type {type(key).__name__}.')


Key: TypeAlias = str | int

KeyPath: TypeAlias = tuple[DictKey | SequenceKey, ...]


def unwrap_key_path(kp: KeyPath) -> tuple[str, ]:
    parts = []
    for part in kp:
        match part:
            case DictKey():
                parts.append(part.key)
            case SequenceKey():
                parts.append(part.idx)
            case _:
                raise RuntimeError(
                    f'Unexpected type of key part: {type(part)}.')
    return tuple(parts)


def to_flat_dict(at: ArrayTree) \
        -> tuple[dict[Key, Array], list[Key], PyTreeDef]:
    ix: list[Key] = []
    res: dict[Key, Array] = {}

    def fn_leaf(leaf: tuple[KeyPath, Any]):
        key = unwrap_key_path(leaf[0])
        nonlocal ix, res
        ix.append(key)
        res[key] = leaf[1]

    def fn_node(*args):
        pass

    leaves, treedef = jax.tree.flatten_with_path(at, is_leaf_takes_path=True)
    treedef.walk(fn_node, fn_leaf, leaves)  # type: ignore[attr-defined]
    return res, ix, treedef


def from_flat_dict[T](ix: list[T], treedef: PyTreeDef,
                      mapping: dict[T, Any]) -> ArrayTree:
    leaves = [mapping[key] for key in ix]
    return jax.tree.unflatten(treedef, leaves)


def reconstruct(flat_dict: Mapping[Key, Any]):
    """Reconstruct array tree from flat dictionary.

    >>> reconstruct({('a', 'a'): 1, ('a', 'b'): 2, ('b', ): 3})
    {'a': {'a': 1, 'b': 2}, 'b': 3}
    >>> reconstruct({(0, 'a'): 1, (0, 'b'): 2, (1, 'a'): 3})
    [{'a': 1, 'b': 2}, {'a': 3}]
    """
    if flat_dict == {}:
        return {}
    if () in flat_dict and len(flat_dict) == 1:
        return flat_dict[()]  # Leaf.
    elif () in flat_dict:
        raise RuntimeError(
            f'Internal nodes and leaves interleave together: {flat_dict}.')

    groups: dict[Key, Mapping[Key, Any]] = defaultdict(dict)
    for key, leaf in flat_dict.items():
        head, *rest = key
        tail = tuple(rest)
        groups[head][tail] = leaf

    if isinstance(key := next(iter(groups)), int):
        return [reconstruct(groups[i]) for i in range(len(groups))]

    return {key: reconstruct(group) for key, group in groups.items()}


def map_param_tree(fn: Callable[[Sequence[ArrayTree | None]], Any],
                   trees: Sequence[ArrayTree] ) -> dict[str, Any]:
    """Apply `fn` to all leaves of union of `trees`.

    >>> xs = {'params': {'kernel': jnp.eye(2)}}
    >>> ys = {'params': {'bias': jnp.zeros(2)}}
    >>> fn = lambda args: [a for a in args if a is not None][0]
    >>> res = map_param_tree(fn, (xs, ys))
    >>> jax.tree.map(jnp.shape, res)
    {'params': {'bias': (2,), 'kernel': (2, 2)}}
    """
    if len(trees) < 1:
        raise RuntimeError('At least one tree requires.')

    mappings, key_indices, _ = zip(*[to_flat_dict(tree) for tree in trees])

    keys = set()
    for key_index in key_indices:
        keys |= set(key_index)

    res: dict[KeyPath, Any] = {}
    for key in sorted(keys):
        args = [m.get(key) for m in mappings]
        res[key] = fn(args)

    return reconstruct(res)


def merge_branch_params(cases: Mapping[BranchKey, Expr]) \
        -> tuple[list[Symbol], PyTreeDef, dict[Symbol, Symbol]]:
    """Merge case parameter symbols by variable path and symbol identity.

    1. Restore param tree.
    2. Symbols at the same paths are the same symbols.
    3. Merge variable trees.
       {/a: s1} + {/a: s2} -> {/a: Sa}
       {/a: s1} + {/b: s2} -> {/a: Sa, /b: Sb}
       {/a: s1, /b: s1} + {/a: s2, /c: s2} -> error: s1 at both /a and /b
    """
    param_trees = []
    for _, expr in cases.items():
        param_tree = external_param_tree(expr)
        param_trees += [param_tree]

    def fn(leaves: Sequence[ArrayTree | None]) -> ArrayTree:
        symbols: Sequence[Symbol] = [x for x in leaves if x is not None]
        if len(symbols) == 0:
            raise RuntimeError(f'No array tree defined: {leaves}.')
        symbol, *symbols = symbols
        return symbol

    # Build union tree of parameter trees.
    params = map_param_tree(fn, param_trees)
    leaves, treedef = jax.tree.flatten(params)

    # Build mapping between shared symbols and branch arm symbols.
    mapping: dict[Symbol, Symbol] = {}
    flat_params, *_ = to_flat_dict(params)
    for param_tree in param_trees:
        flat_param_tree, *_ = to_flat_dict(param_tree)
        for key, sym in flat_param_tree.items():
            if (shared_sym := flat_params[key]) == sym:
                continue
            mapping[sym] = shared_sym

    # Then replace branch symbols with shared ones.
    for _, expr in cases.items():
        replace_symbols(expr, mapping)

    return leaves, treedef, mapping


def branch(selector_name: str,
           cases: Mapping[str | bool | np.bool_, Expr],
           *, name: str | None = None) -> Branch:
    """Build a static branch expression selected by a named kwarg."""
    if not isinstance(selector_name, str) or not selector_name:
        raise ValueError('Branch selector name must be a non-empty string.')
    if not cases:
        raise ValueError('Static branch requires at least one case.')

    normalized: dict[BranchKey, Expr] = {}
    for key, expr in cases.items():
        norm_key = normalize_branch_key(key, selector_name=selector_name)
        if norm_key in normalized:
            raise ValueError(f'Duplicate static branch key {norm_key!r}.')
        if not isinstance(expr, Expr):
            raise TypeError(
                f'Branch case {norm_key!r} must be an {type(Expr).__name__}, '
                f'not {type(expr).__name__}.')
        normalized[norm_key] = clone_expr(expr)

    first_case = next(iter(normalized.values()))
    first_inputs = [*external_inputs(first_case)]
    first_outputs = [*first_case.outputs]
    for key, case in tuple(normalized.items())[1:]:
        validate_symbols(first_inputs, external_inputs(case),
                         f'case {key!r} input')
        validate_symbols(first_outputs, case.outputs, f'case {key!r} output')

    params, var_tree, param_mapping = merge_branch_params(normalized)
    for case in normalized.values():
        replace_symbols(case, param_mapping)

    # Recompute after parameter rewrites in case a boundary intentionally
    # aliases a parameter symbol.
    first_inputs = [*external_inputs(first_case)]
    first_outputs = [*first_case.outputs]
    for case in tuple(normalized.values())[1:]:
        input_mapping = dict(zip(external_inputs(case), first_inputs))
        output_mapping = dict(zip(case.outputs, first_outputs))
        replace_symbols(case, input_mapping | output_mapping)
        set_external_inputs(case, first_inputs)
        case.outputs = first_outputs

    selector = Static(selector_name)
    return Branch(params + [selector] + first_inputs, first_outputs,
                  {'name': name}, selector_name, selector, normalized,
                  var_tree)


def make_mox(fn: Callable[Args, Any]) -> Callable[Args, Mox]:
    """Make a tracing routine for `fn` to obtaine its Module eXpression
    (MoX).

    >>> m = nn.Dense(10)
    >>> batch = jnp.empty((1, 10))
    >>> params = jax.jit(m.init)(jax.random.PRNGKey(42), batch)
    >>> mox = make_mox(m.apply)(params, batch)
    """
    @wraps(fn)
    def wrapper(*args: Args.args, **kwargs: Args.kwargs) -> Mox:
        wf = lu.wrap_init(fn)
        in_args, in_tree = jax.tree.flatten((args, kwargs))
        wf, out_tree_thunk = jax.api_util.flatten_fun(wf, in_tree)

        # Root module expression is incomplete in traversing due to missing
        # outputs. May be, it would be better to construct builder from both
        # `inputs` and `in_tree at once.
        builder = MoxBuilder()
        builder.set_input_tree(in_tree)
        wf = trace_modules(wf, builder)
        with intercept_methods(builder.intercept):
            _ = wf.call_wrapped(*in_args)

        builder.set_output_tree(out_tree_thunk())
        return builder.build()
    return wrapper


class MoxBuilder:

    def __init__(self):
        self.root = Mox([], [], {})
        self.block_stack: list[Mox] = [self.root]
        self.symbols: dict[ModuleTracer, Symbol] = {}

        self.module_stack: list[InterceptorContext] = []  # get_module_path

    def build(self) -> Mox:
        return self.root

    def get_module_path(self) -> str:
        return ''.join(f'/{type(c.module).__qualname__}'
                       for c in self.module_stack)

    def intercept(self, fn: Callable[..., Any], args, kwargs,
                  context: InterceptorContext) -> Any:
        """A hook to intercept call of `flax.linen.Module` method."""
        # TODO(@daskol): Should we run nested tracer?
        # TODO(@daskol): How to flatten in abstract way?
        # TODO(@daskol): Do not ignore `setup` method.
        if context.method_name == 'setup':
            return fn(*args, **kwargs)

        # It is important to access `__dict__` directly since `__getattr__` is
        # overriden.
        params = {f.name: context.module.__dict__.get(f.name)
                  for f in fields(context.module) if f.name != 'parent'}
        module_info = type(context.module), context.method_name
        module_rngs = {}
        for name, rng in context.module.scope.rngs.items():
            [rng_sym] = self.to_symbols([rng.rng])
            module_rngs[name] = LazyRng(rng_sym, rng.suffix)

        # Push on stack partially built module expression object. It will be
        # finalized by the end of this routine.
        child = Mox([], [], params, [], *module_info, rngs=module_rngs)
        parent = self.block_stack[-1]
        parent.children += [child]
        self.block_stack += [child]

        # Flax passes a module function.
        unbound_fn: Callable[..., Any]
        if isinstance(fn, partial):
            unbound_fn = fn.func
        else:
            method_fn = getattr(context.module, context.method_name)
            unbound_fn = (lambda x: x.__func__)(method_fn)

        # Assume that weights are already wrapped up.
        flat_vars, child.var_tree = jax.tree.flatten(context.module.variables)
        child.inputs.extend(self.to_symbols(flat_vars))

        # Flax assumes that the first argument is a weight dictionary (or
        # Module or Scope). Thus, we need to flatten this dictionary for
        # binding and unflatten it in evaluation time.
        args = (context.module, ) + args
        (scope, *in_args), child.in_tree = jax.tree.flatten((args, kwargs))
        trace = find_top_trace(flat_vars + in_args)  # TODO(@daskol): Dynamic!
        in_tracers, call_args = [], []
        for arg in in_args:
            tracer = trace.full_raise(arg)
            in_tracers.append(tracer)
            call_args.append(arg if _is_static_input(arg) else tracer)
        child.inputs.extend(self.to_symbols(in_tracers))  # XXX

        # Flatten function (inputs and outputs) for building intermediate
        # representation.
        wrap_fn = lu.wrap_init(unbound_fn)
        flat_fn, out_tree_fn = jax.api_util.flatten_fun(wrap_fn, child.in_tree)
        outs = flat_fn.call_wrapped(scope, *call_args)

        # TODO(@daskol): Remove code duplication with `_lower`.
        flat_outs, _ = jax.tree.flatten(outs)
        out_tracers = [trace.full_raise(x) for x in flat_outs]

        # TODO(@daskol): Should we introduce `module_call` primitive here? Or
        # just flatten outputs?
        child.out_tree = out_tree_fn()
        child.outputs.extend(self.to_symbols(out_tracers))
        self.block_stack.pop()
        return jax.tree.unflatten(child.out_tree, outs)

    def append(self, prim: jex.core.Primitive, params: dict[str, Any],
               in_tracers: list[ModuleTracer],
               out_tracers: list[ModuleTracer]):
        in_symbols = self.to_symbols(in_tracers)
        out_symbols = self.to_symbols(out_tracers)
        eq = Equation(in_symbols, out_symbols, params, prim)

        block = self.block_stack[-1]
        block.children.append(eq)

    def set_input_tree(self, tree: PyTreeDef):
        self.root.in_tree = tree

    def set_inputs(self, tracers: Sequence[ModuleTracer]):
        self.root.inputs.clear()
        self.root.inputs.extend(self.to_symbols(tracers))

    def set_output_tree(self, tree: PyTreeDef):
        self.root.out_tree = tree

    def set_outputs(self, tracers: Sequence[ModuleTracer] | ModuleTracer):
        if not isinstance(tracers, list | tuple | Sequence):
            tracers = [tracers]
        self.root.outputs.clear()
        self.root.outputs.extend(self.to_symbols(tracers))

    def to_symbols(self, tracers: Sequence[ModuleTracer]) -> Sequence[Symbol]:
        symbols = []
        for tracer in tracers:
            if (symbol := self.symbols.get(tracer)) is None:
                if isinstance(tracer.value, AbstractValue | Tracer):
                    symbol = Var(tracer.aval)
                elif isinstance(tracer.aval, AbstractValue):
                    symbol = Literal(tracer.value, tracer.aval)
                else:
                    raise RuntimeError('Unexpected abstract value of type '
                                       f'{type(tracer.aval)}.')
                self.symbols[tracer] = symbol
            symbols += [symbol]
        return symbols


def fully_qualified_name(ty: Type[nn.Module] | None) -> str:
    if ty is None:
        return '<none>'
    elif ty.__module__ == 'builtins':
        return ty.__qualname__
    else:
        return f'{ty.__module__}.{ty.__qualname__}'


def dump(node: Expr, fileobj: IO[str], *, depth=0):
    indent = '  ' * depth
    match node:
        case Mox():
            if node.is_ephemeral:
                name_ty = 'Ephemeral'
                name = '<none>'
            else:
                name_ty = fully_qualified_name(node.module_ty)
                name = node.params['name']
            print(f'{indent}inputs ={node.inputs}', file=fileobj)
            print(f'{indent}outputs={node.outputs}', file=fileobj)
            keys = (k for k in node.params.keys()
                    if k not in ('name', 'parent'))
            try:
                key = next(keys)
                val = node.params[key]
                attrs = f'{key}={val}'
            except StopIteration:
                attrs = ''
            print(f'{indent}mod {name} : {name_ty}({attrs}) {{ # {depth}',
                  file=fileobj)
            for child in node.children:
                dump(child, fileobj, depth=depth + 1)
            print(f'{indent}}}', file=fileobj)
        case Branch():
            keys = ', '.join(repr(k) for k in node.cases)
            print(f'{indent}inputs ={node.inputs}', file=fileobj)
            print(f'{indent}outputs={node.outputs}', file=fileobj)
            print(f'{indent}branch {node.params.get("name")} : '
                  f'static_branch(selector={node.selector_name}, '
                  f'cases=({keys})) {{ # {depth}', file=fileobj)
            for key, case in node.cases.items():
                print(f'{indent}  case {key!r} {{', file=fileobj)
                dump(case, fileobj, depth=depth + 2)
                print(f'{indent}  }}', file=fileobj)
            print(f'{indent}}}', file=fileobj)
        case Equation():
            print(f'{indent}eq {node.prim.name}', file=fileobj)
            print(f'{indent}inputs ={node.inputs}', file=fileobj)
            print(f'{indent}outputs={node.outputs}', file=fileobj)
            print(f'{indent}{node}', file=fileobj)
        case Expr():
            raise RuntimeError('Unexpected node of type {type(node)}.')


def dump_yson(expr: Expr, fileobj: IO[bytes], indent: int = 2):
    """Serialize module expression `expr` as a YSON-formatted object to
    `fileobj`.
    """
    try:
        from yt.yson import YsonEntity, YsonList, YsonType, dump
    except ImportError as e:
        msg = (
            'Missing YSON packages. Try to install `ytsaurus-client` package '
            'for basic YSON support and `ytsaurus-yson` for fast serialization'
            '/deserialization.')
        raise RuntimeError(msg) from e

    def fmt(val):
        if isinstance(val, bool | int | float):
            return val
        elif isinstance(val, Jaxpr):
            return val.pretty_print(use_color=False)
        elif callable(val):
            name = getattr(val, '__name__', type(val).__name__)
            module = getattr(val, '__module__', type(val).__module__)
            return f'{module}.{name}'
        elif hasattr(val, 'dtype'):
            return str(val.dtype)
        else:
            return str(val)

    def to_yson(expr: Expr) -> YsonType:
        if isinstance(expr, Mox):
            root = YsonList([])
            root.attributes['primitive'] = 'module_call'
            root.attributes['ephemeral'] = fmt(expr.is_ephemeral)
            if expr.module_ty:
                root.attributes['type'] = expr.module_ty.__name__
            root.extend(to_yson(child) for child in expr.children)
        elif isinstance(expr, Branch):
            root = YsonList([])
            root.attributes['primitive'] = 'static_branch'
            root.attributes['selector'] = expr.selector_name
            for key, case in expr.cases.items():
                child = to_yson(case)
                child.attributes['case'] = fmt(key)
                root.append(child)
        elif isinstance(expr, Equation):
            root = YsonEntity()
            root.attributes['primitive'] = expr.prim.name
        else:
            raise RuntimeError('Unexpected expression type {type(expr)}.')
        root.attributes.update({
            k: fmt(v)
            for k, v in expr.params.items() if v is not None
        })
        return root

    root = to_yson(expr)
    kwargs = {}
    if indent > 0:
        kwargs = dict(yson_format='pretty', indent=indent)
    dump(root, fileobj, **kwargs)


def dump_xml(expr: Expr, fileobj: IO[str], indent: int = 2):
    """Serialize module expression `expr` to XML representation."""

    def fmt(val):
        if isinstance(val, bool):
            return str(val).lower()
        elif isinstance(val, Jaxpr):
            return val.pretty_print(use_color=False)
        elif callable(val):
            name = getattr(val, '__name__', type(val).__name__)
            module = getattr(val, '__module__', type(val).__module__)
            return f'{module}.{name}'
        elif hasattr(val, 'dtype'):
            return str(val.dtype)
        else:
            return str(val)

    def build_subtree(parent: Element, node: Expr,
                      case: BranchKey | None = None):
        attrs = {}
        if isinstance(node, Mox):
            tag = 'module_call'
            if node.module_ty:
                attrs['type'] = node.module_ty.__name__
            attrs['ephemeral'] = fmt(node.is_ephemeral)
        elif isinstance(node, Branch):
            tag = 'static_branch'
            attrs['selector'] = node.selector_name
        elif isinstance(node, Equation):
            tag = node.prim.name
        if case is not None:
            attrs['case'] = fmt(case)
        attrs.update({
            k: fmt(v)
            for k, v in node.params.items() if v is not None
        })
        elem = SubElement(parent, tag, attrs)
        if isinstance(node, Mox):
            for child in node.children:
                build_subtree(elem, child)
        elif isinstance(node, Branch):
            for key, child in node.cases.items():
                build_subtree(elem, child, key)

    if isinstance(expr, Mox):
        root = Element('module_call', attrib={'ephemeral': 'true'})
        for child in expr.children:
            build_subtree(root, child)
    elif isinstance(expr, Branch):
        root = Element('static_branch',
                       attrib={'selector': expr.selector_name})
        for key, child in expr.cases.items():
            build_subtree(root, child, key)
    elif isinstance(expr, Equation):
        root = Element(expr.prim.name, attrib={})
    else:
        raise RuntimeError('Unexpected expression type {type(expr)}.')

    et = ElementTree(root)
    if indent:
        indent_etree(et, space=' ' * indent)
    et.write(fileobj, encoding='unicode', xml_declaration=True)


@dataclass(slots=True, frozen=True)
class Token:
    kind: str
    value: str = ''

    @property
    def is_empty(self) -> bool:
        return self.kind == '' and self.value == ''

    @classmethod
    def empty(cls) -> Self:
        return cls('', '')

    def __repr__(self) -> str:
        if self.is_empty:
            return 'ε'
        return f'<{self.kind}>{self.value}'

    def __str__(self) -> str:
        return self.value or self.kind

    _NODE_TYPE: ClassVar = re.compile(
        r'(comment|text|processing-instruction|node)')

    _NAME_CHAR: ClassVar = re.compile(r'[A-z_][0-9A-z_]*')

    _DOUBLE_QUOTED_STR: ClassVar = re.compile(r'"[^"]*"')
    _SINGLE_QUOTED_STR: ClassVar = re.compile(r"'[^']*'")

    _FULL_NUM: ClassVar = re.compile(r'\d+(\.(\d+)?)?')
    _FRAC_NUM: ClassVar = re.compile(r'\.(\d+)?')

    _AXIS_NAME: ClassVar = re.compile(
        r'(ancestor|ancestor-or-self|attribute|child|descendant'
        r'|descendant-or-self|following|following-sibling|namespace|parent'
        r'|preceding|preceding-sibling|self)')


def tokenize_xpath(val: str):
    token = Token.empty()
    while val:
        while val:
            if val[0] != ' ':
                break
            val = val[:1]
        val, token = tokenize_expr_token(val, token)
        yield token


def tokenize_expr_token(val: str, last: Token):
    if val[:2] in ('..', '::'):
        return val[2:], Token(val[:2])
    elif val[:1] in '()[]@,.':
        return val[1:], Token(val[:1])

    _PARSERS = (tokenize_name_test, tokenize_node_type, tokenize_node_operator,
                tokenize_function_name, tokenize_axis_name, tokenize_literal,
                tokenize_number)
    for parser in _PARSERS:
        try:
            val, token = parser(val, last)
        except TypeError:
            pass  # Parser failed to parse: None is parse result.
        else:
            return val, token

    raise NotImplementedError(
        f'Perhaps, this production rule is not implemented at {val[:16]}.')


def tokenize_name_test(val: str, last: Token):
    # If there is a preceding token and the preceding token is not one of @,
    # ::, (, [, , or an Operator, then a * must be recognized as a
    # MultiplyOperator and an NCName must be recognized as an OperatorName.
    if val[:1] == '*':
        if not last.is_empty and last.kind not in ('@', '::', '(', 'Operator'):
            return val[1:], Token('Operator', val[:1])
        else:
            return val[1:], Token('NameTest', val[:1])

    def tokenize_ncname(val: str) -> str | None:
        if (m := Token._NAME_CHAR.match(val)) is not None:
            name = m.group(0)
            return name

    # NCName ':' '*'
    if not (name := tokenize_ncname(val)):
        tail = val[len(name):]
        if tail[:2] == ':*':
            return tail[2:], Token('NameTest', f'{name}:*')

    # QName := NCName : NCName
    if (prefix := tokenize_ncname(val)):
        tail = val[len(prefix):]
        if tail[:1] != ':':
            return tail, Token('NameTest', prefix)
        if (suffix := tokenize_ncname(tail[1:])):
            offset = len(prefix) + 1 + len(suffix)
            return val[offset:], Token('NameTest', f'{prefix}:{suffix}')


def tokenize_node_type(val: str, last: Token):
    if (m := Token._NODE_TYPE.match(val)) is not None:
        value = m.group(0)
        return val[len(value):], Token('NodeType', value)


def tokenize_node_operator(val: str, last: Token):
    if val[:3] in ('and', 'or', 'mod', 'div'):
        return val[3:], Token('Operator', val[:3])
    elif val[:2] in ('//', '!=', '<=', '>='):
        return val[2:], Token('Operator', val[:2])
    elif val[:1] in '*/|+-=':
        return val[1:], Token('Operator', val[:1])


def tokenize_function_name(val: str, last: Token):
    pass


def tokenize_axis_name(val: str, last: Token):
    if (m := Token._AXIS_NAME.match(val)) is not None:
        value = m.group(0)
        return val[len(value):], Token('AxisName', value)


def tokenize_literal(val: str, last: Token):
    for pattern in (Token._DOUBLE_QUOTED_STR, Token._SINGLE_QUOTED_STR):
        if (m := pattern.match(val)):
            value = m.group(0)
            return val[len(value):], Token('Literal', value)


def tokenize_number(val: str, last: Token):
    for pattern in (Token._FULL_NUM, Token._FRAC_NUM):
        if (m := pattern.match(val)):
            value = m.group(0)
            return val[len(value):], Token('Number', value)


class XPath:
    """XML Path expression language expression.

    [1]: https://www.w3.org/TR/xpath-10/
    [1]: https://www.w3.org/TR/xpath-20/
    [1]: https://www.w3.org/TR/xpath-30/
    [1]: https://www.w3.org/TR/xpath-31/
    """

    def __init__(self, xpath: str | Self):
        if isinstance(xpath, XPath):
            self.locs = (*xpath.locs, )
            return
        try:
            tokens = (*tokenize_xpath(xpath), )
        except Exception as e:
            raise RuntimeError(f'Failed to tokenize XPath: {xpath}.') from e

        try:
            self.locs: tuple[LocationStep, ...] = (*parse_xpath(tokens), )
        except Exception as e:
            raise RuntimeError(f'Failed to parse XPath: {xpath}.') from e

    def __repr__(self) -> str:
        return ' '.join(repr(x) for x in self.locs)

    def __str__(self) -> str:
        return '/'.join(str(x) for x in self.locs)


PredicateFn: TypeAlias = Callable[[dict[str, Any]], bool]


@dataclass(slots=True, frozen=True)
class LocationPredicate:
    func: PredicateFn
    desc: str

    def __call__(self, attrs: dict[str, Any]) -> bool:
        return self.func(attrs)

    def __str__(self) -> str:
        return self.desc


@dataclass(slots=True, frozen=True)
class LocationStep:
    axis: str
    node: str
    predicate: tuple[LocationPredicate, ...] = ()

    def __str__(self) -> str:
        pred = ''.join(str(p) for p in self.predicate)
        return f'{self.axis}::{self.node}{pred}'


def parse_xpath(tokens: list[Token]):
    if len(tokens) == 0:
        yield LocationStep('self', 'node()')
    if len(tokens) == 1:
        if tokens[0].kind == 'Operator' and tokens[0].value == '/':
            yield LocationStep('self', 'node()')
            return
    while tokens:
        if tokens[0].kind == 'Operator':
            if tokens[0].value == '/':
                tokens = tokens[1:]
            elif tokens[0].value == '//':
                yield LocationStep('descendant-or-self', 'node()')
                tokens = tokens[1:]
            else:
                raise RuntimeError(f'Unexpected operator: {tokens[0]!r}.')
        else:
            loc, tokens = parse_location_step(tokens)
            yield loc


def parse_location_step(tokens: list[Token]):
    # Abbreviated step first.
    if tokens[0].kind == '.':
        return LocationStep('self', 'node()'), tokens[1:]
    elif tokens[0].kind == '..':
        return LocationStep('parent', 'node()'), tokens[1:]

    if tokens[0].kind == '@':
        axis = 'attribute'
        tokens = tokens[1:]
    elif (len(tokens) > 1 and
          tokens[0].kind == 'AxisName' and tokens[1].kind == '::'):
        axis = tokens[1].value
        tokens = tokens[2:]
    else:
        axis = 'child'  # Default axis specifier.

    if len(tokens) > 0 and tokens[0].kind == 'NameTest':
        node = tokens[0].value
        tokens = tokens[1:]
    elif (len(tokens) > 2 and tokens[0].kind == 'NodeType' and
          tokens[1].kind == '(' and tokens[2].kind == ')'):
        node = f'{tokens[0].value}()'
        tokens = tokens[3:]
    else:
        node = 'node()'

    preds = ()
    while len(tokens) > 5:
        if tokens[0].kind != '[':
            break
        prefix = ''.join(t.kind for t in tokens[:4])
        if prefix != '[@NameTestOperator':
            raise RuntimeError('Failed to parse predicate of a step.')
        if tokens[4].kind not in ('Literal', 'Number'):
            raise RuntimeError('Failed to parse predicate of a step.')
        if tokens[5].kind != ']':
            raise RuntimeError('Failed to parse predicate of a step.')
        key = tokens[2].value
        try:
            val = int(tokens[4].value)
        except ValueError:
            try:
                val = float(tokens[4].value)
            except ValueError:
                val = tokens[4].value[1:-1]
        pred = LocationPredicate(lambda x: x[key] == val,
                                 ''.join(str(t) for t in tokens[:6]))
        preds += (pred, )
        tokens = tokens[6:]

    return LocationStep(axis, node, preds), tokens


def eval_module(read, write, mox: Mox):
    if mox.is_ephemeral:
        raise NotImplementedError('Only concrete modules allowed.')

    def read_safe(var: Symbol | flax.core.scope.Scope) -> Any:
        if isinstance(var, Symbol):
            return read(var)

    def write_safe(var: Symbol, val: Any):
        if var not in mox.inputs:
            write(var, val)

    # Weight and input symbol are all flattened and stored in single list. In
    # order to apply module func to weights and inputs we need to separate
    # them, restore weights, and restore inputs.
    num_vars = mox.var_tree.num_leaves
    var_syms = jax.tree.unflatten(mox.var_tree, mox.inputs[:num_vars])
    var_vals = jax.tree.map(read, var_syms)

    # Materialize RNGs states.
    rngs = {}
    rng_counters = {}
    for name, rng in mox.rngs.items():
        rngs[name] = LazyRng(rng=read_safe(rng.rng), suffix=rng.suffix)
        rng_counters[name] = 0

    with flax.core.bind(var_vals).temporary() as scope:
        # Ad hoc patching of RNG states in scope.
        scope.rngs = rngs
        scope.rng_counters = rng_counters

        mod = mox.module_ty(**mox.params)
        scoped_mod = mod.clone(parent=scope)

        # See func:`bind` in flax/core/scope.py:1105.
        in_vals = jax.tree.map(read_safe, [scope] + mox.inputs[num_vars:])
        (_, *in_args), in_kwargs = jax.tree.unflatten(mox.in_tree, in_vals)

        unbound_fn = getattr(mox.module_ty, mox.entrypoint)
        out_vals = unbound_fn(scoped_mod, *in_args, **in_kwargs)

    # Output symbols are stored as flattened. So, we need to flatten outputs
    # and write result back.
    outputs, out_tree = jax.tree.flatten(out_vals)
    assert out_tree == mox.out_tree, \
        f'Output tree mismatched in eval time: {mox.out_tree} -> {out_tree}.'
    jax.tree.map(write_safe, mox.outputs, outputs)


def eval_equation(read, write, eq: Equation):
    in_vals = [read(x) for x in eq.inputs]  # TODO(@daskol): Trees?
    subfuns, params = eq.prim.get_bind_params(eq.params)
    out_vals = eq.prim.bind(*subfuns, *in_vals, **params)
    if not eq.prim.multiple_results:
        out_vals = [out_vals]
    for sym, val in zip(eq.outputs, out_vals):
        write(sym, val)


def eval_mox(tree: Mox, *args, **kwargs):
    """Evaluate a module expression `tree` with `args` and `kwargs`."""
    env: dict[Symbol, Any] = {}

    def read(var: Symbol) -> Any:
        """Read a symbol from execution context or take literal value."""
        if var in env:
            return env[var]
        if isinstance(var, Literal):
            return var.const
        raise KeyError(f'Variable {var} is undefined.')

    def write(var: Symbol, val: Any):
        assert var not in env, f'Variable {var} has been already defined.'
        env[var] = val

    def fn(node: Expr) -> Mox | None:
        if isinstance(node, Mox):
            if node.is_ephemeral:
                return node
            else:
                return eval_module(read, write, node)
        elif isinstance(node, Equation):
            return eval_equation(read, write, node)

    # Initialize execution context and execute.
    flat_args, in_tree = jax.tree.flatten((args, kwargs))
    assert in_tree == tree.in_tree, \
        f'Arguments and input tree mismatched: {in_tree} vs {tree.in_tree}.'
    jax.tree.map(write, tree.inputs, flat_args)
    map_mox(fn, tree)

    flatten_res = [read(x) for x in tree.outputs]
    if len(flatten_res) == 1:
        return flatten_res[0]
    return flatten_res


def map_mox(fn: Callable[[Expr], Any], tree: Mox):
    """Apply map transformation `fn` to a module `tree`."""
    nodes: list[Expr] = [tree]
    while nodes:
        node: Expr = nodes.pop()
        if isinstance(res := fn(node), Mox):
            nodes += reversed(res.children)


ModulePath: TypeAlias = tuple[str, ...]

# SubFn takes a path `path` to a node `node` which has been selected with
# search query and returns a new node which substitutes original node `node`.
SubFn: TypeAlias = Callable[[ModulePath, Expr], Expr]


def sub(expr: str | XPath | Sequence[Equation | Mox],
        repl: Mox | Equation | SubFn, mox: Mox) -> Mox:
    """Substitute a module expression `mox` with `repl` according to matching
    pattern `expr`.

    Args:
      expr: XPath expression for nodes to substitute.
      repl: replacement for selected nodes.
      mox: a module expression to mutate.

    Return:
      Modified module expression (MoX).
    """
    def default_sub_fn(path: ModulePath, expr: Expr) -> Expr:
        """Safe module expression modification requires deep copy of nodes.
        However, abstract values of `Symbol` can contain JAX traceback object
        which is not deepcopy-able. This is why we shallow-copy `repl` and
        create new symbols.
        """
        obj = copy(repl)
        obj.inputs = [type(s)(s.value) for s in repl.inputs]
        obj.outputs = [type(s)(s.value) for s in repl.outputs]
        return obj

    sub_fn: SubFn
    if isinstance(repl, Expr):
        sub_fn = default_sub_fn
    elif callable(repl):
        sub_fn = repl
    else:
        raise ValueError(f'Unexpected argument `repl` of type {type(repl)}.')

    # Our substitution algorithm is quite straight forward: find nodes of
    # interest, verify type integrity, search parents, and finally replace.
    nodes: list[Equation | Mox] = []
    if isinstance(expr, list | tuple):
        nodes.extend(expr)
    else:
        for node in query(expr, mox):
            if not isinstance(node, Equation | Mox):
                raise RuntimeError(
                    f'XPath expression does not select a node: {expr}.')
            nodes += [node]

    for node in nodes:
        if not (parents := find_parents(mox, node)):
            raise RuntimeError(f'No parent found for node {node}.')
        path = tuple([p.params.get('name') or p.module_ty.__name__
                      for p in parents[1:]])
        path += (node.params.get('name') or node.module_ty.__name__, )
        if (new_node := sub_fn(path, node)) is node:
            continue
        sub_node(mox, node, new_node)

    return mox


def sub_node(mox: Mox, node: Expr, repl: Equation | Mox):
    if not (parents := find_parents(mox, node)):
        raise RuntimeError(f'No parent found for node {node}.')

    *_, parent = parents
    for ix, child in enumerate(parent.children):
        if child is node:
            break
    else:
        raise RuntimeError(
            f'There is no node {node} among childrens of {parent}.')

    rewire_node(node, repl)
    update_in_trees(parents, ix, repl)
    update_var_trees(parents, ix, repl)
    parent.children[ix] = repl


def find_parents(root: Expr, node: Expr) -> tuple[Mox, ...]:
    # TODO(@daskol): We should get rid of `find_parents` and add `parent`
    # reference to `Expr` type.
    match root:
        case Equation():
            return None
        case Mox():
            # Try to find `node` in childrens.
            for expr in root.children:
                if expr is node:
                    return (root, )
            # Otherwise, ask childrens.
            for expr in root.children:
                parents = find_parents(expr, node)
                if parents is not None:
                    return (root, ) + parents


def rewire_node(node: Expr, repl: Expr):
    if isinstance(node, Equation):
        node_inputs = node.inputs
    elif isinstance(node, Mox):
        node_inputs = node.inputs[node.var_tree.num_leaves:]

    if isinstance(repl, Equation):
        repl_params = []
        repl_inputs = repl.inputs
    elif isinstance(repl, Mox):
        repl_params = repl.inputs[:repl.var_tree.num_leaves]
        repl_inputs = repl.inputs[repl.var_tree.num_leaves:]
    repl_aux_inputs = repl_inputs[len(node_inputs):]

    # Check types of output symbols and preserved input symbols.
    validate_symbols(node_inputs, repl_inputs[:len(node_inputs)])
    validate_symbols(node.outputs, repl.outputs)

    # If there are some input symbols then we preserve as many input symbols as
    # possible.
    repl.inputs = repl_params + node_inputs + repl_aux_inputs
    repl.outputs = node.outputs


def update_in_trees(parents: tuple[Mox, ...], ix: int, repl: Mox):
    orig = parents[-1].children[ix]
    arity = len(orig.inputs)
    if isinstance(orig, Mox):
        arity -= orig.var_tree.num_leaves

    if isinstance(repl, Equation):
        repl_inputs = repl.inputs
    elif isinstance(repl, Mox):
        repl_inputs = repl.inputs[repl.var_tree.num_leaves:]
    orig_syms, aux_syms = repl_inputs[:arity], repl_inputs[arity:]
    if orig.inputs[-arity:] != orig_syms:
        raise NotImplementedError(
            'Original input symbols must be preserved at the moment.')

    for offset, p in enumerate(reversed(parents)):
        p.entrypoint = None  # Mark as an ephemeral.

        # The first input in the root node is a proper param dict, not a
        # placeholder for a class instance.
        num_params = p.var_tree.num_leaves
        params = p.inputs[:num_params]
        if (len(parents) - offset - 1) > 0:
            inputs = [object()] + p.inputs[num_params:]  # Scope is the first.
            args, kwargs = jax.tree.unflatten(p.in_tree, inputs)
            args += tuple(aux_syms)
            (_, *p.inputs), p.in_tree = jax.tree.flatten((args, kwargs))
            p.inputs = params + p.inputs
        else:
            args, kwargs = jax.tree.unflatten(p.in_tree, p.inputs)
            args += tuple(aux_syms)
            p.inputs, p.in_tree = jax.tree.flatten((args, kwargs))
        assert params == p.inputs[:num_params], \
            'Weight params symbols are not preserved.'


def update_var_trees(parents: tuple[Mox, ...], ix: int, repl: Mox):
    # Only module expressions have param tree.
    if not isinstance(repl, Mox):
        return

    orig = parents[-1].children[ix]
    orig_name = orig.params.get('name')
    for p in reversed(parents[1:]):
        # Restore child tree.
        num_params = repl.var_tree.num_leaves
        repl_vars = repl.var_tree.unflatten(repl.inputs[:num_params])
        repl_name = repl.params.get('name') or f'{repl.module_ty.__name__}_0'

        # Restore parent tree.
        num_params = p.var_tree.num_leaves
        variables = p.var_tree.unflatten(p.inputs[:num_params])
        if 'params' not in variables:
            variables['params'] = {}
        variables['params'].pop(orig_name, None)

        # Substitute child subtree in parent.
        assert repl_name not in variables['params'], \
            'Duplicated key in variables: fix name generation or fix a tree.'
        variables['params'][repl_name] = repl_vars.get('params', {})
        var_tree = p.var_tree
        inputs, p.var_tree = jax.tree.flatten(variables)
        p.inputs = inputs + p.inputs[num_params:]

        # Check whether `expr` shares the same `var_tree`, i.e. other
        # entrypoint of the same module. If it is the same module then we must
        # ensure that its `var_tree` is the same.
        #
        # TODO(@daskol): Propagate `var_tree` changees down to leaves.
        for child in p.children:
            if child is repl:
                continue
            if not isinstance(child, Mox) or child.var_tree != var_tree:
                continue
            child.inputs = p.inputs[:p.var_tree.num_leaves] \
                        + child.inputs[child.var_tree.num_leaves:]
            child.var_tree = p.var_tree

        p.entrypoint = None  # Mark as an ephemeral.
        if (parent_name := p.params.get('name')):
            orig_name = parent_name  # Top-level module.
        repl = p

    # Restore child var-tree.
    num_params = repl.var_tree.num_leaves
    repl_vars = repl.var_tree.unflatten(repl.inputs[:num_params])
    repl_name = repl.params.get('name') or f'{repl.module_ty.__name__}_0'

    # Restore root in-tree.
    root = parents[0]
    (variables, *inputs), kwargs = root.in_tree.unflatten(root.inputs)

    # Update in-place root params.
    if 'params' not in variables:
        variables['params'] = {}
    variables['params'].pop(orig_name, None)
    assert repl_name not in variables['params'], \
        'Duplicated key in variables: fix name generation or fix a tree.'
    variables['params'].update(repl_vars['params'])
    args = (variables, *inputs)
    root.inputs, root.in_tree = jax.tree.flatten((args, kwargs))


def validate_symbols(actual: list[Symbol], desired: list[Symbol],
                     what: str = ''):
    if what:
        what = f'{what.capitalize()} symbol'
    else:
        what = 'Symbol'
    if len(actual) != len(desired):
        raise RuntimeError(f'Number of {what.lower()} differ: '
                           f'{len(actual)} != {len(desired)}.')
    for pair in zip(actual, desired):
        lhs, rhs = [s.aval for s in pair]
        if lhs.shape != rhs.shape:
            raise RuntimeError(
                f'{what} shapes differ: {lhs.shape} != {rhs.shape}.')
        if lhs.dtype != rhs.dtype:
            raise RuntimeError(
                f'{what} dtypes differ: {lhs.dtype} != {rhs.dtype}.')


def query(expr: str | XPath, mox: Mox) -> Sequence[Any]:
    """Get modules or their properties by XPath expression.

    >>> class ResBlock(nn.Module):
    >>>     @nn.compact
    >>>     def __call__(self, xs):
    >>>         return xs + nn.Dense(10)(xs)
    >>>
    >>> mox = make_mox(ResBlock().init)(jax.random.PRNGKey(42),
    >>>                                 jnp.empty((2, 10))
    >>> # Query all modules with 10 output features.
    >>> query('//[@features=10]')
    [nn.Dense(10)]
    """
    xpath = XPath(expr)
    nodes = (mox, )
    for i, step in enumerate(xpath.locs):
        # TODO(@daskol): Support full range of name locations. Proper
        # implementation requires element nodes with reference to parent
        # element. We could do it with the power of built-in `xml`.
        if step.axis == 'self' and step.node == 'node()':
            pass  # Node set is not changed.
        elif step.axis == 'descendant-or-self' and step.node == 'node()':
            nodes = select_all_descendants(nodes)
        elif step.axis == 'child':
            # TODO(@daskol): Some node type expression ignored (e.g. text).
            if step.node == 'node()':
                nodes = select_children(nodes)
            elif not step.node.endswith('()'):
                nodes = select_nodes(step.node, nodes)

            # If there is no predicate then keep all nodes.
            if step.predicate:
                nodes = filter_nodes(step.predicate, nodes)
        else:
            tail = ''.join(str(x) for x in xpath.locs[i:])
            raise NotImplementedError(
                f'Some XPath features are not supported at {tail}.')
    return nodes


NodeSet: TypeAlias = Sequence[Mox | Equation]


def filter_nodes(predicates: Sequence[LocationPredicate],
                 node_set: NodeSet) -> NodeSet:
    def verify(attrs: dict[str, Any]) -> bool:
        try:
            return all(p(attrs) for p in predicates)
        except Exception:
            return False

    nodes = []
    for node in node_set:
        if isinstance(node, Mox):
            attrs = node.to_dict(False)
        elif isinstance(node, Equation):
            attrs = node.to_dict()
        if verify(attrs):
            nodes += [node]
    return tuple(nodes)


def select_all_descendants(node_set: NodeSet) -> NodeSet:
    nodes = [*node_set]
    for node in node_set:
        if isinstance(node, Mox):
            nodes += select_all_descendants(node.children)
    return tuple(nodes)


def select_children(node_set: NodeSet) -> NodeSet:
    nodes = []
    for node in node_set:
        if isinstance(node, Mox):
            nodes += node.children
    return tuple(nodes)


def select_nodes(name: str, node_set: NodeSet) -> NodeSet:
    def predicate(node: Expr) -> bool:
        match node:
            case Mox():
                return name == 'module_call'
            case Equation():
                return name == node.prim.name

    return tuple([n for n in node_set if predicate(n)])
