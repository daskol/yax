"""Module `yax` implements module tracing routines, module expression (MoX)
building and querying.
"""

import logging
from dataclasses import dataclass, field, fields
from functools import wraps
from io import StringIO
from typing import IO, Any, ParamSpec, Sequence, Type

import flax.linen as nn
import jax
import jax.extend as jex
import jax.extend.linear_util as lu
from flax.linen.module import Callable, InterceptorContext, intercept_methods
from jax.core import MainTrace, ShapedArray, Sublevel, Trace, Tracer, new_main

__all__ = ('Equation', 'Expr', 'Literal', 'Mox', 'Var', 'Symbol', 'mox',
           'mtree_map', 'mtree_query', 'mtree_sub')

# TODO(@daskol): Python 3.12 introduced new type parameter syntax (PEP-0695)
# but some code quality tools (e.g. yapf) do not support this syntax.
Args = ParamSpec('Args')

logger = logging.getLogger(__name__)


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
    def aval(self):
        # TODO(@daskol): Should be cached or precomputed.
        return jax.api_util.shaped_abstractify(self.value)

    def full_lower(self):
        # TODO(@daskol): Full lower to some abstract array type.
        # return jax.core.full_lower(self.value)
        assert isinstance(self, Tracer)
        return self


class ModuleTrace(Trace):

    def __init__(self, main: MainTrace, sublevel: Sublevel, **kwargs) -> None:
        super().__init__(main, sublevel)
        self.builder: MoxBuilder = kwargs.get('builder')

    def pure(self, val):
        return ModuleTracer(self, val)

    def lift(self, val):
        return ModuleTracer(self, val)

    def process_primitive(self, prim: jex.core.Primitive, tracers, params):
        result, _ = prim.abstract_eval(*[x.aval for x in tracers], **params)
        # TODO(@daskol): Add tracers too?
        if prim.multiple_results:
            out_tracers = [ModuleTracer(self, out) for out in result]
            self.builder.append(prim, params, tracers, out_tracers)
        else:
            out_tracers = ModuleTracer(self, result)
            self.builder.append(prim, params, tracers, [out_tracers])
        return out_tracers


@lu.transformation
def _lower(builder: 'MoxBuilder', main: MainTrace, *ins):
    trace: ModuleTrace = main.with_cur_sublevel()
    in_tracers = [trace.lift(x) for x in ins]
    builder.set_inputs(in_tracers)
    out_tracers = yield in_tracers, {}
    builder.set_outputs(out_tracers)
    outs = [trace.full_raise(x).value for x in out_tracers]
    yield outs


@lu.transformation
def _raise(builder, *ins):
    with new_main(ModuleTrace, builder=builder) as main:
        outs = yield (main, *ins), {}
        del main
    yield outs


def trace_modules(wf: lu.WrappedFun, builder: 'MoxBuilder') -> lu.WrappedFun:
    return _raise(_lower(wf, builder), builder)


@dataclass(slots=True)
class Symbol:
    aval: ShapedArray

    def __eq__(self, other) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass(slots=True)
class Var(Symbol):
    pass


@dataclass(init=False, slots=True)
class Literal(Symbol):
    value: Any

    def __init__(self, value: Any, aval: ShapedArray | None = None):
        if aval is None:
            aval = ...  # TODO(@daskol): Get shaped.
        super().__init__(aval)
        self.value = value


@dataclass(slots=True)
class Expr:
    inputs: list[Symbol]
    outputs: list[Var]
    params: dict[str, Any]


@dataclass(slots=True)
class Equation(Expr):
    prim: jex.core.Primitive


@dataclass(repr=False, slots=True)
class Mox(Expr):
    children: list[Expr] = field(default_factory=list)
    module_ty: Type[nn.Module] | None = None
    entrypoint: str | None = None

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


def mox(fn: Callable[Args, Any]) -> Callable[Args, Mox]:
    """Make a tracing routine for `fn` to obtaine its Module eXpression
    (MoX).

    >>> m = nn.Dense(10)
    >>> batch = jnp.empty((1, 10))
    >>> params = jax.jit(m.init)(jax.random.PRNGKey(42), batch)
    >>> mtree = mox(m.apply)(params, batch)
    """
    @wraps(fn)
    def wrapper(*args: Args.args, **kwargs: Args.kwargs) -> Mox:
        wf = lu.wrap_init(fn)
        in_args, in_tree = jax.tree.flatten((args, kwargs))
        wf, _ = jax.api_util.flatten_fun(wf, in_tree)
        logger.debug(in_tree)

        builder = MoxBuilder()
        wf = trace_modules(wf, builder)
        with intercept_methods(builder.intercept):
            _ = wf.call_wrapped(*in_args)

        # out_tree = out_tree_thunk()
        # res = jax.tree.unflatten(out_tree, out_args)
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
        # https://github.com/google/flax/pull/1443/files
        self.module_stack += [context]
        mpath = self.get_module_path()
        logger.debug('enter %s (%s)', type(context.module).__qualname__, mpath)

        in_syms = self.to_symbols(args)
        out_syms = []  # TODO(@daskol): Should we run nested tracer?
        # TODO(@daskol): How to flatten in abstract way?
        # kwsyms = {t: s for t, s in zip(args, self.to_symbols(args))}

        # It is important to access `__dict__` directly since `__getattr__` is
        # overriden.
        params = {f.name: context.module.__dict__.get(f.name)
                  for f in fields(context.module) if f.name != 'parent'}
        module_info = type(context.module), context.method_name
        child = Mox(in_syms, out_syms, params, [], *module_info)

        parent = self.block_stack[-1]
        parent.children += [child]
        self.block_stack += [child]

        # Forward call further and its result as is.
        result = fn(*args, **kwargs)

        logger.debug('exit  %s (%s)', type(context.module).__qualname__, mpath)
        self.block_stack.pop()
        self.module_stack.pop()
        return result

    def append(self, prim: jex.core.Primitive, params: dict[str, Any],
               in_tracers: list[ModuleTracer],
               out_tracers: list[ModuleTracer]):
        in_symbols = self.to_symbols(in_tracers)
        out_symbols = self.to_symbols(out_tracers)
        eq = Equation(in_symbols, out_symbols, params, prim)

        block = self.block_stack[-1]
        block.children.append(eq)

    def set_inputs(self, tracers: Sequence[ModuleTracer]):
        self.root.inputs.clear()
        self.root.inputs.extend(self.to_symbols(tracers))

    def set_outputs(self, tracers: Sequence[ModuleTracer] | ModuleTracer):
        if not isinstance(tracers, list | tuple | Sequence):
            tracers = [tracers]
        self.root.outputs.clear()
        self.root.outputs.extend(self.to_symbols(tracers))

    def to_symbols(self, tracers: Sequence[ModuleTracer]) -> Sequence[Symbol]:
        symbols = []
        for tracer in tracers:
            if (symbol := self.symbols.get(tracer)) is None:
                symbol = Symbol(tracer.aval)
                self.symbols[tracer] = symbol
            symbols += [symbol]
        return symbols


def fully_qualified_name(ty: Type[nn.Module]) -> str:
    if ty.__module__ == 'builtins':
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
        match child:
            case Mox():
                dump(child, fileobj, depth=depth + 1)
            case Equation():
                print(f'{indent}  eq {child.prim.name}', file=fileobj)
                print(f'{indent}  inputs ={child.inputs}', file=fileobj)
                print(f'{indent}  outputs={child.outputs}', file=fileobj)
                print(f'{indent}  {child}', file=fileobj)
    print(f'{indent}}}', file=fileobj)


class XPath:
    """XPath expression."""


def eval_module(read, write, mox: Mox, in_tree) -> None:
    pass


def eval_equation(read, write, eq: Equation) -> None:
    in_vals = [read(x) for x in eq.inputs]  # TODO(@daskol): Trees?
    subfuns, params = eq.prim.get_bind_params(eq.params)
    out_vals = eq.prim.bind(*subfuns, *in_vals, **params)
    if not eq.prim.multiple_results:
        out_vals = [out_vals]
    for sym, val in zip(eq.outputs, out_vals):
        write(sym, val)


def mtree_eval(tree: Mox, *args, **kwargs):
    """Evaluate a module expression `tree` with `args` and `kwargs`."""
    Var = Any
    env: dict[Var, Any] = {}

    def read(var: Var) -> Any:
        return env[var]

    def write(var: Var, val: Any):
        assert var not in env, f'Variable {var} has been already defined.'
        env[var] = val

    def fn(node: Expr) -> Mox | None:
        if isinstance(node, Mox):
            if node.is_ephemeral:
                return node
            else:
                return eval_module(read, write, node, in_tree)
        elif isinstance(node, Equation):
            return eval_equation(read, write, node)

    # Initialize execution context and execute.
    # TODO(@daskol): Use `in_tree`.
    flatten_args, in_tree = jax.tree.flatten(args)
    for symbol, value in zip(tree.inputs, flatten_args):
        write(symbol, value)
    mtree_map(fn, tree)

    flatten_res = [read(x) for x in tree.outputs]
    if len(flatten_res) == 1:
        return flatten_res[0]
    return flatten_res


def mtree_map(fn: Callable[[Expr], Any], tree: Mox):
    """Apply map transformation `fn` to a module `tree`."""
    nodes = [tree]
    while nodes:
        node: Expr = nodes.pop()
        if isinstance(res := fn(node), Mox):
            nodes += res.children


def mtree_sub(expr: str | XPath, tree: Mox, subtree: Mox) -> Mox:
    """Substitute a modules or subtrees of `tree` with `subtree` according to
    matching pattern `expr`.
    """


def mtree_query(expr: str | XPath, tree: Mox) -> Sequence[Any]:
    """Get modules or their properties by XPath expression.

    >>> class ResBlock(nn.Module):
    >>>     @nn.compact
    >>>     def __call__(self, xs):
    >>>         return xs + nn.Dense(10)(xs)
    >>> mtree = mox(ResBlock().init)(jax.random.PRNGKey(42),
    >>>                              jnp.empty((2, 10))
    >>> # Query all modules with 10 output features.
    >>> mtree_query('//[@features=10]')
    [nn.Dense(10)]
    """
