"""Module `yax` implements module tracing routines, module expression (MoX)
building and querying.
"""

import logging
from dataclasses import dataclass, field, fields
from functools import wraps
from io import StringIO
from typing import Any, ParamSpec, Self, Sequence, Type, TypeAlias

import flax.linen as nn
import jax
import jax.extend as jex
import jax.extend.linear_util as lu
import jax.numpy as jnp
from flax.linen.module import Callable, InterceptorContext, intercept_methods
from jax.core import MainTrace, Sublevel, Trace, Tracer, new_main

# TODO(@daskol): Python 3.12 introduced new type parameter syntax (PEP-0695)
# but some code quality tools (e.g. yapf) do not support this syntax.
Args = ParamSpec('Args')

logger = logging.getLogger(__name__)


class ModuleTracer(Tracer):

    def __init__(self, trace: Trace, value):
        super().__init__(trace)
        self.value = value

    @property
    def aval(self):
        # TODO(@daskol): Should be cached or precomputed.
        return jax.api_util.shaped_abstractify(self.value)

    def full_lower(self):
        # TODO(@daskol): Full lower to some abstract array type.
        # return jax.core.full_lower(self.value)
        return jnp.empty(self.value.shape, self.value.dtype)


class ModuleTrace(Trace):

    def __init__(self, main: MainTrace, sublevel: Sublevel, **kwargs) -> None:
        super().__init__(main, sublevel)
        self.builder: MoxBuilder = kwargs.get('builder')

    def pure(self, val):
        return ModuleTracer(self, val)

    def lift(self, val):
        return ModuleTracer(self, val)

    def process_primitive(self, prim: jex.core.Primitive, tracers, params):
        print(f'process_primitive(prim={prim})')
        logger.debug('tracers=%s; params=%s', tracers, params)
        outs, _ = prim.abstract_eval(*[x.aval for x in tracers], **params)
        # TODO(@daskol): Add tracers too?
        self.builder.append(prim, tracers, params)
        if prim.multiple_results:
            return [ModuleTracer(self, out) for out in outs]
        else:
            return ModuleTracer(self, outs)


@lu.transformation
def _lower(main: MainTrace, *ins):
    trace: ModuleTrace = main.with_cur_sublevel()
    in_tracers = [trace.lift(x) for x in ins]
    out_tracers = yield in_tracers, {}
    outs = [trace.full_raise(x).value for x in out_tracers]
    yield outs


@lu.transformation
def _raise(builder, *ins):
    with new_main(ModuleTrace, builder=builder) as main:
        outs = yield (main, *ins), {}
        del main
    yield outs


def trace_modules(wf: lu.WrappedFun, builder: 'MoxBuilder') -> lu.WrappedFun:
    return _raise(_lower(wf), builder)


@dataclass(repr=False)
class Block:
    """Linear block of code.

    Args:
      module:
      entrypoint:
      attrs:
      children:
    """

    # TODO(@daskol): Multiple invokations of the same block produces metadata
    # duplication, i.e. module_ty and attrs (and entrypoint sometimes).

    module_ty: Type[nn.Module]

    entrypoint: str

    attrs: dict[str, Any]

    children: list['Block', 'Instr'] = field(default_factory=list)

    # def __eq__(self, other):
    #     pass

    def __hash__(self):
        return id(self.module)

    def __repr__(self):
        name = f'{self.module_ty.__module__}.{self.module_ty.__qualname__}'
        return name

    @classmethod
    def from_interceptor_context(cls, context: InterceptorContext) -> Self:
        # It is important to access `__dict__` directly since `__getattr__` is
        # overriden.
        attrs = {f.name: context.module.__dict__.get(f.name)
                 for f in fields(context.module) if f.name != 'parent'}
        return cls(type(context.module), context.method_name, attrs)


@dataclass
class Instr:

    name: str


@dataclass(repr=False)
class Root:

    children: list[Block | Instr] = field(default_factory=list)

    def __repr__(self):
        buf = StringIO()
        print_node(self, fileobj=buf)
        return buf.getvalue()


Node: TypeAlias = Block | Instr | Root

# TODO(@daskol): Root node should be an internal node?
Mox: TypeAlias = Root


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
        logger.debug(wf)
        logger.debug('=' * 80)
        in_args, in_tree = jax.tree.flatten((args, kwargs))
        logger.debug(in_tree)
        wf, out_tree_thunk = jax.api_util.flatten_fun(wf, in_tree)
        logger.debug(wf)
        logger.debug('=' * 80)
        builder = MoxBuilder()
        wf = trace_modules(wf, builder)
        logger.debug(wf)
        logger.debug('=' * 80)
        with intercept_methods(builder.intercept):
            out_args = wf.call_wrapped(*in_args)
        logger.debug(out_args)
        out_tree = out_tree_thunk()
        logger.debug(out_tree)
        res = jax.tree.unflatten(out_tree, out_args)
        logger.debug('result: %s', res)
        return builder.build()
    return wrapper


def fully_qualified_name(ty: Type[nn.Module]) -> str:
    if ty.__module__ == 'builtins':
        return ty.__qualname__
    else:
        return f'{ty.__module__}.{ty.__qualname__}'


class MoxBuilder:

    def __init__(self):
        # TODO(@daskol): HuggingFace?
        self.root = Root()
        self.block_stack: list[Block] = [self.root]

        self.module_stack: list[InterceptorContext] = []

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

        child = Block.from_interceptor_context(context)
        parent = self.block_stack[-1]
        parent.children += [child]
        self.block_stack += [child]

        # Forward call further and its result as is.
        result = fn(*args, **kwargs)

        logger.debug('exit  %s (%s)', type(context.module).__qualname__, mpath)
        self.block_stack.pop()
        self.module_stack.pop()
        return result

    def append(self, prim: jex.core.Primitive, tracers, *args, **kwargs):
        # Some models are not inhereted from `flax.linen.Module`. They define
        # an aggregate type on top of `flax.linen.Module` (e.g. HuggingFace
        # `trasnformers`). Thus, there are some ops which are outside of
        # module.
        if len(self.module_stack) == 0:
            print(f'primitive {prim} does not have root module')
            return

        if prim.name == 'pjit':
            params, *_ = args
            jaxpr = params['jaxpr']
            instr = Instr(jaxpr)
        else:
            instr = Instr(prim)
        block = self.block_stack[-1]
        block.children.append(instr)

    def to_string(self):
        buf = StringIO()
        print_node(self.root, fileobj=buf)
        return buf.getvalue()


def print_node(node: Node, depth=0, fileobj=None):
    indent = '  ' * depth
    match node:
        case Root():
            print(f'{indent}mod {{ # {depth}', file=fileobj)
        case Block():
            name_ty = fully_qualified_name(node.module_ty)
            name = node.attrs['name']
            keys = (k for k in node.attrs.keys()
                    if k not in ('name', 'parent'))
            try:
                key = next(keys)
                val = node.attrs[key]
                attrs = f'{key}={val}'
            except StopIteration:
                attrs = ''
            print(f'{indent}mod {name} : {name_ty}({attrs}) {{ # {depth}',
                  file=fileobj)
    for child in node.children:
        match child:
            case Block():
                print_node(child, depth + 1, fileobj=fileobj)
            case Instr(name):
                print(f'{indent}  {child.name}', file=fileobj)
    print(f'{indent}}}', file=fileobj)


class XPath:
    """XPath expression."""


def mtree_map(fn: Callable[[Node], Any], tree: Mox):
    """Apply map transformation `fn` to a module `tree`."""


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
