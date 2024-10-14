import inspect
from functools import wraps

import flax.linen as nn
import jax
import jax.extend as jex
import jax.extend.linear_util as lu
import jax.numpy as jnp
from jax.core import MainTrace, Trace, Tracer, new_main


class FlaxModuleTracer(Tracer):

    def __init__(self, trace: Trace, value):
        super().__init__(trace)
        self.value = value

    @property
    def aval(self):
        # aval = jax.core.get_aval(self.value)
        aval = jax.api_util.shaped_abstractify(self.value)
        self.log('aval', f'aval={aval}', f'val={self.value}')
        return aval

    def full_lower(self):
        self.log('full_lower()', f'value={self.value}')
        res = jax.core.full_lower(self.value)
        res = jnp.empty(self.value.shape, self.value.dtype)
        self.log('full_lower()', f'result={res}')
        return res

    def log(self, method, *args, **kwargs):
        print(f'{self.__class__.__name__}::{method}:', *args, **kwargs)


class FlaxModuleTrace(Trace):

    def pure(self, val):
        return FlaxModuleTracer(self, val)

    def lift(self, val):
        return FlaxModuleTracer(self, val)

    def process_primitive(self, prim: jex.core.Primitive, tracers, params):
        print(f'process_primitive(prim={prim})')
        print('tracers:', tracers)
        print('params:', params)
        match prim.name:
            case 'pjit':
                outs, _ = prim.abstract_eval(*tracers, **params)
                print(outs)
            case _:
                with jax.core.new_sublevel():
                    outs, _ = prim.abstract_eval(*tracers, **params)
                print(outs)
        if prim.multiple_results:
            return [FlaxModuleTracer(self, out) for out in outs]
        else:
            return FlaxModuleTracer(self, outs)


@lu.transformation
def _lower(main: MainTrace, *ins):
    trace: FlaxModuleTrace = main.with_cur_sublevel()
    in_tracers = [trace.lift(x) for x in ins]
    print('_lower()')
    out_tracers = yield in_tracers, {}
    print('_lower()')
    outs = [trace.full_raise(x).value for x in out_tracers]
    yield outs


@lu.transformation
def _raise(*ins):
    with new_main(FlaxModuleTrace) as main:
        outs = yield (main, *ins), {}
        del main
    yield outs


def trace_flax(wf: lu.WrappedFun) -> lu.WrappedFun:
    return _raise(_lower(wf))


def trace(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        wf = lu.wrap_init(fn)
        print(wf)
        print('=' * 80)
        in_args, in_tree = jax.tree.flatten((args, kwargs))
        print(in_tree)
        wf, out_tree_thunk = jax.api_util.flatten_fun(wf, in_tree)
        print(wf)
        print('=' * 80)
        wf = trace_flax(wf)
        print(wf)
        print('=' * 80)
        out_args = wf.call_wrapped(*in_args)
        print(out_args)
        out_tree = out_tree_thunk()
        print(out_tree)
        return jax.tree.unflatten(out_tree, out_args)

    return wrapper


def find_modules():
    modules = []
    for frame_info in inspect.stack():
        if (self := frame_info.frame.f_locals.get('self')) is None:
            continue
        if not isinstance(self, nn.Module):
            continue
        if modules and modules[-1][0] is self:
            continue
        if self.scope is None:
            break
        modules += [(self, self.__class__.__qualname__)]
        print(len(modules), id(self), self, self.scope)
    print()
    print('CURRENT FRAME:', '/' + '/'.join(n for _, n in modules))
    print()
