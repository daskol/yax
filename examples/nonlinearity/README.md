# YAX: Custom Activation Functions (Few-Bit)

In this usage example, we demostrate a gradient-quantized GELU activation
function. Its is implemented in C++ as well as forward and backward rules which
are required for custom VJP in JAX. It should be enough to just build binary
extesion with custom GELU operation as follows.

```shell
meson setup build
mkdir -p subprojects
meson wrap install robin-map
meson wrap install nanobind
```

Additionally, some tests requires memory profiling tools. It requires `protoc`
to generate (un)marshaling routines in Python. By default, `meson` generates
these routines (i.e. `profile_pb2.py`).
