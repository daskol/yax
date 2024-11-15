/**
 * Copyright 2024 Daniel Bershatsky
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>

#include <nanobind/nanobind.h>

#include "kernel_nanobind_helpers.h"

namespace nb = nanobind;

namespace fewbit {

constexpr double kSqrt2 = 1.4142135623730950488;

template <typename T> void Gelu(void *out, void const **ins) {
    auto size = *reinterpret_cast<int64_t const *>(ins[0]);
    auto xs = reinterpret_cast<T const *>(ins[1]);
    auto bits = *reinterpret_cast<int64_t const *>(ins[1]);
    auto bs = reinterpret_cast<T const *>(ins[2]); // Boundaries.

    auto outs = reinterpret_cast<void **>(out);
    auto ys = reinterpret_cast<T *>(outs[0]);
    auto ps = reinterpret_cast<T *>(outs[1]); // Primals.

    for (int64_t ix = 0; ix != size; ++ix) {
        auto tmp = static_cast<double>(xs[ix]);
        ys[ix] = 0.5 * tmp * (1 + std::erf(tmp / kSqrt2));
    }
}

nb::dict Registrations() {
    nb::dict dict;
    dict["cpu_gelu_f32"] = jax::EncapsulateFunction(Gelu<float>);
    dict["cpu_gelu_f64"] = jax::EncapsulateFunction(Gelu<double>);
    return dict;
}

NB_MODULE(_fewbit, m) {
    m.def("registrations", &Registrations);
}

} // namespace fewbit
