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
#include <span>

#include <nanobind/nanobind.h>

#include "kernel_nanobind_helpers.h"

namespace nb = nanobind;

namespace fewbit {

constexpr long double kSqrt2 = 1.4142135623730950488;

template <typename T> constexpr T Gelu(T value) {
    constexpr T sqrt2 = kSqrt2;
    return 0.5 * value * (1 + std::erf(value / sqrt2));
}

template <typename T> void Gelu(void *out, void const **ins) {
    auto size = *reinterpret_cast<int64_t const *>(ins[0]);
    auto xs = reinterpret_cast<T const *>(ins[1]);
    auto ys = reinterpret_cast<T *>(out);

    for (int64_t ix = 0; ix != size; ++ix) {
        ys[ix] = Gelu(xs[ix]);
    }
}

template <typename T>
void GeluFwd(std::span<T const> input, T *output, uint8_t *primals,
             std::span<T const> boundaries) {
    int bit_width = std::popcount(boundaries.size());
    int max_width = 8 * sizeof(uint8_t);
    int threshold = max_width - bit_width;
    int shift = 0;
    for (auto const in : input) {
        // Compute activation function output.
        *output = Gelu(in);
        ++output;

        // Quantize input.
        auto pos = std::lower_bound(boundaries.begin(), boundaries.end(), in);
        auto ix = std::distance(boundaries.begin(), pos);

        // Compress quantization index to bit stream.
        if (shift <= threshold) {
            *primals |= static_cast<uint8_t>(ix << shift);
            shift = (shift + bit_width) % max_width;
        } else {
            auto lo = (ix << shift) & 0xff;
            auto hi = ix >> (max_width - shift);
            shift = shift + bit_width - max_width;
            primals[0] |= static_cast<uint8_t>(lo);
            primals[1] = static_cast<uint8_t>(hi);
            ++primals;
        }
    }
}

template <typename T> void GeluFwd(void *out, void const **ins) {
    auto size = *reinterpret_cast<uint64_t const *>(ins[0]);
    auto xs = reinterpret_cast<T const *>(ins[1]);
    auto bits = *reinterpret_cast<uint64_t const *>(ins[2]);
    auto bs = reinterpret_cast<T const *>(ins[3]); // Boundaries.

    auto outs = reinterpret_cast<void **>(out);
    auto ys = reinterpret_cast<T *>(outs[0]);
    auto ps = reinterpret_cast<uint8_t *>(outs[1]);

    GeluFwd<T>({xs, size}, ys, ps, {bs, (1u << bits) - 1});
}

template <typename T>
void GradientQuantizedBwd(std::span<uint8_t const> primals,
                          std::span<T const> out_cotangets, T *in_cotangents,
                          std::span<T const> levels) {
    int bit_width = std::popcount(levels.size() - 1);
    int max_width = 8 * sizeof(uint8_t);
    int shift = 0;
    auto mask = (1u << bit_width) - 1;
    auto primal = primals.begin();
    for (auto out_cotangent : out_cotangets) {
        // Inflate compressed quantization level indices.
        uint64_t index = (*primal >> shift) & mask;
        shift = shift + bit_width;
        if (shift > max_width) {
            ++primal;
            shift -= max_width;
            index |= (*primal << (bit_width - shift)) & mask;
        }

        // Scale input cotangents with quantization level.
        *in_cotangents = levels[index] * out_cotangent;
        ++in_cotangents;
    }
}

template <typename T> void GradientQuantizedBwd(void *out, void const **ins) {
    auto bits = *reinterpret_cast<uint64_t const *>(ins[0]);
    auto ps = reinterpret_cast<uint8_t const *>(ins[1]); // Primals
    auto ls = reinterpret_cast<T const *>(ins[2]);       // Levels
    auto size = *reinterpret_cast<uint64_t const *>(ins[3]);
    auto xs = reinterpret_cast<T const *>(ins[4]);

    auto ys = reinterpret_cast<T *>(out);

    constexpr auto max_width = 8 * sizeof(uint8_t);
    auto length = (bits * size) / max_width;
    if ((bits * size) % max_width > 0) {
        ++length;
    }

    GradientQuantizedBwd<T>({ps, length}, {xs, size}, ys, {ls, 1u << bits});
}

nb::capsule EncapsulateFunction(void (*fn)(void *, void const **)) {
    return jax::EncapsulateFunction(fn);
}

nb::dict Registrations() {
    nb::dict dict;
    dict["cpu_gelu_f32"] = EncapsulateFunction(Gelu<float>);
    dict["cpu_gelu_f64"] = EncapsulateFunction(Gelu<double>);
    dict["cpu_gelu_fwd_f32"] = EncapsulateFunction(GeluFwd<float>);
    dict["cpu_gelu_fwd_f64"] = EncapsulateFunction(GeluFwd<double>);
    dict["cpu_gradient_quantized_bwd_f32"] =
        EncapsulateFunction(GradientQuantizedBwd<float>);
    dict["cpu_gradient_quantized_bwd_f64"] =
        EncapsulateFunction(GradientQuantizedBwd<double>);
    return dict;
}

NB_MODULE(_fewbit, m) {
    m.def("registrations", &Registrations);
}

} // namespace fewbit
