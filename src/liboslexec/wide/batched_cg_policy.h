// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifndef __OSL_BATCHED_CG_POLICY_H
#define __OSL_BATCHED_CG_POLICY_H

#include <OSL/oslconfig.h>

#ifndef __OSL_BATCHED_CG_SIMD_THRESHOLD
#    define __OSL_BATCHED_CG_SIMD_THRESHOLD 2
#endif

using namespace OSL;

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

namespace {
enum class Param { F, V, WF, WV, WDF, WDV };

// Code gen policy to control how a function will
// be emitted for a batch for a particular parameter list
template<Param... ParamListT> struct BatchedCGPolicy {
    // Work will be done per lane until
    // occupancy reached the simd_threshold
    // then it will be use explicit vectorization
    // over all lanes in the batch
    static constexpr int simd_threshold = __OSL_BATCHED_CG_SIMD_THRESHOLD;
};

// Example usage, specialize for concrete ParamListT
// to change the simd_threshold for WV, WF, WF to 6
// and for WV, WV to 4.
//
//     template<>
//     struct BatchedCGPolicy<Param::WV,Param::WF,Param::WF> {
//         static constexpr int simd_threshold = 6;
//     };
//     template<>
//     struct BatchedCGPolicy<Param::WV,Param::WV> {
//         static constexpr int simd_threshold = 4;
//     };

}  // namespace

}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#endif
