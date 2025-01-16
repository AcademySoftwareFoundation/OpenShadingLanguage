// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifndef __OSL_USE_REFERENCE_INT_HASH
#    define __OSL_USE_REFERENCE_INT_HASH 0
#endif
#if __OSL_USE_REFERENCE_INT_HASH
// incorrect results when vectorizing with reference hash
#    undef OSL_OPENMP_SIMD
#endif

#include "batched_cg_policy.h"
OSL_NAMESPACE_BEGIN
namespace __OSL_WIDE_PVT {

namespace {
template<>
struct BatchedCGPolicy<Param::WDF, Param::WDF, Param::WDF, Param::WF, Param::WF> {
    static constexpr int simd_threshold = 4;
};
template<> struct BatchedCGPolicy<Param::WDF, Param::WDV, Param::WV> {
    static constexpr int simd_threshold = 5;
};
template<>
struct BatchedCGPolicy<Param::WDF, Param::WDV, Param::WDF, Param::WV, Param::WF> {
    static constexpr int simd_threshold = 6;
};
}  // namespace

}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_END

// To improve parallel compile times, split noise with float results and
// Vec3 results into different cpp files
#define __OSL_XMACRO_FLOAT_RESULTS_ONLY
#define __OSL_XMACRO_ARGS (psnoise, PeriodicSNoiseScalar, PeriodicSNoise)
#include "wide_opnoise_periodic_impl_deriv_xmacro.h"
