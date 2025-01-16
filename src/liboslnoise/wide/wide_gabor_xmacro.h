// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/oslconfig.h>

#include "define_opname_macros.h"

#ifdef __OSL_XMACRO_ARGS

#    define __OSL_XMACRO_ANISOTROPIC \
        __OSL_EXPAND(__OSL_XMACRO_ARG1 __OSL_XMACRO_ARGS)
#    define __OSL_XMACRO_FILTER_POLICY \
        __OSL_EXPAND(__OSL_XMACRO_ARG2 __OSL_XMACRO_ARGS)

#endif

#ifndef __OSL_XMACRO_ANISOTROPIC
#    error must define __OSL_XMACRO_ANISOTROPIC to name an enum value representing the desired anisotropic
#endif

#ifndef __OSL_XMACRO_FILTER_POLICY
#    error must define __OSL_XMACRO_FILTER_POLICY to __OSL_XMACRO_FILTER_POLICY or EnabledFilterPolicy
#endif

#include "wide_gabornoise.h"


OSL_NAMESPACE_BEGIN

namespace __OSL_WIDE_PVT {

// Explicit template instantiation
template void
wide_gabor<__OSL_XMACRO_ANISOTROPIC, __OSL_XMACRO_FILTER_POLICY>(
    Masked<Dual2<float>> wResult, Wide<const Dual2<float>> wX,
    NoiseParams const* opt, Block<Vec3>* opt_varying_direction);

template void
wide_gabor<__OSL_XMACRO_ANISOTROPIC, __OSL_XMACRO_FILTER_POLICY>(
    Masked<Dual2<float>> wResult, Wide<const Dual2<float>> wX,
    Wide<const Dual2<float>> wY, NoiseParams const* opt,
    Block<Vec3>* opt_varying_direction);

template void
wide_gabor<__OSL_XMACRO_ANISOTROPIC, __OSL_XMACRO_FILTER_POLICY>(
    Masked<Dual2<float>> wResult, Wide<const Dual2<Vec3>> wP,
    NoiseParams const* opt, Block<Vec3>* opt_varying_direction);


template void
wide_pgabor<__OSL_XMACRO_ANISOTROPIC, __OSL_XMACRO_FILTER_POLICY>(
    Masked<Dual2<float>> wResult, Wide<const Dual2<float>> wX,
    Wide<const float> wXp, NoiseParams const* opt,
    Block<Vec3>* opt_varying_direction);

template void
wide_pgabor<__OSL_XMACRO_ANISOTROPIC, __OSL_XMACRO_FILTER_POLICY>(
    Masked<Dual2<float>> wResult, Wide<const Dual2<float>> wX,
    Wide<const Dual2<float>> wY, Wide<const float> wXp, Wide<const float> wYp,
    NoiseParams const* opt, Block<Vec3>* opt_varying_direction);

template void
wide_pgabor<__OSL_XMACRO_ANISOTROPIC, __OSL_XMACRO_FILTER_POLICY>(
    Masked<Dual2<float>> wResult, Wide<const Dual2<Vec3>> wP,
    Wide<const Vec3> wPp, NoiseParams const* opt,
    Block<Vec3>* opt_varying_direction);

};  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_END
