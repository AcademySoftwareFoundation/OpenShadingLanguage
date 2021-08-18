// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifndef __OSL_WIDTH
#    error must define __OSL_WIDTH to number of SIMD lanes before including this header
#endif

#include <OSL/oslconfig.h>

#include <OSL/dual_vec.h>
#include <OSL/wide.h>

OSL_NAMESPACE_ENTER

struct NoiseParams;

namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

// Foward declaration, implementation is in liboslnoise/wide_gabornoise.h
struct DisabledFilterPolicy {
    static constexpr bool active = false;
};

struct EnabledFilterPolicy {
    static constexpr bool active = true;
};


template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_gabor(Masked<Dual2<float>> wResult, Wide<const Dual2<float>> wX,
           NoiseParams const* opt, Block<Vec3>* opt_varying_direction);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_gabor(Masked<Dual2<float>> wResult, Wide<const Dual2<float>> wX,
           Wide<const Dual2<float>> wY, NoiseParams const* opt,
           Block<Vec3>* opt_varying_direction);

template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_gabor(Masked<Dual2<float>> wResult, Wide<const Dual2<Vec3>> wP,
           NoiseParams const* opt, Block<Vec3>* opt_varying_direction);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_gabor3(Masked<Dual2<Vec3>> wResult, Wide<const Dual2<float>> wX,
            NoiseParams const* opt, Block<Vec3>* opt_varying_direction);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_gabor3(Masked<Dual2<Vec3>> wResult, Wide<const Dual2<float>> wX,
            Wide<const Dual2<float>> wY, NoiseParams const* opt,
            Block<Vec3>* opt_varying_direction);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_gabor3(Masked<Dual2<Vec3>> wResult, Wide<const Dual2<Vec3>> wP,
            NoiseParams const* opt, Block<Vec3>* opt_varying_direction);



//////////////////////////////////////////////////////////////
// Periodic gabor(s)
//////////////////////////////////////////////////////////////

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_pgabor(Masked<Dual2<float>> wResult, Wide<const Dual2<float>> wX,
            Wide<const float> wXp, NoiseParams const* opt,
            Block<Vec3>* opt_varying_direction);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_pgabor(Masked<Dual2<float>> wResult, Wide<const Dual2<float>> wX,
            Wide<const Dual2<float>> wY, Wide<const float> wXp,
            Wide<const float> wYp, NoiseParams const* opt,
            Block<Vec3>* opt_varying_direction);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_pgabor(Masked<Dual2<float>> wResult, Wide<const Dual2<Vec3>> wP,
            Wide<const Vec3> wPp, NoiseParams const* opt,
            Block<Vec3>* opt_varying_direction);



// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_pgabor3(Masked<Dual2<Vec3>> wResult, Wide<const Dual2<float>> wX,
             Wide<const float> wXp, NoiseParams const* opt,
             Block<Vec3>* opt_varying_direction);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_pgabor3(Masked<Dual2<Vec3>> wResult, Wide<const Dual2<float>> wX,
             Wide<const Dual2<float>> wY, Wide<const float> wXp,
             Wide<const float> wYp, NoiseParams const* opt,
             Block<Vec3>* opt_varying_direction);

// Foward declaration, implementation is in wide_gabor.h
template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_pgabor3(Masked<Dual2<Vec3>> wResult, Wide<const Dual2<Vec3>> wP,
             Wide<const Vec3> wPp, NoiseParams const* opt,
             Block<Vec3>* opt_varying_direction);



}  // namespace __OSL_WIDE_PVT

OSL_NAMESPACE_EXIT
