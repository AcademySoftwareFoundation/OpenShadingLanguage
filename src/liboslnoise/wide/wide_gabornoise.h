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

#include <OSL/wide/wide_gabornoise_fwd.h>

#include "../sfm_gabornoise.h"

// Macro to reduce repeated code
#define __OSL_SETUP_WIDE_DIRECTION                                                \
    Block<Vec3> uniformDirection;                                                 \
    if (opt_varying_direction == nullptr) {                                       \
        /* If no varying direction specified, just broadcast out */               \
        /* the uniform value from the NoiseParams (which might be the default) */ \
        assign_all(uniformDirection, opt->direction);                             \
        opt_varying_direction = &uniformDirection;                                \
    }                                                                             \
    Wide<const Vec3> wDirection(*opt_varying_direction);

OSL_NAMESPACE_ENTER

namespace __OSL_WIDE_PVT {

template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_gabor(Masked<Dual2<float>> wResult, Wide<const Dual2<float>> wX,
           NoiseParams const* opt, Block<Vec3>* opt_varying_direction)
{
    OSL_DASSERT(opt);

    OSL_FORCEINLINE_BLOCK
    {
        sfm::GaborUniformParams gup(*opt);
        __OSL_SETUP_WIDE_DIRECTION

#if !OSL_NON_INTEL_CLANG  // Control flow too complex for clang's loop vectorizor
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
#endif
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Dual2<float> x = wX[lane];
            Vec3 direction       = wDirection[lane];
            if (wResult.mask()[lane]) {
                const Dual2<Vec3> P = make_Vec3(x);

                wResult[ActiveLane(lane)]
                    = sfm::scalar_gabor<AnisotropicT, FilterPolicyT>(P, gup,
                                                                     direction);
            }
        }
    }
}

template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_gabor(Masked<Dual2<float>> wResult, Wide<const Dual2<float>> wX,
           Wide<const Dual2<float>> wY, NoiseParams const* opt,
           Block<Vec3>* opt_varying_direction)
{
    OSL_DASSERT(opt);

    OSL_FORCEINLINE_BLOCK
    {
        sfm::GaborUniformParams gup(*opt);
        __OSL_SETUP_WIDE_DIRECTION

#if !OSL_NON_INTEL_CLANG  // Control flow too complex for clang's loop vectorizor
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
#endif
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Dual2<float> x = wX[lane];
            const Dual2<float> y = wY[lane];
            Vec3 direction       = wDirection[lane];
            if (wResult.mask()[lane]) {
                const Dual2<Vec3> P = make_Vec3(x, y);
                wResult[ActiveLane(lane)]
                    = sfm::scalar_gabor<AnisotropicT, FilterPolicyT>(P, gup,
                                                                     direction);
            }
        }
    }
}

template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_gabor(Masked<Dual2<float>> wResult, Wide<const Dual2<Vec3>> wP,
           NoiseParams const* opt, Block<Vec3>* opt_varying_direction)
{
    OSL_DASSERT(opt);

    OSL_FORCEINLINE_BLOCK
    {
        sfm::GaborUniformParams gup(*opt);
        __OSL_SETUP_WIDE_DIRECTION

#if !OSL_NON_INTEL_CLANG  // Control flow too complex for clang's loop vectorizor
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
#endif
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Dual2<Vec3> P = wP[lane];
            Vec3 direction      = wDirection[lane];
            if (wResult.mask()[lane]) {
                wResult[ActiveLane(lane)]
                    = sfm::scalar_gabor<AnisotropicT, FilterPolicyT>(P, gup,
                                                                     direction);
            }
        }
    }
}


template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_gabor3(Masked<Dual2<Vec3>> wResult, Wide<const Dual2<float>> wX,
            NoiseParams const* opt, Block<Vec3>* opt_varying_direction)
{
    OSL_DASSERT(opt);

    OSL_FORCEINLINE_BLOCK
    {
        sfm::GaborUniformParams gup(*opt);
        __OSL_SETUP_WIDE_DIRECTION

#if !OSL_NON_INTEL_CLANG  // Control flow too complex for clang's loop vectorizor
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
#endif
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Dual2<float> x = wX[lane];
            Vec3 direction       = wDirection[lane];
            if (wResult.mask()[lane]) {
                const Dual2<Vec3> P = make_Vec3(x);
                wResult[ActiveLane(lane)]
                    = sfm::scalar_gabor3<AnisotropicT, FilterPolicyT>(P, gup,
                                                                      direction);
            }
        }
    }
}

template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_gabor3(Masked<Dual2<Vec3>> wResult, Wide<const Dual2<float>> wX,
            Wide<const Dual2<float>> wY, NoiseParams const* opt,
            Block<Vec3>* opt_varying_direction)
{
    OSL_DASSERT(opt);

    OSL_FORCEINLINE_BLOCK
    {
        sfm::GaborUniformParams gup(*opt);
        __OSL_SETUP_WIDE_DIRECTION

#if !OSL_NON_INTEL_CLANG  // Control flow too complex for clang's loop vectorizor
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
#endif
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Dual2<float> x = wX[lane];
            const Dual2<float> y = wY[lane];
            Vec3 direction       = wDirection[lane];
            if (wResult.mask()[lane]) {
                const Dual2<Vec3> P = make_Vec3(x, y);
                wResult[ActiveLane(lane)]
                    = sfm::scalar_gabor3<AnisotropicT, FilterPolicyT>(P, gup,
                                                                      direction);
            }
        }
    }
}

template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_gabor3(Masked<Dual2<Vec3>> wResult, Wide<const Dual2<Vec3>> wP,
            NoiseParams const* opt, Block<Vec3>* opt_varying_direction)
{
    OSL_DASSERT(opt);

    OSL_FORCEINLINE_BLOCK
    {
        sfm::GaborUniformParams gup(*opt);
        __OSL_SETUP_WIDE_DIRECTION

#if !OSL_NON_INTEL_CLANG  // Control flow too complex for clang's loop vectorizor
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
#endif
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Dual2<Vec3> P = wP[lane];
            Vec3 direction      = wDirection[lane];
            if (wResult.mask()[lane]) {
                wResult[ActiveLane(lane)]
                    = sfm::scalar_gabor3<AnisotropicT, FilterPolicyT>(P, gup,
                                                                      direction);
            }
        }
    }
}


template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_pgabor(Masked<Dual2<float>> wResult, Wide<const Dual2<float>> wX,
            Wide<const float> wXp, NoiseParams const* opt,
            Block<Vec3>* opt_varying_direction)
{
    OSL_DASSERT(opt);

    OSL_FORCEINLINE_BLOCK
    {
        sfm::GaborUniformParams gup(*opt);
        __OSL_SETUP_WIDE_DIRECTION

#if !OSL_NON_INTEL_CLANG  // Control flow too complex for clang's loop vectorizor
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
#endif
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Dual2<float> x = wX[lane];
            const float xperiod  = wXp[lane];
            Vec3 direction       = wDirection[lane];
            if (wResult.mask()[lane]) {
                const Dual2<Vec3> P = make_Vec3(x);
                Vec3 Pperiod(xperiod, 0.0f, 0.0f);

                wResult[ActiveLane(lane)]
                    = sfm::scalar_pgabor<AnisotropicT, FilterPolicyT>(
                        P, Pperiod, gup, direction);
            }
        }
    }
}


template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_pgabor(Masked<Dual2<float>> wResult, Wide<const Dual2<float>> wX,
            Wide<const Dual2<float>> wY, Wide<const float> wXp,
            Wide<const float> wYp, NoiseParams const* opt,
            Block<Vec3>* opt_varying_direction)
{
    OSL_DASSERT(opt);

    OSL_FORCEINLINE_BLOCK
    {
        sfm::GaborUniformParams gup(*opt);
        __OSL_SETUP_WIDE_DIRECTION

#if !OSL_NON_INTEL_CLANG  // Control flow too complex for clang's loop vectorizor
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
#endif
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Dual2<float> x = wX[lane];
            const Dual2<float> y = wY[lane];

            const float xperiod = wXp[lane];
            const float yperiod = wYp[lane];
            Vec3 direction      = wDirection[lane];
            if (wResult.mask()[lane]) {
                const Dual2<Vec3> P = make_Vec3(x, y);
                Vec3 Pperiod(xperiod, yperiod, 0.0f);

                wResult[ActiveLane(lane)]
                    = sfm::scalar_pgabor<AnisotropicT, FilterPolicyT>(
                        P, Pperiod, gup, direction);
            }
        }
    }
}

template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_pgabor(Masked<Dual2<float>> wResult, Wide<const Dual2<Vec3>> wP,
            Wide<const Vec3> wPp, NoiseParams const* opt,
            Block<Vec3>* opt_varying_direction)
{
    OSL_DASSERT(opt);

    OSL_FORCEINLINE_BLOCK
    {
        sfm::GaborUniformParams gup(*opt);
        __OSL_SETUP_WIDE_DIRECTION

#if !OSL_NON_INTEL_CLANG  // Control flow too complex for clang's loop vectorizor
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
#endif
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Dual2<Vec3> P = wP[lane];
            Vec3 Pperiod        = wPp[lane];
            Vec3 direction      = wDirection[lane];
            if (wResult.mask()[lane]) {
                wResult[ActiveLane(lane)]
                    = sfm::scalar_pgabor<AnisotropicT, FilterPolicyT>(
                        P, Pperiod, gup, direction);
            }
        }
    }
}



template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_pgabor3(Masked<Dual2<Vec3>> wResult, Wide<const Dual2<float>> wX,
             Wide<const float> wXp, NoiseParams const* opt,
             Block<Vec3>* opt_varying_direction)
{
    OSL_DASSERT(opt);

    OSL_FORCEINLINE_BLOCK
    {
        sfm::GaborUniformParams gup(*opt);
        __OSL_SETUP_WIDE_DIRECTION

#if !OSL_NON_INTEL_CLANG  // Control flow too complex for clang's loop vectorizor
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
#endif
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Dual2<float> x = wX[lane];
            const float xperiod  = wXp[lane];
            Vec3 direction       = wDirection[lane];

            if (wResult.mask()[lane]) {
                const Dual2<Vec3> P = make_Vec3(x);
                Vec3 Pperiod(xperiod, 0.0f, 0.0f);
                wResult[ActiveLane(lane)]
                    = sfm::scalar_pgabor3<AnisotropicT, FilterPolicyT>(
                        P, Pperiod, gup, direction);
            }
        }
    }
}


template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_pgabor3(Masked<Dual2<Vec3>> wResult, Wide<const Dual2<float>> wX,
             Wide<const Dual2<float>> wY, Wide<const float> wXp,
             Wide<const float> wYp, NoiseParams const* opt,
             Block<Vec3>* opt_varying_direction)
{
    OSL_DASSERT(opt);

    OSL_FORCEINLINE_BLOCK
    {
        sfm::GaborUniformParams gup(*opt);
        __OSL_SETUP_WIDE_DIRECTION

#if !OSL_NON_INTEL_CLANG  // Control flow too complex for clang's loop vectorizor
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
#endif
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Dual2<float> x = wX[lane];
            const Dual2<float> y = wY[lane];

            const float xperiod = wXp[lane];
            const float yperiod = wYp[lane];
            Vec3 direction      = wDirection[lane];

            if (wResult.mask()[lane]) {
                const Dual2<Vec3> P = make_Vec3(x, y);
                Vec3 Pperiod(xperiod, yperiod, 0.0f);
                wResult[ActiveLane(lane)]
                    = sfm::scalar_pgabor3<AnisotropicT, FilterPolicyT>(
                        P, Pperiod, gup, direction);
            }
        }
    }
}

template<int AnisotropicT, typename FilterPolicyT>
OSL_NOINLINE void
wide_pgabor3(Masked<Dual2<Vec3>> wResult, Wide<const Dual2<Vec3>> wP,
             Wide<const Vec3> wPp, NoiseParams const* opt,
             Block<Vec3>* opt_varying_direction)
{
    OSL_DASSERT(opt);

    OSL_FORCEINLINE_BLOCK
    {
        sfm::GaborUniformParams gup(*opt);
        __OSL_SETUP_WIDE_DIRECTION

#if !OSL_NON_INTEL_CLANG  // Control flow too complex for clang's loop vectorizor
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
#endif
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Dual2<Vec3> P = wP[lane];
            Vec3 Pperiod        = wPp[lane];
            Vec3 direction      = wDirection[lane];
            if (wResult.mask()[lane]) {
                wResult[ActiveLane(lane)]
                    = sfm::scalar_pgabor3<AnisotropicT, FilterPolicyT>(
                        P, Pperiod, gup, direction);
            }
        }
    }
}



}  // namespace __OSL_WIDE_PVT

#undef __OSL_SETUP_WIDE_DIRECTION

OSL_NAMESPACE_EXIT
