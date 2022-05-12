// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of Trigonometric operations
/// NOTE: many functions are left as LLVM IR, but some are better to
/// execute from the library to take advantage of compiler's small vector
/// math library versions.
///
/////////////////////////////////////////////////////////////////////////


#include <cmath>

#include <OSL/oslconfig.h>

#include <OSL/batched_shaderglobals.h>
#include <OSL/dual.h>
#include <OSL/dual_vec.h>
#include <OSL/sfmath.h>
#include <OSL/wide.h>

#include <OpenImageIO/fmath.h>

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

#include "define_opname_macros.h"

#if OSL_FAST_MATH
// OIIO::fast_sin & OIIO::fast_cos are not vectorizing (presumably madd is interfering)
// so use regular sin which compiler should replace with its own fast version
#    define __OSL_XMACRO_ARGS (sin, OIIO::fast_sin, OSL::fast_sin)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (cos, OIIO::fast_cos, OSL::fast_cos)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (tan, OIIO::fast_tan, OSL::fast_tan)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (asin, OIIO::fast_asin, OSL::fast_asin)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (acos, OIIO::fast_acos, OSL::fast_acos)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (atan, OIIO::fast_atan, OSL::fast_atan)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (atan2, OIIO::fast_atan2, OSL::fast_atan2)
#    include "wide_opbinary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (sinh, OIIO::fast_sinh, OSL::fast_sinh)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (cosh, OIIO::fast_cosh, OSL::fast_cosh)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (tanh, OIIO::fast_tanh, OSL::fast_tanh)
#    include "wide_opunary_per_component_xmacro.h"


#else
// try it out and compare performance, maybe compile time flag
#    define __OSL_XMACRO_ARGS (sin, sinf, OSL::sin)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (cos, cosf, OSL::cos)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (tan, tanf, OSL::tan)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (asin, safe_asin, OSL::safe_asin)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (acos, safe_acos, OSL::safe_acos)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (atan, atanf, OSL::atan)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (atan2, atan2f, OSL::atan2)
#    include "wide_opbinary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (sinh, sinhf, OSL::sinh)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (cosh, coshf, OSL::cosh)
#    include "wide_opunary_per_component_xmacro.h"

#    define __OSL_XMACRO_ARGS (tanh, tanhf, OSL::tanh)
#    include "wide_opunary_per_component_xmacro.h"
#endif



static OSL_FORCEINLINE void
impl_sincos(float theta, float& rsine, float& rcosine)
{
#if OSL_FAST_MATH
    OIIO::fast_sincos(theta, &rsine, &rcosine);
#else
    OIIO::sincos(theta, &rsine, &rcosine);
#endif
}



static OSL_FORCEINLINE void
impl_sincos(Vec3 theta, Vec3& rsine, Vec3& rcosine)
{
    impl_sincos(theta.x, rsine.x, rcosine.x);
    impl_sincos(theta.y, rsine.y, rcosine.y);
    impl_sincos(theta.z, rsine.z, rcosine.z);
}


OSL_BATCHOP void
__OSL_OP3(sincos, Wf, Wf, Wf)(void* theta_, void* rsine_, void* rcosine_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> wtheta(theta_);
        Wide<float> wrsine(rsine_);
        Wide<float> wrcosine(rcosine_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float theta = wtheta[lane];
            float rsine;
            float rcosine;
            impl_sincos(theta, rsine, rcosine);
            wrsine[lane]   = rsine;
            wrcosine[lane] = rcosine;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(sincos, Wf, Wf, Wf)(void* theta_, void* rsine_, void* rcosine_,
                                     unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> wtheta(theta_);
        Masked<float> wrsine(rsine_, Mask(mask_value));
        Masked<float> wrcosine(rcosine_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float theta = wtheta[lane];
            if (wrsine.mask()[lane]) {
                float rsine;
                float rcosine;
                impl_sincos(theta, rsine, rcosine);

                wrsine[ActiveLane(lane)]   = rsine;
                wrcosine[ActiveLane(lane)] = rcosine;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(sincos, Wdf, Wdf, Wf)(void* theta_, void* rsine_, void* rcosine_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> wtheta(theta_);
        Wide<Dual2<float>> wrsine(rsine_);
        Wide<float> wrcosine(rcosine_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> theta = wtheta[lane];
            float rsine;
            float rcosine;
            impl_sincos(theta.val(), rsine, rcosine);
            wrsine[lane]   = Dual2<float>(rsine, rcosine * theta.dx(),
                                        rcosine * theta.dy());
            wrcosine[lane] = rcosine;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(sincos, Wdf, Wdf, Wf)(void* theta_, void* rsine_,
                                       void* rcosine_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> wtheta(theta_);
        Masked<Dual2<float>> wrsine(rsine_, Mask(mask_value));
        Masked<float> wrcosine(rcosine_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> theta = wtheta[lane];
            if (wrsine.mask()[lane]) {
                float rsine;
                float rcosine;
                impl_sincos(theta.val(), rsine, rcosine);
                wrsine[ActiveLane(lane)]   = Dual2<float>(rsine,
                                                        rcosine * theta.dx(),
                                                        rcosine * theta.dy());
                wrcosine[ActiveLane(lane)] = rcosine;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(sincos, Wdf, Wf, Wdf)(void* theta_, void* rsine_, void* rcosine_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> wtheta(theta_);
        Wide<float> wrsine(rsine_);
        Wide<Dual2<float>> wrcosine(rcosine_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> theta = wtheta[lane];
            float rsine;
            float rcosine;
            impl_sincos(theta.val(), rsine, rcosine);
            wrsine[lane]   = rsine;
            wrcosine[lane] = Dual2<float>(rcosine, -rsine * theta.dx(),
                                          -rsine * theta.dy());
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(sincos, Wdf, Wf, Wdf)(void* theta_, void* rsine_,
                                       void* rcosine_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> wtheta(theta_);
        Masked<float> wrsine(rsine_, Mask(mask_value));
        Masked<Dual2<float>> wrcosine(rcosine_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> theta = wtheta[lane];
            if (wrsine.mask()[lane]) {
                float rsine;
                float rcosine;
                impl_sincos(theta.val(), rsine, rcosine);
                wrsine[ActiveLane(lane)]   = rsine;
                wrcosine[ActiveLane(lane)] = Dual2<float>(rcosine,
                                                          -rsine * theta.dx(),
                                                          -rsine * theta.dy());
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(sincos, Wdf, Wdf, Wdf)(void* theta_, void* rsine_, void* rcosine_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> wtheta(theta_);
        Wide<Dual2<float>> wrsine(rsine_);
        Wide<Dual2<float>> wrcosine(rcosine_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> theta = wtheta[lane];
            float rsine;
            float rcosine;
            impl_sincos(theta.val(), rsine, rcosine);
            wrsine[lane]   = Dual2<float>(rsine, rcosine * theta.dx(),
                                        rcosine * theta.dy());
            wrcosine[lane] = Dual2<float>(rcosine, -rsine * theta.dx(),
                                          -rsine * theta.dy());
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(sincos, Wdf, Wdf, Wdf)(void* theta_, void* rsine_,
                                        void* rcosine_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> wtheta(theta_);
        Masked<Dual2<float>> wrsine(rsine_, Mask(mask_value));
        Masked<Dual2<float>> wrcosine(rcosine_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> theta = wtheta[lane];
            if (wrsine.mask()[lane]) {
                float rsine;
                float rcosine;
                impl_sincos(theta.val(), rsine, rcosine);
                wrsine[ActiveLane(lane)]   = Dual2<float>(rsine,
                                                        rcosine * theta.dx(),
                                                        rcosine * theta.dy());
                wrcosine[ActiveLane(lane)] = Dual2<float>(rcosine,
                                                          -rsine * theta.dx(),
                                                          -rsine * theta.dy());
            }
        }
    }
}


OSL_BATCHOP void
__OSL_OP3(sincos, Wv, Wv, Wv)(void* theta_, void* rsine_, void* rcosine_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wtheta(theta_);
        Wide<Vec3> wrsine(rsine_);
        Wide<Vec3> wrcosine(rcosine_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 theta = wtheta[lane];
            Vec3 rsine;
            Vec3 rcosine;
            impl_sincos(theta, rsine, rcosine);
            wrsine[lane]   = rsine;
            wrcosine[lane] = rcosine;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(sincos, Wv, Wv, Wv)(void* theta_, void* rsine_, void* rcosine_,
                                     unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wtheta(theta_);
        Masked<Vec3> wrsine(rsine_, Mask(mask_value));
        Masked<Vec3> wrcosine(rcosine_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 theta = wtheta[lane];
            if (wrsine.mask()[lane]) {
                Vec3 rsine;
                Vec3 rcosine;
                impl_sincos(theta, rsine, rcosine);
                wrsine[ActiveLane(lane)]   = rsine;
                wrcosine[ActiveLane(lane)] = rcosine;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(sincos, Wdv, Wdv, Wv)(void* theta_, void* rsine_, void* rcosine_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wtheta(theta_);
        Wide<Dual2<Vec3>> wrsine(rsine_);
        Wide<Vec3> wrcosine(rcosine_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> theta = wtheta[lane];
            Vec3 rsine;
            Vec3 rcosine;
            impl_sincos(theta.val(), rsine, rcosine);
            wrsine[lane] = Dual2<Vec3>(rsine, rcosine * theta.dx(),
                                       rcosine * theta.dy());
            ;
            wrcosine[lane] = rcosine;
        }
    }
}


OSL_BATCHOP void
__OSL_MASKED_OP3(sincos, Wdv, Wdv, Wv)(void* theta_, void* rsine_,
                                       void* rcosine_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wtheta(theta_);
        Masked<Dual2<Vec3>> wrsine(rsine_, Mask(mask_value));
        Masked<Vec3> wrcosine(rcosine_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> theta = wtheta[lane];
            if (wrsine.mask()[lane]) {
                Vec3 rsine;
                Vec3 rcosine;
                impl_sincos(theta.val(), rsine, rcosine);
                wrsine[ActiveLane(lane)]   = Dual2<Vec3>(rsine,
                                                       rcosine * theta.dx(),
                                                       rcosine * theta.dy());
                wrcosine[ActiveLane(lane)] = rcosine;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(sincos, Wdv, Wv, Wdv)(void* theta_, void* rsine_, void* rcosine_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wtheta(theta_);
        Wide<Vec3> wrsine(rsine_);
        Wide<Dual2<Vec3>> wrcosine(rcosine_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> theta = wtheta[lane];
            Vec3 rsine;
            Vec3 rcosine;
            impl_sincos(theta.val(), rsine, rcosine);
            wrsine[lane]   = rsine;
            wrcosine[lane] = Dual2<Vec3>(rcosine, -rsine * theta.dx(),
                                         -rsine * theta.dy());
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(sincos, Wdv, Wv, Wdv)(void* theta_, void* rsine_,
                                       void* rcosine_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wtheta(theta_);
        Masked<Vec3> wrsine(rsine_, Mask(mask_value));
        Masked<Dual2<Vec3>> wrcosine(rcosine_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> theta = wtheta[lane];
            if (wrsine.mask()[lane]) {
                Vec3 rsine;
                Vec3 rcosine;
                impl_sincos(theta.val(), rsine, rcosine);
                wrsine[ActiveLane(lane)]   = rsine;
                wrcosine[ActiveLane(lane)] = Dual2<Vec3>(rcosine,
                                                         -rsine * theta.dx(),
                                                         -rsine * theta.dy());
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(sincos, Wdv, Wdv, Wdv)(void* theta_, void* rsine_, void* rcosine_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wtheta(theta_);
        Wide<Dual2<Vec3>> wrsine(rsine_);
        Wide<Dual2<Vec3>> wrcosine(rcosine_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> theta = wtheta[lane];
            Vec3 rsine;
            Vec3 rcosine;
            impl_sincos(theta.val(), rsine, rcosine);
            wrsine[lane]   = Dual2<Vec3>(rsine, rcosine * theta.dx(),
                                       rcosine * theta.dy());
            wrcosine[lane] = Dual2<Vec3>(rcosine, -rsine * theta.dx(),
                                         -rsine * theta.dy());
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(sincos, Wdv, Wdv, Wdv)(void* theta_, void* rsine_,
                                        void* rcosine_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wtheta(theta_);
        Masked<Dual2<Vec3>> wrsine(rsine_, Mask(mask_value));
        Masked<Dual2<Vec3>> wrcosine(rcosine_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> theta = wtheta[lane];
            if (wrsine.mask()[lane]) {
                Vec3 rsine;
                Vec3 rcosine;
                impl_sincos(theta.val(), rsine, rcosine);
                wrsine[ActiveLane(lane)]   = Dual2<Vec3>(rsine,
                                                       rcosine * theta.dx(),
                                                       rcosine * theta.dy());
                wrcosine[ActiveLane(lane)] = Dual2<Vec3>(rcosine,
                                                         -rsine * theta.dx(),
                                                         -rsine * theta.dy());
            }
        }
    }
}


}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#include "undef_opname_macros.h"
