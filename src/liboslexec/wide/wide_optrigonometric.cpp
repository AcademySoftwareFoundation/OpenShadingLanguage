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
// TODO: compare performance of fast_math vs. math library
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
// Avoid regression in icx 2022.2.0 (should already be fixed in next release)
// TODO: incorporate SSA code changes into OIIO and remove the workaround
#    if !defined(__INTEL_LLVM_COMPILER) || (__INTEL_LLVM_COMPILER != 20220200)
    OIIO::fast_sincos(theta, &rsine, &rcosine);
#    else
    // Adopt Single Statement Assignment (SSA) coding style
    // to create less work for optimizers and code generation
    //
    // Implementation is adapted from https://github.com/OpenImageIO/oiio
    // under the same BSD-3-Clause license
    const float x = theta;
    using OIIO::clamp;
    using OIIO::fast_rint;
    using OIIO::madd;
    const int q    = fast_rint(x * float(M_1_PI));
    float qf       = float(q);
    const float x2 = madd(qf, -0.78515625f * 4, x);
    const float x3 = madd(qf, -0.00024187564849853515625f * 4, x2);
    const float x4 = madd(qf, -3.7747668102383613586e-08f * 4, x3);
    const float x5 = madd(qf, -1.2816720341285448015e-12f * 4, x4);
    const float x6 = float(M_PI_2) - (float(M_PI_2) - x5);  // crush denormals
    const float s  = x6 * x6;
    // NOTE: same exact polynomials as fast_sin and fast_cos above
    const bool q_is_odd = (q & 1) != 0;
    float x7            = x6;
    if (q_is_odd)
        x7 = -x6;
    const float su0 = 2.6083159809786593541503e-06f;
    const float su2 = madd(su0, s, -0.0001981069071916863322258f);
    const float su3 = madd(su2, s, +0.00833307858556509017944336f);
    const float su4 = madd(su3, s, -0.166666597127914428710938f);
    const float su5 = madd(s, su4 * x7, x7);
    const float cu0 = -2.71811842367242206819355e-07f;
    const float cu2 = madd(cu0, s, +2.47990446951007470488548e-05f);
    const float cu3 = madd(cu2, s, -0.00138888787478208541870117f);
    const float cu4 = madd(cu3, s, +0.0416666641831398010253906f);
    const float cu5 = madd(cu4, s, -0.5f);
    const float cu6 = madd(cu5, s, +1.0f);
    float cu        = cu6;
    if (q_is_odd)
        cu = -cu6;
    rsine   = clamp(su5, -1.0f, 1.0f);
    rcosine = clamp(cu, -1.0f, 1.0f);
#    endif
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
