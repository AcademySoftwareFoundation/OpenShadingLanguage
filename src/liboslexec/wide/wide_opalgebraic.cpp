// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader implementation of Algebraic operations
/// NOTE: Execute from the library (vs. LLVM-IR) to take advantage
/// of compiler's small vector math library.
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

OSL_NAMESPACE_BEGIN
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

#include "define_opname_macros.h"

#define __OSL_XMACRO_ARGS (sqrt, OIIO::safe_sqrt, sqrt)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS (inversesqrt, OIIO::safe_inversesqrt, inversesqrt)
#include "wide_opunary_per_component_xmacro.h"

// emitted directly by llvm_gen_wide.cpp
//MAKE_BINARY_FI_OP(safe_div, sfm::safe_div, sfm::safe_div)

#define __OSL_XMACRO_ARGS (floor, floorf)
#include "wide_opunary_per_component_float_or_vector_xmacro.h"

#define __OSL_XMACRO_ARGS (ceil, ceilf)
#include "wide_opunary_per_component_float_or_vector_xmacro.h"

#define __OSL_XMACRO_ARGS (trunc, truncf)
#include "wide_opunary_per_component_float_or_vector_xmacro.h"

#define __OSL_XMACRO_ARGS (round, roundf)
#include "wide_opunary_per_component_float_or_vector_xmacro.h"


static OSL_FORCEINLINE float
impl_sign(float x)
{
    // Avoid nested conditional logic as per language
    // rules the right operand may only be evaluated
    // if the 1st conditional is false.
    // Thus complex control flow vs. just 2 compares
    // and masked assignments
    //return x < 0.0f ? -1.0f : (x==0.0f ? 0.0f : 1.0f);
    float sign = 0.0f;
    if (x < 0.0f)
        sign = -1.0f;
    if (x > 0.0f)
        sign = 1.0f;
    return sign;
}
#define __OSL_XMACRO_ARGS (sign, impl_sign)
#include "wide_opunary_per_component_float_or_vector_xmacro.h"

// TODO: move to dual.h
OSL_FORCEINLINE OSL_HOSTDEVICE Dual2<float>
abs(const Dual2<float>& x)
{
    return x.val() >= 0 ? x : sfm::negate(x);
}

#define __OSL_XMACRO_ARGS (abs, std::abs, abs)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS (fabs, std::abs, abs)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS (abs, std::abs)
#include "wide_opunary_int_xmacro.h"

#define __OSL_XMACRO_ARGS (fabs, std::abs)
#include "wide_opunary_int_xmacro.h"

#define __OSL_XMACRO_ARGS (fmod, OIIO::safe_fmod, safe_fmod)
#include "wide_opbinary_per_component_xmacro.h"
#define __OSL_XMACRO_ARGS (fmod, OIIO::safe_fmod, safe_fmod)
#include "wide_opbinary_per_component_mixed_vector_float_xmacro.h"


static OSL_FORCEINLINE float
impl_step(float edge, float x)
{
    // Avoid ternary, as only constants are in the
    // conditional branches, this may be unnecessary.
    // return x < edge ? 0.0f : 1.0f;
    float result = 0.0f;
    if (x >= edge) {
        result = 1.0f;
    }
    return result;
}


// TODO: consider moving step to batched_llvm_gen.cpp
#define __OSL_XMACRO_ARGS (step, impl_step)
#include "wide_opbinary_per_component_float_or_vector_xmacro.h"



inline Vec3
calculatenormal(const Dual2<Vec3>& tmpP, bool flipHandedness)
{
    // Encourage compiles to test for coherency and skip branch when false
    if (OSL_UNLIKELY(flipHandedness))
        return tmpP.dy().cross(tmpP.dx());
    else
        return tmpP.dx().cross(tmpP.dy());
}



OSL_PRAGMA_WARNING_PUSH
OSL_NONINTEL_CLANG_PRAGMA(GCC diagnostic ignored "-Wpass-failed")

OSL_BATCHOP void
__OSL_OP2(length, Wf, Wv)(void* r_, void* V_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wV(V_);
        Wide<float> wr(r_);

        OSL_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 V   = wV[lane];
            float r  = sfm::length(V);
            wr[lane] = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP2(length, Wf, Wv)(void* r_, void* V_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wV(V_);
        Masked<float> wr(r_, Mask(mask_value));

        OSL_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 V = wV[lane];
            if (wr.mask()[lane]) {
                float r              = sfm::length(V);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}

OSL_PRAGMA_WARNING_POP



OSL_BATCHOP void
__OSL_OP2(length, Wdf, Wdv)(void* r_, void* V_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wV(V_);
        Wide<Dual2<float>> wr(r_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> V  = wV[lane];
            Dual2<float> r = length(V);
            wr[lane]       = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP2(length, Wdf, Wdv)(void* r_, void* V_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wV(V_);
        Masked<Dual2<float>> wr(r_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> V = wV[lane];
            if (wr.mask()[lane]) {
                Dual2<float> r       = length(V);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_PRAGMA_WARNING_PUSH
OSL_NONINTEL_CLANG_PRAGMA(GCC diagnostic ignored "-Wpass-failed")

OSL_BATCHOP void
__OSL_OP2(area, Wf, Wdv)(void* r_, void* DP_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wDP(DP_);

        Wide<float> wr(r_);

        OSL_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> DP = wDP[lane];

            Vec3 N = calculatenormal(DP, false);
            //float r = N.length();
            float r  = sfm::length(N);
            wr[lane] = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP2(area, Wf, Wdv)(void* r_, void* DP_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wDP(DP_);

        Masked<float> wr(r_, Mask(mask_value));

        OSL_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> DP = wDP[lane];
            if (wr.mask()[lane]) {
                Vec3 N = calculatenormal(DP, false);
                //float r = N.length();
                float r              = sfm::length(N);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}

OSL_PRAGMA_WARNING_POP



OSL_BATCHOP void
__OSL_OP3(distance, Wf, Wv, Wv)(void* r_, void* a_, void* b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wA(a_);
        Wide<const Vec3> wB(b_);
        Wide<float> wr(r_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wA[lane];
            Vec3 b = wB[lane];

            // TODO: couldn't we just (b-a).length()?
            float x  = a.x - b.x;
            float y  = a.y - b.y;
            float z  = a.z - b.z;
            float r  = sqrtf(x * x + y * y + z * z);
            wr[lane] = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(distance, Wf, Wv, Wv)(void* r_, void* a_, void* b_,
                                       unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wA(a_);
        Wide<const Vec3> wB(b_);
        Masked<float> wr(r_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wA[lane];
            Vec3 b = wB[lane];
            if (wr.mask()[lane]) {
                // TODO: couldn't we just (b-a).length()?
                float x              = a.x - b.x;
                float y              = a.y - b.y;
                float z              = a.z - b.z;
                float r              = sqrtf(x * x + y * y + z * z);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(distance, Wdf, Wdv, Wv)(void* r_, void* a_, void* b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wA(a_);
        Wide<const Vec3> wB(b_);
        Wide<Dual2<float>> wr(r_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wA[lane];
            Vec3 b        = wB[lane];

            Dual2<float> r = distance(a, b);
            wr[lane]       = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(distance, Wdf, Wdv, Wv)(void* r_, void* a_, void* b_,
                                         unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wA(a_);
        Wide<const Vec3> wB(b_);
        Masked<Dual2<float>> wr(r_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wA[lane];
            Vec3 b        = wB[lane];
            if (wr.mask()[lane]) {
                Dual2<float> r       = distance(a, b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(distance, Wdf, Wv, Wdv)(void* r_, void* a_, void* b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wA(a_);
        Wide<const Dual2<Vec3>> wB(b_);
        Wide<Dual2<float>> wr(r_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a        = wA[lane];
            Dual2<Vec3> b = wB[lane];

            Dual2<float> r = distance(a, b);
            wr[lane]       = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(distance, Wdf, Wv, Wdv)(void* r_, void* a_, void* b_,
                                         unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wA(a_);
        Wide<const Dual2<Vec3>> wB(b_);
        Masked<Dual2<float>> wr(r_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a        = wA[lane];
            Dual2<Vec3> b = wB[lane];
            if (wr.mask()[lane]) {
                Dual2<float> r       = distance(a, b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(distance, Wdf, Wdv, Wdv)(void* r_, void* a_, void* b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wA(a_);
        Wide<const Dual2<Vec3>> wB(b_);
        Wide<Dual2<float>> wr(r_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wA[lane];
            Dual2<Vec3> b = wB[lane];

            Dual2<float> r = distance(a, b);
            wr[lane]       = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(distance, Wdf, Wdv, Wdv)(void* r_, void* a_, void* b_,
                                          unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wA(a_);
        Wide<const Dual2<Vec3>> wB(b_);
        Masked<Dual2<float>> wr(r_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wA[lane];
            Dual2<Vec3> b = wB[lane];
            if (wr.mask()[lane]) {
                Dual2<float> r       = distance(a, b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_PRAGMA_WARNING_PUSH
OSL_NONINTEL_CLANG_PRAGMA(GCC diagnostic ignored "-Wpass-failed")

OSL_BATCHOP void
__OSL_OP2(normalize, Wv, Wv)(void* r_, void* V_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wV(V_);
        Wide<Vec3> wr(r_);

        OSL_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 V   = wV[lane];
            Vec3 N   = sfm::normalize(V);
            wr[lane] = N;
        }
    }
}


OSL_BATCHOP void
__OSL_MASKED_OP2(normalize, Wv, Wv)(void* r_, void* V_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wV(V_);
        Masked<Vec3> wr(r_, Mask(mask_value));

        OSL_OMP_SIMD_LOOP(simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 V = wV[lane];
            if (wr.mask()[lane]) {
                Vec3 N               = sfm::normalize(V);
                wr[ActiveLane(lane)] = N;
            }
        }
    }
}

OSL_PRAGMA_WARNING_POP


OSL_BATCHOP void
__OSL_OP2(normalize, Wdv, Wdv)(void* r_, void* V_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wV(V_);
        Wide<Dual2<Vec3>> wr(r_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> V = wV[lane];
            Dual2<Vec3> N = sfm::normalize(V);
            wr[lane]      = N;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP2(normalize, Wdv, Wdv)(void* r_, void* V_,
                                      unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wV(V_);
        Masked<Dual2<Vec3>> wr(r_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> V = wV[lane];
            if (wr.mask()[lane]) {
                Dual2<Vec3> N        = sfm::normalize(V);
                wr[ActiveLane(lane)] = N;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(cross, Wv, Wv, Wv)(void* result_, void* a_, void* b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wA(a_);
        Wide<const Vec3> wB(b_);
        Wide<Vec3> wr(result_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wA[lane];
            Vec3 b = wB[lane];

            Vec3 r   = a.cross(b);
            wr[lane] = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(cross, Wv, Wv, Wv)(void* result_, void* a_, void* b_,
                                    unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wA(a_);
        Wide<const Vec3> wB(b_);
        Masked<Vec3> wr(result_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wA[lane];
            Vec3 b = wB[lane];
            if (wr.mask()[lane]) {
                Vec3 r               = a.cross(b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(cross, Wdv, Wdv, Wv)(void* result_, void* a_, void* b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wA(a_);
        Wide<const Vec3> wB(b_);
        Wide<Dual2<Vec3>> wr(result_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wA[lane];
            Vec3 b        = wB[lane];

            Dual2<Vec3> dv_b(b);
            Dual2<Vec3> r = cross(a, dv_b);
            wr[lane]      = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(cross, Wdv, Wdv, Wv)(void* result_, void* a_, void* b_,
                                      unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wA(a_);
        Wide<const Vec3> wB(b_);
        Masked<Dual2<Vec3>> wr(result_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wA[lane];
            Vec3 b        = wB[lane];
            Dual2<Vec3> dv_b(b);

            if (wr.mask()[lane]) {
                Dual2<Vec3> r        = cross(a, dv_b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(cross, Wdv, Wv, Wdv)(void* result_, void* a_, void* b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wA(a_);
        Wide<const Dual2<Vec3>> wB(b_);
        Wide<Dual2<Vec3>> wr(result_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wA[lane];
            Dual2<Vec3> dv_a(a);
            Dual2<Vec3> b = wB[lane];

            Dual2<Vec3> r = cross(dv_a, b);
            wr[lane]      = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(cross, Wdv, Wv, Wdv)(void* result_, void* a_, void* b_,
                                      unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wA(a_);
        Wide<const Dual2<Vec3>> wB(b_);
        Masked<Dual2<Vec3>> wr(result_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wA[lane];
            Dual2<Vec3> dv_a(a);
            Dual2<Vec3> b = wB[lane];

            if (wr.mask()[lane]) {
                Dual2<Vec3> r        = cross(dv_a, b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(cross, Wdv, Wdv, Wdv)(void* result_, void* a_, void* b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wA(a_);
        Wide<const Dual2<Vec3>> wB(b_);
        Wide<Dual2<Vec3>> wr(result_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wA[lane];
            Dual2<Vec3> b = wB[lane];

            Dual2<Vec3> r = cross(a, b);
            wr[lane]      = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(cross, Wdv, Wdv, Wdv)(void* result_, void* a_, void* b_,
                                       unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wA(a_);
        Wide<const Dual2<Vec3>> wB(b_);
        Masked<Dual2<Vec3>> wr(result_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wA[lane];
            Dual2<Vec3> b = wB[lane];

            if (wr.mask()[lane]) {
                Dual2<Vec3> r        = cross(a, b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}


OSL_BATCHOP void
__OSL_OP3(dot, Wf, Wv, Wv)(void* result_, void* a_, void* b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wA(a_);
        Wide<const Vec3> wB(b_);
        Wide<float> wr(result_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wA[lane];
            Vec3 b = wB[lane];

            float r  = a.dot(b);
            wr[lane] = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(dot, Wf, Wv, Wv)(void* result_, void* a_, void* b_,
                                  unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wA(a_);
        Wide<const Vec3> wB(b_);
        Masked<float> wr(result_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wA[lane];
            Vec3 b = wB[lane];

            if (wr.mask()[lane]) {
                float r              = a.dot(b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(dot, Wdf, Wdv, Wv)(void* result_, void* a_, void* b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wA(a_);
        Wide<const Vec3> wB(b_);
        Wide<Dual2<float>> wr(result_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wA[lane];
            Vec3 b        = wB[lane];

            Dual2<float> r = dot(a, b);
            wr[lane]       = r;
        }
    }
}


OSL_BATCHOP void
__OSL_MASKED_OP3(dot, Wdf, Wdv, Wv)(void* result_, void* a_, void* b_,
                                    unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wA(a_);
        Wide<const Vec3> wB(b_);
        Masked<Dual2<float>> wr(result_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wA[lane];
            Vec3 b        = wB[lane];

            if (wr.mask()[lane]) {
                Dual2<float> r       = dot(a, b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(dot, Wdf, Wv, Wdv)(void* result_, void* a_, void* b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wA(a_);
        Wide<const Dual2<Vec3>> wB(b_);
        Wide<Dual2<float>> wr(result_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a        = wA[lane];
            Dual2<Vec3> b = wB[lane];

            Dual2<float> r = dot(a, b);
            wr[lane]       = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(dot, Wdf, Wv, Wdv)(void* result_, void* a_, void* b_,
                                    unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wA(a_);
        Wide<const Dual2<Vec3>> wB(b_);
        Masked<Dual2<float>> wr(result_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a        = wA[lane];
            Dual2<Vec3> b = wB[lane];

            if (wr.mask()[lane]) {
                Dual2<float> r       = dot(a, b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP3(dot, Wdf, Wdv, Wdv)(void* result_, void* a_, void* b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wA(a_);
        Wide<const Dual2<Vec3>> wB(b_);
        Wide<Dual2<float>> wr(result_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wA[lane];
            Dual2<Vec3> b = wB[lane];

            Dual2<float> r = dot(a, b);
            wr[lane]       = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(dot, Wdf, Wdv, Wdv)(void* result_, void* a_, void* b_,
                                     unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wA(a_);
        Wide<const Dual2<Vec3>> wB(b_);
        Masked<Dual2<float>> wr(result_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wA[lane];
            Dual2<Vec3> b = wB[lane];
            if (wr.mask()[lane]) {
                Dual2<float> r       = dot(a, b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}


inline float
filter_width(float dx, float dy)
{
    return sqrtf(dx * dx + dy * dy);
}


OSL_BATCHOP void
__OSL_OP2(filterwidth, Wf, Wdf)(void* result_, void* x_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> wX(x_);

        Wide<float> wr(result_);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> x = wX[lane];
            wr[lane]       = filter_width(x.dx(), x.dy());
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP2(filterwidth, Wf, Wdf)(void* result_, void* x_,
                                       unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> wX(x_);

        Masked<float> wr(result_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> x = wX[lane];
            if (wr.mask()[lane]) {
                wr[ActiveLane(lane)] = filter_width(x.dx(), x.dy());
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP2(filterwidth, Wv, Wdv)(void* out, void* x_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wX(x_);

        Wide<Vec3> wr(out);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> x = wX[lane];
            Vec3 r;
            r.x = filter_width(x.dx().x, x.dy().x);
            r.y = filter_width(x.dx().y, x.dy().y);
            r.z = filter_width(x.dx().z, x.dy().z);

            wr[lane] = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP2(filterwidth, Wv, Wdv)(void* out, void* x_,
                                       unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wX(x_);

        Masked<Vec3> wr(out, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> x = wX[lane];
            if (wr.mask()[lane]) {
                Vec3 r;
                r.x = filter_width(x.dx().x, x.dy().x);
                r.y = filter_width(x.dx().y, x.dy().y);
                r.z = filter_width(x.dx().z, x.dy().z);

                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP(calculatenormal)(void* out, void* bsg_, void* P_)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wP(P_);
        Wide<const int> wFlipHandedness(bsg->varying.flipHandedness);
        Wide<Vec3> wr(out);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> P = wP[lane];
            Vec3 N        = calculatenormal(P, wFlipHandedness[lane]);
            wr[lane]      = N;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP(calculatenormal)(void* out, void* bsg_, void* P_,
                                 unsigned int mask_value)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wP(P_);
        Wide<const int> wFlipHandedness(bsg->varying.flipHandedness);
        Masked<Vec3> wr(out, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> P       = wP[lane];
            int flip_handedness = wFlipHandedness[lane];
            //std::cout << "P=" << P.val() << "," << P.dx() << "," << P.dy() << std::endl;

            if (wr.mask()[lane]) {
                Vec3 N               = calculatenormal(P, flip_handedness);
                wr[ActiveLane(lane)] = N;
            }
        }
    }
}


OSL_BATCHOP void
__OSL_OP4(smoothstep, Wf, Wf, Wf, Wf)(void* r_, void* e0_, void* e1_, void* x_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> we0(e0_);
        Wide<const float> we1(e1_);
        Wide<const float> wx(x_);
        Wide<float> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float e0 = we0[lane];
            float e1 = we1[lane];
            float x  = wx[lane];
            float r  = smoothstep(e0, e1, x);
            wr[lane] = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP4(smoothstep, Wf, Wf, Wf, Wf)(void* r_, void* e0_, void* e1_,
                                             void* x_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> we0(e0_);
        Wide<const float> we1(e1_);
        Wide<const float> wx(x_);
        Masked<float> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float e0 = we0[lane];
            float e1 = we1[lane];
            float x  = wx[lane];
            if (wr.mask()[lane]) {
                float r              = smoothstep(e0, e1, x);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP4(smoothstep, Wdf, Wdf, Wdf, Wdf)(void* r_, void* e0_, void* e1_,
                                          void* x_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> we0(e0_);
        Wide<const Dual2<float>> we1(e1_);
        Wide<const Dual2<float>> wx(x_);
        Wide<Dual2<float>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> e0 = we0[lane];
            Dual2<float> e1 = we1[lane];
            Dual2<float> x  = wx[lane];
            Dual2<float> r  = smoothstep(e0, e1, x);
            wr[lane]        = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP4(smoothstep, Wdf, Wdf, Wdf, Wdf)(void* r_, void* e0_, void* e1_,
                                                 void* x_,
                                                 unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> we0(e0_);
        Wide<const Dual2<float>> we1(e1_);
        Wide<const Dual2<float>> wx(x_);
        Masked<Dual2<float>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> e0 = we0[lane];
            Dual2<float> e1 = we1[lane];
            Dual2<float> x  = wx[lane];
            if (wr.mask()[lane]) {
                Dual2<float> r       = smoothstep(e0, e1, x);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP4(smoothstep, Wdf, Wf, Wdf, Wdf)(void* r_, void* e0_, void* e1_,
                                         void* x_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> we0(e0_);
        Wide<const Dual2<float>> we1(e1_);
        Wide<const Dual2<float>> wx(x_);
        Wide<Dual2<float>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float e0        = we0[lane];
            Dual2<float> e1 = we1[lane];
            Dual2<float> x  = wx[lane];
            Dual2<float> r  = smoothstep(Dual2<float>(e0), e1, x);
            wr[lane]        = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP4(smoothstep, Wdf, Wf, Wdf, Wdf)(void* r_, void* e0_, void* e1_,
                                                void* x_,
                                                unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> we0(e0_);
        Wide<const Dual2<float>> we1(e1_);
        Wide<const Dual2<float>> wx(x_);
        Masked<Dual2<float>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float e0        = we0[lane];
            Dual2<float> e1 = we1[lane];
            Dual2<float> x  = wx[lane];
            if (wr.mask()[lane]) {
                Dual2<float> r       = smoothstep(Dual2<float>(e0), e1, x);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP4(smoothstep, Wdf, Wdf, Wf, Wdf)(void* r_, void* e0_, void* e1_,
                                         void* x_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> we0(e0_);
        Wide<const float> we1(e1_);
        Wide<const Dual2<float>> wx(x_);
        Wide<Dual2<float>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> e0 = we0[lane];
            float e1        = we1[lane];
            Dual2<float> x  = wx[lane];
            Dual2<float> r  = smoothstep(e0, Dual2<float>(e1), x);
            wr[lane]        = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP4(smoothstep, Wdf, Wdf, Wf, Wdf)(void* r_, void* e0_, void* e1_,
                                                void* x_,
                                                unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> we0(e0_);
        Wide<const float> we1(e1_);
        Wide<const Dual2<float>> wx(x_);
        Masked<Dual2<float>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> e0 = we0[lane];
            float e1        = we1[lane];
            Dual2<float> x  = wx[lane];
            if (wr.mask()[lane]) {
                Dual2<float> r       = smoothstep(e0, Dual2<float>(e1), x);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}


OSL_BATCHOP void
__OSL_OP4(smoothstep, Wdf, Wdf, Wdf, Wf)(void* r_, void* e0_, void* e1_,
                                         void* x_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> we0(e0_);
        Wide<const Dual2<float>> we1(e1_);
        Wide<const float> wx(x_);
        Wide<Dual2<float>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> e0 = we0[lane];
            Dual2<float> e1 = we1[lane];
            float x         = wx[lane];
            Dual2<float> r  = smoothstep(e0, e1, Dual2<float>(x));
            wr[lane]        = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP4(smoothstep, Wdf, Wdf, Wdf, Wf)(void* r_, void* e0_, void* e1_,
                                                void* x_,
                                                unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> we0(e0_);
        Wide<const Dual2<float>> we1(e1_);
        Wide<const float> wx(x_);
        Masked<Dual2<float>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> e0 = we0[lane];
            Dual2<float> e1 = we1[lane];
            float x         = wx[lane];
            if (wr.mask()[lane]) {
                Dual2<float> r       = smoothstep(e0, e1, Dual2<float>(x));
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP4(smoothstep, Wdf, Wf, Wf, Wdf)(void* r_, void* e0_, void* e1_,
                                        void* x_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> we0(e0_);
        Wide<const float> we1(e1_);
        Wide<const Dual2<float>> wx(x_);
        Wide<Dual2<float>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float e0       = we0[lane];
            float e1       = we1[lane];
            Dual2<float> x = wx[lane];
            Dual2<float> r = smoothstep(Dual2<float>(e0), Dual2<float>(e1), x);
            wr[lane]       = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP4(smoothstep, Wdf, Wf, Wf, Wdf)(void* r_, void* e0_, void* e1_,
                                               void* x_,
                                               unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> we0(e0_);
        Wide<const float> we1(e1_);
        Wide<const Dual2<float>> wx(x_);
        Masked<Dual2<float>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float e0       = we0[lane];
            float e1       = we1[lane];
            Dual2<float> x = wx[lane];
            if (wr.mask()[lane]) {
                Dual2<float> r = smoothstep(Dual2<float>(e0), Dual2<float>(e1),
                                            x);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP4(smoothstep, Wdf, Wdf, Wf, Wf)(void* r_, void* e0_, void* e1_,
                                        void* x_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> we0(e0_);
        Wide<const float> we1(e1_);
        Wide<const float> wx(x_);
        Wide<Dual2<float>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> e0 = we0[lane];
            float e1        = we1[lane];
            float x         = wx[lane];
            Dual2<float> r  = smoothstep(e0, Dual2<float>(e1), Dual2<float>(x));
            wr[lane]        = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP4(smoothstep, Wdf, Wdf, Wf, Wf)(void* r_, void* e0_, void* e1_,
                                               void* x_,
                                               unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> we0(e0_);
        Wide<const float> we1(e1_);
        Wide<const float> wx(x_);
        Masked<Dual2<float>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> e0 = we0[lane];
            float e1        = we1[lane];
            float x         = wx[lane];
            if (wr.mask()[lane]) {
                Dual2<float> r       = smoothstep(e0, Dual2<float>(e1),
                                                  Dual2<float>(x));
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}


OSL_BATCHOP void
__OSL_OP4(smoothstep, Wdf, Wf, Wdf, Wf)(void* r_, void* e0_, void* e1_,
                                        void* x_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> we0(e0_);
        Wide<const Dual2<float>> we1(e1_);
        Wide<const float> wx(x_);
        Wide<Dual2<float>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float e0        = we0[lane];
            Dual2<float> e1 = we1[lane];
            float x         = wx[lane];
            Dual2<float> r  = smoothstep(Dual2<float>(e0), e1, Dual2<float>(x));
            wr[lane]        = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP4(smoothstep, Wdf, Wf, Wdf, Wf)(void* r_, void* e0_, void* e1_,
                                               void* x_,
                                               unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> we0(e0_);
        Wide<const Dual2<float>> we1(e1_);
        Wide<const float> wx(x_);
        Masked<Dual2<float>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float e0        = we0[lane];
            Dual2<float> e1 = we1[lane];
            float x         = wx[lane];
            if (wr.mask()[lane]) {
                Dual2<float> r       = smoothstep(Dual2<float>(e0), e1,
                                                  Dual2<float>(x));
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}

}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_END

#include "undef_opname_macros.h"
