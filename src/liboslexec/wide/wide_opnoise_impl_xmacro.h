// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <limits>

#include <OSL/oslconfig.h>

#include "oslexec_pvt.h"

#include "batched_cg_policy.h"

#include <OSL/Imathx/Imathx.h>
#include <OSL/dual_vec.h>
#include <OSL/oslnoise.h>

#include <OpenImageIO/fmath.h>

using namespace OSL;

OSL_NAMESPACE_BEGIN
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)


#ifdef __OSL_XMACRO_ARGS

#    define __OSL_XMACRO_OPNAME \
        __OSL_EXPAND(__OSL_XMACRO_ARG1 __OSL_XMACRO_ARGS)
#    define __OSL_XMACRO_SFM_IMPLNAME \
        __OSL_EXPAND(__OSL_XMACRO_ARG2 __OSL_XMACRO_ARGS)
#    define __OSL_XMACRO_IMPLNAME \
        __OSL_EXPAND(__OSL_XMACRO_ARG3 __OSL_XMACRO_ARGS)

#endif

#ifndef __OSL_XMACRO_OPNAME
#    error must define __OSL_XMACRO_OPNAME to name of noise operation before including this header
#endif

#ifndef __OSL_XMACRO_SFM_IMPLNAME
#    error must define __OSL_XMACRO_SFM_IMPLNAME to name of SIMD friendly noise implementation before including this header
#endif

#ifndef __OSL_XMACRO_IMPLNAME
#    error must define __OSL_XMACRO_IMPLNAME to name of noise implementation before including this header
#endif

#include "define_opname_macros.h"
#define __OSL_NOISE_OP2(A, B)    __OSL_MASKED_OP2(__OSL_XMACRO_OPNAME, A, B)
#define __OSL_NOISE_OP3(A, B, C) __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, A, B, C)


#ifndef __OSL_XMACRO_VEC3_RESULTS_ONLY

OSL_BATCHOP void
__OSL_NOISE_OP2(Wf, Wf)(char* r_ptr, char* x_ptr, unsigned int mask_value)
{
    Masked<float> wresult(r_ptr, Mask(mask_value));
    Wide<const float> wx(x_ptr);

    typedef BatchedCGPolicy<Param::WF, Param::WF> Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const float x = wx[lane];
                float result;
                blockvec_impl(result, x);
                wresult[lane] = result;
            });
        return;
    }

    __OSL_XMACRO_SFM_IMPLNAME impl;
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const float x = wx[lane];
            if (wresult.mask()[lane]) {
                float result;
                impl(result, x);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_NOISE_OP3(Wf, Wf, Wf)(char* r_ptr, char* x_ptr, char* y_ptr,
                            unsigned int mask_value)
{
    Masked<float> wresult(r_ptr, Mask(mask_value));
    Wide<const float> wx(x_ptr);
    Wide<const float> wy(y_ptr);

    typedef BatchedCGPolicy<Param::WF, Param::WF, Param::WF> Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const float x = wx[lane];
                const float y = wy[lane];
                float result;
                blockvec_impl(result, x, y);
                wresult[lane] = result;
            });
        return;
    }

    __OSL_XMACRO_SFM_IMPLNAME impl;
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const float x = wx[lane];
            const float y = wy[lane];
            if (wresult.mask()[lane]) {
                float result;
                impl(result, x, y);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_NOISE_OP2(Wf, Wv)(char* r_ptr, char* p_ptr, unsigned int mask_value)
{
    Masked<float> wresult(r_ptr, Mask(mask_value));
    Wide<const Vec3> wp(p_ptr);

    typedef BatchedCGPolicy<Param::WF, Param::WV> Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const Vec3 p = wp[lane];
                float result;
                blockvec_impl(result, p);
                wresult[lane] = result;
            });
        return;
    }
    __OSL_XMACRO_SFM_IMPLNAME impl;
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Vec3 p = wp[lane];
            if (wresult.mask()[lane]) {
                float result;
                impl(result, p);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_NOISE_OP3(Wf, Wv, Wf)(char* r_ptr, char* p_ptr, char* t_ptr,
                            unsigned int mask_value)
{
    Masked<float> wresult(r_ptr, Mask(mask_value));
    Wide<const Vec3> wp(p_ptr);
    Wide<const float> wt(t_ptr);
    typedef BatchedCGPolicy<Param::WF, Param::WV, Param::WF> Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const Vec3 p  = wp[lane];
                const float t = wt[lane];
                float result;
                blockvec_impl(result, p, t);
                wresult[lane] = result;
            });
        return;
    }
    __OSL_XMACRO_SFM_IMPLNAME impl;
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Vec3 p  = wp[lane];
            const float t = wt[lane];
            if (wresult.mask()[lane]) {
                float result;
                impl(result, p, t);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}

#endif

#ifndef __OSL_XMACRO_FLOAT_RESULTS_ONLY

OSL_BATCHOP void
__OSL_NOISE_OP2(Wv, Wv)(char* r_ptr, char* p_ptr, unsigned int mask_value)
{
    Masked<Vec3> wresult(r_ptr, Mask(mask_value));
    Wide<const Vec3> wp(p_ptr);
    typedef BatchedCGPolicy<Param::WV, Param::WV> Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const Vec3 p = wp[lane];
                Vec3 result;
                blockvec_impl(result, p);
                wresult[lane] = result;
            });
        return;
    }
    __OSL_XMACRO_SFM_IMPLNAME impl;
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Vec3 p = wp[lane];
            if (wresult.mask()[lane]) {
                Vec3 result;
                impl(result, p);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_NOISE_OP2(Wv, Wf)(char* r_ptr, char* x_ptr, unsigned int mask_value)
{
    Masked<Vec3> wresult(r_ptr, Mask(mask_value));
    Wide<const float> wx(x_ptr);
    typedef BatchedCGPolicy<Param::WV, Param::WF> Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const float x = wx[lane];
                Vec3 result;
                blockvec_impl(result, x);
                wresult[lane] = result;
            });
        return;
    }
    __OSL_XMACRO_SFM_IMPLNAME impl;
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const float x = wx[lane];
            if (wresult.mask()[lane]) {
                Vec3 result;
                impl(result, x);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_NOISE_OP3(Wv, Wv, Wf)(char* r_ptr, char* p_ptr, char* t_ptr,
                            unsigned int mask_value)
{
    Masked<Vec3> wresult(r_ptr, Mask(mask_value));
    Wide<const Vec3> wp(p_ptr);
    Wide<const float> wt(t_ptr);
    typedef BatchedCGPolicy<Param::WV, Param::WV, Param::WF> Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const Vec3 p  = wp[lane];
                const float t = wt[lane];
                Vec3 result;
                blockvec_impl(result, p, t);
                wresult[lane] = result;
            });
        return;
    }
    __OSL_XMACRO_SFM_IMPLNAME impl;
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Vec3 p  = wp[lane];
            const float t = wt[lane];
            if (wresult.mask()[lane]) {
                Vec3 result;
                impl(result, p, t);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_NOISE_OP3(Wv, Wf, Wf)(char* r_ptr, char* x_ptr, char* y_ptr,
                            unsigned int mask_value)
{
    Masked<Vec3> wresult(r_ptr, Mask(mask_value));
    Wide<const float> wx(x_ptr);
    Wide<const float> wy(y_ptr);
    typedef BatchedCGPolicy<Param::WV, Param::WF, Param::WF> Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const float x = wx[lane];
                const float y = wy[lane];
                Vec3 result;
                blockvec_impl(result, x, y);
                wresult[lane] = result;
            });
        return;
    }
    __OSL_XMACRO_SFM_IMPLNAME impl;
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const float x = wx[lane];
            const float y = wy[lane];
            if (wresult.mask()[lane]) {
                Vec3 result;
                impl(result, x, y);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}

#endif



}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_END


#undef __OSL_NOISE_OP2
#undef __OSL_NOISE_OP3

#include "undef_opname_macros.h"
#undef __OSL_XMACRO_FLOAT_RESULTS_ONLY
#undef __OSL_XMACRO_VEC3_RESULTS_ONLY
#undef __OSL_XMACRO_OPNAME
#undef __OSL_XMACRO_SFM_IMPLNAME
#undef __OSL_XMACRO_IMPLNAME
#undef __OSL_XMACRO_SIMDIMPLNAME
#undef __OSL_XMACRO_ARGS
