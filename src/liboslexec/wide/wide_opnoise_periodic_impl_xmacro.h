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

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

/***********************************************************************
 * batched periodic routines callable by the LLVM-generated code.
 */

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
#    error must define __OSL_XMACRO_SFM_IMPLNAME to name of noise implementation before including this header
#endif

#ifndef __OSL_XMACRO_IMPLNAME
#    error must define __OSL_XMACRO_IMPLNAME to name of noise implementation before including this header
#endif


#include "define_opname_macros.h"

#define __OSL_PNOISE_OP3(A, B, C) __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, A, B, C)
#define __OSL_PNOISE_OP5(A, B, C, D, E) \
    __OSL_MASKED_OP5(__OSL_XMACRO_OPNAME, A, B, C, D, E)

#ifndef __OSL_XMACRO_VEC3_RESULTS_ONLY

OSL_BATCHOP void __OSL_PNOISE_OP3(Wf, Wf, Wf)(char* r_ptr, char* x_ptr,
                                              char* px_ptr,
                                              unsigned int mask_value)
{
    Masked<float> wresult(r_ptr, Mask(mask_value));
    Wide<const float> wx(x_ptr);
    Wide<const float> wpx(px_ptr);
    typedef BatchedCGPolicy<Param::WF, Param::WF, Param::WF> Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const float x  = wx[lane];
                const float px = wpx[lane];
                float result;
                blockvec_impl(result, x, px);
                wresult[lane] = result;
            });
        return;
    }
    __OSL_XMACRO_SFM_IMPLNAME impl;
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const float x  = wx[lane];
            const float px = wpx[lane];
            if (wresult.mask()[lane]) {
                float result;
                impl(result, x, px);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}

OSL_BATCHOP void __OSL_PNOISE_OP5(Wf, Wf, Wf, Wf, Wf)(char* r_ptr, char* x_ptr,
                                                      char* y_ptr, char* px_ptr,
                                                      char* py_ptr,
                                                      unsigned int mask_value)
{
    Masked<float> wresult(r_ptr, Mask(mask_value));
    Wide<const float> wx(x_ptr);
    Wide<const float> wy(y_ptr);
    Wide<const float> wpx(px_ptr);
    Wide<const float> wpy(py_ptr);
    typedef BatchedCGPolicy<Param::WF, Param::WF, Param::WF, Param::WF, Param::WF>
        Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const float x  = wx[lane];
                const float y  = wy[lane];
                const float px = wpx[lane];
                const float py = wpy[lane];
                float result;
                blockvec_impl(result, x, y, px, py);
                wresult[lane] = result;
            });
        return;
    }
    __OSL_XMACRO_SFM_IMPLNAME impl;
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const float x  = wx[lane];
            const float y  = wy[lane];
            const float px = wpx[lane];
            const float py = wpy[lane];
            if (wresult.mask()[lane]) {
                float result;
                impl(result, x, y, px, py);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}

OSL_BATCHOP void __OSL_PNOISE_OP3(Wf, Wv, Wv)(char* r_ptr, char* p_ptr,
                                              char* pp_ptr,
                                              unsigned int mask_value)
{
    Masked<float> wresult(r_ptr, Mask(mask_value));
    Wide<const Vec3> wp(p_ptr);
    Wide<const Vec3> wpp(pp_ptr);
    typedef BatchedCGPolicy<Param::WF, Param::WV, Param::WV> Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const Vec3 p  = wp[lane];
                const Vec3 pp = wpp[lane];
                float result;
                blockvec_impl(result, p, pp);
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
            const Vec3 pp = wpp[lane];
            if (wresult.mask()[lane]) {
                float result;
                impl(result, p, pp);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}

OSL_BATCHOP void __OSL_PNOISE_OP5(Wf, Wv, Wf, Wv, Wf)(char* r_ptr, char* p_ptr,
                                                      char* t_ptr, char* pp_ptr,
                                                      char* pt_ptr,
                                                      unsigned int mask_value)
{
    Masked<float> wresult(r_ptr, Mask(mask_value));
    Wide<const Vec3> wp(p_ptr);
    Wide<const float> wt(t_ptr);
    Wide<const Vec3> wpp(pp_ptr);
    Wide<const float> wpt(pt_ptr);
    typedef BatchedCGPolicy<Param::WF, Param::WV, Param::WF, Param::WV, Param::WF>
        Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const Vec3 p   = wp[lane];
                const float t  = wt[lane];
                const Vec3 pp  = wpp[lane];
                const float pt = wpt[lane];
                float result;
                blockvec_impl(result, p, t, pp, pt);
                wresult[lane] = result;
            });
        return;
    }
    __OSL_XMACRO_SFM_IMPLNAME impl;
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Vec3 p   = wp[lane];
            const float t  = wt[lane];
            const Vec3 pp  = wpp[lane];
            const float pt = wpt[lane];
            if (wresult.mask()[lane]) {
                float result;
                impl(result, p, t, pp, pt);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}

#endif

#ifndef __OSL_XMACRO_FLOAT_RESULTS_ONLY

OSL_BATCHOP void __OSL_PNOISE_OP3(Wv, Wf, Wf)(char* r_ptr, char* x_ptr,
                                              char* px_ptr,
                                              unsigned int mask_value)
{
    Masked<Vec3> wresult(r_ptr, Mask(mask_value));
    Wide<const float> wx(x_ptr);
    Wide<const float> wpx(px_ptr);
    typedef BatchedCGPolicy<Param::WV, Param::WF, Param::WF> Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const float x  = wx[lane];
                const float px = wpx[lane];
                Vec3 result;
                blockvec_impl(result, x, px);
                wresult[lane] = result;
            });
        return;
    }
    __OSL_XMACRO_SFM_IMPLNAME impl;
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const float x  = wx[lane];
            const float px = wpx[lane];
            if (wresult.mask()[lane]) {
                Vec3 result;
                impl(result, x, px);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}

OSL_BATCHOP void __OSL_PNOISE_OP5(Wv, Wf, Wf, Wf, Wf)(char* r_ptr, char* x_ptr,
                                                      char* y_ptr, char* px_ptr,
                                                      char* py_ptr,
                                                      unsigned int mask_value)
{
    Masked<Vec3> wresult(r_ptr, Mask(mask_value));
    Wide<const float> wx(x_ptr);
    Wide<const float> wy(y_ptr);
    Wide<const float> wpx(px_ptr);
    Wide<const float> wpy(py_ptr);
    typedef BatchedCGPolicy<Param::WV, Param::WF, Param::WF, Param::WF, Param::WF>
        Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const float x  = wx[lane];
                const float y  = wy[lane];
                const float px = wpx[lane];
                const float py = wpy[lane];
                Vec3 result;
                blockvec_impl(result, x, y, px, py);
                wresult[lane] = result;
            });
        return;
    }
    __OSL_XMACRO_SFM_IMPLNAME impl;
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const float x  = wx[lane];
            const float y  = wy[lane];
            const float px = wpx[lane];
            const float py = wpy[lane];
            if (wresult.mask()[lane]) {
                Vec3 result;
                impl(result, x, y, px, py);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}

OSL_BATCHOP void __OSL_PNOISE_OP3(Wv, Wv, Wv)(char* r_ptr, char* p_ptr,
                                              char* pp_ptr,
                                              unsigned int mask_value)
{
    Masked<Vec3> wresult(r_ptr, Mask(mask_value));
    Wide<const Vec3> wp(p_ptr);
    Wide<const Vec3> wpp(pp_ptr);
    typedef BatchedCGPolicy<Param::WV, Param::WV, Param::WV> Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const Vec3 p  = wp[lane];
                const Vec3 pp = wpp[lane];
                Vec3 result;
                blockvec_impl(result, p, pp);
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
            const Vec3 pp = wpp[lane];
            if (wresult.mask()[lane]) {
                Vec3 result;
                impl(result, p, pp);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}


OSL_BATCHOP void __OSL_PNOISE_OP5(Wv, Wv, Wf, Wv, Wf)(char* r_ptr, char* p_ptr,
                                                      char* t_ptr, char* pp_ptr,
                                                      char* pt_ptr,
                                                      unsigned int mask_value)
{
    Masked<Vec3> wresult(r_ptr, Mask(mask_value));
    Wide<const Vec3> wp(p_ptr);
    Wide<const float> wt(t_ptr);
    Wide<const Vec3> wpp(pp_ptr);
    Wide<const float> wpt(pt_ptr);
    typedef BatchedCGPolicy<Param::WV, Param::WV, Param::WF, Param::WV, Param::WF>
        Policy;
    if (Policy::simd_threshold > __OSL_WIDTH
        || (Policy::simd_threshold > 1
            && wresult.mask().count() < Policy::simd_threshold)) {
        wresult.mask().invoke_foreach<1, Policy::simd_threshold - 1>(
            [=](ActiveLane lane) -> void {
                __OSL_XMACRO_IMPLNAME blockvec_impl;
                const Vec3 p   = wp[lane];
                const float t  = wt[lane];
                const Vec3 pp  = wpp[lane];
                const float pt = wpt[lane];
                Vec3 result;
                blockvec_impl(result, p, t, pp, pt);
                wresult[lane] = result;
            });
        return;
    }
    __OSL_XMACRO_SFM_IMPLNAME impl;
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            const Vec3 p   = wp[lane];
            const float t  = wt[lane];
            const Vec3 pp  = wpp[lane];
            const float pt = wpt[lane];
            if (wresult.mask()[lane]) {
                Vec3 result;
                impl(result, p, t, pp, pt);
                wresult[ActiveLane(lane)] = result;
            }
        }
    }
}

#endif


}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#undef __OSL_PNOISE_OP3
#undef __OSL_PNOISE_OP5

#include "undef_opname_macros.h"
#undef __OSL_XMACRO_FLOAT_RESULTS_ONLY
#undef __OSL_XMACRO_VEC3_RESULTS_ONLY
#undef __OSL_XMACRO_OPNAME
#undef __OSL_XMACRO_SFM_IMPLNAME
#undef __OSL_XMACRO_IMPLNAME
#undef __OSL_XMACRO_ARGS
