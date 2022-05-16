// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#ifdef __OSL_XMACRO_ARGS
#    define __OSL_XMACRO_OPNAME \
        __OSL_EXPAND(__OSL_XMACRO_ARG1 __OSL_XMACRO_ARGS)
#    define __OSL_XMACRO_FLOAT_FUNC \
        __OSL_EXPAND(__OSL_XMACRO_ARG2 __OSL_XMACRO_ARGS)
#    define __OSL_XMACRO_DUAL_FUNC \
        __OSL_EXPAND(__OSL_XMACRO_ARG3 __OSL_XMACRO_ARGS)
#endif

#ifndef __OSL_XMACRO_OPNAME
#    error must define __OSL_XMACRO_OPNAME to name of unary operation before including this header
#endif

#ifndef __OSL_XMACRO_FLOAT_FUNC
#    error must define __OSL_XMACRO_FLOAT_FUNC to name of SIMD friendly unary implementation before including this header
#endif

#ifndef __OSL_XMACRO_DUAL_FUNC
#    error must define __OSL_XMACRO_DUAL_FUNC to name of unary implementation before including this header
#endif

#ifndef __OSL_WIDTH
#    error must define __OSL_WIDTH to number of SIMD lanes before including this header
#endif

#define __OSL_SKIP_UNDEF
#include "wide_opunary_float_xmacro.h"


OSL_BATCHOP void
__OSL_OP2(__OSL_XMACRO_OPNAME, Wv, Wv)(void* r_, void* val_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wval(val_);
        Wide<Vec3> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 val = wval[lane];
            Vec3 r(__OSL_XMACRO_FLOAT_FUNC(val.x),
                   __OSL_XMACRO_FLOAT_FUNC(val.y),
                   __OSL_XMACRO_FLOAT_FUNC(val.z));
            wr[lane] = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP2(__OSL_XMACRO_OPNAME, Wv, Wv)(void* r_, void* val_,
                                              unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wval(val_);
        Masked<Vec3> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 val = wval[lane];
            if (wr.mask()[lane]) {
                Vec3 r(__OSL_XMACRO_FLOAT_FUNC(val.x),
                       __OSL_XMACRO_FLOAT_FUNC(val.y),
                       __OSL_XMACRO_FLOAT_FUNC(val.z));
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



OSL_BATCHOP void
__OSL_OP2(__OSL_XMACRO_OPNAME, Wdv, Wdv)(void* r_, void* val_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wdf(val_);
        Wide<Dual2<Vec3>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> df = wdf[lane];
            Dual2<float> sx(df.val().x, df.dx().x, df.dy().x);
            Dual2<float> sy(df.val().y, df.dx().y, df.dy().y);
            Dual2<float> sz(df.val().z, df.dx().z, df.dy().z);
            Dual2<float> ax = __OSL_XMACRO_DUAL_FUNC(sx);
            Dual2<float> ay = __OSL_XMACRO_DUAL_FUNC(sy);
            Dual2<float> az = __OSL_XMACRO_DUAL_FUNC(sz);
            Dual2<Vec3> r(Vec3(ax.val(), ay.val(), az.val()),
                          Vec3(ax.dx(), ay.dx(), az.dx()),
                          Vec3(ax.dy(), ay.dy(), az.dy()));
            wr[lane] = r;
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP2(__OSL_XMACRO_OPNAME, Wdv, Wdv)(void* r_, void* val_,
                                                unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wdf(val_);
        Masked<Dual2<Vec3>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> df = wdf[lane];
            if (wr.mask()[lane]) {
                Dual2<float> sx(df.val().x, df.dx().x, df.dy().x);
                Dual2<float> sy(df.val().y, df.dx().y, df.dy().y);
                Dual2<float> sz(df.val().z, df.dx().z, df.dy().z);
                Dual2<float> ax = __OSL_XMACRO_DUAL_FUNC(sx);
                Dual2<float> ay = __OSL_XMACRO_DUAL_FUNC(sy);
                Dual2<float> az = __OSL_XMACRO_DUAL_FUNC(sz);
                Dual2<Vec3> r(Vec3(ax.val(), ay.val(), az.val()),
                              Vec3(ax.dx(), ay.dx(), az.dx()),
                              Vec3(ax.dy(), ay.dy(), az.dy()));
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



#undef __OSL_XMACRO_ARGS
#undef __OSL_XMACRO_OPNAME
#undef __OSL_XMACRO_FLOAT_FUNC
#undef __OSL_XMACRO_DUAL_FUNC
