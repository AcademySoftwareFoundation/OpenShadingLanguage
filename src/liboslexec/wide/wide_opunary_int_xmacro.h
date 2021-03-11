// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage
#ifdef __OSL_XMACRO_ARGS
#    define __OSL_XMACRO_OPNAME \
        __OSL_EXPAND(__OSL_XMACRO_ARG1 __OSL_XMACRO_ARGS)
#    define __OSL_XMACRO_INT_FUNC \
        __OSL_EXPAND(__OSL_XMACRO_ARG2 __OSL_XMACRO_ARGS)
#endif

#ifndef __OSL_XMACRO_OPNAME
#    error must define __OSL_XMACRO_OPNAME to name of unary operation before including this header
#endif

#ifndef __OSL_XMACRO_INT_FUNC
#    error must define __OSL_XMACRO_INT_FUNC to name of SIMD friendly unary implementation before including this header
#endif

#ifndef __OSL_WIDTH
#    error must define __OSL_WIDTH to number of SIMD lanes before including this header
#endif


OSL_BATCHOP void __OSL_OP2(__OSL_XMACRO_OPNAME, Wi, Wi)(void* r_, void* val_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const int> wval(val_);
        Wide<int> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            int val  = wval[lane];
            int r    = __OSL_XMACRO_INT_FUNC(val);
            wr[lane] = r;
        }
    }
}

OSL_BATCHOP void __OSL_MASKED_OP2(__OSL_XMACRO_OPNAME, Wi,
                                  Wi)(void* r_, void* val_,
                                      unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const int> wval(val_);
        Masked<int> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            int val = wval[lane];
            if (wr.mask()[lane]) {
                int r                = __OSL_XMACRO_INT_FUNC(val);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}

#undef __OSL_XMACRO_ARGS
#undef __OSL_XMACRO_OPNAME
#undef __OSL_XMACRO_INT_FUNC
