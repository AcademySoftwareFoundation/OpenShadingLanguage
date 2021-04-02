// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage
#ifdef __OSL_XMACRO_ARGS
#    define __OSL_XMACRO_OPNAME \
        __OSL_EXPAND(__OSL_XMACRO_ARG1 __OSL_XMACRO_ARGS)
#    define __OSL_XMACRO_FLOAT_FUNC \
        __OSL_EXPAND(__OSL_XMACRO_ARG2 __OSL_XMACRO_ARGS)
#endif

#ifndef __OSL_XMACRO_OPNAME
#    error must define __OSL_XMACRO_OPNAME to name of unary operation before including this header
#endif

#ifndef __OSL_XMACRO_FLOAT_FUNC
#    error must define __OSL_XMACRO_FLOAT_FUNC to name of SIMD friendly unary implementation before including this header
#endif

#ifndef __OSL_WIDTH
#    error must define __OSL_WIDTH to number of SIMD lanes before including this header
#endif

OSL_BATCHOP void __OSL_OP2(__OSL_XMACRO_OPNAME, Wf, Wf)(void* r_, void* val_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> wval(val_);
        Wide<float> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float val = wval[lane];
            float r   = __OSL_XMACRO_FLOAT_FUNC(val);
            wr[lane]  = r;
        }
    }
}

OSL_BATCHOP void __OSL_MASKED_OP2(__OSL_XMACRO_OPNAME, Wf,
                                  Wf)(void* r_, void* val_,
                                      unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> wval(val_);
        Masked<float> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float val = wval[lane];
            if (wr.mask()[lane]) {
                float r              = __OSL_XMACRO_FLOAT_FUNC(val);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}

OSL_BATCHOP void __OSL_OP2(__OSL_XMACRO_OPNAME, Wv, Wv)(void* r_, void* val_)
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

OSL_BATCHOP void __OSL_MASKED_OP2(__OSL_XMACRO_OPNAME, Wv,
                                  Wv)(void* r_, void* val_,
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


#undef __OSL_XMACRO_ARGS
#undef __OSL_XMACRO_OPNAME
#undef __OSL_XMACRO_FLOAT_FUNC
#undef __OSL_XMACRO_DUAL_FUNC
