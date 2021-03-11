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

OSL_BATCHOP void __OSL_OP3(__OSL_XMACRO_OPNAME, Wf, Wf, Wf)(void* r_, void* a_,
                                                            void* b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> wa(a_);
        Wide<const float> wb(b_);
        Wide<float> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float a  = wa[lane];
            float b  = wb[lane];
            float r  = __OSL_XMACRO_FLOAT_FUNC(a, b);
            wr[lane] = r;
        }
    }
}

OSL_BATCHOP void __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, Wf, Wf,
                                  Wf)(void* r_, void* a_, void* b_,
                                      unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> wa(a_);
        Wide<const float> wb(b_);
        Masked<float> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            float a = wa[lane];
            float b = wb[lane];
            if (wr.mask()[lane]) {
                float r              = __OSL_XMACRO_FLOAT_FUNC(a, b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}

OSL_BATCHOP void __OSL_OP3(__OSL_XMACRO_OPNAME, Wv, Wv, Wv)(void* r_, void* a_,
                                                            void* b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wa(a_);
        Wide<const Vec3> wb(b_);
        Wide<Vec3> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wa[lane];
            Vec3 b = wb[lane];
            Vec3 r(__OSL_XMACRO_FLOAT_FUNC(a.x, b.x),
                   __OSL_XMACRO_FLOAT_FUNC(a.y, b.y),
                   __OSL_XMACRO_FLOAT_FUNC(a.z, b.z));
            wr[lane] = r;
        }
    }
}

OSL_BATCHOP void __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, Wv, Wv,
                                  Wv)(void* r_, void* a_, void* b_,
                                      unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wa(a_);
        Wide<const Vec3> wb(b_);
        Masked<Vec3> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wa[lane];
            Vec3 b = wb[lane];
            if (wr.mask()[lane]) {
                Vec3 r(__OSL_XMACRO_FLOAT_FUNC(a.x, b.x),
                       __OSL_XMACRO_FLOAT_FUNC(a.y, b.y),
                       __OSL_XMACRO_FLOAT_FUNC(a.z, b.z));
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



#undef __OSL_XMACRO_ARGS
#undef __OSL_XMACRO_OPNAME
#undef __OSL_XMACRO_FLOAT_FUNC
#undef __OSL_XMACRO_DUAL_FUNC
