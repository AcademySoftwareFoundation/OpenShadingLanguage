// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifdef __OSL_XMACRO_ARGS

#define __OSL_XMACRO_OPNAME __OSL_EXPAND(__OSL_XMACRO_ARG1 __OSL_XMACRO_ARGS)
#define __OSL_XMACRO_FLOAT_FUNC __OSL_EXPAND(__OSL_XMACRO_ARG2 __OSL_XMACRO_ARGS)
#define __OSL_XMACRO_DUAL_FUNC __OSL_EXPAND(__OSL_XMACRO_ARG3 __OSL_XMACRO_ARGS)

#endif

#ifndef __OSL_XMACRO_OPNAME
#error must define __OSL_XMACRO_OPNAME to name of unary operation before including this header
#endif

#ifndef __OSL_XMACRO_FLOAT_FUNC
#error must define __OSL_XMACRO_FLOAT_FUNC to name of SIMD friendly unary implementation before including this header
#endif

#ifndef __OSL_XMACRO_DUAL_FUNC
#error must define __OSL_XMACRO_DUAL_FUNC to name of unary implementation before including this header
#endif

#ifndef __OSL_WIDTH
#error must define __OSL_WIDTH to number of SIMD lanes before including this header
#endif

#ifndef __OSL_XMACRO_MASKED_ONLY
OSL_BATCHOP void __OSL_OP3(__OSL_XMACRO_OPNAME, Wv,Wv,Wf)
    (void *r_, void *a_, void *b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wa(a_);
        Wide<const float> wb(b_);
        Wide<Vec3> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wa[lane];
            float b = wb[lane];
            Vec3 r (__OSL_XMACRO_FLOAT_FUNC(a.x, b),
                    __OSL_XMACRO_FLOAT_FUNC(a.y, b),
                    __OSL_XMACRO_FLOAT_FUNC(a.z, b));
            wr[lane] = r;
        }
    }
}
#endif

OSL_BATCHOP void __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, Wv,Wv,Wf)
    (void *r_, void *a_, void *b_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wa(a_);
        Wide<const float> wb(b_);
        Masked<Vec3> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wa[lane];
            float b = wb[lane];
            if (wr.mask()[lane]) {
                Vec3 r (__OSL_XMACRO_FLOAT_FUNC(a.x, b),
                        __OSL_XMACRO_FLOAT_FUNC(a.y, b),
                        __OSL_XMACRO_FLOAT_FUNC(a.z, b));
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}


#ifndef __OSL_XMACRO_MASKED_ONLY
OSL_BATCHOP void __OSL_OP3(__OSL_XMACRO_OPNAME, Wdv,Wdv,Wdf)
    (void *r_, void *a_, void *b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wa(a_);
        Wide<const Dual2<float>> wb(b_);
        Wide<Dual2<Vec3>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wa[lane];
            Dual2<float> b = wb[lane];
            /* Swizzle the Dual2<Vec3> into 3 Dual2<float> */
            Dual2<float> ax = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().x, a.dx().x, a.dy().x),
                                        b);
            Dual2<float> ay = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().y, a.dx().y, a.dy().y),
                                        b);
            Dual2<float> az = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().z, a.dx().z, a.dy().z),
                                        b);
            /* Now swizzle back */
            Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),
                           Vec3( ax.dx(),  ay.dx(),  az.dx() ),
                           Vec3( ax.dy(),  ay.dy(),  az.dy() ));
            wr[lane] = r;
        }
    }
}
#endif

OSL_BATCHOP void __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, Wdv,Wdv,Wdf)
    (void *r_, void *a_, void *b_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wa(a_);
        Wide<const Dual2<float>> wb(b_);
        Masked<Dual2<Vec3>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wa[lane];
            Dual2<float> b = wb[lane];
            if (wr.mask()[lane]) {
                /* Swizzle the Dual2<Vec3> into 3 Dual2<float> */
                Dual2<float> ax = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().x, a.dx().x, a.dy().x),
                                b);
                Dual2<float> ay = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().y, a.dx().y, a.dy().y),
                                b);
                Dual2<float> az = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().z, a.dx().z, a.dy().z),
                                b);
                /* Now swizzle back */
                Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),
                      Vec3( ax.dx(),  ay.dx(),  az.dx() ),
                      Vec3( ax.dy(),  ay.dy(),  az.dy() ));
                wr[ActiveLane(lane)] = r;
            }
        }
     }
 }

#ifndef __OSL_XMACRO_MASKED_ONLY
OSL_BATCHOP void __OSL_OP3(__OSL_XMACRO_OPNAME, Wdv,Wv,Wdf)
    (void *r_, void *a_, void *b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wa(a_);
        Wide<const Dual2<float>> wb(b_);
        Wide<Dual2<Vec3>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wa[lane];
            Dual2<Vec3> da(a);
            Dual2<float> b = wb[lane];
            /* Swizzle the Dual2<Vec3> into 3 Dual2<float> */
            Dual2<float> ax = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (da.val().x, da.dx().x, da.dy().x),
                                    b);
            Dual2<float> ay = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (da.val().y, da.dx().y, da.dy().y),
                                        b);
            Dual2<float> az = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (da.val().z, da.dx().z, da.dy().z),
                                        b);
            /* Now swizzle back */
            Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),
                           Vec3( ax.dx(),  ay.dx(),  az.dx() ),
                           Vec3( ax.dy(),  ay.dy(),  az.dy() ));
            wr[lane] = r;
        }
    }
}
#endif

OSL_BATCHOP void __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, Wdv,Wv,Wdf)
    (void *r_, void *a_, void *b_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wa(a_);
        Wide<const Dual2<float>> wb(b_);
        Masked<Dual2<Vec3>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wa[lane];
            Dual2<Vec3> da(a);
            Dual2<float> b = wb[lane];
            if (wr.mask()[lane]) {
                /* Swizzle the Dual2<Vec3> into 3 Dual2<float> */
                Dual2<float> ax = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (da.val().x, da.dx().x, da.dy().x),
                                            b);
                Dual2<float> ay = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (da.val().y, da.dx().y, da.dy().y),
                                            b);
                Dual2<float> az = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (da.val().z, da.dx().z, da.dy().z),
                                            b);
                /* Now swizzle back */
                Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),
                               Vec3( ax.dx(),  ay.dx(),  az.dx() ),
                               Vec3( ax.dy(),  ay.dy(),  az.dy() ));\
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}

#ifndef __OSL_XMACRO_MASKED_ONLY
OSL_BATCHOP void __OSL_OP3(__OSL_XMACRO_OPNAME, Wdv,Wdv,Wf)
    (void *r_, void *a_, void *b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wa(a_);
        Wide<const float> wb(b_);
        Wide<Dual2<Vec3>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wa[lane];
            float b = wb[lane];
            Dual2<float> db(b);
            /* Swizzle the Dual2<Vec3> into 3 Dual2<float> */
            Dual2<float> ax = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().x, a.dx().x, a.dy().x),
                                        db);
            Dual2<float> ay = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().y, a.dx().y, a.dy().y),
                                        db);
            Dual2<float> az = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().z, a.dx().z, a.dy().z),
                                        db);
            /* Now swizzle back */
            Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),
                           Vec3( ax.dx(),  ay.dx(),  az.dx() ),
                           Vec3( ax.dy(),  ay.dy(),  az.dy() ));
            wr[lane] = r;
        }
    }
}
#endif

OSL_BATCHOP void __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, Wdv,Wdv,Wf)
    (void *r_, void *a_, void *b_,unsigned int mask_value )
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wa(a_);
        Wide<const float> wb(b_);
        Masked<Dual2<Vec3>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wa[lane];
            float b = wb[lane];
            Dual2<float> db(b);
            if (wr.mask()[lane]) {
                /* Swizzle the Dual2<Vec3> into 3 Dual2<float> */
                Dual2<float> ax = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().x, a.dx().x, a.dy().x),
                                            db);
                Dual2<float> ay = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().y, a.dx().y, a.dy().y),
                                            db);
                Dual2<float> az = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().z, a.dx().z, a.dy().z),
                                            db);
                /* Now swizzle back */
                Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),
                               Vec3( ax.dx(),  ay.dx(),  az.dx() ),
                               Vec3( ax.dy(),  ay.dy(),  az.dy() ));\
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}



#undef __OSL_XMACRO_ARGS
#undef __OSL_XMACRO_OPNAME
#undef __OSL_XMACRO_FLOAT_FUNC
#undef __OSL_XMACRO_DUAL_FUNC
#undef __OSL_XMACRO_MASKED_ONLY



