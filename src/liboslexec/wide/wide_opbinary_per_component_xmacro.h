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
OSL_BATCHOP void __OSL_OP3(__OSL_XMACRO_OPNAME, Wf,Wf,Wf)
    (void *r_, void *a_, void *b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> wa(a_);
        Wide<const float> wb(b_);
        Wide<float> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            float a = wa[lane];
            float b = wb[lane];
            float r = __OSL_XMACRO_FLOAT_FUNC(a,b);
            wr[lane] = r;
        }
    }
}
#endif

OSL_BATCHOP void __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, Wf,Wf,Wf)
    (void *r_, void *a_, void *b_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> wa(a_);
        Wide<const float> wb(b_);
        Masked<float> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            float a = wa[lane];
            float b = wb[lane];
            if (wr.mask()[lane]) {
                float r = __OSL_XMACRO_FLOAT_FUNC(a,b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}

#ifndef __OSL_XMACRO_MASKED_ONLY
OSL_BATCHOP void __OSL_OP3(__OSL_XMACRO_OPNAME, Wdf,Wdf,Wdf)
    (void *r_, void *a_, void *b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> wa(a_);
        Wide<const Dual2<float>> wb(b_);
        Wide<Dual2<float>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> a = wa[lane];
            Dual2<float> b = wb[lane];
            Dual2<float> r = __OSL_XMACRO_DUAL_FUNC(a,b);
            wr[lane] = r;
        }
    }
}
#endif

OSL_BATCHOP void __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, Wdf,Wdf,Wdf)
    (void *r_, void *a_, void *b_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> wa(a_);
        Wide<const Dual2<float>> wb(b_);
        Masked<Dual2<float>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> a = wa[lane];
            Dual2<float> b = wb[lane];
            if (wr.mask()[lane]) {
                Dual2<float> r = __OSL_XMACRO_DUAL_FUNC(a,b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}

#ifndef __OSL_XMACRO_MASKED_ONLY
OSL_BATCHOP void __OSL_OP3(__OSL_XMACRO_OPNAME, Wdf,Wf,Wdf)
    (void *r_, void *a_, void *b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> wa(a_);
        Wide<const Dual2<float>> wb(b_);
        Wide<Dual2<float>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            float a = wa[lane];
            Dual2<float> b = wb[lane];
            Dual2<float> r = __OSL_XMACRO_DUAL_FUNC(Dual2<float>(a),b);
            wr[lane] = r;
        }
    }
}
#endif

OSL_BATCHOP void __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, Wdf,Wf,Wdf)
    (void *r_, void *a_, void *b_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const float> wa(a_);
        Wide<const Dual2<float>> wb(b_);
        Masked<Dual2<float>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            float a = wa[lane];
            Dual2<float> b = wb[lane];
            if (wr.mask()[lane]) {
                Dual2<float> r = __OSL_XMACRO_DUAL_FUNC(Dual2<float>(a),b);
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}

#ifndef __OSL_XMACRO_MASKED_ONLY
OSL_BATCHOP void __OSL_OP3(__OSL_XMACRO_OPNAME, Wdf,Wdf,Wf)
    (void *r_, void *a_, void *b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> wa(a_);
        Wide<const float> wb(b_);
        Wide<Dual2<float>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> a = wa[lane];
            float b = wb[lane];
            Dual2<float> r = __OSL_XMACRO_DUAL_FUNC(a,Dual2<float>(b));
            wr[lane] = r;
        }
    }
}
#endif

OSL_BATCHOP void __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, Wdf,Wdf,Wf)
    (void *r_, void *a_, void *b_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<float>> wa(a_);
        Wide<const float> wb(b_);
        Masked<Dual2<float>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Dual2<float> a = wa[lane];
            float b = wb[lane];
            if (wr.mask()[lane]) {
                Dual2<float> r = __OSL_XMACRO_DUAL_FUNC(a,Dual2<float>(b));
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}

#ifndef __OSL_XMACRO_MASKED_ONLY
OSL_BATCHOP void __OSL_OP3(__OSL_XMACRO_OPNAME, Wv,Wv,Wv)
    (void *r_, void *a_, void *b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wa(a_);
        Wide<const Vec3> wb(b_);
        Wide<Vec3> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wa[lane];
            Vec3 b = wb[lane];
            Vec3 r (__OSL_XMACRO_FLOAT_FUNC(a.x, b.x),
                    __OSL_XMACRO_FLOAT_FUNC(a.y, b.y),
                    __OSL_XMACRO_FLOAT_FUNC(a.z, b.z));
            wr[lane] = r;
        }
    }
}
#endif

OSL_BATCHOP void __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, Wv,Wv,Wv)
    (void *r_, void *a_, void *b_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wa(a_);
        Wide<const Vec3> wb(b_);
        Masked<Vec3> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Vec3 a = wa[lane];
            Vec3 b = wb[lane];
            if (wr.mask()[lane]) {
                Vec3 r (__OSL_XMACRO_FLOAT_FUNC(a.x, b.x),
                        __OSL_XMACRO_FLOAT_FUNC(a.y, b.y),
                        __OSL_XMACRO_FLOAT_FUNC(a.z, b.z));
                wr[ActiveLane(lane)] = r;
            }
        }
    }
}

#ifndef __OSL_XMACRO_MASKED_ONLY
OSL_BATCHOP void __OSL_OP3(__OSL_XMACRO_OPNAME, Wdv,Wdv,Wdv)
    (void *r_, void *a_, void *b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wa(a_);
        Wide<const Dual2<Vec3>> wb(b_);
        Wide<Dual2<Vec3>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wa[lane];
            Dual2<Vec3> b = wb[lane];
            /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */
            Dual2<float> ax = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().x, a.dx().x, a.dy().x),
                                        Dual2<float> (b.val().x, b.dx().x, b.dy().x));
            Dual2<float> ay = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().y, a.dx().y, a.dy().y),
                                        Dual2<float> (b.val().y, b.dx().y, b.dy().y));
            Dual2<float> az = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().z, a.dx().z, a.dy().z),
                                        Dual2<float> (b.val().z, b.dx().z, b.dy().z));
            /* Now swizzle back */
            Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),
                           Vec3( ax.dx(),  ay.dx(),  az.dx() ),
                           Vec3( ax.dy(),  ay.dy(),  az.dy() ));
            wr[lane] = r;
        }
    }
}
#endif

OSL_BATCHOP void __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, Wdv,Wdv,Wdv)
    (void *r_, void *a_, void *b_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wa(a_);
        Wide<const Dual2<Vec3>> wb(b_);
        Masked<Dual2<Vec3>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wa[lane];
            Dual2<Vec3> b = wb[lane];
            if (wr.mask()[lane]) {
                /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */
                Dual2<float> ax = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().x, a.dx().x, a.dy().x),
                                            Dual2<float> (b.val().x, b.dx().x, b.dy().x));
                Dual2<float> ay = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().y, a.dx().y, a.dy().y),
                                            Dual2<float> (b.val().y, b.dx().y, b.dy().y));
                Dual2<float> az = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().z, a.dx().z, a.dy().z),
                                            Dual2<float> (b.val().z, b.dx().z, b.dy().z));
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
OSL_BATCHOP void __OSL_OP3(__OSL_XMACRO_OPNAME, Wdv,Wv,Wdv)
    (void *r_, void *a_, void *b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wa(a_);
        Wide<const Dual2<Vec3>> wb(b_);
        Wide<Dual2<Vec3>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a(unproxy(wa[lane]));
            Dual2<Vec3> b = wb[lane];
            /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */
            Dual2<float> ax = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().x, a.dx().x, a.dy().x),
                                        Dual2<float> (b.val().x, b.dx().x, b.dy().x));
            Dual2<float> ay = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().y, a.dx().y, a.dy().y),
                                        Dual2<float> (b.val().y, b.dx().y, b.dy().y));
            Dual2<float> az = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().z, a.dx().z, a.dy().z),
                                        Dual2<float> (b.val().z, b.dx().z, b.dy().z));
            /* Now swizzle back */
            Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),
                           Vec3( ax.dx(),  ay.dx(),  az.dx() ),
                           Vec3( ax.dy(),  ay.dy(),  az.dy() ));
            wr[lane] = r;
        }
    }
}
#endif

OSL_BATCHOP void __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, Wdv,Wv,Wdv)
    (void *r_, void *a_, void *b_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Vec3> wa(a_);
        Wide<const Dual2<Vec3>> wb(b_);
        Masked<Dual2<Vec3>> wr(r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a(unproxy(wa[lane]));
            Dual2<Vec3> b = wb[lane];
            if (wr.mask()[lane]) {
                /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */
                Dual2<float> ax = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().x, a.dx().x, a.dy().x),
                                            Dual2<float> (b.val().x, b.dx().x, b.dy().x));
                Dual2<float> ay = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().y, a.dx().y, a.dy().y),
                                            Dual2<float> (b.val().y, b.dx().y, b.dy().y));
                Dual2<float> az = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().z, a.dx().z, a.dy().z),
                                            Dual2<float> (b.val().z, b.dx().z, b.dy().z));
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
OSL_BATCHOP void __OSL_OP3(__OSL_XMACRO_OPNAME, Wdv,Wdv,Wv)
    (void *r_, void *a_, void *b_)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wa(a_);
        Wide<const Vec3> wb(b_);
        Wide<Dual2<Vec3>> wr(r_);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for(int lane=0; lane < __OSL_WIDTH; ++lane) {
            Dual2<Vec3> a = wa[lane];
            Dual2<Vec3> b(unproxy(wb[lane]));
            /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */
            Dual2<float> ax = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().x, a.dx().x, a.dy().x),
                                        Dual2<float> (b.val().x, b.dx().x, b.dy().x));
            Dual2<float> ay = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().y, a.dx().y, a.dy().y),
                                        Dual2<float> (b.val().y, b.dx().y, b.dy().y));
            Dual2<float> az = __OSL_XMACRO_DUAL_FUNC (Dual2<float> (a.val().z, a.dx().z, a.dy().z),
                                        Dual2<float> (b.val().z, b.dx().z, b.dy().z));
            /* Now swizzle back */
            Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),
                           Vec3( ax.dx(),  ay.dx(),  az.dx() ),
                           Vec3( ax.dy(),  ay.dy(),  az.dy() ));
            wr[lane] = r;
        }
    }
}
#endif

OSL_BATCHOP void __OSL_MASKED_OP3(__OSL_XMACRO_OPNAME, Wdv,Wdv,Wv)
    (void *r_, void *a_, void *b_, unsigned int mask_value)
{
    OSL_FORCEINLINE_BLOCK
    {
        Wide<const Dual2<Vec3>> wa(a_);
        Wide<const Vec3> wb (b_);
        Masked<Dual2<Vec3>> wr (r_, Mask(mask_value));
        OSL_OMP_PRAGMA(omp simd simdlen (__OSL_WIDTH))
        for(int lane=0; lane <__OSL_WIDTH; ++lane){
            Dual2<Vec3> a = wa[lane];
            Dual2<Vec3> b (unproxy (wb[lane]));
            if (wr.mask()[lane]) {
                /*Swizzle the Dual2<Vec3>s into 3 Dual2<float>s */
                Dual2<float> ax = __OSL_XMACRO_DUAL_FUNC(Dual2<float> (a.val().x, a.dx().x, a.dy().x ),
                                           Dual2<float> (b.val().x, b.dx().x, b.dy().x));
                Dual2<float> ay = __OSL_XMACRO_DUAL_FUNC(Dual2<float> (a.val().y, a.dx().y, a.dy().y ),
                                           Dual2<float> (b.val().y, b.dx().y, b.dy().y));
                Dual2<float> az = __OSL_XMACRO_DUAL_FUNC(Dual2<float> (a.val().z, a.dx().z, a.dy().z ),
                                           Dual2<float> (b.val().z, b.dx().z, b.dy().z));

                /*Now swizzle back */
                Dual2<Vec3> r (Vec3(ax.val(), ay.val(), az.val()),
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
#undef __OSL_XMACRO_MASKED_ONLY


