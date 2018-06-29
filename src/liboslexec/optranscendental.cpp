/*
Copyright (c) 2017 Intel Inc., et al.
All Rights Reserved.
  
Copyright (c) 2009-2015 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of Transcendental operations
/// NOTE: many functions are left as LLVM IR, but some are better to 
/// execute from the library to take advantage of compiler's small vector
/// math library versions.
///
/////////////////////////////////////////////////////////////////////////

#include <cmath>

#include "oslexec_pvt.h"
#include "OSL/dual.h"
#include "OSL/dual_vec.h"
#include "OSL/Imathx.h"
#include "OSL/wide.h"
#include "sfmath.h"

#include <OpenEXR/ImathFun.h>
#include <OpenImageIO/fmath.h>
#include <OpenImageIO/simd.h>

// TODO: investigate validity of using restrict, mostly concerned about dest == src and if still legal

OSL_NAMESPACE_ENTER
namespace pvt {

#define MAKE_WIDE_UNARY_F_OP(name,floatfunc,dualfunc)         \
OSL_SHADEOP void                                                    \
osl_##name##_w16fw16f (void * /*OSL_RESTRICT*/ r_, void * /*OSL_RESTRICT*/ val_)                        \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<float> wval(val_);						\
		WideAccessor<float> wr(r_);								    \
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			float val = wval[lane];                                 \
			float r = floatfunc(val);                               \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16fw16f_masked (void * /*OSL_RESTRICT*/ r_, void * /*OSL_RESTRICT*/ val_, int mask_value) \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<float> wval(val_);						\
		MaskedAccessor<float> wr(r_, Mask(mask_value));				\
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			float val = wval[lane];                                 \
			float r = floatfunc(val);                               \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
																	\
OSL_SHADEOP void                                                    \
osl_##name##_w16dfw16df (void *r_, void *val_)                      \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Dual2<float>> wval(val_);					\
		WideAccessor<Dual2<float>> wr(r_);							\
		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wr.width)) \
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wr.width)) \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Dual2<float> val = wval[lane];                          \
			Dual2<float> r = dualfunc(val);                         \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16dfw16df_masked (void *r_, void *val_, int mask_value) \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Dual2<float>> wval(val_);					\
		MaskedAccessor<Dual2<float>> wr(r_, Mask(mask_value));		\
		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wr.width)) \
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wr.width)) \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Dual2<float> val = wval[lane];                          \
			Dual2<float> r = dualfunc(val);                         \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}

#define MAKE_WIDE_UNARY_PERCOMPONENT_OP(name,floatfunc,dualfunc)         \
MAKE_WIDE_UNARY_F_OP(name,floatfunc,dualfunc)         \
OSL_SHADEOP void                                                    \
osl_##name##_w16vw16v (void *r_, void *val_)                        \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Vec3> wval(val_);						    \
		WideAccessor<Vec3> wr(r_);								    \
		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wr.width)) \
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wr.width)) \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Vec3 val = wval[lane];                                  \
			Vec3 r (floatfunc(val.x),                               \
					floatfunc(val.y),                               \
				    floatfunc(val.z));                              \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16vw16v_masked (void *r_, void *val_, int mask_value)                        \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Vec3> wval(val_);						    \
		MaskedAccessor<Vec3> wr(r_, Mask(mask_value));								    \
		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wr.width)) \
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wr.width)) \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Vec3 val = wval[lane];                                  \
			Vec3 r (floatfunc(val.x),                               \
					floatfunc(val.y),                               \
				    floatfunc(val.z));                              \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                       \
osl_##name##_w16dvw16dv (void *r_, void *val_)                      \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Dual2<Vec3>> wdf(val_);					\
		WideAccessor<Dual2<Vec3>> wr(r_);							\
		/*OSL_OMP_PRAGMA(omp simd simdlen(wr.width))*/                  \
		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wr.width)) \
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wr.width)) \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Dual2<Vec3> df = wdf[lane];                             \
		    Dual2<float> sx (df.val().x, df.dx().x, df.dy().x);   \
		    Dual2<float> sy (df.val().y, df.dx().y, df.dy().y);   \
		    Dual2<float> sz (df.val().z, df.dx().z, df.dy().z);   \
		    Dual2<float> ax = dualfunc (sx);   \
		    Dual2<float> ay = dualfunc (sy);   \
		    Dual2<float> az = dualfunc (sz);   \
			Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),     \
			               Vec3( ax.dx(),  ay.dx(),  az.dx() ),     \
			               Vec3( ax.dy(),  ay.dy(),  az.dy() ));    \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
OSL_SHADEOP void                       \
osl_##name##_w16dvw16dv_masked (void *r_, void *val_, int mask_value)                      \
{                                                                   \
    OSL_INTEL_PRAGMA(forceinline recursive)                         \
    {                                                               \
        ConstWideAccessor<Dual2<Vec3>> wdf(val_);                   \
        MaskedAccessor<Dual2<Vec3>> wr(r_, Mask(mask_value));        \
        /*OSL_OMP_PRAGMA(omp simd simdlen(wr.width))*/                  \
        OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wr.width)) \
        OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wr.width)) \
        for(int lane=0; lane < wr.width; ++lane) {                  \
            Dual2<Vec3> df = wdf[lane];                             \
            Dual2<float> sx (df.val().x, df.dx().x, df.dy().x);   \
            Dual2<float> sy (df.val().y, df.dx().y, df.dy().y);   \
            Dual2<float> sz (df.val().z, df.dx().z, df.dy().z);   \
            Dual2<float> ax = dualfunc (sx);   \
            Dual2<float> ay = dualfunc (sy);   \
            Dual2<float> az = dualfunc (sz);   \
            Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),     \
                           Vec3( ax.dx(),  ay.dx(),  az.dx() ),     \
                           Vec3( ax.dy(),  ay.dy(),  az.dy() ));    \
            wr[lane] = r;                                           \
        }                                                           \
    }                                                               \
}



#define MAKE_WIDE_BINARY_PERCOMPONENT_OP(name,floatfunc,dualfunc)   \
OSL_SHADEOP void                                                    \
osl_##name##_w16fw16fw16f (void *r_, void *a_, void *b_)            \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<float> wa(a_);						    \
		ConstWideAccessor<float> wb(b_);						    \
		WideAccessor<float> wr(r_);								    \
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			float a = wa[lane];                                     \
			float b = wb[lane];                                     \
			float r = floatfunc(a,b);                               \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16fw16fw16f_masked (void *r_, void *a_, void *b_, int mask_value)            \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<float> wa(a_);						    \
		ConstWideAccessor<float> wb(b_);						    \
		MaskedAccessor<float> wr(r_, Mask(mask_value));				\
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			float a = wa[lane];                                     \
			float b = wb[lane];                                     \
			float r = floatfunc(a,b);                               \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
																	\
OSL_SHADEOP void                                                    \
osl_##name##_w16dfw16dfw16df (void *r_, void *a_, void *b_)         \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Dual2<float>> wa(a_);					    \
		ConstWideAccessor<Dual2<float>> wb(b_);					    \
		WideAccessor<Dual2<float>> wr(r_);							\
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Dual2<float> a = wa[lane];                              \
			Dual2<float> b = wb[lane];                              \
			Dual2<float> r = dualfunc(a,b);                         \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
	                                                                \
OSL_SHADEOP void                                                    \
osl_##name##_w16dfw16dfw16df_masked (void *r_, void *a_, void *b_, int mask_value)         \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Dual2<float>> wa(a_);					    \
		ConstWideAccessor<Dual2<float>> wb(b_);					    \
		MaskedAccessor<Dual2<float>> wr(r_, Mask(mask_value));		\
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Dual2<float> a = wa[lane];                              \
			Dual2<float> b = wb[lane];                              \
			Dual2<float> r = dualfunc(a,b);                         \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
	                                                                \
OSL_SHADEOP void                                                    \
osl_##name##_w16dfw16fw16df (void *r_, void *a_, void *b_)          \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
	    ConstWideAccessor<float> wa(a_);					        \
		ConstWideAccessor<Dual2<float>> wb(b_);					    \
		WideAccessor<Dual2<float>> wr(r_);							\
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			float a = wa[lane];                                     \
			Dual2<float> b = wb[lane];                              \
			Dual2<float> r = dualfunc(Dual2<float>(a),b);           \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
OSL_SHADEOP void                                                    \
osl_##name##_w16dfw16fw16df_masked (void *r_, void *a_, void *b_, int mask_value)          \
{                                                                   \
    OSL_INTEL_PRAGMA(forceinline recursive)                         \
    {                                                               \
        ConstWideAccessor<float> wa(a_);                            \
        ConstWideAccessor<Dual2<float>> wb(b_);                     \
        MaskedAccessor<Dual2<float>> wr(r_, Mask(mask_value));      \
        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
        for(int lane=0; lane < wr.width; ++lane) {                  \
            float a = wa[lane];                                     \
            Dual2<float> b = wb[lane];                              \
            Dual2<float> r = dualfunc(Dual2<float>(a),b);           \
            wr[lane] = r;                                           \
        }                                                           \
    }                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16dfw16dfw16f (void *r_, void *a_, void *b_)          \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Dual2<float>> wa(a_);					    \
		ConstWideAccessor<float> wb(b_);					        \
		WideAccessor<Dual2<float>> wr(r_);							\
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Dual2<float> a = wa[lane];                              \
			float b = wb[lane];                                     \
			Dual2<float> r = dualfunc(a,Dual2<float>(b));           \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16dfw16dfw16f_masked (void *r_, void *a_, void *b_, int mask_value)          \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Dual2<float>> wa(a_);					    \
		ConstWideAccessor<float> wb(b_);					        \
		MaskedAccessor<Dual2<float>> wr(r_, Mask(mask_value));		\
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Dual2<float> a = wa[lane];                              \
			float b = wb[lane];                                     \
			Dual2<float> r = dualfunc(a,Dual2<float>(b));           \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16vw16vw16v (void *r_, void *a_, void *b_)            \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Vec3> wa(a_);						        \
		ConstWideAccessor<Vec3> wb(b_);						        \
		WideAccessor<Vec3> wr(r_);								    \
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Vec3 a = wa[lane];                                      \
			Vec3 b = wb[lane];                                      \
			Vec3 r (floatfunc(a.x, b.x),                            \
					floatfunc(a.y, b.y),                            \
				    floatfunc(a.z, b.z));                           \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16vw16vw16v_masked (void *r_, void *a_, void *b_, int mask_value)            \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Vec3> wa(a_);						        \
		ConstWideAccessor<Vec3> wb(b_);						        \
		MaskedAccessor<Vec3> wr(r_, Mask(mask_value));				\
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Vec3 a = wa[lane];                                      \
			Vec3 b = wb[lane];                                      \
			Vec3 r (floatfunc(a.x, b.x),                            \
					floatfunc(a.y, b.y),                            \
				    floatfunc(a.z, b.z));                           \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16dvw16dvw16dv (void *r_, void *a_, void *b_)         \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Dual2<Vec3>> wa(a_);					    \
		ConstWideAccessor<Dual2<Vec3>> wb(b_);					    \
		WideAccessor<Dual2<Vec3>> wr(r_);							\
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Dual2<Vec3> a = wa[lane];                                \
			Dual2<Vec3> b = wb[lane];                                \
		    /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */   \
		    Dual2<float> ax = dualfunc (Dual2<float> (a.val().x, a.dx().x, a.dy().x),    \
		                                Dual2<float> (b.val().x, b.dx().x, b.dy().x));   \
		    Dual2<float> ay = dualfunc (Dual2<float> (a.val().y, a.dx().y, a.dy().y),    \
		                                Dual2<float> (b.val().y, b.dx().y, b.dy().y));   \
		    Dual2<float> az = dualfunc (Dual2<float> (a.val().z, a.dx().z, a.dy().z),    \
		                                Dual2<float> (b.val().z, b.dx().z, b.dy().z));   \
			/* Now swizzle back */                                  \
			Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),     \
			               Vec3( ax.dx(),  ay.dx(),  az.dx() ),     \
			               Vec3( ax.dy(),  ay.dy(),  az.dy() ));    \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16dvw16dvw16dv_masked (void *r_, void *a_, void *b_, int mask_value)         \
{                                                                   \
    OSL_INTEL_PRAGMA(forceinline recursive)                         \
    {                                                               \
        ConstWideAccessor<Dual2<Vec3>> wa(a_);                      \
        ConstWideAccessor<Dual2<Vec3>> wb(b_);                      \
        MaskedAccessor<Dual2<Vec3>> wr(r_, Mask(mask_value));                           \
        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
        for(int lane=0; lane < wr.width; ++lane) {                  \
            Dual2<Vec3> a = wa[lane];                                \
            Dual2<Vec3> b = wb[lane];                                \
            /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */   \
            Dual2<float> ax = dualfunc (Dual2<float> (a.val().x, a.dx().x, a.dy().x),    \
                                        Dual2<float> (b.val().x, b.dx().x, b.dy().x));   \
            Dual2<float> ay = dualfunc (Dual2<float> (a.val().y, a.dx().y, a.dy().y),    \
                                        Dual2<float> (b.val().y, b.dx().y, b.dy().y));   \
            Dual2<float> az = dualfunc (Dual2<float> (a.val().z, a.dx().z, a.dy().z),    \
                                        Dual2<float> (b.val().z, b.dx().z, b.dy().z));   \
            /* Now swizzle back */                                  \
            Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),     \
                           Vec3( ax.dx(),  ay.dx(),  az.dx() ),     \
                           Vec3( ax.dy(),  ay.dy(),  az.dy() ));    \
            wr[lane] = r;                                           \
        }                                                           \
    }                                                               \
}                                                                   \
                                                                    \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16dvw16vw16dv (void *r_, void *a_, void *b_)         \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Vec3> wa(a_);					    	    \
		ConstWideAccessor<Dual2<Vec3>> wb(b_);					    \
		WideAccessor<Dual2<Vec3>> wr(r_);							\
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Dual2<Vec3> a(unproxy(wa[lane]));                       \
			Dual2<Vec3> b = wb[lane];                               \
		    /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */   \
		    Dual2<float> ax = dualfunc (Dual2<float> (a.val().x, a.dx().x, a.dy().x),    \
		                                Dual2<float> (b.val().x, b.dx().x, b.dy().x));   \
		    Dual2<float> ay = dualfunc (Dual2<float> (a.val().y, a.dx().y, a.dy().y),    \
		                                Dual2<float> (b.val().y, b.dx().y, b.dy().y));   \
		    Dual2<float> az = dualfunc (Dual2<float> (a.val().z, a.dx().z, a.dy().z),    \
		                                Dual2<float> (b.val().z, b.dx().z, b.dy().z));   \
			/* Now swizzle back */                                  \
			Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),     \
			               Vec3( ax.dx(),  ay.dx(),  az.dx() ),     \
			               Vec3( ax.dy(),  ay.dy(),  az.dy() ));    \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16dvw16vw16dv_masked (void *r_, void *a_, void *b_, unsigned int mask_value) \
{                                                                   \
    OSL_INTEL_PRAGMA(forceinline recursive)                         \
    {                                                               \
        ConstWideAccessor<Vec3> wa(a_);                             \
        ConstWideAccessor<Dual2<Vec3>> wb(b_);                      \
        MaskedAccessor<Dual2<Vec3>> wr(r_, Mask(mask_value));       \
        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
        for(int lane=0; lane < wr.width; ++lane) {                  \
            Dual2<Vec3> a(unproxy(wa[lane]));                       \
            Dual2<Vec3> b = wb[lane];                               \
            /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */   \
            Dual2<float> ax = dualfunc (Dual2<float> (a.val().x, a.dx().x, a.dy().x),    \
                                        Dual2<float> (b.val().x, b.dx().x, b.dy().x));   \
            Dual2<float> ay = dualfunc (Dual2<float> (a.val().y, a.dx().y, a.dy().y),    \
                                        Dual2<float> (b.val().y, b.dx().y, b.dy().y));   \
            Dual2<float> az = dualfunc (Dual2<float> (a.val().z, a.dx().z, a.dy().z),    \
                                        Dual2<float> (b.val().z, b.dx().z, b.dy().z));   \
            /* Now swizzle back */                                  \
            Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),     \
                           Vec3( ax.dx(),  ay.dx(),  az.dx() ),     \
                           Vec3( ax.dy(),  ay.dy(),  az.dy() ));    \
            wr[lane] = r;                                           \
        }                                                           \
    }                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16dvw16dvw16v (void *r_, void *a_, void *b_)          \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Dual2<Vec3>> wa(a_);					    \
		ConstWideAccessor<Vec3> wb(b_);					    		\
		WideAccessor<Dual2<Vec3>> wr(r_);							\
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Dual2<Vec3> a = wa[lane];                               \
			Dual2<Vec3> b(unproxy(wb[lane]));                       \
		    /* Swizzle the Dual2<Vec3>'s into 3 Dual2<float>'s */   \
		    Dual2<float> ax = dualfunc (Dual2<float> (a.val().x, a.dx().x, a.dy().x),    \
		                                Dual2<float> (b.val().x, b.dx().x, b.dy().x));   \
		    Dual2<float> ay = dualfunc (Dual2<float> (a.val().y, a.dx().y, a.dy().y),    \
		                                Dual2<float> (b.val().y, b.dx().y, b.dy().y));   \
		    Dual2<float> az = dualfunc (Dual2<float> (a.val().z, a.dx().z, a.dy().z),    \
		                                Dual2<float> (b.val().z, b.dx().z, b.dy().z));   \
			/* Now swizzle back */                                  \
			Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),     \
			               Vec3( ax.dx(),  ay.dx(),  az.dx() ),     \
			               Vec3( ax.dy(),  ay.dy(),  az.dy() ));    \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16dvw16dvw16v_masked (void *r_, void *a_, void *b_, unsigned int mask_value) \
{                                                                                          \
    OSL_INTEL_PRAGMA(forceinline recursive)                                                \
    {                                                                                      \
                                                                                           \
    ConstWideAccessor<Dual2<Vec3>> wa(a_);                                                 \
    ConstWideAccessor<Vec3> wb (b_);                                                       \
    MaskedAccessor<Dual2<Vec3>> wr (r_, Mask(mask_value));                                   \
    OSL_OMP_PRAGMA(omp simd simdlen (wr.width))                                            \
    for(int lane=0; lane <wr.width; ++lane){                                               \
        Dual2<Vec3> a = wa[lane];                                                           \
        Dual2<Vec3> b (unproxy (wb[lane]));                                                \
        /*Swizzle the Dual2<Vec3>s into 3 Dual2<float>s */                                 \
        Dual2<float> ax = dualfunc(Dual2<float> (a.val().x, a.dx().x, a.dy().x ),          \
                                   Dual2<float> (b.val().x, b.dx().x, b.dy().x));          \
        Dual2<float> ay = dualfunc(Dual2<float> (a.val().y, a.dx().y, a.dy().y ),          \
                                   Dual2<float> (b.val().y, b.dx().y, b.dy().y));          \
        Dual2<float> az = dualfunc(Dual2<float> (a.val().z, a.dx().z, a.dy().z ),          \
                                   Dual2<float> (b.val().z, b.dx().z, b.dy().z));          \
                                                                                           \
        /*Now swizzle back */                                                              \
        Dual2<Vec3> r (Vec3(ax.val(), ay.val(), az.val()),                               \
                         Vec3(ax.dx(), ay.dx(), az.dx()),                                  \
                         Vec3(ax.dy(), ay.dy(), az.dy()));                                 \
       wr[lane] = r;                                                                       \
    }                                                                                      \
    }                                                                                      \
}                                                                                          \

// Mixed vec func(vec,float)
#define MAKE_WIDE_BINARY_PERCOMPONENT_VF_OP(name,floatfunc,dualfunc)         \
OSL_SHADEOP void                                                    \
osl_##name##_w16vw16vw16f (void *r_, void *a_, void *b_)            \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Vec3> wa(a_);						        \
		ConstWideAccessor<float> wb(b_);						    \
		WideAccessor<Vec3> wr(r_);								    \
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Vec3 a = wa[lane];                                      \
			float b = wb[lane];                                     \
			Vec3 r (floatfunc(a.x, b),                              \
					floatfunc(a.y, b),                              \
				    floatfunc(a.z, b));                             \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
OSL_SHADEOP void                                                    \
osl_##name##_w16vw16vw16f_masked (void *r_, void *a_, void *b_, unsigned int mask_value)            \
{                                                                   \
    OSL_INTEL_PRAGMA(forceinline recursive)                         \
    {                                                               \
        ConstWideAccessor<Vec3> wa(a_);                             \
        ConstWideAccessor<float> wb(b_);                            \
        MaskedAccessor<Vec3> wr(r_, Mask(mask_value));                                  \
        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
        for(int lane=0; lane < wr.width; ++lane) {                  \
            Vec3 a = wa[lane];                                      \
            float b = wb[lane];                                     \
            Vec3 r (floatfunc(a.x, b),                              \
                    floatfunc(a.y, b),                              \
                    floatfunc(a.z, b));                             \
            wr[lane] = r;                                           \
        }                                                           \
    }                                                               \
}                                                                   \
OSL_SHADEOP void                                                    \
osl_##name##_w16dvw16dvw16df (void *r_, void *a_, void *b_)         \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Dual2<Vec3>> wa(a_);					    \
		ConstWideAccessor<Dual2<float>> wb(b_);					    \
		WideAccessor<Dual2<Vec3>> wr(r_);							\
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Dual2<Vec3> a = wa[lane];                               \
			Dual2<float> b = wb[lane];                              \
		    /* Swizzle the Dual2<Vec3> into 3 Dual2<float> */   \
		    Dual2<float> ax = dualfunc (Dual2<float> (a.val().x, a.dx().x, a.dy().x),    \
		                                b);   \
		    Dual2<float> ay = dualfunc (Dual2<float> (a.val().y, a.dx().y, a.dy().y),    \
		                                b);   \
		    Dual2<float> az = dualfunc (Dual2<float> (a.val().z, a.dx().z, a.dy().z),    \
		                                b);   \
			/* Now swizzle back */                                  \
			Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),     \
			               Vec3( ax.dx(),  ay.dx(),  az.dx() ),     \
			               Vec3( ax.dy(),  ay.dy(),  az.dy() ));    \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16dvw16dvw16df_masked (void *r_, void *a_, void *b_, unsigned int mask_value)         \
{                                                                   \
    OSL_INTEL_PRAGMA(forceinline recursive)                         \
{                                                                   \
        ConstWideAccessor<Dual2<Vec3>> wa(a_);                      \
        ConstWideAccessor<Dual2<float>> wb(b_);                     \
        MaskedAccessor<Dual2<Vec3>> wr(r_, Mask(mask_value));                           \
        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
        for(int lane=0; lane < wr.width; ++lane) {                  \
            Dual2<Vec3> a = wa[lane];                               \
            Dual2<float> b = wb[lane];                              \
            /* Swizzle the Dual2<Vec3> into 3 Dual2<float> */   \
            Dual2<float> ax = dualfunc (Dual2<float> (a.val().x, a.dx().x, a.dy().x),    \
                            b);   \
            Dual2<float> ay = dualfunc (Dual2<float> (a.val().y, a.dx().y, a.dy().y),    \
                            b);   \
            Dual2<float> az = dualfunc (Dual2<float> (a.val().z, a.dx().z, a.dy().z),    \
                            b);   \
            /* Now swizzle back */                                  \
            Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),     \
                  Vec3( ax.dx(),  ay.dx(),  az.dx() ),              \
                  Vec3( ax.dy(),  ay.dy(),  az.dy() ));             \
            wr[lane] = r;                                           \
        }                                                           \
     }                                                              \
 }                                                                  \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16dvw16vw16df (void *r_, void *a_, void *b_)         	\
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
	ConstWideAccessor<Vec3> wa(a_);					    			\
		ConstWideAccessor<Dual2<float>> wb(b_);					    \
		WideAccessor<Dual2<Vec3>> wr(r_);							\
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Vec3 a = wa[lane];                               		\
			Dual2<Vec3> da(a);										\
			Dual2<float> b = wb[lane];                              \
			/* Swizzle the Dual2<Vec3> into 3 Dual2<float> */   	\
			Dual2<float> ax = dualfunc (Dual2<float> (da.val().x, da.dx().x, da.dy().x),    \
										b);   \
			Dual2<float> ay = dualfunc (Dual2<float> (da.val().y, da.dx().y, da.dy().y),    \
										b);   \
			Dual2<float> az = dualfunc (Dual2<float> (da.val().z, da.dx().z, da.dy().z),    \
										b);   \
			/* Now swizzle back */                                  \
			Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),     \
						   Vec3( ax.dx(),  ay.dx(),  az.dx() ),     \
						   Vec3( ax.dy(),  ay.dy(),  az.dy() ));    \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
OSL_SHADEOP void                                                    \
osl_##name##_w16dvw16vw16df_masked (void *r_, void *a_, void *b_, unsigned int mask_value)          \
{                                                                   \
    OSL_INTEL_PRAGMA(forceinline recursive)                         \
    {                                                               \
    ConstWideAccessor<Vec3> wa(a_);                                 \
        ConstWideAccessor<Dual2<float>> wb(b_);                     \
        MaskedAccessor<Dual2<Vec3>> wr(r_, Mask(mask_value));       \
        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
        for(int lane=0; lane < wr.width; ++lane) {                  \
            Vec3 a = wa[lane];                                      \
            Dual2<Vec3> da(a);                                      \
            Dual2<float> b = wb[lane];                              \
            /* Swizzle the Dual2<Vec3> into 3 Dual2<float> */       \
            Dual2<float> ax = dualfunc (Dual2<float> (da.val().x, da.dx().x, da.dy().x),    \
                                        b);   \
            Dual2<float> ay = dualfunc (Dual2<float> (da.val().y, da.dx().y, da.dy().y),    \
                                        b);   \
            Dual2<float> az = dualfunc (Dual2<float> (da.val().z, da.dx().z, da.dy().z),    \
                                        b);   \
            /* Now swizzle back */                                  \
            Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),     \
                           Vec3( ax.dx(),  ay.dx(),  az.dx() ),     \
                           Vec3( ax.dy(),  ay.dy(),  az.dy() ));    \
            wr[lane] = r;                                           \
        }                                                           \
    }                                                               \
}                                                                   \
																	\
OSL_SHADEOP void                                                    \
osl_##name##_w16dvw16dvw16f (void *r_, void *a_, void *b_)          \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Dual2<Vec3>> wa(a_);					    \
		ConstWideAccessor<float> wb(b_);					        \
		WideAccessor<Dual2<Vec3>> wr(r_);							\
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Dual2<Vec3> a = wa[lane];                               \
			float b = wb[lane];                                     \
			Dual2<float> db(b);										\
		    /* Swizzle the Dual2<Vec3> into 3 Dual2<float> */       \
		    Dual2<float> ax = dualfunc (Dual2<float> (a.val().x, a.dx().x, a.dy().x),    \
		                                db);                        \
		    Dual2<float> ay = dualfunc (Dual2<float> (a.val().y, a.dx().y, a.dy().y),    \
		                                db);                        \
		    Dual2<float> az = dualfunc (Dual2<float> (a.val().z, a.dx().z, a.dy().z),    \
		                                db);                        \
			/* Now swizzle back */                                  \
			Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),     \
			               Vec3( ax.dx(),  ay.dx(),  az.dx() ),     \
			               Vec3( ax.dy(),  ay.dy(),  az.dy() ));    \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
OSL_SHADEOP void                                                    \
osl_##name##_w16dvw16dvw16f_masked (void *r_, void *a_, void *b_,unsigned int mask_value )          \
{                                                                   \
    OSL_INTEL_PRAGMA(forceinline recursive)                         \
    {                                                               \
        ConstWideAccessor<Dual2<Vec3>> wa(a_);                      \
        ConstWideAccessor<float> wb(b_);                            \
        MaskedAccessor<Dual2<Vec3>> wr(r_, Mask(mask_value));         \
        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
        for(int lane=0; lane < wr.width; ++lane) {                  \
            Dual2<Vec3> a = wa[lane];                               \
            float b = wb[lane];                                     \
            Dual2<float> db(b);                                     \
            /* Swizzle the Dual2<Vec3> into 3 Dual2<float> */       \
            Dual2<float> ax = dualfunc (Dual2<float> (a.val().x, a.dx().x, a.dy().x),    \
                                        db);                        \
            Dual2<float> ay = dualfunc (Dual2<float> (a.val().y, a.dx().y, a.dy().y),    \
                                        db);                        \
            Dual2<float> az = dualfunc (Dual2<float> (a.val().z, a.dx().z, a.dy().z),    \
                                        db);                        \
            /* Now swizzle back */                                  \
            Dual2<Vec3> r (Vec3( ax.val(), ay.val(), az.val()),     \
                           Vec3( ax.dx(),  ay.dx(),  az.dx() ),     \
                           Vec3( ax.dy(),  ay.dy(),  az.dy() ));    \
            wr[lane] = r;                                           \
        }                                                           \
    }                                                               \
}                                                                   \



#define MAKE_WIDE_BINARY_PERCOMPONENT_F_OR_V_OP(name,floatfunc)   \
OSL_SHADEOP void                                                    \
osl_##name##_w16fw16fw16f (void *r_, void *a_, void *b_)            \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<float> wa(a_);						    \
		ConstWideAccessor<float> wb(b_);						    \
		WideAccessor<float> wr(r_);								    \
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			float a = wa[lane];                                     \
			float b = wb[lane];                                     \
			float r = floatfunc(a,b);                               \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}      																\
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16fw16fw16f_masked (void *r_, void *a_, void *b_, int mask_value)     \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<float> wa(a_);						    \
		ConstWideAccessor<float> wb(b_);						    \
		MaskedAccessor<float> wr(r_, Mask(mask_value));             \
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			float a = wa[lane];                                     \
			float b = wb[lane];                                     \
			float r = floatfunc(a,b);                               \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}      																\
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16vw16vw16v (void *r_, void *a_, void *b_)            \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Vec3> wa(a_);						        \
		ConstWideAccessor<Vec3> wb(b_);						        \
		WideAccessor<Vec3> wr(r_);								    \
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Vec3 a = wa[lane];                                      \
			Vec3 b = wb[lane];                                      \
			Vec3 r (floatfunc(a.x, b.x),                            \
					floatfunc(a.y, b.y),                            \
				    floatfunc(a.z, b.z));                           \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
	                                                                \
OSL_SHADEOP void                                                    \
osl_##name##_w16vw16vw16v_masked (void *r_, void *a_, void *b_, int mask_value) \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Vec3> wa(a_);						        \
		ConstWideAccessor<Vec3> wb(b_);						        \
		MaskedAccessor<Vec3> wr(r_, Mask(mask_value));				\
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Vec3 a = wa[lane];                                      \
			Vec3 b = wb[lane];                                      \
			Vec3 r (floatfunc(a.x, b.x),                            \
					floatfunc(a.y, b.y),                            \
				    floatfunc(a.z, b.z));                           \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \


#define MAKE_WIDE_UNARY_I_OP(name,intfunc)         \
OSL_SHADEOP void                                                    \
osl_##name##_w16iw16i (void *r_, void *val_)                        \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<int> wval(val_);						\
		WideAccessor<int> wr(r_);								    \
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			int val = wval[lane];                                 \
			int r = intfunc(val);                               \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
\
OSL_SHADEOP void                                                    \
osl_##name##_w16iw16i_masked (void *r_, void *val_, unsigned int mask_value)\
{                                                                   \
    OSL_INTEL_PRAGMA(forceinline recursive)                         \
    {                                                               \
        ConstWideAccessor<int> wval(val_);                      \
        MaskedAccessor<int> wr(r_, Mask(mask_value));                                   \
        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
        for(int lane=0; lane < wr.width; ++lane) {                  \
            int val = wval[lane];                                 \
            int r = intfunc(val);                               \
            wr[lane] = r;                                           \
        }                                                           \
    }                                                               \
}                                                                   \

#define MAKE_WIDE_BINARY_FI_OP(name,intfunc,floatfunc)              \
OSL_SHADEOP void                                                    \
osl_##name##_w16iw16iw16i (void *r_, void *a_, void *b_)            \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<int> wa(a_);						\
		ConstWideAccessor<int> wb(b_);						\
		WideAccessor<int> wr(r_);								    \
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			int a = wa[lane];                                 \
			int b = wb[lane];                                 \
			int r = intfunc(a,b);                               \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
\
OSL_SHADEOP void                                                    \
osl_##name##_w16fw16fw16f (void *r_, void *a_, void *b_)            \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<float> wa(a_);							\
		ConstWideAccessor<float> wb(b_);							\
		WideAccessor<float> wr(r_);								    \
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			float a = wa[lane];                                 	\
			float b = wb[lane];                                 	\
			float r = intfunc(a,b);                               	\
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}



#define MAKE_WIDE_TEST_F_OP(name,test_floatfunc)         \
OSL_SHADEOP void                                                    \
osl_##name##_w16iw16f (void *r_, void *val_)                        \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<float> wval(val_);						\
		WideAccessor<int> wr(r_);								    \
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			float val = wval[lane];                                 \
			int r = test_floatfunc(val);                            \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16iw16f_masked (void *r_, void *val_, unsigned int mask_value)\
{                                                                   \
    OSL_INTEL_PRAGMA(forceinline recursive)                         \
    {                                                               \
        ConstWideAccessor<float> wval(val_);                        \
        MaskedAccessor<int> wr(r_, Mask(mask_value));               \
        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
        for(int lane=0; lane < wr.width; ++lane) {                  \
            float val = wval[lane];                                 \
            int r = test_floatfunc(val);                            \
            wr[lane] = r;                                           \
        }                                                           \
    }                                                               \
}                                                                   \


#if OSL_FAST_MATH
// OIIO::fast_sin & OIIO::fast_cos are not vectorizing (presumably madd is interfering)
// so use regular sin which compiler should replace with its own fast version
//MAKE_WIDE_UNARY_PERCOMPONENT_OP (sin  , OIIO::fast_sin  , OSL::fast_sin )
MAKE_WIDE_UNARY_PERCOMPONENT_OP (sin  , sfm::sin      , OSL::sin  )
//MAKE_WIDE_UNARY_PERCOMPONENT_OP (cos  , OIIO::fast_cos  , OSL::fast_cos )
MAKE_WIDE_UNARY_PERCOMPONENT_OP (cos  , sfm::cos      , OSL::cos  )
//MAKE_WIDE_UNARY_PERCOMPONENT_OP (tan  , OIIO::fast_tan  , OSL::fast_tan )
MAKE_WIDE_UNARY_PERCOMPONENT_OP (tan  , sfm::tan  , sfm::tan )
MAKE_WIDE_UNARY_PERCOMPONENT_OP (asin , OIIO::fast_asin , OSL::fast_asin)
MAKE_WIDE_UNARY_PERCOMPONENT_OP (acos , OIIO::fast_acos , OSL::fast_acos)
//MAKE_WIDE_UNARY_PERCOMPONENT_OP (atan , OIIO::fast_atan , OSL::fast_atan)
MAKE_WIDE_UNARY_PERCOMPONENT_OP (atan , sfm::atan , sfm::atan)
//MAKE_WIDE_BINARY_PERCOMPONENT_OP(atan2, OIIO::fast_atan2, OSL::fast_atan2)
MAKE_WIDE_BINARY_PERCOMPONENT_OP(atan2, sfm::atan2, sfm::atan2)
//MAKE_WIDE_UNARY_PERCOMPONENT_OP (sinh , OIIO::fast_sinh , OSL::fast_sinh)
MAKE_WIDE_UNARY_PERCOMPONENT_OP (sinh , sfm::sinh , sfm::sinh)
//MAKE_WIDE_UNARY_PERCOMPONENT_OP (cosh , OIIO::fast_cosh , OSL::fast_cosh)
MAKE_WIDE_UNARY_PERCOMPONENT_OP (cosh , sfm::cosh , sfm::cosh)
//MAKE_WIDE_UNARY_PERCOMPONENT_OP (tanh , OIIO::fast_tanh , OSL::fast_tanh)
MAKE_WIDE_UNARY_PERCOMPONENT_OP (tanh , sfm::tanh , sfm::tanh)
#else
// try it out and compare performance, maybe compile time flag
MAKE_WIDE_UNARY_PERCOMPONENT_OP (sin  , sinf      , OSL::sin  )
MAKE_WIDE_UNARY_PERCOMPONENT_OP (cos  , cosf      , OSL::cos  )
MAKE_WIDE_UNARY_PERCOMPONENT_OP (tan  , tanf      , OSL::tan  )
MAKE_WIDE_UNARY_PERCOMPONENT_OP (asin , safe_asin , OSL::safe_asin )
MAKE_WIDE_UNARY_PERCOMPONENT_OP (acos , safe_acos , OSL::safe_acos )
MAKE_WIDE_UNARY_PERCOMPONENT_OP (atan , atanf     , OSL::atan )
MAKE_WIDE_BINARY_PERCOMPONENT_OP(atan2, atan2f    , atan2)
MAKE_WIDE_UNARY_PERCOMPONENT_OP (sinh , sinhf     , OSL::sinh )
MAKE_WIDE_UNARY_PERCOMPONENT_OP (cosh , coshf     , OSL::cosh )
MAKE_WIDE_UNARY_PERCOMPONENT_OP (tanh , tanhf     , OSL::tanh )
#endif

#if OSL_FAST_MATH
//MAKE_WIDE_UNARY_PERCOMPONENT_OP     (log        , OIIO::fast_log       , fast_log)
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (log        , sfm::log       , sfm::log)
//MAKE_WIDE_UNARY_PERCOMPONENT_OP     (log2       , OIIO::fast_log2      , fast_log2)
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (log2       , sfm::log2    , sfm::log2)
//MAKE_WIDE_UNARY_PERCOMPONENT_OP     (log10      , OIIO::fast_log10     , fast_log10)
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (log10      , sfm::log10     , sfm::log10)
// OIIO::fast_sin & OIIO::fast_cos are not vectorizing (presumably madd is interfering)
// so use regular sin which compiler should replace with its own fast version
//MAKE_WIDE_UNARY_PERCOMPONENT_OP     (exp        , OIIO::fast_exp       , fast_exp)
//MAKE_WIDE_UNARY_PERCOMPONENT_OP     (exp2       , OIIO::fast_exp2      , fast_exp2)
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (exp        , sfm::exp     , sfm::exp)
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (exp2       , sfm::exp2    , sfm::exp2)
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (expm1      , sfm::expm1   , sfm::expm1)
//MAKE_WIDE_BINARY_PERCOMPONENT_OP    (pow        , OIIO::fast_safe_pow  , fast_safe_pow)
MAKE_WIDE_BINARY_PERCOMPONENT_OP    (pow        , sfm::safe_pow  , sfm::safe_pow)
MAKE_WIDE_BINARY_PERCOMPONENT_VF_OP (pow        , sfm::safe_pow  , sfm::safe_pow)
//MAKE_WIDE_UNARY_F_OP     (erf        , OIIO::fast_erf       , erf)
MAKE_WIDE_UNARY_F_OP     (erf        , sfm::erf   , sfm::erf)
//MAKE_WIDE_UNARY_F_OP     (erfc       , OIIO::fast_erfc      , erfc)
MAKE_WIDE_UNARY_F_OP     (erfc       , sfm::erfc  , sfm::erfc)
#else
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (log        , OIIO::safe_log       , safe_log)
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (log2       , OIIO::safe_log2      , safe_log2)
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (log10      , OIIO::safe_log10     , safe_log10)
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (exp        , expf                 , exp)
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (exp2       , exp2f                , exp2)
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (expm1      , expm1f               , expm1)
MAKE_WIDE_BINARY_PERCOMPONENT_OP    (pow        , OIIO::safe_pow       , safe_pow)
//MAKE_BINARY_PERCOMPONENT_VF_OP (pow        , OIIO::safe_pow       , safe_pow)
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (erf        , erff                 , erf)
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (erfc       , erfcf                , erfc)
#endif
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (sqrt       , OIIO::safe_sqrt      , sqrt)
MAKE_WIDE_UNARY_PERCOMPONENT_OP     (inversesqrt, OIIO::safe_inversesqrt, inversesqrt)

// emitted directly by llvm_gen_wide.cpp
//MAKE_WIDE_BINARY_FI_OP(safe_div, sfm::safe_div, sfm::safe_div)

#define MAKE_WIDE_UNARY_F_OR_V_OP(name,floatfunc)           \
OSL_SHADEOP void                                                    \
osl_##name##_w16fw16f (void *r_, void *val_)                        \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<float> wval(val_);						\
		WideAccessor<float> wr(r_);								    \
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			float val = wval[lane];                                 \
			float r = floatfunc(val);                               \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16fw16f_masked (void *r_, void *val_, int mask_value) \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<float> wval(val_);						\
		MaskedAccessor<float> wr(r_, Mask(mask_value));             \
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			float val = wval[lane];                                 \
			float r = floatfunc(val);                               \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
                                                                    \
OSL_SHADEOP void                                                    \
osl_##name##_w16vw16v (void *r_, void *val_)                        \
{                                                                   \
	OSL_INTEL_PRAGMA(forceinline recursive)							\
	{																\
		ConstWideAccessor<Vec3> wval(val_);						    \
		WideAccessor<Vec3> wr(r_);								    \
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
		for(int lane=0; lane < wr.width; ++lane) {                  \
			Vec3 val = wval[lane];                                  \
			Vec3 r (floatfunc(val.x),                               \
					floatfunc(val.y),                               \
				    floatfunc(val.z));                              \
			wr[lane] = r;                                           \
		}                                                           \
	}                                                               \
}                                                                   \
OSL_SHADEOP void                                                    \
osl_##name##_w16vw16v_masked (void *r_, void *val_, int mask_value) \
{                                                                   \
    OSL_INTEL_PRAGMA(forceinline recursive)                         \
    {                                                               \
        ConstWideAccessor<Vec3> wval(val_);                         \
        MaskedAccessor<Vec3> wr(r_, Mask(mask_value));              \
        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))                  \
        for(int lane=0; lane < wr.width; ++lane) {                  \
            Vec3 val = wval[lane];                                  \
            Vec3 r (floatfunc(val.x),                               \
                    floatfunc(val.y),                               \
                    floatfunc(val.z));                              \
            wr[lane] = r;                                           \
        }                                                           \
    }                                                               \
}

MAKE_WIDE_UNARY_F_OR_V_OP (logb, OIIO::fast_logb)
MAKE_WIDE_UNARY_F_OR_V_OP (floor, floorf)
MAKE_WIDE_UNARY_F_OR_V_OP (ceil, ceilf)
MAKE_WIDE_UNARY_F_OR_V_OP (trunc, truncf)
MAKE_WIDE_UNARY_F_OR_V_OP (round, roundf)

static OSL_INLINE float impl_sign (float x) {
    return x < 0.0f ? -1.0f : (x==0.0f ? 0.0f : 1.0f);
}
MAKE_WIDE_UNARY_F_OR_V_OP (sign, impl_sign)


MAKE_WIDE_UNARY_PERCOMPONENT_OP (abs, sfm::absf, sfm::absf);
MAKE_WIDE_UNARY_PERCOMPONENT_OP (fabs, sfm::absf, sfm::absf);
MAKE_WIDE_UNARY_I_OP(abs, sfm::absi);
MAKE_WIDE_UNARY_I_OP(fabs, sfm::absi);

static OSL_INLINE float impl_step (float edge, float x) {
    return x < edge ? 0.0f : 1.0f;
}
// TODO: consider moving step to stdosl.h
MAKE_WIDE_BINARY_PERCOMPONENT_F_OR_V_OP(step, impl_step)


MAKE_WIDE_TEST_F_OP(isnan,OIIO::isnan)
MAKE_WIDE_TEST_F_OP(isinf,OIIO::isinf)
MAKE_WIDE_TEST_F_OP(isfinite,OIIO::isfinite)

static OSL_INLINE void impl_sincos (float theta, float &rsine, float &rcosine) {
#if OSL_FAST_MATH
	//OIIO::fast_sincos(theta, &rsine, &rcosine);
	sfm::sincos(theta, rsine, rcosine);
#else
	OIIO::sincos(theta, &rsine, &rcosine);
#endif
}

static OSL_INLINE void impl_sincos (Vec3 theta, Vec3 &rsine, Vec3 &rcosine) {
	impl_sincos(theta.x, rsine.x, rcosine.x);
	impl_sincos(theta.y, rsine.y, rcosine.y);
	impl_sincos(theta.z, rsine.z, rcosine.z);
}


OSL_SHADEOP void
osl_sincos_w16fw16fw16f (void *theta_, void *rsine_, void *rcosine_)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
		ConstWideAccessor<float> wtheta(theta_);
        WideAccessor<float> wrsine(rsine_);
        WideAccessor<float> wrcosine(rcosine_);

        OSL_OMP_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
			float theta = wtheta[lane];
			float rsine;
			float rcosine;
			impl_sincos(theta, rsine, rcosine);
			wrsine[lane] = rsine;
			wrcosine[lane] = rcosine;
        }
    }
}

OSL_SHADEOP void
osl_sincos_w16fw16fw16f_masked (void *theta_, void *rsine_, void *rcosine_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<float> wtheta(theta_);
        MaskedAccessor<float> wrsine(rsine_, Mask(mask_value));
        MaskedAccessor<float> wrcosine(rcosine_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
            float theta = wtheta[lane];
            float rsine;
            float rcosine;
            impl_sincos(theta, rsine, rcosine);

            wrsine[lane] = rsine;
            wrcosine[lane] = rcosine;
        }
    }
}

OSL_SHADEOP void
osl_sincos_w16dfw16dfw16f (void *theta_, void *rsine_, void *rcosine_)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
		ConstWideAccessor<Dual2<float>> wtheta(theta_);
        WideAccessor<Dual2<float>> wrsine(rsine_);
        WideAccessor<float> wrcosine(rcosine_);

		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wrsine.width))
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
        	Dual2<float> theta = wtheta[lane];
        	float rsine;
			float rcosine;
			impl_sincos(theta.val(), rsine, rcosine);
			wrsine[lane] = Dual2<float>(rsine,  rcosine * theta.dx(),  rcosine * theta.dy());
			wrcosine[lane] = rcosine;
        }
    }
}

OSL_SHADEOP void
osl_sincos_w16dfw16dfw16f_masked (void *theta_, void *rsine_, void *rcosine_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Dual2<float>> wtheta(theta_);
        MaskedAccessor<Dual2<float>> wrsine(rsine_, Mask(mask_value));
        MaskedAccessor<float> wrcosine(rcosine_, Mask(mask_value));

        OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wrsine.width))
        OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
            Dual2<float> theta = wtheta[lane];
            float rsine;
            float rcosine;
            impl_sincos(theta.val(), rsine, rcosine);
            wrsine[lane] = Dual2<float>(rsine,  rcosine * theta.dx(),  rcosine * theta.dy());
            wrcosine[lane] = rcosine;
        }
    }
}

OSL_SHADEOP void
osl_sincos_w16dfw16fw16df (void *theta_, void *rsine_, void *rcosine_)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
		ConstWideAccessor<Dual2<float>> wtheta(theta_);
        WideAccessor<float> wrsine(rsine_);
        WideAccessor<Dual2<float>> wrcosine(rcosine_);

		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wrsine.width))
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
        	Dual2<float> theta = wtheta[lane];
        	float rsine;
			float rcosine;
			impl_sincos(theta.val(), rsine, rcosine);
			wrsine[lane] = rsine;
			wrcosine[lane] = Dual2<float>(rcosine,  -rsine * theta.dx(), -rsine * theta.dy());
        }
    }
}

OSL_SHADEOP void
osl_sincos_w16dfw16fw16df_masked (void *theta_, void *rsine_, void *rcosine_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Dual2<float>> wtheta(theta_);
        MaskedAccessor<float> wrsine(rsine_, Mask(mask_value));
        MaskedAccessor<Dual2<float>> wrcosine(rcosine_, Mask(mask_value));

        OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wrsine.width))
        OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
            Dual2<float> theta = wtheta[lane];
            float rsine;
            float rcosine;
            impl_sincos(theta.val(), rsine, rcosine);
            wrsine[lane] = rsine;
            wrcosine[lane] = Dual2<float>(rcosine,  -rsine * theta.dx(), -rsine * theta.dy());
        }
    }
}

OSL_SHADEOP void
osl_sincos_w16dfw16dfw16df (void *theta_, void *rsine_, void *rcosine_)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
		ConstWideAccessor<Dual2<float>> wtheta(theta_);
        WideAccessor<Dual2<float>> wrsine(rsine_);
        WideAccessor<Dual2<float>> wrcosine(rcosine_);

		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wrsine.width))
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
        	Dual2<float> theta = wtheta[lane];
        	float rsine;
			float rcosine;
			impl_sincos(theta.val(), rsine, rcosine);
			wrsine[lane] = Dual2<float>(rsine,  rcosine * theta.dx(),  rcosine * theta.dy());
			wrcosine[lane] = Dual2<float>(rcosine,  -rsine * theta.dx(), -rsine * theta.dy());
        }
    }
}

OSL_SHADEOP void
osl_sincos_w16dfw16dfw16df_masked (void *theta_, void *rsine_, void *rcosine_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Dual2<float>> wtheta(theta_);
        MaskedAccessor<Dual2<float>> wrsine(rsine_, Mask(mask_value));
        MaskedAccessor<Dual2<float>> wrcosine(rcosine_, Mask (mask_value));

        OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wrsine.width))
        OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
            Dual2<float> theta = wtheta[lane];
            float rsine;
            float rcosine;
            impl_sincos(theta.val(), rsine, rcosine);
            wrsine[lane] = Dual2<float>(rsine,  rcosine * theta.dx(),  rcosine * theta.dy());
            wrcosine[lane] = Dual2<float>(rcosine,  -rsine * theta.dx(), -rsine * theta.dy());
        }
    }
}


OSL_SHADEOP void
osl_sincos_w16vw16vw16v (void *theta_, void *rsine_, void *rcosine_)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
		ConstWideAccessor<Vec3> wtheta(theta_);
        WideAccessor<Vec3> wrsine(rsine_);
        WideAccessor<Vec3> wrcosine(rcosine_);

        OSL_OMP_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
        	Vec3 theta = wtheta[lane];
        	Vec3 rsine;
        	Vec3 rcosine;
        	impl_sincos(theta, rsine, rcosine);
			wrsine[lane] = rsine;
			wrcosine[lane] = rcosine;
        }
    }
}

OSL_SHADEOP void
osl_sincos_w16vw16vw16v_masked (void *theta_, void *rsine_, void *rcosine_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Vec3> wtheta(theta_);
        MaskedAccessor<Vec3> wrsine(rsine_, Mask(mask_value));
        MaskedAccessor<Vec3> wrcosine(rcosine_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
            Vec3 theta = wtheta[lane];
            Vec3 rsine;
            Vec3 rcosine;
            impl_sincos(theta, rsine, rcosine);
            wrsine[lane] = rsine;
            wrcosine[lane] = rcosine;
        }
    }
}

OSL_SHADEOP void
osl_sincos_w16dvw16dvw16v (void *theta_, void *rsine_, void *rcosine_)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
		ConstWideAccessor<Dual2<Vec3>> wtheta(theta_);
        WideAccessor<Dual2<Vec3>> wrsine(rsine_);
        WideAccessor<Vec3> wrcosine(rcosine_);

		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wrsine.width))
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
        	Dual2<Vec3> theta = wtheta[lane];
        	Vec3 rsine;
        	Vec3 rcosine;
        	impl_sincos(theta.val(), rsine, rcosine);
			wrsine[lane] = Dual2<Vec3>(rsine,  rcosine * theta.dx(),  rcosine * theta.dy());;
			wrcosine[lane] = rcosine;
        }
    }
}


OSL_SHADEOP void
osl_sincos_w16dvw16dvw16v_masked (void *theta_, void *rsine_, void *rcosine_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Dual2<Vec3>> wtheta(theta_);
        MaskedAccessor<Dual2<Vec3>> wrsine(rsine_, Mask(mask_value));
        MaskedAccessor<Vec3> wrcosine(rcosine_, Mask(mask_value));

        OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wrsine.width))
        OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
            Dual2<Vec3> theta = wtheta[lane];
            Vec3 rsine;
            Vec3 rcosine;
            impl_sincos(theta.val(), rsine, rcosine);
            wrsine[lane] = Dual2<Vec3>(rsine,  rcosine * theta.dx(),  rcosine * theta.dy());;
            wrcosine[lane] = rcosine;
        }
    }
}

OSL_SHADEOP void
osl_sincos_w16dvw16vw16dv (void *theta_, void *rsine_, void *rcosine_)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
		ConstWideAccessor<Dual2<Vec3>> wtheta(theta_);
        WideAccessor<Vec3> wrsine(rsine_);
        WideAccessor<Dual2<Vec3>> wrcosine(rcosine_);

		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wrsine.width))
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
        	Dual2<Vec3> theta = wtheta[lane];
        	Vec3 rsine;
        	Vec3 rcosine;
        	impl_sincos(theta.val(), rsine, rcosine);
			wrsine[lane] = rsine;
			wrcosine[lane] = Dual2<Vec3>(rcosine,  -rsine * theta.dx(), -rsine * theta.dy());

        }
    }
}

OSL_SHADEOP void
osl_sincos_w16dvw16vw16dv_masked (void *theta_, void *rsine_, void *rcosine_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Dual2<Vec3>> wtheta(theta_);
        MaskedAccessor<Vec3> wrsine(rsine_, Mask(mask_value));
        MaskedAccessor<Dual2<Vec3>> wrcosine(rcosine_, Mask(mask_value));

        OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wrsine.width))
        OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
            Dual2<Vec3> theta = wtheta[lane];
            Vec3 rsine;
            Vec3 rcosine;
            impl_sincos(theta.val(), rsine, rcosine);
            wrsine[lane] = rsine;
            wrcosine[lane] = Dual2<Vec3>(rcosine,  -rsine * theta.dx(), -rsine * theta.dy());

        }
    }
}

OSL_SHADEOP void
osl_sincos_w16dvw16dvw16dv (void *theta_, void *rsine_, void *rcosine_)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
		ConstWideAccessor<Dual2<Vec3>> wtheta(theta_);
        WideAccessor<Dual2<Vec3>> wrsine(rsine_);
        WideAccessor<Dual2<Vec3>> wrcosine(rcosine_);

		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wrsine.width))
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
        	Dual2<Vec3> theta = wtheta[lane];
        	Vec3 rsine;
        	Vec3 rcosine;
        	impl_sincos(theta.val(), rsine, rcosine);
			wrsine[lane] = Dual2<Vec3>(rsine,  rcosine * theta.dx(),  rcosine * theta.dy());;
			wrcosine[lane] = Dual2<Vec3>(rcosine,  -rsine * theta.dx(), -rsine * theta.dy());
        }
    }
}

OSL_SHADEOP void
osl_sincos_w16dvw16dvw16dv_masked (void *theta_, void *rsine_, void *rcosine_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Dual2<Vec3>> wtheta(theta_);
        MaskedAccessor<Dual2<Vec3>> wrsine(rsine_, Mask(mask_value));
        MaskedAccessor<Dual2<Vec3>> wrcosine(rcosine_, Mask(mask_value));

        OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wrsine.width))
        OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wrsine.width))
        for(int lane=0; lane < wrsine.width; ++lane) {
            Dual2<Vec3> theta = wtheta[lane];
            Vec3 rsine;
            Vec3 rcosine;
            impl_sincos(theta.val(), rsine, rcosine);
            wrsine[lane] = Dual2<Vec3>(rsine,  rcosine * theta.dx(),  rcosine * theta.dy());;
            wrcosine[lane] = Dual2<Vec3>(rcosine,  -rsine * theta.dx(), -rsine * theta.dy());
        }
    }
}

// NEEDED
#if 1

inline Vec3 calculatenormal(const Dual2<Vec3> &tmpP, bool flipHandedness)
{
    if (flipHandedness)
        return tmpP.dy().cross( tmpP.dx());
    else
        return tmpP.dx().cross( tmpP.dy());
}




OSL_SHADEOP void
osl_length_w16fw16v(void *r_, void *V_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Vec3> wV(V_);
		WideAccessor<float> wr(r_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Vec3 V = wV[lane];
		    float r = simdFriendlyLength(V);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_length_w16fw16v_masked(void *r_, void *V_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Vec3> wV(V_);
		MaskedAccessor<float> wr(r_, Mask(mask_value));

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Vec3 V = wV[lane];
		    float r = simdFriendlyLength(V);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_length_w16dfw16dv(void *r_, void *V_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<Vec3>> wV(V_);
		WideAccessor<Dual2<float>> wr(r_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<Vec3> V = wV[lane];
			Dual2<float> r = length(V);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_length_w16dfw16dv_masked(void *r_, void *V_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<Vec3>> wV(V_);
		MaskedAccessor<Dual2<float>> wr(r_, Mask(mask_value));

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<Vec3> V = wV[lane];
			Dual2<float> r = length(V);
			wr[lane] = r;
		}
	}
}


OSL_SHADEOP void
osl_area_w16(void *r_, void *DP_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<Vec3>> wDP(DP_);
	    
		WideAccessor<float> wr(r_);
	
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<Vec3> DP = wDP[lane];
			
		    Vec3 N = calculatenormal(DP, false);
		    //float r = N.length();
		    float r = simdFriendlyLength(N);
			wr[lane] = r;
		}
	}	
}

OSL_SHADEOP void
osl_area_w16_masked(void *r_, void *DP_, unsigned int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<Vec3>> wDP(DP_);
	    
		MaskedAccessor<float> wr(r_, Mask(mask_value));
	
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<Vec3> DP = wDP[lane];
			
		    Vec3 N = calculatenormal(DP, false);
		    //float r = N.length();
		    float r = simdFriendlyLength(N);
			wr[lane] = r;
		}
	}	
}


OSL_SHADEOP void
osl_distance_w16fw16vw16v(void *r_, void *a_, void *b_)
{

	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Vec3> wA(a_);
		ConstWideAccessor<Vec3> wB(b_);
		WideAccessor<float> wr(r_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Vec3 a = wA[lane];
			Vec3 b = wB[lane];

			// TODO: couldn't we just (b-a).length()?
		   float x = a[0] - b[0];
		   float y = a[1] - b[1];
		   float z = a[2] - b[2];
		   float r = sqrtf (x*x + y*y + z*z);
		   wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_distance_w16fw16vw16v_masked(void *r_, void *a_, void *b_, int mask_value)
{

	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Vec3> wA(a_);
		ConstWideAccessor<Vec3> wB(b_);
		MaskedAccessor<float> wr(r_, Mask(mask_value));

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Vec3 a = wA[lane];
			Vec3 b = wB[lane];

			// TODO: couldn't we just (b-a).length()?
		   float x = a[0] - b[0];
		   float y = a[1] - b[1];
		   float z = a[2] - b[2];
		   float r = sqrtf (x*x + y*y + z*z);
		   wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_distance_w16dfw16dvw16v(void *r_, void *a_, void *b_)
{

	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<Vec3>> wA(a_);
		ConstWideAccessor<Vec3> wB(b_);
		WideAccessor<Dual2<float>> wr(r_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {

			Dual2<Vec3> a = wA[lane];
			Vec3 b = wB[lane];

		    Dual2<float> r = distance(a,b);
		    wr[lane] = r;

		}
	}
}

OSL_SHADEOP void
osl_distance_w16dfw16dvw16v_masked(void *r_, void *a_, void *b_, int mask_value)
{

	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<Vec3>> wA(a_);
		ConstWideAccessor<Vec3> wB(b_);
		MaskedAccessor<Dual2<float>> wr(r_, Mask(mask_value));

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {

			Dual2<Vec3> a = wA[lane];
			Vec3 b = wB[lane];

		    Dual2<float> r = distance(a,b);
		    wr[lane] = r;

		}
	}
}

OSL_SHADEOP void
osl_distance_w16dfw16vw16dv(void *r_, void *a_, void *b_)
{

    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Vec3> wA(a_);
        ConstWideAccessor<Dual2<Vec3>> wB(b_);
        WideAccessor<Dual2<float>> wr(r_);

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {

            Vec3 a = wA[lane];
            Dual2<Vec3> b = wB[lane];

            Dual2<float> r = distance(a,b);
            wr[lane] = r;

        }
    }
}

OSL_SHADEOP void
osl_distance_w16dfw16vw16dv_masked(void *r_, void *a_, void *b_, int mask_value)
{

    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Vec3> wA(a_);
        ConstWideAccessor<Dual2<Vec3>> wB(b_);
        MaskedAccessor<Dual2<float>> wr(r_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {

            Vec3 a = wA[lane];
            Dual2<Vec3> b = wB[lane];

            Dual2<float> r = distance(a,b);
            wr[lane] = r;

        }
    }
}

OSL_SHADEOP void
osl_distance_w16dfw16dvw16dv(void *r_, void *a_, void *b_)
{

    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Dual2<Vec3>> wA(a_);
        ConstWideAccessor<Dual2<Vec3>> wB(b_);
        WideAccessor<Dual2<float>> wr(r_);

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {
            Dual2<Vec3> a = wA[lane];
            Dual2<Vec3> b = wB[lane];

            Dual2<float> r = distance(a,b);
            wr[lane] = r;
        }
    }
}

OSL_SHADEOP void
osl_distance_w16dfw16dvw16dv_masked(void *r_, void *a_, void *b_, int mask_value)
{

    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Dual2<Vec3>> wA(a_);
        ConstWideAccessor<Dual2<Vec3>> wB(b_);
        MaskedAccessor<Dual2<float>> wr(r_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {

            Dual2<Vec3> a = wA[lane];
            Dual2<Vec3> b = wB[lane];

            Dual2<float> r = distance(a,b);
            wr[lane] = r;

        }
    }
}




OSL_SHADEOP void
osl_normalize_w16vw16v(void *r_, void *V_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Vec3> wV(V_);
		WideAccessor<Vec3> wr(r_);
	
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Vec3 V = wV[lane];
		    Vec3 N = simdFriendlyNormalize(V);
			wr[lane] = N;
		}
	}	
}

#if OSL_EXPERIMENTAL_NORMALIZE_MASKED
// Experimental code that includes mask inside __builtin_expect to see how it affects code generation
OSL_INLINE float HackedaccessibleTinyLength(const Vec3 &N)
{
    float absX = (N.x >= float (0))? N.x: -N.x;
    float absY = (N.y >= float (0))? N.y: -N.y;
    float absZ = (N.z >= float (0))? N.z: -N.z;

    float max = absX;

    if (max < absY)
    max = absY;

    if (max < absZ)
    max = absZ;

    if (max == float (0))
    return float (0);

    //
    // Do not replace the divisions by max with multiplications by 1/max.
    // Computing 1/max can overflow but the divisions below will always
    // produce results less than or equal to 1.
    //

    absX /= max;
    absY /= max;
    absZ /= max;

    return max * Imath::Math<float>::sqrt (absX * absX + absY * absY + absZ * absZ);
}

OSL_INLINE
float HackedsimdFriendlyLength(const Vec3 &N, bool isLaneActive)
{
    float length2 = N.dot (N);

    if (__builtin_expect(isLaneActive && (length2 < float (2) * Imath::limits<float>::smallest()), 0))
        return HackedaccessibleTinyLength(N);

    return Imath::Math<float>::sqrt (length2);
}


OSL_INLINE Vec3
HackedsimdFriendlyNormalize(const Vec3 &N, bool isLaneActive)
{
    float l = HackedsimdFriendlyLength(N, isLaneActive);

    if (l == float (0))
        return Vec3 (float (0));

    return Vec3 (N.x / l, N.y / l, N.z / l);
}


OSL_SHADEOP void
osl_normalize_w16vw16v_masked(void *r_, void *V_, int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Vec3> wV(V_);
        MaskedAccessor<Vec3> wr(r_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {
            // Hacked by A.W. to see if __builtin_expect behaves better when mask applied to it
            Vec3 V = wV[lane];
//              Vec3 N = simdFriendlyNormalize(V);
                Vec3 N = HackedsimdFriendlyNormalize(V,wr.mask()[lane]);
                wr[lane] = N;
        }
    }
}
#else
OSL_SHADEOP void
osl_normalize_w16vw16v_masked(void *r_, void *V_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Vec3> wV(V_);
		MaskedAccessor<Vec3> wr(r_, Mask(mask_value));

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Vec3 V = wV[lane];
		    Vec3 N = simdFriendlyNormalize(V);
			wr[lane] = N;
		}
	}
}
#endif

OSL_SHADEOP void
osl_normalize_w16dvw16dv(void *r_, void *V_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<Vec3>> wV(V_);
		WideAccessor<Dual2<Vec3>> wr(r_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<Vec3> V = wV[lane];
		    Dual2<Vec3> N = simdFriendlyNormalize(V);
			wr[lane] = N;
		}
	}
}

OSL_SHADEOP void
osl_normalize_w16dvw16dv_masked(void *r_, void *V_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<Vec3>> wV(V_);
		MaskedAccessor<Dual2<Vec3>> wr(r_, Mask(mask_value));

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<Vec3> V = wV[lane];
		    Dual2<Vec3> N = simdFriendlyNormalize(V);
			wr[lane] = N;
		}
	}
}



OSL_SHADEOP void
osl_cross_w16vw16vw16v (void *result_, void *a_, void *b_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Vec3> wA(a_);
		ConstWideAccessor<Vec3> wB(b_);
		WideAccessor<Vec3> wr(result_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Vec3 a = wA[lane];
			Vec3 b = wB[lane];

		    Vec3 r = a.cross(b);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_cross_w16vw16vw16v_masked (void *result_, void *a_, void *b_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Vec3> wA(a_);
		ConstWideAccessor<Vec3> wB(b_);
		MaskedAccessor<Vec3> wr(result_, Mask(mask_value));

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {

			Vec3 a = wA[lane];
			Vec3 b = wB[lane];

		    Vec3 r = a.cross(b);
			wr[lane] = r;

		}
	}
}

OSL_SHADEOP void
osl_cross_w16dvw16dvw16dv (void *result_, void *a_, void *b_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<Vec3>> wA(a_);
		ConstWideAccessor<Dual2<Vec3>> wB(b_);
		WideAccessor<Dual2<Vec3>> wr(result_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<Vec3> a = wA[lane];
			Dual2<Vec3> b = wB[lane];

			Dual2<Vec3> r = cross(a,b);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_cross_w16dvw16dvw16dv_masked (void *result_, void *a_, void *b_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<Vec3>> wA(a_);
		ConstWideAccessor<Dual2<Vec3>> wB(b_);
		MaskedAccessor<Dual2<Vec3>> wr(result_, Mask(mask_value));

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {

			Dual2<Vec3> a = wA[lane];
			Dual2<Vec3> b = wB[lane];

			Dual2<Vec3> r = cross(a,b);
			wr[lane] = r;

		}
	}
}

OSL_SHADEOP void
osl_cross_w16dvw16dvw16v (void *result_, void *a_, void *b_)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Dual2<Vec3>> wA(a_);
        ConstWideAccessor<Vec3> wB(b_);
        WideAccessor<Dual2<Vec3>> wr(result_);

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {
            Dual2<Vec3> a = wA[lane];
            Vec3 b = wB[lane];

            Dual2<Vec3> r = cross(a,b);
            wr[lane] = r;
        }
    }
}

OSL_SHADEOP void
osl_cross_w16dvw16dvw16v_masked (void *result_, void *a_, void *b_, int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Dual2<Vec3>> wA(a_);
        ConstWideAccessor<Vec3> wB(b_);
        MaskedAccessor<Dual2<Vec3>> wr(result_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {

            Dual2<Vec3> a = wA[lane];
            Vec3 b = wB[lane];

            Dual2<Vec3> r = cross(a,b);
            wr[lane] = r;

        }
    }
}

OSL_SHADEOP void
osl_cross_w16dvw16vw16dv (void *result_, void *a_, void *b_)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Vec3> wA(a_);
        ConstWideAccessor<Dual2<Vec3>> wB(b_);
        WideAccessor<Dual2<Vec3>> wr(result_);

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {
            Vec3 a = wA[lane];
            Dual2<Vec3> b = wB[lane];

            Dual2<Vec3> r = cross(a,b);
            wr[lane] = r;
        }
    }
}

OSL_SHADEOP void
osl_cross_w16dvw16vw16dv_masked (void *result_, void *a_, void *b_, int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Vec3> wA(a_);
        ConstWideAccessor<Dual2<Vec3>> wB(b_);
        MaskedAccessor<Dual2<Vec3>> wr(result_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {

            Vec3 a = wA[lane];
            Dual2<Vec3> b = wB[lane];

            Dual2<Vec3> r = cross(a,b);
            wr[lane] = r;

        }
    }
}

OSL_SHADEOP void
osl_dot_w16fw16vw16v (void *result_, void *a_, void *b_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Vec3> wA(a_);
		ConstWideAccessor<Vec3> wB(b_);
		WideAccessor<float> wr(result_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Vec3 a = wA[lane];
			Vec3 b = wB[lane];

		    float r = a.dot(b);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_dot_w16fw16vw16v_masked (void *result_, void *a_, void *b_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Vec3> wA(a_);
		ConstWideAccessor<Vec3> wB(b_);
		MaskedAccessor<float> wr(result_, Mask(mask_value));

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {

			Vec3 a = wA[lane];
			Vec3 b = wB[lane];

		    float r = a.dot(b);
			wr[lane] = r;

		}
	}
}


OSL_SHADEOP void
osl_dot_w16dfw16dvw16dv (void *result_, void *a_, void *b_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<Vec3>> wA(a_);
		ConstWideAccessor<Dual2<Vec3>> wB(b_);
		WideAccessor<Dual2<float>> wr(result_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<Vec3> a = wA[lane];
			Dual2<Vec3> b = wB[lane];

			Dual2<float> r = dot(a, b);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_dot_w16dfw16dvw16dv_masked (void *result_, void *a_, void *b_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<Vec3>> wA(a_);
		ConstWideAccessor<Dual2<Vec3>> wB(b_);
		MaskedAccessor<Dual2<float>> wr(result_, Mask(mask_value));

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {

			Dual2<Vec3> a = wA[lane];
			Dual2<Vec3> b = wB[lane];

			Dual2<float> r = dot(a, b);
			wr[lane] = r;

		}
	}
}


OSL_SHADEOP void
osl_dot_w16dfw16dvw16v (void *result_, void *a_, void *b_)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Dual2<Vec3>> wA(a_);
        ConstWideAccessor<Vec3> wB(b_);
        WideAccessor<Dual2<float>> wr(result_);

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {
            Dual2<Vec3> a = wA[lane];
            Vec3 b = wB[lane];

            Dual2<float> r = dot(a, b);
            wr[lane] = r;
        }
    }
}


OSL_SHADEOP void
osl_dot_w16dfw16dvw16v_masked (void *result_, void *a_, void *b_, int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Dual2<Vec3>> wA(a_);
        ConstWideAccessor<Vec3> wB(b_);
        MaskedAccessor<Dual2<float>> wr(result_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {

            Dual2<Vec3> a = wA[lane];
            Vec3 b = wB[lane];

            Dual2<float> r = dot(a, b);
            wr[lane] = r;

        }
    }
}

OSL_SHADEOP void
osl_dot_w16dfw16vw16dv (void *result_, void *a_, void *b_)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Vec3> wA(a_);
        ConstWideAccessor<Dual2<Vec3>> wB(b_);
        WideAccessor<Dual2<float>> wr(result_);

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {
            Vec3 a = wA[lane];
            Dual2<Vec3> b = wB[lane];

            Dual2<float> r = dot(a, b);
            wr[lane] = r;
        }
    }
}

OSL_SHADEOP void
osl_dot_w16dfw16vw16dv_masked (void *result_, void *a_, void *b_, int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Vec3> wA(a_);
        ConstWideAccessor<Dual2<Vec3>> wB(b_);
        MaskedAccessor<Dual2<float>> wr(result_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {

            Vec3 a = wA[lane];
            Dual2<Vec3> b = wB[lane];

            Dual2<float> r = dot(a, b);
            wr[lane] = r;

        }
    }
}



inline float filter_width(float dx, float dy)
{
    return sqrtf(dx*dx + dy*dy);
}

OSL_SHADEOP void osl_filterwidth_w16fw16df(void *result_, void *x_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<float>> wX(x_);

		WideAccessor<float> wr(result_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
		    Dual2<float> x = wX[lane];
		    wr[lane] = filter_width(x.dx(), x.dy());
		}
	}
}

OSL_SHADEOP void osl_filterwidth_w16fw16df_masked (void *result_, void *x_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Dual2<float>> wX(x_);

        MaskedAccessor<float> wr(result_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {
            Dual2<float> x = wX[lane];
            wr[lane] = filter_width(x.dx(), x.dy());
        }
    }
}

OSL_SHADEOP void osl_filterwidth_w16vw16dv(void *out, void *x_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<Vec3>> wX(x_);

		WideAccessor<Vec3> wr(out);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
		    Dual2<Vec3> x = wX[lane];
		    Vec3 r;
		    r.x = filter_width (x.dx().x, x.dy().x);
		    r.y = filter_width (x.dx().y, x.dy().y);
		    r.z = filter_width (x.dx().z, x.dy().z);

		    wr[lane] = r;
		}
	}
}

OSL_SHADEOP void osl_filterwidth_w16vw16dv_masked(void *out, void *x_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Dual2<Vec3>> wX(x_);

        MaskedAccessor<Vec3> wr(out, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {
            Dual2<Vec3> x = wX[lane];
            Vec3 r;
            r.x = filter_width (x.dx().x, x.dy().x);
            r.y = filter_width (x.dx().y, x.dy().y);
            r.z = filter_width (x.dx().z, x.dy().z);

            wr[lane] = r;
        }
    }
}

OSL_SHADEOP void osl_calculatenormal_batched(void *out, void *sgb_, void *P_)
{
	ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<Vec3>> wP(P_);
		ConstWideAccessor<int> wFlipHandedness(sgb->varyingData().flipHandedness);
		WideAccessor<Vec3> wr(out);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
		    Dual2<Vec3> P = wP[lane];
		    Vec3 N = calculatenormal(P, wFlipHandedness[lane]);
		    wr[lane] = N;
		}
	}
}

OSL_SHADEOP void
osl_smoothstep_w16fw16fw16fw16f (void *r_, void *e0_, void *e1_, void *x_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<float> we0(e0_);
		ConstWideAccessor<float> we1(e1_);
		ConstWideAccessor<float> wx(x_);
		WideAccessor<float> wr(r_);
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			float e0 = we0[lane];
			float e1 = we1[lane];
			float x = wx[lane];
			float r = smoothstep(e0, e1, x);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_smoothstep_w16fw16fw16fw16f_masked (void *r_, void *e0_, void *e1_, void *x_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<float> we0(e0_);
		ConstWideAccessor<float> we1(e1_);
		ConstWideAccessor<float> wx(x_);
		MaskedAccessor<float> wr(r_, Mask(mask_value));
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			float e0 = we0[lane];
			float e1 = we1[lane];
			float x = wx[lane];
			float r = smoothstep(e0, e1, x);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_smoothstep_w16dfw16dfw16dfw16df (void *r_, void *e0_, void *e1_, void *x_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<float>> we0(e0_);
		ConstWideAccessor<Dual2<float>> we1(e1_);
		ConstWideAccessor<Dual2<float>> wx(x_);
		WideAccessor<Dual2<float>> wr(r_);
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<float> e0 = we0[lane];
			Dual2<float> e1 = we1[lane];
			Dual2<float> x = wx[lane];
			Dual2<float> r = smoothstep(e0, e1, x);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_smoothstep_w16dfw16dfw16dfw16df_masked (void *r_, void *e0_, void *e1_, void *x_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<float>> we0(e0_);
		ConstWideAccessor<Dual2<float>> we1(e1_);
		ConstWideAccessor<Dual2<float>> wx(x_);
		MaskedAccessor<Dual2<float>> wr(r_, Mask(mask_value));
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<float> e0 = we0[lane];
			Dual2<float> e1 = we1[lane];
			Dual2<float> x = wx[lane];
			Dual2<float> r = smoothstep(e0, e1, x);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_smoothstep_w16dfw16fw16dfw16df (void *r_, void *e0_, void *e1_, void *x_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<float> we0(e0_);
		ConstWideAccessor<Dual2<float>> we1(e1_);
		ConstWideAccessor<Dual2<float>> wx(x_);
		WideAccessor<Dual2<float>> wr(r_);
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			float e0 = we0[lane];
			Dual2<float> e1 = we1[lane];
			Dual2<float> x = wx[lane];
			Dual2<float> r = smoothstep(Dual2<float>(e0), e1, x);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_smoothstep_w16dfw16fw16dfw16df_masked (void *r_, void *e0_, void *e1_, void *x_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<float> we0(e0_);
		ConstWideAccessor<Dual2<float>> we1(e1_);
		ConstWideAccessor<Dual2<float>> wx(x_);
		MaskedAccessor<Dual2<float>> wr(r_, Mask(mask_value));
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			float e0 = we0[lane];
			Dual2<float> e1 = we1[lane];
			Dual2<float> x = wx[lane];
			Dual2<float> r = smoothstep(Dual2<float>(e0), e1, x);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_smoothstep_w16dfw16dfw16fw16df (void *r_, void *e0_, void *e1_, void *x_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<float>> we0(e0_);
		ConstWideAccessor<float> we1(e1_);
		ConstWideAccessor<Dual2<float>> wx(x_);
		WideAccessor<Dual2<float>> wr(r_);
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<float> e0 = we0[lane];
			float e1 = we1[lane];
			Dual2<float> x = wx[lane];
			Dual2<float> r = smoothstep(e0, Dual2<float>(e1), x);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_smoothstep_w16dfw16dfw16fw16df_masked (void *r_, void *e0_, void *e1_, void *x_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<float>> we0(e0_);
		ConstWideAccessor<float> we1(e1_);
		ConstWideAccessor<Dual2<float>> wx(x_);
		MaskedAccessor<Dual2<float>> wr(r_, Mask(mask_value));
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<float> e0 = we0[lane];
			float e1 = we1[lane];
			Dual2<float> x = wx[lane];
			Dual2<float> r = smoothstep(e0, Dual2<float>(e1), x);
			wr[lane] = r;
		}
	}
}


OSL_SHADEOP void
osl_smoothstep_w16dfw16dfw16dfw16f (void *r_, void *e0_, void *e1_, void *x_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<float>> we0(e0_);
		ConstWideAccessor<Dual2<float>> we1(e1_);
		ConstWideAccessor<float> wx(x_);
		WideAccessor<Dual2<float>> wr(r_);
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<float> e0 = we0[lane];
			Dual2<float> e1 = we1[lane];
			float x = wx[lane];
			Dual2<float> r = smoothstep(e0, e1, Dual2<float>(x));
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_smoothstep_w16dfw16dfw16dfw16f_masked (void *r_, void *e0_, void *e1_, void *x_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<float>> we0(e0_);
		ConstWideAccessor<Dual2<float>> we1(e1_);
		ConstWideAccessor<float> wx(x_);
		MaskedAccessor<Dual2<float>> wr(r_, Mask(mask_value));
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<float> e0 = we0[lane];
			Dual2<float> e1 = we1[lane];
			float x = wx[lane];
			Dual2<float> r = smoothstep(e0, e1, Dual2<float>(x));
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_smoothstep_w16dfw16fw16fw16df (void *r_, void *e0_, void *e1_, void *x_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<float> we0(e0_);
		ConstWideAccessor<float> we1(e1_);
		ConstWideAccessor<Dual2<float>> wx(x_);
		WideAccessor<Dual2<float>> wr(r_);
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			float e0 = we0[lane];
			float e1 = we1[lane];
			Dual2<float> x = wx[lane];
			Dual2<float> r = smoothstep(Dual2<float>(e0), Dual2<float>(e1), x);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_smoothstep_w16dfw16fw16fw16df_masked (void *r_, void *e0_, void *e1_, void *x_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<float> we0(e0_);
		ConstWideAccessor<float> we1(e1_);
		ConstWideAccessor<Dual2<float>> wx(x_);
		MaskedAccessor<Dual2<float>> wr(r_, Mask(mask_value));
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			float e0 = we0[lane];
			float e1 = we1[lane];
			Dual2<float> x = wx[lane];
			Dual2<float> r = smoothstep(Dual2<float>(e0), Dual2<float>(e1), x);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_smoothstep_w16dfw16dfw16fw16f (void *r_, void *e0_, void *e1_, void *x_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<float>> we0(e0_);
		ConstWideAccessor<float> we1(e1_);
		ConstWideAccessor<float> wx(x_);
		WideAccessor<Dual2<float>> wr(r_);
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<float> e0 = we0[lane];
			float e1 = we1[lane];
			float x = wx[lane];
			Dual2<float> r = smoothstep(e0, Dual2<float>(e1), Dual2<float>(x));
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_smoothstep_w16dfw16dfw16fw16f_masked (void *r_, void *e0_, void *e1_, void *x_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<float>> we0(e0_);
		ConstWideAccessor<float> we1(e1_);
		ConstWideAccessor<float> wx(x_);
		MaskedAccessor<Dual2<float>> wr(r_, Mask(mask_value));
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<float> e0 = we0[lane];
			float e1 = we1[lane];
			float x = wx[lane];
			Dual2<float> r = smoothstep(e0, Dual2<float>(e1), Dual2<float>(x));
			wr[lane] = r;
		}
	}
}


OSL_SHADEOP void
osl_smoothstep_w16dfw16fw16dfw16f (void *r_, void *e0_, void *e1_, void *x_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<float> we0(e0_);
		ConstWideAccessor<Dual2<float>> we1(e1_);
		ConstWideAccessor<float> wx(x_);
		WideAccessor<Dual2<float>> wr(r_);
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			float e0 = we0[lane];
			Dual2<float> e1 = we1[lane];
			float x = wx[lane];
			Dual2<float> r = smoothstep(Dual2<float>(e0), e1, Dual2<float>(x));
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_smoothstep_w16dfw16fw16dfw16f_masked (void *r_, void *e0_, void *e1_, void *x_, int mask_value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<float> we0(e0_);
		ConstWideAccessor<Dual2<float>> we1(e1_);
		ConstWideAccessor<float> wx(x_);
		MaskedAccessor<Dual2<float>> wr(r_, Mask(mask_value));
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			float e0 = we0[lane];
			Dual2<float> e1 = we1[lane];
			float x = wx[lane];
			Dual2<float> r = smoothstep(Dual2<float>(e0), e1, Dual2<float>(x));
			wr[lane] = r;
		}
	}
}

// Asked if the raytype includes a bit pattern.
OSL_SHADEOP int
osl_raytype_bit_batched (void *sgb_, int bit)
{
	ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    return (sgb->uniform().raytype & bit) != 0;
}

// Asked if the raytype is a name we can't know until mid-shader.
OSL_SHADEOP int osl_raytype_name_batched (void *sgb_, void *name)
{
	ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    int bit = sgb->uniform().context->shadingsys().raytype_bit (USTR(name));
    return (sgb->uniform().raytype & bit) != 0;
}

#endif

} // namespace pvt
OSL_NAMESPACE_EXIT
