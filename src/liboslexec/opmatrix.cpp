/*
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
/// Shader interpreter implementation of matrix operations.
///
/////////////////////////////////////////////////////////////////////////

#include <OpenImageIO/fmath.h>
#include <OpenImageIO/simd.h>

#include <iostream>
#include <cmath>

#include "oslexec_pvt.h"
#include "OSL/dual.h"
#include "OSL/dual_vec.h"
#include "OSL/wide.h"


OSL_NAMESPACE_ENTER
namespace pvt {


// Matrix ops
OSL_SHADEOP void
osl_mul_mfm (void *r, float a, void * b)
{
    MAT(r) = MAT(b) * a;
}

OSL_SHADEOP void
osl_mul_w16mw16fw16m(void *wr_, void *wa_, void * wb_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<float> wa(wa_);
		ConstWideAccessor<Matrix44> wb(wb_);
		WideAccessor<Matrix44> wr(wr_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			float a = wa[lane];
			Matrix44 b = wb[lane];
			Matrix44 r = b * a;
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_mul_mmf (void *r, void *a, float b)
{
    MAT(r) = MAT(a) * b;
}

OSL_SHADEOP void
osl_mul_w16mw16mw16f(void *wr_, void *wa_, void * wb_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<Matrix44> wa(wa_);
		ConstWideAccessor<float> wb(wb_);
		WideAccessor<Matrix44> wr(wr_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			Matrix44 a = wa[lane];
			float b = wb[lane];
			Matrix44 r = a * b;
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_mul_mmm (void *r, void *a, void *b)
{
    MAT(r) = MAT(a) * MAT(b);
}

OSL_SHADEOP void
osl_mul_w16mw16mw16m(void *wr_, void *wa_, void * wb_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<Matrix44> wa(wa_);
		ConstWideAccessor<Matrix44> wb(wb_);
		WideAccessor<Matrix44> wr(wr_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			Matrix44 a = wa[lane];
			Matrix44 b = wb[lane];
			Matrix44 r = a * b;
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_mul_mff (void *r, float a, float b)
{
    float c = a * b;
    MAT(r) = Matrix44 (c,0.0f,0.0f,0.0f,
    				   0.0f,c,0.0f,0.0f,
			           0.0f,0.0f,c,0.0f,
			           0.0f,0.0f,0.0f,c);
}

OSL_SHADEOP void
osl_mul_w16mw16fw16f(void *wr_, void *wa_, void * wb_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<float> wa(wa_);
		ConstWideAccessor<float> wb(wb_);
		WideAccessor<Matrix44> wr(wr_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			float a = wa[lane];
			float b = wb[lane];
			float c = a * b;
			Matrix44 r(c,0.0f,0.0f,0.0f,
					   0.0f,c,0.0f,0.0f,
					   0.0f,0.0f,c,0.0f,
					   0.0f,0.0f,0.0f,c);
			wr[lane] = r;
		}
	}
}


OSL_SHADEOP void
osl_div_mmm (void *r, void *a, void *b)
{
    MAT(r) = MAT(a) * MAT(b).inverse();
}

OSL_SHADEOP void
osl_div_w16mw16mw16m(void *wr_, void *wa_, void * wb_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<Matrix44> wa(wa_);
		ConstWideAccessor<Matrix44> wb(wb_);
		WideAccessor<Matrix44> wr(wr_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			Matrix44 a = wa[lane];
			Matrix44 b = wb[lane];
			Matrix44 r = a * b.inverse();
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_div_mmf (void *r, void *a, float b)
{
    MAT(r) = MAT(a) * (1.0f/b);
}

OSL_SHADEOP void
osl_div_w16mw16mw16f(void *wr_, void *wa_, void * wb_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<Matrix44> wa(wa_);
		ConstWideAccessor<float> wb(wb_);
		WideAccessor<Matrix44> wr(wr_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			Matrix44 a = wa[lane];
			float b = wb[lane];
			Matrix44 r = a * (1.0f/b);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_div_mfm (void *r, float a, void *b)
{
    MAT(r) = a * MAT(b).inverse();
}


OSL_SHADEOP void
osl_div_w16mw16fw16m(void *wr_, void *wa_, void * wb_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<float> wa(wa_);
		ConstWideAccessor<Matrix44> wb(wb_);
		WideAccessor<Matrix44> wr(wr_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			float a = wa[lane];
			Matrix44 b = wb[lane];
			Matrix44 r = a * b.inverse();
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_div_mff (void *r, float a, float b)
{
    float c = (b == 0) ? 0.0f : (a / b);
    MAT(r) = Matrix44 (c,0.0f,0.0f,0.0f,
			   	       0.0f,c,0.0f,0.0f,
			           0.0f,0.0f,c,0.0f,
			           0.0f,0.0f,0.0f,c);
}

OSL_SHADEOP void
osl_div_w16mw16fw16f(void *wr_, void *wa_, void * wb_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<float> wa(wa_);
		ConstWideAccessor<float> wb(wb_);
		WideAccessor<Matrix44> wr(wr_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			float a = wa[lane];
			float b = wb[lane];
		    float c = (b == 0) ? 0.0f : (a / b);
			Matrix44 r(c,0.0f,0.0f,0.0f,
					   0.0f,c,0.0f,0.0f,
					   0.0f,0.0f,c,0.0f,
					   0.0f,0.0f,0.0f,c);
			wr[lane] = r;
		}
	}
}





OSL_SHADEOP void
osl_transpose_mm (void *r, void *m)
{
    MAT(r) = MAT(m).transposed();
}

OSL_SHADEOP void
osl_transpose_w16mw16m(void *wr_, void *wm_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<Matrix44> wm(wm_);
		WideAccessor<Matrix44> wr(wr_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			Matrix44 m = wm[lane];
			Matrix44 r = m.transposed();
			wr[lane] = r;
		}
	}
}



OSL_SHADEOP int
osl_get_matrix (void *sg_, void *r, const char *from)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    ShadingContext *ctx = (ShadingContext *)sg->context;
    if (USTR(from) == Strings::common ||
            USTR(from) == ctx->shadingsys().commonspace_synonym()) {
        MAT(r).makeIdentity ();
        return true;
    }
    if (USTR(from) == Strings::shader) {
        ctx->renderer()->get_matrix (sg, MAT(r), sg->shader2common, sg->time);
        return true;
    }
    if (USTR(from) == Strings::object) {
        ctx->renderer()->get_matrix (sg, MAT(r), sg->object2common, sg->time);
        return true;
    }
    int ok = ctx->renderer()->get_matrix (sg, MAT(r), USTR(from), sg->time);
    if (! ok) {
        MAT(r).makeIdentity();
        ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
        if (ctx->shadingsys().unknown_coordsys_error())
            ctx->error ("Unknown transformation \"%s\"", from);
    }
    return ok;
}

#if 0
OSL_SHADEOP int
osl_get_matrix_batched (void *sgb_, void *r, const char *from)
{
	ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = sgb->uniform().context;

    if (USTR(from) == Strings::common ||
            USTR(from) == ctx->shadingsys().commonspace_synonym()) {
        MAT(r).makeIdentity ();
        return true;
    }
    if (USTR(from) == Strings::shader) {
        ctx->renderer()->get_matrix (sg, MAT(r), sg->shader2common, sg->time);
        return true;
    }
    if (USTR(from) == Strings::object) {
        ctx->renderer()->get_matrix (sg, MAT(r), sg->object2common, sg->time);
        return true;
    }
    int ok = ctx->renderer()->get_matrix (sg, MAT(r), USTR(from), sg->time);
    if (! ok) {
        MAT(r).makeIdentity();
        ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
        if (ctx->shadingsys().unknown_coordsys_error())
            ctx->error ("Unknown transformation \"%s\"", from);
    }
    return ok;
}
#endif

OSL_INLINE Mask
impl_osl_wide_get_matrix_masked (void *sgb_, void *wr, const char *from, WeakMask weak_mask)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;
	Wide<Matrix44> & wrm = WMAT(wr);
    if (USTR(from) == Strings::common ||
            USTR(from) == ctx->shadingsys().commonspace_synonym()) {
    	OSL_INTEL_PRAGMA("forceinline recursive")
    	{
			
			Matrix44 ident;
			ident.makeIdentity();
			OSL_INTEL_PRAGMA("omp simd simdlen(wrm.width)")				    	
			for(int lane=0; lane < wrm.width; ++lane) {
				wrm.set(lane, ident);
			}
    	}
        return Mask(true);
    }
    
	if (USTR(from) == Strings::shader) {
		ctx->batched_renderer()->get_matrix (sgb, wrm, sgb->varyingData().shader2common, sgb->varyingData().time, weak_mask);
		// NOTE: matching scalar version of code which ignores the renderservices return value
        return Mask(true);
	}
	if (USTR(from) == Strings::object) {
		ctx->batched_renderer()->get_matrix (sgb, wrm, sgb->varyingData().object2common, sgb->varyingData().time, weak_mask);
		// NOTE: matching scalar version of code which ignores the renderservices return value
        return Mask(true);
	}
	
	Mask succeeded = ctx->batched_renderer()->get_matrix (sgb, wrm, USTR(from), sgb->varyingData().time, weak_mask);
    Mask failedLanes = succeeded.invert(weak_mask);
    if (failedLanes.any_on())
    {
        Matrix44 ident;
        ident.makeIdentity();
        OSL_INTEL_PRAGMA("omp simd simdlen(wrm.width)")
        for(int lane=0; lane < wrm.width; ++lane) {
            if (failedLanes[lane]) {
                wrm.set(lane, ident);
            }
        }
		ShadingContext *ctx = sgb->uniform().context;
		if (ctx->shadingsys().unknown_coordsys_error())
		{
			ASSERT(weak_mask.any_on());
			ctx->error ("Unknown transformation \"%s\"", from);			
		}
	}
    return succeeded;
}


OSL_SHADEOP int
osl_get_inverse_matrix (void *sg_, void *r, const char *to)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    ShadingContext *ctx = (ShadingContext *)sg->context;
    if (USTR(to) == Strings::common ||
        USTR(to) == ctx->shadingsys().commonspace_synonym()) {
        MAT(r).makeIdentity ();
        return true;
    }
    if (USTR(to) == Strings::shader) {
        ctx->renderer()->get_inverse_matrix (sg, MAT(r), sg->shader2common, sg->time);
        return true;
    }
    if (USTR(to) == Strings::object) {
        ctx->renderer()->get_inverse_matrix (sg, MAT(r), sg->object2common, sg->time);
        return true;
    }
    int ok = ctx->renderer()->get_inverse_matrix (sg, MAT(r), USTR(to), sg->time);
    if (! ok) {
        MAT(r).makeIdentity ();
        ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
        if (ctx->shadingsys().unknown_coordsys_error())
            ctx->error ("Unknown transformation \"%s\"", to);
    }
    return ok;
}

OSL_INLINE Mask
impl_osl_wide_get_inverse_matrix_masked (void *sgb_, void *wr, const char *to, WeakMask weak_mask)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;
	Wide<Matrix44> & wrm = WMAT(wr);
    
    if (USTR(to) == Strings::common ||
            USTR(to) == ctx->shadingsys().commonspace_synonym()) {
    	
    	Matrix44 ident;
    	ident.makeIdentity();
    	for(int lane=0; lane < wrm.width; ++lane) {
    		wrm.set(lane, ident);
    	}
        return Mask(true);
    }
    if (USTR(to) == Strings::shader) {
    	ctx->batched_renderer()->get_inverse_matrix (sgb, wrm, sgb->varyingData().shader2common, sgb->varyingData().time, weak_mask);        
		// NOTE: matching scalar version of code which ignores the renderservices return value
        return Mask(true);
    }
    if (USTR(to) == Strings::object) {
    	ctx->batched_renderer()->get_inverse_matrix (sgb, wrm, sgb->varyingData().object2common, sgb->varyingData().time, weak_mask);        
		// NOTE: matching scalar version of code which ignores the renderservices return value
        return Mask(true);
    }

	// Based on the 1 function that calls this function
	// the results of the failed data lanes will get overwritten
	// so no need to make sure that the values are valid (assuming FP exceptions are disabled)
    Mask succeeded = ctx->batched_renderer()->get_inverse_matrix (sgb, wrm, USTR(to), sgb->varyingData().time, weak_mask);

    Mask failedLanes = succeeded.invert(weak_mask);
    if (failedLanes.any_on())
	{
        Matrix44 ident;
        ident.makeIdentity();
        for(int lane=0; lane < wrm.width; ++lane) {
            if (failedLanes[lane]) {
                wrm.set(lane, ident);
            }
        }
		ShadingContext *ctx = sgb->uniform().context;
		if (ctx->shadingsys().unknown_coordsys_error())
		{
			ctx->error ("Unknown transformation \"%s\"", to);			
		}
	}
    return succeeded;
    
}


OSL_SHADEOP int
osl_prepend_matrix_from (void *sg, void *r, const char *from)
{
    Matrix44 m;
    bool ok = osl_get_matrix ((ShaderGlobals *)sg, &m, from);
    if (ok)
        MAT(r) = m * MAT(r);
    else {
        ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
        if (ctx->shadingsys().unknown_coordsys_error())
            ctx->error ("Unknown transformation \"%s\"", from);
    }
    return ok;
}


OSL_SHADEOP void
osl_prepend_matrix_from_w16m_batched (void *sgb, void *wr, const char *from)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		Wide<Matrix44> wMfrom;
		/*Mask succeeded =*/
        impl_osl_wide_get_matrix_masked ((ShaderGlobalsBatch *)sgb, &wMfrom, from, Mask(true));

		Wide<Matrix44> & wrm = WMAT(wr);

		// No savings from using weak_mask or succeeded
		OSL_INTEL_PRAGMA("omp simd simdlen(wrm.width)")
		for(int lane=0; lane < wrm.width; ++lane) {
			Matrix44 mat_From = wMfrom.get(lane);
			Matrix44 mat_input = wrm.get(lane);

			Matrix44 result = mat_From * mat_input;

			wrm.set(lane, result);
		}
	}
}



OSL_SHADEOP int
osl_get_from_to_matrix (void *sg, void *r, const char *from, const char *to)
{
    Matrix44 Mfrom, Mto;
    int ok = osl_get_matrix ((ShaderGlobals *)sg, &Mfrom, from);
    ok &= osl_get_inverse_matrix ((ShaderGlobals *)sg, &Mto, to);
    MAT(r) = Mfrom * Mto;
    return ok;
}

OSL_INLINE Mask
impl_osl_wide_get_from_to_matrix_masked (void *sgb, void *wr, const char *from, const char *to, WeakMask weak_mask)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		Wide<Matrix44> wMfrom, wMto;
        Mask succeeded = impl_osl_wide_get_matrix_masked ((ShaderGlobalsBatch *)sgb, &wMfrom, from, weak_mask);
        succeeded &= impl_osl_wide_get_inverse_matrix_masked ((ShaderGlobalsBatch *)sgb, &wMto, to, weak_mask);
		
		Wide<Matrix44> & wrm = WMAT(wr);
	
		// No savings from using weak_mask or succeeded
		OSL_INTEL_PRAGMA("omp simd simdlen(wrm.width)")
		for(int lane=0; lane < wrm.width; ++lane) {
			Matrix44 mat_From = wMfrom.get(lane);
			Matrix44 mat_To = wMto.get(lane);		
	
			Matrix44 result = mat_From * mat_To;
			
			wrm.set(lane, result);
		}
		return succeeded;
	}
}

OSL_SHADEOP int
osl_get_from_to_matrix_w16m_batched (void *sgb, void *r, const char *from, const char *to)
{
    return impl_osl_wide_get_from_to_matrix_masked(sgb, r, from, to, Mask(true)).value();
}




// point = M * point
inline void osl_transform_vmv(void *result, const Matrix44 &M, void* v_)
{
	//std::cout << "osl_transform_vmv" << std::endl;
   const Vec3 &v = VEC(v_);
   robust_multVecMatrix (M, v, VEC(result));
}




OSL_INLINE void avoidAliasingRobustMultVecMatrix(
	const Wide<Matrix44>& wx,
	const Wide< Imath::Vec3<float> >& wsrc,
	MaskedAccessor<Vec3>& wdst)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		OSL_INTEL_PRAGMA("omp simd simdlen(wdst.width)")
		for(int index=0; index < wdst.width; ++index)
		{
		   const Matrix44 x = wx.get(index);
		   Imath::Vec3<float> src = wsrc.get(index);
		   
		   //std::cout << "----src>" << src << std::endl;
		   
		   Imath::Vec3<float> dst;	   
	
		   
		   //robust_multVecMatrix(x, src, dst);
		   
			// Avoid alising issues when mixing data member access vs. 
			// reinterpret casted array based access.
			// Legally, compiler could assume no alaising because they are technically			
			// different types.  As they actually over the same memory incorrect
			// code generation can ensue
#if 0
		   float a = src[0] * x[0][0] + src[1] * x[1][0] + src[2] * x[2][0] + x[3][0];
		    float b = src[0] * x[0][1] + src[1] * x[1][1] + src[2] * x[2][1] + x[3][1];
		    float c = src[0] * x[0][2] + src[1] * x[1][2] + src[2] * x[2][2] + x[3][2];
		    float w = src[0] * x[0][3] + src[1] * x[1][3] + src[2] * x[2][3] + x[3][3];
#else
			   float a = src.x * x[0][0] + src.y * x[1][0] + src.z * x[2][0] + x[3][0];
			    float b = src.x * x[0][1] + src.y * x[1][1] + src.z * x[2][1] + x[3][1];
			    float c = src.x * x[0][2] + src.y * x[1][2] + src.z * x[2][2] + x[3][2];
			    float w = src.x * x[0][3] + src.y * x[1][3] + src.z * x[2][3] + x[3][3];
		    
#endif

		    if (__builtin_expect(w != 0, 1)) {
		       dst.x = a / w;
		       dst.y = b / w;
		       dst.z = c / w;
		    } else {
		       dst.x = 0;
		       dst.y = 0;
		       dst.z = 0;
		    }		   
	    
		   //std::cout << "----dst>" << dst << std::endl;
		   
		   wdst[index] = dst;
		   
		   //Imath::Vec3<float> verify = wdst.get(index);
		   //std::cout << "---->" << verify << "<-----" << std::endl;
		}
	}
}

OSL_INLINE void
avoidAliasingMultDirMatrix (const Matrix44 &M, const Vec3 &src, Vec3 &dst)
{
	float a = src.x * M[0][0] + src.y * M[1][0] + src.z * M[2][0];
	float b = src.x * M[0][1] + src.y * M[1][1] + src.z * M[2][1];
	float c = src.x * M[0][2] + src.y * M[1][2] + src.z * M[2][2];

	dst.x = a;
	dst.y = b;
	dst.z = c;
    
}

/// Multiply a matrix times a vector with derivatives to obtain
/// a transformed vector with derivatives.
OSL_INLINE void
avoidAliasingRobustMultVecMatrix (
	const Wide<Matrix44> &WM, 
	const Wide<Dual2<Vec3>> &win, 
	MaskedAccessor<Dual2<Vec3>> &wout)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		OSL_INTEL_PRAGMA("omp simd simdlen(wout.width)")
		for(int index=0; index < wout.width; ++index)
		{
			const Matrix44 M = WM.get(index);			
			const Dual2<Vec3> in = win.get(index);
	
			// Rearrange into a Vec3<Dual2<float> >
			Imath::Vec3<Dual2<float> > din, dout;
			
			// Avoid alising issues when mixing data member access vs. 
			// reinterpret casted array based access.
			// Legally, compiler could assume no alaising because they are technically			
			// different types.  As they actually over the same memory incorrect
			// code generation can ensue
#if 0
			for (int i = 0;  i < 3;  ++i)
				din[i].set (in.val()[i], in.dx()[i], in.dy()[i]);
			
			Dual2<float> a = din[0] * M[0][0] + din[1] * M[1][0] + din[2] * M[2][0] + M[3][0];
			Dual2<float> b = din[0] * M[0][1] + din[1] * M[1][1] + din[2] * M[2][1] + M[3][1];
			Dual2<float> c = din[0] * M[0][2] + din[1] * M[1][2] + din[2] * M[2][2] + M[3][2];
			Dual2<float> w = din[0] * M[0][3] + din[1] * M[1][3] + din[2] * M[2][3] + M[3][3];
#else
			din.x.set (in.val().x, in.dx().x, in.dy().x);
			din.y.set (in.val().y, in.dx().y, in.dy().y);
			din.z.set (in.val().z, in.dx().z, in.dy().z);
			
			Dual2<float> a = din.x * M[0][0] + din.y * M[1][0] + din.z * M[2][0] + M[3][0];
			Dual2<float> b = din.x * M[0][1] + din.y * M[1][1] + din.z * M[2][1] + M[3][1];
			Dual2<float> c = din.x * M[0][2] + din.y * M[1][2] + din.z * M[2][2] + M[3][2];
			Dual2<float> w = din.x * M[0][3] + din.y * M[1][3] + din.z * M[2][3] + M[3][3];
#endif
			
		
			if (w.val() != 0) {
			   dout.x = a / w;
			   dout.y = b / w;
			   dout.z = c / w;
			} else {
			   dout.x = 0;
			   dout.y = 0;
			   dout.z = 0;
			}
		
			Dual2<Vec3> out;
			// Rearrange back into Dual2<Vec3>
#if 0
			out.set (Vec3 (dout[0].val(), dout[1].val(), dout[2].val()),
					 Vec3 (dout[0].dx(),  dout[1].dx(),  dout[2].dx()),
					 Vec3 (dout[0].dy(),  dout[1].dy(),  dout[2].dy()));
#else
			out.set (Vec3 (dout.x.val(), dout.y.val(), dout.z.val()),
					 Vec3 (dout.x.dx(),  dout.y.dx(),  dout.z.dx()),
					 Vec3 (dout.x.dy(),  dout.y.dy(),  dout.z.dy()));
#endif
			
			wout[index] = out;
		   
		   //Imath::Vec3<float> verify = wdst.get(index);
		   //std::cout << "---->" << verify << "<-----" << std::endl;
		}
	}
    
}


inline void osl_transform_wvwmwv(void *result, const Wide<Matrix44> &M, void* v_, Mask mask)
{
   MaskedAccessor<Vec3> resultRef(result, mask);
	
   const Wide<Vec3> &v = WVEC(v_);
   avoidAliasingRobustMultVecMatrix (M, v, resultRef);
}


// TODO: do we need this control of optimization level?  
// Remove after verifying correct results
OSL_INTEL_PRAGMA("intel optimization_level 2")

inline void osl_transform_dvmdv(void *result, const Matrix44 &M, void* v_)
{
   const Dual2<Vec3> &v = DVEC(v_);
   robust_multVecMatrix (M, v, DVEC(result));
}

inline void osl_transform_wdvwmwdv(void *result, const Wide<Matrix44> &M, void* v_, Mask mask)
{
   MaskedAccessor<Dual2<Vec3>> resultRef(result, mask);
   
   const Wide<Dual2<Vec3>> &v = WDVEC(v_);
   avoidAliasingRobustMultVecMatrix (M, v, resultRef);
}

// vector = M * vector
inline void osl_transformv_vmv(void *result, const Matrix44 &M, void* v_)
{
   const Vec3 &v = VEC(v_);
   M.multDirMatrix (v, VEC(result));
}

inline void osl_transformv_wvwmwv(void *result, const Wide<Matrix44> &wM, void* v_, Mask mask)
{
   //const Vec3 &v = VEC(v_);
	
   OSL_INTEL_PRAGMA("forceinline recursive")
   {			   
	   const Wide<Vec3> &wv = WVEC(v_);
	   MaskedAccessor<Vec3> resultRef(result, mask);   
	   
	   OSL_INTEL_PRAGMA("omp simd simdlen(resultRef.width)")
	   for(int i=0; i < resultRef.width; ++i)
	   {
		   Matrix44 M = wM.get(i);
		   Vec3 v = wv.get(i);
		   Vec3 r;

		   // Do to illegal aliasing in OpenEXR version
		   // we call our own flavor without aliasing
		   //M.multDirMatrix (v, VEC(result));
		   avoidAliasingMultDirMatrix(M, v, r);
		   
		   resultRef[i] = r;
	   }
   }	
}

inline void osl_transformv_dvmdv(void *result, const Matrix44 &M, void* v_)
{
   const Dual2<Vec3> &v = DVEC(v_);
   multDirMatrix (M, v, DVEC(result));
}

inline void
avoidAliasingMultDirMatrix (const Matrix44 &M, const Dual2<Vec3> &in, Dual2<Vec3> &out)
{
	avoidAliasingMultDirMatrix(M, in.val(), out.val());
	avoidAliasingMultDirMatrix(M, in.dx(), out.dx());
	avoidAliasingMultDirMatrix(M, in.dy(), out.dy());
}

inline void osl_transformv_wdvwmwdv(void *result, const Wide<Matrix44> &wM, void* v_, Mask mask)
{
   //const Dual2<Vec3> &v = DVEC(v_);
   //multDirMatrix (M, v, DVEC(result));
   OSL_INTEL_PRAGMA("forceinline recursive")
   {		
	   MaskedAccessor<Dual2<Vec3>> resultRef(result, mask);

	   const Wide<Dual2<Vec3>> &wv = WDVEC(v_);
		  	   
	   OSL_INTEL_PRAGMA("omp simd simdlen(resultRef.width)")
	   for(int i=0; i < resultRef.width; ++i)
	   {
		   Dual2<Vec3> v = wv.get(i);
		   Matrix44 M = wM.get(i);
		   Dual2<Vec3> r;
		   
		   avoidAliasingMultDirMatrix (M, v, r);
	   
		   resultRef[i] = r;
	   }
   }	
}

// normal = M * normal
inline void osl_transformn_vmv(void *result, const Matrix44 &M, void* v_)
{
   const Vec3 &v = VEC(v_);
   M.inverse().transposed().multDirMatrix (v, VEC(result));
}

inline void osl_transformn_wvwmwv(void *result, const Wide<Matrix44> &wM, void* v_, Mask mask)
{
	//std::cout << "osl_transform_vmv" << std::endl;
	
   OSL_INTEL_PRAGMA("forceinline recursive")
   {		
	   
	   const Wide<Vec3> &wv = WVEC(v_);
	   //Wide<Vec3> &wr = WVEC(result);
	   MaskedAccessor<Vec3> resultRef(result, mask);
	   	   
	   OSL_INTEL_PRAGMA("omp simd simdlen(resultRef.width)")
	   for(int i=0; i < resultRef.width; ++i)
	   {
		   Vec3 v = wv.get(i);
		   Matrix44 M = wM.get(i);
		   Vec3 r;
		   
		   M.inverse().transposed().multDirMatrix (v, r);
		   
		   resultRef[i] = r;
	   }
   }
}


inline void osl_transformn_dvmdv(void *result, const Matrix44 &M, void* v_)
{
   const Dual2<Vec3> &v = DVEC(v_);
   multDirMatrix (M.inverse().transposed(), v, DVEC(result));
}

inline void osl_transformn_wdvwmwdv(void *result, const Wide<Matrix44> &wM, void* v_, Mask mask)
{
   
   //multDirMatrix (M.inverse().transposed(), v, DVEC(result));
   
   OSL_INTEL_PRAGMA("forceinline recursive")
   {		
	   MaskedAccessor<Dual2<Vec3>> resultRef(result, mask);

	   const Wide<Dual2<Vec3>> &wv = WDVEC(v_);
	      
	   OSL_INTEL_PRAGMA("omp simd simdlen(resultRef.width)")
	   for(int i=0; i < resultRef.width; ++i)
	   {
		   Dual2<Vec3> v = wv.get(i);
		   Matrix44 M = wM.get(i);
		   Dual2<Vec3> r;
		   
		   multDirMatrix (M.inverse().transposed(), v, r);
		   
	   
		   resultRef[i] = r;
	   }
   }   
}


OSL_SHADEOP void
osl_transformv_w16vw16mw16v (void *r_, void *matrix_, void *s_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<Vec3> wsource(s_);
		ConstWideAccessor<Matrix44> wmatrix(matrix_);
		WideAccessor<Vec3> wr(r_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			Vec3 s = wsource[lane];
			Matrix44 m = wmatrix[lane];
			Vec3 r;

			avoidAliasingMultDirMatrix (m, s, r);

			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_transformv_w16dvw16mw16dv (void *r_, void *matrix_, void *s_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<Dual2<Vec3>> wsource(s_);
		ConstWideAccessor<Matrix44> wmatrix(matrix_);
		WideAccessor<Dual2<Vec3>> wr(r_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<Vec3> s = wsource[lane];
			Matrix44 m = wmatrix[lane];
			Dual2<Vec3> r;

			avoidAliasingMultDirMatrix (m, s, r);

			wr[lane] = r;
		}
	}
}



OSL_SHADEOP int
osl_transform_triple (void *sg_, void *Pin, int Pin_derivs,
                      void *Pout, int Pout_derivs,
                      void *from, void *to, int vectype)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    Matrix44 M;
    int ok;
    Pin_derivs &= Pout_derivs;   // ignore derivs if output doesn't need it
    if (USTR(from) == Strings::common)
        ok = osl_get_inverse_matrix (sg, &M, (const char *)to);
    else if (USTR(to) == Strings::common)
        ok = osl_get_matrix (sg, &M, (const char *)from);
    else
        ok = osl_get_from_to_matrix (sg, &M, (const char *)from,
                                     (const char *)to);
    if (ok) {
        if (vectype == TypeDesc::POINT) {
            if (Pin_derivs)
                osl_transform_dvmdv(Pout, M, Pin);
            else
                osl_transform_vmv(Pout, M, Pin);
        } else if (vectype == TypeDesc::VECTOR) {
            if (Pin_derivs)
                osl_transformv_dvmdv(Pout, M, Pin);
            else
                osl_transformv_vmv(Pout, M, Pin);
        } else if (vectype == TypeDesc::NORMAL) {
            if (Pin_derivs)
                osl_transformn_dvmdv(Pout, M, Pin);
            else
                osl_transformn_vmv(Pout, M, Pin);
        }
        else ASSERT(0);
    } else {
        *(Vec3 *)Pout = *(Vec3 *)Pin;
        if (Pin_derivs) {
            ((Vec3 *)Pout)[1] = ((Vec3 *)Pin)[1];
            ((Vec3 *)Pout)[2] = ((Vec3 *)Pin)[2];
        }
    }
    if (Pout_derivs && !Pin_derivs) {
        ((Vec3 *)Pout)[1].setValue (0.0f, 0.0f, 0.0f);
        ((Vec3 *)Pout)[2].setValue (0.0f, 0.0f, 0.0f);
    }
    return ok;
}



OSL_SHADEOP void 
osl_wide_transform_triple (void *sgb_, void *Pin, int Pin_derivs,
                      void *Pout, int Pout_derivs,
					  void * from, void * to, int vectype, unsigned int mask_value)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;
    
    Mask mask(mask_value);
    
    //ASSERT(Pin != Pout);

    Wide<Matrix44> M;
    Pin_derivs &= Pout_derivs;   // ignore derivs if output doesn't need it
    
    Mask succeeded;
    // Avoid matrix concatenation if possible by detecting when the 
    // adjacent matrix would be identity
    // We don't expect both from and to == common, so we are not
    // optimizing for it
    if (USTR(from) == Strings::common ||
            USTR(from) == ctx->shadingsys().commonspace_synonym()) {
        succeeded = impl_osl_wide_get_inverse_matrix_masked (sgb, &M, (const char *)to, mask);
    } else if (USTR(to) == Strings::common ||
            USTR(to) == ctx->shadingsys().commonspace_synonym()) {
        succeeded = impl_osl_wide_get_matrix_masked(sgb, &M, (const char *)from, mask);
    } else {
        succeeded = impl_osl_wide_get_from_to_matrix_masked (sgb, &M, (const char *)from,
										 (const char *)to, mask);
    }
    

    {
        // only operate on active lanes
        Mask activeMask = mask & succeeded;
        // TODO:  consider templatising this function and having
    	// a specific version for each vec type, as we know the type 
    	// at code gen time we can call a specific version versus
    	// the cost testing the type here.
        if (vectype == TypeDesc::POINT) {
            if (Pin_derivs) {
                osl_transform_wdvwmwdv(Pout, M, Pin, activeMask);
            } else {
                osl_transform_wvwmwv(Pout, M, Pin, activeMask);
            }
        } else if (vectype == TypeDesc::VECTOR) {
            if (Pin_derivs) {
                osl_transformv_wdvwmwdv(Pout, M, Pin, activeMask);
            } else {
                osl_transformv_wvwmwv(Pout, M, Pin, activeMask);
            }
        } else if (vectype == TypeDesc::NORMAL) {
            if (Pin_derivs)
                osl_transformn_wdvwmwdv(Pout, M, Pin, activeMask);
            else {
                osl_transformn_wvwmwv(Pout, M, Pin, activeMask);
            }
        }
        else {        	
        	ASSERT(0);
        }
    }
    OSL_INTEL_PRAGMA("forceinline recursive")
    {		
        // if Pin != Pout, we still need to copy inactive data over to Pout
        // Handle cleaning up any data lanes that did not succeed
        if ((Pin != Pout) && succeeded.any_off(mask))
		{
			// For any lanes we failed to get a matrix for
			// just copy the output to the input values
			WideAccessor<Vec3> inVec(Pin);
			// NOTE:  As we only only want to copy lanes that failed, 
			// we will invert our success mask
            Mask failed = succeeded.invert(mask);
			MaskedAccessor<Vec3> outVec(Pout, failed);
			OSL_INTEL_PRAGMA("omp simd simdlen(outVec.width)")
			for(int i=0; i< outVec.width; ++i)
			{
				outVec[i] = inVec[i];
			}
			if (Pin_derivs) {
				WideAccessor<Vec3> inDx(Pin, 1 /*derivIndex*/);
				WideAccessor<Vec3> inDy(Pin, 2 /*derivIndex*/);
				MaskedAccessor<Vec3> outDx(Pout, failed, 1 /*derivIndex*/);
				MaskedAccessor<Vec3> outDy(Pout, failed, 2 /*derivIndex*/);
				OSL_INTEL_PRAGMA("omp simd simdlen(outDx.width)")
				for(int i=0; i< outDx.width; ++i)
				{
					outDx[i] = inDx[i];
					outDy[i] = inDy[i];
				}        	
			}    	
		}
        // Handle zeroing derivs if necessary
		if (Pout_derivs && !Pin_derivs) {
			WideAccessor<Vec3> outDx(Pout, 1 /*derivIndex*/);
			WideAccessor<Vec3> outDy(Pout, 2 /*derivIndex*/);
			OSL_INTEL_PRAGMA("omp simd simdlen(outDx.width)")
			for(int i=0; i< outDx.width; ++i)
			{	    	
				outDx[i] = Vec3(0.0f, 0.0f, 0.0f);
				outDy[i] = Vec3(0.0f, 0.0f, 0.0f);
			}        	
		}
    }
}


OSL_SHADEOP int
osl_transform_triple_nonlinear (void *sg_, void *Pin, int Pin_derivs,
                                void *Pout, int Pout_derivs,
                                void *from, void *to,
                                int vectype)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    RendererServices *rend = sg->renderer;
    if (rend->transform_points (sg, USTR(from), USTR(to), sg->time,
                                (const Vec3 *)Pin, (Vec3 *)Pout, 1,
                                (TypeDesc::VECSEMANTICS)vectype)) {
        // Renderer had a direct way to transform the points between the
        // two spaces.
        if (Pout_derivs) {
            if (Pin_derivs) {
                rend->transform_points (sg, USTR(from), USTR(to), sg->time,
                                        (const Vec3 *)Pin+1,
                                        (Vec3 *)Pout+1, 2, TypeDesc::VECTOR);
            } else {
                ((Vec3 *)Pout)[1].setValue (0.0f, 0.0f, 0.0f);
                ((Vec3 *)Pout)[2].setValue (0.0f, 0.0f, 0.0f);
            }
        }
        return true;
    }

    // Renderer couldn't or wouldn't transform directly
    return osl_transform_triple (sg, Pin, Pin_derivs, Pout, Pout_derivs,
                                 from, to, vectype);
}



// Calculate the determinant of a 2x2 matrix.
template <typename F>
inline F det2x2(F a, F b, F c, F d)
{
    return a * d - b * c;
}

// calculate the determinant of a 3x3 matrix in the form:
//     | a1,  b1,  c1 |
//     | a2,  b2,  c2 |
//     | a3,  b3,  c3 |
template <typename F>
inline F det3x3(F a1, F a2, F a3, F b1, F b2, F b3, F c1, F c2, F c3)
{
    return a1 * det2x2( b2, b3, c2, c3 )
         - b1 * det2x2( a2, a3, c2, c3 )
         + c1 * det2x2( a2, a3, b2, b3 );
}

// calculate the determinant of a 4x4 matrix.
template <typename F>
inline F det4x4(const Imath::Matrix44<F> &m)
{
    // assign to individual variable names to aid selecting correct elements
    F a1 = m[0][0], b1 = m[0][1], c1 = m[0][2], d1 = m[0][3];
    F a2 = m[1][0], b2 = m[1][1], c2 = m[1][2], d2 = m[1][3];
    F a3 = m[2][0], b3 = m[2][1], c3 = m[2][2], d3 = m[2][3];
    F a4 = m[3][0], b4 = m[3][1], c4 = m[3][2], d4 = m[3][3];
    return a1 * det3x3( b2, b3, b4, c2, c3, c4, d2, d3, d4)
         - b1 * det3x3( a2, a3, a4, c2, c3, c4, d2, d3, d4)
         + c1 * det3x3( a2, a3, a4, b2, b3, b4, d2, d3, d4)
         - d1 * det3x3( a2, a3, a4, b2, b3, b4, c2, c3, c4);
}

OSL_SHADEOP float
osl_determinant_fm (void *m)
{
    return det4x4 (MAT(m));
}

OSL_SHADEOP void
osl_determinant_w16fw16m(void *wr_, void * wm_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<Matrix44> wm(wm_);
		WideAccessor<float> wr(wr_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			Matrix44 m = wm[lane];
			float r = det4x4(m);
			wr[lane] = r;
		}
	}
}



} // namespace pvt
OSL_NAMESPACE_EXIT
