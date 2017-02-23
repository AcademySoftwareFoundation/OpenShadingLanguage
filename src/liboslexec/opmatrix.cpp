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


#if OSL_USE_WIDE_LLVM_BACKEND
typedef int wide_int __attribute__((vector_size(SimdLaneCount*sizeof(int))));
#endif


// Matrix ops

OSL_SHADEOP void
osl_mul_mm (void *r, void *a, void *b)
{
    MAT(r) = MAT(a) * MAT(b);
}

OSL_SHADEOP void
osl_mul_mf (void *r, void *a, float b)
{
    MAT(r) = MAT(a) * b;
}

OSL_SHADEOP void
osl_mul_m_ff (void *r, float a, float b)
{
    float f = a * b;
    MAT(r) = Matrix44 (f,0,0,0, 0,f,0,0, 0,0,f,0, 0,0,0,f);
}



OSL_SHADEOP void
osl_div_mm (void *r, void *a, void *b)
{
    MAT(r) = MAT(a) * MAT(b).inverse();
}

OSL_SHADEOP void
osl_div_mf (void *r, void *a, float b)
{
    MAT(r) = MAT(a) * (1.0f/b);
}

OSL_SHADEOP void
osl_div_fm (void *r, float a, void *b)
{
    MAT(r) = a * MAT(b).inverse();
}

OSL_SHADEOP void
osl_div_m_ff (void *r, float a, float b)
{
    float f = (b == 0) ? 0.0f : (a / b);
    MAT(r) = Matrix44 (f,0,0,0, 0,f,0,0, 0,0,f,0, 0,0,0,f);
}



OSL_SHADEOP void
osl_transpose_mm (void *r, void *m)
{
    MAT(r) = MAT(m).transposed();
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

#if OSL_USE_WIDE_LLVM_BACKEND
OSL_SHADEOP int
osl_wide_get_matrix (void *sgb_, void *wr, const char *from)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;
	Wide<Matrix44> & wrm = WMAT(wr);
    if (USTR(from) == Strings::common ||
            USTR(from) == ctx->shadingsys().commonspace_synonym()) {
    	Matrix44 ident;
    	ident.makeIdentity();
		OSL_INTEL_PRAGMA("simd")
    	for(int lane=0; lane < SimdLaneCount; ++lane) {
    		wrm.set(lane, ident);
    	}
        return true;
    }
    
    // TODO:  consider passing wide matrix through to renderer
	if (USTR(from) == Strings::shader) {
		ctx->renderer()->get_matrix (sgb, wrm, sgb->varyingData().shader2common, sgb->varyingData().time);
		return true;
	}
	if (USTR(from) == Strings::object) {
		ctx->renderer()->get_matrix (sgb, wrm, sgb->varyingData().object2common, sgb->varyingData().time);
		return true;
	}
	int wok = true;
	for(int lane=0; lane < SimdLaneCount; ++lane) {    
		Matrix44 r;
		int ok = ctx->renderer()->get_matrix (sgb, r, USTR(from), sgb->varyingData().time.get(lane));
		if (! ok) {
			wok = false;
			r.makeIdentity();
			ShadingContext *ctx = sgb->uniform().context;
			if (ctx->shadingsys().unknown_coordsys_error())
				ctx->error ("Unknown transformation \"%s\"", from);
		}
		wrm.set(lane, r);
	}
    return wok;

}
#endif



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

#if OSL_USE_WIDE_LLVM_BACKEND
OSL_SHADEOP int
osl_wide_get_inverse_matrix (void *sgb_, void *wr, const char *to)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;
	Wide<Matrix44> & wrm = WMAT(wr);
    
    if (USTR(to) == Strings::common ||
            USTR(to) == ctx->shadingsys().commonspace_synonym()) {
    	
    	Matrix44 ident;
    	ident.makeIdentity();
    	for(int lane=0; lane < SimdLaneCount; ++lane) {
    		wrm.set(lane, ident);
    	}
        return true;
    }
    if (USTR(to) == Strings::shader) {
    	ctx->renderer()->get_inverse_matrix (sgb, wrm, sgb->varyingData().shader2common, sgb->varyingData().time);        
        return true;
    }
    if (USTR(to) == Strings::object) {
    	ctx->renderer()->get_inverse_matrix (sgb, wrm, sgb->varyingData().object2common, sgb->varyingData().time);        
        return true;
    }

	int wok = true;
	for(int lane=0; lane < SimdLaneCount; ++lane) {    
		Matrix44 r;
	    int ok = ctx->renderer()->get_inverse_matrix (sgb, r, USTR(to), sgb->varyingData().time.get(lane));
		if (! ok) {
			wok = false;
			r.makeIdentity();
			ShadingContext *ctx = sgb->uniform().context;
			if (ctx->shadingsys().unknown_coordsys_error())
				ctx->error ("Unknown transformation \"%s\"", to);
		}
		wrm.set(lane, r);
	}
    return wok;
    
}
#endif



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



OSL_SHADEOP int
osl_get_from_to_matrix (void *sg, void *r, const char *from, const char *to)
{
    Matrix44 Mfrom, Mto;
    int ok = osl_get_matrix ((ShaderGlobals *)sg, &Mfrom, from);
    ok &= osl_get_inverse_matrix ((ShaderGlobals *)sg, &Mto, to);
    MAT(r) = Mfrom * Mto;
    return ok;
}

#if OSL_USE_WIDE_LLVM_BACKEND

OSL_SHADEOP int
osl_wide_get_from_to_matrix (void *sgb, void *wr, const char *from, const char *to)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		Wide<Matrix44> wMfrom, wMto;
		int ok = osl_wide_get_matrix ((ShaderGlobalsBatch *)sgb, &wMfrom, from);
		ok &= osl_wide_get_inverse_matrix ((ShaderGlobalsBatch *)sgb, &wMto, to);
		
		Wide<Matrix44> & wrm = WMAT(wr);
	
		OSL_INTEL_PRAGMA("simd assert")
		for(int lane=0; lane < SimdLaneCount; ++lane) {    
			Matrix44 mat_From = wMfrom.get(lane);
			Matrix44 mat_To = wMto.get(lane);		
	
			Matrix44 result = mat_From * mat_To;
			
			wrm.set(lane, result);
		}
		return ok;
	}
}

#endif


// point = M * point
inline void osl_transform_vmv(void *result, const Matrix44 &M, void* v_)
{
	//std::cout << "osl_transform_vmv" << std::endl;
   const Vec3 &v = VEC(v_);
   robust_multVecMatrix (M, v, VEC(result));
}

inline void osl_transform_wvwmwv(void *result, const Wide<Matrix44> &M, void* v_)
{
	//std::cout << "osl_transform_vmv" << std::endl;
	
   const Wide<Vec3> &v = WVEC(v_);
   robust_multVecMatrix (M, v, WVEC(result));
}


OSL_INTEL_PRAGMA("intel optimization_level 2")

inline void osl_transform_dvmdv(void *result, const Matrix44 &M, void* v_)
{
   const Dual2<Vec3> &v = DVEC(v_);
   robust_multVecMatrix (M, v, DVEC(result));
}

inline void osl_transform_wdvwmwdv(void *result, const Wide<Matrix44> &M, void* v_)
{
   const Wide<Dual2<Vec3>> &v = WDVEC(v_);
   robust_multVecMatrix (M, v, WDVEC(result));
}

// vector = M * vector
inline void osl_transformv_vmv(void *result, const Matrix44 &M, void* v_)
{
   const Vec3 &v = VEC(v_);
   M.multDirMatrix (v, VEC(result));
}

inline void osl_transformv_dvmdv(void *result, const Matrix44 &M, void* v_)
{
   const Dual2<Vec3> &v = DVEC(v_);
   multDirMatrix (M, v, DVEC(result));
}

// normal = M * normal
inline void osl_transformn_vmv(void *result, const Matrix44 &M, void* v_)
{
   const Vec3 &v = VEC(v_);
   M.inverse().transposed().multDirMatrix (v, VEC(result));
}

inline void osl_transformn_dvmdv(void *result, const Matrix44 &M, void* v_)
{
   const Dual2<Vec3> &v = DVEC(v_);
   multDirMatrix (M.inverse().transposed(), v, DVEC(result));
}



OSL_SHADEOP int
osl_transform_triple (void *sg_, void *Pin, int Pin_derivs,
                      void *Pout, int Pout_derivs,
                      void *from, void *to, int vectype)
{
    static ustring u_common ("common");
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    Matrix44 M;
    int ok;
    Pin_derivs &= Pout_derivs;   // ignore derivs if output doesn't need it
    if (USTR(from) == u_common)
        ok = osl_get_inverse_matrix (sg, &M, (const char *)to);
    else if (USTR(to) == u_common)
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



#if OSL_USE_WIDE_LLVM_BACKEND
OSL_SHADEOP wide_int
osl_wide_transform_triple (void *sgb_, void *Pin, int Pin_derivs,
                      void *Pout, int Pout_derivs,
					  void * from, void * to, int vectype)
{
    static ustring u_common ("common");
    ASSERT(Pin != Pout);
    
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    
    Wide<Matrix44> M;
    int ok;
    Pin_derivs &= Pout_derivs;   // ignore derivs if output doesn't need it
    
#if 0 
    if (USTR(from) == u_common)
        ok = osl_get_inverse_matrix (sgb, &M, (const char *)to);
    else if (USTR(to) == u_common)
        ok = osl_get_matrix (sgb, &M, (const char *)from);
    else
#endif
        ok = osl_wide_get_from_to_matrix (sgb, &M, (const char *)from,
                                     (const char *)to);
    if (ok) {
        if (vectype == TypeDesc::POINT) {
            if (Pin_derivs) {
                osl_transform_wdvwmwdv(Pout, M, Pin);
            } else {
                osl_transform_wvwmwv(Pout, M, Pin);
            }
#if 0 
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
#endif
        }
        else {
        	std::cout << "osl_wide_transform_triple vectype=" << vectype << std::endl;
        	ASSERT(0);
        }
    } else {
    	ASSERT(0);
        *(Vec3 *)Pout = *(Vec3 *)Pin;
        if (Pin_derivs) {
            ((Vec3 *)Pout)[1] = ((Vec3 *)Pin)[1];
            ((Vec3 *)Pout)[2] = ((Vec3 *)Pin)[2];
        }
    }
    if (Pout_derivs && !Pin_derivs) {
    	ASSERT(0);
        ((Vec3 *)Pout)[1].setValue (0.0f, 0.0f, 0.0f);
        ((Vec3 *)Pout)[2].setValue (0.0f, 0.0f, 0.0f);
    }
    
    wide_int wresult = { ok, ok, ok, ok};
    return wresult;
}
#endif


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




} // namespace pvt
OSL_NAMESPACE_EXIT
