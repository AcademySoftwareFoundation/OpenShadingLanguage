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
#include "OSL/Imathx.h"
#include "OSL/wide.h"


OSL_NAMESPACE_ENTER

namespace pvt {

// Matrix ops
OSL_SHADEOP void
osl_mul_mfm (void *r, float a, void * b)
{
    MAT(r) = MAT(b) * a;
}

// flatten is workaround to enable inlining of non-inlined methods
OSL_SHADEOP OSL_CLANG_ATTRIBUTE(flatten) void
osl_mul_w16mw16fw16m(void *wr_, void *wa_, void * wb_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<float> wa(wa_);
		ConstWideAccessor<Matrix44> wb(wb_);
		WideAccessor<Matrix44> wr(wr_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			float a = wa[lane];
			Matrix44 b = wb[lane];
			Matrix44 r = b * a;
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP OSL_CLANG_ATTRIBUTE(flatten) void
osl_mul_w16mw16fw16m_masked(void *wr_, void *wa_, void * wb_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<float> wa(wa_);
        ConstWideAccessor<Matrix44> wb(wb_);
        MaskedAccessor<Matrix44> wr(wr_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
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

// flatten is workaround to enable inlining of non-inlined methods
OSL_SHADEOP OSL_CLANG_ATTRIBUTE(flatten) void
osl_mul_w16mw16mw16f(void *wr_, void *wa_, void * wb_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Matrix44> wa(wa_);
		ConstWideAccessor<float> wb(wb_);
		WideAccessor<Matrix44> wr(wr_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Matrix44 a = wa[lane];
			float b = wb[lane];
			Matrix44 r = a * b;
			wr[lane] = r;
		}
	}
}

// flatten is workaround to enable inlining of non-inlined methods
OSL_SHADEOP OSL_CLANG_ATTRIBUTE(flatten) void
osl_mul_w16mw16mw16f_masked(void *wr_, void *wa_, void * wb_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Matrix44> wa(wa_);
        ConstWideAccessor<float> wb(wb_);
        MaskedAccessor<Matrix44> wr(wr_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
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

// flatten is workaround to enable inlining of non-inlined methods
OSL_SHADEOP OSL_CLANG_ATTRIBUTE(flatten) void
osl_mul_w16mw16mw16m(void *wr_, void *wa_, void * wb_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Matrix44> wa(wa_);
		ConstWideAccessor<Matrix44> wb(wb_);
		WideAccessor<Matrix44> wr(wr_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Matrix44 a = wa[lane];
			Matrix44 b = wb[lane];
			// Need inlinable version for vectorization
			// Matrix44 r = a * b;
			Matrix44 r;
			inlinedMultMatrixMatrix(a, b, r);
			wr[lane] = r;
		}
	}
}

// flatten is workaround to enable inlining of non-inlined methods
OSL_SHADEOP OSL_CLANG_ATTRIBUTE(flatten) void
osl_mul_w16mw16mw16m_masked(void *wr_, void *wa_, void * wb_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Matrix44> wa(wa_);
        ConstWideAccessor<Matrix44> wb(wb_);
        MaskedAccessor<Matrix44> wr(wr_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {
            Matrix44 a = wa[lane];
            Matrix44 b = wb[lane];
            // Need inlinable version for vectorization
            // Matrix44 r = a * b;
            Matrix44 r;
            inlinedMultMatrixMatrix(a, b, r);
            wr[lane] = r;
        }
    }
}

OSL_SHADEOP void
osl_div_mmm (void *r, void *a, void *b)
{
    MAT(r) = MAT(a) * MAT(b).inverse();
}


// flatten is workaround to enable inlining of non-inlined methods
OSL_SHADEOP OSL_CLANG_ATTRIBUTE(flatten) void
osl_div_w16mw16mw16m_masked(void *wr_, void *wa_, void * wb_, unsigned int mask_value)
{
    ConstWideAccessor<Matrix44> wa(wa_);
    ConstWideAccessor<Matrix44> wb(wb_);
    MaskedAccessor<Matrix44> wr(wr_, Mask(mask_value));

    int allAreAffine = 1;
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        OSL_OMP_PRAGMA(omp simd simdlen(wb.width))
        for(int lane=0; lane < wb.width; ++lane) {
            if (wr.mask().is_on(lane)) {
                Matrix44 m = wb[lane];
                if ((m.x[0][3] != 0.0f || m.x[1][3] != 0.0f || m.x[2][3] != 0.0f || m.x[3][3] != 1.0f)) {
                    allAreAffine = 0;
                }
            }
        }
    }

    if (allAreAffine) {
        OSL_INTEL_PRAGMA(forceinline recursive)
        {
            // Workaround clang omp loop analysis issue by using its native pragma
            // Suspect a different implementation of Matrix44 with proper inlining,
            // user defined copy constructor, and a POD compatible default constructor
            // would solve the issue
            OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wr.width))
            OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wr.width))
            for(int lane=0; lane < wr.width; ++lane) {
                Matrix44 a = wa[lane];
                Matrix44 b = wb[lane];
                // Need inlineable version
                //Matrix44 r = a * b.inverse();
                Matrix44 r;
                inlinedMultMatrixMatrix(a, affineInvert(b), r);
                wr[lane] = r;
            }
        }
    } else {
        for(int lane=0; lane < wr.width; ++lane) {
            if (wr.mask().is_on(lane)) {
                Matrix44 a = wa[lane];
                Matrix44 b = wb[lane];
                Matrix44 r = a * b.inverse();
                wr[lane] = r;
            }
        }
    }
}

OSL_SHADEOP void
osl_div_mmf (void *r, void *a, float b)
{
    MAT(r) = MAT(a) * (1.0f/b);
}

// flatten is workaround to enable inlining of non-inlined methods
OSL_SHADEOP OSL_CLANG_ATTRIBUTE(flatten) void
osl_div_w16mw16mw16f(void *wr_, void *wa_, void * wb_)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Matrix44> wa(wa_);
        ConstWideAccessor<float> wb(wb_);
        WideAccessor<Matrix44> wr(wr_);

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {
            Matrix44 a = wa[lane];
            float b = wb[lane];
            Matrix44 r = a * (1.0f/b);
            wr[lane] = r;
        }
    }
}

// flatten is workaround to enable inlining of non-inlined methods
OSL_SHADEOP OSL_CLANG_ATTRIBUTE(flatten) void
osl_div_w16mw16mw16f_masked(void *wr_, void *wa_, void * wb_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Matrix44> wa(wa_);
        ConstWideAccessor<float> wb(wb_);
        MaskedAccessor<Matrix44> wr(wr_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
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


// flatten is workaround to enable inlining of non-inlined methods
OSL_SHADEOP OSL_CLANG_ATTRIBUTE(flatten) void
osl_div_w16mw16fw16m_masked(void *wr_, void *wa_, void * wb_, unsigned int mask_value)
{
    ConstWideAccessor<float> wa(wa_);
    ConstWideAccessor<Matrix44> wb(wb_);
    MaskedAccessor<Matrix44> wr(wr_, Mask(mask_value));

    int allAreAffine = 1;
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        OSL_OMP_PRAGMA(omp simd simdlen(wb.width))
        for(int lane=0; lane < wb.width; ++lane) {
            if (wr.mask().is_on(lane)) {
                Matrix44 m = wb[lane];
                if ((m.x[0][3] != 0.0f || m.x[1][3] != 0.0f || m.x[2][3] != 0.0f || m.x[3][3] != 1.0f)) {
                    allAreAffine = 0;
                }
            }
        }
    }

    if (allAreAffine) {
        OSL_INTEL_PRAGMA(forceinline recursive)
        {
            // Workaround clang omp loop analysis issue by using its native pragma
            // Suspect a different implementation of Matrix44 with proper inlining,
            // user defined copy constructor, and a POD compatible default constructor
            // would solve the issue
            OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wr.width))
            OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wr.width))
            for(int lane=0; lane < wr.width; ++lane) {
                float a = wa[lane];
                Matrix44 b = wb[lane];
                Matrix44 r = a * affineInvert(b);
                wr[lane] = r;
            }
        }
    } else {
        for(int lane=0; lane < wr.width; ++lane) {
            if (wr.mask().is_on(lane)) {
                float a = wa[lane];
                Matrix44 b = wb[lane];
                Matrix44 r = a * b.inverse();
                wr[lane] = r;
            }
        }
    }
}


OSL_SHADEOP void
osl_transpose_mm (void *r, void *m)
{
    MAT(r) = MAT(m).transposed();
}

// flatten is workaround to enable inlining of non-inlined methods
OSL_SHADEOP OSL_CLANG_ATTRIBUTE(flatten) void
osl_transpose_w16mw16m(void *wr_, void *wm_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Matrix44> wm(wm_);
		WideAccessor<Matrix44> wr(wr_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Matrix44 m = wm[lane];
			// Call inlineable transposed
			//Matrix44 r = m.transposed();
			Matrix44 r = inlinedTransposed(m);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP OSL_CLANG_ATTRIBUTE(flatten) void
osl_transpose_w16mw16m_masked(void *wr_, void *wm_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Matrix44> wm(wm_);
        MaskedAccessor<Matrix44> wr(wr_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {
            Matrix44 m = wm[lane];
            // Call inlineable transposed
            //Matrix44 r = m.transposed();
            Matrix44 r = inlinedTransposed(m);
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


OSL_INLINE Mask
impl_get_uniform_from_matrix_batched (void *sgb_, MaskedAccessor<Matrix44> wrm, const char *from)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;
    if (USTR(from) == Strings::common ||
            USTR(from) == ctx->shadingsys().commonspace_synonym()) {
        Matrix44 ident;
        ident.makeIdentity();
        OSL_INTEL_PRAGMA(forceinline recursive)
        {
			OSL_OMP_PRAGMA(omp simd simdlen(wrm.width))
			for(int lane=0; lane < wrm.width; ++lane) {
				wrm[lane] = ident;
			}
    	}
        return Mask(true);
    }
    
	if (USTR(from) == Strings::shader) {
		ctx->batched_renderer()->get_matrix (sgb, wrm, sgb->varyingData().shader2common, sgb->varyingData().time);
		// NOTE: matching scalar version of code which ignores the renderservices return value
        return Mask(true);
	}
	if (USTR(from) == Strings::object) {
		ctx->batched_renderer()->get_matrix (sgb, wrm, sgb->varyingData().object2common, sgb->varyingData().time);
		// NOTE: matching scalar version of code which ignores the renderservices return value
        return Mask(true);
	}
	
	Mask succeeded = ctx->batched_renderer()->get_matrix (sgb, wrm, USTR(from), sgb->varyingData().time);
    Mask failedLanes = succeeded.invert(wrm.mask());
    if (failedLanes.any_on())
    {
        Matrix44 ident;
        ident.makeIdentity();
        OSL_INTEL_PRAGMA(forceinline recursive)
        {
            OSL_OMP_PRAGMA(omp simd simdlen(wrm.width))
            for(int lane=0; lane < wrm.width; ++lane) {
                if (failedLanes[lane]) {
                    wrm[lane] = ident;
                }
            }
        }
		ShadingContext *ctx = sgb->uniform().context;
		if (ctx->shadingsys().unknown_coordsys_error())
		{
			ASSERT(wrm.mask().any_on());
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

static OSL_INLINE Mask
impl_get_uniform_to_inverse_matrix_batched (void *sgb_, MaskedAccessor<Matrix44> wrm, const char *to)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = sgb->uniform().context;
    
    if (USTR(to) == Strings::common ||
            USTR(to) == ctx->shadingsys().commonspace_synonym()) {
    	
        Matrix44 ident;
        ident.makeIdentity();
        OSL_INTEL_PRAGMA(forceinline recursive)
        {
            OSL_OMP_PRAGMA(omp simd simdlen(wrm.width))
            for(int lane=0; lane < wrm.width; ++lane) {
                wrm[lane] = ident;
            }
        }
        return Mask(true);
    }
    if (USTR(to) == Strings::shader) {
    	ctx->batched_renderer()->get_inverse_matrix (sgb, wrm, sgb->varyingData().shader2common, sgb->varyingData().time);
		// NOTE: matching scalar version of code which ignores the renderservices return value
        return Mask(true);
    }
    if (USTR(to) == Strings::object) {
    	ctx->batched_renderer()->get_inverse_matrix (sgb, wrm, sgb->varyingData().object2common, sgb->varyingData().time);
		// NOTE: matching scalar version of code which ignores the renderservices return value
        return Mask(true);
    }

	// Based on the 1 function that calls this function
	// the results of the failed data lanes will get overwritten
	// so no need to make sure that the values are valid (assuming FP exceptions are disabled)
    Mask succeeded = ctx->batched_renderer()->get_inverse_matrix (sgb, wrm, USTR(to), sgb->varyingData().time);

    Mask failedLanes = succeeded.invert(wrm.mask());
    if (failedLanes.any_on())
	{
        Matrix44 ident;
        ident.makeIdentity();
        OSL_INTEL_PRAGMA(forceinline recursive)
        {
            OSL_OMP_PRAGMA(omp simd simdlen(wrm.width))
            for(int lane=0; lane < wrm.width; ++lane) {
                if (failedLanes[lane]) {
                    wrm[lane] = ident;
                }
            }
        }
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

// flatten is workaround to enable inlining of non-inlined methods
template<typename ResultAccessorT, typename FromAccessorT, typename ToAccessorT>
static OSL_INLINE OSL_CLANG_ATTRIBUTE(flatten) void
impl_wide_mat_multiply(ResultAccessorT wresult, FromAccessorT wfrom, ToAccessorT wto)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        static constexpr int width = wresult.width;
        // No savings from using a WeakMask
        OSL_OMP_PRAGMA(omp simd simdlen(width))
        for(int lane=0; lane < wresult.width; ++lane) {
            Matrix44 mat_From = wfrom[lane];
            Matrix44 mat_To = wto[lane];

            // Need to call inlinable version
            //Matrix44 result = mat_From * mat_To;
            Matrix44 result;
            inlinedMultMatrixMatrix(mat_From, mat_To, result);

            wresult[lane] = result;
        }
    }
}

static OSL_INLINE Mask
impl_get_varying_from_matrix_batched(ShaderGlobalsBatch *sgb, ShadingContext *ctx, ConstWideAccessor<ustring> wFrom, MaskedAccessor<Matrix44> wMfrom)
{
    // Deal with a varying 'from' space
    ustring commonspace_synonym = ctx->shadingsys().commonspace_synonym();

	Mask commonSpaceMask(false);
	Mask shaderSpaceMask(false);
	Mask objectSpaceMask(false);
	Mask namedSpaceMask(false);

    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        OSL_OMP_PRAGMA(omp simd simdlen(wFrom.width))
        for(int lane=0; lane < wFrom.width; ++lane) {
            if (wMfrom.mask()[lane]) {
                ustring from = wFrom[lane];
                if (from == Strings::common ||
                    from == commonspace_synonym) {
                    commonSpaceMask.set_on(lane);
                } else if (from == Strings::shader) {
                    shaderSpaceMask.set_on(lane);
                } else if (from == Strings::object) {
                    objectSpaceMask.set_on(lane);
                } else {
                    namedSpaceMask.set_on(lane);
                }
            }
        }
    }

	if (commonSpaceMask.any_on())
	{
        Matrix44 ident;
        ident.makeIdentity();
        OSL_INTEL_PRAGMA(forceinline recursive)
        {
            MaskedAccessor<Matrix44> mfrom(wMfrom.data(), commonSpaceMask);
            OSL_OMP_PRAGMA(omp simd simdlen(mfrom.width))
            for(int lane=0; lane < mfrom.width; ++lane) {
                mfrom[lane] = ident;
            }
	    }
	}
    const auto & sgbv = sgb->varyingData();
	if (shaderSpaceMask.any_on())
	{
		MaskedAccessor<Matrix44> mfrom(wMfrom.data(), shaderSpaceMask);
        ctx->batched_renderer()->get_matrix (sgb, mfrom, sgbv.shader2common, sgbv.time);
		// NOTE: matching scalar version of code which ignores the renderservices return value
	}
	if (objectSpaceMask.any_on())
	{
		MaskedAccessor<Matrix44> mfrom(wMfrom.data(), objectSpaceMask);
        ctx->batched_renderer()->get_matrix (sgb, mfrom, sgbv.object2common, sgbv.time);
		// NOTE: matching scalar version of code which ignores the renderservices return value
	}
	// Only named lookups can fail, so we can just subtract those lanes
	Mask succeeded(wMfrom.mask());
	if (namedSpaceMask.any_on())
	{
		MaskedAccessor<Matrix44> mfrom(wMfrom.data(), namedSpaceMask);

        Mask success = ctx->batched_renderer()->get_matrix (sgb, mfrom, wFrom, sgbv.time);

        Mask failedLanes = success.invert(namedSpaceMask);
        if (failedLanes.any_on())
    	{
            Matrix44 ident;
            ident.makeIdentity();
            MaskedAccessor<Matrix44> mto_failed(wMfrom.data(), failedLanes);
            OSL_INTEL_PRAGMA(forceinline recursive)
            {
                OSL_OMP_PRAGMA(omp simd simdlen(mto_failed.width))
                for(int lane=0; lane < mto_failed.width; ++lane) {
                    mto_failed[lane] = ident;
                }
            }
    		if (ctx->shadingsys().unknown_coordsys_error())
    		{
                for(int lane=0; lane < mto_failed.width; ++lane) {
                	if (failedLanes[lane]) {
                		ustring from = wFrom[lane];
                		ctx->error (Mask(Lane(lane)), "Unknown transformation \"%s\"", from);
                	}
                }
    		}

    		// Remove any failed lanes from the success mask
    		succeeded &= ~failedLanes;
    	}
	}
	return succeeded;
}

OSL_SHADEOP void
osl_prepend_matrix_from_w16ms_batched (void *sgb, void *wr, const char *from)
{
    Wide<Matrix44> wMfrom;
    MaskedAccessor<Matrix44> from_matrix(wMfrom, Mask(true));
    /*Mask succeeded =*/
    impl_get_uniform_from_matrix_batched ((ShaderGlobalsBatch *)sgb, from_matrix, from);

    WideAccessor<Matrix44> wrm(wr);

    impl_wide_mat_multiply(wrm, from_matrix, wrm);
}

OSL_SHADEOP void
osl_prepend_matrix_from_w16ms_masked (void *sgb, void *wr, const char *from, int mask_value)
{
    Wide<Matrix44> wMfrom;
    MaskedAccessor<Matrix44> from_matrix(wMfrom, Mask(mask_value));
    /*Mask succeeded =*/
    impl_get_uniform_from_matrix_batched ((ShaderGlobalsBatch *)sgb, from_matrix, from);

    MaskedAccessor<Matrix44> wrm(wr, Mask(mask_value));

    impl_wide_mat_multiply(wrm, from_matrix, wrm);
}

OSL_SHADEOP void
osl_prepend_matrix_from_w16mw16s_batched (void *sgb_, void *wr, void * w_from_name)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;

    ConstWideAccessor<ustring> wFromName(w_from_name);

    Wide<Matrix44> wMfrom;
    MaskedAccessor<Matrix44> from_matrix(wMfrom, Mask(true));
    /*Mask succeeded =*/
    impl_get_varying_from_matrix_batched(sgb, ctx, wFromName, from_matrix);

    WideAccessor<Matrix44> wrm(wr);
    impl_wide_mat_multiply(wrm, from_matrix, wrm);
}

OSL_SHADEOP void
osl_prepend_matrix_from_w16mw16s_masked (void *sgb_, void *wr, void * w_from_name, int mask_value)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;

    ConstWideAccessor<ustring> wFromName(w_from_name);

    Wide<Matrix44> wMfrom;
    MaskedAccessor<Matrix44> from_matrix(wMfrom, Mask(mask_value));
    /*Mask succeeded =*/
    impl_get_varying_from_matrix_batched(sgb, ctx, wFromName, from_matrix);

    MaskedAccessor<Matrix44> wrm(wr, Mask(mask_value));

    impl_wide_mat_multiply(wrm, from_matrix, wrm);
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


static OSL_INLINE Mask
impl_get_varying_to_matrix_batched(
	ShaderGlobalsBatch *sgb,
	ShadingContext *ctx,
	ConstWideAccessor<ustring> wTo,
	MaskedAccessor<Matrix44> wMto)
{
    // Deal with a varying 'to' space
    ustring commonspace_synonym = ctx->shadingsys().commonspace_synonym();

	Mask commonSpaceMask(false);
	Mask shaderSpaceMask(false);
	Mask objectSpaceMask(false);
	Mask namedSpaceMask(false);

    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        OSL_OMP_PRAGMA(omp simd simdlen(wTo.width))
        for(int lane=0; lane < wTo.width; ++lane) {
            if (wMto.mask()[lane]) {
                ustring to = wTo[lane];

                if (to == Strings::common ||
                    to == commonspace_synonym) {
                    commonSpaceMask.set_on(lane);
                } else if (to == Strings::shader) {
                    shaderSpaceMask.set_on(lane);
                } else if (to == Strings::object) {
                    objectSpaceMask.set_on(lane);
                } else {
                    namedSpaceMask.set_on(lane);
                }
            }
        }
    }

	if (commonSpaceMask.any_on())
	{
        Matrix44 ident;
        ident.makeIdentity();
        OSL_INTEL_PRAGMA(forceinline recursive)
        {
            MaskedAccessor<Matrix44> mto(wMto.data(), commonSpaceMask);
            OSL_OMP_PRAGMA(omp simd simdlen(mto.width))
            for(int lane=0; lane < mto.width; ++lane) {
                mto[lane] = ident;
            }
	    }
	}
    const auto & sgbv = sgb->varyingData();
	if (shaderSpaceMask.any_on())
	{
		MaskedAccessor<Matrix44> mto(wMto.data(), shaderSpaceMask);
        ctx->batched_renderer()->get_inverse_matrix (sgb, mto, sgbv.shader2common, sgbv.time);
		// NOTE: matching scalar version of code which ignores the renderservices return value
	}
	if (objectSpaceMask.any_on())
	{
		MaskedAccessor<Matrix44> mto(wMto.data(), objectSpaceMask);
        ctx->batched_renderer()->get_inverse_matrix (sgb, mto, sgbv.object2common, sgbv.time);
		// NOTE: matching scalar version of code which ignores the renderservices return value
	}
	// Only named lookups can fail, so we can just subtract those lanes
	Mask succeeded(wMto.mask());
	if (namedSpaceMask.any_on())
	{
		MaskedAccessor<Matrix44> mto(wMto.data(), namedSpaceMask);

        Mask success = ctx->batched_renderer()->get_inverse_matrix (sgb, mto, wTo, sgbv.time);

        Mask failedLanes = success.invert(namedSpaceMask);
        if (failedLanes.any_on())
    	{
            Matrix44 ident;
            ident.makeIdentity();
			MaskedAccessor<Matrix44> mto(wMto.data(), failedLanes);
	        OSL_INTEL_PRAGMA(forceinline recursive)
	        {
                OSL_OMP_PRAGMA(omp simd simdlen(mto.width))
                for(int lane=0; lane < mto.width; ++lane) {
                    mto[lane] = ident;
                }
	        }
    		if (ctx->shadingsys().unknown_coordsys_error())
    		{
                for(int lane=0; lane < mto.width; ++lane) {
                	if (failedLanes[lane]) {
                		ustring to = wTo[lane];
                		ctx->error (Mask(Lane(lane)), "Unknown transformation \"%s\"", to);
                	}
                }
    		}

    		// Remove any failed lanes from the success mask
    		succeeded &= ~failedLanes;
    	}
	}
	return succeeded;
}


// flatten is workaround to enable inlining of non-inlined methods
OSL_INLINE OSL_CLANG_ATTRIBUTE(flatten) Mask
impl_get_uniform_from_to_matrix_batched (ShaderGlobalsBatch *sgb, MaskedAccessor<Matrix44> wrm, const char *from, const char *to)
{
    Wide<Matrix44> wMfrom, wMto;
    MaskedAccessor<Matrix44> from_matrix(wMfrom, wrm.mask());
    Mask succeeded = impl_get_uniform_from_matrix_batched (sgb, from_matrix, from);

    // NOTE: even if we failed to get a from matrix, it should have been set to
    // identity, so we still need to try to get the to matrix for the original mask
    MaskedAccessor<Matrix44> to_matrix(wMto, wrm.mask());
    succeeded &= impl_get_uniform_to_inverse_matrix_batched (sgb, to_matrix, to);

    impl_wide_mat_multiply(wrm, from_matrix, to_matrix);
    return succeeded;
}

OSL_SHADEOP int
osl_get_from_to_matrix_w16mss_batched (void *sgb_, void *wr, const char *from, const char *to)
{
	ShaderGlobalsBatch * sgb = (ShaderGlobalsBatch *)sgb_;
	MaskedAccessor<Matrix44> wrm(wr,Mask(true));
    return impl_get_uniform_from_to_matrix_batched(sgb, wrm, from, to).value();
}

OSL_SHADEOP int
osl_get_from_to_matrix_w16mss_masked (void *sgb_, void *wr, const char *from, const char *to, int mask_value)
{
	ShaderGlobalsBatch * sgb = (ShaderGlobalsBatch *)sgb_;
	MaskedAccessor<Matrix44> wrm(wr,Mask(mask_value));
    return impl_get_uniform_from_to_matrix_batched(sgb, wrm, from, to).value();
}


OSL_SHADEOP int
osl_get_from_to_matrix_w16msw16s_batched (void *sgb_, void *wr, const char *from, void * w_to_ptr)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;
    Wide<Matrix44> wMfrom;
    MaskedAccessor<Matrix44> from_matrix(wMfrom, Mask(true));
    Mask succeeded = impl_get_uniform_from_matrix_batched (sgb, from_matrix, from);

    ConstWideAccessor<ustring> wToSpace(w_to_ptr);
    Wide<Matrix44> wMto;
    // NOTE: even if we failed to get a from matrix, it should have been set to
    // identity, so we still need to try to get the to matrix for the original mask
    MaskedAccessor<Matrix44> to_matrix(wMto, Mask(true));
    succeeded &= impl_get_varying_to_matrix_batched(sgb, ctx, wToSpace, to_matrix);

    // No savings from using succeeded
    impl_wide_mat_multiply(WideAccessor<Matrix44>(wr), from_matrix, to_matrix);
    return succeeded.value();
}

OSL_SHADEOP int
osl_get_from_to_matrix_w16msw16s_masked (void *sgb_, void *wr, const char *from, void * w_to_ptr, int mask_value)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;
    Wide<Matrix44> wMfrom;
    MaskedAccessor<Matrix44> from_matrix(wMfrom, Mask(mask_value));
    Mask succeeded = impl_get_uniform_from_matrix_batched (sgb, from_matrix, from);

    ConstWideAccessor<ustring> wToSpace(w_to_ptr);
    Wide<Matrix44> wMto;
    // NOTE: even if we failed to get a from matrix, it should have been set to
    // identity, so we still need to try to get the to matrix for the original mask
    MaskedAccessor<Matrix44> to_matrix(wMto, Mask(mask_value));
    succeeded &= impl_get_varying_to_matrix_batched(sgb, ctx, wToSpace, to_matrix);

    MaskedAccessor<Matrix44> wrm(wr, Mask(mask_value));
    impl_wide_mat_multiply(wrm, from_matrix, to_matrix);
    return succeeded.value();
}



OSL_SHADEOP int
osl_get_from_to_matrix_w16mw16ss_batched (void *sgb_, void *wr, void  *w_from_ptr, const char * to)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;

    ConstWideAccessor<ustring> wFromName(w_from_ptr);

    Wide<Matrix44> wMto;
    MaskedAccessor<Matrix44> to_matrix(wMto, Mask(true));
    Mask succeeded = impl_get_uniform_to_inverse_matrix_batched (sgb, to_matrix, to);

    Wide<Matrix44> wMfrom;
    // NOTE: even if we failed to get a to matrix, it should have been set to
    // identity, so we still need to try to get the to matrix for the original mask
    MaskedAccessor<Matrix44> from_matrix(wMfrom, Mask(true));
    succeeded &= impl_get_varying_from_matrix_batched(sgb, ctx, wFromName, from_matrix);

    // No savings from using succeeded
    impl_wide_mat_multiply(WideAccessor<Matrix44>(wr), from_matrix, to_matrix);
    return succeeded.value();
}

OSL_SHADEOP int
osl_get_from_to_matrix_w16mw16ss_masked (void *sgb_, void *wr, void  *w_from_ptr, const char * to, int mask_value)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;

    ConstWideAccessor<ustring> wFromName(w_from_ptr);

    Wide<Matrix44> wMto;
    MaskedAccessor<Matrix44> to_matrix(wMto, Mask(mask_value));
    Mask succeeded = impl_get_uniform_to_inverse_matrix_batched (sgb, to_matrix, to);

    Wide<Matrix44> wMfrom;
    // NOTE: even if we failed to get a to matrix, it should have been set to
    // identity, so we still need to try to get the to matrix for the original mask
    MaskedAccessor<Matrix44> from_matrix(wMfrom, Mask(mask_value));
    succeeded &= impl_get_varying_from_matrix_batched(sgb, ctx, wFromName, from_matrix);

    MaskedAccessor<Matrix44> wrm(wr, Mask(mask_value));
    impl_wide_mat_multiply(wrm, from_matrix, to_matrix);
    return succeeded.value();
}


OSL_SHADEOP int
osl_get_from_to_matrix_w16mw16sw16s_batched (void *sgb_, void *wr, void  *w_from_ptr, void * w_to_ptr)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;

    ConstWideAccessor<ustring> wFromName(w_from_ptr);

    Wide<Matrix44> wMfrom;
    MaskedAccessor<Matrix44> from_matrix(wMfrom, Mask(true));
    Mask succeeded = impl_get_varying_from_matrix_batched(sgb, ctx, wFromName, from_matrix);

    ConstWideAccessor<ustring> wToSpace(w_to_ptr);
    Wide<Matrix44> wMto;
    // NOTE: even if we failed to get a from matrix, it should have been set to
    // identity, so we still need to try to get the to matrix for the original mask
    MaskedAccessor<Matrix44> to_matrix(wMto, Mask(true));
    succeeded &= impl_get_varying_to_matrix_batched(sgb, ctx, wToSpace, to_matrix);

    // No savings from using succeeded
    impl_wide_mat_multiply(WideAccessor<Matrix44>(wr), from_matrix, to_matrix);
    return succeeded.value();
}

OSL_SHADEOP int
osl_get_from_to_matrix_w16mw16sw16s_masked (void *sgb_, void *wr, void  *w_from_ptr, void * w_to_ptr, int mask_value)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;

    ConstWideAccessor<ustring> wFromName(w_from_ptr);

    Wide<Matrix44> wMfrom;
    MaskedAccessor<Matrix44> from_matrix(wMfrom, Mask(mask_value));
    Mask succeeded = impl_get_varying_from_matrix_batched(sgb, ctx, wFromName, from_matrix);

    ConstWideAccessor<ustring> wToSpace(w_to_ptr);
    Wide<Matrix44> wMto;
    // NOTE: even if we failed to get a from matrix, it should have been set to
    // identity, so we still need to try to get the to matrix for the original mask
    MaskedAccessor<Matrix44> to_matrix(wMto, Mask(mask_value));
    succeeded &= impl_get_varying_to_matrix_batched(sgb, ctx, wToSpace, to_matrix);

    MaskedAccessor<Matrix44> wrm(wr, Mask(mask_value));
    impl_wide_mat_multiply(wrm, from_matrix, to_matrix);
    return succeeded.value();
}


// point = M * point
inline void osl_transform_vmv(void *result, const Matrix44 &M, void* v_)
{
	//std::cout << "osl_transform_vmv" << std::endl;
   const Vec3 &v = VEC(v_);
   robust_multVecMatrix (M, v, VEC(result));
}

OSL_INLINE void
avoidAliasingRobustMultVecMatrix(const Matrix44 &M, const Vec3 &src, Vec3 &dst)
{
	// Avoid aliasing issues when mixing data member access vs.
	// reinterpret casted array based access.
	// Legally, compiler could assume no alaising because they are technically
	// different types.  As they actually over the same memory incorrect
	// code generation can ensue
#if 0
   float a = src[0] * M[0][0] + src[1] * M[1][0] + src[2] * M[2][0] + M[3][0];
    float b = src[0] * M[0][1] + src[1] * M[1][1] + src[2] * M[2][1] + M[3][1];
    float c = src[0] * M[0][2] + src[1] * M[1][2] + src[2] * M[2][2] + M[3][2];
    float w = src[0] * M[0][3] + src[1] * M[1][3] + src[2] * M[2][3] + M[3][3];
#else
	float a = src.x * M[0][0] + src.y * M[1][0] + src.z * M[2][0] + M[3][0];
	float b = src.x * M[0][1] + src.y * M[1][1] + src.z * M[2][1] + M[3][1];
	float c = src.x * M[0][2] + src.y * M[1][2] + src.z * M[2][2] + M[3][2];
	float w = src.x * M[0][3] + src.y * M[1][3] + src.z * M[2][3] + M[3][3];
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
}

OSL_INLINE void
avoidAliasingRobustMultVecMatrix(const Matrix44 &M, const Dual2<Vec3> &src, Dual2<Vec3> &dst)
{
	// Avoid aliasing issues when mixing data member access vs.
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
			// Rearrange into a Vec3<Dual2<float> >
			Imath::Vec3<Dual2<float> > din, dout;

			din.x.set (src.val().x, src.dx().x, src.dy().x);
			din.y.set (src.val().y, src.dx().y, src.dy().y);
			din.z.set (src.val().z, src.dx().z, src.dy().z);

			Dual2<float> a = din.x * M[0][0] + din.y * M[1][0] + din.z * M[2][0] + M[3][0];
			Dual2<float> b = din.x * M[0][1] + din.y * M[1][1] + din.z * M[2][1] + M[3][1];
			Dual2<float> c = din.x * M[0][2] + din.y * M[1][2] + din.z * M[2][2] + M[3][2];
			Dual2<float> w = din.x * M[0][3] + din.y * M[1][3] + din.z * M[2][3] + M[3][3];
#endif


			if (w.val() != 0.0f) {
			   dout.x = a / w;
			   dout.y = b / w;
			   dout.z = c / w;
			} else {
			   dout.x = 0.0f;
			   dout.y = 0.0f;
			   dout.z = 0.0f;
			}

			// Rearrange back into Dual2<Vec3>
#if 0
			out.set (Vec3 (dout[0].val(), dout[1].val(), dout[2].val()),
					 Vec3 (dout[0].dx(),  dout[1].dx(),  dout[2].dx()),
					 Vec3 (dout[0].dy(),  dout[1].dy(),  dout[2].dy()));
#else
			dst.set (Vec3 (dout.x.val(), dout.y.val(), dout.z.val()),
					 Vec3 (dout.x.dx(),  dout.y.dx(),  dout.z.dx()),
					 Vec3 (dout.x.dy(),  dout.y.dy(),  dout.z.dy()));
#endif
}

OSL_INLINE void
avoidAliasingMultDirMatrix (const Matrix44 &M, const Vec3 &src, Vec3 &dst)
{
	float a = src.x * M.x[0][0] + src.y * M.x[1][0] + src.z * M.x[2][0];
	float b = src.x * M.x[0][1] + src.y * M.x[1][1] + src.z * M.x[2][1];
	float c = src.x * M.x[0][2] + src.y * M.x[1][2] + src.z * M.x[2][2];

	dst.x = a;
	dst.y = b;
	dst.z = c;
    
}


inline void osl_transform_dvmdv(void *result, const Matrix44 &M, void* v_)
{
   const Dual2<Vec3> &v = DVEC(v_);
   robust_multVecMatrix (M, v, DVEC(result));
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

/// Multiply a matrix times a vector with derivatives to obtain
/// a transformed vector with derivatives.
inline void
avoidAliasingMultDirMatrix (const Matrix44 &M, const Dual2<Vec3> &in, Dual2<Vec3> &out)
{
	avoidAliasingMultDirMatrix(M, in.val(), out.val());
	avoidAliasingMultDirMatrix(M, in.dx(), out.dx());
	avoidAliasingMultDirMatrix(M, in.dy(), out.dy());
}

// normal = M * normal
inline void osl_transformn_vmv(void *result, const Matrix44 &M, void* v_)
{
   const Vec3 &v = VEC(v_);
   M.inverse().transposed().multDirMatrix (v, VEC(result));
}

static inline void
osl_transformn_dvmdv(void *result, const Matrix44 &M, void* v_)
{
   const Dual2<Vec3> &v = DVEC(v_);
   multDirMatrix (M.inverse().transposed(), v, DVEC(result));
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


// NOTE:  For batched transforms, a different dispatch approach is used.
// Instead of calling a single transform_triple function with lots of
// conditionals to select/dispatch the correct code, a 2 steps are taken.
// First call an explicitly named function (osl_build_transform_matrix_??_masked)
// is called that represents the uniformity for the different from & to spaces
// is called building a transform matrix.
// Second call an explicitly named function (osl_transform_[point|vector|normal]_??_masked)
// is called that represents the uniformity and data types of the src and destination triples.
// Also zeroing of derives is left to the code generator.

OSL_SHADEOP int
osl_build_transform_matrix_ss_masked (void *sgb_, void *WM_,
					  void * from_, void * to_, unsigned int mask_value)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;

    Mask mask(mask_value);
    MaskedAccessor<Matrix44> mm(WM_, mask);
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;

    ustring from = USTR(from_);
    ustring to = USTR(to_);

    Mask succeeded;
    // Avoid matrix concatenation if possible by detecting when the
    // adjacent matrix would be identity
    // We don't expect both from and to == common, so we are not
    // optimizing for it
    if (from == Strings::common ||
            from == ctx->shadingsys().commonspace_synonym()) {
        succeeded = impl_get_uniform_to_inverse_matrix_batched (sgb, mm, to.c_str());
    } else if (to == Strings::common ||
            to == ctx->shadingsys().commonspace_synonym()) {
        succeeded = impl_get_uniform_from_matrix_batched(sgb, mm, from.c_str());
    } else {
        succeeded = impl_get_uniform_from_to_matrix_batched (sgb, mm, from.c_str(),
										 to.c_str());
    }
    return succeeded.value();
}

OSL_SHADEOP int
osl_build_transform_matrix_w16ss_masked (void *sgb_, void *WM_,
					  void * wfrom_, void * to_, unsigned int mask_value)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;

    Mask mask(mask_value);
    MaskedAccessor<Matrix44> wrm(WM_, mask);

    ConstWideAccessor<ustring> wfrom_space(wfrom_);

    ustring to_space = USTR(to_);

    Wide<Matrix44> wMfrom, wMto;
    MaskedAccessor<Matrix44> from_matrix(wMfrom, wrm.mask());
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;

    Mask succeeded = impl_get_varying_from_matrix_batched (sgb, ctx, wfrom_space, from_matrix);
    MaskedAccessor<Matrix44> to_matrix(wMto, wrm.mask() & succeeded);
    succeeded &= impl_get_uniform_to_inverse_matrix_batched (sgb, to_matrix, to_space.c_str());

    impl_wide_mat_multiply(wrm, from_matrix, to_matrix);
    return succeeded.value();
}

OSL_SHADEOP int
osl_build_transform_matrix_sw16s_masked (void *sgb_, void *WM_,
					  void * from_, void * wto_, unsigned int mask_value)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;

    Mask mask(mask_value);
    MaskedAccessor<Matrix44> wrm(WM_, mask);

    ustring from = USTR(from_);
    ConstWideAccessor<ustring> wto_space(wto_);

    Wide<Matrix44> wMfrom, wMto;
    MaskedAccessor<Matrix44> from_matrix(wMfrom, wrm.mask());
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;

    Mask succeeded = impl_get_uniform_from_matrix_batched (sgb, from_matrix, from.c_str());
    MaskedAccessor<Matrix44> to_matrix(wMto, wrm.mask() & succeeded);
    succeeded &= impl_get_varying_to_matrix_batched(sgb, ctx, wto_space, to_matrix);

    impl_wide_mat_multiply(wrm, from_matrix, to_matrix);

    return succeeded.value();
}

OSL_SHADEOP int
osl_build_transform_matrix_w16sw16s_masked (void *sgb_, void *WM_,
					  void * wfrom_, void * wto_, unsigned int mask_value)
{
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;

    Mask mask(mask_value);
    MaskedAccessor<Matrix44> wrm(WM_, mask);

    ConstWideAccessor<ustring> wfrom_space(wfrom_);
    ConstWideAccessor<ustring> wto_space(wto_);

    Wide<Matrix44> wMfrom, wMto;
    MaskedAccessor<Matrix44> from_matrix(wMfrom, wrm.mask());
    ShadingContext *ctx = (ShadingContext *)sgb->uniform().context;

    Mask succeeded = impl_get_varying_from_matrix_batched (sgb, ctx, wfrom_space, from_matrix);
    MaskedAccessor<Matrix44> to_matrix(wMto, wrm.mask() & succeeded);
    succeeded &= impl_get_varying_to_matrix_batched(sgb, ctx, wto_space, to_matrix);

    impl_wide_mat_multiply(wrm, from_matrix, to_matrix);

    return succeeded.value();
}

template <typename InputAccessorT>
static OSL_INLINE void
impl_copy_untransformed_lanes(
	InputAccessorT inVec,
	void * Pout, Mask succeeded, Mask op_mask)
{
	static constexpr int width = InputAccessorT::width;
	typedef typename InputAccessorT::value_type data_type;
    // if Pin != Pout, we still need to copy inactive data over to Pout
    // Handle cleaning up any data lanes that did not succeed
    if (((void *)&inVec.data() != Pout))
    {
        // For any lanes we failed to get a matrix for
        // just copy the input to the output values
        // NOTE:  As we only only want to copy lanes that failed,
        // we will invert our success mask
        Mask failed = succeeded.invert() & op_mask;
        if (__builtin_expect(failed.any_on(), 0)) {
            OSL_INTEL_PRAGMA(forceinline recursive)
            {
                MaskedAccessor<data_type, width> failedOutVec(Pout, failed);
                OSL_OMP_PRAGMA(omp simd simdlen(width))
                for(int i=0; i< failedOutVec.width; ++i)
                {
                    failedOutVec[i] = inVec[i];
                }
            }
        }
    }
}

template <typename InputAccessorT, typename MatrixAccessorT>
static OSL_INLINE void
impl_transform_point_masked(void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	static constexpr int width = InputAccessorT::width;
	typedef typename InputAccessorT::value_type data_type;

	// ignore derivs because output doesn't need it
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		Mask mask(mask_value);
		Mask succeeded(mask_transform);

		InputAccessorT inPoints(Pin);
		// only operate on active lanes
		Mask activeMask = mask & succeeded;

		MaskedAccessor<data_type, width> wresult(Pout, activeMask);
		MatrixAccessorT wM(transform);

		// Transform with Vector semantics
		OSL_OMP_PRAGMA(omp simd simdlen(width))
		for(int i=0; i < wresult.width; ++i)
		{
			const Matrix44 m = wM[i];
			const data_type v = inPoints[i];
			data_type r;

			// Do to illegal aliasing in OpenEXR version
			// we call our own flavor without aliasing
			robust_multVecMatrix(m, v, r);

			wresult[i] = r;
		}

		impl_copy_untransformed_lanes(inPoints, Pout, succeeded, mask);
	}
}

OSL_SHADEOP void
osl_transform_point_vw16vm_masked (void *Pin,
                      void *Pout,
                      void * transform, unsigned int mask_transform, unsigned int mask_value)
{
    // TODO: see if we can get gen_transform to call the vvm version then do a masked broadcast
    impl_transform_point_masked<ConstUniformAccessor<Vec3>, ConstUniformAccessor<Matrix44>> (Pin, Pout, transform, mask_transform, mask_value);
}

OSL_SHADEOP void
osl_transform_point_vw16vw16m_masked (void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	impl_transform_point_masked<ConstUniformAccessor<Vec3>, ConstWideAccessor<Matrix44>> (Pin, Pout, transform, mask_transform, mask_value);
}

OSL_SHADEOP void
osl_transform_point_w16vw16vw16m_masked (void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	impl_transform_point_masked<ConstWideAccessor<Vec3>, ConstWideAccessor<Matrix44>> (Pin, Pout, transform, mask_transform, mask_value);
}

OSL_SHADEOP void
osl_transform_point_w16dvw16dvw16m_masked (void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	impl_transform_point_masked<ConstWideAccessor<Dual2<Vec3>>, ConstWideAccessor<Matrix44>> (Pin, Pout, transform, mask_transform, mask_value);
}

OSL_SHADEOP void
osl_transform_point_w16vw16vm_masked (void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	impl_transform_point_masked<ConstWideAccessor<Vec3>, ConstUniformAccessor<Matrix44>> (Pin, Pout, transform, mask_transform, mask_value);
}

OSL_SHADEOP void
osl_transform_point_w16dvw16dvm_masked (void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	impl_transform_point_masked<ConstWideAccessor<Dual2<Vec3>>, ConstUniformAccessor<Matrix44>> (Pin, Pout, transform, mask_transform, mask_value);
}


template <typename InputAccessorT, typename MatrixAccessorT>
static OSL_INLINE void
impl_transform_vector_masked (void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	static constexpr int width = InputAccessorT::width;
	typedef typename InputAccessorT::value_type data_type;

    // ignore derivs because output doesn't need it
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        Mask mask(mask_value);
        Mask succeeded(mask_transform);

        InputAccessorT inPoints(Pin);
        // only operate on active lanes
        Mask activeMask = mask & succeeded;

    	MaskedAccessor<data_type, width> wresult(Pout, activeMask);
    	MatrixAccessorT wM(transform);

    	// Transform with Vector semantics
		OSL_OMP_PRAGMA(omp simd simdlen(width))
		for(int i=0; i < wresult.width; ++i)
		{
			Matrix44 m = wM[i];
			data_type v = inPoints[i];
			data_type r;

			// Do to illegal aliasing in OpenEXR version
			// we call our own flavor without aliasing
			//M.multDirMatrix (v, VEC(result));
			avoidAliasingMultDirMatrix(m, v, r);

			wresult[i] = r;
		}

	    impl_copy_untransformed_lanes(inPoints, Pout, succeeded, mask);
    }
}

OSL_SHADEOP void
osl_transform_vector_vw16vm_masked(void *Pin,
                      void *Pout,
                      void * transform, unsigned int mask_transform, unsigned int mask_value)
{
    // TODO: see if we can get gen_transform to call the vvm version then do a masked broadcast
    impl_transform_vector_masked<ConstUniformAccessor<Vec3>, ConstUniformAccessor<Matrix44>>
        (Pin, Pout, transform, mask_transform, mask_value);
}

OSL_SHADEOP void
osl_transform_vector_vw16vw16m_masked(void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	impl_transform_vector_masked<ConstUniformAccessor<Vec3>, ConstWideAccessor<Matrix44>>
		(Pin, Pout, transform, mask_transform, mask_value);
}

OSL_SHADEOP void
osl_transform_vector_w16vw16vw16m_masked (void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	impl_transform_vector_masked<ConstWideAccessor<Vec3>, ConstWideAccessor<Matrix44>>
		(Pin, Pout, transform, mask_transform, mask_value);
}

OSL_SHADEOP void
osl_transform_vector_w16dvw16dvw16m_masked(void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	impl_transform_vector_masked<ConstWideAccessor<Dual2<Vec3>>, ConstWideAccessor<Matrix44>>
		(Pin, Pout, transform, mask_transform, mask_value);
}


OSL_SHADEOP void
osl_transform_vector_w16vw16vm_masked (void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	impl_transform_vector_masked<ConstWideAccessor<Vec3>, ConstUniformAccessor<Matrix44>>
		(Pin, Pout, transform, mask_transform, mask_value);
}


OSL_SHADEOP void
osl_transform_vector_w16dvw16dvm_masked(void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	impl_transform_vector_masked<ConstWideAccessor<Dual2<Vec3>>, ConstUniformAccessor<Matrix44>>
		(Pin, Pout, transform, mask_transform, mask_value);
}


template <typename InputAccessorT, typename MatrixAccessorT>
static OSL_INLINE void
impl_transform_normal_masked (void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	static constexpr int width = InputAccessorT::width;
	typedef typename InputAccessorT::value_type data_type;

    Mask mask(mask_value);
    Mask succeeded(mask_transform);

    // only operate on active lanes
    Mask activeMask = mask & succeeded;

    MaskedAccessor<data_type, width> wresult(Pout, activeMask);
    InputAccessorT inPoints(Pin);
    MatrixAccessorT wM(transform);

    // Transform with Normal semantics
    int allAreAffine = 1;
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        // Detect if all the data lanes of the matrix are affine
        OSL_OMP_PRAGMA(omp simd simdlen(width))
        for(int lane=0; lane < wM.width; ++lane) {
            if (wresult.mask().is_on(lane)) {
                Matrix44 m = wM[lane];
                if ((m.x[0][3] != 0.0f || m.x[1][3] != 0.0f || m.x[2][3] != 0.0f || m.x[3][3] != 1.0f)) {
                    allAreAffine = 0;
                }
            }
        }
    }

    if (allAreAffine) {
        OSL_INTEL_PRAGMA(forceinline recursive)
        {
            // Optimized SIMD path for affine matrix
            OSL_OMP_PRAGMA(omp simd simdlen(width))
            for(int i=0; i < wresult.width; ++i)
            {
                data_type v = inPoints[i];
                Matrix44 M = wM[i];
                data_type r;

                //inlinedTransposed(affineInvert(M)).multDirMatrix (v, r);
                multDirMatrix (inlinedTransposed(affineInvert(M)), v, r);

                wresult[i] = r;
            }
        }
    } else {
        // Backup slow path for non-affine matrix
        for(int lane=0; lane < wresult.width; ++lane)
        {
            if (wresult.mask().is_on(lane)) {
                data_type v = inPoints[lane];
                Matrix44 M = wM[lane];
                data_type r;

                //M.inverse().transposed().multDirMatrix (v, r);
                multDirMatrix (M.inverse().transposed(), v, r);

                wresult[lane] = r;
            }
        }
    }

    impl_copy_untransformed_lanes(inPoints, Pout, succeeded, mask);
}

OSL_SHADEOP void
osl_transform_normal_vw16vm_masked (void *Pin,
                      void *Pout,
                      void * transform, unsigned int mask_transform, unsigned int mask_value)
{
    // TODO: see if we can get gen_transform to call the vvm version then do a masked broadcast
    impl_transform_normal_masked<ConstUniformAccessor<Vec3>, ConstUniformAccessor<Matrix44>> (Pin, Pout, transform, mask_transform, mask_value);
}


OSL_SHADEOP void
osl_transform_normal_vw16vw16m_masked (void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	impl_transform_normal_masked<ConstUniformAccessor<Vec3>, ConstWideAccessor<Matrix44>> (Pin, Pout, transform, mask_transform, mask_value);
}

OSL_SHADEOP void
osl_transform_normal_w16vw16vw16m_masked (void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	impl_transform_normal_masked<ConstWideAccessor<Vec3>, ConstWideAccessor<Matrix44>> (Pin, Pout, transform, mask_transform, mask_value);
}

OSL_SHADEOP void
osl_transform_normal_w16dvw16dvw16m_masked (void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	impl_transform_normal_masked<ConstWideAccessor<Dual2<Vec3>>, ConstWideAccessor<Matrix44>> (Pin, Pout, transform, mask_transform, mask_value);
}






OSL_SHADEOP void
osl_transform_normal_w16vw16vm_masked (void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	impl_transform_normal_masked<ConstWideAccessor<Vec3>, ConstUniformAccessor<Matrix44>> (Pin, Pout, transform, mask_transform, mask_value);
}

OSL_SHADEOP void
osl_transform_normal_w16dvw16dvm_masked (void *Pin,
                      void *Pout,
					  void * transform, unsigned int mask_transform, unsigned int mask_value)
{
	impl_transform_normal_masked<ConstWideAccessor<Dual2<Vec3>>, ConstUniformAccessor<Matrix44>> (Pin, Pout, transform, mask_transform, mask_value);
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
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Matrix44> wm(wm_);
		WideAccessor<float> wr(wr_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Matrix44 m = wm[lane];
			float r = det4x4(m);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_determinant_w16fw16m_masked(void *wr_, void * wm_, unsigned int mask_value)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
        ConstWideAccessor<Matrix44> wm(wm_);
        MaskedAccessor<float> wr(wr_, Mask(mask_value));

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {
            Matrix44 m = wm[lane];
            float r = det4x4(m);
            wr[lane] = r;
        }
    }
}

} // namespace pvt
OSL_NAMESPACE_EXIT
