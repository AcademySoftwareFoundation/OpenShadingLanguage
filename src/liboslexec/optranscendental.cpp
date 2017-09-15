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
#include <OSL/Imathx.h>
#include "OSL/wide.h"


OSL_NAMESPACE_ENTER
namespace pvt {

OSL_SHADEOP void
osl_pow_w16fw16fw16f (void *r_, void *base_, void *exponent_)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<float> wbase(base_);
		ConstWideAccessor<float> wexponent(exponent_);
		WideAccessor<float> wr(r_);
	
		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			float base = wbase[lane];
			float exponent = wexponent[lane];
			// TODO: perhaps a custom pow implementation to take
			// advantage of the exponent being the same?
			float r = powf(base,exponent);
			wr[lane] = r;
		}
	}	
}

OSL_SHADEOP void
osl_pow_w16fw16fw16f_masked (void *r_, void *base_, void *exponent_, int mask_)
{
    OSL_INTEL_PRAGMA(forceinline recursive)
    {
		ConstWideAccessor<float> wbase(base_);
		ConstWideAccessor<float> wexponent(exponent_);
        const Mask mask(mask_);
        WideAccessor<float> wr(r_);

        OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
        for(int lane=0; lane < wr.width; ++lane) {
            if (mask[lane]) {
                float base = wbase[lane];
                float exponent = wexponent[lane];
                // TODO: perhaps a custom pow implementation to take
                // advantage of the exponent being the same?
                float r = powf(base,exponent);
                wr[lane] = r;
            }
        }
    }
}

OSL_SHADEOP void
osl_pow_w16vw16vw16f (void *r_, void *base_, void *exponent_)
{
	//std::cout << "Made it to osl_pow_w16vw16vw16f" << std::endl;
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Vec3> wbase(base_);
		ConstWideAccessor<float> wexponent(exponent_);
		WideAccessor<Vec3> wr(r_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Vec3 base = wbase[lane];
			float exponent = wexponent[lane];
			//std::cout << "lane[" << lane << "] base = " << base << " exp=" << exponent << std::endl;
			Vec3 r;
			// TODO: perhaps a custom pow implementation to take
			// advantage of the exponent being the same?
			r.x = powf(base.x,exponent);
			r.y = powf(base.y,exponent);
			r.z = powf(base.z,exponent);
			wr[lane] = r;
		}
	}	
}

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
osl_acos_w16fw16f (void *result_, void *value_)
{
#ifndef OSL_FAST_MATH
	#error INCOMPLETE safe version incomplete
#endif
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<float> wV(value_);
		WideAccessor<float> wr(result_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			float V = wV[lane];

		    float r = acos(V);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_acos_w16dfw16df (void *result_, void *a_)
{
#ifndef OSL_FAST_MATH
	#error INCOMPLETE safe version incomplete
#endif
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		ConstWideAccessor<Dual2<float>> wA(a_);
		WideAccessor<Dual2<float>> wr(result_);

		OSL_OMP_PRAGMA(omp simd simdlen(wr.width))
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<float> a = wA[lane];

#ifdef __clang__
			// acos is not vectorizing through clang
			// https://llvm.org/docs/Vectorizers.html
			// acos is not a supported function call
			// so try different implementation
			float arccosa = OIIO::fast_acos(a.val());
#else
			float arccosa = acos(a.val());
#endif

			float denom   = fabsf(a.val()) < 1.0f ? -1.0f / sqrtf(1.0f - a.val() * a.val()) : 0.0f;
			Dual2<float> r(arccosa, denom * a.dx(), denom * a.dy());

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


} // namespace pvt
OSL_NAMESPACE_EXIT
