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
#include "OSL/wide.h"


OSL_NAMESPACE_ENTER
namespace pvt {

OSL_SHADEOP void
osl_pow_w16fw16fw16f (void *r_, void *base_, void *exponent_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		const Wide<float> &wbase = WFLOAT(base_);
		const Wide<float> &wexponent = WFLOAT(exponent_);
		Wide<float> &wr = WFLOAT(r_);
	
		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")				
		for(int lane=0; lane < wr.width; ++lane) {
			float base = wbase.get(lane);
			float exponent = wexponent.get(lane);
			// TODO: perhaps a custom pow implementation to take
			// advantage of the exponent being the same?
			float r = powf(base,exponent);
			wr.set(lane, r);
		}
	}	
}

OSL_SHADEOP void
osl_pow_w16fw16fw16f_masked (void *r_, void *base_, void *exponent_, int mask_)
{
    OSL_INTEL_PRAGMA("forceinline recursive")
    {
        const Wide<float> &wbase = WFLOAT(base_);
        const Wide<float> &wexponent = WFLOAT(exponent_);
        const Mask mask(mask_);
        Wide<float> &wr = WFLOAT(r_);

        OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
        for(int lane=0; lane < wr.width; ++lane) {
            if (mask[lane]) {
                float base = wbase.get(lane);
                float exponent = wexponent.get(lane);
                // TODO: perhaps a custom pow implementation to take
                // advantage of the exponent being the same?
                float r = powf(base,exponent);
                wr.set(lane, r);
            }
        }
    }
}

OSL_SHADEOP void
osl_pow_w16vw16vw16f (void *r_, void *base_, void *exponent_)
{
	//std::cout << "Made it to osl_pow_w16vw16vw16f" << std::endl;
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		const Wide<Vec3> &wbase = WVEC(base_);
		const Wide<float> &wexponent = WFLOAT(exponent_);
		Wide<Vec3> &wr = WVEC(r_);
	
		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")				
		for(int lane=0; lane < wr.width; ++lane) {
			Vec3 base = wbase.get(lane);
			float exponent = wexponent.get(lane);
			//std::cout << "lane[" << lane << "] base = " << base << " exp=" << exponent << std::endl;
			Vec3 r;
			// TODO: perhaps a custom pow implementation to take
			// advantage of the exponent being the same?
			r.x = powf(base.x,exponent);
			r.y = powf(base.y,exponent);
			r.z = powf(base.z,exponent);
			wr.set(lane, r);
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


// Imath::Vec3::lengthTiny is private
// local copy here no changes
inline float accessibleTinyLength(const Vec3 &N)
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

// because lengthTiny does alot of work including another
// sqrt, we really want to skip that if possible because
// with SIMD execution, we end up doing the sqrt twice
// and blending the results.  Although code could be 
// refactored to do a single sqrt, think its better
// to skip the code block as we don't expect near 0 lengths
// TODO: get OpenEXR ImathVec to update to similar, don't think
// it can cause harm
inline float simdFriendlyLength(const Vec3 &N)
{
	float length2 = N.dot (N);

	if (__builtin_expect(length2 < float (2) * Imath::limits<float>::smallest(), 0))
		return accessibleTinyLength(N);

	return Imath::Math<float>::sqrt (length2);
}

inline Vec3 simdFriendlyNormalize(const Vec3 &N)
{
    float l = simdFriendlyLength(N);

    if (l == float (0))
    	return Vec3 (float (0));

    return Vec3 (N.x / l, N.y / l, N.z / l);
}

OSL_SHADEOP void
osl_length_w16fw16v(void *r_, void *V_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<Vec3> wV(V_);
		WideAccessor<float> wr(r_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
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
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<Vec3> wV(V_);
		MaskedAccessor<float> wr(r_, Mask(mask_value));

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
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
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		const Wide<Dual2<Vec3>> &wDP = WDVEC(DP_);
	    
		Wide<float> &wr = WFLOAT(r_);
	
		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")		
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<Vec3> DP = wDP.get(lane);
			
		    Vec3 N = calculatenormal(DP, false);
		    //float r = N.length();
		    float r = simdFriendlyLength(N);
			wr.set(lane, r);
		}
	}	
}

OSL_SHADEOP void
osl_area_w16_masked(void *r_, void *DP_, unsigned int mask_value)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		const Wide<Dual2<Vec3>> &wDP = WDVEC(DP_);
	    
		MaskedAccessor<float> wr(r_, Mask(mask_value));
	
		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")		
		for(int lane=0; lane < wr.width; ++lane) {
			Dual2<Vec3> DP = wDP.get(lane);
			
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
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<Vec3> wV(V_);
		WideAccessor<Vec3> wr(r_);
	
		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")		
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
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<Vec3> wV(V_);
		MaskedAccessor<Vec3> wr(r_, Mask(mask_value));

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			Vec3 V = wV[lane];
		    Vec3 N = simdFriendlyNormalize(V);
			wr[lane] = N;
		}
	}
}


OSL_SHADEOP void
osl_cross_w16vw16vw16v (void *result_, void *a_, void *b_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<Vec3> wA(a_);
		ConstWideAccessor<Vec3> wB(b_);
		WideAccessor<Vec3> wr(result_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
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
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<Vec3> wA(a_);
		ConstWideAccessor<Vec3> wB(b_);
		MaskedAccessor<Vec3> wr(result_, Mask(mask_value));

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			Vec3 a = wA[lane];
			Vec3 b = wB[lane];

		    Vec3 r = a.cross(b);
			wr[lane] = r;
		}
	}
}

OSL_SHADEOP void
osl_dot_w16fw16vw16v (void *result_, void *a_, void *b_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<Vec3> wA(a_);
		ConstWideAccessor<Vec3> wB(b_);
		WideAccessor<float> wr(result_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			Vec3 a = wA[lane];
			Vec3 b = wB[lane];

		    float r = a.dot(b);
			wr[lane] = r;
		}
	}
}



OSL_SHADEOP void
osl_acos_w16fw16f (void *result_, void *value_)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		ConstWideAccessor<float> wV(value_);
		WideAccessor<float> wr(result_);

		OSL_INTEL_PRAGMA("omp simd simdlen(wr.width)")
		for(int lane=0; lane < wr.width; ++lane) {
			float V = wV[lane];

		    float r = acos(V);
			wr[lane] = r;
		}
	}
}



} // namespace pvt
OSL_NAMESPACE_EXIT
