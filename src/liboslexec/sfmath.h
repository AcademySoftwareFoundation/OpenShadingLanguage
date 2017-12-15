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


#include <cmath>

#include "oslexec_pvt.h"
#include "OSL/dual.h"
#include "OSL/dual_vec.h"
#include "OSL/Imathx.h"
#include "OSL/wide.h"

#include <OpenEXR/ImathFun.h>
#include <OpenImageIO/fmath.h>


OSL_NAMESPACE_ENTER
namespace pvt {



// non SIMD version, should be scalar code meant to be used
// inside SIMD loops
// SIMD FRIENDLY MATH
namespace sfm
{
	// Math code derived from OpenImageIO/fmath.h
	// including it's copyrights in the namespace
	/*
	  Copyright 2008-2014 Larry Gritz and the other authors and contributors.
	  All Rights Reserved.

	  Redistribution and use in source and binary forms, with or without
	  modification, are permitted provided that the following conditions are
	  met:
	  * Redistributions of source code must retain the above copyright
		notice, this list of conditions and the following disclaimer.
	  * Redistributions in binary form must reproduce the above copyright
		notice, this list of conditions and the following disclaimer in the
		documentation and/or other materials provided with the distribution.
	  * Neither the name of the software's owners nor the names of its
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

	  (This is the Modified BSD License)

	  A few bits here are based upon code from NVIDIA that was also released
	  under the same modified BSD license, and marked as:
		 Copyright 2004 NVIDIA Corporation. All Rights Reserved.

	  Some parts of this file were first open-sourced in Open Shading Language,
	  then later moved here. The original copyright notice was:
		 Copyright (c) 2009-2014 Sony Pictures Imageworks Inc., et al.

	  Many of the math functions were copied from or inspired by other
	  public domain sources or open source packages with compatible licenses.
	  The individual functions give references were applicable.
	*/

	/// Fused multiply and add: (a*b + c)
	OSL_INLINE float madd (float a, float b, float c) {
		return a * b + c;
	}

	OSL_INLINE float absf (float x)
	{
	    return x >= 0.0f ? x : -x;
	}

	OSL_INLINE Dual2<float> absf (const Dual2<float> &x)
	{
#ifdef __clang__
		Dual2<float> r;
		if (x.val() >= 0.0f) {
			r = x;
		} else {
			r = -x;

		}
		return r;
#else
	    return x.val() >= 0.0 ? x : -x;
#endif
	}

	OSL_INLINE int absi (int x)
	{
	    return x >= 0 ? x : -x;
	}

	template <typename IN_TYPE, typename OUT_TYPE>
	OSL_INLINE OUT_TYPE bit_cast (const IN_TYPE val) {
		static_assert(sizeof(IN_TYPE) == sizeof(OUT_TYPE), "when casting between types they must be the same size");
		union {
			IN_TYPE inVal;
			OUT_TYPE outVal;
		} in_or_out;
		in_or_out.inVal = val;
		return in_or_out.outVal;
	}

	OSL_INLINE int bitcast_to_int (float x) { return bit_cast<float,int>(x); }
	OSL_INLINE float bitcast_to_float (int x) { return bit_cast<int,float>(x); }

	using OIIO::clamp;

	template<typename T>
	T log2 (const T& xval); // undefined

	template<>
	OSL_INLINE float log2<float> (const float& xval) {
	    // NOTE: clamp to avoid special cases and make result "safe" from large negative values/nans
	    float x = clamp (xval, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
	    // based on https://github.com/LiraNuna/glsl-sse2/blob/master/source/vec4.h
	    unsigned bits = bit_cast<float, unsigned>(x);
	    int exponent = int(bits >> 23) - 127;
	    float f = bit_cast<unsigned, float>((bits & 0x007FFFFF) | 0x3f800000) - 1.0f;
	    // Examined 2130706432 values of log2 on [1.17549435e-38,3.40282347e+38]: 0.0797524457 avg ulp diff, 3713596 max ulp, 7.62939e-06 max error
	    // ulp histogram:
	    //  0  = 97.46%
	    //  1  =  2.29%
	    //  2  =  0.11%
	    float f2 = f * f;
	    float f4 = f2 * f2;
	    float hi = madd(f, -0.00931049621349f,  0.05206469089414f);
	    float lo = madd(f,  0.47868480909345f, -0.72116591947498f);
	    hi = madd(f, hi, -0.13753123777116f);
	    hi = madd(f, hi,  0.24187369696082f);
	    hi = madd(f, hi, -0.34730547155299f);
	    lo = madd(f, lo,  1.442689881667200f);
	    return ((f4 * hi) + (f * lo)) + exponent;
	}

	template<>
	OSL_INLINE Dual2<float> log2<Dual2<float>>(const Dual2<float> &a)
	{
	    float loga = log2(a.val());
	    float aln2 = a.val() * float(M_LN2);
	    float inva = aln2 < std::numeric_limits<float>::min() ? 0.0f : 1.0f / aln2;
	    return Dual2<float> (loga, inva * a.dx(), inva * a.dy());
	}

	template<typename T>
	inline T log (const T& x) {
	    // Examined 2130706432 values of logf on [1.17549435e-38,3.40282347e+38]: 0.313865375 avg ulp diff, 5148137 max ulp, 7.62939e-06 max error
	    return log2(x) * T(M_LN2);
	}

	template<typename T>
	inline T log10 (const T& x) {
	    // Examined 2130706432 values of log10f on [1.17549435e-38,3.40282347e+38]: 0.631237033 avg ulp diff, 4471615 max ulp, 3.8147e-06 max error
	    return log2(x) * T(M_LN2 / M_LN10);
	}

	template<typename T>
	OSL_INLINE T exp2 (const T & xval); // undefined

	template<>
	OSL_INLINE float exp2<float> (const float & xval) {
#if 0
		return std::exp2(xval);
#else
	    // clamp to safe range for final addition
	    float x = clamp (xval, -126.0f, 126.0f);
	    // range reduction
	    int m = static_cast<int>(x); x -= m;
	    x = 1.0f - (1.0f - x); // crush denormals (does not affect max ulps!)
	    // 5th degree polynomial generated with sollya
	    // Examined 2247622658 values of exp2 on [-126,126]: 2.75764912 avg ulp diff, 232 max ulp
	    // ulp histogram:
	    //  0  = 87.81%
	    //  1  =  4.18%
	    float r = 1.33336498402e-3f;
	    r = madd(x, r, 9.810352697968e-3f);
	    r = madd(x, r, 5.551834031939e-2f);
	    r = madd(x, r, 0.2401793301105f);
	    r = madd(x, r, 0.693144857883f);
	    r = madd(x, r, 1.0f);
	    // multiply by 2 ^ m by adding in the exponent
	    // NOTE: left-shift of negative number is undefined behavior

	    // Clang: loop not vectorized: unsafe dependent memory operations in loop
	    return bit_cast<unsigned, float>(bit_cast<float, unsigned>(r) + (static_cast<unsigned>(m) << 23));

	    // Clang: loop not vectorized: unsafe dependent memory operations in loop
//		union {
//			float float_val;
//			unsigned int uint_val;
//		} float_or_uint;
//		float_or_uint.float_val = r;
//		float_or_uint.uint_val += (static_cast<unsigned>(m) << 23);
//		return float_or_uint.float_val;
#endif
	}

	template<>
	OSL_INLINE Dual2<float> exp2<Dual2<float>>(const Dual2<float> &a)
	{
	    float exp2a = sfm::template exp2(a.val());
	    return Dual2<float> (exp2a, exp2a*float(M_LN2)*a.dx(), exp2a*float(M_LN2)*a.dy());
	}

	template <typename T>
	inline T exp (const T& x) {
	    // Examined 2237485550 values of exp on [-87.3300018,87.3300018]: 2.6666452 avg ulp diff, 230 max ulp
	    return exp2(x * T(1 / M_LN2));
	}

	inline float expm1 (float x) {
	    if (absf(x) < 0.03f) {
	        float y = 1.0f - (1.0f - x); // crush denormals
	        return copysignf(madd(0.5f, y * y, y), x);
	    } else
	        return exp(x) - 1.0f;
	}

	OSL_INLINE Dual2<float> expm1(const Dual2<float> &a)
	{
	    float expm1a = expm1(a.val());
	    float expa   = exp  (a.val());
	    return Dual2<float> (expm1a, expa * a.dx(), expa * a.dy());
	}

	OSL_INLINE float erf(const float x)
	{
		// Examined 1082130433 values of erff on [0,4]: 1.93715e-06 max error
		// Abramowitz and Stegun, 7.1.28
		const float a1 = 0.0705230784f;
		const float a2 = 0.0422820123f;
		const float a3 = 0.0092705272f;
		const float a4 = 0.0001520143f;
		const float a5 = 0.0002765672f;
		const float a6 = 0.0000430638f;
		const float a = absf(x);
		const float b = 1.0f - (1.0f - a); // crush denormals
		const float r = madd(madd(madd(madd(madd(madd(a6, b, a5), b, a4), b, a3), b, a2), b, a1), b, 1.0f);
		const float s = r * r; // ^2
		const float t = s * s; // ^4
		const float u = t * t; // ^8
		const float v = u * u; // ^16
		return copysign(1.0f - 1.0f / v, x);
	}

	OSL_INLINE Dual2<float> erf(const Dual2<float> &a)
	{
		float erfa = erf (a.val());
		float two_over_sqrt_pi = 1.128379167095512573896158903f;
		float derfadx = exp(-a.val() * a.val()) * two_over_sqrt_pi;
		return Dual2<float> (erfa, derfadx * a.dx(), derfadx * a.dy());
	}

	OSL_INLINE float erfc (float x)
	{
	    // Examined 2164260866 values of erfcf on [-4,4]: 1.90735e-06 max error
	    // ulp histogram:
	    //   0  = 80.30%
	    return 1.0f - erf(x);
	}

	OSL_INLINE Dual2<float> erfc(const Dual2<float> &a)
	{
	    float erfa = erfc (a.val());
	    float two_over_sqrt_pi = -1.128379167095512573896158903f;
	    float derfadx = exp(-a.val() * a.val()) * two_over_sqrt_pi;
	    //float derfadx = std::exp(-a.val() * a.val()) * two_over_sqrt_pi;
	    return Dual2<float> (erfa, derfadx * a.dx(), derfadx * a.dy());
	}



	OSL_INLINE float safe_pow (float x, float y) {
	    if (y == 0) return 1.0f; // x^0=1
	    if (x == 0) return 0.0f; // 0^y=0
	    // be cheap & exact for special case of squaring and identity
	    if (y == 1.0f)
	        return x;
	    if (y == 2.0f)
	        return std::min (x*x, std::numeric_limits<float>::max());
	    float sign = 1.0f;
	    if (x < 0) {
	        // if x is negative, only deal with integer powers
	        // powf returns NaN for non-integers, we will return 0 instead
	        int ybits = bitcast_to_int(y) & 0x7fffffff;
	        if (ybits >= 0x4b800000) {
	            // always even int, keep positive
	        } else if (ybits >= 0x3f800000) {
	            // bigger than 1, check
	            int k = (ybits >> 23) - 127;  // get exponent
	            int j =  ybits >> (23 - k);   // shift out possible fractional bits
	            if ((j << (23 - k)) == ybits) // rebuild number and check for a match
	                sign = bit_cast<int, float>(0x3f800000 | (j << 31)); // +1 for even, -1 for odd
	            else
	                return 0.0f; // not integer
	        } else {
	            return 0.0f; // not integer
	        }
	    }
	    return sign * exp2(y * log2(std::abs(x)));

	}

	OSL_INLINE Dual2<float> safe_pow(const Dual2<float> &u, const Dual2<float> &v)
	{
	    // NOTE: same issue as above (fast_safe_pow does even more clamping)
	    float powuvm1 = sfm::safe_pow (u.val(), v.val() - 1.0f);
	    float powuv   = powuvm1 * u.val();
	    float logu    = u.val() > 0 ? log(u.val()) : 0.0f;
	    return Dual2<float> ( powuv, v.val()*powuvm1 * u.dx() + logu*powuv * v.dx(),
	                                 v.val()*powuvm1 * u.dy() + logu*powuv * v.dy() );
	}

	// NOTE: safe_fmod identical to regular version, they just weren't accessible from
	// llvm_ops.cpp, so they could be moved to common place
	OSL_INLINE float safe_fmod (float a, float b) {
	    return (b != 0.0f) ? std::fmod (a,b) : 0.0f;
	}

	OSL_INLINE Dual2<float> safe_fmod (const Dual2<float> &a, const Dual2<float> &b) {
	    return Dual2<float> (safe_fmod (a.val(), b.val()), a.dx(), a.dy());
	}

#if 0 // emitted directly by llvm_gen_wide.cpp
	OSL_INLINE float safe_div(float a, float b) {
	    return (b != 0.0f) ? (a / b) : 0.0f;
	}

	OSL_INLINE int safe_div(int a, int b) {
	    return (b != 0) ? (a / b) : 0;
	}
#endif


	/// Round to nearest integer, returning as an int.
	OSL_INLINE int fast_rint (float x) {
	    // used by sin/cos/tan range reduction
	#if 1
	    // single roundps instruction on SSE4.1+ (for gcc/clang at least)
	    //return static_cast<int>(rintf(x));
		OSL_INTEL_PRAGMA(forceinline)
		return rintf(x);
	#else
	    // emulate rounding by adding/substracting 0.5
	    return static_cast<int>(x + copysignf(0.5f, x));
	    //return (x >= 0.0f) ? static_cast<int>(x + 0.5f) : static_cast<int>(x - 0.5f);

	    //return static_cast<int>(x +  (x >= 0.0f) ? 0.5f : - 0.5f);
	    //float pad = (x >= 0.0f) ? 0.5f : - 0.5f;
	    //return static_cast<int>(x + pad);
	    //return nearbyint(x);
#endif
	}


	OSL_INLINE float sin (float x) {
	    // very accurate argument reduction from SLEEF
	    // starts failing around x=262000
	    // Results on: [-2pi,2pi]
	    // Examined 2173837240 values of sin: 0.00662760244 avg ulp diff, 2 max ulp, 1.19209e-07 max error
	    int q = fast_rint (x * float(M_1_PI));
	    float qf = q;
	    x = madd(qf, -0.78515625f*4, x);
	    x = madd(qf, -0.00024187564849853515625f*4, x);
	    x = madd(qf, -3.7747668102383613586e-08f*4, x);
	    x = madd(qf, -1.2816720341285448015e-12f*4, x);
	    x = float(M_PI_2) - (float(M_PI_2) - x); // crush denormals
	    float s = x * x;
	    if ((q & 1) != 0) x = -x;
	    // this polynomial approximation has very low error on [-pi/2,+pi/2]
	    // 1.19209e-07 max error in total over [-2pi,+2pi]
	    float u = 2.6083159809786593541503e-06f;
	    u = madd(u, s, -0.0001981069071916863322258f);
	    u = madd(u, s, +0.00833307858556509017944336f);
	    u = madd(u, s, -0.166666597127914428710938f);
	    u = madd(s, u * x, x);
	    // For large x, the argument reduction can fail and the polynomial can be
	    // evaluated with arguments outside the valid internal. Just clamp the bad
	    // values away (setting to 0.0f means no branches need to be generated).
	    //if (fabsf(u) > 1.0f) u = 0.0f;
	    if (absf(u) > 1.0f) u = 0.0f;
	    return u;
	}
	OSL_INLINE float cos (float x) {
	    // same argument reduction as fast_sin
	    int q = fast_rint (x * float(M_1_PI));
	    float qf = q;
	    x = madd(qf, -0.78515625f*4, x);
	    x = madd(qf, -0.00024187564849853515625f*4, x);
	    x = madd(qf, -3.7747668102383613586e-08f*4, x);
	    x = madd(qf, -1.2816720341285448015e-12f*4, x);
	    x = float(M_PI_2) - (float(M_PI_2) - x); // crush denormals
	    float s = x * x;
	    // polynomial from SLEEF's sincosf, max error is
	    // 4.33127e-07 over [-2pi,2pi] (98% of values are "exact")
	    float u = -2.71811842367242206819355e-07f;
	    u = madd(u, s, +2.47990446951007470488548e-05f);
	    u = madd(u, s, -0.00138888787478208541870117f);
	    u = madd(u, s, +0.0416666641831398010253906f);
	    u = madd(u, s, -0.5f);
	    u = madd(u, s, +1.0f);
	    if ((q & 1) != 0) u = -u;
	    if (fabsf(u) > 1.0f) u = 0.0f;
	    return u;
	}
	OSL_INLINE void sincos (float x, float & sine, float& cosine) {
	    // same argument reduction as fast_sin
	    int q = fast_rint (x * float(M_1_PI));
	    float qf = q;
	    x = madd(qf, -0.78515625f*4, x);
	    x = madd(qf, -0.00024187564849853515625f*4, x);
	    x = madd(qf, -3.7747668102383613586e-08f*4, x);
	    x = madd(qf, -1.2816720341285448015e-12f*4, x);
	    x = float(M_PI_2) - (float(M_PI_2) - x); // crush denormals
	    float s = x * x;
	    // NOTE: same exact polynomials as fast_sin and fast_cos above
	    if ((q & 1) != 0) x = -x;
	    float su = 2.6083159809786593541503e-06f;
	    su = madd(su, s, -0.0001981069071916863322258f);
	    su = madd(su, s, +0.00833307858556509017944336f);
	    su = madd(su, s, -0.166666597127914428710938f);
	    su = madd(s, su * x, x);
	    float cu = -2.71811842367242206819355e-07f;
	    cu = madd(cu, s, +2.47990446951007470488548e-05f);
	    cu = madd(cu, s, -0.00138888787478208541870117f);
	    cu = madd(cu, s, +0.0416666641831398010253906f);
	    cu = madd(cu, s, -0.5f);
	    cu = madd(cu, s, +1.0f);
	    if ((q & 1) != 0) cu = -cu;
//	    if (fabsf(su) > 1.0f) su = 0.0f;
//	    if (fabsf(cu) > 1.0f) cu = 0.0f;
	    if (absf(su) > 1.0f) su = 0.0f;
	    if (absf(cu) > 1.0f) cu = 0.0f;
	    sine   = su;
	    cosine = cu;
	}

	// NOTE: this approximation is only valid on [-8192.0,+8192.0], it starts becoming
	// really poor outside of this range because the reciprocal amplifies errors
	OSL_INLINE float tan (float x) {
	    // derived from SLEEF implementation
	    // note that we cannot apply the "denormal crush" trick everywhere because
	    // we sometimes need to take the reciprocal of the polynomial
	    int q = fast_rint (x * float(2 * M_1_PI));
	    float qf = q;
	    x = madd(qf, -0.78515625f*2, x);
	    x = madd(qf, -0.00024187564849853515625f*2, x);
	    x = madd(qf, -3.7747668102383613586e-08f*2, x);
	    x = madd(qf, -1.2816720341285448015e-12f*2, x);
	    if ((q & 1) == 0)
	    x = float(M_PI_4) - (float(M_PI_4) - x); // crush denormals (only if we aren't inverting the result later)
	    float s = x * x;
	    float u = 0.00927245803177356719970703f;
	    u = madd(u, s, 0.00331984995864331722259521f);
	    u = madd(u, s, 0.0242998078465461730957031f);
	    u = madd(u, s, 0.0534495301544666290283203f);
	    u = madd(u, s, 0.133383005857467651367188f);
	    u = madd(u, s, 0.333331853151321411132812f);
	    u = madd(s, u * x, x);
	    if ((q & 1) != 0) u = -1.0f / u;
	    return u;
	}

	OSL_INLINE Dual2<float> tan(const Dual2<float> &a)
	{
	    float tana  = sfm::tan (a.val());
	    float cosa  = sfm::cos (a.val());
	    float sec2a = 1 / (cosa * cosa);
	    return Dual2<float> (tana, sec2a * a.dx(), sec2a * a.dy());
	}

	OSL_INLINE float atan (float x) {
	    const float a = absf(x);
	    const float k = a > 1.0f ? 1 / a : a;
	    const float s = 1.0f - (1.0f - k); // crush denormals
	    const float t = s * s;
	    // http://mathforum.org/library/drmath/view/62672.html
	    // Examined 4278190080 values of atan: 2.36864877 avg ulp diff, 302 max ulp, 6.55651e-06 max error      // (with  denormals)
	    // Examined 4278190080 values of atan: 171160502 avg ulp diff, 855638016 max ulp, 6.55651e-06 max error // (crush denormals)
	    float r = s * madd(0.43157974f, t, 1.0f) / madd(madd(0.05831938f, t, 0.76443945f), t, 1.0f);
	    if (a > 1.0f) r = 1.570796326794896557998982f - r;
	    return copysignf(r, x);
	}

	OSL_INLINE Dual2<float> atan(const Dual2<float> &a)
	{
	    float arctana = sfm::atan(a.val());
	    float denom   = 1.0f / (1.0f + a.val() * a.val());
	    return Dual2<float> (arctana, denom * a.dx(), denom * a.dy());

	}

	OSL_INLINE float atan2 (float y, float x) {
	    // based on atan approximation above
	    // the special cases around 0 and infinity were tested explicitly
	    // the only case not handled correctly is x=NaN,y=0 which returns 0 instead of nan
	    const float a = absf(x);
	    const float b = absf(y);

	    const float k = (b == 0) ? 0.0f : ((a == b) ? 1.0f : (b > a ? a / b : b / a));
	    const float s = 1.0f - (1.0f - k); // crush denormals
	    const float t = s * s;

	    float r = s * madd(0.43157974f, t, 1.0f) / madd(madd(0.05831938f, t, 0.76443945f), t, 1.0f);

	    if (b > a) r = 1.570796326794896557998982f - r; // account for arg reduction
	    if (bit_cast<float, unsigned>(x) & 0x80000000u) // test sign bit of x
	        r = float(M_PI) - r;
	    return copysignf(r, y);
	}

	OSL_INLINE Dual2<float> atan2(const Dual2<float> &y, const Dual2<float> &x)
	{
	    float atan2xy = sfm::atan2(y.val(), x.val());
	    float denom = (x.val() == 0 && y.val() == 0) ? 0.0f : 1.0f / (x.val() * x.val() + y.val() * y.val());
	    return Dual2<float> ( atan2xy, (y.val()*x.dx() - x.val()*y.dx())*denom,
	                                   (y.val()*x.dy() - x.val()*y.dy())*denom );
	}

	OSL_INLINE float cosh (float x) {
	    // Examined 2237485550 values of cosh on [-87.3300018,87.3300018]: 1.78256726 avg ulp diff, 178 max ulp
	    float e = sfm::exp(fabsf(x));
	    return 0.5f * e + 0.5f / e;
	}

	OSL_INLINE float sinh (float x) {
	    float a = absf(x);
	    if (__builtin_expect(a > 1.0f, 0)) {
	        // Examined 53389559 values of sinh on [1,87.3300018]: 33.6886442 avg ulp diff, 178 max ulp
	        float e = sfm::exp(a);
	        return copysignf(0.5f * e - 0.5f / e, x);
	    } else {
	        a = 1.0f - (1.0f - a); // crush denorms
	        float a2 = a * a;
	        // degree 7 polynomial generated with sollya
	        // Examined 2130706434 values of sinh on [-1,1]: 1.19209e-07 max error
	        float r = 2.03945513931e-4f;
	        r = madd(r, a2, 8.32990277558e-3f);
	        r = madd(r, a2, 0.1666673421859f);
	        r = madd(r * a, a2, a);
	        return copysignf(r, x);
	    }
	}

	inline float tanh (float x) {
	    // Examined 4278190080 values of tanh on [-3.40282347e+38,3.40282347e+38]: 3.12924e-06 max error
	    // NOTE: ulp error is high because of sub-optimal handling around the origin
	    float e = sfm::exp(2.0f * fabsf(x));
	    return copysignf(1 - 2 / (1 + e), x);
	}

	OSL_INLINE Dual2<float> cosh(const Dual2<float> &a)
	{
	    float cosha = sfm::cosh(a.val());
	    float sinha = sfm::sinh(a.val());
	    return Dual2<float> (cosha, sinha * a.dx(), sinha * a.dy());
	}

	OSL_INLINE Dual2<float> sinh(const Dual2<float> &a)
	{
	    float cosha = sfm::cosh(a.val());
	    float sinha = sfm::sinh(a.val());
	    return Dual2<float> (sinha, cosha * a.dx(), cosha * a.dy());
	}

	OSL_INLINE Dual2<float> tanh(const Dual2<float> &a)
	{
	    float tanha = sfm::tanh(a.val());
	    float cosha = sfm::cosh(a.val());
	    float sech2a = 1 / (cosha * cosha);
	    return Dual2<float> (tanha, sech2a * a.dx(), sech2a * a.dy());
	}


	// Considering having functionally equivalent versions of Vec3, Color3, Matrix44
	// with slight modifications to inlining and implmentation to avoid aliasing and
	// improve likelyhood of proper privation of local variables within a SIMD loop
	#if 0

	namespace Imath {
	template <class T> class Matrix44
	{
	  public:
		OSL_INLINE Matrix44() {}
		OSL_INLINE Matrix44(const Matrix44 &other);

	    //-------------------
	    // Access to elements
	    //-------------------

	    T           x00;
	    T           x01;
	    T           x02;
	    T           x03;

	    T           x10;
	    T           x11;
	    T           x12;
	    T           x13;

	    T           x20;
	    T           x21;
	    T           x22;
	    T           x23;

	    T           x30;
	    T           x31;
	    T           x32;
	    T           x33;

	    OSL_INLINE Matrix44 (T a, T b, T c, T d, T e, T f, T g, T h,
	              T i, T j, T k, T l, T m, T n, T o, T p);

	                                // a b c d
	                                // e f g h
	                                // i j k l
	                                // m n o p
	};

	template <class T>
	Matrix44<T>::Matrix44 (const Matrix44 &other)
	: x00(other.x00)
	, x01(other.x01)
	, x02(other.x02)
	, x03(other.x03)
	, x10(other.x10)
	, x11(other.x11)
	, x12(other.x12)
	, x13(other.x13)
	, x20(other.x20)
	, x21(other.x21)
	, x22(other.x22)
	, x23(other.x23)
	, x30(other.x30)
	, x31(other.x31)
	, x32(other.x32)
	, x33(other.x33)
	{}

	template <class T>
	Matrix44<T>::Matrix44 (T a, T b, T c, T d, T e, T f, T g, T h,
	                       T i, T j, T k, T l, T m, T n, T o, T p)
	: x00(a)
	, x01(b)
	, x02(c)
	, x03(d)
	, x10(e)
	, x11(f)
	, x12(g)
	, x13(h)
	, x20(i)
	, x21(j)
	, x22(k)
	, x23(l)
	, x30(m)
	, x31(n)
	, x32(o)
	, x33(p)
	{
	//    x[0][0] = a;
	//    x[0][1] = b;
	//    x[0][2] = c;
	//    x[0][3] = d;
	//    x[1][0] = e;
	//    x[1][1] = f;
	//    x[1][2] = g;
	//    x[1][3] = h;
	//    x[2][0] = i;
	//    x[2][1] = j;
	//    x[2][2] = k;
	//    x[2][3] = l;
	//    x[3][0] = m;
	//    x[3][1] = n;
	//    x[3][2] = o;
	//    x[3][3] = p;
	}
	} // namespace Imath

	typedef Imath::Matrix44<Float> Matrix44;
#endif


} // namespace sfm


} // namespace pvt

#if 0
template <int WidthT>
struct Wide<fast::Matrix44, WidthT>
{
	typedef fast::Matrix44 value_type;
	static constexpr int width = WidthT;
	//Wide<float, WidthT> x[4][4];
	Wide<float, WidthT>           x00;
	Wide<float, WidthT>           x01;
	Wide<float, WidthT>           x02;
	Wide<float, WidthT>           x03;

	Wide<float, WidthT>           x10;
	Wide<float, WidthT>           x11;
	Wide<float, WidthT>           x12;
	Wide<float, WidthT>           x13;

	Wide<float, WidthT>           x20;
	Wide<float, WidthT>           x21;
	Wide<float, WidthT>           x22;
	Wide<float, WidthT>           x23;

	Wide<float, WidthT>           x30;
	Wide<float, WidthT>           x31;
	Wide<float, WidthT>           x32;
	Wide<float, WidthT>           x33;

	OSL_INLINE void
	set(int index, const value_type & value)
	{
		x00.set(index, value.x00);
		x01.set(index, value.x01);
		x02.set(index, value.x02);
		x03.set(index, value.x03);
		x10.set(index, value.x10);
		x11.set(index, value.x11);
//		x12.set(index, value.x12);
		//x13.set(index, value.x13);
		x12.set(index, 0.0f);
		x13.set(index, 0.0f);
		//x20.set(index, value.x20);

		x21.set(index, value.x21);
		x22.set(index, value.x22);
		x23.set(index, value.x23);
		x30.set(index, value.x30);
		x31.set(index, value.x31);
		x32.set(index, value.x32);
		x33.set(index, value.x33);
	}

	OSL_INLINE value_type
	get(int index) const
	{
		return value_type(
			x00.get(index), x01.get(index), x02.get(index), x03.get(index),
			x10.get(index), x11.get(index), x12.get(index), x13.get(index),
			x20.get(index), x21.get(index), x22.get(index), x23.get(index),
			x30.get(index), x31.get(index), x32.get(index), x33.get(index));
	}

#if 0
	OSL_INLINE void
	set(int index, const value_type & value)
	{
		x[0][0].set(index, value.x00);
		x[0][1].set(index, value.x01);
		x[0][2].set(index, value.x02);
		x[0][3].set(index, value.x03);
		x[1][0].set(index, value.x10);
		x[1][1].set(index, value.x11);
		x[1][2].set(index, value.x12);
		x[1][3].set(index, value.x13);
		x[2][0].set(index, value.x20);
		x[2][1].set(index, value.x21);
		x[2][2].set(index, value.x22);
		x[2][3].set(index, value.x23);
		x[3][0].set(index, value.x30);
		x[3][1].set(index, value.x31);
		x[3][2].set(index, value.x32);
		x[3][3].set(index, value.x33);
	}

	OSL_INLINE value_type
	get(int index) const
	{
		return value_type(
			x[0][0].get(index), x[0][1].get(index), x[0][2].get(index), x[0][3].get(index),
			x[1][0].get(index), x[1][1].get(index), x[1][2].get(index), x[1][3].get(index),
			x[2][0].get(index), x[2][1].get(index), x[2][2].get(index), x[2][3].get(index),
			x[3][0].get(index), x[3][1].get(index), x[3][2].get(index), x[3][3].get(index));
	}
#endif
	OSL_INLINE Wide() {}
	Wide(const Wide &value) = delete;
//	OSL_INLINE Wide(const Wide &value)
//	: x00(value.x00),
//	x01(value.x01),
//	x02(value.x02),
//	x03(value.x03),
//	x10(value.x10),
//	x11(value.x11),
//	x12(value.x12),
//	x13(value.x13),
//	x20(value.x20),
//	x21(value.x21),
//	x22(value.x22),
//	x23(value.x23),
//	x30(value.x30),
//	x31(value.x31),
//	x32(value.x32),
//	x33(value.x33)
//	{}


};

template <>
struct WideTraits<fast::Matrix44> {
	static bool matches(const TypeDesc &type_desc) {
		return type_desc.basetype == TypeDesc::FLOAT &&
		    type_desc.aggregate == TypeDesc::MATRIX44; }
};
#endif


OSL_NAMESPACE_EXIT
