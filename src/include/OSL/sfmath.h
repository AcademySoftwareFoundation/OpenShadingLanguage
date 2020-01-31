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
#pragma once


#include <cmath>

#include "dual.h"
#include "dual_vec.h"
#include "Imathx.h"

#include <OpenEXR/ImathFun.h>
#include <OpenEXR/ImathMatrix.h>
#include <OpenImageIO/fmath.h>


OSL_NAMESPACE_ENTER

#ifdef __OSL_WIDE_PVT
    namespace __OSL_WIDE_PVT {
#else
    namespace pvt {
#endif



// SIMD FRIENDLY MATH
// Scalar code meant to be used from inside
// compiler vectorized SIMD loops.
// No intrinsics or assembly, just vanilla C++
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

    // Math code derived from OpenEXR/ImathMatrix.h
    // including it's copyrights in the namespace
    /*
       Copyright (c) 2002-2012, Industrial Light & Magic, a division of Lucas
       Digital Ltd. LLC

       All rights reserved.

       Redistribution and use in source and binary forms, with or without
       modification, are permitted provided that the following conditions are
       met:
       *       Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
       *       Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following disclaimer
       in the documentation and/or other materials provided with the
       distribution.
       *       Neither the name of Industrial Light & Magic nor the names of
       its contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

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

	/// return the greatest integer <= x
	OSL_FORCEINLINE int ifloor (float x) {
		//    return (int) x - ((x < 0) ? 1 : 0);

		// std::floor is another option, however that appears to be
		// a function call right now, and this sequence appears cheaper
		//return static_cast<int>(x - ((x < 0.0f) ? 1.0f : 0.0f));

		// This factoring should allow the expensive float to integer
		// conversion to happen at the same time the comparison is
		// in an out of order CPU
		return (static_cast<int>(x)) - ((x < 0.0f) ? 1 : 0);
	}

    /// Fused multiply and add: (a*b + c)
    OSL_FORCEINLINE float madd (float a, float b, float c) {
        // Avoid simulating FMA on non-FMA hardware
        // This can result in differences between FMA and non-FMA hardware
        // but a simulated result is really slow
        return a * b + c;
    }

    /// Identical to OIIO::lerp(a,b,u), but always inlined
    /// to not inhibit CLANG vectorization
    template <class T, class Q>
    OSL_FORCEINLINE T
    lerp (const T& v0, const T& v1, const Q& x)
    {
        // NOTE: a*(1-x) + b*x is much more numerically stable than a+x*(b-a)
        return v0*(Q(1)-x) + v1*x;
    }

    /// Identical to OIIO::bilerp(a,b,c,d,u,v), but always inlined
    /// to not inhibit CLANG vectorization
    template <class T, class Q>
    OSL_FORCEINLINE T
    bilerp(const T& v0, const T& v1, const T& v2, const T& v3, const Q& s, const Q& t)
    {
        // NOTE: a*(t-1) + b*t is much more numerically stable than a+t*(b-a)
        Q s1 = Q(1) - s;
        return T ((Q(1)-t)*(v0*s1 + v1*s) + t*(v2*s1 + v3*s));
    }

    /// Identical to OIIO::trilerp(a,b,c,d,e,f,g,h,u,v,w), but always inlined
    /// to not inhibit CLANG vectorization
    template <class T, class Q>
    OSL_FORCEINLINE T
    trilerp (const T & v0, const T & v1, const T & v2, const T & v3, const T & v4, const T & v5, const T & v6, const T & v7, const Q &s, const Q &t, const Q & r)
    {
        // NOTE: a*(t-1) + b*t is much more numerically stable than a+t*(b-a)
        Q s1 = Q(1) - s;
        Q t1 = Q(1) - t;
        Q r1 = Q(1) - r;

        return T (r1*(t1*(v0*s1 + v1*s) + t*(v2*s1 + v3*s)) +
                   r*(t1*(v4*s1 + v5*s) + t*(v6*s1 + v7*s)));
    }

    // Native OIIO::isinf wasn't vectorizing and was branchy
    // this slightly perturbed version fairs better and is branch free
    // when vectorized
    OSL_FORCEINLINE int isinf (float x) {

        // Based on algorithm in OIIO missing_math.h for _MSC_VER < 1800
        int r = 0;
        // NOTE: using bitwise | to avoid branches
        if (!(OIIO::isfinite(x)|OIIO::isnan(x))) {
            r = static_cast<int>(copysignf(1.0f,x));
        }
        return r;
    }

    // Exists mainly to allow the same function name to work
    // with Dual2 and float
    OSL_FORCEINLINE float absf (float x)
    {
#if 0
        //return x >= 0.0f ? x : -x;
#elif 0
        // Move negation operation out of conditional
        // so result should be masked blend or ternary
        float neg_x = -x;
        return (x >= 0.0f) ? x : neg_x;
#else
        // gcc header use builtin abs that does bit twiddling 2 instructions
        return std::abs(x);
#endif
    }

    template<typename T> OSL_FORCEINLINE T negate(const T &x) {
        #if OSL_FAST_MATH
            // Compiler using a constant bit mask to perform negation,
            // and reading a constant involves accessing its memory location.
            // Alternatively the compiler can create a 0 value in register
            // in a constant time not involving the memory subsystem.
            // So we can subtract from 0 to effectively negate a value.
            // Handling of +0.0f and -0.0f might differ from IEE here.
            // But in graphics practice we don't see a problem with codes
            // using this approach and a measurable 10%(+|-5%) performance gain
            return T(0) - x;
        #else
            return -x;
        #endif
    }

    OSL_FORCEINLINE Dual2<float> absf (const Dual2<float> &x)
    {
        // Avoid ternary ops whose operands have side effects
        // in favor of code that executes both sides masked
        // return x.val() >= 0.0f ? x : -x;

        // NOTE: negation happens outside of conditional, then is blended based on the condition
        Dual2<float> neg_x = negate(x);

        bool cond = x.val() < 0.0f;
        // Blend per builtin component to allow
        // the compiler to track builtins and privatize the data layout
        // versus requiring a stack location.
        float val = x.val();
        if (cond) {
            val = neg_x.val();
        }

        float dx = x.dx();
        if (cond) {
            dx = neg_x.dx();
        }

        float dy = x.dy();
        if (cond) {
            dy = neg_x.dy();
        }

        return Dual2<float>(val, dx, dy);
    }

    OSL_FORCEINLINE int absi (int x)
    {
        //return x >= 0 ? x : -x;
        return std::abs(x);
    }
#if 0
    template <typename IN_TYPE, typename OUT_TYPE>
    OSL_FORCEINLINE OUT_TYPE bit_cast (const IN_TYPE val) {
        static_assert(sizeof(IN_TYPE) == sizeof(OUT_TYPE), "when casting between types they must be the same size");
        union {
            IN_TYPE inVal;
            OUT_TYPE outVal;
        } in_or_out;
        in_or_out.inVal = val;
        return in_or_out.outVal;
    }
#else
    // C++20 has std::bit_cast, although explicit SIMD may still
    // be unhappy
    template <typename IN_TYPE, typename OUT_TYPE>
    OSL_FORCEINLINE OUT_TYPE bit_cast (const IN_TYPE val) {
        static_assert(sizeof(IN_TYPE) == sizeof(OUT_TYPE), "when casting between types they must be the same size");
        OUT_TYPE r;
        memcpy(&r, &val, sizeof(OUT_TYPE));
        return r;
    }
#endif

#ifdef __INTEL_COMPILER
    // Although the union based type punning in the bit_cast template
    // is legal and supported by strict type analysis, a vectorizing compiler
    // may take it too literally and create a temporary and attempt to share
    // memory location, as a union is supposed to do.
    // Within a SIMD loop, this however can result in a AOS of unions, with
    // a gather to pull the data back out in the new type.
    // Theoretically a compiler could try and use a SOA of its union when
    // certain conditions are met.  Until support to that level is reached,
    // we can instead use compiler specific intrinsics to perform the cast
    // Specialize the template for casts which compiler intrinsics exist
    template <>
    OSL_FORCEINLINE uint32_t bit_cast<float, uint32_t> (const float val) {
          static_assert(sizeof(float) == sizeof(uint32_t), "when casting between types they must be the same size");
          return static_cast<uint32_t>(_castf32_u32(val));
    }
    template <>
    OSL_FORCEINLINE int32_t bit_cast<float, int32_t> (const float val) {
          static_assert(sizeof(float) == sizeof(int32_t), "when casting between types they must be the same size");
          return static_cast<int32_t>(_castf32_u32(val));
    }
    template <>
    OSL_FORCEINLINE float bit_cast<uint32_t, float> (const uint32_t val) {
          static_assert(sizeof(uint32_t) == sizeof(float), "when casting between types they must be the same size");
          return _castu32_f32(val);
    }
    template <>
    OSL_FORCEINLINE float bit_cast<int32_t, float> (const int32_t val) {
          static_assert(sizeof(int32_t) == sizeof(float), "when casting between types they must be the same size");
          return _castu32_f32(val);
    }

    template <>
    OSL_FORCEINLINE uint64_t bit_cast<double, uint64_t> (const double val) {
          static_assert(sizeof(double) == sizeof(uint64_t), "when casting between types they must be the same size");
          return static_cast<uint64_t>(_castf64_u64(val));
    }
    template <>
    OSL_FORCEINLINE int64_t bit_cast<double, int64_t> (const double val) {
          static_assert(sizeof(double) == sizeof(int64_t), "when casting between types they must be the same size");
          return static_cast<int64_t>(_castf64_u64(val));
    }
    template <>
    OSL_FORCEINLINE double bit_cast<uint64_t, double> (const uint64_t val) {
          static_assert(sizeof(uint64_t) == sizeof(double), "when casting between types they must be the same size");
          return _castu64_f64(val);
    }
    template <>
    OSL_FORCEINLINE double bit_cast<int64_t, double> (const int64_t val) {
          static_assert(sizeof(int64_t) == sizeof(double), "when casting between types they must be the same size");
          return _castu64_f64(val);
    }
#endif

    OSL_FORCEINLINE int bitcast_to_int (float x) { return bit_cast<float,int>(x); }
    OSL_FORCEINLINE float bitcast_to_float (int x) { return bit_cast<int,float>(x); }

    /// clamp a to bounds [low,high].
    template <class T>
    OSL_FORCEINLINE T
    clamp (T a, T low, T high)
    {
        // OIIO clamp only does the 2nd comparison in the else
        // block, which will generate extra code in a SIMD masking scenario
        //return (a < low) ? low : ((a > high) ? high : a);

        // The following should result in a max and min instruction, thats it
        if (a < low) {
            a = low;
        }
        if (a > high) {
            a = high;
        }
        return a;
    }

    template<typename T>
    T log2 (const T& xval); // undefined

    template<>
    OSL_FORCEINLINE float log2<float> (const float& xval) {
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
    OSL_FORCEINLINE Dual2<float> log2<Dual2<float>>(const Dual2<float> &a)
    {
        float loga = log2(a.val());
        float aln2 = a.val() * float(M_LN2);
        float inva = aln2 < std::numeric_limits<float>::min() ? 0.0f : 1.0f / aln2;
        return Dual2<float> (loga, inva * a.dx(), inva * a.dy());
    }

    template<typename T>
    OSL_FORCEINLINE T log (const T& x) {
        // Examined 2130706432 values of logf on [1.17549435e-38,3.40282347e+38]: 0.313865375 avg ulp diff, 5148137 max ulp, 7.62939e-06 max error
        return log2(x) * T(M_LN2);
    }

    template<typename T>
    OSL_FORCEINLINE T log10 (const T& x) {
        // Examined 2130706432 values of log10f on [1.17549435e-38,3.40282347e+38]: 0.631237033 avg ulp diff, 4471615 max ulp, 3.8147e-06 max error
        return log2(x) * T(M_LN2 / M_LN10);
    }

    OSL_FORCEINLINE float logb (float x) {
        // don't bother with denormals
        x = absf(x);
        if (x < std::numeric_limits<float>::min()) x = std::numeric_limits<float>::min();
        if (x > std::numeric_limits<float>::max()) x = std::numeric_limits<float>::max();
        unsigned bits = bit_cast<float, unsigned>(x);
        return float (int(bits >> 23) - 127);
    }

    template<typename T>
    OSL_FORCEINLINE T exp2 (const T & xval); // undefined

    template<>
    OSL_FORCEINLINE float exp2<float> (const float & xval) {
#if OSL_NON_INTEL_CLANG
        // Not ideal, but CLANG was unhappy using the bitcast/memcpy/reinter_cast/union
        // inside an explicit SIMD loop, so revert to calling the standard version
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
//        union {
//            float float_val;
//            unsigned int uint_val;
//        } float_or_uint;
//        float_or_uint.float_val = r;
//        float_or_uint.uint_val += (static_cast<unsigned>(m) << 23);
//        return float_or_uint.float_val;
#endif
    }

    template<>
    OSL_FORCEINLINE Dual2<float> exp2<Dual2<float>>(const Dual2<float> &a)
    {
        float exp2a = sfm::template exp2(a.val());
        return Dual2<float> (exp2a, exp2a*float(M_LN2)*a.dx(), exp2a*float(M_LN2)*a.dy());
    }

    template <typename T>
    OSL_FORCEINLINE T exp (const T& x) {
        // Examined 2237485550 values of exp on [-87.3300018,87.3300018]: 2.6666452 avg ulp diff, 230 max ulp
        return sfm::exp2(x * T(1 / M_LN2));
    }

    OSL_FORCEINLINE float expm1 (float x) {
        if (absf(x) < 0.03f) {
            float y = 1.0f - (1.0f - x); // crush denormals
            return copysignf(madd(0.5f, y * y, y), x);
        } else
            return exp(x) - 1.0f;
    }

    OSL_FORCEINLINE Dual2<float> expm1(const Dual2<float> &a)
    {
        float expm1a = expm1(a.val());
        float expa   = exp  (a.val());
        return Dual2<float> (expm1a, expa * a.dx(), expa * a.dy());
    }

    OSL_FORCEINLINE float erf(const float x)
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

    OSL_FORCEINLINE Dual2<float> erf(const Dual2<float> &a)
    {
        float erfa = erf (a.val());
        float two_over_sqrt_pi = 1.128379167095512573896158903f;
        float derfadx = exp(-a.val() * a.val()) * two_over_sqrt_pi;
        return Dual2<float> (erfa, derfadx * a.dx(), derfadx * a.dy());
    }

    OSL_FORCEINLINE float erfc (float x)
    {
        // Examined 2164260866 values of erfcf on [-4,4]: 1.90735e-06 max error
        // ulp histogram:
        //   0  = 80.30%
        return 1.0f - erf(x);
    }

    OSL_FORCEINLINE Dual2<float> erfc(const Dual2<float> &a)
    {
        float erfa = erfc (a.val());
        float two_over_sqrt_pi = -1.128379167095512573896158903f;
        float derfadx = exp(-a.val() * a.val()) * two_over_sqrt_pi;
        //float derfadx = std::exp(-a.val() * a.val()) * two_over_sqrt_pi;
        return Dual2<float> (erfa, derfadx * a.dx(), derfadx * a.dy());
    }



    OSL_FORCEINLINE float safe_pow (float x, float y) {
        if (y == 0) return 1.0f; // x^0=1
        if (x == 0) return 0.0f; // 0^y=0
        // be cheap & exact for special case of squaring and identity
        if (y == 1.0f)
            return x;
        if (y == 2.0f)
            return std::min (x*x, std::numeric_limits<float>::max());
        float sign = 1.0f;
        if (OSL_UNLIKELY(x < 0)) {
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

    OSL_FORCEINLINE Dual2<float> safe_pow(const Dual2<float> &u, const Dual2<float> &v)
    {
        // NOTE: same issue as above (fast_safe_pow does even more clamping)
        float powuvm1 = sfm::safe_pow (u.val(), v.val() - 1.0f);
        float powuv   = powuvm1 * u.val();
        float logu    = u.val() > 0 ? log(u.val()) : 0.0f;
        return Dual2<float> ( powuv, v.val()*powuvm1 * u.dx() + logu*powuv * v.dx(),
                                     v.val()*powuvm1 * u.dy() + logu*powuv * v.dy() );
    }

    OSL_FORCEINLINE float safe_fmod (float a, float b) {
        //return (b != 0.0f) ? std::fmod (a,b) : 0.0f;
        if (OSL_LIKELY(b != 0.0f)) {
            // return std::fmod (a,b);
            // std::fmod was getting called serially instead
            // of vectorizing, so we will just do the
            // calculation ourselves
            //
            // The floating-point remainder of the division operation
            // a/b is a - N*b, where N = a/b with its fractional part truncated.
            int N = static_cast<int>(a/b);
            return a - N*b;
        }
        return 0.0f;
    }

    OSL_FORCEINLINE Dual2<float> safe_fmod (const Dual2<float> &a, const Dual2<float> &b) {
        return Dual2<float> (safe_fmod (a.val(), b.val()), a.dx(), a.dy());
    }

#if 0 // emitted directly by llvm_gen_wide.cpp
    OSL_FORCEINLINE float safe_div(float a, float b) {
        return (b != 0.0f) ? (a / b) : 0.0f;
    }

    OSL_FORCEINLINE int safe_div(int a, int b) {
        return (b != 0) ? (a / b) : 0;
    }
#endif


    /// Round to nearest integer, returning as an int.
    OSL_FORCEINLINE int fast_rint (float x) {
        // used by sin/cos/tan range reduction
    #if 0
        // single roundps instruction on SSE4.1+ (for gcc/clang at least)
        //return static_cast<int>(rintf(x));
        return rintf(x);
    #else
        // emulate rounding by adding/substracting 0.5
        return static_cast<int>(x + copysignf(0.5f, x));

        // Other possible factorings
        //return (x >= 0.0f) ? static_cast<int>(x + 0.5f) : static_cast<int>(x - 0.5f);
        //return static_cast<int>(x +  (x >= 0.0f) ? 0.5f : - 0.5f);
        //float pad = (x >= 0.0f) ? 0.5f : - 0.5f;
        //return static_cast<int>(x + pad);
        //return nearbyint(x);
#endif
    }


    OSL_FORCEINLINE float sin (float x) {
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

        /* ORIGINAL OIIO 1.7 did the following, which would create a discontinuity
         * should u be just 1ulp past 1.0.  This was happening with FMA hardware
         * do to reduced amount of rounding for intermediate results.
         * It makes more sense to just perform a true clamping operation.
         *   // For large x, the argument reduction can fail and the polynomial can be
         *   // evaluated with arguments outside the valid internal. Just clamp the bad
         *   // values away (setting to 0.0f means no branches need to be generated).
         *   if (fabsf(u) > 1.0f) u = 0.0f;
         */
        return clamp(u,-1.0f, 1.0f);
    }
    OSL_FORCEINLINE float cos (float x) {
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
        /* ORIGINAL OIIO 1.7 did the following, which would create a discontinuity
         * should u be just 1ulp past 1.0.  This was happening with FMA hardware
         * do to reduced amount of rounding for intermediate results.
         * It makes more sense to just perform a true clamping operation.
         *   if (fabsf(u) > 1.0f) u = 0.0f;
         */
        return clamp(u,-1.0f, 1.0f);
    }
    OSL_FORCEINLINE void sincos (float x, float & sine, float& cosine) {
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
        /* ORIGINAL OIIO 1.7 did the following, which would create a discontinuity
         * should su or cu be just 1ulp past 1.0.  This was happening with FMA hardware
         * do to reduced amount of rounding for intermediate results.
         * It makes more sense to just perform a true clamping operation.
         *   if (fabsf(su) > 1.0f) su = 0.0f;
         *   if (fabsf(cu) > 1.0f) cu = 0.0f;
         */
        sine   = clamp(su,-1.0f, 1.0f);;
        cosine = clamp(cu,-1.0f, 1.0f);;
    }

    // NOTE: this approximation is only valid on [-8192.0,+8192.0], it starts becoming
    // really poor outside of this range because the reciprocal amplifies errors
    OSL_FORCEINLINE float tan (float x) {
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

    OSL_FORCEINLINE Dual2<float> tan(const Dual2<float> &a)
    {
        float tana  = sfm::tan (a.val());
        float cosa  = sfm::cos (a.val());
        float sec2a = 1 / (cosa * cosa);
        return Dual2<float> (tana, sec2a * a.dx(), sec2a * a.dy());
    }

    OSL_FORCEINLINE float atan (float x) {
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

    OSL_FORCEINLINE Dual2<float> atan(const Dual2<float> &a)
    {
        float arctana = sfm::atan(a.val());
        float denom   = 1.0f / (1.0f + a.val() * a.val());
        return Dual2<float> (arctana, denom * a.dx(), denom * a.dy());

    }

    OSL_FORCEINLINE float atan2 (float y, float x) {
        // based on atan approximation above
        // the special cases around 0 and infinity were tested explicitly
        // the only case not handled correctly is x=NaN,y=0 which returns 0 instead of nan
        const float a = absf(x);
        const float b = absf(y);

        //const float k = (b == 0) ? 0.0f : ((a == b) ? 1.0f : (b > a ? a / b : b / a));
        // When applying to all lanes in SIMD, we end up doing extra masking and 2 divides.
        // So lets just do 1 divide and swap the parameters instead.
        // And if we are going to do a doing a divide anyway, when a == b it should be 1.0f anyway
        // so lets not bother special casing it.
        bool b_is_greater_than_a = b > a;
        float sa = b_is_greater_than_a ? b : a;
        float sb = b_is_greater_than_a ? a : b;
        const float k = (b == 0) ? 0.0f : sb/sa;

        const float s = 1.0f - (1.0f - k); // crush denormals
        const float t = s * s;

        float r = s * madd(0.43157974f, t, 1.0f) / madd(madd(0.05831938f, t, 0.76443945f), t, 1.0f);

        if (b_is_greater_than_a) r = 1.570796326794896557998982f - r; // account for arg reduction
        // TODO:  investigate if testing x < 0.0f is more efficient or even the same
        if (bit_cast<float, unsigned>(x) & 0x80000000u) // test sign bit of x
            r = float(M_PI) - r;
        return copysignf(r, y);
    }

    OSL_FORCEINLINE Dual2<float> atan2(const Dual2<float> &y, const Dual2<float> &x)
    {
        float atan2xy = sfm::atan2(y.val(), x.val());
        // NOTE: using bitwise & to avoid branches
        float denom = (x.val() == 0 & y.val() == 0) ? 0.0f : 1.0f / (x.val() * x.val() + y.val() * y.val());
        return Dual2<float> ( atan2xy, (y.val()*x.dx() - x.val()*y.dx())*denom,
                                       (y.val()*x.dy() - x.val()*y.dy())*denom );
    }

    OSL_FORCEINLINE float cosh (float x) {
        // Examined 2237485550 values of cosh on [-87.3300018,87.3300018]: 1.78256726 avg ulp diff, 178 max ulp
        float e = sfm::exp(absf(x));
        return 0.5f * e + 0.5f / e;
    }

    OSL_FORCEINLINE float sinh (float x) {
        float a = absf(x);
        if (OSL_UNLIKELY(a > 1.0f)) {
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
        float e = sfm::exp(2.0f * absf(x));
        return copysignf(1 - 2 / (1 + e), x);
    }

    OSL_FORCEINLINE Dual2<float> cosh(const Dual2<float> &a)
    {
        float cosha = sfm::cosh(a.val());
        float sinha = sfm::sinh(a.val());
        return Dual2<float> (cosha, sinha * a.dx(), sinha * a.dy());
    }

    OSL_FORCEINLINE Dual2<float> sinh(const Dual2<float> &a)
    {
        float cosha = sfm::cosh(a.val());
        float sinha = sfm::sinh(a.val());
        return Dual2<float> (sinha, cosha * a.dx(), cosha * a.dy());
    }

    OSL_FORCEINLINE Dual2<float> tanh(const Dual2<float> &a)
    {
        float tanha = sfm::tanh(a.val());
        float cosha = sfm::cosh(a.val());
        float sech2a = 1 / (cosha * cosha);
        return Dual2<float> (tanha, sech2a * a.dx(), sech2a * a.dy());
    }

    OSL_FORCEINLINE Dual2<float> sin(const Dual2<float> &a)
    {
        float sina, cosa;
        sfm::sincos (a.val(), sina, cosa);
        return Dual2<float> (sina, cosa * a.dx(), cosa * a.dy());
    }

    OSL_FORCEINLINE Dual2<float> cos(const Dual2<float> &a)
    {
        float sina, cosa;
        sfm::sincos (a.val(), sina, cosa);
        return Dual2<float> (cosa, -sina * a.dx(), -sina * a.dy());
    }


    // because lengthTiny does alot of work including another
    // sqrt, we really want to skip that if possible because
    // with SIMD execution, we end up doing the sqrt twice
    // and blending the results.  Although code could be
    // refactored to do a single sqrt, think its better
    // to skip the code block as we don't expect near 0 lengths
    // TODO: get OpenEXR ImathVec to update to similar, don't think
    // it can cause harm

    // Imath::Vec3::lengthTiny is private
    // local copy here no changes
    OSL_FORCEINLINE float accessibleTinyLength(const Vec3 &N)
    {
//        float absX = (N.x >= float (0))? N.x: -N.x;
//        float absY = (N.y >= float (0))? N.y: -N.y;
//        float absZ = (N.z >= float (0))? N.z: -N.z;
        // gcc builtin for abs is 2 instructions using bit twiddling vs. compares
        float absX = absf(N.x);
        float absY = absf(N.y);
        float absZ = absf(N.z);

        float max = absX;

        if (max < absY)
            max = absY;

        if (max < absZ)
            max = absZ;

        if (OSL_UNLIKELY(max == 0.0f))
            return 0.0f;

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

    OSL_FORCEINLINE
    float length(const Vec3 &N)
    {
        float length2 = N.dot (N);

        if (OSL_UNLIKELY(length2 < float (2) * Imath::limits<float>::smallest()))
            return accessibleTinyLength(N);

        return Imath::Math<float>::sqrt (length2);
    }

    OSL_FORCEINLINE Vec3
    normalize(const Vec3 &N)
    {
        float l = length(N);

        if (OSL_UNLIKELY(l == float (0)))
            return Vec3 (float (0));

        return Vec3 (N.x / l, N.y / l, N.z / l);
    }


    OSL_FORCEINLINE Dual2<Vec3>
    normalize (const Dual2<Vec3> &a)
    {
        // NOTE: using bitwise & to avoid branches
        if (OSL_UNLIKELY(a.val().x == 0.0f & a.val().y == 0.0f & a.val().z == 0.0f)) {
            return Dual2<Vec3> (Vec3(0.0f, 0.0f, 0.0f),
                                Vec3(0.0f, 0.0f, 0.0f),
                                Vec3(0.0f, 0.0f, 0.0f));
        } else {
            Dual2<float> ax (a.val().x, a.dx().x, a.dy().x);
            Dual2<float> ay (a.val().y, a.dx().y, a.dy().y);
            Dual2<float> az (a.val().z, a.dx().z, a.dy().z);
            Dual2<float> inv_length = 1.0f / sqrt(ax*ax + ay*ay + az*az);
            ax = ax*inv_length;
            ay = ay*inv_length;
            az = az*inv_length;
            return Dual2<Vec3> (Vec3(ax.val(), ay.val(), az.val()),
                                Vec3(ax.dx(),  ay.dx(),  az.dx() ),
                                Vec3(ax.dy(),  ay.dy(),  az.dy() ));
        }
    }


    template<typename T>
    OSL_FORCEINLINE
    T max_val(T left, T right)
    {
        return (right > left)? right: left;
    }

    class Matrix33 : public Imath::Matrix33<float>
    {
    public:
        typedef Imath::Matrix33<float> parent;

        OSL_FORCEINLINE Matrix33 (Imath::Uninitialized uninit)
        : parent(uninit)
        {}

        // Avoid the memset that is part of the Imath::Matrix33
        // default constructor
        OSL_FORCEINLINE Matrix33 ()
        : parent(1.0f, 0.0f, 0.0f,
                                 0.0f, 1.0f, 0.0f,
                                 0.0f, 0.0f, 1.0f)
        {}

        OSL_FORCEINLINE Matrix33 (float a, float b, float c, float d, float e, float f, float g, float h, float i)
        : parent(a,b,c,d,e,f,g,h,i)
        {}

        OSL_FORCEINLINE Matrix33 (const Imath::Matrix33<float> &a)
        : parent(a)
        {}

        // Avoid the memcpy that is part of the Imath::Matrix33
        OSL_FORCEINLINE
        Matrix33 (const float a[3][3])
        : Imath::Matrix33<float>(
            a[0][0], a[0][1], a[0][2],
            a[1][0], a[1][1], a[1][2],
            a[2][0], a[2][1], a[2][2])
        {}


        // Avoid the memcpy that is part of Imath::Matrix33::operator=
        OSL_FORCEINLINE Matrix33 &
        operator = (const Matrix33 &v)
        {
            parent::x[0][0] = v.x[0][0];
            parent::x[0][1] = v.x[0][1];
            parent::x[0][2] = v.x[0][2];

            parent::x[1][0] = v.x[1][0];
            parent::x[1][1] = v.x[1][1];
            parent::x[1][2] = v.x[1][2];

            parent::x[2][0] = v.x[2][0];
            parent::x[2][1] = v.x[2][1];
            parent::x[2][2] = v.x[2][2];

            return *this;
        }


        // Avoid Imath::Matrix33::operator * that
        // initializing values to 0 before overwriting them
        // Also manually unroll its nested loops
        OSL_FORCEINLINE Matrix33
        operator * (const Matrix33 &v) const
        {
            Matrix33 tmp(Imath::UNINITIALIZED);

            tmp.x[0][0] = parent::x[0][0] * v.x[0][0] +
                          parent::x[0][1] * v.x[1][0] +
                          parent::x[0][2] * v.x[2][0];
            tmp.x[0][1] = parent::x[0][0] * v.x[0][1] +
                    parent::x[0][1] * v.x[1][1] +
                    parent::x[0][2] * v.x[2][1];
            tmp.x[0][2] = parent::x[0][0] * v.x[0][2] +
                    parent::x[0][1] * v.x[1][2] +
                    parent::x[0][2] * v.x[2][2];

            tmp.x[1][0] = parent::x[1][0] * v.x[0][0] +
                    parent::x[1][1] * v.x[1][0] +
                    parent::x[1][2] * v.x[2][0];
            tmp.x[1][1] = parent::x[1][0] * v.x[0][1] +
                    parent::x[1][1] * v.x[1][1] +
                    parent::x[1][2] * v.x[2][1];
            tmp.x[1][2] = parent::x[1][0] * v.x[0][2] +
                    parent::x[1][1] * v.x[1][2] +
                    parent::x[1][2] * v.x[2][2];

            tmp.x[2][0] = parent::x[2][0] * v.x[0][0] +
                    parent::x[2][1] * v.x[1][0] +
                    parent::x[2][2] * v.x[2][0];
            tmp.x[2][1] = parent::x[2][0] * v.x[0][1] +
                    parent::x[2][1] * v.x[1][1] +
                    parent::x[2][2] * v.x[2][1];
            tmp.x[2][2] = parent::x[2][0] * v.x[0][2] +
                    parent::x[2][1] * v.x[1][2] +
                    parent::x[2][2] * v.x[2][2];

            return tmp;
        }
    };


    OSL_FORCEINLINE sfm::Matrix33
    make_matrix33_cols (const Vec3 &a, const Vec3 &b, const Vec3 &c)
    {
        return sfm::Matrix33 (a.x, b.x, c.x,
                         a.y, b.y, c.y,
                         a.z, b.z, c.z);
    }



    // Considering having functionally equivalent versions of Vec3, Color3, Matrix44
    // with slight modifications to inlining and implementation to avoid aliasing and
    // improve likelyhood of proper privation of local variables within a SIMD loop

} // namespace sfm

} // namespace __OSL_WIDE_PVT or pvt




OSL_NAMESPACE_EXIT
