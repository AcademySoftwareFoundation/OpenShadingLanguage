/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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


#define UINT_MAX	0xffffffff
#define UINT64_MAX	0xffffffffffffffffU
#define FLT_MIN		1.175494351e-38F
#define FLT_MAX		3.402823466e+38F
#define FLT_EPSILON 1.192092896e-07F

#define M_E        2.71828182845904523536
#define M_LOG2E    1.44269504088896340736
#define M_LOG10E   0.434294481903251827651
#define M_LN2      0.693147180559945309417
#define M_LN10     2.30258509299404568402
#define M_PI       3.14159265358979323846
#define M_PI_2     1.57079632679489661923
#define M_PI_4     0.785398163397448309616
#define M_1_PI     0.318309886183790671538
#define M_2_PI     0.636619772367581343076
#define M_2_SQRTPI 1.12837916709551257390
#define M_SQRT2    1.41421356237309504880
#define M_SQRT1_2  0.707106781186547524401


#ifndef uint32_t
typedef unsigned int uint32_t;
#endif

#ifndef uint64_t
typedef unsigned long long uint64_t;
#endif


template <typename T>
inline T min(T a, T b)
{
	return ((a < b) ? a : b);
}

template <typename T>
inline T max(T a, T b)
{
	return ((a > b) ? a : b);
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
//
// INTEGER HELPER FUNCTIONS
//
// A variety of handy functions that operate on integers.
//

/// Quick test for whether an integer is a power of 2.
///
template<typename T>
inline bool
ispow2 (T x)
{
    // Numerous references for this bit trick are on the web.  The
    // principle is that x is a power of 2 <=> x == 1<<b <=> x-1 is
    // all 1 bits for bits < b.
    return (x & (x-1)) == 0 && (x >= 0);
}



/// Round up to next higher power of 2 (return x if it's already a power
/// of 2).
inline int
pow2roundup (int x)
{
    // Here's a version with no loops.
    if (x < 0)
        return 0;
    // Subtract 1, then round up to a power of 2, that way if we are
    // already a power of 2, we end up with the same number.
    --x;
    // Make all bits past the first 1 also be 1, i.e. 0001xxxx -> 00011111
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    // Now we have 2^n-1, by adding 1, we make it a power of 2 again
    return x+1;
}



/// Round down to next lower power of 2 (return x if it's already a power
/// of 2).
inline int
pow2rounddown (int x)
{
    // Make all bits past the first 1 also be 1, i.e. 0001xxxx -> 00011111
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    // Strip off all but the high bit, i.e. 00011111 -> 00010000
    // That's the power of two <= the original x
    return x & ~(x >> 1);
}



/// Round up to the next whole multiple of m.
///
inline int
round_to_multiple (int x, int m)
{
    return ((x + m - 1) / m) * m;
}



/// Round up to the next whole multiple of m, for the special case where
/// m is definitely a power of 2 (somewhat simpler than the more general
/// round_to_multiple). This is a template that should work for any
// integer type.
template<typename T>
inline T
round_to_multiple_of_pow2 (T x, T m)
{
    DASSERT (ispow2 (m));
    return (x + m - 1) & (~(m-1));
}



/// Multiply two unsigned 32-bit ints safely, carefully checking for
/// overflow, and clamping to uint32_t's maximum value.
inline uint32_t
clamped_mult32 (uint32_t a, uint32_t b)
{
    const uint32_t Err = UINT_MAX;
    uint64_t r = (uint64_t)a * (uint64_t)b;   // Multiply into a bigger int
    return r < Err ? (uint32_t)r : Err;
}



/// Multiply two unsigned 64-bit ints safely, carefully checking for
/// overflow, and clamping to uint64_t's maximum value.
inline uint64_t
clamped_mult64 (uint64_t a, uint64_t b)
{
    uint64_t ab = a*b;
    if (b && ab/b != a)
        return UINT64_MAX;
    else
        return ab;
}



/// Bitwise circular rotation left by k bits (for 32 bit unsigned integers)
inline uint32_t rotl32 (uint32_t x, int k) {
    return (x<<k) | (x>>(32-k));
}

/// Bitwise circular rotation left by k bits (for 64 bit unsigned integers)
inline uint64_t rotl64 (uint64_t x, int k) {
    return (x<<k) | (x>>(64-k));
}


// (end of integer helper functions)
////////////////////////////////////////////////////////////////////////////




////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
//
// FLOAT UTILITY FUNCTIONS
//
// Helper/utility functions: clamps, blends, interpolations...
//


inline float fast_fabs(float x)
{
	union { float f; uint32_t i; } v = {x};
	v.i &= 0x7FFFFFFF;
	return v.f;
}

inline float fast_copysign(float x, float y)
{
	union { float fval; uint32_t lval; } hx, hy;

    hx.fval = x;
    hy.fval = y;
      
    hx.lval &= 0x7fffffff;
    hx.lval |= hy.lval & 0x80000000;
      
    return hx.fval;
}

inline float fast_sqrt(float x)
{
    // prevent from division by zero
    if (x <= 0.0f) { return 0.0f; }
    union {
        uint32_t intPart;
        float floatPart;
    } convertor, convertor2;
    convertor.floatPart = x;
    convertor2.floatPart = x;
    convertor.intPart = 0x1fbcf800 + (convertor.intPart >> 1);
    convertor2.intPart = 0x5f3759df - (convertor2.intPart >> 1);
    float y = 0.5f * (convertor.floatPart + (x * convertor2.floatPart));
    // two more iterations
    y = 0.5f * (y + x / y); // absolute error within 0.07
    y = 0.5f * (y + x / y); // absolute error within 0.000031
    return y;
}

/// clamp a to bounds [low,high].
template <class T>
inline T
clamp (T a, T low, T high)
{
    return (a < low) ? low : ((a > high) ? high : a);
}



/// Multiply and add: (a * b) + c
template <typename T>
inline T madd (const T& a, const T& b, const T& c) {
    // NOTE:  in the future we may want to explicitly ask for a fused
    // multiply-add in a specialized version for float.
    // NOTE2: GCC/ICC will turn this (for float) into a FMA unless
    // explicitly asked not to, clang seems to leave the code alone.
    return a * b + c;
}



/// Linearly interpolate values v0-v1 at x: v0*(1-x) + v1*x.
/// This is a template, and so should work for any types.
template <class T, class Q>
inline T
lerp (const T& v0, const T& v1, const Q& x)
{
    // NOTE: a*(1-x) + b*x is much more numerically stable than a+x*(b-a)
    return v0*(Q(1)-x) + v1*x;
}



/// Bilinearly interoplate values v0-v3 (v0 upper left, v1 upper right,
/// v2 lower left, v3 lower right) at coordinates (s,t) and return the
/// result.  This is a template, and so should work for any types.
template <class T, class Q>
inline T
bilerp(const T& v0, const T& v1, const T& v2, const T& v3, const Q& s, const Q& t)
{
    // NOTE: a*(t-1) + b*t is much more numerically stable than a+t*(b-a)
    Q s1 = Q(1) - s;
    return T ((Q(1)-t)*(v0*s1 + v1*s) + t*(v2*s1 + v3*s));
}



/// Bilinearly interoplate arrays of values v0-v3 (v0 upper left, v1
/// upper right, v2 lower left, v3 lower right) at coordinates (s,t),
/// storing the results in 'result'.  These are all vectors, so do it
/// for each of 'n' contiguous values (using the same s,t interpolants).
template <class T, class Q>
inline void
bilerp (const T *v0, const T *v1,
        const T *v2, const T *v3,
        Q s, Q t, int n, T *result)
{
    Q s1 = Q(1) - s;
    Q t1 = Q(1) - t;
    for (int i = 0;  i < n;  ++i)
        result[i] = T (t1*(v0[i]*s1 + v1[i]*s) + t*(v2[i]*s1 + v3[i]*s));
}



/// Bilinearly interoplate arrays of values v0-v3 (v0 upper left, v1
/// upper right, v2 lower left, v3 lower right) at coordinates (s,t),
/// SCALING the interpolated value by 'scale' and then ADDING to
/// 'result'.  These are all vectors, so do it for each of 'n'
/// contiguous values (using the same s,t interpolants).
template <class T, class Q>
inline void
bilerp_mad (const T *v0, const T *v1,
            const T *v2, const T *v3,
            Q s, Q t, Q scale, int n, T *result)
{
    Q s1 = Q(1) - s;
    Q t1 = Q(1) - t;
    for (int i = 0;  i < n;  ++i)
        result[i] += T (scale * (t1*(v0[i]*s1 + v1[i]*s) +
                                  t*(v2[i]*s1 + v3[i]*s)));
}



/// Trilinearly interoplate arrays of values v0-v7 (v0 upper left top, v1
/// upper right top, ...) at coordinates (s,t,r), and return the
/// result.  This is a template, and so should work for any types.
template <class T, class Q>
inline T
trilerp (T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, Q s, Q t, Q r)
{
    // NOTE: a*(t-1) + b*t is much more numerically stable than a+t*(b-a)
    Q s1 = Q(1) - s;
    Q t1 = Q(1) - t;
    Q r1 = Q(1) - r;
    return T (r1*(t1*(v0*s1 + v1*s) + t*(v2*s1 + v3*s)) +
               r*(t1*(v4*s1 + v5*s) + t*(v6*s1 + v7*s)));
}



/// Trilinearly interoplate arrays of values v0-v7 (v0 upper left top, v1
/// upper right top, ...) at coordinates (s,t,r),
/// storing the results in 'result'.  These are all vectors, so do it
/// for each of 'n' contiguous values (using the same s,t,r interpolants).
template <class T, class Q>
inline void
trilerp (const T *v0, const T *v1, const T *v2, const T *v3,
         const T *v4, const T *v5, const T *v6, const T *v7,
         Q s, Q t, Q r, int n, T *result)
{
    Q s1 = Q(1) - s;
    Q t1 = Q(1) - t;
    Q r1 = Q(1) - r;
    for (int i = 0;  i < n;  ++i)
        result[i] = T (r1*(t1*(v0[i]*s1 + v1[i]*s) + t*(v2[i]*s1 + v3[i]*s)) +
                        r*(t1*(v4[i]*s1 + v5[i]*s) + t*(v6[i]*s1 + v7[i]*s)));
}



/// Trilinearly interoplate arrays of values v0-v7 (v0 upper left top, v1
/// upper right top, ...) at coordinates (s,t,r),
/// SCALING the interpolated value by 'scale' and then ADDING to
/// 'result'.  These are all vectors, so do it for each of 'n'
/// contiguous values (using the same s,t,r interpolants).
template <class T, class Q>
inline void
trilerp_mad (const T *v0, const T *v1, const T *v2, const T *v3,
             const T *v4, const T *v5, const T *v6, const T *v7,
             Q s, Q t, Q r, Q scale, int n, T *result)
{
    Q r1 = Q(1) - r;
    bilerp_mad (v0, v1, v2, v3, s, t, scale*r1, n, result);
    bilerp_mad (v4, v5, v6, v7, s, t, scale*r, n, result);
}



/// Evaluate B-spline weights in w[0..3] for the given fraction.  This
/// is an important component of performing a cubic interpolation.
template <typename T>
inline void evalBSplineWeights (T w[4], T fraction)
{
    T one_frac = 1 - fraction;
    w[0] = T(1.0 / 6.0) * one_frac * one_frac * one_frac;
    w[1] = T(2.0 / 3.0) - T(0.5) * fraction * fraction * (2 - fraction);
    w[2] = T(2.0 / 3.0) - T(0.5) * one_frac * one_frac * (2 - one_frac);
    w[3] = T(1.0 / 6.0) * fraction * fraction * fraction;
}


/// Evaluate B-spline derivative weights in w[0..3] for the given
/// fraction.  This is an important component of performing a cubic
/// interpolation with derivatives.
template <typename T>
inline void evalBSplineWeightDerivs (T dw[4], T fraction)
{
    T one_frac = 1 - fraction;
    dw[0] = -T(0.5) * one_frac * one_frac;
    dw[1] =  T(0.5) * fraction * (3 * fraction - 4);
    dw[2] = -T(0.5) * one_frac * (3 * one_frac - 4);
    dw[3] =  T(0.5) * fraction * fraction;
}



/// Bicubically interoplate arrays of pointers arranged in a 4x4 pattern
/// with val[0] pointing to the data in the upper left corner, val[15]
/// pointing to the lower right) at coordinates (s,t), storing the
/// results in 'result'.  These are all vectors, so do it for each of
/// 'n' contiguous values (using the same s,t interpolants).
template <class T>
inline void
bicubic_interp (const T **val, T s, T t, int n, T *result)
{
    for (int c = 0;  c < n;  ++c)
        result[c] = T(0);
    T wx[4]; evalBSplineWeights (wx, s);
    T wy[4]; evalBSplineWeights (wy, t);
    for (int j = 0;  j < 4;  ++j) {
        for (int i = 0;  i < 4;  ++i) {
            T w = wx[i] * wy[j];
            for (int c = 0;  c < n;  ++c)
                result[c] += w * val[j*4+i][c];
        }
    }
}



/// Return (x-floor(x)) and put (int)floor(x) in *xint.  This is similar
/// to the built-in modf, but returns a true int, always rounds down
/// (compared to modf which rounds toward 0), and always returns
/// frac >= 0 (comapred to modf which can return <0 if x<0).
inline float
floorfrac (float x, int *xint)
{
    // Find the greatest whole number <= x.  This cast is faster than
    // calling floorf.
    int i = (int) x - (x < 0.0f ? 1 : 0);
    *xint = i;
    return x - static_cast<float>(i);   // Return the fraction left over
}



/// Convert degrees to radians.
template <typename T>
inline T radians (T deg) { return deg * T(M_PI / 180.0); }

/// Convert radians to degrees
template <typename T>
inline T degrees (T rad) { return rad * T(180.0 / M_PI); }


// (end of float helper functions)
////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
//
// CONVERSION
//
// Type and range conversion helper functions and classes.


/// Helper template to let us tell if two types are the same.
template<typename T, typename U> struct is_same { static const bool value = false; };
template<typename T> struct is_same<T,T> { static const bool value = true; };



template <typename IN_TYPE, typename OUT_TYPE>
inline OUT_TYPE bit_cast (const IN_TYPE in) {
	union { IN_TYPE in_val; OUT_TYPE out_val; } cvt;
    cvt.in_val = in;
	return cvt.out_val;
}


/// Simple conversion of a (presumably non-negative) float into a
/// rational.  This does not attempt to find the simplest fraction
/// that approximates the float, for example 52.83 will simply
/// return 5283/100.  This does not attempt to gracefully handle
/// floats that are out of range that could be easily int/int.
inline void
float_to_rational (float f, unsigned int &num, unsigned int &den)
{
    if (f <= 0) {   // Trivial case of zero, and handle all negative values
        num = 0;
        den = 1;
    } else if ((int)(1.0/f) == (1.0/f)) { // Exact results for perfect inverses
        num = 1;
        den = (int)f;
    } else {
        num = (int)f;
        den = 1;
        while (fast_fabs(f-static_cast<float>(num)) > 0.00001f && den < 1000000) {
            den *= 10;
            f *= 10;
            num = (int)f;
        }
    }
}



/// Simple conversion of a float into a rational.  This does not attempt
/// to find the simplest fraction that approximates the float, for
/// example 52.83 will simply return 5283/100.  This does not attempt to
/// gracefully handle floats that are out of range that could be easily
/// int/int.
inline void
float_to_rational (float f, int &num, int &den)
{
    unsigned int n, d;
    float_to_rational (fast_fabs(f), n, d);
    num = (f >= 0) ? (int)n : -(int)n;
    den = (int) d;
}


// (end of conversion helpers)
////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
//
// FAST & APPROXIMATE MATH
//
// The functions named "fast_*" provide a set of replacements to libm that
// are much faster at the expense of some accuracy and robust handling of
// extreme values. One design goal for these approximation was to avoid
// branches as much as possible and operate on single precision values only
// so that SIMD versions should be straightforward ports We also try to
// implement "safe" semantics (ie: clamp to valid range where possible)
// natively since wrapping these inline calls in another layer would be
// wasteful.
//
// Some functions are fast_safe_*, which is both a faster approximation as
// well as clamped input domain to ensure no NaN, Inf, or divide by zero.
//


/// Round to nearest integer, returning as an int.
inline int fast_rint (float x) {
    // used by sin/cos/tan range reduction
    // emulate rounding by adding/substracting 0.5
    return static_cast<int>(x + fast_copysign(0.5f, x));
}


inline float fast_sin (float x) {
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
    if (fast_fabs(u) > 1.0f) u = 0.0f;
    return u;
}


inline float fast_cos (float x) {
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
    if (fast_fabs(u) > 1.0f) u = 0.0f;
    return u;
}

inline void fast_sincos (float x, float* sine, float* cosine) {
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
    if (fast_fabs(su) > 1.0f) su = 0.0f;
    if (fast_fabs(cu) > 1.0f) cu = 0.0f;
    *sine   = su;
    *cosine = cu;
}

// NOTE: this approximation is only valid on [-8192.0,+8192.0], it starts becoming
// really poor outside of this range because the reciprocal amplifies errors
inline float fast_tan (float x) {
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

/// Fast, approximate sin(x*M_PI) with maximum absolute error of 0.000918954611.
/// Adapted from http://devmaster.net/posts/9648/fast-and-accurate-sine-cosine#comment-76773
inline float fast_sinpi (float x)
{
	// Fast trick to strip the integral part off, so our domain is [-1,1]
	const float z = x - ((x + 25165824.0f) - 25165824.0f);
    const float y = z - z * fast_fabs(z);
    const float Q = 3.10396624f;
    const float P = 3.584135056f; // P = 16-4*Q
    return y * (Q + P * fast_fabs(y));
    /* The original article used used inferior constants for Q and P and
     * so had max error 1.091e-3.
     *
     * The optimal value for Q was determined by exhaustive search, minimizing
     * the absolute numerical error relative to float(std::sin(double(phi*M_PI)))
     * over the interval [0,2] (which is where most of the invocations happen).
     * 
     * The basic idea of this approximation starts with the coarse approximation:
     *      sin(pi*x) ~= f(x) =  4 * (x - x * abs(x))
     *
     * This approximation always _over_ estimates the target. On the otherhand, the
     * curve:
     *      sin(pi*x) ~= f(x) * abs(f(x)) / 4
     *
     * always lies _under_ the target. Thus we can simply numerically search for the
     * optimal constant to LERP these curves into a more precise approximation.
     * After folding the constants together and simplifying the resulting math, we
     * end up with the compact implementation below.
     *
     * NOTE: this function actually computes sin(x * pi) which avoids one or two
     * mults in many cases and guarantees exact values at integer periods.
     */
}

/// Fast approximate cos(x*M_PI) with ~0.1% absolute error.
inline float fast_cospi (float x)
{
    return fast_sinpi (x+0.5f);
}

inline float fast_acos (float x) {
    const float f = fast_fabs(x);
    const float m = (f < 1.0f) ? 1.0f - (1.0f - f) : 1.0f; // clamp and crush denormals
    // based on http://www.pouet.net/topic.php?which=9132&page=2
    // 85% accurate (ulp 0)
    // Examined 2130706434 values of acos: 15.2000597 avg ulp diff, 4492 max ulp, 4.51803e-05 max error // without "denormal crush"
    // Examined 2130706434 values of acos: 15.2007108 avg ulp diff, 4492 max ulp, 4.51803e-05 max error // with "denormal crush"
    const float a = fast_sqrt(1.0f - m) * (1.5707963267f + m * (-0.213300989f + m * (0.077980478f + m * -0.02164095f)));
    return x < 0 ? float(M_PI) - a : a;
}

inline float fast_asin (float x) {
    // based on acosf approximation above
    // max error is 4.51133e-05 (ulps are higher because we are consistently off by a little amount)
    const float f = fast_fabs(x);
    const float m = (f < 1.0f) ? 1.0f - (1.0f - f) : 1.0f; // clamp and crush denormals
    const float a = float(M_PI_2) - fast_sqrt(1.0f - m) * (1.5707963267f + m * (-0.213300989f + m * (0.077980478f + m * -0.02164095f)));
    return fast_copysign(a, x);
}

inline float fast_atan (float x) {
    const float a = fast_fabs(x);
    const float k = a > 1.0f ? 1 / a : a;
    const float s = 1.0f - (1.0f - k); // crush denormals
    const float t = s * s;
    // http://mathforum.org/library/drmath/view/62672.html
    // Examined 4278190080 values of atan: 2.36864877 avg ulp diff, 302 max ulp, 6.55651e-06 max error      // (with  denormals)
    // Examined 4278190080 values of atan: 171160502 avg ulp diff, 855638016 max ulp, 6.55651e-06 max error // (crush denormals)
    float r = s * madd(0.43157974f, t, 1.0f) / madd(madd(0.05831938f, t, 0.76443945f), t, 1.0f);
    if (a > 1.0f) r = 1.570796326794896557998982f - r;
    return fast_copysign(r, x);
}

inline float fast_atan2 (float y, float x) {
    // based on atan approximation above
    // the special cases around 0 and infinity were tested explicitly
    // the only case not handled correctly is x=NaN,y=0 which returns 0 instead of nan
    const float a = fast_fabs(x);
    const float b = fast_fabs(y);

    const float k = (b == 0) ? 0.0f : ((a == b) ? 1.0f : (b > a ? a / b : b / a));
    const float s = 1.0f - (1.0f - k); // crush denormals
    const float t = s * s;

    float r = s * madd(0.43157974f, t, 1.0f) / madd(madd(0.05831938f, t, 0.76443945f), t, 1.0f);

    if (b > a) r = 1.570796326794896557998982f - r; // account for arg reduction
    if (bit_cast<float, unsigned>(x) & 0x80000000u) // test sign bit of x
        r = float(M_PI) - r;
    return fast_copysign(r, y);
}

static inline float fast_log2 (float x) {
    // NOTE: clamp to avoid special cases and make result "safe" from large negative values/nans
    if (x < FLT_MIN) x = FLT_MIN;
    if (x > FLT_MAX) x = FLT_MAX;
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

inline float fast_log (float x) {
    // Examined 2130706432 values of logf on [1.17549435e-38,3.40282347e+38]: 0.313865375 avg ulp diff, 5148137 max ulp, 7.62939e-06 max error
    return fast_log2(x) * float(M_LN2);
}

inline float fast_log10 (float x) {
    // Examined 2130706432 values of log10f on [1.17549435e-38,3.40282347e+38]: 0.631237033 avg ulp diff, 4471615 max ulp, 3.8147e-06 max error
    return fast_log2(x) * float(M_LN2 / M_LN10);
}

inline float fast_logb (float x) {
    // don't bother with denormals
    x = fast_fabs(x);
    if (x < FLT_MIN) x = FLT_MIN;
    if (x > FLT_MAX) x = FLT_MAX;
    unsigned bits = bit_cast<float, unsigned>(x);
    return int(bits >> 23) - 127;
}

inline float fast_exp2 (float x) {
    // clamp to safe range for final addition
    if (x < -126.0f) x = -126.0f;
    if (x >  126.0f) x =  126.0f;
    // range reduction
    int m = int(x); x -= m;
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
    return bit_cast<unsigned, float>(bit_cast<float, unsigned>(r) + (unsigned(m) << 23));
}

inline float fast_exp (float x) {
    // Examined 2237485550 values of exp on [-87.3300018,87.3300018]: 2.6666452 avg ulp diff, 230 max ulp
    return fast_exp2(x * float(1 / M_LN2));
}

inline float fast_exp10 (float x) {
    // Examined 2217701018 values of exp10 on [-37.9290009,37.9290009]: 2.71732409 avg ulp diff, 232 max ulp
    return fast_exp2(x * float(M_LN10 / M_LN2));
}

inline float fast_expm1 (float x) {
    if (fast_fabs(x) < 1e-5f) {
        x = 1.0f - (1.0f - x); // crush denormals
        return madd(0.5f, x * x, x);
    } else
        return fast_exp(x) - 1.0f;
}

inline float fast_sinh (float x) {
    float a = fast_fabs(x);
    if (a > 1.0f) {
        // Examined 53389559 values of sinh on [1,87.3300018]: 33.6886442 avg ulp diff, 178 max ulp
        float e = fast_exp(a);
        return fast_copysign(0.5f * e - 0.5f / e, x);
    } else {
        a = 1.0f - (1.0f - a); // crush denorms
        float a2 = a * a;
        // degree 7 polynomial generated with sollya
        // Examined 2130706434 values of sinh on [-1,1]: 1.19209e-07 max error
        float r = 2.03945513931e-4f;
        r = madd(r, a2, 8.32990277558e-3f);
        r = madd(r, a2, 0.1666673421859f);
        r = madd(r * a, a2, a);
        return fast_copysign(r, x);
    }
}

inline float fast_cosh (float x) {
    // Examined 2237485550 values of cosh on [-87.3300018,87.3300018]: 1.78256726 avg ulp diff, 178 max ulp
    float e = fast_exp(fast_fabs(x));
    return 0.5f * e + 0.5f / e;
}

inline float fast_tanh (float x) {
    // Examined 4278190080 values of tanh on [-3.40282347e+38,3.40282347e+38]: 3.12924e-06 max error
    // NOTE: ulp error is high because of sub-optimal handling around the origin
    float e = fast_exp(2.0f * fast_fabs(x));
    return fast_copysign(1 - 2 / (1 + e), x);
}

inline float fast_safe_pow (float x, float y) {
    if (y == 0) return 1.0f; // x^0=1
    if (x == 0) return 0.0f; // 0^y=0
    // be cheap & exact for special case of squaring and identity
    if (y == 1.0f)
        return x;
    if (y == 2.0f)
        return min (x*x, FLT_MAX);
    float sign = 1.0f;
    if (x < 0) {
        // if x is negative, only deal with integer powers
        // powf returns NaN for non-integers, we will return 0 instead
        int ybits = bit_cast<float, int>(y) & 0x7fffffff;
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
    return sign * fast_exp2(y * fast_log2(fast_fabs(x)));
}

inline float fast_erf (float x)
{
    // Examined 1082130433 values of erff on [0,4]: 1.93715e-06 max error
    // Abramowitz and Stegun, 7.1.28
    const float a1 = 0.0705230784f;
    const float a2 = 0.0422820123f;
    const float a3 = 0.0092705272f;
    const float a4 = 0.0001520143f;
    const float a5 = 0.0002765672f;
    const float a6 = 0.0000430638f;
    const float a = fast_fabs(x);
    const float b = 1.0f - (1.0f - a); // crush denormals
    const float r = madd(madd(madd(madd(madd(madd(a6, b, a5), b, a4), b, a3), b, a2), b, a1), b, 1.0f);
    const float s = r * r; // ^2
    const float t = s * s; // ^4
    const float u = t * t; // ^8
    const float v = u * u; // ^16
    return fast_copysign(1.0f - 1.0f / v, x);
}

inline float fast_erfc (float x)
{
    // Examined 2164260866 values of erfcf on [-4,4]: 1.90735e-06 max error
    // ulp histogram:
    //   0  = 80.30%
    return 1.0f - fast_erf(x);
}

inline float fast_ierf (float x)
{
    // from: Approximating the erfinv function by Mike Giles
    // to avoid trouble at the limit, clamp input to 1-eps
    float a = fast_fabs(x); if (a > 0.99999994f) a = 0.99999994f;
    float w = -fast_log((1.0f - a) * (1.0f + a)), p;
    if (w < 5.0f) {
        w = w - 2.5f;
        p =  2.81022636e-08f;
        p = madd(p, w,  3.43273939e-07f);
        p = madd(p, w, -3.5233877e-06f );
        p = madd(p, w, -4.39150654e-06f);
        p = madd(p, w,  0.00021858087f );
        p = madd(p, w, -0.00125372503f );
        p = madd(p, w, -0.00417768164f );
        p = madd(p, w,  0.246640727f   );
        p = madd(p, w,  1.50140941f    );
    } else {
        w = fast_sqrt(w) - 3.0f;
        p = -0.000200214257f;
        p = madd(p, w,  0.000100950558f);
        p = madd(p, w,  0.00134934322f );
        p = madd(p, w, -0.00367342844f );
        p = madd(p, w,  0.00573950773f );
        p = madd(p, w, -0.0076224613f  );
        p = madd(p, w,  0.00943887047f );
        p = madd(p, w,  1.00167406f    );
        p = madd(p, w,  2.83297682f    );
    }
    return p * x;
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
//
// SAFE MATH
//
// The functions named "safe_*" are versions with various internal clamps
// or other deviations from IEEE standards with the specific intent of
// never producing NaN or Inf values or throwing exceptions. But within the
// valid range, they should be full precision and match IEEE standards.
//


/// Safe (clamping) sqrt: safe_sqrt(x<0) returns 0, not NaN.
template <typename T>
inline T safe_sqrt (T x) {
    return x >= T(0) ? fast_sqrt(x) : T(0);
}

/// Safe (clamping) inverse sqrt: safe_inversesqrt(x<=0) returns 0.
template <typename T>
inline T safe_inversesqrt (T x) {
    return x > T(0) ? T(1) / fast_sqrt(x) : T(0);
}


/// Safe (clamping) arcsine: clamp to the valid domain.
template <typename T>
inline T safe_asin (T x) {
    if (x <= T(-1)) return T(-M_PI_2);
    if (x >= T(+1)) return T(+M_PI_2);
    return fast_asin(x);
}

/// Safe (clamping) arccosine: clamp to the valid domain.
template <typename T>
inline T safe_acos (T x) {
    if (x <= T(-1)) return T(M_PI);
    if (x >= T(+1)) return T(0);
    return fast_acos(x);
}


/// Safe log2: clamp to valid domain.
inline float safe_log2 (float x) {
    // match clamping from fast version
    if (x < FLT_MIN) x = FLT_MIN;
    if (x > FLT_MAX) x = FLT_MAX;
    return fast_log2(x);
}

/// Safe log: clamp to valid domain.
inline float safe_log (float x) {
    // slightly different than fast version since clamping happens before scaling
    if (x < FLT_MIN) x = FLT_MIN;
    if (x > FLT_MAX) x = FLT_MAX;
    return fast_log(x);
}

/// Safe log10: clamp to valid domain.
inline float safe_log10 (float x) {
    // slightly different than fast version since clamping happens before scaling
    if (x < FLT_MIN) x = FLT_MIN;
    if (x > FLT_MAX) x = FLT_MAX;
    return fast_log10(x);
}

/// Safe logb: clamp to valid domain.
inline float safe_logb (float x) {
    return (x != 0.0f) ? fast_logb(x) : -FLT_MAX;
}

// (end of safe_* functions)
////////////////////////////////////////////////////////////////////////////

namespace Imath {

template <class T>
inline T
abs (T a)
{
    return (a > T(0)) ? a : -a;
}

template <class T>
inline int
floor (T x)
{
    return (x >= 0)? int (x): -(int (-x) + (-x > int (-x)));
}

template <class T>
inline int
ceil (T x)
{
    return -floor (-x);
}

template <class T>
inline int
round (T x)
{
    return floor (x + T(0.5));
}

template <class T>
inline int
trunc (T x)
{
    return (x >= 0) ? int(x) : -int(-x);
}

template <class T>
inline T 
fmod(T a, T b)
{
    return (a - b * T(floor(a / b)));
}

inline bool 
finitef (float f)
{
    union {float f; int i;} u;
    u.f = f;

    return (u.i & 0x7f800000) != 0x7f800000;
}

template <class T> struct limits
{
    static T	min();
    static T	max();
    static T	smallest();
    static T	epsilon();
    static bool	isIntegral();
    static bool	isSigned();
};

template <>
struct limits <float>
{
    static float		min()		{return -FLT_MAX;}
    static float		max()		{return FLT_MAX;}
    static float		smallest()	{return FLT_MIN;}
    static float		epsilon()	{return FLT_EPSILON;}
    static bool			isIntegral()	{return false;}
    static bool			isSigned()	{return true;}
};

template <class T>
struct Math
{
	static T fabs(T x);
	static T sqrt(T x);
	static T sin(T x);
	static T cos(T x);
};

template <>
struct Math<float>
{
	static float fabs(float x) { return fast_fabs(x); }
	static float sqrt(float x) { return fast_sqrt(x); }
	static float sin(float x) { return fast_sin(x); }
	static float cos(float x) { return fast_cos(x); }
};

}
