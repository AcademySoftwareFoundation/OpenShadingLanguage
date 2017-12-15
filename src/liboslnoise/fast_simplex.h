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

#include <limits>

#include <OSL/dual.h>
#include <OSL/dual_vec.h>
#include <OSL/wide.h>
#include <OpenImageIO/hash.h>
#include <OpenImageIO/simd.h>

OSL_NAMESPACE_ENTER


///////////////////////////////////////////////////////////////////////
// Implementation follows...
//
// Users don't need to worry about this part
///////////////////////////////////////////////////////////////////////

namespace pvt {


template<int WidthT>
OSL_NOINLINE  void
fast_simplexnoise3(WideAccessor<float, WidthT> wresult, ConstWideAccessor<Vec3, WidthT> wp);

template<int WidthT>
OSL_NOINLINE  void
fast_simplexnoise3(WideAccessor<Vec3, WidthT> wresult, ConstWideAccessor<Vec3, WidthT> wp);

template<int WidthT>
OSL_NOINLINE  void
fast_simplexnoise4(WideAccessor<Vec3, WidthT> wresult,
                        ConstWideAccessor<Vec3, WidthT> wp,
                        ConstWideAccessor<float,WidthT> wt);

template<int WidthT>
OSL_NOINLINE  void
fast_usimplexnoise1(WideAccessor<Vec3, WidthT> wresult, ConstWideAccessor<float, WidthT> wx);

template<int WidthT>
OSL_NOINLINE  void
fast_usimplexnoise3(WideAccessor<float, WidthT> wresult, ConstWideAccessor<Vec3, WidthT> wp);

template<int WidthT>
OSL_NOINLINE  void
fast_usimplexnoise3(WideAccessor<Vec3, WidthT> wresult, ConstWideAccessor<Vec3, WidthT> wp);

template<int WidthT>
OSL_NOINLINE  void
fast_usimplexnoise4(WideAccessor<Vec3, WidthT> wresult,
                        ConstWideAccessor<Vec3, WidthT> wp,
                        ConstWideAccessor<float,WidthT> wt);

namespace {

// return the greatest integer <= x
inline int quick_floor (float x) {
	//	return (int) x - ((x < 0) ? 1 : 0);
	
	// std::floor is another option, however that appears to be
	// a function call right now, and this sequence appears cheaper
	//return static_cast<int>(x - ((x < 0.0f) ? 1.0f : 0.0f));
	
	// This factoring should allow the expensive float to integer
	// conversion to happen at the same time the comparison is
	// in an out of order CPU
	return (static_cast<int>(x)) - ((x < 0.0f) ? 1 : 0);
}

#if 0

// return the greatest integer <= x, for 4 values at once
OIIO_FORCEINLINE int4 quick_floor (const float4& x) {
#if 0
    // Even on SSE 4.1, this is actually very slightly slower!
    // Continue to test on future architectures.
    return floori(x);
#else
    int4 b (x);  // truncates
    int4 isneg = bitcast_to_int4 (x < float4::Zero());
    return b + isneg;
    // Trick here (thanks, Cycles, for letting me spy on your code): the
    // comparison will return (int)-1 for components that are less than
    // zero, and adding that is the same as subtracting one!
#endif
}


// convert a 32 bit integer into a floating point number in [0,1]
inline float bits_to_01 (unsigned int bits) {
    // divide by 2^32-1
    return bits * (1.0f / std::numeric_limits<unsigned int>::max());
    // TODO:  I am not sure the above is numerically correct
    //return static_cast<float>(bits)/std::numeric_limits<unsigned int>::max();
    }


// Perform a bjmix (see OpenImageIO/hash.h) on 4 sets of values at once.
OIIO_FORCEINLINE void
bjmix (int4 &a, int4 &b, int4 &c)
{
    using OIIO::simd::rotl32;
    a -= c;  a ^= rotl32(c, 4);  c += b;
    b -= a;  b ^= rotl32(a, 6);  a += c;
    c -= b;  c ^= rotl32(b, 8);  b += a;
    a -= c;  a ^= rotl32(c,16);  c += b;
    b -= a;  b ^= rotl32(a,19);  a += c;
    c -= b;  c ^= rotl32(b, 4);  b += a;
}


// Perform a bjfinal (see OpenImageIO/hash.h) on 4 sets of values at once.
OIIO_FORCEINLINE int4
bjfinal (const int4& a_, const int4& b_, const int4& c_)
{
    using OIIO::simd::rotl32;
	int4 a(a_), b(b_), c(c_);
    c ^= b; c -= rotl32(b,14);
    a ^= c; a -= rotl32(c,11);
    b ^= a; b -= rotl32(a,25);
    c ^= b; c -= rotl32(b,16);
    a ^= c; a -= rotl32(c,4);
    b ^= a; b -= rotl32(a,14);
    c ^= b; c -= rotl32(b,24);
    return c;
}


/// hash an array of N 32 bit values into a pseudo-random value
/// based on my favorite hash: http://burtleburtle.net/bob/c/lookup3.c
/// templated so that the compiler can unroll the loops for us
template <int N>
inline unsigned int
inthash (const unsigned int k[N]) {
    // now hash the data!
    unsigned int a, b, c, len = N;
    a = b = c = 0xdeadbeef + (len << 2) + 13;
    while (len > 3) {
        a += k[0];
        b += k[1];
        c += k[2];
        OIIO::bjhash::bjmix(a, b, c);
        len -= 3;
        k += 3;
    }
    switch (len) {
        case 3 : c += k[2];
        case 2 : b += k[1];
        case 1 : a += k[0];
        c = OIIO::bjhash::bjfinal(a, b, c);
        case 0:
            break;
    }
    return c;
}


// Do four 2D hashes simultaneously.
inline int4
inthash_simd (const int4& key_x, const int4& key_y)
{
    const int len = 2;
    const int seed_ = (0xdeadbeef + (len << 2) + 13);
    static const OIIO_SIMD4_ALIGN int seed[4] = { seed_,seed_,seed_,seed_};
    int4 a = (*(int4*)&seed)+key_x, b = (*(int4*)&seed)+key_y, c = (*(int4*)&seed);
    return bjfinal (a, b, c);
}


// Do four 3D hashes simultaneously.
inline int4
inthash_simd (const int4& key_x, const int4& key_y, const int4& key_z)
{
    const int len = 3;
    const int seed_ = (0xdeadbeef + (len << 2) + 13);
    static const OIIO_SIMD4_ALIGN int seed[4] = { seed_,seed_,seed_,seed_};
    int4 a = (*(int4*)&seed)+key_x, b = (*(int4*)&seed)+key_y, c = (*(int4*)&seed)+key_z;
    return bjfinal (a, b, c);
}



// Do four 3D hashes simultaneously.
inline int4
inthash_simd (const int4& key_x, const int4& key_y, const int4& key_z, const int4& key_w)
{
    const int len = 4;
    const int seed_ = (0xdeadbeef + (len << 2) + 13);
    static const OIIO_SIMD4_ALIGN int seed[4] = { seed_,seed_,seed_,seed_};
    int4 a = (*(int4*)&seed)+key_x, b = (*(int4*)&seed)+key_y, c = (*(int4*)&seed)+key_z;
    bjmix (a, b, c);
    a += key_w;
    return bjfinal(a, b, c);
}







// Define select(bool,truevalue,falsevalue) template that works for a
// variety of types that we can use for both scalars and vectors. Because ?:
// won't work properly in template code with vector ops.
template <typename B, typename F>
OIIO_FORCEINLINE F select (const B& b, const F& t, const F& f) { return b ? t : f; }

template <> OIIO_FORCEINLINE int4 select (const mask4& b, const int4& t, const int4& f) {
    return blend (f, t, b);
}

template <> OIIO_FORCEINLINE float4 select (const mask4& b, const float4& t, const float4& f) {
    return blend (f, t, b);
}

template <> OIIO_FORCEINLINE float4 select (const int4& b, const float4& t, const float4& f) {
    return blend (f, t, mask4(b));
}

template <> OIIO_FORCEINLINE Dual2<float4>
select (const mask4& b, const Dual2<float4>& t, const Dual2<float4>& f) {
    return Dual2<float4> (blend (f.val(), t.val(), b),
                          blend (f.dx(),  t.dx(),  b),
                          blend (f.dy(),  t.dy(),  b));
}

template <>
OIIO_FORCEINLINE Dual2<float4> select (const int4& b, const Dual2<float4>& t, const Dual2<float4>& f) {
    return select (mask4(b), t, f);
}



// Define negate_if(value,bool) that will work for both scalars and vectors,
// as well as Dual2's of both.
template<typename FLOAT, typename BOOL>
OIIO_FORCEINLINE FLOAT negate_if (const FLOAT& val, const BOOL& b) {
    return b ? -val : val;
}

template<> OIIO_FORCEINLINE float4 negate_if (const float4& val, const int4& b) {
    // Special case negate_if for SIMD -- can do it with bit tricks, no branches
    int4 highbit (0x80000000);
    return bitcast_to_float4 (bitcast_to_int4(val) ^ (blend0 (highbit, mask4(b))));
}

// Special case negate_if for SIMD -- can do it with bit tricks, no branches
template<> OIIO_FORCEINLINE Dual2<float4> negate_if (const Dual2<float4>& val, const int4& b)
{
    return Dual2<float4> (negate_if (val.val(), b),
                          negate_if (val.dx(),  b),
                          negate_if (val.dy(),  b));
}


// Define shuffle<> template that works with Dual2<float4> analogously to
// how it works for float4.
template<int i0, int i1, int i2, int i3>
OIIO_FORCEINLINE Dual2<float4> shuffle (const Dual2<float4>& a)
{
    return Dual2<float4> (OIIO::simd::shuffle<i0,i1,i2,i3>(a.val()),
                          OIIO::simd::shuffle<i0,i1,i2,i3>(a.dx()),
                          OIIO::simd::shuffle<i0,i1,i2,i3>(a.dy()));
}

template<int i>
OIIO_FORCEINLINE Dual2<float4> shuffle (const Dual2<float4>& a)
{
    return Dual2<float4> (OIIO::simd::shuffle<i>(a.val()),
                          OIIO::simd::shuffle<i>(a.dx()),
                          OIIO::simd::shuffle<i>(a.dy()));
}

// Define extract<> that works with Dual2<float4> analogously to how it
// works for float4.
template<int i>
OIIO_FORCEINLINE Dual2<float> extract (const Dual2<float4>& a)
{
    return Dual2<float> (OIIO::simd::extract<i>(a.val()),
                         OIIO::simd::extract<i>(a.dx()),
                         OIIO::simd::extract<i>(a.dy()));
}



// Equivalent to OIIO::bilerp (a, b, c, d, u, v), but if abcd are already
// packed into a float4. We assume T is float and VECTYPE is float4,
// but it also works if T is Dual2<float> and VECTYPE is Dual2<float4>.
template<typename T, typename VECTYPE>
OIIO_FORCEINLINE T bilerp (VECTYPE abcd, T u, T v) {
    VECTYPE xx = OIIO::lerp (abcd, OIIO::simd::shuffle<1,1,3,3>(abcd), u);
    return OIIO::simd::extract<0>(OIIO::lerp (xx,OIIO::simd::shuffle<2>(xx), v));
}

// Equivalent to OIIO::bilerp (a, b, c, d, u, v), but if abcd are already
// packed into a float4 and uv are already packed into the first two
// elements of a float4. We assume VECTYPE is float4, but it also works if
// VECTYPE is Dual2<float4>.
OIIO_FORCEINLINE Dual2<float> bilerp (const Dual2<float4>& abcd, const Dual2<float4>& uv) {
    Dual2<float4> xx = OIIO::lerp (abcd, shuffle<1,1,3,3>(abcd), shuffle<0>(uv));
    return extract<0>(OIIO::lerp (xx,shuffle<2>(xx), shuffle<1>(uv)));
}


// Equivalent to OIIO::trilerp (a, b, c, d, e, f, g, h, u, v, w), but if
// abcd and efgh are already packed into float4's and uvw are packed into
// the first 3 elements of a float4.
OIIO_FORCEINLINE float trilerp (const float4& abcd, const float4& efgh, const float4& uvw) {
    // Interpolate along z axis by w
    float4 xy = OIIO::lerp (abcd, efgh, OIIO::simd::shuffle<2>(uvw));
    // Interpolate along x axis by u
    float4 xx = OIIO::lerp (xy, OIIO::simd::shuffle<1,1,3,3>(xy), OIIO::simd::shuffle<0>(uvw));
    // interpolate along y axis by v
    return OIIO::simd::extract<0>(OIIO::lerp (xx, OIIO::simd::shuffle<2>(xx), OIIO::simd::shuffle<1>(uvw)));
}



// always return a value inside [0,b) - even for negative numbers
inline int imod(int a, int b) {
    a %= b;
    return a < 0 ? a + b : a;
}

// imod four values at once
inline int4 imod(const int4& a, int b) {
    int4 c = a % b;
    return c + select(c < 0, int4(b), int4::Zero());
}

// floorfrac return quick_floor as well as the fractional remainder
// FIXME: already implemented inside OIIO but can't easily override it for duals
//        inside a different namespace
inline float floorfrac(float x, int* i) {
    *i = quick_floor(x);
    return x - *i;
}

// floorfrac with derivs
inline Dual2<float> floorfrac(const Dual2<float> &x, int* i) {
    float frac = floorfrac(x.val(), i);
    // slope of x is not affected by this operation
    return Dual2<float>(frac, x.dx(), x.dy());
}

// floatfrac for four sets of values at once.
inline float4 floorfrac(const float4& x, int4 * i) {
#if 0
    float4 thefloor = floor(x);
    *i = int4(thefloor);
    return x-thefloor;
#else
    int4 thefloor = quick_floor (x);
    *i = thefloor;
    return x - float4(thefloor);
#endif
}

// floorfrac with derivs, computed on 4 values at once.
inline Dual2<float4> floorfrac(const Dual2<float4> &x, int4* i) {
    float4 frac = floorfrac(x.val(), i);
    // slope of x is not affected by this operation
    return Dual2<float4>(frac, x.dx(), x.dy());
}


// Perlin 'fade' function. Can be overloaded for float, Dual2, as well
// as float4 / Dual2<float4>.
template <typename T>
inline T fade (const T &t) { 
   return t * t * t * (t * (t * T(6.0f) - T(15.0f)) + T(10.0f));
}


// 1,2,3 and 4 dimensional gradient functions - perform a dot product against a
// randomly chosen vector. Note that the gradient vector is not normalized, but
// this only affects the overall "scale" of the result, so we simply account for
// the scale by multiplying in the corresponding "perlin" function.
// These factors were experimentally calculated to be:
//    1D:   0.188
//    2D:   0.507
//    3D:   0.936
//    4D:   0.870

template <typename T>
inline T grad (int hash, const T &x) {
    int h = hash & 15;
    float g = 1 + (h & 7);  // 1, 2, .., 8
    if (h&8) g = -g;        // random sign
    return g * x;           // dot-product
}

template <typename I, typename T>
inline T grad (const I &hash, const T &x, const T &y) {
    // 8 possible directions (+-1,+-2) and (+-2,+-1)
    I h = hash & 7;
    T u = select (h<4, x, y);
    T v = 2.0f * select (h<4, y, x);
    // compute the dot product with (x,y).
    return negate_if(u, h&1) + negate_if(v, h&2);
}

template <typename I, typename T>
inline T grad (const I &hash, const T &x, const T &y, const T &z) {
    // use vectors pointing to the edges of the cube
    I h = hash & 15;
    T u = select (h<8, x, y);
    T v = select (h<4, y, select ((h==I(12))|(h==I(14)), x, z));
    return negate_if(u,h&1) + negate_if(v,h&2);
}

template <typename I, typename T>
inline T grad (const I &hash, const T &x, const T &y, const T &z, const T &w) {
    // use vectors pointing to the edges of the hypercube
    I h = hash & 31;
    T u = select (h<24, x, y);
    T v = select (h<16, y, z);
    T s = select (h<8 , z, w);
    return negate_if(u,h&1) + negate_if(v,h&2) + negate_if(s,h&4);
}

typedef Imath::Vec3<int> Vec3i;

inline Vec3 grad (const Vec3i &hash, float x) {
    return Vec3 (grad (hash.x, x),
                 grad (hash.y, x),
                 grad (hash.z, x));
}

inline Dual2<Vec3> grad (const Vec3i &hash, Dual2<float> x) {
    Dual2<float> rx = grad (hash.x, x);
    Dual2<float> ry = grad (hash.y, x);
    Dual2<float> rz = grad (hash.z, x);
    return make_Vec3 (rx, ry, rz);
}


inline Vec3 grad (const Vec3i &hash, float x, float y) {
    return Vec3 (grad (hash.x, x, y),
                 grad (hash.y, x, y),
                 grad (hash.z, x, y));
}

inline Dual2<Vec3> grad (const Vec3i &hash, Dual2<float> x, Dual2<float> y) {
    Dual2<float> rx = grad (hash.x, x, y);
    Dual2<float> ry = grad (hash.y, x, y);
    Dual2<float> rz = grad (hash.z, x, y);
    return make_Vec3 (rx, ry, rz);
}

inline Vec3 grad (const Vec3i &hash, float x, float y, float z) {
    return Vec3 (grad (hash.x, x, y, z),
                 grad (hash.y, x, y, z),
                 grad (hash.z, x, y, z));
}

inline Dual2<Vec3> grad (const Vec3i &hash, Dual2<float> x, Dual2<float> y, Dual2<float> z) {
    Dual2<float> rx = grad (hash.x, x, y, z);
    Dual2<float> ry = grad (hash.y, x, y, z);
    Dual2<float> rz = grad (hash.z, x, y, z);
    return make_Vec3 (rx, ry, rz);
}

inline Vec3 grad (const Vec3i &hash, float x, float y, float z, float w) {
    return Vec3 (grad (hash.x, x, y, z, w),
                 grad (hash.y, x, y, z, w),
                 grad (hash.z, x, y, z, w));
}

inline Dual2<Vec3> grad (const Vec3i &hash, Dual2<float> x, Dual2<float> y, Dual2<float> z, Dual2<float> w) {
    Dual2<float> rx = grad (hash.x, x, y, z, w);
    Dual2<float> ry = grad (hash.y, x, y, z, w);
    Dual2<float> rz = grad (hash.z, x, y, z, w);
    return make_Vec3 (rx, ry, rz);
}
#endif



// Gradient directions for 3D.
// These vectors are based on the midpoints of the 12 edges of a cube.
// A larger array of random unit length vectors would also do the job,
// but these 12 (including 4 repeats to make the array length a power
// of two) work better. They are not random, they are carefully chosen
// to represent a small, isotropic set of directions.
// Store in SOA data layout using our Wide helper
static const Wide<Vec3,16> fast_grad3lut_wide(
	Vec3(  1.0f,  0.0f,  1.0f ), Vec3(  0.0f,  1.0f,  1.0f ), // 12 cube edges
	Vec3( -1.0f,  0.0f,  1.0f ), Vec3( 0.0f, -1.0f,  1.0f ),
	Vec3(  1.0f,  0.0f, -1.0f ), Vec3( 0.0f,  1.0f, -1.0f ),
	Vec3( -1.0f,  0.0f, -1.0f ), Vec3(  0.0f, -1.0f, -1.0f ),
	Vec3(  1.0f, -1.0f,  0.0f ), Vec3( 1.0f,  1.0f,  0.0f ),
	Vec3( -1.0f,  1.0f,  0.0f ), Vec3( -1.0f, -1.0f,  0.0f ),
	Vec3(  1.0f,  0.0f,  1.0f ), Vec3( -1.0f,  0.0f,  1.0f ), // 4 repeats to make 16
	Vec3(  0.0f,  1.0f, -1.0f ), Vec3( 0.0f, -1.0f, -1.0f ));

static const Wide<Vec4,32> fast_grad4lut_wide(
  Vec4( 0.0f, 1.0f, 1.0f, 1.0f ),  Vec4( 0.0f, 1.0f, 1.0f, -1.0f ),  Vec4( 0.0f, 1.0f, -1.0f, 1.0f ),  Vec4( 0.0f, 1.0f, -1.0f, -1.0f ), // 32 tesseract edges
  Vec4( 0.0f, -1.0f, 1.0f, 1.0f ), Vec4( 0.0f, -1.0f, 1.0f, -1.0f ), Vec4( 0.0f, -1.0f, -1.0f, 1.0f ), Vec4( 0.0f, -1.0f, -1.0f, -1.0f ),
  Vec4( 1.0f, 0.0f, 1.0f, 1.0f ),  Vec4( 1.0f, 0.0f, 1.0f, -1.0f ),  Vec4( 1.0f, 0.0f, -1.0f, 1.0f ),  Vec4( 1.0f, 0.0f, -1.0f, -1.0f ),
  Vec4( -1.0f, 0.0f, 1.0f, 1.0f ), Vec4( -1.0f, 0.0f, 1.0f, -1.0f ), Vec4( -1.0f, 0.0f, -1.0f, 1.0f ), Vec4( -1.0f, 0.0f, -1.0f, -1.0f ),
  Vec4( 1.0f, 1.0f, 0.0f, 1.0f ),  Vec4( 1.0f, 1.0f, 0.0f, -1.0f ),  Vec4( 1.0f, -1.0f, 0.0f, 1.0f ),  Vec4( 1.0f, -1.0f, 0.0f, -1.0f ),
  Vec4( -1.0f, 1.0f, 0.0f, 1.0f ), Vec4( -1.0f, 1.0f, 0.0f, -1.0f ), Vec4( -1.0f, -1.0f, 0.0f, 1.0f ), Vec4( -1.0f, -1.0f, 0.0f, -1.0f ),
  Vec4( 1.0f, 1.0f, 1.0f, 0.0f ),  Vec4( 1.0f, 1.0f, -1.0f, 0.0f ),  Vec4( 1.0f, -1.0f, 1.0f, 0.0f ),  Vec4( 1.0f, -1.0f, -1.0f, 0.0f ),
  Vec4( -1.0f, 1.0f, 1.0f, 0.0f ), Vec4( -1.0f, 1.0f, -1.0f, 0.0f ), Vec4( -1.0f, -1.0f, 1.0f, 0.0f ), Vec4( -1.0f, -1.0f, -1.0f, 0.0f ));

static const unsigned char fast_simplex[64][4] = {
  {0,1,2,3},{0,1,3,2},{0,0,0,0},{0,2,3,1},{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,2,3,0},
  {0,2,1,3},{0,0,0,0},{0,3,1,2},{0,3,2,1},{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,3,2,0},
  {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},
  {1,2,0,3},{0,0,0,0},{1,3,0,2},{0,0,0,0},{0,0,0,0},{0,0,0,0},{2,3,0,1},{2,3,1,0},
  {1,0,2,3},{1,0,3,2},{0,0,0,0},{0,0,0,0},{0,0,0,0},{2,0,3,1},{0,0,0,0},{2,1,3,0},
  {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},
  {2,0,1,3},{0,0,0,0},{0,0,0,0},{0,0,0,0},{3,0,1,2},{3,0,2,1},{0,0,0,0},{3,1,2,0},
  {2,1,0,3},{0,0,0,0},{0,0,0,0},{0,0,0,0},{3,1,0,2},{0,0,0,0},{3,2,0,1},{3,2,1,0}};


//#define OSL_VERIFY_SIMPLEX3 1	
#ifdef OSL_VERIFY_SIMPLEX3
		static void osl_verify_fail(int lineNumber, const char *expression)
		{
			std::cout << "Line " << __LINE__ << " failed OSL_VERIFY(" << expression << ")" << std::endl; 
			exit(1);
		}
	#define OSL_VERIFY(EXPR) if((EXPR)== false) osl_verify_fail(__LINE__, #EXPR); 
#endif

struct fast {
	
	static inline uint32_t
	scramble (uint32_t v0, uint32_t v1=0, uint32_t v2=0)
	{
	    return OIIO::bjhash::bjfinal (v0, v1, v2^0xdeadbeef);
	}    	
	
	
	template<int seedT>
	static inline float
	grad1 (int i)
	{
	    int h = scramble (i, seedT);
	    float g = 1.0f + (h & 7);   // Gradient value is one of 1.0, 2.0, ..., 8.0
	    if (h & 8)
	        g = -g;   // Make half of the gradients negative
	    return g;
	}

	template<int seedT>
	static inline const Vec3 
	grad3 (int i, int j, int k)
	{
	    int h = scramble (i, j, scramble (k, seedT));
	    
	    //return fast_grad3lut[h & 15];
	    return fast_grad3lut_wide.get(h & 15);
	}

	template<int seedT>
	static inline const Vec4
	grad4 (int i, int j, int k, int l)
	{
	    int h = scramble (i, j, scramble (k, l, seedT));

	    // return fast_grad4lut[h & 31];
	    return fast_grad4lut_wide.get(h & 31);
	}

	// 1D simplex noise with derivative.
	// If the last argument is not null, the analytic derivative
	// is also calculated.
	template<int seedT>
	static inline float
	simplexnoise1 (float x)
	{
	    int i0 = quick_floor(x);
	    int i1 = i0 + 1;
	    float x0 = x - i0;
	    float x1 = x0 - 1.0f;

	    float x20 = x0*x0;
	    float t0 = 1.0f - x20;
	    //  if(t0 < 0.0f) t0 = 0.0f; // Never happens for 1D: x0<=1 always
	    float t20 = t0 * t0;
	    float t40 = t20 * t20;
	    float gx0 = grad1<seedT> (i0);
	    float n0 = t40 * gx0 * x0;

	    float x21 = x1*x1;
	    float t1 = 1.0f - x21;
	    //  if(t1 < 0.0f) t1 = 0.0f; // Never happens for 1D: |x1|<=1 always
	    float t21 = t1 * t1;
	    float t41 = t21 * t21;
	    float gx1 = grad1<seedT> (i1);
	    float n1 = t41 * gx1 * x1;

	    // Sum up and scale the result.  The scale is empirical, to make it
	    // cover [-1,1], and to make it approximately match the range of our
	    // Perlin noise implementation.
	    const float scale = 0.36f;

	    return scale * (n0 + n1);
	}

	// 3D simplex noise with derivatives.
	// If the last tthree arguments are not null, the analytic derivative
	// (the 3D gradient of the scalar noise field) is also calculated.
	template<int seedT>
	static inline float
	simplexnoise3 (float x, float y, float z)
	{
	    // Skewing factors for 3D simplex grid:
	    const float F3 = 0.333333333;   // = 1/3
	    const float G3 = 0.166666667;   // = 1/6

	    // Skew the input space to determine which simplex cell we're in
	    float s = (x+y+z)*F3; // Very nice and simple skew factor for 3D
	    float xs = x+s;
	    float ys = y+s;
	    float zs = z+s;

	    int i = quick_floor(xs);
	    int j = quick_floor(ys);
	    int k = quick_floor(zs);

	    float t = (float)(i+j+k)*G3; 
	    float X0 = i-t; // Unskew the cell origin back to (x,y,z) space
	    float Y0 = j-t;
	    float Z0 = k-t;
	    
	    float x0 = x-X0; // The x,y,z distances from the cell origin
	    float y0 = y-Y0;
	    float z0 = z-Z0;

	    // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
	    // Determine which simplex we are in.
	    int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
	    int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords

	    {
	    	// NOTE:  The GLSL version of the flags produced different results
	    	// These flags are derived directly from the conditional logic 
	    	// which is repeated in the verification code block following
	        int bg0 = (x0 >= y0);
	        int bg1 = (y0 >= z0);
	        int bg2 = (x0 >= z0);
			int nbg0 = !bg0;
			int nbg1 = !bg1;
			int nbg2 = !bg2;
	        i1 = bg0 & (bg1 | bg2);
	        j1 = nbg0 & bg1;
	        k1 =  nbg1 & ((bg0 & nbg2) | nbg0) ;
	        i2 = bg0 | (bg1 & bg2);
	        j2 = bg1 | nbg0;
	        k2 = (bg0 & nbg1) | (nbg0 &(nbg1 | nbg2));
	    }
#ifdef OSL_VERIFY_SIMPLEX3  // Keep around to validate the bit logic above
	    {
	    	   if (x0>=y0) {
	    		        if (y0>=z0) {
	    		            OSL_VERIFY(i1==1); 
	    		            OSL_VERIFY(j1==0);
	    		            OSL_VERIFY(k1==0);
	    		            OSL_VERIFY(i2==1);
	    		            OSL_VERIFY(j2==1);
	    		            OSL_VERIFY(k2==0);  /* X Y Z order */
	    		        } else if (x0>=z0) {
	    		        	OSL_VERIFY(i1==1); 
	    		        	OSL_VERIFY(j1==0); 
	    		        	OSL_VERIFY(k1==0); 
	    		        	OSL_VERIFY(i2==1);
	    		        	OSL_VERIFY(j2==0);
	    		        	OSL_VERIFY(k2==1);  /* X Z Y order */
	    		        } else {
	    		        	OSL_VERIFY(i1==0);
	    		        	OSL_VERIFY(j1==0);
	    		        	OSL_VERIFY(k1==1);
	    		        	OSL_VERIFY(i2==1);
	    		        	OSL_VERIFY(j2==0);
	    		        	OSL_VERIFY(k2==1);  /* Z X Y order */
	    		        }
	    		    } else { // x0<y0
	    		        if (y0<z0) {
	    		        	OSL_VERIFY(i1==0); 
	    		            OSL_VERIFY(j1==0); 
	    		            OSL_VERIFY(k1==1); 
	    		            OSL_VERIFY(i2==0); 
	    		            OSL_VERIFY(j2==1); 
	    		            OSL_VERIFY(k2==1);  /* Z Y X order */
	    		        } else if (x0<z0) {
	    		        	OSL_VERIFY(i1==0); 
	    		        	OSL_VERIFY(j1==1); 
	    		        	OSL_VERIFY(k1==0); 
	    		        	OSL_VERIFY(i2==0); 
	    		        	OSL_VERIFY(j2==1); 
	    		        	OSL_VERIFY(k2==1);  /* Y Z X order */
	    		        } else {
	    		        	OSL_VERIFY(i1==0); 
	    		        	OSL_VERIFY(j1==1); 
	    		        	OSL_VERIFY(k1==0); 
	    		        	OSL_VERIFY(i2==1); 
	    		        	OSL_VERIFY(j2==1);
	    		            OSL_VERIFY(k2==0);  /* Y X Z order */
	    		        }
	    		    }	    	
	    }
	#endif

	    // A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
	    // a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
	    // a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z),
	    // where c = 1/6.
	    float x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
	    float y1 = y0 - j1 + G3;
	    float z1 = z0 - k1 + G3;
	    float x2 = x0 - i2 + 2.0f * G3; // Offsets for third corner in (x,y,z) coords
	    float y2 = y0 - j2 + 2.0f * G3;
	    float z2 = z0 - k2 + 2.0f * G3;
	    float x3 = x0 - 1.0f + 3.0f * G3; // Offsets for last corner in (x,y,z) coords
	    float y3 = y0 - 1.0f + 3.0f * G3;
	    float z3 = z0 - 1.0f + 3.0f * G3;

	    
	    // As we do not expect any coherency between data lanes
	    // Hoisted work out of conditionals to encourage masking blending
    	// versus a check for coherency
	    // In other words we will do all the work, all the time versus
	    // trying to manage it on a per lane basis.
	    // NOTE: this may be slower if used for serial vs. simd
	    Vec3 g0 = grad3<seedT>(i, j, k);
	    Vec3 g1 = grad3<seedT>(i+i1, j+j1, k+k1);
	    Vec3 g2 = grad3<seedT>(i+i2, j+j2, k+k2);
	    Vec3 g3 = grad3<seedT>(i+1, j+1, k+1);
	    
	    // Calculate the contribution from the four corners
	    float t0 = 0.5f - x0*x0 - y0*y0 - z0*z0;
        float t20 = t0 * t0;
        float t40 = t20 * t20;
        // NOTE: avoid array access of points, always use
        // the real data members to avoid aliasing issues
        //n0 = t40 * (g0[0] * x0 + g0[1] * y0 + g0[2] * z0);
        float tn0 = t40 * (g0.x * x0 + g0.y * y0 + g0.z * z0);
	    float n0 = (t0 >= 0.0f) ? tn0 : 0.0f;

	    float t1 = 0.5f - x1*x1 - y1*y1 - z1*z1;
    	float t21 = t1 * t1;
    	float t41 = t21 * t21;
        float tn1 = t41 * (g1.x * x1 + g1.y * y1 + g1.z * z1);
	    float n1 = (t1 >= 0.0f) ? tn1 : 0.0f;

	    float t2 = 0.5f - x2*x2 - y2*y2 - z2*z2;
    	float t22 = t2 * t2;
    	float t42 = t22 * t22;
        float tn2 = t42 * (g2.x * x2 + g2.y * y2 + g2.z * z2);
	    float n2 = (t2 >= 0.0f) ? tn2 : 0.0f;

	    float t3 = 0.5f - x3*x3 - y3*y3 - z3*z3;
    	float t23 = t3 * t3;
    	float t43 = t23 * t23;
        float tn3 = t43 * (g3.x * x3 + g3.y * y3 + g3.z * z3);
	    float n3 = (t3 >= 0.0f) ? tn3 : 0.0f;
	    
	    // Sum up and scale the result.  The scale is empirical, to make it
	    // cover [-1,1], and to make it approximately match the range of our
	    // Perlin noise implementation.
	    constexpr float scale = 68.0f;
	    float noise = scale * (n0 + n1 + n2 + n3);
	
	    return noise;		
	}	


	// 4D simplex noise with derivatives.
	// If the last four arguments are not null, the analytic derivative
	// (the 4D gradient of the scalar noise field) is also calculated.
	template<int seedT>
	static inline float
	simplexnoise4 (float x, float y, float z, float w)
	{
	    // The skewing and unskewing factors are hairy again for the 4D case
	    const float F4 = 0.309016994; // F4 = (Math.sqrt(5.0)-1.0)/4.0
	    const float G4 = 0.138196601; // G4 = (5.0-Math.sqrt(5.0))/20.0

	    // Gradients at simplex corners
	    //const float *g0 = zero, *g1 = zero, *g2 = zero, *g3 = zero, *g4 = zero;

	    // Noise contributions from the four simplex corners
	    //float n0=0.0f, n1=0.0f, n2=0.0f, n3=0.0f, n4=0.0f;
	    //float t20 = 0.0f, t21 = 0.0f, t22 = 0.0f, t23 = 0.0f, t24 = 0.0f;
	    //float t40 = 0.0f, t41 = 0.0f, t42 = 0.0f, t43 = 0.0f, t44 = 0.0f;

	    // Skew the (x,y,z,w) space to determine which cell of 24 simplices we're in
	    float s = (x + y + z + w) * F4; // Factor for 4D skewing
	    float xs = x + s;
	    float ys = y + s;
	    float zs = z + s;
	    float ws = w + s;
	    int i = quick_floor(xs);
	    int j = quick_floor(ys);
	    int k = quick_floor(zs);
	    int l = quick_floor(ws);

	    float t = (i + j + k + l) * G4; // Factor for 4D unskewing
	    float X0 = i - t; // Unskew the cell origin back to (x,y,z,w) space
	    float Y0 = j - t;
	    float Z0 = k - t;
	    float W0 = l - t;

	    float x0 = x - X0;  // The x,y,z,w distances from the cell origin
	    float y0 = y - Y0;
	    float z0 = z - Z0;
	    float w0 = w - W0;

	    // For the 4D case, the simplex is a 4D shape I won't even try to describe.
	    // To find out which of the 24 possible simplices we're in, we need to
	    // determine the magnitude ordering of x0, y0, z0 and w0.
	    // The method below is a reasonable way of finding the ordering of x,y,z,w
	    // and then find the correct traversal order for the simplex we’re in.
	    // First, six pair-wise comparisons are performed between each possible pair
	    // of the four coordinates, and then the results are used to add up binary
	    // bits for an integer index into a precomputed lookup table, simplex[].
	    int c1 = (x0 > y0) ? 32 : 0;
	    int c2 = (x0 > z0) ? 16 : 0;
	    int c3 = (y0 > z0) ? 8 : 0;
	    int c4 = (x0 > w0) ? 4 : 0;
	    int c5 = (y0 > w0) ? 2 : 0;
	    int c6 = (z0 > w0) ? 1 : 0;
	    int c = c1 | c2 | c3 | c4 | c5 | c6; // '|' is mostly faster than '+'

	    int i1, j1, k1, l1; // The integer offsets for the second simplex corner
	    int i2, j2, k2, l2; // The integer offsets for the third simplex corner
	    int i3, j3, k3, l3; // The integer offsets for the fourth simplex corner

	    // simplex[c] is a 4-vector with the numbers 0, 1, 2 and 3 in some order.
	    // Many values of c will never occur, since e.g. x>y>z>w makes x<z, y<w and x<w
	    // impossible. Only the 24 indices which have non-zero entries make any sense.
	    // We use a thresholding to set the coordinates in turn from the largest magnitude.
	    // The number 3 in the "simplex" array is at the position of the largest coordinate.

	    // TODO: get rid of this lookup, try it with pure conditionals,
	    // TODO: This should not be required, backport it from Bill's GLSL code!
	    i1 = fast_simplex[c][0]>=3 ? 1 : 0;
	    j1 = fast_simplex[c][1]>=3 ? 1 : 0;
	    k1 = fast_simplex[c][2]>=3 ? 1 : 0;
	    l1 = fast_simplex[c][3]>=3 ? 1 : 0;
	    // The number 2 in the "simplex" array is at the second largest coordinate.
	    i2 = fast_simplex[c][0]>=2 ? 1 : 0;
	    j2 = fast_simplex[c][1]>=2 ? 1 : 0;
	    k2 = fast_simplex[c][2]>=2 ? 1 : 0;
	    l2 = fast_simplex[c][3]>=2 ? 1 : 0;
	    // The number 1 in the "simplex" array is at the second smallest coordinate.
	    i3 = fast_simplex[c][0]>=1 ? 1 : 0;
	    j3 = fast_simplex[c][1]>=1 ? 1 : 0;
	    k3 = fast_simplex[c][2]>=1 ? 1 : 0;
	    l3 = fast_simplex[c][3]>=1 ? 1 : 0;
	    // The fifth corner has all coordinate offsets = 1, so no need to look that up.

	    float x1 = x0 - i1 + G4; // Offsets for second corner in (x,y,z,w) coords
	    float y1 = y0 - j1 + G4;
	    float z1 = z0 - k1 + G4;
	    float w1 = w0 - l1 + G4;
	    float x2 = x0 - i2 + 2.0f * G4; // Offsets for third corner in (x,y,z,w) coords
	    float y2 = y0 - j2 + 2.0f * G4;
	    float z2 = z0 - k2 + 2.0f * G4;
	    float w2 = w0 - l2 + 2.0f * G4;
	    float x3 = x0 - i3 + 3.0f * G4; // Offsets for fourth corner in (x,y,z,w) coords
	    float y3 = y0 - j3 + 3.0f * G4;
	    float z3 = z0 - k3 + 3.0f * G4;
	    float w3 = w0 - l3 + 3.0f * G4;
	    float x4 = x0 - 1.0f + 4.0f * G4; // Offsets for last corner in (x,y,z,w) coords
	    float y4 = y0 - 1.0f + 4.0f * G4;
	    float z4 = z0 - 1.0f + 4.0f * G4;
	    float w4 = w0 - 1.0f + 4.0f * G4;

	    // As we do not expect any coherency between data lanes
	    // Hoisted work out of conditionals to encourage masking blending
    	// versus a check for coherency
	    // In other words we will do all the work, all the time versus
	    // trying to manage it on a per lane basis.
	    // NOTE: this may be slower if used for serial vs. simd
	    Vec4 g0 = grad4<seedT>(i, j, k, l);
	    Vec4 g1 = grad4<seedT>(i+i1, j+j1, k+k1, l+l1);
	    Vec4 g2 = grad4<seedT>(i+i2, j+j2, k+k2, l+l2);
	    Vec4 g3 = grad4<seedT>(i+i3, j+j3, k+k3, l+l3);
	    Vec4 g4 = grad4<seedT>(i+1, j+1, k+1, l+1);

	    // Calculate the contribution from the five corners
	    float t0 = 0.5f - x0*x0 - y0*y0 - z0*z0 - w0*w0;
        // NOTE: avoid array access of points, always use
        // the real data members to avoid aliasing issues

        float t20 = t0 * t0;
        float t40 = t20 * t20;
        float tn0 = t40 * (g0.x * x0 + g0.y * y0 + g0.z * z0 + g0.w * w0);
	    float n0 = (t0 >= 0.0f) ? tn0 : 0.0f;

	    float t1 = 0.5f - x1*x1 - y1*y1 - z1*z1 - w1*w1;
    	float t21 = t1 * t1;
    	float t41 = t21 * t21;
		float tn1 = t41 * (g1.x * x1 + g1.y * y1 + g1.z * z1 + g1.w * w1);
	    float n1 = (t1 >= 0.0f) ? tn1 : 0.0f;

	    float t2 = 0.5f - x2*x2 - y2*y2 - z2*z2 - w2*w2;
    	float t22 = t2 * t2;
    	float t42 = t22 * t22;
        float tn2 = t42 * (g2.x * x2 + g2.y * y2 + g2.z * z2 + g2.w * w2);
	    float n2 = (t2 >= 0.0f) ? tn2 : 0.0f;

	    float t3 = 0.5f - x3*x3 - y3*y3 - z3*z3 - w3*w3;
    	float t23 = t3 * t3;
    	float t43 = t23 * t23;
        float tn3 = t43 * (g3.x * x3 + g3.y * y3 + g3.z * z3 + g3.w * w3);
	    float n3 = (t3 >= 0.0f) ? tn3 : 0.0f;

	    float t4 = 0.5f - x4*x4 - y4*y4 - z4*z4 - w4*w4;
    	float t24 = t4 * t4;
    	float t44 = t24 * t24;
        float tn4 = t44 * (g4.x * x4 + g4.y * y4 + g4.z * z4 + g4.w * w4);
	    float n4 = (t4 >= 0.0f) ? tn4 : 0.0f;

	    // Sum up and scale the result.  The scale is empirical, to make it
	    // cover [-1,1], and to make it approximately match the range of our
	    // Perlin noise implementation.
	    const float scale = 54.0f;
	    float noise = scale * (n0 + n1 + n2 + n3 + n4);

	    return noise;
	}

};

} // anonymous namespace


template<int WidthT>
void
fast_simplexnoise3(WideAccessor<float, WidthT> wresult, ConstWideAccessor<Vec3, WidthT> wp)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
#ifndef OSL_VERIFY_SIMPLEX3  			
		// Workaround clang omp when it cant perform a runtime pointer check
        // to ensure no overlap in output variables
        // But we can tell clang to assume its safe
		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(WidthT))
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(WidthT))
#endif
		for(int i=0; i< WidthT; ++i) {
			Vec3 p = wp[i];

			//float result = simplexnoise3 (p.x, p.y, p.z);
			float result = fast::simplexnoise3<0/* seed */>(p.x, p.y, p.z);
			wresult[i] = result;
		}
	}
}
    
template<int WidthT>
void
fast_simplexnoise3(WideAccessor<Vec3, WidthT> wresult, ConstWideAccessor<Vec3, WidthT> wp)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
#ifndef OSL_VERIFY_SIMPLEX3  			
		// Workaround clang omp when it cant perform a runtime pointer check
        // to ensure no overlap in output variables
        // But we can tell clang to assume its safe
		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(WidthT))
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(WidthT))
#endif
		for(int i=0; i< WidthT; ++i) {
			Vec3 p = wp[i];

			//float result = simplexnoise3 (p.x, p.y, p.z);
			Vec3 result;
			result.x = fast::simplexnoise3<0/* seed */>(p.x, p.y, p.z);
			result.y = fast::simplexnoise3<1/* seed */>(p.x, p.y, p.z);
			result.z = fast::simplexnoise3<2/* seed */>(p.x, p.y, p.z);
			wresult[i] = result;
		}
	}
}


template<int WidthT>
void
fast_simplexnoise4(WideAccessor<Vec3, WidthT> wresult,
                        ConstWideAccessor<Vec3, WidthT> wp,
                        ConstWideAccessor<float,WidthT> wt)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
#ifndef OSL_VERIFY_SIMPLEX3
		// Workaround clang omp when it cant perform a runtime pointer check
        // to ensure no overlap in output variables
        // But we can tell clang to assume its safe
		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(WidthT))
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(WidthT))
#endif
		for(int i=0; i< WidthT; ++i) {
			Vec3 p = wp[i];
			float t = wt[i];

			Vec3 result;
			result.x = fast::simplexnoise4<0/* seed */>(p.x, p.y, p.z, t);
			result.y = fast::simplexnoise4<1/* seed */>(p.x, p.y, p.z, t);
			result.z = fast::simplexnoise4<2/* seed */>(p.x, p.y, p.z, t);
			wresult[i] = result;
		}
	}
}
    


// USimplex
template<int WidthT>
void
fast_usimplexnoise1(WideAccessor<Vec3, WidthT> wresult, ConstWideAccessor<float, WidthT> wx)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		// Workaround clang omp when it cant perform a runtime pointer check
        // to ensure no overlap in output variables
        // But we can tell clang to assume its safe
		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(WidthT))
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(WidthT))

		for(int i=0; i< WidthT; ++i) {
			float x = wx[i];

			Vec3 result;
			result.x = 0.5f * (fast::simplexnoise1<0/* seed */>(x) + 1.0f);
			result.y = 0.5f * (fast::simplexnoise1<1/* seed */>(x) + 1.0f);
			result.z = 0.5f * (fast::simplexnoise1<2/* seed */>(x) + 1.0f);
			wresult[i] = result;
		}
	}
}


template<int WidthT>
void
fast_usimplexnoise3(WideAccessor<float, WidthT> wresult, ConstWideAccessor<Vec3, WidthT> wp)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
#ifndef OSL_VERIFY_SIMPLEX3  
		// Workaround clang omp when it cant perform a runtime pointer check
        // to ensure no overlap in output variables
        // But we can tell clang to assume its safe
		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(WidthT))
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(WidthT))
#endif
		for(int i=0; i< WidthT; ++i) {
			Vec3 p = wp[i];

			float result = 0.5f * (fast::simplexnoise3<0/* seed */>(p.x, p.y, p.z) + 1.0f);

			wresult[i] = result;
		}
	}
}
    
    
    
template<int WidthT>
void
fast_usimplexnoise3(WideAccessor<Vec3, WidthT> wresult, ConstWideAccessor<Vec3, WidthT>  wp)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
#ifndef OSL_VERIFY_SIMPLEX3  
		// Workaround clang omp when it cant perform a runtime pointer check
        // to ensure no overlap in output variables
        // But we can tell clang to assume its safe
		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(WidthT))
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(WidthT))
#endif
		for(int i=0; i< WidthT; ++i) {
			Vec3 p = wp[i];

			Vec3 result;
			result.x = 0.5f * (fast::simplexnoise3<0/* seed */>(p.x, p.y, p.z) + 1.0f);
			result.y = 0.5f * (fast::simplexnoise3<1/* seed */>(p.x, p.y, p.z) + 1.0f);
			result.z = 0.5f * (fast::simplexnoise3<2/* seed */>(p.x, p.y, p.z) + 1.0f);

			wresult[i] = result;
		}
	}
}


template<int WidthT>
void
fast_usimplexnoise4 (WideAccessor<Vec3, WidthT> wresult,
						ConstWideAccessor<Vec3, WidthT> wp,
						ConstWideAccessor<float,WidthT> wt)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
#ifndef OSL_VERIFY_SIMPLEX3
		// Workaround clang omp when it cant perform a runtime pointer check
        // to ensure no overlap in output variables
        // But we can tell clang to assume its safe
		OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(WidthT))
		OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(WidthT))
#endif
		for(int i=0; i< WidthT; ++i) {
			Vec3 p = wp[i];
			float t = wt[i];

			Vec3 result;
			result.x = 0.5f * (fast::simplexnoise4<0/* seed */>(p.x, p.y, p.z, t) + 1.0f);
			result.y = 0.5f * (fast::simplexnoise4<1/* seed */>(p.x, p.y, p.z, t) + 1.0f);
			result.z = 0.5f * (fast::simplexnoise4<2/* seed */>(p.x, p.y, p.z, t) + 1.0f);
			wresult[i] = result;
		}
	}
}




}; // namespace pvt


OSL_NAMESPACE_EXIT
