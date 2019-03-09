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
#include <OpenImageIO/fmath.h>
#include <OpenImageIO/hash.h>
#include <OpenImageIO/simd.h>

OSL_NAMESPACE_ENTER


namespace oslnoise {

/////////////////////////////////////////////////////////////////////////
//
// Simple public API for computing the same noise functions that you get
// from OSL shaders.
//
// These match the properties of OSL noises that have the same names,
// please see the OSL specification for more detailed descriptions, we
// won't recapitulate it here.
//
// For the sake of compactness, we express the noise functions as
// templates for a either one or two domain parameters, which may be:
//     (float)                 // 1-D domain noise, 1-argument variety
//     (float, float)          // 2-D domain noise, 2-argument variety
//     (const Vec3 &)          // 3-D domain noise, 1-argument variety
//     (const Vec3 &, float)   // 4-D domain noise, 2-argument variety
// And the range type may be
//     float noisename ()      // float-valued noise
//     Vec3  vnoisename()      // vector-valued noise
// Note that in OSL we can overload function calls by return type, but we
// can't in C++, so we prepend a "v" in front of the names of functions
// that return vector-valued noise.
//
/////////////////////////////////////////////////////////////////////////

// Signed Perlin-like noise on 1-4 dimensional domain, range [-1,1].
template <typename S >             OSL_HOSTDEVICE float snoise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE float snoise (S x, T y);
template <typename S >             OSL_HOSTDEVICE Vec3  vsnoise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE Vec3  vsnoise (S x, T y);

// Unsigned Perlin-like noise on 1-4 dimensional domain, range [0,1].
template <typename S >             OSL_HOSTDEVICE float noise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE float noise (S x, T y);
template <typename S >             OSL_HOSTDEVICE Vec3  vnoise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE Vec3  vnoise (S x, T y);

// Cell noise on 1-4 dimensional domain, range [0,1].
// cellnoise is constant within each unit cube (cell) on the domain, but
// discontinuous at integer boundaries (and uncorrellated from cell to
// cell).
template <typename S >             OSL_HOSTDEVICE float cellnoise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE float cellnoise (S x, T y);
template <typename S >             OSL_HOSTDEVICE Vec3  vcellnoise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE Vec3  vcellnoise (S x, T y);

// Hash noise on 1-4 dimensional domain, range [0,1].
// hashnoise is like cellnoise, but without the 'floor' -- in other words,
// it's an uncorrellated hash that is different for every floating point
// value.
template <typename S >             OSL_HOSTDEVICE float hashnoise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE float hashnoise (S x, T y);
template <typename S >             OSL_HOSTDEVICE Vec3  vhashnoise (S x);
template <typename S, typename T>  OSL_HOSTDEVICE Vec3  vhashnoise (S x, T y);

// FIXME -- eventually consider adding to the public API:
//  * periodic varieties
//  * varieties with derivatives
//  * varieties that take/return simd::float3 rather than Imath::Vec3f.
//  * exposing the simplex & gabor varieties


}   // namespace oslnoise



///////////////////////////////////////////////////////////////////////
// Implementation follows...
//
// Users don't need to worry about this part
///////////////////////////////////////////////////////////////////////

struct NoiseParams;

namespace pvt {
using namespace OIIO::simd;
using namespace OIIO::bjhash;

typedef void (*NoiseGenericFunc)(int outdim, float *out, bool derivs,
                                 int indim, const float *in,
                                 const float *period, NoiseParams *params);
typedef void (*NoiseImplFunc)(float *out, const float *in,
                              const float *period, NoiseParams *params);

OSLNOISEPUBLIC OSL_HOSTDEVICE
float simplexnoise1 (float x, int seed=0, float *dnoise_dx=NULL);

OSLNOISEPUBLIC OSL_HOSTDEVICE
float simplexnoise2 (float x, float y, int seed=0,
                     float *dnoise_dx=NULL, float *dnoise_dy=NULL);

OSLNOISEPUBLIC OSL_HOSTDEVICE
float simplexnoise3 (float x, float y, float z, int seed=0,
                     float *dnoise_dx=NULL, float *dnoise_dy=NULL,
                     float *dnoise_dz=NULL);

OSLNOISEPUBLIC OSL_HOSTDEVICE
float simplexnoise4 (float x, float y, float z, float w, int seed=0,
                     float *dnoise_dx=NULL, float *dnoise_dy=NULL,
                     float *dnoise_dz=NULL, float *dnoise_dw=NULL);


namespace {

// convert a 32 bit integer into a floating point number in [0,1]
inline OSL_HOSTDEVICE float bits_to_01 (unsigned int bits) {
    // divide by 2^32-1
    return bits * (1.0f / std::numeric_limits<unsigned int>::max());
}


#ifndef __CUDA_ARCH__
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
#endif


/// hash an array of N 32 bit values into a pseudo-random value
/// based on my favorite hash: http://burtleburtle.net/bob/c/lookup3.c
/// templated so that the compiler can unroll the loops for us
template <int N> OSL_HOSTDEVICE
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


#ifndef __CUDA_ARCH__
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
#endif



struct CellNoise {
    OSL_HOSTDEVICE CellNoise () { }

    inline OSL_HOSTDEVICE void operator() (float &result, float x) const {
        unsigned int iv[1];
        iv[0] = OIIO::ifloor (x);
        hash1<1> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float y) const {
        unsigned int iv[2];
        iv[0] = OIIO::ifloor (x);
        iv[1] = OIIO::ifloor (y);
        hash1<2> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p) const {
        unsigned int iv[3];
        iv[0] = OIIO::ifloor (p.x);
        iv[1] = OIIO::ifloor (p.y);
        iv[2] = OIIO::ifloor (p.z);
        hash1<3> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t) const {
        unsigned int iv[4];
        iv[0] = OIIO::ifloor (p.x);
        iv[1] = OIIO::ifloor (p.y);
        iv[2] = OIIO::ifloor (p.z);
        iv[3] = OIIO::ifloor (t);
        hash1<4> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x) const {
        unsigned int iv[2];
        iv[0] = OIIO::ifloor (x);
        hash3<2> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y) const {
        unsigned int iv[3];
        iv[0] = OIIO::ifloor (x);
        iv[1] = OIIO::ifloor (y);
        hash3<3> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p) const {
        unsigned int iv[4];
        iv[0] = OIIO::ifloor (p.x);
        iv[1] = OIIO::ifloor (p.y);
        iv[2] = OIIO::ifloor (p.z);
        hash3<4> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t) const {
        unsigned int iv[5];
        iv[0] = OIIO::ifloor (p.x);
        iv[1] = OIIO::ifloor (p.y);
        iv[2] = OIIO::ifloor (p.z);
        iv[3] = OIIO::ifloor (t);
        hash3<5> (result, iv);
    }

private:
    template <int N> OSL_HOSTDEVICE
    inline void hash1 (float &result, const unsigned int k[N]) const {
        result = bits_to_01(inthash<N>(k));
    }

    template <int N> OSL_HOSTDEVICE
    inline void hash3 (Vec3 &result, unsigned int k[N]) const {
        k[N-1] = 0; result.x = bits_to_01 (inthash<N> (k));
        k[N-1] = 1; result.y = bits_to_01 (inthash<N> (k));
        k[N-1] = 2; result.z = bits_to_01 (inthash<N> (k));
    }
};



struct PeriodicCellNoise {
    OSL_HOSTDEVICE PeriodicCellNoise () { }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float px) const {
        unsigned int iv[1];
        iv[0] = OIIO::ifloor (wrap (x, px));
        hash1<1> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float y,
                                           float px, float py) const {
        unsigned int iv[2];
        iv[0] = OIIO::ifloor (wrap (x, px));
        iv[1] = OIIO::ifloor (wrap (y, py));
        hash1<2> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p,
                                           const Vec3 &pp) const {
        unsigned int iv[3];
        iv[0] = OIIO::ifloor (wrap (p.x, pp.x));
        iv[1] = OIIO::ifloor (wrap (p.y, pp.y));
        iv[2] = OIIO::ifloor (wrap (p.z, pp.z));
        hash1<3> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t,
                            const Vec3 &pp, float tt) const {
        unsigned int iv[4];
        iv[0] = OIIO::ifloor (wrap (p.x, pp.x));
        iv[1] = OIIO::ifloor (wrap (p.y, pp.y));
        iv[2] = OIIO::ifloor (wrap (p.z, pp.z));
        iv[3] = OIIO::ifloor (wrap (t, tt));
        hash1<4> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float px) const {
        unsigned int iv[2];
        iv[0] = OIIO::ifloor (wrap (x, px));
        hash3<2> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y,
                                           float px, float py) const {
        unsigned int iv[3];
        iv[0] = OIIO::ifloor (wrap (x, px));
        iv[1] = OIIO::ifloor (wrap (y, py));
        hash3<3> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, const Vec3 &pp) const {
        unsigned int iv[4];
        iv[0] = OIIO::ifloor (wrap (p.x, pp.x));
        iv[1] = OIIO::ifloor (wrap (p.y, pp.y));
        iv[2] = OIIO::ifloor (wrap (p.z, pp.z));
        hash3<4> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t,
                                           const Vec3 &pp, float tt) const {
        unsigned int iv[5];
        iv[0] = OIIO::ifloor (wrap (p.x, pp.x));
        iv[1] = OIIO::ifloor (wrap (p.y, pp.y));
        iv[2] = OIIO::ifloor (wrap (p.z, pp.z));
        iv[3] = OIIO::ifloor (wrap (t, tt));
        hash3<5> (result, iv);
    }

private:
    template <int N> OSL_HOSTDEVICE
    inline void hash1 (float &result, const unsigned int k[N]) const {
        result = bits_to_01(inthash<N>(k));
    }

    template <int N> OSL_HOSTDEVICE
    inline void hash3 (Vec3 &result, unsigned int k[N]) const {
        k[N-1] = 0; result.x = bits_to_01 (inthash<N> (k));
        k[N-1] = 1; result.y = bits_to_01 (inthash<N> (k));
        k[N-1] = 2; result.z = bits_to_01 (inthash<N> (k));
    }

    inline OSL_HOSTDEVICE float wrap (float s, float period) const {
        period = floorf (period);
        if (period < 1.0f)
            period = 1.0f;
        return s - period * floorf (s / period);
    }

    inline OSL_HOSTDEVICE Vec3 wrap (const Vec3 &s, const Vec3 &period) {
        return Vec3 (wrap (s[0], period[0]),
                     wrap (s[1], period[1]),
                     wrap (s[2], period[2]));
    }
};



inline OSL_HOSTDEVICE int
inthashi (int x)
{
    unsigned int i[1];
    i[0] = (unsigned int)x;
    return (int) inthash<1>(i);
}

inline OSL_HOSTDEVICE int
inthashf (float x)
{
    unsigned int i[1];
    i[0] = OIIO::bit_cast<float,unsigned int>(x);
    return (int) inthash<1>(i);
}

inline OSL_HOSTDEVICE int
inthashf (float x, float y)
{
    unsigned int i[2];
    i[0] = OIIO::bit_cast<float,unsigned int>(x);
    i[1] = OIIO::bit_cast<float,unsigned int>(y);
    return (int) inthash<2>(i);
}


inline OSL_HOSTDEVICE int
inthashf (const float *x)
{
    unsigned int i[3];
    i[0] = OIIO::bit_cast<float,unsigned int>(x[0]);
    i[1] = OIIO::bit_cast<float,unsigned int>(x[1]);
    i[2] = OIIO::bit_cast<float,unsigned int>(x[2]);
    return (int) inthash<3>(i);
}


inline OSL_HOSTDEVICE int
inthashf (const float *x, float y)
{
    unsigned int i[4];
    i[0] = OIIO::bit_cast<float,unsigned int>(x[0]);
    i[1] = OIIO::bit_cast<float,unsigned int>(x[1]);
    i[2] = OIIO::bit_cast<float,unsigned int>(x[2]);
    i[3] = OIIO::bit_cast<float,unsigned int>(y);
    return (int) inthash<4>(i);
}



struct HashNoise {
    OSL_HOSTDEVICE HashNoise () { }

    inline OSL_HOSTDEVICE void operator() (float &result, float x) const {
        unsigned int iv[1];
        iv[0] = OIIO::bit_cast<float,unsigned int> (x);
        hash1<1> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float y) const {
        unsigned int iv[2];
        iv[0] = OIIO::bit_cast<float,unsigned int> (x);
        iv[1] = OIIO::bit_cast<float,unsigned int> (y);
        hash1<2> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p) const {
        unsigned int iv[3];
        iv[0] = OIIO::bit_cast<float,unsigned int> (p.x);
        iv[1] = OIIO::bit_cast<float,unsigned int> (p.y);
        iv[2] = OIIO::bit_cast<float,unsigned int> (p.z);
        hash1<3> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t) const {
        unsigned int iv[4];
        iv[0] = OIIO::bit_cast<float,unsigned int> (p.x);
        iv[1] = OIIO::bit_cast<float,unsigned int> (p.y);
        iv[2] = OIIO::bit_cast<float,unsigned int> (p.z);
        iv[3] = OIIO::bit_cast<float,unsigned int> (t);
        hash1<4> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x) const {
        unsigned int iv[2];
        iv[0] = OIIO::bit_cast<float,unsigned int> (x);
        hash3<2> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y) const {
        unsigned int iv[3];
        iv[0] = OIIO::bit_cast<float,unsigned int> (x);
        iv[1] = OIIO::bit_cast<float,unsigned int> (y);
        hash3<3> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p) const {
        unsigned int iv[4];
        iv[0] = OIIO::bit_cast<float,unsigned int> (p.x);
        iv[1] = OIIO::bit_cast<float,unsigned int> (p.y);
        iv[2] = OIIO::bit_cast<float,unsigned int> (p.z);
        hash3<4> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t) const {
        unsigned int iv[5];
        iv[0] = OIIO::bit_cast<float,unsigned int> (p.x);
        iv[1] = OIIO::bit_cast<float,unsigned int> (p.y);
        iv[2] = OIIO::bit_cast<float,unsigned int> (p.z);
        iv[3] = OIIO::bit_cast<float,unsigned int> (t);
        hash3<5> (result, iv);
    }

private:
    template <int N> OSL_HOSTDEVICE
    inline void hash1 (float &result, const unsigned int k[N]) const {
        result = bits_to_01(inthash<N>(k));
    }

    template <int N> OSL_HOSTDEVICE
    inline void hash3 (Vec3 &result, unsigned int k[N]) const {
        k[N-1] = 0; result.x = bits_to_01 (inthash<N> (k));
        k[N-1] = 1; result.y = bits_to_01 (inthash<N> (k));
        k[N-1] = 2; result.z = bits_to_01 (inthash<N> (k));
    }
};



struct PeriodicHashNoise {
    OSL_HOSTDEVICE PeriodicHashNoise () { }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float px) const {
        unsigned int iv[1];
        iv[0] = OIIO::bit_cast<float,unsigned int> (wrap (x, px));
        hash1<1> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float y,
                                           float px, float py) const {
        unsigned int iv[2];
        iv[0] = OIIO::bit_cast<float,unsigned int> (wrap (x, px));
        iv[1] = OIIO::bit_cast<float,unsigned int> (wrap (y, py));
        hash1<2> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p,
                                           const Vec3 &pp) const {
        unsigned int iv[3];
        iv[0] = OIIO::bit_cast<float,unsigned int> (wrap (p.x, pp.x));
        iv[1] = OIIO::bit_cast<float,unsigned int> (wrap (p.y, pp.y));
        iv[2] = OIIO::bit_cast<float,unsigned int> (wrap (p.z, pp.z));
        hash1<3> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t,
                                           const Vec3 &pp, float tt) const {
        unsigned int iv[4];
        iv[0] = OIIO::bit_cast<float,unsigned int> (wrap (p.x, pp.x));
        iv[1] = OIIO::bit_cast<float,unsigned int> (wrap (p.y, pp.y));
        iv[2] = OIIO::bit_cast<float,unsigned int> (wrap (p.z, pp.z));
        iv[3] = OIIO::bit_cast<float,unsigned int> (wrap (t, tt));
        hash1<4> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float px) const {
        unsigned int iv[2];
        iv[0] = OIIO::bit_cast<float,unsigned int> (wrap (x, px));
        hash3<2> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y,
                                           float px, float py) const {
        unsigned int iv[3];
        iv[0] = OIIO::bit_cast<float,unsigned int> (wrap (x, px));
        iv[1] = OIIO::bit_cast<float,unsigned int> (wrap (y, py));
        hash3<3> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, const Vec3 &pp) const {
        unsigned int iv[4];
        iv[0] = OIIO::bit_cast<float,unsigned int> (wrap (p.x, pp.x));
        iv[1] = OIIO::bit_cast<float,unsigned int> (wrap (p.y, pp.y));
        iv[2] = OIIO::bit_cast<float,unsigned int> (wrap (p.z, pp.z));
        hash3<4> (result, iv);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t,
                                           const Vec3 &pp, float tt) const {
        unsigned int iv[5];
        iv[0] = OIIO::bit_cast<float,unsigned int> (wrap (p.x, pp.x));
        iv[1] = OIIO::bit_cast<float,unsigned int> (wrap (p.y, pp.y));
        iv[2] = OIIO::bit_cast<float,unsigned int> (wrap (p.z, pp.z));
        iv[3] = OIIO::bit_cast<float,unsigned int> (wrap (t, tt));
        hash3<5> (result, iv);
    }

private:
    template <int N> OSL_HOSTDEVICE
    inline void hash1 (float &result, const unsigned int k[N]) const {
        result = bits_to_01(inthash<N>(k));
    }

    template <int N> OSL_HOSTDEVICE
    inline void hash3 (Vec3 &result, unsigned int k[N]) const {
        k[N-1] = 0; result.x = bits_to_01 (inthash<N> (k));
        k[N-1] = 1; result.y = bits_to_01 (inthash<N> (k));
        k[N-1] = 2; result.z = bits_to_01 (inthash<N> (k));
    }

    inline OSL_HOSTDEVICE float wrap (float s, float period) const {
        period = floorf (period);
        if (period < 1.0f)
            period = 1.0f;
        return s - period * floorf (s / period);
    }

    inline OSL_HOSTDEVICE Vec3 wrap (const Vec3 &s, const Vec3 &period) {
        return Vec3 (wrap (s[0], period[0]),
                     wrap (s[1], period[1]),
                     wrap (s[2], period[2]));
    }
};




// Define select(bool,truevalue,falsevalue) template that works for a
// variety of types that we can use for both scalars and vectors. Because ?:
// won't work properly in template code with vector ops.
template <typename B, typename F> OSL_HOSTDEVICE
OIIO_FORCEINLINE F select (const B& b, const F& t, const F& f) { return b ? t : f; }

#ifndef __CUDA_ARCH__
template <> OIIO_FORCEINLINE int4 select (const bool4& b, const int4& t, const int4& f) {
    return blend (f, t, b);
}

template <> OIIO_FORCEINLINE float4 select (const bool4& b, const float4& t, const float4& f) {
    return blend (f, t, b);
}

template <> OIIO_FORCEINLINE float4 select (const int4& b, const float4& t, const float4& f) {
    return blend (f, t, bool4(b));
}

template <> OIIO_FORCEINLINE Dual2<float4>
select (const bool4& b, const Dual2<float4>& t, const Dual2<float4>& f) {
    return Dual2<float4> (blend (f.val(), t.val(), b),
                          blend (f.dx(),  t.dx(),  b),
                          blend (f.dy(),  t.dy(),  b));
}

template <>
OIIO_FORCEINLINE Dual2<float4> select (const int4& b, const Dual2<float4>& t, const Dual2<float4>& f) {
    return select (bool4(b), t, f);
}
#endif



// Define negate_if(value,bool) that will work for both scalars and vectors,
// as well as Dual2's of both.
template<typename FLOAT, typename BOOL> OSL_HOSTDEVICE
OIIO_FORCEINLINE FLOAT negate_if (const FLOAT& val, const BOOL& b) {
    return b ? -val : val;
}

#ifndef __CUDA_ARCH__
template<> OIIO_FORCEINLINE float4 negate_if (const float4& val, const int4& b) {
    // Special case negate_if for SIMD -- can do it with bit tricks, no branches
    int4 highbit (0x80000000);
    return bitcast_to_float4 (bitcast_to_int4(val) ^ (blend0 (highbit, bool4(b))));
}

// Special case negate_if for SIMD -- can do it with bit tricks, no branches
template<> OIIO_FORCEINLINE Dual2<float4> negate_if (const Dual2<float4>& val, const int4& b)
{
    return Dual2<float4> (negate_if (val.val(), b),
                          negate_if (val.dx(),  b),
                          negate_if (val.dy(),  b));
}
#endif


#ifndef __CUDA_ARCH__
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
#endif



// Equivalent to OIIO::bilerp (a, b, c, d, u, v), but if abcd are already
// packed into a float4. We assume T is float and VECTYPE is float4,
// but it also works if T is Dual2<float> and VECTYPE is Dual2<float4>.
template<typename T, typename VECTYPE> OSL_HOSTDEVICE
OIIO_FORCEINLINE T bilerp (VECTYPE abcd, T u, T v) {
    VECTYPE xx = OIIO::lerp (abcd, OIIO::simd::shuffle<1,1,3,3>(abcd), u);
    return OIIO::simd::extract<0>(OIIO::lerp (xx,OIIO::simd::shuffle<2>(xx), v));
}

#ifndef __CUDA_ARCH__
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
#endif



// always return a value inside [0,b) - even for negative numbers
inline OSL_HOSTDEVICE int imod(int a, int b) {
    a %= b;
    return a < 0 ? a + b : a;
}

#ifndef __CUDA_ARCH__
// imod four values at once
inline int4 imod(const int4& a, int b) {
    int4 c = a % b;
    return c + select(c < 0, int4(b), int4::Zero());
}
#endif

// floorfrac return ifloor as well as the fractional remainder
// FIXME: already implemented inside OIIO but can't easily override it for duals
//        inside a different namespace
inline OSL_HOSTDEVICE float floorfrac(float x, int* i) {
    *i = OIIO::ifloor(x);
    return x - *i;
}

// floorfrac with derivs
inline OSL_HOSTDEVICE Dual2<float> floorfrac(const Dual2<float> &x, int* i) {
    float frac = floorfrac(x.val(), i);
    // slope of x is not affected by this operation
    return Dual2<float>(frac, x.dx(), x.dy());
}

#ifndef __CUDA_ARCH__
// floatfrac for four sets of values at once.
inline float4 floorfrac(const float4& x, int4 * i) {
#if 0
    float4 thefloor = floor(x);
    *i = int4(thefloor);
    return x-thefloor;
#else
    int4 thefloor = OIIO::simd::ifloor (x);
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
#endif


// Perlin 'fade' function. Can be overloaded for float, Dual2, as well
// as float4 / Dual2<float4>.
template <typename T> OSL_HOSTDEVICE
inline T fade (const T &t) {
   return t * t * t * (t * (t * T(6.0f) - T(15.0f)) + T(10.0f));
}



// 1,2,3 and 4 dimensional gradient functions - perform a dot product against a
// randomly chosen vector. Note that the gradient vector is not normalized, but
// this only affects the overal "scale" of the result, so we simply account for
// the scale by multiplying in the corresponding "perlin" function.
// These factors were experimentally calculated to be:
//    1D:   0.188
//    2D:   0.507
//    3D:   0.936
//    4D:   0.870

template <typename T> OSL_HOSTDEVICE
inline T grad (int hash, const T &x) {
    int h = hash & 15;
    float g = 1 + (h & 7);  // 1, 2, .., 8
    if (h&8) g = -g;        // random sign
    return g * x;           // dot-product
}

template <typename I, typename T> OSL_HOSTDEVICE
inline T grad (const I &hash, const T &x, const T &y) {
    // 8 possible directions (+-1,+-2) and (+-2,+-1)
    I h = hash & 7;
    T u = select (h<4, x, y);
    T v = 2.0f * select (h<4, y, x);
    // compute the dot product with (x,y).
    return negate_if(u, h&1) + negate_if(v, h&2);
}

template <typename I, typename T> OSL_HOSTDEVICE
inline T grad (const I &hash, const T &x, const T &y, const T &z) {
    // use vectors pointing to the edges of the cube
    I h = hash & 15;
    T u = select (h<8, x, y);
    T v = select (h<4, y, select ((h==I(12))|(h==I(14)), x, z));
    return negate_if(u,h&1) + negate_if(v,h&2);
}

template <typename I, typename T> OSL_HOSTDEVICE
inline T grad (const I &hash, const T &x, const T &y, const T &z, const T &w) {
    // use vectors pointing to the edges of the hypercube
    I h = hash & 31;
    T u = select (h<24, x, y);
    T v = select (h<16, y, z);
    T s = select (h<8 , z, w);
    return negate_if(u,h&1) + negate_if(v,h&2) + negate_if(s,h&4);
}

typedef Imath::Vec3<int> Vec3i;

inline OSL_HOSTDEVICE Vec3 grad (const Vec3i &hash, float x) {
    return Vec3 (grad (hash.x, x),
                 grad (hash.y, x),
                 grad (hash.z, x));
}

inline OSL_HOSTDEVICE Dual2<Vec3> grad (const Vec3i &hash, Dual2<float> x) {
    Dual2<float> rx = grad (hash.x, x);
    Dual2<float> ry = grad (hash.y, x);
    Dual2<float> rz = grad (hash.z, x);
    return make_Vec3 (rx, ry, rz);
}


inline OSL_HOSTDEVICE Vec3 grad (const Vec3i &hash, float x, float y) {
    return Vec3 (grad (hash.x, x, y),
                 grad (hash.y, x, y),
                 grad (hash.z, x, y));
}

inline OSL_HOSTDEVICE Dual2<Vec3> grad (const Vec3i &hash, Dual2<float> x, Dual2<float> y) {
    Dual2<float> rx = grad (hash.x, x, y);
    Dual2<float> ry = grad (hash.y, x, y);
    Dual2<float> rz = grad (hash.z, x, y);
    return make_Vec3 (rx, ry, rz);
}

inline OSL_HOSTDEVICE Vec3 grad (const Vec3i &hash, float x, float y, float z) {
    return Vec3 (grad (hash.x, x, y, z),
                 grad (hash.y, x, y, z),
                 grad (hash.z, x, y, z));
}

inline OSL_HOSTDEVICE Dual2<Vec3> grad (const Vec3i &hash, Dual2<float> x, Dual2<float> y, Dual2<float> z) {
    Dual2<float> rx = grad (hash.x, x, y, z);
    Dual2<float> ry = grad (hash.y, x, y, z);
    Dual2<float> rz = grad (hash.z, x, y, z);
    return make_Vec3 (rx, ry, rz);
}

inline OSL_HOSTDEVICE Vec3 grad (const Vec3i &hash, float x, float y, float z, float w) {
    return Vec3 (grad (hash.x, x, y, z, w),
                 grad (hash.y, x, y, z, w),
                 grad (hash.z, x, y, z, w));
}

inline OSL_HOSTDEVICE Dual2<Vec3> grad (const Vec3i &hash, Dual2<float> x, Dual2<float> y, Dual2<float> z, Dual2<float> w) {
    Dual2<float> rx = grad (hash.x, x, y, z, w);
    Dual2<float> ry = grad (hash.y, x, y, z, w);
    Dual2<float> rz = grad (hash.z, x, y, z, w);
    return make_Vec3 (rx, ry, rz);
}

template <typename T> OSL_HOSTDEVICE
inline T scale1 (const T &result) { return 0.2500f * result; }
template <typename T> OSL_HOSTDEVICE
inline T scale2 (const T &result) { return 0.6616f * result; }
template <typename T> OSL_HOSTDEVICE
inline T scale3 (const T &result) { return 0.9820f * result; }
template <typename T> OSL_HOSTDEVICE
inline T scale4 (const T &result) { return 0.8344f * result; }



struct HashScalar {
    OSL_HOSTDEVICE int operator() (int x) const {
        unsigned int iv[1];
        iv[0] = x;
        return inthash<1> (iv);
    }

    OSL_HOSTDEVICE int operator() (int x, int y) const {
        unsigned int iv[2];
        iv[0] = x;
        iv[1] = y;
        return inthash<2> (iv);
    }

    OSL_HOSTDEVICE int operator() (int x, int y, int z) const {
        unsigned int iv[3];
        iv[0] = x;
        iv[1] = y;
        iv[2] = z;
        return inthash<3> (iv);
    }

    OSL_HOSTDEVICE int operator() (int x, int y, int z, int w) const {
        unsigned int iv[4];
        iv[0] = x;
        iv[1] = y;
        iv[2] = z;
        iv[3] = w;
        return inthash<4> (iv);
    }

#ifndef __CUDA_ARCH__
    // 4 2D hashes at once!
    OIIO_FORCEINLINE int4 operator() (const int4& x, const int4& y) const {
        return inthash_simd (x, y);
    }

    // 4 3D hashes at once!
    OIIO_FORCEINLINE int4 operator() (const int4& x, const int4& y, const int4& z) const {
        return inthash_simd (x, y, z);
    }

    // 4 3D hashes at once!
    OIIO_FORCEINLINE int4 operator() (const int4& x, const int4& y, const int4& z, const int4& w) const {
        return inthash_simd (x, y, z, w);
    }
#endif

};

struct HashVector {
    OSL_HOSTDEVICE Vec3i operator() (int x) const {
        unsigned int iv[1];
        iv[0] = x;
        return hash3<1> (iv);
    }

    OSL_HOSTDEVICE Vec3i operator() (int x, int y) const {
        unsigned int iv[2];
        iv[0] = x;
        iv[1] = y;
        return hash3<2> (iv);
    }

    OSL_HOSTDEVICE Vec3i operator() (int x, int y, int z) const {
        unsigned int iv[3];
        iv[0] = x;
        iv[1] = y;
        iv[2] = z;
        return hash3<3> (iv);
    }

    OSL_HOSTDEVICE Vec3i operator() (int x, int y, int z, int w) const {
        unsigned int iv[4];
        iv[0] = x;
        iv[1] = y;
        iv[2] = z;
        iv[3] = w;
        return hash3<4> (iv);
    }

    template <int N> OSL_HOSTDEVICE
    Vec3i hash3 (unsigned int k[N]) const {
        Vec3i result;
        unsigned int h = inthash<N> (k);
        // we only need the low-order bits to be random, so split out
        // the 32 bit result into 3 parts for each channel
        result.x = (h      ) & 0xFF;
        result.y = (h >> 8 ) & 0xFF;
        result.z = (h >> 16) & 0xFF;
        return result;
    }

#ifndef __CUDA_ARCH__
    // Vector hash of 4 3D points at once
    OIIO_FORCEINLINE void operator() (int4 *result, const int4& x, const int4& y) const {
        int4 h = inthash_simd (x, y);
        result[0] = (h        ) & 0xFF;
        result[1] = (srl(h,8 )) & 0xFF;
        result[2] = (srl(h,16)) & 0xFF;
    }

    // Vector hash of 4 3D points at once
    OIIO_FORCEINLINE void operator() (int4 *result, const int4& x, const int4& y, const int4& z) const {
        int4 h = inthash_simd (x, y, z);
        result[0] = (h        ) & 0xFF;
        result[1] = (srl(h,8 )) & 0xFF;
        result[2] = (srl(h,16)) & 0xFF;
    }

    // Vector hash of 4 3D points at once
    OIIO_FORCEINLINE void operator() (int4 *result, const int4& x, const int4& y, const int4& z, const int4& w) const {
        int4 h = inthash_simd (x, y, z, w);
        result[0] = (h        ) & 0xFF;
        result[1] = (srl(h,8 )) & 0xFF;
        result[2] = (srl(h,16)) & 0xFF;
    }
#endif

};

struct HashScalarPeriodic {
    OSL_HOSTDEVICE HashScalarPeriodic (float px) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
    }
    OSL_HOSTDEVICE HashScalarPeriodic (float px, float py) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
        m_py = OIIO::ifloor(py); if (m_py < 1) m_py = 1;
    }
    OSL_HOSTDEVICE HashScalarPeriodic (float px, float py, float pz) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
        m_py = OIIO::ifloor(py); if (m_py < 1) m_py = 1;
        m_pz = OIIO::ifloor(pz); if (m_pz < 1) m_pz = 1;
    }
    OSL_HOSTDEVICE HashScalarPeriodic (float px, float py, float pz, float pw) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
        m_py = OIIO::ifloor(py); if (m_py < 1) m_py = 1;
        m_pz = OIIO::ifloor(pz); if (m_pz < 1) m_pz = 1;
        m_pw = OIIO::ifloor(pw); if (m_pw < 1) m_pw = 1;
    }

    int m_px, m_py, m_pz, m_pw;

    OSL_HOSTDEVICE int operator() (int x) const {
        unsigned int iv[1];
        iv[0] = imod (x, m_px);
        return inthash<1> (iv);
    }

    OSL_HOSTDEVICE int operator() (int x, int y) const {
        unsigned int iv[2];
        iv[0] = imod (x, m_px);
        iv[1] = imod (y, m_py);
        return inthash<2> (iv);
    }

    OSL_HOSTDEVICE int operator() (int x, int y, int z) const {
        unsigned int iv[3];
        iv[0] = imod (x, m_px);
        iv[1] = imod (y, m_py);
        iv[2] = imod (z, m_pz);
        return inthash<3> (iv);
    }

    OSL_HOSTDEVICE int operator() (int x, int y, int z, int w) const {
        unsigned int iv[4];
        iv[0] = imod (x, m_px);
        iv[1] = imod (y, m_py);
        iv[2] = imod (z, m_pz);
        iv[3] = imod (w, m_pw);
        return inthash<4> (iv);
    }

#ifndef __CUDA_ARCH__
    // 4 2D hashes at once!
    int4 operator() (const int4& x, const int4& y) const {
        return inthash_simd (imod(x,m_px), imod(y,m_py));
    }

    // 4 3D hashes at once!
    int4 operator() (const int4& x, const int4& y, const int4& z) const {
        return inthash_simd (imod(x,m_px), imod(y,m_py), imod(z,m_pz));
    }

    // 4 4D hashes at once
    int4 operator() (const int4& x, const int4& y, const int4& z, const int4& w) const {
        return inthash_simd (imod(x,m_px), imod(y,m_py), imod(z,m_pz), imod(w,m_pw));
    }
#endif

};

struct HashVectorPeriodic {
    OSL_HOSTDEVICE HashVectorPeriodic (float px) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
    }
    OSL_HOSTDEVICE HashVectorPeriodic (float px, float py) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
        m_py = OIIO::ifloor(py); if (m_py < 1) m_py = 1;
    }
    OSL_HOSTDEVICE HashVectorPeriodic (float px, float py, float pz) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
        m_py = OIIO::ifloor(py); if (m_py < 1) m_py = 1;
        m_pz = OIIO::ifloor(pz); if (m_pz < 1) m_pz = 1;
    }
    OSL_HOSTDEVICE HashVectorPeriodic (float px, float py, float pz, float pw) {
        m_px = OIIO::ifloor(px); if (m_px < 1) m_px = 1;
        m_py = OIIO::ifloor(py); if (m_py < 1) m_py = 1;
        m_pz = OIIO::ifloor(pz); if (m_pz < 1) m_pz = 1;
        m_pw = OIIO::ifloor(pw); if (m_pw < 1) m_pw = 1;
    }

    int m_px, m_py, m_pz, m_pw;

    OSL_HOSTDEVICE Vec3i operator() (int x) const {
        unsigned int iv[1];
        iv[0] = imod (x, m_px);
        return hash3<1> (iv);
    }

    OSL_HOSTDEVICE Vec3i operator() (int x, int y) const {
        unsigned int iv[2];
        iv[0] = imod (x, m_px);
        iv[1] = imod (y, m_py);
        return hash3<2> (iv);
    }

    OSL_HOSTDEVICE Vec3i operator() (int x, int y, int z) const {
        unsigned int iv[3];
        iv[0] = imod (x, m_px);
        iv[1] = imod (y, m_py);
        iv[2] = imod (z, m_pz);
        return hash3<3> (iv);

    }

    OSL_HOSTDEVICE Vec3i operator() (int x, int y, int z, int w) const {
        unsigned int iv[4];
        iv[0] = imod (x, m_px);
        iv[1] = imod (y, m_py);
        iv[2] = imod (z, m_pz);
        iv[3] = imod (w, m_pw);
        return hash3<4> (iv);
    }

    template <int N> OSL_HOSTDEVICE
    Vec3i hash3 (unsigned int k[N]) const {
        Vec3i result;
        unsigned int h = inthash<N> (k);
        // we only need the low-order bits to be random, so split out
        // the 32 bit result into 3 parts for each channel
        result.x = (h      ) & 0xFF;
        result.y = (h >> 8 ) & 0xFF;
        result.z = (h >> 16) & 0xFF;
        return result;
    }

#ifndef __CUDA_ARCH__
    // Vector hash of 4 3D points at once
    void operator() (int4 *result, const int4& x, const int4& y) const {
        int4 h = inthash_simd (imod(x,m_px), imod(y,m_py));
        result[0] = (h        ) & 0xFF;
        result[1] = (srl(h,8 )) & 0xFF;
        result[2] = (srl(h,16)) & 0xFF;
    }

    // Vector hash of 4 3D points at once
    void operator() (int4 *result, const int4& x, const int4& y, const int4& z) const {
        int4 h = inthash_simd (imod(x,m_px), imod(y,m_py), imod(z,m_pz));
        result[0] = (h        ) & 0xFF;
        result[1] = (srl(h,8 )) & 0xFF;
        result[2] = (srl(h,16)) & 0xFF;
    }

    // Vector hash of 4 4D points at once
    void operator() (int4 *result, const int4& x, const int4& y, const int4& z, const int4& w) const {
        int4 h = inthash_simd (imod(x,m_px), imod(y,m_py), imod(z,m_pz), imod(w,m_pw));
        result[0] = (h        ) & 0xFF;
        result[1] = (srl(h,8 )) & 0xFF;
        result[2] = (srl(h,16)) & 0xFF;
    }
#endif
};



template <typename V, typename H, typename T> OSL_HOSTDEVICE
inline void perlin (V& result, H& hash, const T &x) {
    int X; T fx = floorfrac(x, &X);
    T u = fade(fx);

    result = OIIO::lerp(grad (hash (X  ), fx     ),
                        grad (hash (X+1), fx-1.0f), u);
    result = scale1 (result);
}


template <typename H> OSL_HOSTDEVICE
inline void perlin (float &result, const H &hash, const float &x, const float &y)
{
    // result = 0.0f; return;
#if OIIO_SIMD
    int4 XY;
    float4 fxy = floorfrac (float4(x,y,0.0f), &XY);
    float4 uv = fade(fxy);  // Note: will be (fade(fx), fade(fy), 0, 0)

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously.
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = OIIO::simd::shuffle<0>(XY) + (*(int4*)i0101);
    int4 cornery = OIIO::simd::shuffle<1>(XY) + (*(int4*)i0011);
    int4 corner_hash = hash (cornerx, cornery);
    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    float4 remainderx = OIIO::simd::shuffle<0>(fxy) - (*(float4*)f0101);
    float4 remaindery = OIIO::simd::shuffle<1>(fxy) - (*(float4*)f0011);
    float4 corner_grad = grad (corner_hash, remainderx, remaindery);
    result = scale2 (bilerp (corner_grad, uv[0], uv[1]));

#else
    // ORIGINAL, non-SIMD
    int X; float fx = floorfrac(x, &X);
    int Y; float fy = floorfrac(y, &Y);

    float u = fade(fx);
    float v = fade(fy);

    result = OIIO::bilerp (grad (hash (X  , Y  ), fx     , fy     ),
                           grad (hash (X+1, Y  ), fx-1.0f, fy     ),
                           grad (hash (X  , Y+1), fx     , fy-1.0f),
                           grad (hash (X+1, Y+1), fx-1.0f, fy-1.0f), u, v);
    result = scale2 (result);
#endif
}


template <typename H> OSL_HOSTDEVICE
inline void perlin (float &result, const H &hash,
                    const float &x, const float &y, const float &z)
{
#if OIIO_SIMD
    // result = 0; return;
#if 0
    // You'd think it would be faster to do the floorfrac in parallel, but
    // according to my timings, it is not. I don't understand exactly why.
    int4 XYZ;
    float4 fxyz = floorfrac (float4(x,y,z), &XYZ);
    float4 uvw = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = shuffle<0>(XYZ) + (*(int4*)i0101);
    int4 cornery = shuffle<1>(XYZ) + (*(int4*)i0011);
    int4 cornerz = shuffle<2>(XYZ);
#else
    int X; float fx = floorfrac(x, &X);
    int Y; float fy = floorfrac(y, &Y);
    int Z; float fz = floorfrac(z, &Z);
    float4 fxyz (fx, fy, fz); // = floorfrac (xyz, &XYZ);
    float4 uvw = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);
    int4 cornerz = Z;
#endif
    int4 corner_hash_z0 = hash (cornerx, cornery, cornerz);
    int4 corner_hash_z1 = hash (cornerx, cornery, cornerz+int4::One());

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    float4 remainderx = OIIO::simd::shuffle<0>(fxyz) - (*(float4*)f0101);
    float4 remaindery = OIIO::simd::shuffle<1>(fxyz) - (*(float4*)f0011);
    float4 remainderz = OIIO::simd::shuffle<2>(fxyz);
    float4 corner_grad_z0 = grad (corner_hash_z0, remainderx, remaindery, remainderz);
    float4 corner_grad_z1 = grad (corner_hash_z1, remainderx, remaindery, remainderz-float4::One());

    result = scale3 (trilerp (corner_grad_z0, corner_grad_z1, uvw));

#else
    // ORIGINAL -- non-SIMD
    int X; float fx = floorfrac(x, &X);
    int Y; float fy = floorfrac(y, &Y);
    int Z; float fz = floorfrac(z, &Z);
    float u = fade(fx);
    float v = fade(fy);
    float w = fade(fz);
    result = OIIO::trilerp (grad (hash (X  , Y  , Z  ), fx     , fy     , fz     ),
                            grad (hash (X+1, Y  , Z  ), fx-1.0f, fy     , fz     ),
                            grad (hash (X  , Y+1, Z  ), fx     , fy-1.0f, fz     ),
                            grad (hash (X+1, Y+1, Z  ), fx-1.0f, fy-1.0f, fz     ),
                            grad (hash (X  , Y  , Z+1), fx     , fy     , fz-1.0f),
                            grad (hash (X+1, Y  , Z+1), fx-1.0f, fy     , fz-1.0f),
                            grad (hash (X  , Y+1, Z+1), fx     , fy-1.0f, fz-1.0f),
                            grad (hash (X+1, Y+1, Z+1), fx-1.0f, fy-1.0f, fz-1.0f),
                            u, v, w);
    result = scale3 (result);
#endif
}



template <typename H> OSL_HOSTDEVICE
inline void perlin (float &result, const H &hash,
                    const float &x, const float &y, const float &z, const float &w)
{
#if OIIO_SIMD
    // result = 0; return;

    int4 XYZW;
    float4 fxyzw = floorfrac (float4(x,y,z,w), &XYZW);
    float4 uvts = fade (fxyzw);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = OIIO::simd::shuffle<0>(XYZW) + (*(int4*)i0101);
    int4 cornery = OIIO::simd::shuffle<1>(XYZW) + (*(int4*)i0011);
    int4 cornerz = OIIO::simd::shuffle<2>(XYZW);
    int4 cornerz1 = cornerz + int4::One();
    int4 cornerw = OIIO::simd::shuffle<3>(XYZW);

    int4 corner_hash_z0 = hash (cornerx, cornery, cornerz,  cornerw);
    int4 corner_hash_z1 = hash (cornerx, cornery, cornerz1, cornerw);
    int4 cornerw1 = cornerw + int4::One();
    int4 corner_hash_z2 = hash (cornerx, cornery, cornerz,  cornerw1);
    int4 corner_hash_z3 = hash (cornerx, cornery, cornerz1, cornerw1);

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    float4 remainderx = OIIO::simd::shuffle<0>(fxyzw) - (*(float4*)f0101);
    float4 remaindery = OIIO::simd::shuffle<1>(fxyzw) - (*(float4*)f0011);
    float4 remainderz = OIIO::simd::shuffle<2>(fxyzw);
    float4 remainderz1 = remainderz - float4::One();
    float4 remainderw = OIIO::simd::shuffle<3>(fxyzw);
    float4 corner_grad_z0 = grad (corner_hash_z0, remainderx, remaindery, remainderz,  remainderw);
    float4 corner_grad_z1 = grad (corner_hash_z1, remainderx, remaindery, remainderz1, remainderw);
    float4 remainderw1 = remainderw - float4::One();
    float4 corner_grad_z2 = grad (corner_hash_z2, remainderx, remaindery, remainderz,  remainderw1);
    float4 corner_grad_z3 = grad (corner_hash_z3, remainderx, remaindery, remainderz1, remainderw1);

    result = scale4 (OIIO::lerp (trilerp (corner_grad_z0, corner_grad_z1, uvts),
                                 trilerp (corner_grad_z2, corner_grad_z3, uvts),
                                 OIIO::simd::extract<3>(uvts)));

#else
    // ORIGINAL -- non-SIMD
    int X; float fx = floorfrac(x, &X);
    int Y; float fy = floorfrac(y, &Y);
    int Z; float fz = floorfrac(z, &Z);
    int W; float fw = floorfrac(w, &W);

    float u = fade(fx);
    float v = fade(fy);
    float t = fade(fz);
    float s = fade(fw);

    result = OIIO::lerp (
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W  ), fx     , fy     , fz     , fw     ),
                              grad (hash (X+1, Y  , Z  , W  ), fx-1.0f, fy     , fz     , fw     ),
                              grad (hash (X  , Y+1, Z  , W  ), fx     , fy-1.0f, fz     , fw     ),
                              grad (hash (X+1, Y+1, Z  , W  ), fx-1.0f, fy-1.0f, fz     , fw     ),
                              grad (hash (X  , Y  , Z+1, W  ), fx     , fy     , fz-1.0f, fw     ),
                              grad (hash (X+1, Y  , Z+1, W  ), fx-1.0f, fy     , fz-1.0f, fw     ),
                              grad (hash (X  , Y+1, Z+1, W  ), fx     , fy-1.0f, fz-1.0f, fw     ),
                              grad (hash (X+1, Y+1, Z+1, W  ), fx-1.0f, fy-1.0f, fz-1.0f, fw     ),
                              u, v, t),
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W+1), fx     , fy     , fz     , fw-1.0f),
                              grad (hash (X+1, Y  , Z  , W+1), fx-1.0f, fy     , fz     , fw-1.0f),
                              grad (hash (X  , Y+1, Z  , W+1), fx     , fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X+1, Y+1, Z  , W+1), fx-1.0f, fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X  , Y  , Z+1, W+1), fx     , fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y  , Z+1, W+1), fx-1.0f, fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X  , Y+1, Z+1, W+1), fx     , fy-1.0f, fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y+1, Z+1, W+1), fx-1.0f, fy-1.0f, fz-1.0f, fw-1.0f),
                              u, v, t),
               s);
    result = scale4 (result);
#endif
}



template <typename H> OSL_HOSTDEVICE
inline void perlin (Dual2<float> &result, const H &hash,
                    const Dual2<float> &x, const Dual2<float> &y)
{
#if OIIO_SIMD
    // result = 0; return;
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    Dual2<float4> fxy (float4(fx.val(), fy.val(), 0.0f),
                       float4(fx.dx(), fy.dx(), 0.0f),
                       float4(fx.dy(), fy.dy(), 0.0f));
    Dual2<float4> uv = fade (fxy);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously.
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);
    int4 corner_hash = hash (cornerx, cornery);

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    Dual2<float4> remainderx = shuffle<0>(fxy) - (*(float4*)f0101);
    Dual2<float4> remaindery = shuffle<1>(fxy) - (*(float4*)f0011);
    Dual2<float4> corner_grad = grad (corner_hash, remainderx, remaindery);

    result = scale2 (bilerp (corner_grad, uv));
#else
    // Non-SIMD case
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    Dual2<float> u = fade(fx);
    Dual2<float> v = fade(fy);
    result = OIIO::bilerp (grad (hash (X  , Y  ), fx     , fy     ),
                           grad (hash (X+1, Y  ), fx-1.0f, fy     ),
                           grad (hash (X  , Y+1), fx     , fy-1.0f),
                           grad (hash (X+1, Y+1), fx-1.0f, fy-1.0f), u, v);
    result = scale2 (result);
#endif
}




template <typename H> OSL_HOSTDEVICE
inline void perlin (Dual2<float> &result, const H &hash,
                    const Dual2<float> &x, const Dual2<float> &y, const Dual2<float> &z)
{
#if OIIO_SIMD
    // result = 0; return;

    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    Dual2<float4> fxyz (float4(fx.val(), fy.val(), fz.val()),
                        float4(fx.dx(), fy.dx(), fz.dx()),
                        float4(fx.dy(), fy.dy(), fz.dy()));
    Dual2<float4> uvw = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);
    int4 cornerz = Z;
    int4 corner_hash_z0 = hash (cornerx, cornery, cornerz);
    int4 corner_hash_z1 = hash (cornerx, cornery, cornerz+int4::One());

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    Dual2<float4> remainderx = shuffle<0>(fxyz) - (*(float4*)f0101);
    Dual2<float4> remaindery = shuffle<1>(fxyz) - (*(float4*)f0011);
    Dual2<float4> remainderz = shuffle<2>(fxyz);

    Dual2<float4> corner_grad_z0 = grad (corner_hash_z0, remainderx, remaindery, remainderz);
    Dual2<float4> corner_grad_z1 = grad (corner_hash_z1, remainderx, remaindery, remainderz-float4::One());

    // Interpolate along the z axis first
    Dual2<float4> xy = OIIO::lerp (corner_grad_z0, corner_grad_z1, shuffle<2>(uvw));
    // Interpolate along x axis
    Dual2<float4> xx = OIIO::lerp (xy, shuffle<1,1,3,3>(xy), shuffle<0>(uvw));
    // interpolate along y axis
    result = scale3 (extract<0>(OIIO::lerp (xx,shuffle<2>(xx), shuffle<1>(uvw))));

#else
    // Non-SIMD case
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    Dual2<float> u = fade(fx);
    Dual2<float> v = fade(fy);
    Dual2<float> w = fade(fz);
    result = OIIO::trilerp (grad (hash (X  , Y  , Z  ), fx     , fy     , fz     ),
                            grad (hash (X+1, Y  , Z  ), fx-1.0f, fy     , fz     ),
                            grad (hash (X  , Y+1, Z  ), fx     , fy-1.0f, fz     ),
                            grad (hash (X+1, Y+1, Z  ), fx-1.0f, fy-1.0f, fz     ),
                            grad (hash (X  , Y  , Z+1), fx     , fy     , fz-1.0f),
                            grad (hash (X+1, Y  , Z+1), fx-1.0f, fy     , fz-1.0f),
                            grad (hash (X  , Y+1, Z+1), fx     , fy-1.0f, fz-1.0f),
                            grad (hash (X+1, Y+1, Z+1), fx-1.0f, fy-1.0f, fz-1.0f),
                            u, v, w);
    result = scale3 (result);
#endif
}




template <typename H> OSL_HOSTDEVICE
inline void perlin (Dual2<float> &result, const H &hash,
                    const Dual2<float> &x, const Dual2<float> &y,
                    const Dual2<float> &z, const Dual2<float> &w)
{
#if OIIO_SIMD
    // result = 0; return;

    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    int W; Dual2<float> fw = floorfrac(w, &W);
    Dual2<float4> fxyzw (float4(fx.val(), fy.val(), fz.val(), fw.val()),
                         float4(fx.dx (), fy.dx (), fz.dx (), fw.dx ()),
                         float4(fx.dy (), fy.dy (), fz.dy (), fw.dy ()));
    Dual2<float4> uvts = fade (fxyzw);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);
    int4 cornerz = Z;
    int4 cornerz1 = Z + int4::One();
    int4 cornerw = W;
    int4 cornerw1 = W + int4::One();
    int4 corner_hash_z0 = hash (cornerx, cornery, cornerz,  cornerw);
    int4 corner_hash_z1 = hash (cornerx, cornery, cornerz1, cornerw);
    int4 corner_hash_z2 = hash (cornerx, cornery, cornerz,  cornerw1);
    int4 corner_hash_z3 = hash (cornerx, cornery, cornerz1, cornerw1);

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    Dual2<float4> remainderx  = shuffle<0>(fxyzw) - (*(float4*)f0101);
    Dual2<float4> remaindery  = shuffle<1>(fxyzw) - (*(float4*)f0011);
    Dual2<float4> remainderz  = shuffle<2>(fxyzw);
    Dual2<float4> remainderz1 = remainderz - float4::One();
    Dual2<float4> remainderw  = shuffle<3>(fxyzw);
    Dual2<float4> remainderw1 = remainderw - float4::One();

    Dual2<float4> corner_grad_z0 = grad (corner_hash_z0, remainderx, remaindery, remainderz,  remainderw);
    Dual2<float4> corner_grad_z1 = grad (corner_hash_z1, remainderx, remaindery, remainderz1, remainderw);
    Dual2<float4> corner_grad_z2 = grad (corner_hash_z2, remainderx, remaindery, remainderz,  remainderw1);
    Dual2<float4> corner_grad_z3 = grad (corner_hash_z3, remainderx, remaindery, remainderz1, remainderw1);

    // Interpolate along the w axis first
    Dual2<float4> xyz0 = OIIO::lerp (corner_grad_z0, corner_grad_z2, shuffle<3>(uvts));
    Dual2<float4> xyz1 = OIIO::lerp (corner_grad_z1, corner_grad_z3, shuffle<3>(uvts));
    Dual2<float4> xy = OIIO::lerp (xyz0, xyz1, shuffle<2>(uvts));
    // Interpolate along x axis
    Dual2<float4> xx = OIIO::lerp (xy, shuffle<1,1,3,3>(xy), shuffle<0>(uvts));
    // interpolate along y axis
    result = scale4 (extract<0>(OIIO::lerp (xx,shuffle<2>(xx), shuffle<1>(uvts))));

#else
    // ORIGINAL -- non-SIMD
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    int W; Dual2<float> fw = floorfrac(w, &W);

    Dual2<float> u = fade(fx);
    Dual2<float> v = fade(fy);
    Dual2<float> t = fade(fz);
    Dual2<float> s = fade(fw);

    result = OIIO::lerp (
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W  ), fx     , fy     , fz     , fw     ),
                              grad (hash (X+1, Y  , Z  , W  ), fx-1.0f, fy     , fz     , fw     ),
                              grad (hash (X  , Y+1, Z  , W  ), fx     , fy-1.0f, fz     , fw     ),
                              grad (hash (X+1, Y+1, Z  , W  ), fx-1.0f, fy-1.0f, fz     , fw     ),
                              grad (hash (X  , Y  , Z+1, W  ), fx     , fy     , fz-1.0f, fw     ),
                              grad (hash (X+1, Y  , Z+1, W  ), fx-1.0f, fy     , fz-1.0f, fw     ),
                              grad (hash (X  , Y+1, Z+1, W  ), fx     , fy-1.0f, fz-1.0f, fw     ),
                              grad (hash (X+1, Y+1, Z+1, W  ), fx-1.0f, fy-1.0f, fz-1.0f, fw     ),
                              u, v, t),
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W+1), fx     , fy     , fz     , fw-1.0f),
                              grad (hash (X+1, Y  , Z  , W+1), fx-1.0f, fy     , fz     , fw-1.0f),
                              grad (hash (X  , Y+1, Z  , W+1), fx     , fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X+1, Y+1, Z  , W+1), fx-1.0f, fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X  , Y  , Z+1, W+1), fx     , fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y  , Z+1, W+1), fx-1.0f, fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X  , Y+1, Z+1, W+1), fx     , fy-1.0f, fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y+1, Z+1, W+1), fx-1.0f, fy-1.0f, fz-1.0f, fw-1.0f),
                              u, v, t),
               s);
    result = scale4 (result);
#endif
}




template <typename H> OSL_HOSTDEVICE
inline void perlin (Vec3 &result, const H &hash,
                    const float &x, const float &y)
{
#if OIIO_SIMD
    // result.setValue(0,0,0); return;
    int4 XYZ;
    float4 fxyz = floorfrac (float4(x,y,0), &XYZ);
    float4 uv = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = OIIO::simd::shuffle<0>(XYZ) + (*(int4*)i0101);
    int4 cornery = OIIO::simd::shuffle<1>(XYZ) + (*(int4*)i0011);

    // We actually derive 3 hashes (one for each output dimension) for each
    // corner.
    int4 corner_hash[3];
    hash (corner_hash, cornerx, cornery);

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    float4 remainderx = OIIO::simd::shuffle<0>(fxyz) - (*(float4*)f0101);
    float4 remaindery = OIIO::simd::shuffle<1>(fxyz) - (*(float4*)f0011);
    for (int i = 0; i < 3; ++i) {
        float4 corner_grad = grad (corner_hash[i], remainderx, remaindery);
        // Do the bilinear interpolation with SIMD. Here's the fastest way
        // I've found to do it.
        float4 xx = OIIO::lerp (corner_grad, OIIO::simd::shuffle<1,1,3,3>(corner_grad), uv[0]);
        result[i] = scale2 (OIIO::simd::extract<0>(OIIO::lerp (xx,OIIO::simd::shuffle<2>(xx), uv[1])));
    }
#else
    // Non-SIMD case
    typedef float T;
    int X; T fx = floorfrac(x, &X);
    int Y; T fy = floorfrac(y, &Y);
    T u = fade(fx);
    T v = fade(fy);
    result = OIIO::bilerp (grad (hash (X  , Y  ), fx     , fy     ),
                           grad (hash (X+1, Y  ), fx-1.0f, fy     ),
                           grad (hash (X  , Y+1), fx     , fy-1.0f),
                           grad (hash (X+1, Y+1), fx-1.0f, fy-1.0f), u, v);
    result = scale2 (result);
#endif
}



template <typename H> OSL_HOSTDEVICE
inline void perlin (Vec3 &result, const H &hash,
                    const float &x, const float &y, const float &z)
{
#if OIIO_SIMD
    // result.setValue(0,0,0); return;
#if 0
    // You'd think it would be faster to do the floorfrac in parallel, but
    // according to my timings, it is not. Come back and understand why.
    int4 XYZ;
    float4 fxyz = floorfrac (float4(x,y,z), &XYZ);
    float4 uvw = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    int4 cornerx = shuffle<0>(XYZ) + int4(0,1,0,1);
    int4 cornery = shuffle<1>(XYZ) + int4(0,0,1,1);
    int4 cornerz = shuffle<2>(XYZ);
#else
    int X; float fx = floorfrac(x, &X);
    int Y; float fy = floorfrac(y, &Y);
    int Z; float fz = floorfrac(z, &Z);
    float4 fxyz (fx, fy, fz); // = floorfrac (xyz, &XYZ);
    float4 uvw = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);
    int4 cornerz = Z;
#endif

    // We actually derive 3 hashes (one for each output dimension) for each
    // corner.
    int4 corner_hash_z0[3], corner_hash_z1[3];
    hash (corner_hash_z0, cornerx, cornery, cornerz);
    hash (corner_hash_z1, cornerx, cornery, cornerz+int4::One());

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    float4 remainderx = OIIO::simd::shuffle<0>(fxyz) - (*(float4*)f0101);
    float4 remaindery = OIIO::simd::shuffle<1>(fxyz) - (*(float4*)f0011);
    float4 remainderz0 = OIIO::simd::shuffle<2>(fxyz);
    float4 remainderz1 = OIIO::simd::shuffle<2>(fxyz) - float4::One();
    for (int i = 0; i < 3; ++i) {
        float4 corner_grad_z0 = grad (corner_hash_z0[i], remainderx, remaindery, remainderz0);
        float4 corner_grad_z1 = grad (corner_hash_z1[i], remainderx, remaindery, remainderz1);

        // Interpolate along the z axis first
        float4 xy = OIIO::lerp (corner_grad_z0, corner_grad_z1, OIIO::simd::shuffle<2>(uvw));
        // Interpolate along x axis
        float4 xx = OIIO::lerp (xy, OIIO::simd::shuffle<1,1,3,3>(xy), OIIO::simd::shuffle<0>(uvw));
        // interpolate along y axis
        result[i] = scale3 (OIIO::simd::extract<0>(OIIO::lerp (xx,OIIO::simd::shuffle<2>(xx), OIIO::simd::shuffle<1>(uvw))));
    }
#else
    // ORIGINAL -- non-SIMD
    int X; float fx = floorfrac(x, &X);
    int Y; float fy = floorfrac(y, &Y);
    int Z; float fz = floorfrac(z, &Z);
    float u = fade(fx);
    float v = fade(fy);
    float w = fade(fz);
    result = OIIO::trilerp (grad (hash (X  , Y  , Z  ), fx     , fy     , fz      ),
                            grad (hash (X+1, Y  , Z  ), fx-1.0f, fy     , fz      ),
                            grad (hash (X  , Y+1, Z  ), fx     , fy-1.0f, fz      ),
                            grad (hash (X+1, Y+1, Z  ), fx-1.0f, fy-1.0f, fz      ),
                            grad (hash (X  , Y  , Z+1), fx     , fy     , fz-1.0f ),
                            grad (hash (X+1, Y  , Z+1), fx-1.0f, fy     , fz-1.0f ),
                            grad (hash (X  , Y+1, Z+1), fx     , fy-1.0f, fz-1.0f ),
                            grad (hash (X+1, Y+1, Z+1), fx-1.0f, fy-1.0f, fz-1.0f ),
                            u, v, w);
    result = scale3 (result);
#endif
}



template <typename H> OSL_HOSTDEVICE
inline void perlin (Vec3 &result, const H &hash,
                    const float &x, const float &y, const float &z, const float &w)
{
#if OIIO_SIMD
    // result.setValue(0,0,0); return;

    int4 XYZW;
    float4 fxyzw = floorfrac (float4(x,y,z,w), &XYZW);
    float4 uvts = fade (fxyzw);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = OIIO::simd::shuffle<0>(XYZW) + (*(int4*)i0101);
    int4 cornery = OIIO::simd::shuffle<1>(XYZW) + (*(int4*)i0011);
    int4 cornerz = OIIO::simd::shuffle<2>(XYZW);
    int4 cornerw = OIIO::simd::shuffle<3>(XYZW);

    // We actually derive 3 hashes (one for each output dimension) for each
    // corner.
    int4 corner_hash_z0[3], corner_hash_z1[3];
    int4 corner_hash_z2[3], corner_hash_z3[3];
    hash (corner_hash_z0, cornerx, cornery, cornerz, cornerw);
    hash (corner_hash_z1, cornerx, cornery, cornerz+int4::One(), cornerw);
    hash (corner_hash_z2, cornerx, cornery, cornerz, cornerw+int4::One());
    hash (corner_hash_z3, cornerx, cornery, cornerz+int4::One(), cornerw+int4::One());

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    float4 remainderx = OIIO::simd::shuffle<0>(fxyzw) - (*(float4*)f0101);
    float4 remaindery = OIIO::simd::shuffle<1>(fxyzw) - (*(float4*)f0011);
    float4 remainderz = OIIO::simd::shuffle<2>(fxyzw);
    float4 remainderw = OIIO::simd::shuffle<3>(fxyzw);
//    float4 remainderz0 = shuffle<2>(fxyz);
//    float4 remainderz1 = shuffle<2>(fxyz) - 1.0f;
    for (int i = 0; i < 3; ++i) {
        float4 corner_grad_z0 = grad (corner_hash_z0[i], remainderx, remaindery, remainderz, remainderw);
        float4 corner_grad_z1 = grad (corner_hash_z1[i], remainderx, remaindery, remainderz-float4::One(), remainderw);
        float4 corner_grad_z2 = grad (corner_hash_z2[i], remainderx, remaindery, remainderz, remainderw-float4::One());
        float4 corner_grad_z3 = grad (corner_hash_z3[i], remainderx, remaindery, remainderz-float4::One(), remainderw-float4::One());
        result[i] = scale4 (OIIO::lerp (trilerp (corner_grad_z0, corner_grad_z1, uvts),
                                        trilerp (corner_grad_z2, corner_grad_z3, uvts),
                                        OIIO::simd::extract<3>(uvts)));
    }
#else
    // ORIGINAL -- non-SIMD
    int X; float fx = floorfrac(x, &X);
    int Y; float fy = floorfrac(y, &Y);
    int Z; float fz = floorfrac(z, &Z);
    int W; float fw = floorfrac(w, &W);

    float u = fade(fx);
    float v = fade(fy);
    float t = fade(fz);
    float s = fade(fw);

    result = OIIO::lerp (
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W  ), fx     , fy     , fz     , fw     ),
                              grad (hash (X+1, Y  , Z  , W  ), fx-1.0f, fy     , fz     , fw     ),
                              grad (hash (X  , Y+1, Z  , W  ), fx     , fy-1.0f, fz     , fw     ),
                              grad (hash (X+1, Y+1, Z  , W  ), fx-1.0f, fy-1.0f, fz     , fw     ),
                              grad (hash (X  , Y  , Z+1, W  ), fx     , fy     , fz-1.0f, fw     ),
                              grad (hash (X+1, Y  , Z+1, W  ), fx-1.0f, fy     , fz-1.0f, fw     ),
                              grad (hash (X  , Y+1, Z+1, W  ), fx     , fy-1.0f, fz-1.0f, fw     ),
                              grad (hash (X+1, Y+1, Z+1, W  ), fx-1.0f, fy-1.0f, fz-1.0f, fw     ),
                              u, v, t),
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W+1), fx     , fy     , fz     , fw-1.0f),
                              grad (hash (X+1, Y  , Z  , W+1), fx-1.0f, fy     , fz     , fw-1.0f),
                              grad (hash (X  , Y+1, Z  , W+1), fx     , fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X+1, Y+1, Z  , W+1), fx-1.0f, fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X  , Y  , Z+1, W+1), fx     , fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y  , Z+1, W+1), fx-1.0f, fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X  , Y+1, Z+1, W+1), fx     , fy-1.0f, fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y+1, Z+1, W+1), fx-1.0f, fy-1.0f, fz-1.0f, fw-1.0f),
                              u, v, t),
               s);
    result = scale4 (result);
#endif
}



template <typename H> OSL_HOSTDEVICE
inline void perlin (Dual2<Vec3> &result, const H &hash,
                    const Dual2<float> &x, const Dual2<float> &y)
{
    // result = Vec3(0,0,0); return;
#if OIIO_SIMD
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    Dual2<float4> fxyz (float4(fx.val(), fy.val(), 0.0f),
                        float4(fx.dx(),  fy.dx(),  0.0f),
                        float4(fx.dy(),  fy.dy(),  0.0f));
    Dual2<float4> uvw = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);

    // We actually derive 3 hashes (one for each output dimension) for each
    // corner.
    int4 corner_hash[3];
    hash (corner_hash, cornerx, cornery);

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    Dual2<float4> remainderx = shuffle<0>(fxyz) - (*(float4*)f0101);
    Dual2<float4> remaindery = shuffle<1>(fxyz) - (*(float4*)f0011);
    Dual2<float> r[3];
    for (int i = 0; i < 3; ++i) {
        Dual2<float4> corner_grad = grad (corner_hash[i], remainderx, remaindery);
        // Interpolate along x axis
        Dual2<float4> xx = OIIO::lerp (corner_grad, shuffle<1,1,3,3>(corner_grad), shuffle<0>(uvw));
        // interpolate along y axis
        r[i] = scale2 (extract<0>(OIIO::lerp (xx,shuffle<2>(xx), shuffle<1>(uvw))));
    }
    result.set (Vec3 (r[0].val(), r[1].val(), r[2].val()),
                Vec3 (r[0].dx(),  r[1].dx(),  r[2].dx()),
                Vec3 (r[0].dy(),  r[1].dy(),  r[2].dy()));

#else
    // ORIGINAL -- non-SIMD
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    Dual2<float> u = fade(fx);
    Dual2<float> v = fade(fy);
    result = OIIO::bilerp (grad (hash (X  , Y  ), fx     , fy     ),
                           grad (hash (X+1, Y  ), fx-1.0f, fy     ),
                           grad (hash (X  , Y+1), fx     , fy-1.0f),
                           grad (hash (X+1, Y+1), fx-1.0f, fy-1.0f),
                           u, v);
    result = scale2 (result);
#endif
}



template <typename H> OSL_HOSTDEVICE
inline void perlin (Dual2<Vec3> &result, const H &hash,
                    const Dual2<float> &x, const Dual2<float> &y, const Dual2<float> &z)
{
    // result = Vec3(0,0,0); return;
#if OIIO_SIMD
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    Dual2<float4> fxyz (float4(fx.val(), fy.val(), fz.val()),
                        float4(fx.dx(),  fy.dx(),  fz.dx()),
                        float4(fx.dy(),  fy.dy(),  fz.dy()));
    Dual2<float4> uvw = fade (fxyz);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);
    int4 cornerz = Z;

    // We actually derive 3 hashes (one for each output dimension) for each
    // corner.
    int4 corner_hash_z0[3], corner_hash_z1[3];
    hash (corner_hash_z0, cornerx, cornery, cornerz);
    hash (corner_hash_z1, cornerx, cornery, cornerz+int4::One());

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    Dual2<float4> remainderx = shuffle<0>(fxyz) - (*(float4*)f0101);
    Dual2<float4> remaindery = shuffle<1>(fxyz) - (*(float4*)f0011);
    Dual2<float4> remainderz0 = shuffle<2>(fxyz);
    Dual2<float4> remainderz1 = shuffle<2>(fxyz) - float4::One();
    Dual2<float> r[3];
    for (int i = 0; i < 3; ++i) {
        Dual2<float4> corner_grad_z0 = grad (corner_hash_z0[i], remainderx, remaindery, remainderz0);
        Dual2<float4> corner_grad_z1 = grad (corner_hash_z1[i], remainderx, remaindery, remainderz1);

        // Interpolate along the z axis first
        Dual2<float4> xy = OIIO::lerp (corner_grad_z0, corner_grad_z1, shuffle<2>(uvw));
        // Interpolate along x axis
        Dual2<float4> xx = OIIO::lerp (xy, shuffle<1,1,3,3>(xy), shuffle<0>(uvw));
        // interpolate along y axis
        r[i] = scale3 (extract<0>(OIIO::lerp (xx,shuffle<2>(xx), shuffle<1>(uvw))));
    }
    result.set (Vec3 (r[0].val(), r[1].val(), r[2].val()),
                Vec3 (r[0].dx(),  r[1].dx(),  r[2].dx()),
                Vec3 (r[0].dy(),  r[1].dy(),  r[2].dy()));

#else
    // ORIGINAL -- non-SIMD
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    Dual2<float> u = fade(fx);
    Dual2<float> v = fade(fy);
    Dual2<float> w = fade(fz);
    result = OIIO::trilerp (grad (hash (X  , Y  , Z  ), fx     , fy     , fz      ),
                            grad (hash (X+1, Y  , Z  ), fx-1.0f, fy     , fz      ),
                            grad (hash (X  , Y+1, Z  ), fx     , fy-1.0f, fz      ),
                            grad (hash (X+1, Y+1, Z  ), fx-1.0f, fy-1.0f, fz      ),
                            grad (hash (X  , Y  , Z+1), fx     , fy     , fz-1.0f ),
                            grad (hash (X+1, Y  , Z+1), fx-1.0f, fy     , fz-1.0f ),
                            grad (hash (X  , Y+1, Z+1), fx     , fy-1.0f, fz-1.0f ),
                            grad (hash (X+1, Y+1, Z+1), fx-1.0f, fy-1.0f, fz-1.0f ),
                            u, v, w);
    result = scale3 (result);
#endif
}



template <typename H> OSL_HOSTDEVICE
inline void perlin (Dual2<Vec3> &result, const H &hash,
                    const Dual2<float> &x, const Dual2<float> &y,
                    const Dual2<float> &z, const Dual2<float> &w)
{
    // result = Vec3(0,0,0); return;
#if OIIO_SIMD

    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    int W; Dual2<float> fw = floorfrac(w, &W);
    Dual2<float4> fxyzw (float4(fx.val(), fy.val(), fz.val(), fw.val()),
                         float4(fx.dx (), fy.dx (), fz.dx (), fw.dx ()),
                         float4(fx.dy (), fy.dy (), fz.dy (), fw.dy ()));
    Dual2<float4> uvts = fade (fxyzw);

    // We parallelize primarily by computing the hashes and gradients at the
    // integer lattice corners simultaneously. We need 8 total (for 3D), so
    // we do two sets of 4. (Future opportunity to do all 8 simultaneously
    // with AVX.)
    static const OIIO_SIMD4_ALIGN int i0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN int i0011[4] = {0,0,1,1};
    int4 cornerx = X + (*(int4*)i0101);
    int4 cornery = Y + (*(int4*)i0011);
    int4 cornerz = Z;
    int4 cornerz1 = Z + int4::One();
    int4 cornerw = W;
    int4 cornerw1 = W + int4::One();

    // We actually derive 3 hashes (one for each output dimension) for each
    // corner.
    int4 corner_hash_z0[3], corner_hash_z1[3], corner_hash_z2[3], corner_hash_z3[3];
    hash (corner_hash_z0, cornerx, cornery, cornerz,  cornerw);
    hash (corner_hash_z1, cornerx, cornery, cornerz1, cornerw);
    hash (corner_hash_z2, cornerx, cornery, cornerz,  cornerw1);
    hash (corner_hash_z3, cornerx, cornery, cornerz1, cornerw1);

    static const OIIO_SIMD4_ALIGN float f0101[4] = {0,1,0,1};
    static const OIIO_SIMD4_ALIGN float f0011[4] = {0,0,1,1};
    Dual2<float4> remainderx  = shuffle<0>(fxyzw) - (*(float4*)f0101);
    Dual2<float4> remaindery  = shuffle<1>(fxyzw) - (*(float4*)f0011);
    Dual2<float4> remainderz  = shuffle<2>(fxyzw);
    Dual2<float4> remainderz1 = remainderz - float4::One();
    Dual2<float4> remainderw  = shuffle<3>(fxyzw);
    Dual2<float4> remainderw1 = remainderw - float4::One();

    Dual2<float> r[3];
    for (int i = 0; i < 3; ++i) {
        Dual2<float4> corner_grad_z0 = grad (corner_hash_z0[i], remainderx, remaindery, remainderz,  remainderw);
        Dual2<float4> corner_grad_z1 = grad (corner_hash_z1[i], remainderx, remaindery, remainderz1, remainderw);
        Dual2<float4> corner_grad_z2 = grad (corner_hash_z2[i], remainderx, remaindery, remainderz,  remainderw1);
        Dual2<float4> corner_grad_z3 = grad (corner_hash_z3[i], remainderx, remaindery, remainderz1, remainderw1);

        // Interpolate along the w axis first
        Dual2<float4> xyz0 = OIIO::lerp (corner_grad_z0, corner_grad_z2, shuffle<3>(uvts));
        Dual2<float4> xyz1 = OIIO::lerp (corner_grad_z1, corner_grad_z3, shuffle<3>(uvts));
        Dual2<float4> xy = OIIO::lerp (xyz0, xyz1, shuffle<2>(uvts));
        // Interpolate along x axis
        Dual2<float4> xx = OIIO::lerp (xy, shuffle<1,1,3,3>(xy), shuffle<0>(uvts));
        // interpolate along y axis
        r[i] = scale4 (extract<0>(OIIO::lerp (xx,shuffle<2>(xx), shuffle<1>(uvts))));

    }
    result.set (Vec3 (r[0].val(), r[1].val(), r[2].val()),
                Vec3 (r[0].dx(),  r[1].dx(),  r[2].dx()),
                Vec3 (r[0].dy(),  r[1].dy(),  r[2].dy()));

#else
    // ORIGINAL -- non-SIMD
    int X; Dual2<float> fx = floorfrac(x, &X);
    int Y; Dual2<float> fy = floorfrac(y, &Y);
    int Z; Dual2<float> fz = floorfrac(z, &Z);
    int W; Dual2<float> fw = floorfrac(w, &W);

    Dual2<float> u = fade(fx);
    Dual2<float> v = fade(fy);
    Dual2<float> t = fade(fz);
    Dual2<float> s = fade(fw);

    result = OIIO::lerp (
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W  ), fx     , fy     , fz     , fw     ),
                              grad (hash (X+1, Y  , Z  , W  ), fx-1.0f, fy     , fz     , fw     ),
                              grad (hash (X  , Y+1, Z  , W  ), fx     , fy-1.0f, fz     , fw     ),
                              grad (hash (X+1, Y+1, Z  , W  ), fx-1.0f, fy-1.0f, fz     , fw     ),
                              grad (hash (X  , Y  , Z+1, W  ), fx     , fy     , fz-1.0f, fw     ),
                              grad (hash (X+1, Y  , Z+1, W  ), fx-1.0f, fy     , fz-1.0f, fw     ),
                              grad (hash (X  , Y+1, Z+1, W  ), fx     , fy-1.0f, fz-1.0f, fw     ),
                              grad (hash (X+1, Y+1, Z+1, W  ), fx-1.0f, fy-1.0f, fz-1.0f, fw     ),
                              u, v, t),
               OIIO::trilerp (grad (hash (X  , Y  , Z  , W+1), fx     , fy     , fz     , fw-1.0f),
                              grad (hash (X+1, Y  , Z  , W+1), fx-1.0f, fy     , fz     , fw-1.0f),
                              grad (hash (X  , Y+1, Z  , W+1), fx     , fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X+1, Y+1, Z  , W+1), fx-1.0f, fy-1.0f, fz     , fw-1.0f),
                              grad (hash (X  , Y  , Z+1, W+1), fx     , fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y  , Z+1, W+1), fx-1.0f, fy     , fz-1.0f, fw-1.0f),
                              grad (hash (X  , Y+1, Z+1, W+1), fx     , fy-1.0f, fz-1.0f, fw-1.0f),
                              grad (hash (X+1, Y+1, Z+1, W+1), fx-1.0f, fy-1.0f, fz-1.0f, fw-1.0f),
                              u, v, t),
               s);
    result = scale4 (result);
#endif
}



struct Noise {
    OSL_HOSTDEVICE Noise () { }

    inline OSL_HOSTDEVICE void operator() (float &result, float x) const {
        HashScalar h;
        perlin(result, h, x);
        result = 0.5f * (result + 1);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float y) const {
        HashScalar h;
        perlin(result, h, x, y);
        result = 0.5f * (result + 1);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p) const {
        HashScalar h;
        perlin(result, h, p.x, p.y, p.z);
        result = 0.5f * (result + 1);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t) const {
        HashScalar h;
        perlin(result, h, p.x, p.y, p.z, t);
        result = 0.5f * (result + 1);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x) const {
        HashVector h;
        perlin(result, h, x);
        result = 0.5f * (result + Vec3(1, 1, 1));
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y) const {
        HashVector h;
        perlin(result, h, x, y);
        result = 0.5f * (result + Vec3(1, 1, 1));
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p) const {
        HashVector h;
        perlin(result, h, p.x, p.y, p.z);
        result = 0.5f * (result + Vec3(1, 1, 1));
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t) const {
        HashVector h;
        perlin(result, h, p.x, p.y, p.z, t);
        result = 0.5f * (result + Vec3(1, 1, 1));
    }

    // dual versions

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x) const {
        HashScalar h;
        perlin(result, h, x);
        result = 0.5f * (result + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        HashScalar h;
        perlin(result, h, x, y);
        result = 0.5f * (result + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p) const {
        HashScalar h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz);
        result = 0.5f * (result + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        HashScalar h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz, t);
        result = 0.5f * (result + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const {
        HashVector h;
        perlin(result, h, x);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1, 1, 1));
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        HashVector h;
        perlin(result, h, x, y);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1, 1, 1));
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {
        HashVector h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1, 1, 1));
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        HashVector h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz, t);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1, 1, 1));
    }
};

struct SNoise {
    OSL_HOSTDEVICE SNoise () { }

    inline OSL_HOSTDEVICE void operator() (float &result, float x) const {
        HashScalar h;
        perlin(result, h, x);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float y) const {
        HashScalar h;
        perlin(result, h, x, y);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p) const {
        HashScalar h;
        perlin(result, h, p.x, p.y, p.z);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t) const {
        HashScalar h;
        perlin(result, h, p.x, p.y, p.z, t);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x) const {
        HashVector h;
        perlin(result, h, x);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y) const {
        HashVector h;
        perlin(result, h, x, y);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p) const {
        HashVector h;
        perlin(result, h, p.x, p.y, p.z);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t) const {
        HashVector h;
        perlin(result, h, p.x, p.y, p.z, t);
    }


    // dual versions

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x) const {
        HashScalar h;
        perlin(result, h, x);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        HashScalar h;
        perlin(result, h, x, y);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p) const {
        HashScalar h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        HashScalar h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz, t);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const {
        HashVector h;
        perlin(result, h, x);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        HashVector h;
        perlin(result, h, x, y);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {
        HashVector h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        HashVector h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz, t);
    }
};



struct PeriodicNoise {
    OSL_HOSTDEVICE PeriodicNoise () { }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float px) const {
        HashScalarPeriodic h(px);
        perlin(result, h, x);
        result = 0.5f * (result + 1);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float y, float px, float py) const {
        HashScalarPeriodic h(px, py);
        perlin(result, h, x, y);
        result = 0.5f * (result + 1);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, const Vec3 &pp) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z);
        perlin(result, h, p.x, p.y, p.z);
        result = 0.5f * (result + 1);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t, const Vec3 &pp, float pt) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z, pt);
        perlin(result, h, p.x, p.y, p.z, t);
        result = 0.5f * (result + 1);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float px) const {
        HashVectorPeriodic h(px);
        perlin(result, h, x);
        result = 0.5f * (result + Vec3(1, 1, 1));
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y, float px, float py) const {
        HashVectorPeriodic h(px, py);
        perlin(result, h, x, y);
        result = 0.5f * (result + Vec3(1, 1, 1));
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, const Vec3 &pp) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z);
        perlin(result, h, p.x, p.y, p.z);
        result = 0.5f * (result + Vec3(1, 1, 1));
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t, const Vec3 &pp, float pt) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z, pt);
        perlin(result, h, p.x, p.y, p.z, t);
        result = 0.5f * (result + Vec3(1, 1, 1));
    }

    // dual versions

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x, float px) const {
        HashScalarPeriodic h(px);
        perlin(result, h, x);
        result = 0.5f * (result + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y,
                                           float px, float py) const {
        HashScalarPeriodic h(px, py);
        perlin(result, h, x, y);
        result = 0.5f * (result + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p, const Vec3 &pp) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z);
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz);
        result = 0.5f * (result + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p, const Dual2<float> &t,
                                           const Vec3 &pp, float pt) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z, pt);
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz, t);
        result = 0.5f * (result + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, float px) const {
        HashVectorPeriodic h(px);
        perlin(result, h, x);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1, 1, 1));
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y,
                                           float px, float py) const {
        HashVectorPeriodic h(px, py);
        perlin(result, h, x, y);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1, 1, 1));
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Vec3 &pp) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z);
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1, 1, 1));
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t,
                                           const Vec3 &pp, float pt) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z, pt);
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz, t);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1, 1, 1));
    }
};

struct PeriodicSNoise {
    OSL_HOSTDEVICE PeriodicSNoise () { }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float px) const {
        HashScalarPeriodic h(px);
        perlin(result, h, x);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float y, float px, float py) const {
        HashScalarPeriodic h(px, py);
        perlin(result, h, x, y);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, const Vec3 &pp) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z);
        perlin(result, h, p.x, p.y, p.z);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t, const Vec3 &pp, float pt) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z, pt);
        perlin(result, h, p.x, p.y, p.z, t);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float px) const {
        HashVectorPeriodic h(px);
        perlin(result, h, x);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y, float px, float py) const {
        HashVectorPeriodic h(px, py);
        perlin(result, h, x, y);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, const Vec3 &pp) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z);
        perlin(result, h, p.x, p.y, p.z);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t, const Vec3 &pp, float pt) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z, pt);
        perlin(result, h, p.x, p.y, p.z, t);
    }

    // dual versions

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x, float px) const {
        HashScalarPeriodic h(px);
        perlin(result, h, x);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y,
                                           float px, float py) const {
        HashScalarPeriodic h(px, py);
        perlin(result, h, x, y);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p, const Vec3 &pp) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z);
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p, const Dual2<float> &t,
                                           const Vec3 &pp, float pt) const {
        HashScalarPeriodic h(pp.x, pp.y, pp.z, pt);
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz, t);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, float px) const {
        HashVectorPeriodic h(px);
        perlin(result, h, x);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y,
                                           float px, float py) const {
        HashVectorPeriodic h(px, py);
        perlin(result, h, x, y);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Vec3 &pp) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z);
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t,
                                           const Vec3 &pp, float pt) const {
        HashVectorPeriodic h(pp.x, pp.y, pp.z, pt);
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz, t);
    }
};



struct SimplexNoise {
    OSL_HOSTDEVICE SimplexNoise () { }

    inline OSL_HOSTDEVICE void operator() (float &result, float x) const {
        result = simplexnoise1 (x);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float y) const {
        result = simplexnoise2 (x, y);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p) const {
        result = simplexnoise3 (p.x, p.y, p.z);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t) const {
        result = simplexnoise4 (p.x, p.y, p.z, t);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x) const {
        result[0] = simplexnoise1 (x, 0);
        result[1] = simplexnoise1 (x, 1);
        result[2] = simplexnoise1 (x, 2);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y) const {
        result[0] = simplexnoise2 (x, y, 0);
        result[1] = simplexnoise2 (x, y, 1);
        result[2] = simplexnoise2 (x, y, 2);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p) const {
        result[0] = simplexnoise3 (p.x, p.y, p.z, 0);
        result[1] = simplexnoise3 (p.x, p.y, p.z, 1);
        result[2] = simplexnoise3 (p.x, p.y, p.z, 2);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t) const {
        result[0] = simplexnoise4 (p.x, p.y, p.z, t, 0);
        result[1] = simplexnoise4 (p.x, p.y, p.z, t, 1);
        result[2] = simplexnoise4 (p.x, p.y, p.z, t, 2);
    }


    // dual versions

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x,
                                           int seed=0) const {
        float r, dndx;
        r = simplexnoise1 (x.val(), seed, &dndx);
        result.set (r, dndx * x.dx(), dndx * x.dy());
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x,
                                           const Dual2<float> &y, int seed=0) const {
        float r, dndx, dndy;
        r = simplexnoise2 (x.val(), y.val(), seed, &dndx, &dndy);
        result.set (r, dndx * x.dx() + dndy * y.dx(),
                       dndx * x.dy() + dndy * y.dy());
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                                           int seed=0) const {
        float r, dndx, dndy, dndz;
        r = simplexnoise3 (p.val()[0], p.val()[1], p.val()[2],
                           seed, &dndx, &dndy, &dndz);
        result.set (r, dndx * p.dx()[0] + dndy * p.dx()[1] + dndz * p.dx()[2],
                       dndx * p.dy()[0] + dndy * p.dy()[1] + dndz * p.dy()[2]);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                                           const Dual2<float> &t, int seed=0) const {
        float r, dndx, dndy, dndz, dndt;
        r = simplexnoise4 (p.val()[0], p.val()[1], p.val()[2], t.val(),
                           seed, &dndx, &dndy, &dndz, &dndt);
        result.set (r, dndx * p.dx()[0] + dndy * p.dx()[1] + dndz * p.dx()[2] + dndt * t.dx(),
                       dndx * p.dy()[0] + dndy * p.dy()[1] + dndz * p.dy()[2] + dndt * t.dy());
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, x, 0);
        (*this)(r1, x, 1);
        (*this)(r2, x, 2);
        result = make_Vec3 (r0, r1, r2);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, x, y, 0);
        (*this)(r1, x, y, 1);
        (*this)(r2, x, y, 2);
        result = make_Vec3 (r0, r1, r2);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, p, 0);
        (*this)(r1, p, 1);
        (*this)(r2, p, 2);
        result = make_Vec3 (r0, r1, r2);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, p, t, 0);
        (*this)(r1, p, t, 1);
        (*this)(r2, p, t, 2);
        result = make_Vec3 (r0, r1, r2);
    }
};



// Unsigned simplex noise
struct USimplexNoise {
    OSL_HOSTDEVICE USimplexNoise () { }

    inline OSL_HOSTDEVICE void operator() (float &result, float x) const {
        result = 0.5f * (simplexnoise1 (x) + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, float x, float y) const {
        result = 0.5f * (simplexnoise2 (x, y) + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p) const {
        result = 0.5f * (simplexnoise3 (p.x, p.y, p.z) + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (float &result, const Vec3 &p, float t) const {
        result = 0.5f * (simplexnoise4 (p.x, p.y, p.z, t) + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x) const {
        result[0] = 0.5f * (simplexnoise1 (x, 0) + 1.0f);
        result[1] = 0.5f * (simplexnoise1 (x, 1) + 1.0f);
        result[2] = 0.5f * (simplexnoise1 (x, 2) + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, float x, float y) const {
        result[0] = 0.5f * (simplexnoise2 (x, y, 0) + 1.0f);
        result[1] = 0.5f * (simplexnoise2 (x, y, 1) + 1.0f);
        result[2] = 0.5f * (simplexnoise2 (x, y, 2) + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p) const {
        result[0] = 0.5f * (simplexnoise3 (p.x, p.y, p.z, 0) + 1.0f);
        result[1] = 0.5f * (simplexnoise3 (p.x, p.y, p.z, 1) + 1.0f);
        result[2] = 0.5f * (simplexnoise3 (p.x, p.y, p.z, 2) + 1.0f);
    }

    inline OSL_HOSTDEVICE void operator() (Vec3 &result, const Vec3 &p, float t) const {
        result[0] = 0.5f * (simplexnoise4 (p.x, p.y, p.z, t, 0) + 1.0f);
        result[1] = 0.5f * (simplexnoise4 (p.x, p.y, p.z, t, 1) + 1.0f);
        result[2] = 0.5f * (simplexnoise4 (p.x, p.y, p.z, t, 2) + 1.0f);
    }

    // dual versions

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x,
                                           int seed=0) const {
        float r, dndx;
        r = simplexnoise1 (x.val(), seed, &dndx);
        r = 0.5f * (r + 1.0f);
        dndx *= 0.5f;
        result.set (r, dndx * x.dx(), dndx * x.dy());
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<float> &x,
                                           const Dual2<float> &y, int seed=0) const {
        float r, dndx, dndy;
        r = simplexnoise2 (x.val(), y.val(), seed, &dndx, &dndy);
        r = 0.5f * (r + 1.0f);
        dndx *= 0.5f;
        dndy *= 0.5f;
        result.set (r, dndx * x.dx() + dndy * y.dx(),
                       dndx * x.dy() + dndy * y.dy());
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                                           int seed=0) const {
        float r, dndx, dndy, dndz;
        r = simplexnoise3 (p.val()[0], p.val()[1], p.val()[2],
                           seed, &dndx, &dndy, &dndz);
        r = 0.5f * (r + 1.0f);
        dndx *= 0.5f;
        dndy *= 0.5f;
        dndz *= 0.5f;
        result.set (r, dndx * p.dx()[0] + dndy * p.dx()[1] + dndz * p.dx()[2],
                       dndx * p.dy()[0] + dndy * p.dy()[1] + dndz * p.dy()[2]);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<float> &result, const Dual2<Vec3> &p,
                                           const Dual2<float> &t, int seed=0) const {
        float r, dndx, dndy, dndz, dndt;
        r = simplexnoise4 (p.val()[0], p.val()[1], p.val()[2], t.val(),
                           seed, &dndx, &dndy, &dndz, &dndt);
        r = 0.5f * (r + 1.0f);
        dndx *= 0.5f;
        dndy *= 0.5f;
        dndz *= 0.5f;
        dndt *= 0.5f;
        result.set (r, dndx * p.dx()[0] + dndy * p.dx()[1] + dndz * p.dx()[2] + dndt * t.dx(),
                       dndx * p.dy()[0] + dndy * p.dy()[1] + dndz * p.dy()[2] + dndt * t.dy());
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, x, 0);
        (*this)(r1, x, 1);
        (*this)(r2, x, 2);
        result = make_Vec3 (r0, r1, r2);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, x, y, 0);
        (*this)(r1, x, y, 1);
        (*this)(r2, x, y, 2);
        result = make_Vec3 (r0, r1, r2);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, p, 0);
        (*this)(r1, p, 1);
        (*this)(r2, p, 2);
        result = make_Vec3 (r0, r1, r2);
    }

    inline OSL_HOSTDEVICE void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        Dual2<float> r0, r1, r2;
        (*this)(r0, p, t, 0);
        (*this)(r1, p, t, 1);
        (*this)(r2, p, t, 2);
        result = make_Vec3 (r0, r1, r2);
    }
};

} // anonymous namespace



OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<float> gabor (const Dual2<Vec3> &P, const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<float> gabor (const Dual2<float> &x, const Dual2<float> &y,
                    const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<float> gabor (const Dual2<float> &x, const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<Vec3> gabor3 (const Dual2<Vec3> &P, const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<Vec3> gabor3 (const Dual2<float> &x, const Dual2<float> &y,
                    const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<Vec3> gabor3 (const Dual2<float> &x, const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<float> pgabor (const Dual2<Vec3> &P, const Vec3 &Pperiod,
                     const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<float> pgabor (const Dual2<float> &x, const Dual2<float> &y,
                     float xperiod, float yperiod, const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<float> pgabor (const Dual2<float> &x, float xperiod,
                     const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<Vec3> pgabor3 (const Dual2<Vec3> &P, const Vec3 &Pperiod,
                     const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<Vec3> pgabor3 (const Dual2<float> &x, const Dual2<float> &y,
                     float xperiod, float yperiod, const NoiseParams *opt);

OSLNOISEPUBLIC OSL_HOSTDEVICE
Dual2<Vec3> pgabor3 (const Dual2<float> &x, float xperiod,
                     const NoiseParams *opt);



}; // namespace pvt

namespace oslnoise {

#define DECLNOISE(name,impl)                    \
    template <class S> OSL_HOSTDEVICE           \
    inline float name (S x) {                   \
        pvt::impl noise;                        \
        float r;                                \
        noise (r, x);                           \
        return r;                               \
    }                                           \
                                                \
    template <class S, class T> OSL_HOSTDEVICE  \
    inline float name (S x, T y) {              \
        pvt::impl noise;                        \
        float r;                                \
        noise (r, x, y);                        \
        return r;                               \
    }                                           \
                                                \
    template <class S> OSL_HOSTDEVICE           \
    inline Vec3 v ## name (S x) {               \
        pvt::impl noise;                        \
        Vec3 r;                                 \
        noise (r, x);                           \
        return r;                               \
    }                                           \
                                                \
    template <class S, class T> OSL_HOSTDEVICE  \
    inline Vec3 v ## name (S x, T y) {          \
        pvt::impl noise;                        \
        Vec3 r;                                 \
        noise (r, x, y);                        \
        return r;                               \
    }


DECLNOISE (snoise, SNoise)
DECLNOISE (noise, Noise)
DECLNOISE (cellnoise, CellNoise)
DECLNOISE (hashnoise, HashNoise)

#undef DECLNOISE

}   // namespace oslnoise


OSL_NAMESPACE_EXIT
