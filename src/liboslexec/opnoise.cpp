/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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

#include <limits>

#include "oslexec_pvt.h"
#include "oslops.h"

#include "OpenImageIO/varyingref.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {

namespace {

/// return the greatest integer <= x
inline int quick_floor (float x) {
    return (int) x - ((x < 0) ? 1 : 0);
}

/// convert a 32 bit integer into a floating point number in [0,1]
inline float bits_to_01 (unsigned int bits) {
    // divide by 2^32-1
    return bits * (1.0f / std::numeric_limits<unsigned int>::max());
}

/// hash an array of N 32 bit values into a pseudo-random value
/// based on my favorite hash: http://burtleburtle.net/bob/c/lookup3.c
/// templated so that the compiler can unroll the loops for us
template <int N>
inline unsigned int
inthash (const unsigned int k[N]) {
    // define some handy macros
#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))
#define mix(a,b,c) \
{ \
    a -= c;  a ^= rot(c, 4);  c += b; \
    b -= a;  b ^= rot(a, 6);  a += c; \
    c -= b;  c ^= rot(b, 8);  b += a; \
    a -= c;  a ^= rot(c,16);  c += b; \
    b -= a;  b ^= rot(a,19);  a += c; \
    c -= b;  c ^= rot(b, 4);  b += a; \
}
#define final(a,b,c) \
{ \
    c ^= b; c -= rot(b,14); \
    a ^= c; a -= rot(c,11); \
    b ^= a; b -= rot(a,25); \
    c ^= b; c -= rot(b,16); \
    a ^= c; a -= rot(c,4);  \
    b ^= a; b -= rot(a,14); \
    c ^= b; c -= rot(b,24); \
}
    // now hash the data!
    unsigned int a, b, c, len = N;
    a = b = c = 0xdeadbeef + (len << 2) + 13;
    while (len > 3) {
        a += k[0];
        b += k[1];
        c += k[2];
        mix(a, b, c);
        len -= 3;
        k += 3;
    }
    switch (len) {
        case 3 : c += k[2];
        case 2 : b += k[1];
        case 1 : a += k[0];
        final(a, b, c);
        case 0:
            break;
    }
    return c;
    // macros not needed anymore
#undef rot
#undef mix
#undef final
}

struct CellNoise {
    CellNoise (ShadingExecution *) { }

    inline void operator() (float &result, float x) {
        unsigned int iv[1];
        iv[0] = quick_floor (x);
        hash1<1> (result, iv);
    }

    inline void operator() (float &result, float x, float y) {
        unsigned int iv[2];
        iv[0] = quick_floor (x);
        iv[1] = quick_floor (y);
        hash1<2> (result, iv);
    }

    inline void operator() (float &result, const Vec3 &p) {
        unsigned int iv[3];
        iv[0] = quick_floor (p.x);
        iv[1] = quick_floor (p.y);
        iv[2] = quick_floor (p.z);
        hash1<3> (result, iv);
    }

    inline void operator() (float &result, const Vec3 &p, float t) {
        unsigned int iv[4];
        iv[0] = quick_floor (p.x);
        iv[1] = quick_floor (p.y);
        iv[2] = quick_floor (p.z);
        iv[3] = quick_floor (t);
        hash1<4> (result, iv);
    }

    inline void operator() (Vec3 &result, float x) {
        unsigned int iv[2];
        iv[0] = quick_floor (x);
        hash3<2> (result, iv);
    }

    inline void operator() (Vec3 &result, float x, float y) {
        unsigned int iv[3];
        iv[0] = quick_floor (x);
        iv[1] = quick_floor (y);
        hash3<3> (result, iv);
    }

    inline void operator() (Vec3 &result, const Vec3 &p) {
        unsigned int iv[4];
        iv[0] = quick_floor (p.x);
        iv[1] = quick_floor (p.y);
        iv[2] = quick_floor (p.z);
        hash3<4> (result, iv);
    }

    inline void operator() (Vec3 &result, const Vec3 &p, float t) {
        unsigned int iv[5];
        iv[0] = quick_floor (p.x);
        iv[1] = quick_floor (p.y);
        iv[2] = quick_floor (p.z);
        iv[3] = quick_floor (t);
        hash3<5> (result, iv);
    }

private:
    template <int N>
    inline void hash1 (float &result, const unsigned int k[N]) {
        result = bits_to_01(inthash<N>(k));
    }

    template <int N>
    inline void hash3 (Vec3 &result, unsigned int k[N]) {
        k[N-1] = 0; result.x = bits_to_01 (inthash<N> (k));
        k[N-1] = 1; result.y = bits_to_01 (inthash<N> (k));
        k[N-1] = 2; result.z = bits_to_01 (inthash<N> (k));
    }
};

// helper functions for perlin noise 

// always return a value inside [0,b) - even for negative numbers
inline int imod(int a, int b) {
    a %= b;
    return a < 0 ? a + b : a;
}

// floorfrac return quick_floor as well as the fractional remainder
// FIXME: already implemented inside OIIO but can't easily override it for duals
//        inside a different namespace
inline float floorfrac(float x, int* i) {
    *i = quick_floor(x);
    return x - *i;
}

inline Dual2<float> floorfrac(const Dual2<float> &x, int* i) {
    float frac = floorfrac(x.val(), i);
    // slope of x is not affected by this operation
    return Dual2<float>(frac, x.dx(), x.dy());
}

template <typename T>
inline T fade (const T &t) { 
   return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); 
}


// lerp implementation, a few obstacles prevent us from using Imath::lerp():
//   * Original perlin code was written assuming lerp(t,a,b) which is different
//     from Imath::lerp(a,b,t)
//   * Imath version uses 1-t internally which doesn't work for duals. We need
//     to use (1.0f-t) for the templated operator-() to get chosen
//   * This call recieves some arguments which are not handled trivially by
//     templated overloads such as lerp(Dual2<float>, Vec3, Vec3)

float lerp(float t, float a, float b) {
    return (1.0f - t) * a + t * b;
}

Vec3 lerp(float t, Vec3 a, Vec3 b) {
    return (1.0f - t) * a + t * b;
}

Dual2<float> lerp(const Dual2<float> &t, float a, float b) {
    return (1.0f - t) * a + t * b;
}

Dual2<float> lerp(const Dual2<float> &t, const Dual2<float> &a, const Dual2<float> &b) {
    return (1.0f - t) * a + t * b;
}

Dual2<Vec3> lerp(const Dual2<float> &t, const Vec3 &a, const Vec3 &b) {
    return Dual2<Vec3>((1.0f - t.val()) * a + t.val() * b,
                       (b - a) * t.dx(),
                       (b - a) * t.dy());
}

Dual2<Vec3> lerp(const Dual2<float> &t, const Dual2<Vec3> &a, const Dual2<Vec3> &b) {
    Dual2<float> ax(a.val().x, a.dx().x, a.dy().x);
    Dual2<float> ay(a.val().y, a.dx().y, a.dy().y);
    Dual2<float> az(a.val().z, a.dx().z, a.dy().z);
    Dual2<float> bx(b.val().x, b.dx().x, b.dy().x);
    Dual2<float> by(b.val().y, b.dx().y, b.dy().y);
    Dual2<float> bz(b.val().z, b.dx().z, b.dy().z);
    Dual2<float> lerpx = (1.0f - t) * ax + t * bx;
    Dual2<float> lerpy = (1.0f - t) * ay + t * by;
    Dual2<float> lerpz = (1.0f - t) * az + t * bz;
    return Dual2<Vec3>(Vec3(lerpx.val(), lerpy.val(), lerpz.val()),
                       Vec3(lerpx.dx() , lerpy.dx() , lerpz.dx() ),
                       Vec3(lerpx.dy() , lerpy.dy() , lerpz.dy() ));
}


// 1,2,3 and 4 dimensional gradient functions - perform a dot product against a
// randomly chosen edge vector of the hypercube, when the number of edges is not
// exactly a power of 2 (such as in dimension 3), replicate the edges to avoid
// an expensive mod operation.

template <typename T>
inline T grad (int hash, const T &x) {
    static const float G1[2] = { -1, 1 };

    int h = hash & 0x1;
    return x * G1[h];
}

template <typename T>
inline T grad (int hash, const T &x, const T &y) {
    static const float G2[4][2] = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };

    int h = hash & 0x3;
    return x * G2[h][0] + y * G2[h][1];
}
 
template <typename T>
inline T grad (int hash, const T &x, const T &y, const T &z) {
    static const float G3[16][3] = {
            { 1, 1, 0 }, { -1, 1, 0 }, { 1, -1, 0 }, { -1, -1, 0 }, { 1, 0, 1 }, { -1, 0, 1 },
            { 1, 0, -1 }, { -1, 0, -1 }, { 0, 1, 1 }, { 0, -1, 1 }, { 0, 1, -1 }, { 0, -1, -1 },
            { 1, 1, 0 }, { -1, 1, 0 }, { 0, -1, 1 }, { 0, -1, -1 } };

    int h = hash & 15;
    return x * G3[h][0] + y * G3[h][1] + z * G3[h][2];
}

template <typename T>
inline T grad (int hash, const T &x, const T &y, const T &z, const T &w) {
    static const float G4[32][4] = {
            { -1, -1, -1, 0 }, { -1, -1, 1, 0 }, { -1, 1, -1, 0 }, { -1, 1, 1, 0 }, { 1, -1, -1, 0 },
            { 1, -1, 1, 0 }, { 1, 1, -1, 0 }, { 1, 1, 1, 0 }, { -1, -1, 0, -1 }, { -1, 1, 0, -1 },
            { 1, -1, 0, -1 }, { 1, 1, 0, -1 }, { -1, -1, 0, 1 }, { -1, 1, 0, 1 }, { 1, -1, 0, 1 },
            { 1, 1, 0, 1 }, { -1, 0, -1, -1 }, { 1, 0, -1, -1 }, { -1, 0, -1, 1 }, { 1, 0, -1, 1 },
            { -1, 0, 1, -1 }, { 1, 0, 1, -1 }, { -1, 0, 1, 1 }, { 1, 0, 1, 1 }, { 0, -1, -1, -1 },
            { 0, -1, -1, 1 }, { 0, -1, 1, -1 }, { 0, -1, 1, 1 }, { 0, 1, -1, -1 }, { 0, 1, -1, 1 },
            { 0, 1, 1, -1 }, { 0, 1, 1, 1 } };

    int h = hash & 31;
    return x * G4[h][0] + y * G4[h][1] + z * G4[h][2] + w * G4[h][3];
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
    return Dual2<Vec3> (Vec3(rx.val(), ry.val(), rz.val()),
                        Vec3(rx.dx() , ry.dx() , rz.dx() ),
                        Vec3(rx.dy() , ry.dy() , rz.dy() ));
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
    return Dual2<Vec3> (Vec3(rx.val(), ry.val(), rz.val()),
                        Vec3(rx.dx() , ry.dx() , rz.dx() ),
                        Vec3(rx.dy() , ry.dy() , rz.dy() ));
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
    return Dual2<Vec3> (Vec3(rx.val(), ry.val(), rz.val()),
                        Vec3(rx.dx() , ry.dx() , rz.dx() ),
                        Vec3(rx.dy() , ry.dy() , rz.dy() ));
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
    return Dual2<Vec3> (Vec3(rx.val(), ry.val(), rz.val()),
                        Vec3(rx.dx() , ry.dx() , rz.dx() ),
                        Vec3(rx.dy() , ry.dy() , rz.dy() ));
}

template <typename V, typename H, typename T>
inline void perlin (V& result, H& hash, const T &x) {
    int X; T fx = floorfrac(x, &X);
    T u = fade(fx);

    result = lerp (u, grad (hash (X  ), fx     ),
                      grad (hash (X+1), fx-1.0f));
}

template <typename V, typename H, typename T>
inline void perlin (V &result, const H &hash, const T &x, const T &y) {
    int X; T fx = floorfrac(x, &X);
    int Y; T fy = floorfrac(y, &Y);

    T u = fade(fx);
    T v = fade(fy);

    result = lerp (v, lerp (u, grad (hash (X  , Y  ), fx     , fy     ),
                               grad (hash (X+1, Y  ), fx-1.0f, fy     )),
                      lerp (u, grad (hash (X  , Y+1), fx     , fy-1.0f),
                               grad (hash (X+1, Y+1), fx-1.0f, fy-1.0f)));
}


template <typename V, typename H, typename T>
inline void perlin (V &result, const H &hash, const T &x, const T &y, const T &z) {
    int X; T fx = floorfrac(x, &X);
    int Y; T fy = floorfrac(y, &Y);
    int Z; T fz = floorfrac(z, &Z);

    T u = fade(fx);
    T v = fade(fy);
    T w = fade(fz);

    result = lerp (w, lerp (v, lerp (u, grad (hash (X  , Y  , Z  ), fx     , fy     , fz      ),
                                        grad (hash (X+1, Y  , Z  ), fx-1.0f, fy     , fz      )),
                               lerp (u, grad (hash (X  , Y+1, Z  ), fx     , fy-1.0f, fz      ),
                                        grad (hash (X+1, Y+1, Z  ), fx-1.0f, fy-1.0f, fz      ))),
                      lerp (v, lerp (u, grad (hash (X  , Y  , Z+1), fx     , fy     , fz-1.0f ),
                                        grad (hash (X+1, Y  , Z+1), fx-1.0f, fy     , fz-1.0f )),
                               lerp (u, grad (hash (X  , Y+1, Z+1), fx     , fy-1.0f, fz-1.0f ),
                                        grad (hash (X+1, Y+1, Z+1), fx-1.0f, fy-1.0f, fz-1.0f ))));
}

template <typename V, typename H, typename T>
inline void perlin (V &result, const H &hash, const T &x, const T &y, const T &z, const T &w) {
    int X; T fx = floorfrac(x, &X);
    int Y; T fy = floorfrac(y, &Y);
    int Z; T fz = floorfrac(z, &Z);
    int W; T fw = floorfrac(w, &W);

    T u = fade(fx);
    T v = fade(fy);
    T t = fade(fz);
    T s = fade(fw);

    result = lerp (s, lerp (t, lerp (v, lerp (u, grad (hash (X  , Y  , Z  , W  ), fx     , fy     , fz     , fw     ),
                                                 grad (hash (X+1, Y  , Z  , W  ), fx-1.0f, fy     , fz     , fw     )),
                                        lerp (u, grad (hash (X  , Y+1, Z  , W  ), fx     , fy-1.0f, fz     , fw     ),
                                                 grad (hash (X+1, Y+1, Z  , W  ), fx-1.0f, fy-1.0f, fz     , fw     ))),
                               lerp (v, lerp (u, grad (hash (X  , Y  , Z+1, W  ), fx     , fy     , fz-1.0f, fw     ),
                                                 grad (hash (X+1, Y  , Z+1, W  ), fx-1.0f, fy     , fz-1.0f, fw     )),
                                        lerp (u, grad (hash (X  , Y+1, Z+1, W  ), fx     , fy-1.0f, fz-1.0f, fw     ),
                                                 grad (hash (X+1, Y+1, Z+1, W  ), fx-1.0f, fy-1.0f, fz-1.0f, fw     )))),
                      lerp (t, lerp (v, lerp (u, grad (hash (X  , Y  , Z  , W+1), fx     , fy     , fz     , fw-1.0f),
                                                 grad (hash (X+1, Y  , Z  , W+1), fx-1.0f, fy     , fz     , fw-1.0f)),
                                        lerp (u, grad (hash (X  , Y+1, Z  , W+1), fx     , fy-1.0f, fz     , fw-1.0f),
                                                 grad (hash (X+1, Y+1, Z  , W+1), fx-1.0f, fy-1.0f, fz     , fw-1.0f))),
                               lerp (v, lerp (u, grad (hash (X  , Y  , Z+1, W+1), fx     , fy     , fz-1.0f, fw-1.0f),
                                                 grad (hash (X+1, Y  , Z+1, W+1), fx-1.0f, fy     , fz-1.0f, fw-1.0f)),
                                        lerp (u, grad (hash (X  , Y+1, Z+1, W+1), fx     , fy-1.0f, fz-1.0f, fw-1.0f),
                                                 grad (hash (X+1, Y+1, Z+1, W+1), fx-1.0f, fy-1.0f, fz-1.0f, fw-1.0f)))));

}

struct HashScalar {
    int operator() (int x) const {
        unsigned int iv[1];
        iv[0] = x;
        return inthash<1> (iv);
    }

    int operator() (int x, int y) const {
        unsigned int iv[2];
        iv[0] = x;
        iv[1] = y;
        return inthash<2> (iv);
    }

    int operator() (int x, int y, int z) const {
        unsigned int iv[3];
        iv[0] = x;
        iv[1] = y;
        iv[2] = z;
        return inthash<3> (iv);
    }

    int operator() (int x, int y, int z, int w) const {
        unsigned int iv[4];
        iv[0] = x;
        iv[1] = y;
        iv[2] = z;
        iv[3] = w;
        return inthash<4> (iv);
    }
};

struct HashVector {
    Vec3i operator() (int x) const {
        unsigned int iv[1];
        iv[0] = x;
        return hash3<1> (iv);
    }

    Vec3i operator() (int x, int y) const {
        unsigned int iv[2];
        iv[0] = x;
        iv[1] = y;
        return hash3<2> (iv);
    }

    Vec3i operator() (int x, int y, int z) const {
        unsigned int iv[3];
        iv[0] = x;
        iv[1] = y;
        iv[2] = z;
        return hash3<3> (iv);
    }

    Vec3i operator() (int x, int y, int z, int w) const {
        unsigned int iv[4];
        iv[0] = x;
        iv[1] = y;
        iv[2] = z;
        iv[3] = w;
        return hash3<4> (iv);
    }

    template <int N>
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
};

struct HashScalarPeriodic {
    HashScalarPeriodic (int px) : px(px < 1 ? 1: px) {}
    HashScalarPeriodic (int px, int py) : px(px < 1 ? 1: px), py(py < 1 ? 1: py) {}
    HashScalarPeriodic (int px, int py, int pz) : px(px < 1 ? 1: px), py(py < 1 ? 1: py), pz(pz < 1 ? 1: pz) {}
    HashScalarPeriodic (int px, int py, int pz, int pw) : px(px < 1 ? 1: px), py(py < 1 ? 1: py), pz(pz < 1 ? 1: pz), pw(pw < 1 ? 1: pw) {}

    int px, py, pz, pw;

    int operator() (int x) const {
        unsigned int iv[1];
        iv[0] = imod (x, px);
        return inthash<1> (iv);
    }

    int operator() (int x, int y) const {
        unsigned int iv[2];
        iv[0] = imod (x, px);
        iv[1] = imod (y, py);
        return inthash<2> (iv);
    }

    int operator() (int x, int y, int z) const {
        unsigned int iv[3];
        iv[0] = imod (x, px);
        iv[1] = imod (y, py);
        iv[2] = imod (z, pz);
        return inthash<3> (iv);
    }

    int operator() (int x, int y, int z, int w) const {
        unsigned int iv[4];
        iv[0] = imod (x, px);
        iv[1] = imod (y, py);
        iv[2] = imod (z, pz);
        iv[3] = imod (w, pz);
        return inthash<4> (iv);
    }
};

struct HashVectorPeriodic {
    HashVectorPeriodic (int px) : px(px < 1 ? 1: px) {}
    HashVectorPeriodic (int px, int py) : px(px < 1 ? 1: px), py(py < 1 ? 1: py) {}
    HashVectorPeriodic (int px, int py, int pz) : px(px < 1 ? 1: px), py(py < 1 ? 1: py), pz(pz < 1 ? 1: pz) {}
    HashVectorPeriodic (int px, int py, int pz, int pw) : px(px < 1 ? 1: px), py(py < 1 ? 1: py), pz(pz < 1 ? 1: pz), pw(pw < 1 ? 1: pw) {}

    int px, py, pz, pw;

    Vec3i operator() (int x) const {
        unsigned int iv[1];
        iv[0] = imod (x, px);
        return hash3<1> (iv);
    }

    Vec3i operator() (int x, int y) const {
        unsigned int iv[2];
        iv[0] = imod (x, px);
        iv[1] = imod (y, py);
        return hash3<2> (iv);
    }

    Vec3i operator() (int x, int y, int z) const {
        unsigned int iv[3];
        iv[0] = imod (x, px);
        iv[1] = imod (y, py);
        iv[2] = imod (z, pz);
        return hash3<3> (iv);

    }

    Vec3i operator() (int x, int y, int z, int w) const {
        unsigned int iv[4];
        iv[0] = imod (x, px);
        iv[1] = imod (y, py);
        iv[2] = imod (z, pz);
        iv[3] = imod (w, pw);
        return hash3<4> (iv);
    }

    template <int N>
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
};

struct Noise {
    Noise (ShadingExecution *) { }

    inline void operator() (float &result, float x) const {
        HashScalar h;
        perlin(result, h, x);
        result = 0.5f * (result + 1);
    }

    inline void operator() (float &result, float x, float y) const {
        HashScalar h;
        perlin(result, h, x, y);
        result = 0.5f * (result + 1);
    }

    inline void operator() (float &result, const Vec3 &p) const {
        HashScalar h;
        perlin(result, h, p.x, p.y, p.z);
        result = 0.5f * (result + 1);
    }

    inline void operator() (float &result, const Vec3 &p, float t) const {
        HashScalar h;
        perlin(result, h, p.x, p.y, p.z, t);
        result = 0.5f * (result + 1);
    }

    inline void operator() (Vec3 &result, float x) const {
        HashVector h;
        perlin(result, h, x);
        result = 0.5f * (result + Vec3(1, 1, 1));
    }

    inline void operator() (Vec3 &result, float x, float y) const {
        HashVector h;
        perlin(result, h, x, y);
        result = 0.5f * (result + Vec3(1, 1, 1));
    }

    inline void operator() (Vec3 &result, const Vec3 &p) const {
        HashVector h;
        perlin(result, h, p.x, p.y, p.z);
        result = 0.5f * (result + Vec3(1, 1, 1));
    }

    inline void operator() (Vec3 &result, const Vec3 &p, float t) const {
        HashVector h;
        perlin(result, h, p.x, p.y, p.z, t);
        result = 0.5f * (result + Vec3(1, 1, 1));
    }

    // dual versions

    inline void operator() (Dual2<float> &result, const Dual2<float> &x) const {
        HashScalar h;
        perlin(result, h, x);
        result = 0.5f * (result + 1.0f);
    }

    inline void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        HashScalar h;
        perlin(result, h, x, y);
        result = 0.5f * (result + 1.0f);
    }

    inline void operator() (Dual2<float> &result, const Dual2<Vec3> &p) const {
        HashScalar h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz);
        result = 0.5f * (result + 1.0f);
    }

    inline void operator() (Dual2<float> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        HashScalar h;        
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz, t);
        result = 0.5f * (result + 1.0f);
    }

    inline void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const {
        HashVector h;
        perlin(result, h, x);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1, 1, 1));
    }

    inline void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        HashVector h;
        perlin(result, h, x, y);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1, 1, 1));
    }

    inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {
        HashVector h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1, 1, 1));
    }

    inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        HashVector h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz, t);
        result = Vec3(0.5f, 0.5f, 0.5f) * (result + Vec3(1, 1, 1));
    }
};

struct SNoise {
    SNoise (ShadingExecution *) { }

    inline void operator() (float &result, float x) const {
        HashScalar h;
        perlin(result, h, x);
    }

    inline void operator() (float &result, float x, float y) const {
        HashScalar h;
        perlin(result, h, x, y);
    }

    inline void operator() (float &result, const Vec3 &p) const {
        HashScalar h;
        perlin(result, h, p.x, p.y, p.z);
    }

    inline void operator() (float &result, const Vec3 &p, float t) const {
        HashScalar h;
        perlin(result, h, p.x, p.y, p.z, t);
    }

    inline void operator() (Vec3 &result, float x) const {
        HashVector h;
        perlin(result, h, x);
    }

    inline void operator() (Vec3 &result, float x, float y) const {
        HashVector h;
        perlin(result, h, x, y);
    }

    inline void operator() (Vec3 &result, const Vec3 &p) const {
        HashVector h;
        perlin(result, h, p.x, p.y, p.z);
    }

    inline void operator() (Vec3 &result, const Vec3 &p, float t) const {
        HashVector h;
        perlin(result, h, p.x, p.y, p.z, t);
    }


    // dual versions

    inline void operator() (Dual2<float> &result, const Dual2<float> &x) const {
        HashScalar h;
        perlin(result, h, x);
    }

    inline void operator() (Dual2<float> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        HashScalar h;
        perlin(result, h, x, y);
    }

    inline void operator() (Dual2<float> &result, const Dual2<Vec3> &p) const {
        HashScalar h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz);
    }

    inline void operator() (Dual2<float> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        HashScalar h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz, t);
    }

    inline void operator() (Dual2<Vec3> &result, const Dual2<float> &x) const {
        HashVector h;
        perlin(result, h, x);
    }

    inline void operator() (Dual2<Vec3> &result, const Dual2<float> &x, const Dual2<float> &y) const {
        HashVector h;
        perlin(result, h, x, y);
    }

    inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p) const {
        HashVector h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz);
    }

    inline void operator() (Dual2<Vec3> &result, const Dual2<Vec3> &p, const Dual2<float> &t) const {
        HashVector h;
        Dual2<float> px(p.val().x, p.dx().x, p.dy().x);
        Dual2<float> py(p.val().y, p.dx().y, p.dy().y);
        Dual2<float> pz(p.val().z, p.dx().z, p.dy().z);
        perlin(result, h, px, py, pz, t);
    }
};

// TODO: periodic noise functor

} // anonymous namespace



template <typename FUNCTION>
DECLOP (generic_noise_function_noderivs)
{
    ASSERT (nargs == 2 || nargs == 3);
    OpImpl impl = NULL;
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));

    // type check first args
    ASSERT (Result.typespec().is_float() || Result.typespec().is_triple());
    ASSERT (A.typespec().is_float() || A.typespec().is_triple());

    if (nargs == 2) {
        // either ff or fp
        if (Result.typespec().is_float() && A.typespec().is_float())
            impl = unary_op_noderivs<float, float, FUNCTION>;
        else if (Result.typespec().is_float() && A.typespec().is_triple())
            impl = unary_op_noderivs<float, Vec3, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_float())
            impl = unary_op_noderivs<Vec3, float, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_triple())
            impl = unary_op_noderivs<Vec3, Vec3, FUNCTION>;
    } else if (nargs == 3) {
        // either fff or fpf
        Symbol &B (exec->sym (args[2]));
        ASSERT (B.typespec().is_float());
        if (Result.typespec().is_float() && A.typespec().is_float())
            impl = binary_op_noderivs<float, float, float, FUNCTION>;
        else if (Result.typespec().is_float() && A.typespec().is_triple())
            impl = binary_op_noderivs<float, Vec3, float, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_float())
            impl = binary_op_noderivs<Vec3, float, float, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_triple())
            impl = binary_op_noderivs<Vec3, Vec3, float, FUNCTION>;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        exec->error_arg_types ();
        ASSERT (0 && "Function arg type can't be handled");
    }
}

template <typename FUNCTION>
DECLOP (generic_noise_function)
{
    ASSERT (nargs == 2 || nargs == 3);
    OpImpl impl = NULL;
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));

    // type check first args
    ASSERT (Result.typespec().is_float() || Result.typespec().is_triple());
    ASSERT (A.typespec().is_float() || A.typespec().is_triple());

    if (nargs == 2) {
        // either ff or fp
        if (Result.typespec().is_float() && A.typespec().is_float())
            impl = unary_op<float, float, FUNCTION>;
        else if (Result.typespec().is_float() && A.typespec().is_triple())
            impl = unary_op<float, Vec3, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_float())
            impl = unary_op<Vec3, float, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_triple())
            impl = unary_op<Vec3, Vec3, FUNCTION>;
    } else if (nargs == 3) {
        // either fff or fpf
        Symbol &B (exec->sym (args[2]));
        ASSERT (B.typespec().is_float());
        if (Result.typespec().is_float() && A.typespec().is_float())
            impl = binary_op<float, float, float, FUNCTION>;
        else if (Result.typespec().is_float() && A.typespec().is_triple())
            impl = binary_op<float, Vec3, float, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_float())
            impl = binary_op<Vec3, float, float, FUNCTION>;
        else if (Result.typespec().is_triple() && A.typespec().is_triple())
            impl = binary_op<Vec3, Vec3, float, FUNCTION>;
    }

    if (impl) {
        impl (exec, nargs, args, runflags, beginpoint, endpoint);
        // Use the specialized one for next time!  Never have to check the
        // types or do the other sanity checks again.
        // FIXME -- is this thread-safe?
        exec->op().implementation (impl);
    } else {
        exec->error_arg_types ();
        ASSERT (0 && "Function arg type can't be handled");
    }
}

DECLOP (OP_cellnoise)
{
    // NOTE: cellnoise is a step function which is locally flat
    //       therefore its derivatives are always 0
    generic_noise_function_noderivs<CellNoise> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}



DECLOP (OP_noise)
{
    generic_noise_function<Noise> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}



DECLOP (OP_snoise)
{
    generic_noise_function<SNoise> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
