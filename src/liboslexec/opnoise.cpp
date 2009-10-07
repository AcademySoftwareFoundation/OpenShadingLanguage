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
#include "OpenImageIO/fmath.h"


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

inline float fade (float t) { 
   return t * t * t * (t * (t * 6 - 15) + 10); 
}

// FIXME: original perlin code was written assuming this order of args to lerp
template <typename T>
T lerp(float x, const T &a, const T &b) {
    return Imath::lerp(a, b, x);
}

// 1,2,3 and 4 dimensional gradient functions - perform a dot product against a
// randomly chosen edge vector of the hypercube, when the number of edges is not
// exactly a power of 2 (such as in dimension 3), replicate the edges to avoid
// an expensive mod operation.

inline float grad (int hash, float x) {
    static const float G1[2] = { -1, 1 };

    int h = hash & 0x1;
    return x * G1[h];
}

inline float grad (int hash, float x, float y) {
    static const float G2[4][2] = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };

    int h = hash & 0x3;
    return x * G2[h][0] + y * G2[h][1];
}
 
inline float grad (int hash, float x, float y, float z) {
    static const float G3[16][3] = {
            { 1, 1, 0 }, { -1, 1, 0 }, { 1, -1, 0 }, { -1, -1, 0 }, { 1, 0, 1 }, { -1, 0, 1 },
            { 1, 0, -1 }, { -1, 0, -1 }, { 0, 1, 1 }, { 0, -1, 1 }, { 0, 1, -1 }, { 0, -1, -1 },
            { 1, 1, 0 }, { -1, 1, 0 }, { 0, -1, 1 }, { 0, -1, -1 } };

    int h = hash & 15;
    return x * G3[h][0] + y * G3[h][1] + z * G3[h][2];
}
 
inline float grad (int hash, float x, float y, float z, float w) {
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

inline Vec3 grad (const Vec3i &hash, float x, float y) {
    return Vec3 (grad (hash.x, x, y),
                 grad (hash.y, x, y),
                 grad (hash.z, x, y));
}

inline Vec3 grad (const Vec3i &hash, float x, float y, float z) {
    return Vec3 (grad (hash.x, x, y, z),
                 grad (hash.y, x, y, z),
                 grad (hash.z, x, y, z));
}

inline Vec3 grad (const Vec3i &hash, float x, float y, float z, float w) {
    return Vec3 (grad (hash.x, x, y, z, w),
                 grad (hash.y, x, y, z, w),
                 grad (hash.z, x, y, z, w));
}

template <typename V, typename H>
inline void perlin (V& result, H& hash, float x) {
    int X; x = floorfrac(x, &X);
    float u = fade(x);

    result = lerp (u, grad (hash (X  ), x  ),
                      grad (hash (X+1), x-1));
}

template <typename V, typename H>
inline void perlin (V &result, const H &hash, float x, float y) {
    int X; x = floorfrac(x, &X);
    int Y; y = floorfrac(y, &Y);

    float u = fade(x);
    float v = fade(y);

    result = lerp (v, lerp (u, grad (hash (X  , Y  ), x  , y  ),
                               grad (hash (X+1, Y  ), x-1, y  )),
                      lerp (u, grad (hash (X  , Y+1), x  , y-1),
                               grad (hash (X+1, Y+1), x-1, y-1)));
}


template <typename V, typename H>
inline void perlin (V &result, const H &hash, float x, float y, float z) {
    int X; x = floorfrac(x, &X);
    int Y; y = floorfrac(y, &Y);
    int Z; z = floorfrac(z, &Z);

    float u = fade(x);
    float v = fade(y);
    float w = fade(z);

    result = lerp (w, lerp (v, lerp (u, grad (hash (X  , Y  , Z  ), x  , y  , z   ),
                                        grad (hash (X+1, Y  , Z  ), x-1, y  , z   )),
                               lerp (u, grad (hash (X  , Y+1, Z  ), x  , y-1, z   ),
                                        grad (hash (X+1, Y+1, Z  ), x-1, y-1, z   ))),
                      lerp (v, lerp (u, grad (hash (X  , Y  , Z+1), x  , y  , z-1 ),
                                        grad (hash (X+1, Y  , Z+1), x-1, y  , z-1 )),
                               lerp (u, grad (hash (X  , Y+1, Z+1), x  , y-1, z-1 ),
                                        grad (hash (X+1, Y+1, Z+1), x-1, y-1, z-1 ))));
}

template <typename V, typename H>
inline void perlin (V &result, const H &hash, float x, float y, float z, float w) {
    int X; x = floorfrac(x, &X);
    int Y; y = floorfrac(y, &Y);
    int Z; z = floorfrac(z, &Z);
    int W; w = floorfrac(w, &W);

    float u = fade(x);
    float v = fade(y);
    float t = fade(z);
    float s = fade(w);

    result = lerp (s, lerp (t, lerp (v, lerp (u, grad (hash (X  , Y  , Z  , W  ), x  , y  , z  , w  ),
                                                 grad (hash (X+1, Y  , Z  , W  ), x-1, y  , z  , w  )),
                                        lerp (u, grad (hash (X  , Y+1, Z  , W  ), x  , y-1, z  , w  ),
                                                 grad (hash (X+1, Y+1, Z  , W  ), x-1, y-1, z  , w  ))),
                               lerp (v, lerp (u, grad (hash (X  , Y  , Z+1, W  ), x  , y  , z-1, w  ),
                                                 grad (hash (X+1, Y  , Z+1, W  ), x-1, y  , z-1, w  )),
                                        lerp (u, grad (hash (X  , Y+1, Z+1, W  ), x  , y-1, z-1, w  ),
                                                 grad (hash (X+1, Y+1, Z+1, W  ), x-1, y-1, z-1, w  )))),
                      lerp (t, lerp (v, lerp (u, grad (hash (X  , Y  , Z  , W+1), x  , y  , z  , w-1),
                                                 grad (hash (X+1, Y  , Z  , W+1), x-1, y  , z  , w-1)),
                                        lerp (u, grad (hash (X  , Y+1, Z  , W+1), x  , y-1, z  , w-1),
                                                 grad (hash (X+1, Y+1, Z  , W+1), x-1, y-1, z  , w-1))),
                               lerp (v, lerp (u, grad (hash (X  , Y  , Z+1, W+1), x  , y  , z-1, w-1),
                                                 grad (hash (X+1, Y  , Z+1, W+1), x-1, y  , z-1, w-1)),
                                        lerp (u, grad (hash (X  , Y+1, Z+1, W+1), x  , y-1, z-1, w-1),
                                                 grad (hash (X+1, Y+1, Z+1, W+1), x-1, y-1, z-1, w-1)))));

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
        unsigned int iv[2];
        iv[0] = x;
        return hash3<2> (iv);
    }

    Vec3i operator() (int x, int y) const {
        unsigned int iv[3];
        iv[0] = x;
        iv[1] = y;
        return hash3<3> (iv);
    }

    Vec3i operator() (int x, int y, int z) const {
        unsigned int iv[4];
        iv[0] = x;
        iv[1] = y;
        iv[2] = z;
        return hash3<4> (iv);
    }

    Vec3i operator() (int x, int y, int z, int w) const {
        unsigned int iv[5];
        iv[0] = x;
        iv[1] = y;
        iv[2] = z;
        iv[3] = w;
        return hash3<5> (iv);
    }

    template <int N>
    Vec3i hash3 (unsigned int k[N]) const {
        Vec3i result;
        k[N-1] = 0; result.x = inthash<N> (k); 
        k[N-1] = 1; result.y = inthash<N> (k);
        k[N-1] = 2; result.z = inthash<N> (k);
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
        unsigned int iv[2];
        iv[0] = imod (x, px);
        return hash3<2> (iv);
    }

    Vec3i operator() (int x, int y) const {
        unsigned int iv[3];
        iv[0] = imod (x, px);
        iv[1] = imod (y, py);
        return hash3<3> (iv);
    }

    Vec3i operator() (int x, int y, int z) const {
        unsigned int iv[4];
        iv[0] = imod (x, px);
        iv[1] = imod (y, py);
        iv[2] = imod (z, pz);
        return hash3<4> (iv);

    }

    Vec3i operator() (int x, int y, int z, int w) const {
        unsigned int iv[5];
        iv[0] = imod (x, px);
        iv[1] = imod (y, py);
        iv[2] = imod (z, pz);
        iv[3] = imod (w, pw);
        return hash3<5> (iv);
    }

    template <int N>
    Vec3i hash3 (unsigned int k[N]) const {
        Vec3i result;
        k[N-1] = 0; result.x = inthash<N> (k); 
        k[N-1] = 1; result.y = inthash<N> (k);
        k[N-1] = 2; result.z = inthash<N> (k);
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



DECLOP (OP_cellnoise)
{
    // NOTE: cellnoise is a step function which is locally flat
    //       therefore its derivatives are always 0
    generic_noise_function_noderivs<CellNoise> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}



DECLOP (OP_noise)
{
    // FIXME: we _do_ want to compute accurate derivatives here
    generic_noise_function_noderivs<Noise> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}



DECLOP (OP_snoise)
{
    // FIXME: we _do_ want to compute accurate derivatives here
    generic_noise_function_noderivs<SNoise> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
