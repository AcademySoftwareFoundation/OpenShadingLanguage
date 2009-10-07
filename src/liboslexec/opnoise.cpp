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

class CellNoise {
public:
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

} // anonymous namespace



DECLOP (OP_cellnoise)
{
    ASSERT (nargs == 2 || nargs == 3);
    OpImpl impl = NULL;
    Symbol &Result (exec->sym (args[0]));
    Symbol &A (exec->sym (args[1]));

    // type check first args
    ASSERT (Result.typespec().is_float() || Result.typespec().is_triple());
    ASSERT (A.typespec().is_float() || A.typespec().is_triple());

    // NOTE: cellnoise is a step function which is locally flat
    //       therefore its derivatives are always 0

    if (nargs == 2) {
        // either ff or fp
        if (Result.typespec().is_float() && A.typespec().is_float())
            impl = unary_op_noderivs<float, float, CellNoise>;
        else if (Result.typespec().is_float() && A.typespec().is_triple())
            impl = unary_op_noderivs<float, Vec3, CellNoise>;
        else if (Result.typespec().is_triple() && A.typespec().is_float())
            impl = unary_op_noderivs<Vec3, float, CellNoise>;
        else if (Result.typespec().is_triple() && A.typespec().is_triple())
            impl = unary_op_noderivs<Vec3, Vec3, CellNoise>;
    } else if (nargs == 3) {
        // either fff or fpf
        Symbol &B (exec->sym (args[2]));
        ASSERT (B.typespec().is_float());
        if (Result.typespec().is_float() && A.typespec().is_float())
            impl = binary_op_noderivs<float, float, float, CellNoise>;
        else if (Result.typespec().is_float() && A.typespec().is_triple())
            impl = binary_op_noderivs<float, Vec3, float, CellNoise>;
        else if (Result.typespec().is_triple() && A.typespec().is_float())
            impl = binary_op_noderivs<Vec3, float, float, CellNoise>;
        else if (Result.typespec().is_triple() && A.typespec().is_triple())
            impl = binary_op_noderivs<Vec3, Vec3, float, CellNoise>;
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

}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
