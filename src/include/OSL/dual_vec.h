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



/// \file
///
/// Dual<> extensions specifically for dealing with Imath Vec, Color, and
/// Matrix types.
///
/// In general, it's reasonable to handle a vector-with-derivs as a vector-
/// of-floats-with-derivs, i.e. Vec<Dual<float>>. But OSL's design chose
/// specifically require that any data with derivs has the same data layout
/// as without derivs, just repeated for the partials.
///
/// Thus, OSL represents vectors-with-derivs as Dual<Vec<float>>, NOT as
/// Vec<Dual<float>. So there are some special cases we need to deal with.
///


#pragma once

#include <OSL/oslconfig.h>
#include <OSL/dual.h>
#include "Imathx.h"

OSL_NAMESPACE_ENTER

#if 0 // appears unused
/// Templated trick to be able to derive what type we use to represent
/// a vector, given a scalar, automatically using the right kind of Dual.
template<class T> struct Vec3FromScalar { typedef Imath::Vec3<T> type; };
template<class T, int P> struct Vec3FromScalar<Dual<T,P>> { typedef Dual<Imath::Vec3<T>,P> type; };

/// Templated trick to be able to derive what type we use to represent
/// a color, given a scalar, automatically using the right kind of Dual2.
template<class T> struct Color3FromScalar { typedef Imath::Color3<T> type; };
template<class T, int P> struct Color3FromScalar<Dual<T,P>> { typedef Dual<Imath::Color3<T>,P> type; };
#endif

/// Templated trick to be able to derive the scalar component type of
/// a vector, whether a VecN or a Dual2<VecN>.
template<class T> struct ScalarFromVec { typedef typename T::BaseType type; };
template<class T, int P> struct ScalarFromVec<Dual<T,P>> { typedef Dual<typename T::BaseType,P> type; };


/// A uniform way to assemble a Vec3 from float and a Dual<Vec3>
/// from Dual<float>.
OSL_HOSTDEVICE inline Vec3
make_Vec3 (float x, float y, float z)
{
    return Vec3 (x, y, z);
}

template<int P>
OSL_HOSTDEVICE inline Dual<Vec3,P>
make_Vec3 (const Dual<Vec3::BaseType,P> &x, const Dual<Vec3::BaseType,P> &y, const Dual<Vec3::BaseType,P> &z)
{
    Dual<Vec3,P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i).setValue (x.elem(i), y.elem(i), z.elem(i));
    });
    return result;
}


/// Make a Dual<Vec3> from a single Dual<Float> x coordinate, and 0
/// for the other components.
template<int P>
OSL_HOSTDEVICE inline Dual<Vec3,P>
make_Vec3 (const Dual<Vec3::BaseType,P> &x)
{
    Dual<Vec3,P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i).setValue (x.elem(i), 0.0f, 0.0f);
    });
    return result;
}


template<int P>
OSL_HOSTDEVICE inline Dual<Vec3,P>
make_Vec3 (const Dual<Vec3::BaseType,P> &x, const Dual<Vec3::BaseType,P> &y)
{
    Dual<Vec3,P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i).setValue (x.elem(i), y.elem(i), 0.0f);
    });
    return result;
}


/// A uniform way to assemble a Color3 from float and a Dual<Color3>
/// from Dual<float>.
OSL_HOSTDEVICE inline Color3
make_Color3 (float x, float y, float z)
{
    return Color3 (x, y, z);
}

template<int P>
OSL_HOSTDEVICE inline Dual<Color3,P>
make_Color3 (const Dual<Color3::BaseType,P> &x, const Dual<Color3::BaseType,P> &y, const Dual<Color3::BaseType,P> &z)
{
    Dual<Color3, P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i).setValue (x.elem(i), y.elem(i), z.elem(i));
    });
    return result;
}

/// A uniform way to assemble a Vec2 from float and a Dual<Vec2>
/// from Dual<float>.
OSL_HOSTDEVICE inline Vec2
make_Vec2 (float x, float y)
{
    return Vec2 (x, y);
}

template<int P>
OSL_HOSTDEVICE inline Dual<Vec2,P>
make_Vec2 (const Dual<Vec2::BaseType,P> &x, const Dual<Vec2::BaseType,P> &y)
{
    Dual<Vec2,P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i).setValue (x.elem(i), y.elem(i));
    });
    return result;
}


/// Instead of index based access, explicitly use _x, _y, _z suffixes to
/// avoid Vec3::operator[] that uses non-conforming code creating aliasing issues
/// comp_x(X) comp_y(X) comp_z(X) is a uniform way to extract a single component from a Vec3 or
/// Dual<Vec3>.
///
/// comp_x(Vec3,c) returns a float as the x component of the vector.
/// comp_x(Dual<Vec3>) returns a Dual<float> of the x component (with
/// derivs).

OSL_HOSTDEVICE OSL_INLINE float
comp_x (const Vec3 &v)
{
    return v.x;
}

template<int P>
OSL_HOSTDEVICE OSL_INLINE OIIO_CONSTEXPR14 Dual<Vec3::BaseType, P>
comp_x (const Dual<Vec3,P> &v)
{
    Dual<Vec3::BaseType, P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i) = v.elem(i).x;
    });
    return result;
}

OSL_HOSTDEVICE OSL_INLINE float
comp_y (const Vec3 &v)
{
    return v.y;
}

template<int P>
OSL_HOSTDEVICE OSL_INLINE OIIO_CONSTEXPR14 Dual<Vec3::BaseType,P>
comp_y (const Dual<Vec3,P> &v)
{
    Dual<Vec3::BaseType,P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i) = v.elem(i).y;
    });
    return result;
}


OSL_HOSTDEVICE OSL_INLINE float
comp_z (const Vec3 &v)
{
    return v.z;
}

template<int P>
OSL_HOSTDEVICE OSL_INLINE OIIO_CONSTEXPR14 Dual<Vec3::BaseType,P>
comp_z (const Dual<Vec3,P> &v)
{
    Dual<Vec3::BaseType,P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i) = v.elem(i).z;
    });
    return result;
}

OSL_HOSTDEVICE OSL_INLINE float
comp_x (const Color3 &v)
{
    return v.x;
}

OSL_HOSTDEVICE OSL_INLINE float
comp_y (const Color3 &v)
{
    return v.y;
}

OSL_HOSTDEVICE OSL_INLINE float
comp_z (const Color3 &v)
{
    return v.z;
}

template<int P>
OSL_HOSTDEVICE OSL_INLINE OIIO_CONSTEXPR14 Dual<Color3::BaseType,P>
comp_x (const Dual<Color3,P> &v)
{
    Dual<Color3::BaseType, P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i) = v.elem(i).x;
    });
    return result;
}

template<int P>
OSL_HOSTDEVICE OSL_INLINE OIIO_CONSTEXPR14 Dual<Color3::BaseType,P>
comp_y (const Dual<Color3,P> &v)
{
    Dual<Color3::BaseType,P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i) = v.elem(i).y;
    });
    return result;
}

template<int P>
OSL_HOSTDEVICE OSL_INLINE OIIO_CONSTEXPR14 Dual<Color3::BaseType,P>
comp_z (const Dual<Color3,P> &v)
{
    Dual<Color3::BaseType,P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i) = v.elem(i).z;
    });
    return result;
}



OSL_HOSTDEVICE OSL_INLINE float
comp_x (const Vec2 &v)
{
    return v.x;
}

template<int P>
OSL_HOSTDEVICE OSL_INLINE OIIO_CONSTEXPR14 Dual<Vec2::BaseType,P>
comp_x (const Dual<Vec2,P> &v)
{
    Dual<Vec2::BaseType,P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i) = v.elem(i).x;
    });
    return result;
}


OSL_HOSTDEVICE OSL_INLINE float
comp_y (const Vec2 &v)
{
    return v.y;
}

template<int P>
OSL_HOSTDEVICE OSL_INLINE OIIO_CONSTEXPR14 Dual<Vec2::BaseType,P>
comp_y (const Dual<Vec2,P> &v)
{
    Dual<Vec2::BaseType,P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i) = v.elem(i).y;
    });
    return result;
}




/// Multiply a 3x3 matrix by a 3-vector, with derivs.
///
template <class T, int P>
OSL_HOSTDEVICE inline void
multMatrix (const Imath::Matrix33<T> &M, const Dual<Vec3,P> &src,
            Dual<Vec3,P> &dst)
{
    // The simplest way to express this is to break up the Dual<Vec> into
    // Vec<Dual>, do the usual matrix math, then reshuffle again.
    Dual<Vec3::BaseType,P> src0 = comp_x(src), src1 = comp_y(src), src2 = comp_z(src);
    Dual<Vec3::BaseType,P> a = src0 * M.x[0][0] + src1 * M.x[1][0] + src2 * M.x[2][0];
    Dual<Vec3::BaseType,P> b = src0 * M.x[0][1] + src1 * M.x[1][1] + src2 * M.x[2][1];
    Dual<Vec3::BaseType,P> c = src0 * M.x[0][2] + src1 * M.x[1][2] + src2 * M.x[2][2];
    dst = make_Vec3 (a, b, c);
}


/// Multiply a row 3-vector (with derivatives) by a 3x3 matrix (no derivs).
///
template <class T, int P> OSL_HOSTDEVICE inline OIIO_CONSTEXPR14
Dual<Vec3,P>
operator* (const Dual<Vec3,P> &src, const Imath::Matrix33<T> &M)
{
    // The simplest way to express this is to break up the Dual<Vec> into
    // Vec<Dual>, do the usual matrix math, then reshuffle again.
    Dual<Vec3::BaseType,P> src0 = comp_x(src), src1 = comp_y(src), src2 = comp_z(src);
    Dual<Vec3::BaseType,P> a = src0 * M[0][0] + src1 * M[1][0] + src2 * M[2][0];
    Dual<Vec3::BaseType,P> b = src0 * M[0][1] + src1 * M[1][1] + src2 * M[2][1];
    Dual<Vec3::BaseType,P> c = src0 * M[0][2] + src1 * M[1][2] + src2 * M[2][2];
    return make_Vec3 (a, b, c);
}


/// Multiply a row 3-vector (with derivatives) by a 3x3 matrix (no derivs).
///
template <class T, int P> OSL_HOSTDEVICE inline OIIO_CONSTEXPR14
Dual<Color3,P>
operator* (const Dual<Color3,P> &src, const Imath::Matrix33<T> &M)
{
    // The simplest way to express this is to break up the Dual<Vec> into
    // Vec<Dual>, do the usual matrix math, then reshuffle again.
    Dual<Color3::BaseType,P> src0 = comp_x(src), src1 = comp_y(src), src2 = comp_z(src);
    Dual<Color3::BaseType,P> a = src0 * M[0][0] + src1 * M[1][0] + src2 * M[2][0];
    Dual<Color3::BaseType,P> b = src0 * M[0][1] + src1 * M[1][1] + src2 * M[2][1];
    Dual<Color3::BaseType,P> c = src0 * M[0][2] + src1 * M[1][2] + src2 * M[2][2];
    return make_Color3 (a, b, c);
}

template <class S>
OSL_HOSTDEVICE inline void
robust_multVecMatrix(const Imath::Matrix44<S>& M, const Vec3& src, Vec3& dst)
{
    auto a = src.x * M.x[0][0] + src.y * M.x[1][0] + src.z * M.x[2][0] + M.x[3][0];
    auto b = src.x * M.x[0][1] + src.y * M.x[1][1] + src.z * M.x[2][1] + M.x[3][1];
    auto c = src.x * M.x[0][2] + src.y * M.x[1][2] + src.z * M.x[2][2] + M.x[3][2];
    auto w = src.x * M.x[0][3] + src.y * M.x[1][3] + src.z * M.x[2][3] + M.x[3][3];

    if (OSL_EXPECT_TRUE(! equalVal (w, Vec3::BaseType(0)))) {
        dst.x = a / w;
        dst.y = b / w;
        dst.z = c / w;
    } else {
        dst.x = Vec3::BaseType(0);
        dst.y = Vec3::BaseType(0);
        dst.z = Vec3::BaseType(0);
    }
}


/// Multiply a matrix times a vector with derivatives to obtain
/// a transformed vector with derivatives.
template <class S, int P>
OSL_HOSTDEVICE inline void
robust_multVecMatrix (const Imath::Matrix44<S> &M,
                      const Dual<Vec3,P> &in, Dual<Vec3,P> &out)
{
    // Rearrange into a Vec3<Dual<float>>
    // Avoid aliasing issues by not using Vec3::operator[]
    Imath::Vec3<Dual<Vec3::BaseType,P>> din(comp_x(in), comp_y(in), comp_z(in)), dout;

    auto a = din.x * M.x[0][0] + din.y * M.x[1][0] + din.z * M.x[2][0] + M.x[3][0];
    auto b = din.x * M.x[0][1] + din.y * M.x[1][1] + din.z * M.x[2][1] + M.x[3][1];
    auto c = din.x * M.x[0][2] + din.y * M.x[1][2] + din.z * M.x[2][2] + M.x[3][2];
    auto w = din.x * M.x[0][3] + din.y * M.x[1][3] + din.z * M.x[2][3] + M.x[3][3];

    if (OSL_EXPECT_TRUE(!equalVal (w, Vec3::BaseType(0)))) {
       dout.x = a / w;
       dout.y = b / w;
       dout.z = c / w;
    } else {
       dout.x = Vec3::BaseType(0);
       dout.y = Vec3::BaseType(0);
       dout.z = Vec3::BaseType(0);
    }

    // Rearrange back into Dual<Vec3>
    out = make_Vec3 (dout.x, dout.y, dout.z);
}

/// Multiply a matrix times a direction with derivatives to obtain
/// a transformed direction with derivatives.
template <class S, int P>
OSL_HOSTDEVICE inline void
multDirMatrix (const Imath::Matrix44<S> &M,
               const Dual<Vec3,P> &in, Dual<Vec3,P> &out)
{
    OSL_INDEX_LOOP(i, P+1, {
        M.multDirMatrix (in.elem(i), out.elem(i));
    });
}

// Return value version multDirMatrix
template <class S, int P>
OSL_HOSTDEVICE inline Dual<Vec3,P>
multiplyDirByMatrix (const Imath::Matrix44<S> &M,
               const Dual<Vec3,P> &in)
{
    Dual<Vec3,P> out;
    OSL_INDEX_LOOP(i, P+1, {
        out.elem(i) = multiplyDirByMatrix(M, in.elem(i));
    });
    return out;
}

template<int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<Vec3::BaseType, P>
dot (const Dual<Vec3,P> &a, const Dual<Vec3,P> &b)
{
    auto ax = comp_x (a);
    auto ay = comp_y (a);
    auto az = comp_z (a);
    auto bx = comp_x (b);
    auto by = comp_y (b);
    auto bz = comp_z (b);
    return ax*bx + ay*by + az*bz;
}



template<int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<Vec3::BaseType,P>
dot (const Dual<Vec3,P> &a, const Vec3 &b)
{
    auto ax = comp_x (a);
    auto ay = comp_y (a);
    auto az = comp_z (a);
    auto bx = comp_x (b);
    auto by = comp_y (b);
    auto bz = comp_z (b);
    return ax*bx + ay*by + az*bz;
}



template<int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<Vec3::BaseType,P>
dot (const Vec3 &a, const Dual<Vec3,P> &b)
{
    return dot (b, a);
}



template<int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<Vec2::BaseType,P>
dot (const Dual<Vec2,P> &a, const Dual<Vec2,P> &b)
{
    auto ax = comp_x (a);
    auto ay = comp_y (a);
    auto bx = comp_x (b);
    auto by = comp_y (b);
    return ax*bx + ay*by;
}



template<int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<Vec2::BaseType,P>
dot (const Dual<Vec2,P> &a, const Vec2 &b)
{
    auto ax = comp_x (a);
    auto ay = comp_y (a);
    auto bx = comp_x (b);
    auto by = comp_y (b);
    return ax*bx + ay*by;
}



template<int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<Vec2::BaseType,P>
dot (const Vec2 &a, const Dual<Vec2,P> &b)
{
    auto ax = comp_x (a);
    auto ay = comp_y (a);
    auto bx = comp_x (b);
    auto by = comp_y (b);
    return ax*bx + ay*by;
}



template<int P>
OSL_HOSTDEVICE inline Dual<Vec3,P>
cross (const Dual<Vec3,P> &a, const Dual<Vec3,P> &b)
{
    auto ax = comp_x (a);
    auto ay = comp_y (a);
    auto az = comp_z (a);
    auto bx = comp_x (b);
    auto by = comp_y (b);
    auto bz = comp_z (b);
    auto nx = ay*bz - az*by;
    auto ny = az*bx - ax*bz;
    auto nz = ax*by - ay*bx;
    return make_Vec3 (nx, ny, nz);
}



template<int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<Vec3::BaseType,P>
length (const Dual<Vec3,P> &a)
{
    auto ax = comp_x (a);
    auto ay = comp_y (a);
    auto az = comp_z (a);
    return sqrt(ax*ax + ay*ay + az*az);
}



template<int P>
OSL_HOSTDEVICE inline Dual<Vec3,P>
normalize (const Dual<Vec3,P> &a)
{
    auto ax = comp_x (a);
    auto ay = comp_y (a);
    auto az = comp_z (a);
    auto len = sqrt(ax * ax + ay * ay + az * az);
    if (OSL_EXPECT_TRUE(len > Vec3::BaseType(0))) {
        auto invlen = Vec3::BaseType(1) / len;
        auto nax = ax * invlen;
        auto nay = ay * invlen;
        auto naz = az * invlen;
        return make_Vec3 (nax, nay, naz);
    } else {
        return Vec3(0,0,0);
    }
}



template<int P>
OSL_HOSTDEVICE inline Dual<Vec3::BaseType,P>
distance (const Dual<Vec3,P> &a, const Dual<Vec3,P> &b)
{
    return length (a - b);
}


template<int P>
OSL_HOSTDEVICE inline Dual<Vec3::BaseType,P>
distance (const Dual<Vec3,P> &a, const Vec3 &b)
{
    return length (a - b);
}


template<int P>
OSL_HOSTDEVICE inline Dual<Vec3::BaseType,P>
distance (const Vec3 &a, const Dual<Vec3,P> &b)
{
    return length (a - b);
}

OSL_NAMESPACE_EXIT
