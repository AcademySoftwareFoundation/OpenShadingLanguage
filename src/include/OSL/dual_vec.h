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

OSL_NAMESPACE_ENTER


/// Templated trick to be able to derive what type we use to represent
/// a vector, given a scalar, automatically using the right kind of Dual.
template<class T> struct Vec3FromScalar { typedef Imath::Vec3<T> type; };
template<class T, int P> struct Vec3FromScalar<Dual<T,P>> { typedef Dual<Imath::Vec3<T>,P> type; };

/// Templated trick to be able to derive what type we use to represent
/// a color, given a scalar, automatically using the right kind of Dual2.
template<class T> struct Color3FromScalar { typedef Imath::Color3<T> type; };
template<class T, int P> struct Color3FromScalar<Dual<T,P>> { typedef Dual<Imath::Color3<T>,P> type; };

/// Templated trick to be able to derive the scalar component type of
/// a vector, whether a VecN or a Dual2<VecN>.
template<class T> struct ScalarFromVec {};
template<> struct ScalarFromVec<Vec2> { typedef Float type; };
template<> struct ScalarFromVec<Vec3> { typedef Float type; };
template<> struct ScalarFromVec<Color3> { typedef Float type; };
template<> struct ScalarFromVec<Dual<Vec2>> { typedef Dual<Float> type; };
template<> struct ScalarFromVec<Dual<Vec3>> { typedef Dual<Float> type; };
template<> struct ScalarFromVec<Dual<Color3>> { typedef Dual<Float> type; };
template<> struct ScalarFromVec<Dual2<Vec2>> { typedef Dual2<Float> type; };
template<> struct ScalarFromVec<Dual2<Vec3>> { typedef Dual2<Float> type; };
template<> struct ScalarFromVec<Dual2<Color3>> { typedef Dual2<Float> type; };



/// A uniform way to assemble a Vec3 from float and a Dual<Vec3>
/// from Dual<float>.
OSL_HOSTDEVICE inline Vec3
make_Vec3 (float x, float y, float z)
{
    return Vec3 (x, y, z);
}

template<class T, int P>
OSL_HOSTDEVICE inline Dual<Imath::Vec3<T>,P>
make_Vec3 (const Dual<T,P> &x, const Dual<T,P> &y, const Dual<T,P> &z)
{
    Dual<Imath::Vec3<T>,P> result;
    for (int i = 0; i <= P; ++i)
        result.elem(i).setValue (x.elem(i), y.elem(i), z.elem(i));
    return result;
}


/// Make a Dual<Vec3> from a single Dual<Float> x coordinate, and 0
/// for the other components.
template<class T, int P>
OSL_HOSTDEVICE inline Dual<Imath::Vec3<T>,P>
make_Vec3 (const Dual<T,P> &x)
{
    Dual<Imath::Vec3<T>,P> result;
    for (int i = 0; i <= P; ++i)
        result.elem(i).setValue (x.elem(i), 0.0f, 0.0f);
    return result;
}


template<class T, int P>
OSL_HOSTDEVICE inline Dual<Imath::Vec3<T>,P>
make_Vec3 (const Dual<T,P> &x, const Dual<T,P> &y)
{
    Dual<Imath::Vec3<T>,P> result;
    for (int i = 0; i <= P; ++i)
        result.elem(i).setValue (x.elem(i), y.elem(i), 0.0f);
    return result;
}



/// A uniform way to assemble a Color3 from float and a Dual<Color3>
/// from Dual<float>.
OSL_HOSTDEVICE inline Color3
make_Color3 (float x, float y, float z)
{
    return Color3 (x, y, z);
}

template<class T, int P>
OSL_HOSTDEVICE inline Dual<Imath::Color3<T>,P>
make_Color3 (const Dual<T,P> &x, const Dual<T,P> &y, const Dual<T,P> &z)
{
    Dual<Imath::Color3<T>,P> result;
    for (int i = 0; i <= P; ++i)
        result.elem(i).setValue (x.elem(i), y.elem(i), z.elem(i));
    return result;
}



/// A uniform way to assemble a Vec2 from float and a Dual<Vec2>
/// from Dual<float>.
OSL_HOSTDEVICE inline Vec2
make_Vec2 (float x, float y)
{
    return Vec2 (x, y);
}

template<class T, int P>
OSL_HOSTDEVICE inline Dual<Imath::Vec2<T>,P>
make_Vec2 (const Dual<T,P> &x, const Dual<T,P> &y)
{
    Dual<Imath::Vec2<T>,P> result;
    for (int i = 0; i <= P; ++i)
        result.elem(i).setValue (x.elem(i), y.elem(i));
    return result;
}



/// comp(X,c) is a uniform way to extract a single component from a Vec3 or
/// Dual<Vec3>.
///
/// comp(Vec3,c) returns a float as the c-th component of the vector.
/// comp(Dual<Vec3>,c) returns a Dual<float> of the c-th component (with
/// derivs).

OSL_HOSTDEVICE inline float
comp (const Vec3 &v, int c)
{
    return v[c];
}


template<class T, int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<T,P>
comp (const Dual<Imath::Vec3<T>,P> &v, int c)
{
    Dual<T,P> result;
    for (int i = 0; i <= P; ++i)
        result.elem(i) = v.elem(i)[c];
    return result;
}



template<class T, int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<T,P>
comp (const Dual<Imath::Color3<T>,P> &v, int c)
{
    Dual<T,P> result;
    for (int i = 0; i <= P; ++i)
        result.elem(i) = v.elem(i)[c];
    return result;
}


OSL_HOSTDEVICE inline float
comp (const Vec2 &v, int c)
{
    return v[c];
}


template<class T, int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<T,P>
comp (const Dual<Imath::Vec2<T>,P> &v, int c)
{
    Dual<T,P> result;
    for (int i = 0; i <= P; ++i)
        result.elem(i) = v.elem(i)[c];
    return result;
}




/// Multiply a 3x3 matrix by a 3-vector, with derivs.
///
template <class S, class T, int P>
OSL_HOSTDEVICE inline void
multMatrix (const Imath::Matrix33<T> &M, const Dual<Imath::Vec3<S>,P> &src,
            Dual<Imath::Vec3<S>,P> &dst)
{
    // The simplest way to express this is to break up the Dual<Vec> into
    // Vec<Dual>, do the usual matrix math, then reshuffle again.
    Dual<S,P> src0 = comp(src,0), src1 = comp(src,1), src2 = comp(src,2);
    Dual<S,P> a = src0 * M[0][0] + src1 * M[1][0] + src2 * M[2][0];
    Dual<S,P> b = src0 * M[0][1] + src1 * M[1][1] + src2 * M[2][1];
    Dual<S,P> c = src0 * M[0][2] + src1 * M[1][2] + src2 * M[2][2];
    dst = make_Vec3 (a, b, c);
}


/// Multiply a row 3-vector (with derivatives) by a 3x3 matrix (no derivs).
///
template <class S, class T, int P> OSL_HOSTDEVICE inline OIIO_CONSTEXPR14
Dual<Imath::Vec3<S>,P>
operator* (const Dual<Imath::Vec3<S>,P> &src, const Imath::Matrix33<T> &M)
{
    // The simplest way to express this is to break up the Dual<Vec> into
    // Vec<Dual>, do the usual matrix math, then reshuffle again.
    Dual<S,P> src0 = comp(src,0), src1 = comp(src,1), src2 = comp(src,2);
    Dual<S,P> a = src0 * M[0][0] + src1 * M[1][0] + src2 * M[2][0];
    Dual<S,P> b = src0 * M[0][1] + src1 * M[1][1] + src2 * M[2][1];
    Dual<S,P> c = src0 * M[0][2] + src1 * M[1][2] + src2 * M[2][2];
    return make_Vec3 (a, b, c);
}


/// Multiply a row 3-vector (with derivatives) by a 3x3 matrix (no derivs).
///
template <class S, class T, int P> OSL_HOSTDEVICE inline OIIO_CONSTEXPR14
Dual<Imath::Color3<S>,P>
operator* (const Dual<Imath::Color3<S>,P> &src, const Imath::Matrix33<T> &M)
{
    // The simplest way to express this is to break up the Dual<Vec> into
    // Vec<Dual>, do the usual matrix math, then reshuffle again.
    Dual<S,P> src0 = comp(src,0), src1 = comp(src,1), src2 = comp(src,2);
    Dual<S,P> a = src0 * M[0][0] + src1 * M[1][0] + src2 * M[2][0];
    Dual<S,P> b = src0 * M[0][1] + src1 * M[1][1] + src2 * M[2][1];
    Dual<S,P> c = src0 * M[0][2] + src1 * M[1][2] + src2 * M[2][2];
    return make_Color3 (a, b, c);
}


template <class S, class T>
OSL_HOSTDEVICE inline void
robust_multVecMatrix(const Imath::Matrix44<S>& x, const Imath::Vec3<T>& src, Imath::Vec3<T>& dst)
{
    auto a = src[0] * x[0][0] + src[1] * x[1][0] + src[2] * x[2][0] + x[3][0];
    auto b = src[0] * x[0][1] + src[1] * x[1][1] + src[2] * x[2][1] + x[3][1];
    auto c = src[0] * x[0][2] + src[1] * x[1][2] + src[2] * x[2][2] + x[3][2];
    auto w = src[0] * x[0][3] + src[1] * x[1][3] + src[2] * x[2][3] + x[3][3];

    if (! equalVal (w, T(0))) {
        dst.x = a / w;
        dst.y = b / w;
        dst.z = c / w;
    } else {
        dst.x = 0;
        dst.y = 0;
        dst.z = 0;
    }
}


/// Multiply a matrix times a vector with derivatives to obtain
/// a transformed vector with derivatives.
template <class S, class T, int P>
OSL_HOSTDEVICE inline void
robust_multVecMatrix (const Imath::Matrix44<S> &M,
                      const Dual<Imath::Vec3<T>,P> &in, Dual<Imath::Vec3<T>,P> &out)
{
    // Rearrange into a Vec3<Dual<float>>
    Imath::Vec3<Dual<T,P>> din, dout;
    for (int i = 0;  i < 3;  ++i) {
        din[i] = comp (in, i);
    }

    auto a = din[0] * M[0][0] + din[1] * M[1][0] + din[2] * M[2][0] + M[3][0];
    auto b = din[0] * M[0][1] + din[1] * M[1][1] + din[2] * M[2][1] + M[3][1];
    auto c = din[0] * M[0][2] + din[1] * M[1][2] + din[2] * M[2][2] + M[3][2];
    auto w = din[0] * M[0][3] + din[1] * M[1][3] + din[2] * M[2][3] + M[3][3];

    if (! equalVal (w, T(0))) {
       dout.x = a / w;
       dout.y = b / w;
       dout.z = c / w;
    } else {
       dout.x = T(0);
       dout.y = T(0);
       dout.z = T(0);
    }

    // Rearrange back into Dual<Vec3>
    out = make_Vec3 (dout[0], dout[1], dout[2]);
}

/// Multiply a matrix times a direction with derivatives to obtain
/// a transformed direction with derivatives.
template <class S, class T, int P>
OSL_HOSTDEVICE inline void
multDirMatrix (const Imath::Matrix44<S> &M,
               const Dual<Imath::Vec3<T>,P> &in, Dual<Imath::Vec3<T>,P> &out)
{
    for (int i = 0; i <= P; ++i)
        M.multDirMatrix (in.elem(i), out.elem(i));
}





template<class T, int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<T,P>
dot (const Dual<Imath::Vec3<T>,P> &a, const Dual<Imath::Vec3<T>,P> &b)
{
    auto ax = comp (a, 0);
    auto ay = comp (a, 1);
    auto az = comp (a, 2);
    auto bx = comp (b, 0);
    auto by = comp (b, 1);
    auto bz = comp (b, 2);
    return ax*bx + ay*by + az*bz;
}



template<class T, int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<T,P>
dot (const Dual<Imath::Vec3<T>,P> &a, const Imath::Vec3<T> &b)
{
    auto ax = comp (a, 0);
    auto ay = comp (a, 1);
    auto az = comp (a, 2);
    auto bx = comp (b, 0);
    auto by = comp (b, 1);
    auto bz = comp (b, 2);
    return ax*bx + ay*by + az*bz;
}



template<class T, int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<T,P>
dot (const Imath::Vec3<T> &a, const Dual<Imath::Vec3<T>,P> &b)
{
    return dot (b, a);
}



template<class T, int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<T,P>
dot (const Dual<Imath::Vec2<T>,P> &a, const Dual<Imath::Vec2<T>,P> &b)
{
    auto ax = comp (a, 0);
    auto ay = comp (a, 1);
    auto bx = comp (b, 0);
    auto by = comp (b, 1);
    return ax*bx + ay*by;
}



template<class T, int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<T,P>
dot (const Dual<Imath::Vec2<T>,P> &a, const Vec2 &b)
{
    auto ax = comp (a, 0);
    auto ay = comp (a, 1);
    auto bx = comp (b, 0);
    auto by = comp (b, 1);
    return ax*bx + ay*by;
}



template<class T, int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<T,P>
dot (const Vec2 &a, const Dual<Imath::Vec2<T>,P> &b)
{
    auto ax = comp (a, 0);
    auto ay = comp (a, 1);
    auto bx = comp (b, 0);
    auto by = comp (b, 1);
    return ax*bx + ay*by;
}



template<class T, int P>
OSL_HOSTDEVICE inline Dual<Imath::Vec3<T>,P>
cross (const Dual<Imath::Vec3<T>,P> &a, const Dual<Imath::Vec3<T>,P> &b)
{
    auto ax = comp (a, 0);
    auto ay = comp (a, 1);
    auto az = comp (a, 2);
    auto bx = comp (b, 0);
    auto by = comp (b, 1);
    auto bz = comp (b, 2);
    auto nx = ay*bz - az*by;
    auto ny = az*bx - ax*bz;
    auto nz = ax*by - ay*bx;
    return make_Vec3 (nx, ny, nz);
}



template<class T, int P>
OSL_HOSTDEVICE inline OIIO_CONSTEXPR14 Dual<T,P>
length (const Dual<Imath::Vec3<T>,P> &a)
{
    auto ax = comp (a, 0);
    auto ay = comp (a, 1);
    auto az = comp (a, 2);
    return sqrt(ax*ax + ay*ay + az*az);
}



template<class T, int P>
OSL_HOSTDEVICE inline Dual<Imath::Vec3<T>,P>
normalize (const Dual<Imath::Vec3<T>,P> &a)
{
    // NOTE: math must be consistent with osl_normalize_vv
    // TODO: math for derivative elements could be further optimized ...
    auto ax = comp (a, 0);
    auto ay = comp (a, 1);
    auto az = comp (a, 2);
    auto len = sqrt(ax * ax + ay * ay + az * az);
    if (len > T(0)) {
        auto invlen = T(1) / len;
        ax = ax * invlen;
        ay = ay * invlen;
        az = az * invlen;
        return make_Vec3 (ax, ay, az);
    } else {
        return Vec3(0,0,0);
    }
}



template<class T, int P>
OSL_HOSTDEVICE inline Dual<T,P>
distance (const Dual<Imath::Vec3<T>,P> &a, const Dual<Imath::Vec3<T>,P> &b)
{
    return length (a - b);
}


template<class T, int P>
OSL_HOSTDEVICE inline Dual<T,P>
distance (const Dual<Imath::Vec3<T>,P> &a, const Imath::Vec3<T> &b)
{
    return length (a - b);
}


template<class T, int P>
OSL_HOSTDEVICE inline Dual<T,P>
distance (const Imath::Vec3<T> &a, const Dual<Imath::Vec3<T>,P> &b)
{
    return length (a - b);
}



OSL_NAMESPACE_EXIT
