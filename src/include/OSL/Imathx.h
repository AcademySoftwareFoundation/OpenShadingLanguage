/*
Copyright (c) 2012 Sony Pictures Imageworks Inc., et al.
and
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


// Extensions to Imath classes for use in OSL's internals.
// 
// The original Imath classes bear the "new BSD" license (same as
// ours above) and this copyright:
// Copyright (c) 2002, Industrial Light & Magic, a division of
// Lucas Digital Ltd. LLC.  All rights reserved.


#pragma once

#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>
#include <OpenEXR/ImathColor.h>

#include "matrix22.h"


OSL_NAMESPACE_ENTER


/// 3x3 matrix transforming a 3-vector.  This is curiously not supplied
/// by Imath, so we define it ourselves.
template <class S, class T>
inline void
multMatrix (const Imath::Matrix33<T> &M, const Imath::Vec3<S> &src,
            Imath::Vec3<S> &dst)
{
    // Changed all Vec3 subscripts to access data members versus array casts
    S a = src.x * M.x[0][0] + src.y * M.x[1][0] + src.z * M.x[2][0];
    S b = src.x * M.x[0][1] + src.y * M.x[1][1] + src.z * M.x[2][1];
    S c = src.x * M.x[0][2] + src.y * M.x[1][2] + src.z * M.x[2][2];
    dst.x = a;
    dst.y = b;
    dst.z = c;
}


/// Express dot product as a function rather than a method.
template<class T>
inline T
dot (const Imath::Vec2<T> &a, const Imath::Vec2<T> &b)
{
    return a.dot (b);
}


/// Express dot product as a function rather than a method.
template<class T>
inline T
dot (const Imath::Vec3<T> &a, const Imath::Vec3<T> &b)
{
    return a.dot (b);
}



/// Return the determinant of a 2x2 matrix.
template <class T>
inline
T determinant (const Imathx::Matrix22<T> &M)
{
    return M.x[0][0]*M.x[1][1] - M.x[0][1]*M.x[1][0];
}

// Imath::Vec3::lengthTiny is private
// local copy here no changes
OSL_INLINE float accessibleTinyLength(const Vec3 &N)
{
    float absX = (N.x >= float (0))? N.x: -N.x;
    float absY = (N.y >= float (0))? N.y: -N.y;
    float absZ = (N.z >= float (0))? N.z: -N.z;

    float max = absX;

    if (max < absY)
	max = absY;

    if (max < absZ)
	max = absZ;

    if (max == float (0))
	return float (0);

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

// because lengthTiny does alot of work including another
// sqrt, we really want to skip that if possible because
// with SIMD execution, we end up doing the sqrt twice
// and blending the results.  Although code could be
// refactored to do a single sqrt, think its better
// to skip the code block as we don't expect near 0 lengths
// TODO: get OpenEXR ImathVec to update to similar, don't think
// it can cause harm
OSL_INLINE
float simdFriendlyLength(const Vec3 &N)
{
	float length2 = N.dot (N);

	if (__builtin_expect(length2 < float (2) * Imath::limits<float>::smallest(), 0))
		return accessibleTinyLength(N);

	return Imath::Math<float>::sqrt (length2);
}

OSL_INLINE Vec3
simdFriendlyNormalize(const Vec3 &N)
{
    float l = simdFriendlyLength(N);

    if (l == float (0))
    	return Vec3 (float (0));

    return Vec3 (N.x / l, N.y / l, N.z / l);
}


// flatten is workaround to enable inlining of non-inlined methods
static OSL_INLINE OSL_CLANG_ATTRIBUTE(flatten) Matrix44
affineInvert(const Matrix44 &m)
{
    //assert(__builtin_expect(m.x[0][3] == 0.0f && m.x[1][3] == 0.0f && m.x[2][3] == 0.0f && m.x[3][3] == 1.0f, 1))
	Matrix44 s (m.x[1][1] * m.x[2][2] - m.x[2][1] * m.x[1][2],
				m.x[2][1] * m.x[0][2] - m.x[0][1] * m.x[2][2],
				m.x[0][1] * m.x[1][2] - m.x[1][1] * m.x[0][2],
				0.0f,

				m.x[2][0] * m.x[1][2] - m.x[1][0] * m.x[2][2],
				m.x[0][0] * m.x[2][2] - m.x[2][0] * m.x[0][2],
				m.x[1][0] * m.x[0][2] - m.x[0][0] * m.x[1][2],
				0.0f,

				m.x[1][0] * m.x[2][1] - m.x[2][0] * m.x[1][1],
				m.x[2][0] * m.x[0][1] - m.x[0][0] * m.x[2][1],
				m.x[0][0] * m.x[1][1] - m.x[1][0] * m.x[0][1],
				0.0f,

				0.0f,
				0.0f,
				0.0f,
				1.0f);

	float r = m.x[0][0] * s.x[0][0] + m.x[0][1] * s.x[1][0] + m.x[0][2] * s.x[2][0];
	float abs_r = IMATH_INTERNAL_NAMESPACE::abs (r);


	int may_have_divided_by_zero = 0;
	if (__builtin_expect(abs_r < 1.0f, 0))
	{
		float mr = abs_r / Imath::limits<float>::smallest();
		OSL_INTEL_PRAGMA(unroll)
		for (int i = 0; i < 3; ++i)
		{
			OSL_INTEL_PRAGMA(unroll)
			for (int j = 0; j < 3; ++j)
			{
				if (mr <= IMATH_INTERNAL_NAMESPACE::abs (s.x[i][j]))
				{
					may_have_divided_by_zero = 1;
				}
			}
		}
	}

	OSL_INTEL_PRAGMA(unroll)
	for (int i = 0; i < 3; ++i)
	{
		OSL_INTEL_PRAGMA(unroll)
		for (int j = 0; j < 3; ++j)
		{
			s.x[i][j] /= r;
		}
	}

	s.x[3][0] = -m.x[3][0] * s.x[0][0] - m.x[3][1] * s.x[1][0] - m.x[3][2] * s.x[2][0];
	s.x[3][1] = -m.x[3][0] * s.x[0][1] - m.x[3][1] * s.x[1][1] - m.x[3][2] * s.x[2][1];
	s.x[3][2] = -m.x[3][0] * s.x[0][2] - m.x[3][1] * s.x[1][2] - m.x[3][2] * s.x[2][2];

	if (__builtin_expect(may_have_divided_by_zero == 1, 0))
	{
		s = Matrix44();
	}
	return s;
}

// In order to have inlinable Matrix44*float
// Override with a more specific version than
// template <class T>
// inline Matrix44<T>
// operator * (T a, const Matrix44<T> &v);

OSL_INLINE Matrix44
operator * (float a, const Matrix44 &v)
{
    return Matrix44 (v.x[0][0] * a,
                     v.x[0][1] * a,
                     v.x[0][2] * a,
                     v.x[0][3] * a,
                     v.x[1][0] * a,
                     v.x[1][1] * a,
                     v.x[1][2] * a,
                     v.x[1][3] * a,
                     v.x[2][0] * a,
                     v.x[2][1] * a,
                     v.x[2][2] * a,
                     v.x[2][3] * a,
                     v.x[3][0] * a,
                     v.x[3][1] * a,
                     v.x[3][2] * a,
                     v.x[3][3] * a);
}

OSL_INLINE Matrix44
inlinedTransposed (const Matrix44 &m)
{
    return Matrix44 (m.x[0][0],
                     m.x[1][0],
                     m.x[2][0],
                     m.x[3][0],
                     m.x[0][1],
                     m.x[1][1],
                     m.x[2][1],
                     m.x[3][1],
                     m.x[0][2],
                     m.x[1][2],
                     m.x[2][2],
                     m.x[3][2],
                     m.x[0][3],
                     m.x[1][3],
                     m.x[2][3],
                     m.x[3][3]);
}

// Inlinable version to enable vectorization

// Inlinable version to enable vectorization
OSL_INLINE void
inlinedMultMatrixMatrix (const Matrix44 &a,
                       const Matrix44 &b,
                       Matrix44 &c)
{
    // original version did casting from 2d with known offsets
    // which only requires 1 pointer to a version with
    // 3 different pointers and could cause aliasing issues
    // Concerned that vectorizor might be doing work than necessary
    // so made a simpler version below
#if 0
    register const float *  ap = &a.x[0][0];
    register const float *  bp = &b.x[0][0];
    register       float *  cp = &c.x[0][0];

    register float a0, a1, a2, a3;

    a0 = ap[0];
    a1 = ap[1];
    a2 = ap[2];
    a3 = ap[3];

    cp[0]  = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12];
    cp[1]  = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13];
    cp[2]  = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14];
    cp[3]  = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15];

    a0 = ap[4];
    a1 = ap[5];
    a2 = ap[6];
    a3 = ap[7];

    cp[4]  = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12];
    cp[5]  = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13];
    cp[6]  = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14];
    cp[7]  = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15];

    a0 = ap[8];
    a1 = ap[9];
    a2 = ap[10];
    a3 = ap[11];

    cp[8]  = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12];
    cp[9]  = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13];
    cp[10] = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14];
    cp[11] = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15];

    a0 = ap[12];
    a1 = ap[13];
    a2 = ap[14];
    a3 = ap[15];

    cp[12] = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12];
    cp[13] = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13];
    cp[14] = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14];
    cp[15] = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15];
#else
    const float a00 = a.x[0][0];
    const float a01 = a.x[0][1];
    const float a02 = a.x[0][2];
    const float a03 = a.x[0][3];

    c.x[0][0]  = a00 * b.x[0][0]  + a01 * b.x[1][0]  + a02 * b.x[2][0]  + a03 * b.x[3][0];
    c.x[0][1]  = a00 * b.x[0][1]  + a01 * b.x[1][1]  + a02 * b.x[2][1]  + a03 * b.x[3][1];
    c.x[0][2]  = a00 * b.x[0][2]  + a01 * b.x[1][2]  + a02 * b.x[2][2] + a03 * b.x[3][2];
    c.x[0][3]  = a00 * b.x[0][3]  + a01 * b.x[1][3]  + a02 * b.x[2][3] + a03 * b.x[3][3];

    const float a10 = a.x[1][0];
    const float a11 = a.x[1][1];
    const float a12 = a.x[1][2];
    const float a13 = a.x[1][3];

    c.x[1][0]  = a10 * b.x[0][0]  + a11 * b.x[1][0]  + a12 * b.x[2][0]  + a13 * b.x[3][0];
    c.x[1][1]  = a10 * b.x[0][1]  + a11 * b.x[1][1]  + a12 * b.x[2][1]  + a13 * b.x[3][1];
    c.x[1][2]  = a10 * b.x[0][2]  + a11 * b.x[1][2]  + a12 * b.x[2][2] + a13 * b.x[3][2];
    c.x[1][3]  = a10 * b.x[0][3]  + a11 * b.x[1][3]  + a12 * b.x[2][3] + a13 * b.x[3][3];

    const float a20 = a.x[2][0];
    const float a21 = a.x[2][1];
    const float a22 = a.x[2][2];
    const float a23 = a.x[2][3];

    c.x[2][0]  = a20 * b.x[0][0]  + a21 * b.x[1][0]  + a22 * b.x[2][0]  + a23 * b.x[3][0];
    c.x[2][1]  = a20 * b.x[0][1]  + a21 * b.x[1][1]  + a22 * b.x[2][1]  + a23 * b.x[3][1];
    c.x[2][2] = a20 * b.x[0][2]  + a21 * b.x[1][2]  + a22 * b.x[2][2] + a23 * b.x[3][2];
    c.x[2][3] = a20 * b.x[0][3]  + a21 * b.x[1][3]  + a22 * b.x[2][3] + a23 * b.x[3][3];

    const float a30 = a.x[3][0];
    const float a31 = a.x[3][1];
    const float a32 = a.x[3][2];
    const float a33 = a.x[3][3];

    c.x[3][0] = a30 * b.x[0][0]  + a31 * b.x[1][0]  + a32 * b.x[2][0]  + a33 * b.x[3][0];
    c.x[3][1] = a30 * b.x[0][1]  + a31 * b.x[1][1]  + a32 * b.x[2][1]  + a33 * b.x[3][1];
    c.x[3][2] = a30 * b.x[0][2]  + a31 * b.x[1][2]  + a32 * b.x[2][2] + a33 * b.x[3][2];
    c.x[3][3] = a30 * b.x[0][3]  + a31 * b.x[1][3]  + a32 * b.x[2][3] + a33 * b.x[3][3];
#endif
}

namespace fast {

#if 0
// Considering having functionally equivalent versions of Vec3, Color3, Matrix44
// with slight modifications to inlining and implmentation to avoid aliasing and
// improve likelyhood of proper privation of local variables within a SIMD loop
///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2004-2012, Industrial Light & Magic, a division of Lucas

namespace Imath {

template <class T>
using Vec3  = ::Imath::Vec3<T>;


template <class T>
class Color3: public Vec3 <T>
{
  public:

    //-------------
    // Constructors
    //-------------

    OSL_INLINE Color3 ();			// no initialization
    OSL_INLINE explicit Color3 (T a);	// (a a a)
    OSL_INLINE Color3 (T a, T b, T c);	// (a b c)


    //---------------------------------
    // Copy constructors and assignment
    //---------------------------------

    OSL_INLINE Color3 (const Color3 &c);
    template <class S> OSL_INLINE Color3 (const Vec3<S> &v);

    OSL_INLINE const Color3 &	operator = (const Color3 &c);


    //------------------------
    // Component-wise addition
    //------------------------

    OSL_INLINE const Color3 &	operator += (const Color3 &c);
    OSL_INLINE Color3		operator + (const Color3 &c) const;


    //---------------------------
    // Component-wise subtraction
    //---------------------------

    OSL_INLINE const Color3 &	operator -= (const Color3 &c);
    OSL_INLINE Color3		operator - (const Color3 &c) const;


    //------------------------------------
    // Component-wise multiplication by -1
    //------------------------------------

    OSL_INLINE Color3		operator - () const;
    OSL_INLINE const Color3 &	negate ();


    //------------------------------
    // Component-wise multiplication
    //------------------------------

    OSL_INLINE const Color3 &	operator *= (const Color3 &c);
    OSL_INLINE const Color3 &	operator *= (T a);
    OSL_INLINE Color3		operator * (const Color3 &c) const;
    OSL_INLINE Color3		operator * (T a) const;


    //------------------------
    // Component-wise division
    //------------------------

    OSL_INLINE const Color3 &	operator /= (const Color3 &c);
    OSL_INLINE const Color3 &	operator /= (T a);
    OSL_INLINE Color3		operator / (const Color3 &c) const;
    OSL_INLINE Color3		operator / (T a) const;
};


//-------------------------
// Implementation of Color3
//-------------------------

template <class T>
Color3<T>::Color3 (): Vec3 <T> ()
{
    // empty
}

template <class T>
Color3<T>::Color3 (T a): Vec3 <T> (a)
{
    // empty
}

template <class T>
Color3<T>::Color3 (T a, T b, T c): Vec3 <T> (a, b, c)
{
    // empty
}

template <class T>
Color3<T>::Color3 (const Color3 &c): Vec3 <T> (c)
{
    // empty
}

template <class T>
template <class S>
Color3<T>::Color3 (const Vec3<S> &v): Vec3 <T> (v)
{
    //empty
}

template <class T>
const Color3<T> &
Color3<T>::operator = (const Color3 &c)
{
    //*((Vec3<T> *) this) = c;
	Vec3<T>::operator=(c);
    return *this;
}

template <class T>
const Color3<T> &
Color3<T>::operator += (const Color3 &c)
{
    //*((Vec3<T> *) this) += c;
	Vec3<T>::operator+=(c);
    return *this;
}

template <class T>
Color3<T>
Color3<T>::operator + (const Color3 &c) const
{
//    return Color3 (*(Vec3<T> *)this + (const Vec3<T> &)c);
    return Color3 (Vec3<T>::operator + (c));
	//return c;
}

template <class T>
const Color3<T> &
Color3<T>::operator -= (const Color3 &c)
{
    //*((Vec3<T> *) this) -= c;
	Vec3<T>::operator-=(c);
    return *this;
}

template <class T>
Color3<T>
Color3<T>::operator - (const Color3 &c) const
{
    //return Color3 (*(Vec3<T> *)this - (const Vec3<T> &)c);
	return Color3 (Vec3<T>::operator-(c));
}

template <class T>
Color3<T>
Color3<T>::operator - () const
{
    //return Color3 (-(*(Vec3<T> *)this));
	return Color3 (Vec3<T>::operator-());
}

template <class T>
const Color3<T> &
Color3<T>::negate ()
{
    //((Vec3<T> *) this)->negate();
	Vec3<T>::negate();
    return *this;
}

template <class T>
const Color3<T> &
Color3<T>::operator *= (const Color3 &c)
{
    //*((Vec3<T> *) this) *= c;
	Vec3<T>::operator *= (c);
    return *this;
}

template <class T>
const Color3<T> &
Color3<T>::operator *= (T a)
{
//    *((Vec3<T> *) this) *= a;
	Vec3<T>::operator *= (a);
    return *this;
}

template <class T>
Color3<T>
Color3<T>::operator * (const Color3 &c) const
{
    //return Color3 (*(Vec3<T> *)this * (const Vec3<T> &)c);
	return Color3 (Vec3<T>::operator * (c));
}

template <class T>
Color3<T>
Color3<T>::operator * (T a) const
{
    //return Color3 (*(Vec3<T> *)this * a);
	return Color3 (Vec3<T>::operator * (a));
}

template <class T>
const Color3<T> &
Color3<T>::operator /= (const Color3 &c)
{
    //*((Vec3<T> *) this) /= c;
	Vec3<T>::operator /=(c);
    return *this;
}

template <class T>
const Color3<T> &
Color3<T>::operator /= (T a)
{
//    *((Vec3<T> *) this) /= a;
	Vec3<T>::operator /=(a);
    return *this;
}

template <class T>
Color3<T>
Color3<T>::operator / (const Color3 &c) const
{
    //return Color3 (*(Vec3<T> *)this / (const Vec3<T> &)c);
	return Color3 (Vec3<T>::operator / (c));
}

template <class T>
Color3<T>
Color3<T>::operator / (T a) const
{
    //return Color3 (*(Vec3<T> *)this / a);
	return Color3 (Vec3<T>::operator / (a));
}


} // namespace Imath

typedef Imath::Color3<float> Color3;

#endif


} // fast


OSL_NAMESPACE_EXIT
