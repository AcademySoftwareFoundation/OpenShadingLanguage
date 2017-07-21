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
    S a = src[0] * M[0][0] + src[1] * M[1][0] + src[2] * M[2][0];
    S b = src[0] * M[0][1] + src[1] * M[1][1] + src[2] * M[2][1];
    S c = src[0] * M[0][2] + src[1] * M[1][2] + src[2] * M[2][2];
    dst[0] = a;
    dst[1] = b;
    dst[2] = c;
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
T determinant (const Imathx::Matrix22<T> &m)
{
    return m[0][0]*m[1][1] - m[0][1]*m[1][0];
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

OSL_NAMESPACE_EXIT
