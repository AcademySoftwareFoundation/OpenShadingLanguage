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


OSL_NAMESPACE_EXIT
