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

#ifndef OSL_DUAL_VEC_H
#define OSL_DUAL_VEC_H

#include "oslconfig.h"
#include "dual.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {


/// Templated trick to be able to derive what type we use to represent
/// a vector, given a scalar, automatically using the right kind of Dual2.
template<typename T> struct Vec3FromScalar {};
template<> struct Vec3FromScalar<float> { typedef Vec3 type; };
template<> struct Vec3FromScalar<Dual2<float> > { typedef Dual2<Vec3> type; };


/// A uniform way to assemble a Vec3 from float and a Dual2<Vec3>
/// from Dual2<float>.
inline Vec3
make_Vec3 (float x, float y, float z)
{
    return Vec3 (x, y, z);
}

inline Dual2<Vec3>
make_Vec3 (Dual2<float> &x, Dual2<float> &y, Dual2<float> &z)
{
    return Dual2<Vec3> (Vec3 (x.val(), y.val(), z.val()),
                        Vec3 (x.dx(), y.dx(), z.dx()),
                        Vec3 (x.dy(), y.dy(), z.dy()));
}




/// Templated trick to be able to derive what type we use to represent
/// a color, given a scalar, automatically using the right kind of Dual2.
template<typename T> struct Color3FromScalar {};
template<> struct Color3FromScalar<float> { typedef Color3 type; };
template<> struct Color3FromScalar<Dual2<float> > { typedef Dual2<Color3> type; };


/// A uniform way to assemble a Color3 from float and a Dual2<Color3>
/// from Dual2<float>.
inline Color3
make_Color3 (float x, float y, float z)
{
    return Color3 (x, y, z);
}

inline Dual2<Color3>
make_Color3 (Dual2<float> &x, Dual2<float> &y, Dual2<float> &z)
{
    return Dual2<Color3> (Color3 (x.val(), y.val(), z.val()),
                          Color3 (x.dx(), y.dx(), z.dx()),
                          Color3 (x.dy(), y.dy(), z.dy()));
}



}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif


#endif /* OSL_DUAL_VEC_H */
