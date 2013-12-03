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
#ifndef OSL_DUAL_VEC_H
#define OSL_DUAL_VEC_H

#include "oslconfig.h"
#include "dual.h"

OSL_NAMESPACE_ENTER


/// Templated trick to be able to derive what type we use to represent
/// a vector, given a scalar, automatically using the right kind of Dual2.
template<typename T> struct Vec3FromScalar {};
template<> struct Vec3FromScalar<float> { typedef Vec3 type; };
template<> struct Vec3FromScalar<Dual2<float> > { typedef Dual2<Vec3> type; };

/// Templated trick to be able to derive the scalar component type of
/// a vector, whether a VecN or a Dual2<VecN>.
template<typename T> struct ScalarFromVec {};
template<> struct ScalarFromVec<Vec3> { typedef Float type; };
template<> struct ScalarFromVec<Vec2> { typedef Float type; };
template<> struct ScalarFromVec<Dual2<Vec3> > { typedef Dual2<Float> type; };
template<> struct ScalarFromVec<Dual2<Vec2> > { typedef Dual2<Float> type; };



/// A uniform way to assemble a Vec3 from float and a Dual2<Vec3>
/// from Dual2<float>.
inline Vec3
make_Vec3 (float x, float y, float z)
{
    return Vec3 (x, y, z);
}

inline Dual2<Vec3>
make_Vec3 (const Dual2<float> &x, const Dual2<float> &y, const Dual2<float> &z)
{
    return Dual2<Vec3> (Vec3 (x.val(), y.val(), z.val()),
                        Vec3 (x.dx(),  y.dx(),  z.dx()),
                        Vec3 (x.dy(),  y.dy(),  z.dy()));
}


/// Make a Dual2<Vec3> from a single Dual2<Float> x coordinate, and 0
/// for the other components.
inline Dual2<Vec3>
make_Vec3 (const Dual2<Float> &x)
{
    return Dual2<Vec3> (Vec3 (x.val(), 0.0f, 0.0f),
                        Vec3 (x.dx(),  0.0f, 0.0f),
                        Vec3 (x.dy(),  0.0f, 0.0f));
}


inline Dual2<Vec3>
make_Vec3 (const Dual2<Float> &x, const Dual2<Float> &y)
{
    return Dual2<Vec3> (Vec3 (x.val(), y.val(), 0.0f),
                        Vec3 (x.dx(),  y.dx(),  0.0f),
                        Vec3 (x.dy(),  y.dy(),  0.0f));
}




/// A uniform way to assemble a Vec2 from float and a Dual2<Vec2>
/// from Dual2<float>.
inline Vec2
make_Vec2 (float x, float y)
{
    return Vec2 (x, y);
}

inline Dual2<Vec2>
make_Vec2 (const Dual2<float> &x, const Dual2<float> &y)
{
    return Dual2<Vec2> (Vec2 (x.val(), y.val()),
                        Vec2 (x.dx(),  y.dx()),
                        Vec2 (x.dy(),  y.dy()));
}



/// A uniform way to extract a single component from a Vec3 or Dual2<Vec3>
inline float
comp (const Vec3 &v, int c)
{
    return v[c];
}


inline Dual2<float>
comp (const Dual2<Vec3> &v, int c)
{
    return Dual2<float> (v.val()[c], v.dx()[c], v.dy()[c]);
}



/// Multiply a 3x3 matrix by a 3-vector, with derivs.
///
template <class S, class T>
inline void
multMatrix (const Imath::Matrix33<T> &M, const Dual2<Imath::Vec3<S> > &src,
            Dual2<Imath::Vec3<S> > &dst)
{
    Dual2<float> src0 = comp(src,0), src1 = comp(src,1), src2 = comp(src,2);
    Dual2<S> a = src0 * M[0][0] + src1 * M[1][0] + src2 * M[2][0];
    Dual2<S> b = src0 * M[0][1] + src1 * M[1][1] + src2 * M[2][1];
    Dual2<S> c = src0 * M[0][2] + src1 * M[1][2] + src2 * M[2][2];
    dst = make_Vec3 (a, b, c);
}



/// Multiply a matrix times a vector with derivatives to obtain
/// a transformed vector with derivatives.
inline void
multVecMatrix (const Matrix44 &M, Dual2<Vec3> &in, Dual2<Vec3> &out)
{
    // Rearrange into a Vec3<Dual2<float> >
    Imath::Vec3<Dual2<float> > din, dout;
    for (int i = 0;  i < 3;  ++i)
        din[i].set (in.val()[i], in.dx()[i], in.dy()[i]);

    // N.B. the following function has a divide by 'w'
    M.multVecMatrix (din, dout);

    // Rearrange back into Dual2<Vec3>
    out.set (Vec3 (dout[0].val(), dout[1].val(), dout[2].val()),
             Vec3 (dout[0].dx(),  dout[1].dx(),  dout[2].dx()),
             Vec3 (dout[0].dy(),  dout[1].dy(),  dout[2].dy()));
}

/// Multiply a matrix times a direction with derivatives to obtain
/// a transformed direction with derivatives.
inline void
multDirMatrix (const Matrix44 &M, Dual2<Vec3> &in, Dual2<Vec3> &out)
{
    M.multDirMatrix (in.val(), out.val());
    M.multDirMatrix (in.dx(), out.dx());
    M.multDirMatrix (in.dy(), out.dy());
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
make_Color3 (const Dual2<float> &x, const Dual2<float> &y, const Dual2<float> &z)
{
    return Dual2<Color3> (Color3 (x.val(), y.val(), z.val()),
                          Color3 (x.dx(), y.dx(), z.dx()),
                          Color3 (x.dy(), y.dy(), z.dy()));
}



/// Various operator* permuations between Dual2<float> and Dual2<Vec3> 
// datatypes.
inline Dual2<Vec3> 
operator* (float a, const Dual2<Vec3> &b)
{
    return Dual2<Vec3>(a*b.val(), a*b.dx(), a*b.dy());
}

inline Dual2<Vec3> 
operator* (const Dual2<Vec3> &a, float b)
{
    return Dual2<Vec3>(a.val()*b, a.dx()*b, a.dy()*b);
}

inline Dual2<Vec3> 
operator* (const Vec3 &a, const Dual2<float> &b)
{
    return Dual2<Vec3>(a*b.val(), a*b.dx(), a*b.dy());
}

inline Dual2<Vec3> 
operator* (const Dual2<Vec3> &a, const Dual2<float> &b)
{
    return Dual2<Vec3>(a.val()*b.val(), 
                       a.val()*b.dx() + a.dx()*b.val(),
                       a.val()*b.dy() + a.dy()*b.val());
}


inline Dual2<float>
dot (const Dual2<Vec3> &a, const Dual2<Vec3> &b)
{
    Dual2<float> ax (a.val().x, a.dx().x, a.dy().x);
    Dual2<float> ay (a.val().y, a.dx().y, a.dy().y);
    Dual2<float> az (a.val().z, a.dx().z, a.dy().z);
    Dual2<float> bx (b.val().x, b.dx().x, b.dy().x);
    Dual2<float> by (b.val().y, b.dx().y, b.dy().y);
    Dual2<float> bz (b.val().z, b.dx().z, b.dy().z);
    return ax*bx + ay*by + az*bz;
}



inline Dual2<float>
dot (const Dual2<Vec3> &a, const Vec3 &b)
{
    Dual2<float> ax (a.val().x, a.dx().x, a.dy().x);
    Dual2<float> ay (a.val().y, a.dx().y, a.dy().y);
    Dual2<float> az (a.val().z, a.dx().z, a.dy().z);
    return ax*b.x + ay*b.y + az*b.z;
}



inline Dual2<float>
dot (const Vec3 &a, const Dual2<Vec3> &b)
{
    Dual2<float> bx (b.val().x, b.dx().x, b.dy().x);
    Dual2<float> by (b.val().y, b.dx().y, b.dy().y);
    Dual2<float> bz (b.val().z, b.dx().z, b.dy().z);
    return a.x*bx + a.y*by + a.z*bz;
}



inline Dual2<float>
dot (const Dual2<Vec2> &a, const Dual2<Vec2> &b)
{
    Dual2<float> ax (a.val().x, a.dx().x, a.dy().x);
    Dual2<float> ay (a.val().y, a.dx().y, a.dy().y);
    Dual2<float> bx (b.val().x, b.dx().x, b.dy().x);
    Dual2<float> by (b.val().y, b.dx().y, b.dy().y);
    return ax*bx + ay*by;
}



inline Dual2<float>
dot (const Dual2<Vec2> &a, const Vec2 &b)
{
    Dual2<float> ax (a.val().x, a.dx().x, a.dy().x);
    Dual2<float> ay (a.val().y, a.dx().y, a.dy().y);
    return ax*b.x + ay*b.y;
}



inline Dual2<float>
dot (const Vec2 &a, const Dual2<Vec2> &b)
{
    Dual2<float> bx (b.val().x, b.dx().x, b.dy().x);
    Dual2<float> by (b.val().y, b.dx().y, b.dy().y);
    return a.x*bx + a.y*by;
}



inline Dual2<Vec3>
cross (const Dual2<Vec3> &a, const Dual2<Vec3> &b)
{
    Dual2<float> ax (a.val().x, a.dx().x, a.dy().x);
    Dual2<float> ay (a.val().y, a.dx().y, a.dy().y);
    Dual2<float> az (a.val().z, a.dx().z, a.dy().z);
    Dual2<float> bx (b.val().x, b.dx().x, b.dy().x);
    Dual2<float> by (b.val().y, b.dx().y, b.dy().y);
    Dual2<float> bz (b.val().z, b.dx().z, b.dy().z);

    Dual2<float> nx = ay*bz - az*by;
    Dual2<float> ny = az*bx - ax*bz;
    Dual2<float> nz = ax*by - ay*bx;

    return Dual2<Vec3> (Vec3(nx.val(), ny.val(), nz.val()),
                        Vec3(nx.dx(),  ny.dx(),  nz.dx()  ),
                        Vec3(nx.dy(),  ny.dy(),  nz.dy()  ));
}



inline Dual2<float>
length (const Dual2<Vec3> &a)
{
    Dual2<float> ax (a.val().x, a.dx().x, a.dy().x);
    Dual2<float> ay (a.val().y, a.dx().y, a.dy().y);
    Dual2<float> az (a.val().z, a.dx().z, a.dy().z);
    return sqrt(ax*ax + ay*ay + az*az);
}



inline Dual2<Vec3>
normalize (const Dual2<Vec3> &a)
{
    if (a.val().x == 0 && a.val().y == 0 && a.val().z == 0) {
        return Dual2<Vec3> (Vec3(0, 0, 0),
                            Vec3(0, 0, 0),
                            Vec3(0, 0, 0));
    } else {
        Dual2<float> ax (a.val().x, a.dx().x, a.dy().x);
        Dual2<float> ay (a.val().y, a.dx().y, a.dy().y);
        Dual2<float> az (a.val().z, a.dx().z, a.dy().z);
        Dual2<float> inv_length = 1.0f / sqrt(ax*ax + ay*ay + az*az);
        ax = ax*inv_length;
        ay = ay*inv_length;
        az = az*inv_length;
        return Dual2<Vec3> (Vec3(ax.val(), ay.val(), az.val()),
                            Vec3(ax.dx(),  ay.dx(),  az.dx() ),
                            Vec3(ax.dy(),  ay.dy(),  az.dy() ));
    }
}



inline Dual2<float>
distance (const Dual2<Vec3> &a, const Dual2<Vec3> &b)
{
    return length (a - b);
}



OSL_NAMESPACE_EXIT

#endif /* OSL_DUAL_VEC_H */
