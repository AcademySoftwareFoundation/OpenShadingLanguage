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

#ifndef OSLOPS_H
#define OSLOPS_H

#include "OpenImageIO/typedesc.h"

#include "oslexec.h"
#include "osl_pvt.h"
#include "oslexec_pvt.h"
#include "dual.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {

// Utility to convert something that looks like runflags to spans
template<class RF>
inline bool runflags_to_spans (RF &rf, int beg, int end,
                               int *indices, int &nindices, bool onval = true)
{
    bool any_off = false;
    for (int i = beg;  i < end;  ++i) {
        if ((rf[i] != 0) == onval) {
            indices[nindices++] = i;
            while (i < end && rf[i] == onval)  // Skip to next 0
                ++i;
            indices[nindices++] = i;
            any_off |= (i < end);  // stopped because of an off point
        } else {
            any_off = true;
        }
    }
    return any_off;
}

// Utility to copy spans-to-spans, but only when the thing that looks like
// runflags matches the 'onval'.
template<class RF>
inline bool spans_runflags_to_spans (int *spans, int spanlength, RF &rf,
                                     int *indices, int &nindices, bool onval = true)
{
    bool any_off = false;
    for (int s = 0;  s < spanlength;  s += 2)
        any_off |= runflags_to_spans (rf, spans[s], spans[s+1],
                                      indices, nindices, onval);
    return any_off;
}


#define CLOSURE_PREPARE(name, classname)    \
void name(RendererServices *, int id, void *data) \
{                                                 \
    memset(data, 0, sizeof(classname));           \
    new (data) classname();                       \
}

// Proxy type that derives from Vec3 but allows some additional operations
// not normally supported by Imath::Vec3.  This is purely for convenience.
class VecProxy : public Vec3 {
public:
    VecProxy () { }
    VecProxy (float a) : Vec3(a,a,a) { }
    VecProxy (float a, float b, float c) : Vec3(a,b,c) { }
    VecProxy (const Vec3& v) : Vec3(v) { }

    friend VecProxy operator+ (const Vec3 &v, float f) {
        return VecProxy (v.x+f, v.y+f, v.z+f);
    }
    friend VecProxy operator+ (float f, const Vec3 &v) {
        return VecProxy (v.x+f, v.y+f, v.z+f);
    }
    friend VecProxy operator- (const Vec3 &v, float f) {
        return VecProxy (v.x-f, v.y-f, v.z-f);
    }
    friend VecProxy operator- (float f, const Vec3 &v) {
        return VecProxy (f-v.x, f-v.y, f-v.z);
    }
    friend VecProxy operator* (const Vec3 &v, int f) {
        return VecProxy (v.x*f, v.y*f, v.z*f);
    }
    friend VecProxy operator* (int f, const Vec3 &v) {
        return VecProxy (v.x*f, v.y*f, v.z*f);
    }
    friend VecProxy operator/ (const Vec3 &v, int f) {
        if (f == 0)
            return VecProxy(0.0);
        return VecProxy (v.x/f, v.y/f, v.z/f);
    }
    friend VecProxy operator/ (float f, const Vec3 &v) {
        return VecProxy (v.x == 0.0 ? 0.0 : f/v.x, 
                         v.y == 0.0 ? 0.0 : f/v.y,
                         v.z == 0.0 ? 0.0 : f/v.z);
    }
    friend VecProxy operator/ (int f, const Vec3 &v) {
        return VecProxy (v.x == 0.0 ? 0.0 : f/v.x, 
                         v.y == 0.0 ? 0.0 : f/v.y,
                         v.z == 0.0 ? 0.0 : f/v.z);
    }
    friend bool operator== (const Vec3 &v, float f) {
        return v.x == f && v.y == f && v.z == f;
    }
    friend bool operator== (const Vec3 &v, int f) {
        return v.x == f && v.y == f && v.z == f;
    }
    friend bool operator== (float f, const Vec3 &v) {
        return v.x == f && v.y == f && v.z == f;
    }
    friend bool operator== (int f, const Vec3 &v) {
        return v.x == f && v.y == f && v.z == f;
    }

    friend bool operator!= (const Vec3 &v, float f) {
        return v.x != f || v.y != f || v.z != f;
    }
    friend bool operator!= (const Vec3 &v, int f) {
        return v.x != f || v.y != f || v.z != f;
    }
    friend bool operator!= (float f, const Vec3 &v) {
        return v.x != f || v.y != f || v.z != f;
    }
    friend bool operator!= (int f, const Vec3 &v) {
        return v.x != f || v.y != f || v.z != f;
    }
};



// Proxy type that derives from Matrix44 but allows assignment of a float
// to mean f*Identity.
class MatrixProxy : public Matrix44 {
public:
    MatrixProxy () { }
    MatrixProxy (float a, float b, float c, float d,
                 float e, float f, float g, float h,
                 float i, float j, float k, float l,
                 float m, float n, float o, float p)
        : Matrix44 (a,b,c,d, e,f,g,h, i,j,k,l, m,n,o,p) { }

    MatrixProxy (float f) : Matrix44 (f,0,0,0, 0,f,0,0, 0,0,f,0, 0,0,0,f) { }

    const MatrixProxy& operator= (float f) {
        *this = MatrixProxy (f);
        return *this;
    }

    friend bool operator== (const MatrixProxy &m, float f) {
        MatrixProxy comp (f);
        return m == comp;
    }
    friend bool operator== (const MatrixProxy &m, int f) {
        MatrixProxy comp (f);
        return m == comp;
    }
    friend bool operator== (float f, const MatrixProxy &m) { return m == f; }
    friend bool operator== (int f, const MatrixProxy &m) { return m == f; }

    friend bool operator!= (const MatrixProxy &m, float f) {
        MatrixProxy comp (f);
        return m != comp;
    }
    friend bool operator!= (const MatrixProxy &m, int f) {
        MatrixProxy comp (f);
        return m != comp;
    }
    friend bool operator!= (float f, const MatrixProxy &m) { return m != f; }
    friend bool operator!= (int f, const MatrixProxy &m) { return m != f; }
};



}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif


#endif /* OSLOPS_H */
