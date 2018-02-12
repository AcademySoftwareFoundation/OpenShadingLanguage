/*
Copyright (c) 2009-2018 Sony Pictures Imageworks Inc., et al.
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

#include <OpenImageIO/fmath.h>

#include <OSL/dual_vec.h>
#include <OSL/oslconfig.h>
#include <vector>


// This file contains stripped-down versions of the scene objects from
// testrender/raytracer.h.
//
// The primitives don't included the render-time functions (intersect, etc.),
// since those operations are performed on the GPU.
//
// See the source files for sphere and quad in the cuda subdirectory for the
// implementations.

OSL_NAMESPACE_ENTER

struct Camera {
    Camera() {} // leave uninitialized
    Camera(Vec3 eye, Vec3 dir, Vec3 up, float fov, int w, int h) :
        eye(eye),
        dir(dir.normalize()),
        invw(1.0f / w), invh(1.0f / h) {
        float k = OIIO::fast_tan(fov * float(M_PI / 360));
        Vec3 right = dir.cross(up).normalize();
        cx = right * (w * k / h);
        cy = (cx.cross(dir)).normalize() * k;
    }

    Vec3 eye, dir, cx, cy;
    float invw, invh;
};


struct Primitive {
    Primitive(int shaderID, bool isLight) : shaderID(shaderID), isLight(isLight) {}

    int shaderid() const { return shaderID; }
    bool islight() const { return isLight; }

private:
    int shaderID;
    bool isLight;
};


struct Sphere : public Primitive {
    Sphere(Vec3 c, float r, int shaderID, bool isLight)
        : Primitive(shaderID, isLight), c(c), r2(r * r) {
        ASSERT(r > 0);
    }

    Vec3  c;
    float r2;
};


struct Quad : public Primitive {
    Quad(const Vec3& p, const Vec3& ex, const Vec3& ey, int shaderID, bool isLight)
        : Primitive(shaderID, isLight), p(p), ex(ex), ey(ey) {
        n = ex.cross(ey);
        a = n.length();
        n = n.normalize();
        eu = 1 / ex.length2();
        ev = 1 / ey.length2();
    }

    Vec3 p, ex, ey, n;
    float a, eu, ev;
};

OSL_NAMESPACE_EXIT
