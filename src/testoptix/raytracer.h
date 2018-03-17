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

#include <optix_world.h>



// Converts from Imath::Vec3 to optix::float3
optix::float3 vec3_to_float3 (const OSL::Vec3& vec)
{
    return optix::make_float3 (vec.x, vec.y, vec.z);
}


// The primitives don't included the intersection routines, etc., from the
// versions in testrender, since those operations are performed on the GPU.
//
// See the source files in the cuda subdirectory for the implementations.

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

    virtual ~Primitive() {}

    int shaderid() const { return shaderID; }
    bool islight() const { return isLight; }

    virtual void setOptixVariables (optix::Geometry geom, optix::Program  bounds,
                                    optix::Program intersect) const = 0;

private:
    int shaderID;
    bool isLight;
};


struct Sphere : public Primitive {
    Sphere(Vec3 c, float r, int shaderID, bool isLight)
        : Primitive(shaderID, isLight), c(c), r2(r * r) {
        ASSERT(r > 0);
    }

    virtual void setOptixVariables (optix::Geometry geom, optix::Program  bounds,
                                    optix::Program intersect) const
    {
        geom->setPrimitiveCount (1u);
        geom->setBoundingBoxProgram (bounds);
        geom->setIntersectionProgram (intersect);

        geom["sphere"]->setFloat (optix::make_float4(c.x, c.y, c.z, sqrtf(r2)));
        geom["r2"]->setFloat (r2);
        geom["a" ]->setFloat (M_PIf * (r2 * r2));
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

    virtual void setOptixVariables (optix::Geometry geom, optix::Program bounds,
                                    optix::Program intersect) const
    {
        geom->setPrimitiveCount (1u);
        geom->setBoundingBoxProgram (bounds);
        geom->setIntersectionProgram (intersect);

        geom["p" ]->setFloat (vec3_to_float3 (p));
        geom["ex"]->setFloat (vec3_to_float3 (ex));
        geom["ey"]->setFloat (vec3_to_float3 (ey));
        geom["n" ]->setFloat (vec3_to_float3 (n));
        geom["eu"]->setFloat (eu);
        geom["ev"]->setFloat (ev);
        geom["a" ]->setFloat (a);
    }

    Vec3 p, ex, ey, n;
    float a, eu, ev;
};


struct Scene {
    void create_geom_programs (optix::Context optix_ctx, const std::string& sphere_ptx,
                               const std::string& quad_ptx)
    {
        // The bounds program is used to construct axis-aligned bounding boxes
        // for each primitive when the acceleration structure is being created.
        sphere_bounds    = optix_ctx->createProgramFromPTXString (sphere_ptx, "bounds");
        quad_bounds      = optix_ctx->createProgramFromPTXString (quad_ptx,   "bounds");

        // The intersection program is used to perform ray-geometry intersections.
        sphere_intersect = optix_ctx->createProgramFromPTXString (sphere_ptx, "intersect");
        quad_intersect   = optix_ctx->createProgramFromPTXString (quad_ptx,   "intersect");
    }

    void add_sphere(const Sphere& s) {
        spheres.push_back(s);
    }

    void add_quad(const Quad& q) {
        quads.push_back(q);
    }

    std::vector<Sphere> spheres;
    std::vector<Quad> quads;

    optix::Program sphere_intersect;
    optix::Program sphere_bounds;
    optix::Program quad_intersect;
    optix::Program quad_bounds;

    std::vector<optix::Material> optix_mtls;
};

OSL_NAMESPACE_EXIT
