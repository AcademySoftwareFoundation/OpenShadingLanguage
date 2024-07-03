// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <vector>

#include <OpenImageIO/fmath.h>

#include "optix_compat.h"
#include "render_params.h"
#include "bvh.h"
#include <OSL/dual_vec.h>
#include <OSL/oslconfig.h>


#if OSL_USE_OPTIX
#    include <optix.h>
#    include <vector_functions.h>  // from CUDA
#endif

// The primitives don't included the intersection routines, etc., from the
// versions in testrender, since those operations are performed on the GPU.
//
// See the source files in the cuda subdirectory for the implementations.


OSL_NAMESPACE_ENTER

class OptixRenderer;  // FIXME -- should not be here


// build two vectors orthogonal to the first, assumes n is normalized
inline void
ortho(const Vec3& n, Vec3& x, Vec3& y)
{
    x = (fabsf(n.x) > .01f ? Vec3(n.z, 0, -n.x) : Vec3(0, -n.z, n.y))
            .normalize();
    y = n.cross(x);
}


// Note: not used in OptiX mode
struct Ray {
    enum RayType {
        CAMERA       = 1,
        SHADOW       = 2,
        REFLECTION   = 4,
        REFRACTION   = 8,
        DIFFUSE      = 16,
        GLOSSY       = 32,
        SUBSURFACE   = 64,
        DISPLACEMENT = 128
    };

    Ray(const Vec3& o, const Vec3& d, float radius, float spread,
        RayType raytype)
        : origin(o)
        , direction(d)
        , radius(radius)
        , spread(spread)
        , raytype(static_cast<int>(raytype))
    {
    }

    Vec3 point(float t) const { return origin + direction * t; }
    Dual2<Vec3> dual_direction() const
    {
        Dual2<Vec3> v;
        v.val() = direction;
        ortho(direction, v.dx(), v.dy());
        v.dx() *= spread;
        v.dy() *= spread;
        return v;
    }

    Dual2<Vec3> point(Dual2<float> t) const
    {
        const float r = radius + spread * t.val();
        Dual2<Vec3> p;
        p.val() = point(t.val());
        ortho(direction, p.dx(), p.dy());
        p.dx() *= r;
        p.dy() *= r;
        return p;
    }

    Vec3 origin, direction;
    float radius, spread;
    int raytype;
};



struct Camera {
    Camera() {}

    // Set where the camera sits and looks at.
    void lookat(const Vec3& eye, const Vec3& dir, const Vec3& up, float fov)
    {
        this->eye = eye;
        this->dir = dir.normalized();
        this->up  = up;
        this->fov = fov;
        finalize();
    }

    // Set resolution
    void resolution(int w, int h)
    {
        xres = w;
        yres = h;
        invw = 1.0f / w;
        invh = 1.0f / h;
        finalize();
    }

    // Compute all derived values based on camera parameters.
    void finalize()
    {
        float k    = OIIO::fast_tan(fov * float(M_PI / 360));
        Vec3 right = dir.cross(up).normalize();
        cx         = right * (xres * k / yres);
        cy         = (cx.cross(dir)).normalize() * k;
    }

    // Get a ray for the given screen coordinates.
    Ray get(float x, float y) const
    {
        const Vec3 v = (cx * (x * invw - 0.5f) + cy * (0.5f - y * invh) + dir)
                           .normalize();
        const float cos_a = dir.dot(v);
        const float spread
            = sqrtf(invw * invh * cx.length() * cy.length() * cos_a) * cos_a;
        return Ray(eye, v, 0, spread, Ray::CAMERA);
    }

    // Specified by user:
    Vec3 eye { 0, 0, 0 };
    Vec3 dir { 0, 0, -1 };
    Vec3 up { 0, 1, 0 };
    float fov { 90 };
    int xres { 1 };
    int yres { 1 };

    // Computed:
    Vec3 cx, cy;
    float invw, invh;
};

struct TriangleIndices {
    unsigned a, b, c;
};

struct LightSample {
    Vec3 dir;
    float dist;
    float pdf;
};

using ShaderMap = std::unordered_map<std::string, int>;

struct Scene {
    void add_sphere(const Vec3& c, float r, int shaderID, int resolution);

    void add_quad(const Vec3& p, const Vec3& ex, const Vec3& ey, int shaderID, int resolution);

    // add models parsed from a .obj file
    void add_model(const std::string& filename, const ShaderMap& shadermap, int shaderID, OIIO::ErrorHandler& errhandler);

    int num_prims() const { return triangles.size(); }

    void prepare(OIIO::ErrorHandler& errhandler);

    Intersection intersect(const Ray& r, const float tmax, const unsigned skipID1, const unsigned skipID2 = ~0u) const;

    LightSample sample(int primID, const Vec3& x, float xi, float yi) const
    {
        // A Low-Distortion Map Between Triangle and Square
        // Eric Heitz, 2019
        if (yi > xi ) {
            xi *= 0.5f;
            yi -= xi;
        } else {
            yi *= 0.5f;
            xi -= yi;
        }
        const Vec3 va = verts[triangles[primID].a];
        const Vec3 vb = verts[triangles[primID].b];
        const Vec3 vc = verts[triangles[primID].c];
        const Vec3 n = (va - vb).cross(va - vc);

        Vec3 l   = ((1 - xi - yi) * va + xi * vb + yi * vc) - x;
        float d2 = l.length2();
        Vec3 dir = l.normalize();
        // length of n is twice the area
        float pdf = d2 / (0.5f * fabsf(dir.dot(n)));
        return { dir, sqrtf(d2), pdf };
    }

    float shapepdf(int primID, const Vec3& x, const Vec3& p) const
    {
        const Vec3 va = verts[triangles[primID].a];
        const Vec3 vb = verts[triangles[primID].b];
        const Vec3 vc = verts[triangles[primID].c];
        const Vec3 n = (va - vb).cross(va - vc);

        Vec3 l   = p - x;
        float d2 = l.length2();
        Vec3 dir = l.normalize();
        // length of n is twice the area
        return d2 / (0.5f * fabsf(dir.dot(n)));
    }

    float primitivearea(int primID) const
    {
        const Vec3 va = verts[triangles[primID].a];
        const Vec3 vb = verts[triangles[primID].b];
        const Vec3 vc = verts[triangles[primID].c];
        return 0.5f * (va - vb).cross(va - vc).length();
    }

    Vec3 normal(const Dual2<Vec3>& p, int primID) const
    {
        const Vec3 va = verts[triangles[primID].a];
        const Vec3 vb = verts[triangles[primID].b];
        const Vec3 vc = verts[triangles[primID].c];
        return (va - vb).cross(va - vc).normalize();
    }

    Dual2<Vec2> uv(const Dual2<Vec3>& p, const Dual2<Vec3>& n, Vec3& dPdu,
                   Vec3& dPdv, int primID) const
    {
        return Dual2<Vec2>(Vec2(0, 0));
    }

    int shaderid(int primID) const
    {
        return shaderids[primID];
    }

    // basic triangle data
    std::vector<Vec3> verts;
    std::vector<TriangleIndices> triangles;
    std::vector<int> shaderids;
    // acceleration structure (built over triangles)
    std::unique_ptr<BVH> bvh;
};

OSL_NAMESPACE_EXIT
