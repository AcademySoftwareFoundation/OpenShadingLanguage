// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <vector>

#include <OpenImageIO/fmath.h>

#include "../testshade/render_state.h"
#include "optix_compat.h"
#include "render_params.h"
#include <OSL/dual_vec.h>
#include <OSL/oslconfig.h>
#include "bvh.h"

#if OSL_USE_OPTIX
#    include <optix.h>
#    include <vector_functions.h>  // from CUDA
#endif

#ifdef __CUDACC__
#    include "cuda/rend_lib.h"
#endif

// The primitives don't included the intersection routines, etc., from the
// versions in testrender, since those operations are performed on the GPU.
//
// See the source files in the cuda subdirectory for the implementations.


OSL_NAMESPACE_BEGIN

class OptixRenderer;  // FIXME -- should not be here


// build two vectors orthogonal to the first, assumes n is normalized
inline OSL_HOSTDEVICE void
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

    OSL_HOSTDEVICE
    Ray(const Vec3& o, const Vec3& d, float radius, float spread,
        float roughness, RayType raytype)
        : origin(o)
        , direction(d)
        , radius(radius)
        , spread(spread)
        , roughness(roughness)
        , raytype(static_cast<int>(raytype))
    {
    }

    OSL_HOSTDEVICE
    Vec3 point(float t) const { return origin + direction * t; }

    OSL_HOSTDEVICE
    Dual2<Vec3> dual_direction() const
    {
        Dual2<Vec3> v;
        v.val() = direction;
        ortho(direction, v.dx(), v.dy());
        v.dx() *= spread;
        v.dy() *= spread;
        return v;
    }

    OSL_HOSTDEVICE
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
    float radius, spread, roughness;
    int raytype;
};



struct Camera {
    OSL_HOSTDEVICE Camera() {}

    // Set where the camera sits and looks at.
    OSL_HOSTDEVICE
    void lookat(const Vec3& eye, const Vec3& dir, const Vec3& up, float fov)
    {
        this->eye = eye;
        this->dir = dir.normalized();
        this->up  = up;
        this->fov = fov;
        finalize();
    }

    // Set resolution
    OSL_HOSTDEVICE
    void resolution(int w, int h)
    {
        xres = w;
        yres = h;
        invw = 1.0f / w;
        invh = 1.0f / h;
        finalize();
    }

    // Compute all derived values based on camera parameters.
    OSL_HOSTDEVICE
    void finalize()
    {
        float k    = OIIO::fast_tan(fov * float(M_PI / 360));
        Vec3 right = dir.cross(up).normalize();
        cx         = right * (xres * k / yres);
        cy         = (cx.cross(dir)).normalize() * k;
    }

    // Get a ray for the given screen coordinates.
    OSL_HOSTDEVICE
    Ray get(float x, float y) const
    {
        // TODO: On CUDA devices, the normalize() operation can result in vector
        // components with magnitudes slightly greater than 1.0, which can cause
        // downstream computations to blow up and produce NaNs. Normalizing the
        // vector again avoids this issue.
        const Vec3 v = (cx * (x * invw - 0.5f) + cy * (0.5f - y * invh) + dir)
#ifndef __CUDACC__
                           .normalize();
#else
                           .normalize()
                           .normalized();
#endif

        const float cos_a = dir.dot(v);
        const float spread
            = sqrtf(invw * invh * cx.length() * cy.length() * cos_a) * cos_a;
        return Ray(eye, v, 0, spread, 0.0f, Ray::CAMERA);
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
    int a, b, c;
};

struct LightSample {
    Vec3 dir;
    float dist;
    float pdf;
    float u, v;
};

using ShaderMap = std::unordered_map<std::string, int>;

struct Scene {
#ifndef __CUDACC__
    void add_sphere(const Vec3& c, float r, int shaderID, int resolution);

    void add_quad(const Vec3& p, const Vec3& ex, const Vec3& ey, int shaderID,
                  int resolution);

    // add models parsed from a .obj file
    void add_model(const std::string& filename, const ShaderMap& shadermap,
                   int shaderID, OIIO::ErrorHandler& errhandler);

    int num_prims() const { return triangles.size(); }

    void prepare(OIIO::ErrorHandler& errhandler);
#endif

    // NB: OptiX needs to populate the ShaderGlobals in the closest-hit program,
    //     so we need to pass along a pointer to the struct.
    OSL_HOSTDEVICE
    Intersection intersect(const Ray& r, const float tmax,
                           const unsigned skipID1,
                           const unsigned skipID2 = ~0u) const;

    OSL_HOSTDEVICE
    LightSample sample(int primID, const Vec3& x, float xi, float yi) const
    {
        // A Low-Distortion Map Between Triangle and Square
        // Eric Heitz, 2019
        if (yi > xi) {
            xi *= 0.5f;
            yi -= xi;
        } else {
            yi *= 0.5f;
            xi -= yi;
        }
        const Vec3 va = verts[triangles[primID].a];
        const Vec3 vb = verts[triangles[primID].b];
        const Vec3 vc = verts[triangles[primID].c];
        const Vec3 n  = (va - vb).cross(va - vc);

        Vec3 l   = ((1 - xi - yi) * va + xi * vb + yi * vc) - x;
        float d2 = l.length2();
        Vec3 dir = l.normalize();
        // length of n is twice the area
        float pdf = d2 / (0.5f * fabsf(dir.dot(n)));
        return { dir, sqrtf(d2), pdf, xi, yi };
    }

    OSL_HOSTDEVICE
    float shapepdf(int primID, const Vec3& x, const Vec3& p) const
    {
        const Vec3 va = verts[triangles[primID].a];
        const Vec3 vb = verts[triangles[primID].b];
        const Vec3 vc = verts[triangles[primID].c];
        const Vec3 n  = (va - vb).cross(va - vc);

        Vec3 l   = p - x;
        float d2 = l.length2();
        Vec3 dir = l.normalize();
        // length of n is twice the area
        return d2 / (0.5f * fabsf(dir.dot(n)));
    }

    OSL_HOSTDEVICE
    float primitivearea(int primID) const
    {
        const Vec3 va = verts[triangles[primID].a];
        const Vec3 vb = verts[triangles[primID].b];
        const Vec3 vc = verts[triangles[primID].c];
        return 0.5f * (va - vb).cross(va - vc).length();
    }

    OSL_HOSTDEVICE
    Vec3 normal(const Dual2<Vec3>& p, Vec3& Ng, int primID, float u,
                float v) const
    {
        const Vec3 va = verts[triangles[primID].a];
        const Vec3 vb = verts[triangles[primID].b];
        const Vec3 vc = verts[triangles[primID].c];
        Ng            = (va - vb).cross(va - vc).normalize();

        // this triangle doesn't have vertex normals, just use face normal
        if (n_triangles[primID].a < 0)
            return Ng;

        // use vertex normals
        const Vec3 na = normals[n_triangles[primID].a];
        const Vec3 nb = normals[n_triangles[primID].b];
        const Vec3 nc = normals[n_triangles[primID].c];
        return ((1 - u - v) * na + u * nb + v * nc).normalize();
    }

    OSL_HOSTDEVICE
    Dual2<Vec2> uv(const Dual2<Vec3>& p, const Vec3& n, Vec3& dPdu, Vec3& dPdv,
                   int primID, float u, float v) const
    {
        if (uv_triangles[primID].a < 0)
            return Dual2<Vec2>(Vec2(0, 0));
        const Vec2 ta = uvs[uv_triangles[primID].a];
        const Vec2 tb = uvs[uv_triangles[primID].b];
        const Vec2 tc = uvs[uv_triangles[primID].c];

        const Vec3 va = verts[triangles[primID].a];
        const Vec3 vb = verts[triangles[primID].b];
        const Vec3 vc = verts[triangles[primID].c];

        const Vec2 dt02 = ta - tc, dt12 = tb - tc;
        const Vec3 dp02 = va - vc, dp12 = vb - vc;
        // TODO: could use Kahan's algorithm here
        // https://pharr.org/matt/blog/2019/11/03/difference-of-floats
        const float det = dt02.x * dt12.y - dt02.y * dt12.x;
        if (det != 0) {
            Float invdet = 1 / det;
            dPdu         = (dt12.y * dp02 - dt02.y * dp12) * invdet;
            dPdv         = (-dt12.x * dp02 + dt02.x * dp12) * invdet;
            // TODO: smooth out dPdu and dPdv by storing per vertex tangents
        }
        return Dual2<Vec2>((1 - u - v) * ta + u * tb + v * tc);
    }

    OSL_HOSTDEVICE int shaderid(int primID) const { return shaderids[primID]; }

#ifndef __CUDACC__
    // basic triangle data
    std::vector<Vec3> verts;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;
    std::vector<TriangleIndices> triangles;
    std::vector<TriangleIndices> uv_triangles;
    std::vector<TriangleIndices> n_triangles;
    std::vector<int> shaderids;
    std::vector<int>
        last_index;  // one entry per mesh, stores the last triangle index (+1) -- also is the start triangle of the next mesh
    // acceleration structure (built over triangles)
    std::unique_ptr<BVH> bvh;
#else
    const Vec3* verts;
    const Vec3* normals;
    const Vec2* uvs;
    const TriangleIndices* triangles;
    const TriangleIndices* uv_triangles;
    const TriangleIndices* n_triangles;
    const int* shaderids;
    OptixTraversableHandle handle;
#endif
};

OSL_NAMESPACE_END
