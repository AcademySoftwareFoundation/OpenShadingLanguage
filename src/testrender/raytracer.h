// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <vector>

#include <OpenImageIO/fmath.h>

#include "optix_compat.h"
#include "render_params.h"
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



// Note: The primitives only use the intersection routines, etc., for CPU.
// They aren't used for OptiX mode, where instead the equivalent are in
// the cuda subdirectory.


struct Primitive {
    Primitive(int shaderID, bool isLight) : shaderID(shaderID), isLight(isLight)
    {
    }
    virtual ~Primitive() {}

    int shaderid() const { return shaderID; }
    bool islight() const { return isLight; }
    void getBounds(float& minx, float& miny, float& minz, float& maxx,
                   float& maxy, float& maxz) const;

#if OSL_USE_OPTIX
    virtual void setOptixVariables(void* data) const = 0;
#endif

private:
    int shaderID;
    bool isLight;
};


struct Sphere final : public Primitive {
    Sphere(Vec3 c, float r, int shaderID, bool isLight)
        : Primitive(shaderID, isLight), c(c), r(r), r2(r * r)
    {
        OSL_DASSERT(r > 0);
    }

    void getBounds(float& minx, float& miny, float& minz, float& maxx,
                   float& maxy, float& maxz) const
    {
        minx = c.x - r;
        miny = c.y - r;
        minz = c.z - r;
        maxx = c.x + r;
        maxy = c.y + r;
        maxz = c.z + r;
    }

    // returns distance to nearest hit or 0
    Dual2<float> intersect(const Ray& r, bool self) const
    {
        Dual2<Vec3> oc   = c - r.origin;
        Dual2<float> b   = dot(oc, r.direction);
        Dual2<float> det = b * b - dot(oc, oc) + r2;
        if (det.val() >= 0) {
            det            = sqrt(det);
            Dual2<float> x = b - det;
            Dual2<float> y = b + det;
            return self ? (fabsf(x.val()) > fabsf(y.val())
                               ? (x.val() > 0 ? x : 0)
                               : (y.val() > 0 ? y : 0))
                        : (x.val() > 0 ? x : (y.val() > 0 ? y : 0));
        }
        return 0;  // no hit
    }

    float surfacearea() const { return float(M_PI) * r2; }

    Dual2<Vec3> normal(const Dual2<Vec3>& p) const { return normalize(p - c); }

    Dual2<Vec2> uv(const Dual2<Vec3>& /*p*/, const Dual2<Vec3>& n, Vec3& dPdu,
                   Vec3& dPdv) const
    {
        Dual2<float> nx(n.val().x, n.dx().x, n.dy().x);
        Dual2<float> ny(n.val().y, n.dx().y, n.dy().y);
        Dual2<float> nz(n.val().z, n.dx().z, n.dy().z);
        Dual2<float> u = (atan2(nx, nz) + Dual2<float>(M_PI)) * 0.5f
                         * float(M_1_PI);
        Dual2<float> v = safe_acos(ny) * float(M_1_PI);
        // Review of sphere parameterization:
        //    x = r * -sin(2pi*u) * sin(pi*v)
        //    y = r * cos(pi*v)
        //    z = r * -cos(2pi*u) * sin(pi*v)
        // partial derivs:
        //    dPdu.x = -r * sin(pi*v) * 2pi * cos(2pi*u)
        //    dPdu.y = 0
        //    dPdu.z = r * sin(pi*v) * 2pi * sin(2pi*u)
        //    dPdv.x = r * -cos(pi*v) * pi * sin(2pi*u)
        //    dPdv.y = r * -pi * sin(pi*v)
        //    dPdv.z = r * -cos(pi*v) * pi * cos(2pi*u)
        const float pi = float(M_PI);
        float twopiu   = 2.0f * pi * u.val();
        float sin2piu, cos2piu;
        OIIO::sincos(twopiu, &sin2piu, &cos2piu);
        float sinpiv, cospiv;
        OIIO::sincos(pi * v.val(), &sinpiv, &cospiv);
        float pir = pi * r;
        dPdu.x    = -2.0f * pir * sinpiv * cos2piu;
        dPdu.y    = 0.0f;
        dPdu.z    = 2.0f * pir * sinpiv * sin2piu;
        dPdv.x    = -pir * cospiv * sin2piu;
        dPdv.y    = -pir * sinpiv;
        dPdv.z    = -pir * cospiv * cos2piu;
        return make_Vec2(u, v);
    }

    // return a direction towards a point on the sphere
    Vec3 sample(const Vec3& x, float xi, float yi, float& pdf) const
    {
        const float TWOPI = float(2 * M_PI);
        float cmax2       = 1 - r2 / (c - x).length2();
        float cmax        = cmax2 > 0 ? sqrtf(cmax2) : 0;
        float cos_a       = 1 - xi + xi * cmax;
        float sin_a       = sqrtf(1 - cos_a * cos_a);
        float phi         = TWOPI * yi;
        float sp, cp;
        OIIO::fast_sincos(phi, &sp, &cp);
        Vec3 sw = (c - x).normalize(), su, sv;
        ortho(sw, su, sv);
        pdf = 1 / (TWOPI * (1 - cmax));
        return (su * (cp * sin_a) + sv * (sp * sin_a) + sw * cos_a).normalize();
    }

    float shapepdf(const Vec3& x, const Vec3& /*p*/) const
    {
        const float TWOPI = float(2 * M_PI);
        float cmax2       = 1 - r2 / (c - x).length2();
        float cmax        = cmax2 > 0 ? sqrtf(cmax2) : 0;
        return 1 / (TWOPI * (1 - cmax));
    }

#if OSL_USE_OPTIX
    virtual void setOptixVariables(void* data) const
    {
        SphereParams* sphere_data = reinterpret_cast<SphereParams*>(data);
        sphere_data->c            = make_float3(c.x, c.y, c.z);
        sphere_data->r2           = r2;
        sphere_data->a            = M_PI * (r2 * r2);
        sphere_data->shaderID     = shaderid();
    }
#endif

private:
    Vec3 c;
    float r, r2;
};



struct Quad final : public Primitive {
    Quad(const Vec3& p, const Vec3& ex, const Vec3& ey, int shaderID,
         bool isLight)
        : Primitive(shaderID, isLight), p(p), ex(ex), ey(ey)
    {
        n  = ex.cross(ey);
        a  = n.length();
        n  = n.normalize();
        eu = 1 / ex.length2();
        ev = 1 / ey.length2();
    }

    void getBounds(float& minx, float& miny, float& minz, float& maxx,
                   float& maxy, float& maxz) const
    {
        const Vec3 p0 = p;
        const Vec3 p1 = p + ex;
        const Vec3 p2 = p + ex + ey;
        const Vec3 p3 = p + ey;
        minx          = std::min(p0.x, std::min(p1.x, std::min(p2.x, p3.x)));
        miny          = std::min(p0.y, std::min(p1.y, std::min(p2.y, p3.y)));
        minz          = std::min(p0.z, std::min(p1.z, std::min(p2.z, p3.z)));
        maxx          = std::max(p0.x, std::max(p1.x, std::max(p2.x, p3.x)));
        maxy          = std::max(p0.y, std::max(p1.y, std::max(p2.y, p3.y)));
        maxz          = std::max(p0.z, std::max(p1.z, std::max(p2.z, p3.z)));
    }

    // returns distance to nearest hit or 0
    Dual2<float> intersect(const Ray& r, bool self) const
    {
        if (self)
            return 0;
        Dual2<float> dn = dot(r.direction, n);
        Dual2<float> en = dot(p - r.origin, n);
        if (dn.val() * en.val() > 0) {
            Dual2<float> t  = en / dn;
            Dual2<Vec3> h   = r.point(t) - p;
            Dual2<float> dx = dot(h, ex) * eu;
            Dual2<float> dy = dot(h, ey) * ev;
            if (dx.val() >= 0 && dx.val() < 1 && dy.val() >= 0 && dy.val() < 1)
                return t;
        }
        return 0;  // no hit
    }

    float surfacearea() const { return a; }

    Dual2<Vec3> normal(const Dual2<Vec3>& /*p*/) const
    {
        return Dual2<Vec3>(n, Vec3(0, 0, 0), Vec3(0, 0, 0));
    }

    Dual2<Vec2> uv(const Dual2<Vec3>& p, const Dual2<Vec3>& /*n*/, Vec3& dPdu,
                   Vec3& dPdv) const
    {
        Dual2<Vec3> h  = p - this->p;
        Dual2<float> u = dot(h, ex) * eu;
        Dual2<float> v = dot(h, ey) * ev;
        dPdu           = ex;
        dPdv           = ey;
        return make_Vec2(u, v);
    }

    // return a direction towards a point on the sphere
    Vec3 sample(const Vec3& x, float xi, float yi, float& pdf) const
    {
        Vec3 l   = (p + xi * ex + yi * ey) - x;
        float d2 = l.length2();
        Vec3 dir = l.normalize();
        pdf      = d2 / (a * fabsf(dir.dot(n)));
        return dir;
    }

    float shapepdf(const Vec3& x, const Vec3& p) const
    {
        Vec3 l   = p - x;
        float d2 = l.length2();
        Vec3 dir = l.normalize();
        return d2 / (a * fabsf(dir.dot(n)));
    }

#if OSL_USE_OPTIX
    virtual void setOptixVariables(void* data) const
    {
        QuadParams* quad_data = reinterpret_cast<QuadParams*>(data);
        quad_data->p          = make_float3(p.x, p.y, p.z);
        quad_data->ex         = make_float3(ex.x, ex.y, ex.z);
        quad_data->ey         = make_float3(ey.x, ey.y, ey.z);
        quad_data->n          = make_float3(n.x, n.y, n.z);
        quad_data->eu         = eu;
        quad_data->ev         = ev;
        quad_data->a          = a;
        quad_data->shaderID   = shaderid();
    }
#endif

private:
    Vec3 p, ex, ey, n;
    float a, eu, ev;
};



struct Scene {
    void add_sphere(const Sphere& s) { spheres.push_back(s); }

    void add_quad(const Quad& q) { quads.push_back(q); }

    int num_prims() const { return spheres.size() + quads.size(); }

    bool intersect(const Ray& r, Dual2<float>& t, int& primID) const
    {
        const int ns   = spheres.size();
        const int nq   = quads.size();
        const int self = primID;  // remember which object we started from
        t              = std::numeric_limits<float>::infinity();
        primID         = -1;  // reset ID
        for (int i = 0; i < ns; i++) {
            Dual2<float> d = spheres[i].intersect(r, self == i);
            if (d.val() > 0 && d.val() < t.val()) {  // found valid hit?
                t      = d;
                primID = i;
            }
        }
        for (int i = 0; i < nq; i++) {
            Dual2<float> d = quads[i].intersect(r, self == (i + ns));
            if (d.val() > 0 && d.val() < t.val()) {  // found valid hit?
                t      = d;
                primID = i + ns;
            }
        }
        return primID >= 0;
    }

    Vec3 sample(int primID, const Vec3& x, float xi, float yi, float& pdf) const
    {
        if (primID < int(spheres.size()))
            return spheres[primID].sample(x, xi, yi, pdf);
        primID -= spheres.size();
        return quads[primID].sample(x, xi, yi, pdf);
    }

    float shapepdf(int primID, const Vec3& x, const Vec3& p) const
    {
        if (primID < int(spheres.size()))
            return spheres[primID].shapepdf(x, p);
        primID -= spheres.size();
        return quads[primID].shapepdf(x, p);
    }

    float surfacearea(int primID) const
    {
        if (primID < int(spheres.size()))
            return spheres[primID].surfacearea();
        primID -= spheres.size();
        return quads[primID].surfacearea();
    }

    Dual2<Vec3> normal(const Dual2<Vec3>& p, int primID) const
    {
        if (primID < int(spheres.size()))
            return spheres[primID].normal(p);
        primID -= spheres.size();
        return quads[primID].normal(p);
    }

    Dual2<Vec2> uv(const Dual2<Vec3>& p, const Dual2<Vec3>& n, Vec3& dPdu,
                   Vec3& dPdv, int primID) const
    {
        if (primID < int(spheres.size()))
            return spheres[primID].uv(p, n, dPdu, dPdv);
        primID -= spheres.size();
        return quads[primID].uv(p, n, dPdu, dPdv);
    }

    int shaderid(int primID) const
    {
        if (primID < int(spheres.size()))
            return spheres[primID].shaderid();
        primID -= spheres.size();
        return quads[primID].shaderid();
    }

    bool islight(int primID) const
    {
        if (primID < int(spheres.size()))
            return spheres[primID].islight();
        primID -= spheres.size();
        return quads[primID].islight();
    }

    std::vector<Sphere> spheres;
    std::vector<Quad> quads;
};

OSL_NAMESPACE_EXIT
