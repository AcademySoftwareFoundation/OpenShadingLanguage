#pragma once

#include <OpenImageIO/fmath.h>

#include "OSL/dual_vec.h"
#include "OSL/oslconfig.h"
#include <vector>

OSL_NAMESPACE_ENTER

struct Ray {
    Ray(const Dual2<Vec3>& o, const Dual2<Vec3>& d) : o(o), d(d) {}

    Vec3 point(float t) const {
        return o.val() + d.val() * t;
    }

    Dual2<Vec3> point(Dual2<float> t) const {
        return o + d * t;
    }

    Dual2<Vec3> o, d;
};

// build two vectors orthogonal to the first, assumes n is normalized
void ortho(const Vec3&n, Vec3& x, Vec3& y) {
    x = (fabsf(n.x) >.01f ? Vec3(n.z, 0, -n.x) : Vec3(0, -n.z, n.y)).normalize();
    y = n.cross(x);
}

struct Camera {
    Camera() {} // leave uninitialized
    Camera(Vec3 eye, Vec3 dir, Vec3 up, float fov, int w, int h) :
        eye(eye),
        dir(dir.normalize()),
        invw(1.0f / w), invh(1.0f / h) {
        float k = tanf(fov * float(M_PI / 360));
        Vec3 right = dir.cross(up).normalize();
        cx = right * (w * k / h);
        cy = (cx.cross(dir)).normalize() * k;
    }

    Ray get(float x, float y) const {
        Dual2<Vec3> v = cx * (Dual2<float>(x, 1, 0) * invw - 0.5f) +
                        cy * (0.5f - Dual2<float>(y, 0, -1) * invh) +
                        dir;
        return Ray(eye, normalize(v));
    }

private:
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
    Sphere(Vec3 c, float r, int shaderID, bool isLight) : Primitive(shaderID, isLight), c(c), r2(r * r) {
        ASSERT(r > 0);
    }

    // returns distance to nearest hit or 0
    Dual2<float> intersect(const Ray &r, bool self) const {
        Dual2<Vec3>  oc = c - r.o;
        Dual2<float> b = dot(oc, r.d);
        Dual2<float> det = b * b - dot(oc, oc) + r2;
        if (det.val() >= 0) {
            det = sqrt(det);
            Dual2<float> x = b - det;
            Dual2<float> y = b + det;
            return self ? (fabsf(x.val())>fabsf(y.val())?(x.val()>0?x:0):(y.val()>0?y:0))
                        : (x.val()>0?x:(y.val()>0?y:0));
        }
        return 0; // no hit
    }

    float surfacearea() const {
        return float(M_PI) * r2;
    }

    Dual2<Vec3> normal(const Dual2<Vec3>& p) const {
        return normalize(p - c);
    }

    Dual2<Vec2> uv(const Dual2<Vec3>& p, const Dual2<Vec3>& n, Vec3& dPdu, Vec3& dPdv) const {
        Dual2<float> nx(n.val().x, n.dx().x, n.dy().x);
        Dual2<float> ny(n.val().y, n.dx().y, n.dy().y);
        Dual2<float> nz(n.val().z, n.dx().z, n.dy().z);
        Dual2<float> u = (atan2(nx, nz) + Dual2<float>(M_PI)) * 0.5f * float(M_1_PI);
        Dual2<float> v = safe_acos(ny) * float(M_1_PI);
        float xz2 = nx.val() * nx.val() + nz.val() * nz.val();
        if (xz2 > 0) {
            const float PI = float(M_PI);
            const float TWOPI = float(2 * M_PI);
            float xz = sqrtf(xz2);
            float inv = 1 / xz;
            dPdu.x = -TWOPI * nx.val();
            dPdu.y = TWOPI * nz.val();
            dPdu.z = 0;
            dPdv.x = -PI * nz.val() * inv * ny.val();
            dPdv.y = -PI * nx.val() * inv * ny.val();
            dPdv.z = PI * xz;
        } else {
            // pick arbitrary axes for poles to avoid division by 0
            if (ny.val() > 0) {
                dPdu = Vec3(0, 0, 1);
                dPdv = Vec3(1, 0, 0);
            } else {
                dPdu = Vec3( 0, 0, 1);
                dPdv = Vec3(-1, 0, 0);
            }
        }
        return make_Vec2(u, v);
    }

    // return a direction towards a point on the sphere
    Vec3 sample(const Vec3& x, float xi, float yi, float& invpdf) const {
        const float TWOPI = float(2 * M_PI);
        float cmax2 = 1 - r2 / (c - x).length2();
        float cmax = cmax2>0 ? sqrtf(cmax2) : 0;
        float cos_a = 1 - xi + xi * cmax;
        float sin_a = sqrtf(1 - cos_a * cos_a);
        float phi = TWOPI * yi;
        Vec3 sw = (c - x).normalize(), su, sv;
        ortho(sw, su, sv);
        invpdf = TWOPI * (1 - cmax);
        return (su * (cosf(phi) * sin_a) +
                sv * (sinf(phi) * sin_a) +
                sw * cos_a).normalize();
    }

    float shapepdf(const Vec3& x, const Vec3& p) const {
        const float TWOPI = float(2 * M_PI);
        float cmax2 = 1 - r2 / (c - x).length2();
        float cmax = cmax2>0 ? sqrtf(cmax2) : 0;
        return 1 / (TWOPI * (1 - cmax));
    }

private:
    Vec3  c;
    float r2;
};

struct Quad : public Primitive {
    Quad(const Vec3& p, const Vec3& ex, const Vec3& ey, int shaderID, bool isLight) : Primitive(shaderID, isLight), p(p), ex(ex), ey(ey) {
        n = ex.cross(ey);
        a = n.length();
        n = n.normalize();
        eu = 1 / ex.length2();
        ev = 1 / ey.length2();
    }

    // returns distance to nearest hit or 0
    Dual2<float> intersect(const Ray &r, bool self) const {
        if (self) return 0;
        Dual2<float> dn = dot(r.d, n);
        Dual2<float> en = dot(p - r.o, n);
        if (dn.val() * en.val() > 0) {
            Dual2<float> t = en / dn;
            Dual2<Vec3>  h = r.point(t) - p;
            Dual2<float> dx = dot(h, ex) * eu;
            Dual2<float> dy = dot(h, ey) * ev;
            if (dx.val() >= 0 && dx.val() < 1 && dy.val() >= 0 && dy.val() < 1)
                return t;
        }
        return 0; // no hit
    }

    float surfacearea() const {
        return a;
    }

    Dual2<Vec3> normal(const Dual2<Vec3>& p) const {
        return Dual2<Vec3>(n, Vec3(0, 0, 0), Vec3(0, 0, 0));
    }

    Dual2<Vec2> uv(const Dual2<Vec3>& p, const Dual2<Vec3>& n, Vec3& dPdu, Vec3& dPdv) const {
        Dual2<Vec3>  h = p - this->p;
        Dual2<float> u = dot(h, ex) * eu;
        Dual2<float> v = dot(h, ey) * ev;
        dPdu = ex;
        dPdv = ey;
        return make_Vec2(u, v);
    }

    // return a direction towards a point on the sphere
    Vec3 sample(const Vec3& x, float xi, float yi, float& invpdf) const {
        Vec3 l = (p + xi * ex + yi * ey) - x;
        float d2 = l.length2();
        Vec3 dir = l.normalize();
        invpdf = a * fabsf(dir.dot(n)) / d2;
        return dir;
    }

    float shapepdf(const Vec3& x, const Vec3& p) const {
        Vec3 l = p - x;
        float d2 = l.length2();
        Vec3 dir = l.normalize();
        return d2 / (a * fabsf(dir.dot(n)));
    }

private:
    Vec3 p, ex, ey, n;
    float a, eu, ev;
};

struct Scene {
    void add_sphere(const Sphere& s) {
        spheres.push_back(s);
    }

    void add_quad(const Quad& q) {
        quads.push_back(q);
    }

    int num_prims() const {
        return spheres.size() + quads.size();
    }

    bool intersect(const Ray& r, Dual2<float>& t, int& primID) const {
        const int ns = spheres.size();
        const int nq = quads.size();
        const int self = primID; // remember which object we started from
        t = std::numeric_limits<float>::infinity();
        primID = -1; // reset ID
        for (int i = 0; i < ns; i++) {
            Dual2<float> d = spheres[i].intersect(r, self == i);
            if (d.val() > 0 && d.val() < t.val()) { // found valid hit?
                t = d;
                primID = i;
            }
        }
        for (int i = 0; i < nq; i++) {
            Dual2<float> d = quads[i].intersect(r, self == (i + ns));
            if (d.val() > 0 && d.val() < t.val()) { // found valid hit?
                t = d;
                primID = i + ns;
            }
        }
        return primID >= 0;
    }

    Vec3 sample(int primID, const Vec3& x, float xi, float yi, float& invpdf) const {
        if (primID < int(spheres.size()))
            return spheres[primID].sample(x, xi, yi, invpdf);
        primID -= spheres.size();
        return quads[primID].sample(x, xi, yi, invpdf);
    }

    float shapepdf(int primID, const Vec3& x, const Vec3& p) const {
        if (primID < int(spheres.size()))
            return spheres[primID].shapepdf(x, p);
        primID -= spheres.size();
        return quads[primID].shapepdf(x, p);
    }

    float surfacearea(int primID) const {
        if (primID < int(spheres.size()))
            return spheres[primID].surfacearea();
        primID -= spheres.size();
        return quads[primID].surfacearea();
    }

    Dual2<Vec3> normal(const Dual2<Vec3>& p, int primID) const {
        if (primID < int(spheres.size()))
            return spheres[primID].normal(p);
        primID -= spheres.size();
        return quads[primID].normal(p);
    }

    Dual2<Vec2> uv(const Dual2<Vec3>& p, const Dual2<Vec3>& n, Vec3& dPdu, Vec3& dPdv, int primID) const {
        if (primID < int(spheres.size()))
            return spheres[primID].uv(p, n, dPdu, dPdv);
        primID -= spheres.size();
        return quads[primID].uv(p, n, dPdu, dPdv);
    }

    int shaderid(int primID) const {
        if (primID < int(spheres.size()))
            return spheres[primID].shaderid();
        primID -= spheres.size();
        return quads[primID].shaderid();
    }

    bool islight(int primID) const {
        if (primID < int(spheres.size()))
            return spheres[primID].islight();
        primID -= spheres.size();
        return quads[primID].islight();
    }
private:
    std::vector<Sphere> spheres;
    std::vector<Quad> quads;
};

OSL_NAMESPACE_EXIT
