// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// The code from "Ray Tracing in One Weekend" (https://raytracing.github.io)
// that knows nothing about OSL.

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

////////////////////////////////// UTILITIES ///////////////////////////////////

// Constants

inline constexpr double infinity = std::numeric_limits<double>::infinity();
inline constexpr double pi       = 3.1415926535897932385;

// Utility Functions

inline double
degrees_to_radians(double degrees)
{
    return degrees * pi / 180.0;
}

// Each rendering thread gets an independent generator.
inline thread_local uint64_t random_state = 0;

inline void
seed_random_generator(unsigned int seed)
{
    random_state = seed;
}

// Returns a random real in [0,1). This is SplitMix64 rather than <random>,
// whose distributions are not guaranteed to give the same numbers in every
// standard library. That way every platform renders with the same numbers.
inline double
random_double()
{
    uint64_t z = (random_state += 0x9e3779b97f4a7c15ull);
    z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z          = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    // Keep the top 53 bits, which is all a double's mantissa can hold.
    return static_cast<double>(z >> 11) * 0x1.0p-53;
}

// Returns a random real in [min,max).
inline double
random_double(double min, double max)
{
    return min + (max - min) * random_double();
}



///////////////////////////////////// VEC3 /////////////////////////////////////

class vec3 {
public:
    double e[3];

    vec3() : e { 0, 0, 0 } {}
    vec3(double e0, double e1, double e2) : e { e0, e1, e2 } {}

    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }

    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    double operator[](int i) const { return e[i]; }
    double& operator[](int i) { return e[i]; }

    vec3& operator+=(const vec3& v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    vec3& operator*=(double t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    vec3& operator/=(double t) { return *this *= 1 / t; }

    double length() const { return std::sqrt(length_squared()); }

    double length_squared() const
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    bool near_zero() const
    {
        constexpr double s = 1e-8;
        return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s)
               && (std::fabs(e[2]) < s);
    }

    static vec3 random(double min, double max)
    {
        double x = random_double(min, max);
        double y = random_double(min, max);
        double z = random_double(min, max);
        return vec3(x, y, z);
    }
};

// Aliases of vec3, for clarity.
using point3 = vec3;
using color  = vec3;

// Vector Utility Functions

inline vec3
operator+(const vec3& u, const vec3& v)
{
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3
operator-(const vec3& u, const vec3& v)
{
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline vec3
operator*(const vec3& u, const vec3& v)
{
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline vec3
operator*(double t, const vec3& v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline vec3
operator*(const vec3& v, double t)
{
    return t * v;
}

inline vec3
operator/(const vec3& v, double t)
{
    return (1 / t) * v;
}

inline double
dot(const vec3& u, const vec3& v)
{
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

inline vec3
cross(const vec3& u, const vec3& v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline vec3
unit_vector(const vec3& v)
{
    return v / v.length();
}

inline vec3
random_unit_vector()
{
    while (true) {
        vec3 p       = vec3::random(-1, 1);
        double lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1)
            return p / std::sqrt(lensq);
    }
}

inline vec3
reflect(const vec3& v, const vec3& n)
{
    return v - 2 * dot(v, n) * n;
}

inline vec3
refract(const vec3& uv, const vec3& n, double etai_over_etat)
{
    double cos_theta = std::fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp  = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel
        = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}



///////////////////////////////////// RAY //////////////////////////////////////

class ray {
public:
    ray() = default;

    ray(const point3& origin, const vec3& direction)
        : orig(origin), dir(direction)
    {
    }

    const point3& origin() const { return orig; }
    const vec3& direction() const { return dir; }

    point3 at(double t) const { return orig + t * dir; }

private:
    point3 orig;
    vec3 dir;
};



/////////////////////////////////// INTERVAL ///////////////////////////////////

class interval {
public:
    double min, max;

    // Default interval is empty
    interval() : min(+infinity), max(-infinity) {}

    interval(double min, double max) : min(min), max(max) {}

    double size() const { return max - min; }

    bool contains(double x) const { return min <= x && x <= max; }

    bool surrounds(double x) const { return min < x && x < max; }
};



/////////////////////////////////// HITTABLE ///////////////////////////////////

// Defined in osl/material.h, since it is the class that runs OSL shaders.
class material;

class hit_record {
public:
    point3 p;
    vec3 normal;
    std::shared_ptr<material> mat;
    double t        = 0;
    bool front_face = false;

    // Surface parameterization, passed on to the shader globals.
    double u = 0;
    double v = 0;
    vec3 dpdu;
    vec3 dpdv;

    void set_face_normal(const ray& r, const vec3& outward_normal)
    {
        // `outward_normal` is assumed to have unit length.
        front_face = dot(r.direction(), outward_normal) < 0;
        normal     = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    virtual ~hittable() = default;

    [[nodiscard]] virtual bool hit(const ray& r, interval ray_t,
                                   hit_record& rec) const = 0;
};


//////////////////////////////// HITTABLE_LIST /////////////////////////////////

class hittable_list : public hittable {
public:
    std::vector<std::shared_ptr<hittable>> objects;

    hittable_list() = default;

    void add(std::shared_ptr<hittable> object)
    {
        objects.push_back(std::move(object));
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override
    {
        hit_record temp_rec;
        bool hit_anything     = false;
        double closest_so_far = ray_t.max;

        for (const auto& object : objects) {
            if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything   = true;
                closest_so_far = temp_rec.t;
                rec            = temp_rec;
            }
        }

        return hit_anything;
    }
};



//////////////////////////////////// SPHERE ////////////////////////////////////

class sphere : public hittable {
public:
    sphere(const point3& center, double radius, std::shared_ptr<material> mat)
        : center(center), radius(std::fmax(0, radius)), mat(std::move(mat))
    {
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override
    {
        vec3 oc  = center - r.origin();
        double a = r.direction().length_squared();
        double h = dot(r.direction(), oc);
        double c = oc.length_squared() - radius * radius;

        double discriminant = h * h - a * c;
        if (discriminant < 0)
            return false;

        double sqrtd = std::sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        double root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t               = root;
        rec.p               = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;
        set_sphere_uv(outward_normal, rec);

        return true;
    }

private:
    // Spherical UVs and tangents at the unit outward normal `p`.
    // u = phi / (2*pi), v = theta / pi.
    void set_sphere_uv(const vec3& p, hit_record& rec) const
    {
        // Rounding can push this past 1 near a pole and make acos a NaN.
        double cos_theta = std::fmin(std::fmax(-p.y(), -1.0), 1.0);

        double theta = std::acos(cos_theta);
        double phi   = std::atan2(-p.z(), p.x()) + pi;

        rec.u = phi / (2 * pi);
        rec.v = theta / pi;

        double sin_theta = std::sin(theta);
        double sin_phi = std::sin(phi), cos_phi = std::cos(phi);

        rec.dpdu = 2 * pi * radius
                   * vec3(sin_theta * sin_phi, 0, sin_theta * cos_phi);
        rec.dpdv = pi * radius
                   * vec3(-cos_theta * cos_phi, sin_theta, cos_theta * sin_phi);
    }

    point3 center;
    double radius;
    std::shared_ptr<material> mat;
};
