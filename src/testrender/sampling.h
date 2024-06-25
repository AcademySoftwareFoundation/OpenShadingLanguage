// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>
#include <OpenImageIO/fmath.h>
#include <OpenImageIO/hash.h>
#include <algorithm>
#include <cmath>

OSL_NAMESPACE_ENTER

struct TangentFrame {
    // build frame from unit normal
    static TangentFrame from_normal(const Vec3& n)
    {
        // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
        const float sign = copysignf(1.0f, n.z);
        const float a    = -1 / (sign + n.z);
        const float b    = n.x * n.y * a;
        const Vec3 u = Vec3(1 + sign * n.x * n.x * a, sign * b, -sign * n.x);
        const Vec3 v = Vec3(b, sign + n.y * n.y * a, -n.y);
        return { u, v, n };
    }

    // build frame from unit normal and unit tangent
    // fallsback to an arbitrary basis if the tangent is 0 or colinear with n
    static TangentFrame from_normal_and_tangent(const Vec3& n, const Vec3& t)
    {
        Vec3 x      = t - n * dot(n, t);
        float xlen2 = dot(x, x);
        if (xlen2 > 0) {
            x *= 1.0f / sqrtf(xlen2);
            return { x, n.cross(x), n };
        } else {
            // degenerate case, fallback to generic tangent frame
            return from_normal(n);
        }
    }

    // transform vector
    Vec3 get(float x, float y, float z) const { return x * u + y * v + z * w; }

    // untransform vector
    float getx(const Vec3& a) const { return a.dot(u); }
    float gety(const Vec3& a) const { return a.dot(v); }
    float getz(const Vec3& a) const { return a.dot(w); }

    Vec3 tolocal(const Vec3& a) const
    {
        return Vec3(a.dot(u), a.dot(v), a.dot(w));
    }
    Vec3 toworld(const Vec3& a) const { return get(a.x, a.y, a.z); }

    Vec3 u, v, w;
};

struct Sampling {
    /// Warp the unit disk onto the unit sphere
    /// http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric-map.html
    static void to_unit_disk(float& x, float& y)
    {
        const float PI_OVER_4 = float(M_PI_4);
        const float PI_OVER_2 = float(M_PI_2);
        float phi, r;
        float a = 2 * x - 1;
        float b = 2 * y - 1;
        if (a * a > b * b) {  // use squares instead of absolute values
            r   = a;
            phi = PI_OVER_4 * (b / a);
        } else if (b != 0) {  // b is largest
            r   = b;
            phi = PI_OVER_2 - PI_OVER_4 * (a / b);
        } else {  // a == b == 0
            r   = 0;
            phi = 0;
        }
        OIIO::fast_sincos(phi, &x, &y);
        x *= r;
        y *= r;
    }

    static void sample_cosine_hemisphere(const Vec3& N, float rndx, float rndy,
                                         Vec3& out, float& pdf)
    {
        to_unit_disk(rndx, rndy);
        float cos_theta = sqrtf(std::max(1 - rndx * rndx - rndy * rndy, 0.0f));
        out = TangentFrame::from_normal(N).get(rndx, rndy, cos_theta);
        pdf = cos_theta * float(M_1_PI);
    }

    static void sample_uniform_hemisphere(const Vec3& N, float rndx, float rndy,
                                          Vec3& out, float& pdf)
    {
        float phi       = float(2 * M_PI) * rndx;
        float cos_theta = rndy;
        float sin_theta = sqrtf(1 - cos_theta * cos_theta);
        out = TangentFrame::from_normal(N).get(sin_theta * cosf(phi),
                                               sin_theta * sinf(phi),
                                               cos_theta);
        pdf = float(0.5 * M_1_PI);
    }
};

// Multiple Importance Sampling helper functions
struct MIS {
    // for the function below, enumerate the cases for:
    // the sampled function being a weight or eval,
    // the "other" function being a weight or eval
    enum MISMode { WEIGHT_WEIGHT, WEIGHT_EVAL, EVAL_WEIGHT };

    // Evaluates the weight factor for doing MIS when computing a product of two
    // functions such as light * brdf.
    // Provides options depending how the functions being multiplied together are
    // expressed (controlled by the enum above).
    // Centralizing the handling of the pdfs this way ensures that all numerical
    // cases can be enumerated and handled robustly without arbitrary epsilons.
    template<MISMode mode>
    static inline float power_heuristic(float sampled_pdf, float other_pdf)
    {
        // NOTE: inf is ok!
        assert(sampled_pdf >= 0);
        assert(other_pdf >= 0);

        float r, mis;
        if (sampled_pdf > other_pdf) {
            r   = other_pdf / sampled_pdf;
            mis = 1 / (1 + r * r);
        } else if (sampled_pdf < other_pdf) {
            r   = sampled_pdf / other_pdf;
            mis = 1 - 1 / (1 + r * r);
        } else {
            // avoid (possible, but extremely rare) inf/inf cases
            assert(sampled_pdf == other_pdf);
            r   = 1.0f;
            mis = 0.5f;
        }
        assert(r >= 0);
        assert(r <= 1);
        assert(mis >= 0);
        assert(mis <= 1);
        const float MAX = std::numeric_limits<float>::max();
        switch (mode) {
        case WEIGHT_WEIGHT:
            return std::min(other_pdf, MAX) * mis;  // avoid inf * 0
        case WEIGHT_EVAL: return mis;
        case EVAL_WEIGHT:
            return mis
                   * ((other_pdf > sampled_pdf)
                          ? std::min(1 / r, MAX)
                          : r);  // NOTE: mis goes to 0 faster than 1/r goes to inf
        }
        return 0;
    }

    // Encapsulates the balance heuristic when evaluating a sum of functions
    // such as a BRDF mixture. This updates a (weight, pdf) pair with a new one
    // to represent the sum of both. b is the probability of choosing the provided
    // weight. A running sum should be started with a weight and pdf of 0.
    static inline void update_eval(Color3* w, float* pdf, Color3 ow, float opdf,
                                   float b)
    {
        // NOTE: inf is ok!
        assert(*pdf >= 0);
        assert(opdf >= 0);
        assert(b >= 0);
        assert(b <= 1);

        // make sure 1 / b is not inf
        // note that if the weight has components > 1 ow can still overflow, but
        // well designed BSDFs should keep weight <= 1
        if (b > std::numeric_limits<float>::min()) {
            opdf *= b;
            ow *= 1 / b;
            float mis;
            if (*pdf < opdf)
                mis = 1 / (1 + *pdf / opdf);
            else if (opdf < *pdf)
                mis = 1 - 1 / (1 + opdf / *pdf);
            else
                mis = 0.5f;  // avoid (rare) inf/inf

            *w = *w * (1 - mis) + ow * mis;
            *pdf += opdf;
        }

        assert(*pdf >= 0);
    }
};

// "Practical Hash-based Owen Scrambling" - Brent Burley - JCGT 2020
//    https://jcgt.org/published/0009/04/01/
struct Sampler {
    Sampler(int px, int py, int si)
        : seed(((px & 2047) << 22) | ((py & 2047) << 11))
        , index(reversebits(si))
    {
        assert(si < (1 << 24));
    }

    Vec3 get()
    {
        static const uint32_t zmatrix[24] = {
            // 2^24 precision (reversed)
            0x000001u, 0x000003u, 0x000006u, 0x000009u, 0x000017u, 0x00003au,
            0x000071u, 0x0000a3u, 0x000116u, 0x000339u, 0x000677u, 0x0009aau,
            0x001601u, 0x003903u, 0x007706u, 0x00aa09u, 0x010117u, 0x03033au,
            0x060671u, 0x0909a3u, 0x171616u, 0x3a3939u, 0x717777u, 0xa3aaaau
        };
        seed += 4;  // advance depth for next call
        uint32_t scrambled_index = owen_scramble(index, hash(seed - 4))
                                   & 0xFFFFFF;
        uint32_t result_x = scrambled_index;  // already reversed
        uint32_t result_y = 0;
        uint32_t result_z = 0;
        uint32_t ymatrix  = 1;
        for (int c = 0; c < 24; c++) {
            uint32_t bit = (scrambled_index >> c) & 1;
            result_y ^= bit * ymatrix;
            result_z ^= bit * zmatrix[c];
            ymatrix ^= ymatrix
                       << 1;  // generate procedurally instead of storing this
        }
        // scramble results and scale by 2^-24 to guarantee equally spaced values in [0,1)
        return {
            (owen_scramble(result_x, hash(seed - 3)) >> 8) * 5.96046448e-8f,
            (owen_scramble(result_y, hash(seed - 2)) >> 8) * 5.96046448e-8f,
            (owen_scramble(result_z, hash(seed - 1)) >> 8) * 5.96046448e-8f
        };
    }

private:
    uint32_t seed, index;

    static uint32_t hash(uint32_t s)
    {
        // https://github.com/skeeto/hash-prospector
        s ^= s >> 16;
        s *= 0x21f0aaadu;
        s ^= s >> 15;
        s *= 0xd35a2d97u;
        s ^= s >> 15;
        return s;
    }

    static uint32_t reversebits(uint32_t x)
    {
#if defined(__clang__)
        return __builtin_bitreverse32(x);
#else
        x = (x << 16) | (x >> 16);
        x = ((x & 0x00ff00ff) << 8) | ((x & 0xff00ff00) >> 8);
        x = ((x & 0x0f0f0f0f) << 4) | ((x & 0xf0f0f0f0) >> 4);
        x = ((x & 0x33333333) << 2) | ((x & 0xcccccccc) >> 2);
        x = ((x & 0x55555555) << 1) | ((x & 0xaaaaaaaa) >> 1);
        return x;
#endif
    }

    static uint32_t owen_scramble(uint32_t p, uint32_t s)
    {
        // https://psychopath.io/post/2021_01_30_building_a_better_lk_hash
        // assumes reversed input
        p ^= p * 0x3d20adea;
        p += s;
        p *= (s >> 16) | 1;
        p ^= p * 0x05526c56;
        p ^= p * 0x53a22864;
        return reversebits(p);
    }
};


OSL_NAMESPACE_EXIT
