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
    TangentFrame(const Vec3& n) : w(n) {
        u = (fabsf(w.x) >.01f ? Vec3(w.z, 0, -w.x) :
                                Vec3(0, -w.z, w.y)).normalize();
        v = w.cross(u);
    }

    // build frame from unit normal and unit tangent
    TangentFrame(const Vec3& n, const Vec3& t) : w(n) {
        v = w.cross(t);
        u = v.cross(w);
    }

    // transform vector
    Vec3 get(float x, float y, float z) const {
        return x * u + y * v + z * w;
    }

    // untransform vector
    float getx(const Vec3& a) const { return a.dot(u); }
    float gety(const Vec3& a) const { return a.dot(v); }
    float getz(const Vec3& a) const { return a.dot(w); }

    Vec3 tolocal(const Vec3 &a) const {
      return Vec3(a.dot(u), a.dot(v), a.dot(w));
    }
    Vec3 toworld(const Vec3 &a) const {
      return get(a.x, a.y, a.z);
    }

private:
    Vec3 u, v, w;
};

struct Sampling {
    /// Warp the unit disk onto the unit sphere
    /// http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric-map.html
    static void to_unit_disk(float& x, float& y) {
        const float PI_OVER_4 = float(M_PI_4);
        const float PI_OVER_2 = float(M_PI_2);
        float phi, r;
        float a = 2 * x - 1;
        float b = 2 * y - 1;
        if (a * a > b * b) { // use squares instead of absolute values
            r = a;
            phi = PI_OVER_4 * (b / a);
        } else if (b != 0) { // b is largest
            r = b;
            phi = PI_OVER_2 - PI_OVER_4 * (a / b);
        } else { // a == b == 0
            r = 0;
            phi = 0;
        }
        OIIO::fast_sincos(phi, &x, &y);
        x *= r;
        y *= r;
    }

    static void sample_cosine_hemisphere(const Vec3& N, float rndx, float rndy, Vec3& out, float& pdf) {
        to_unit_disk(rndx, rndy);
        float cos_theta = sqrtf(std::max(1 - rndx * rndx - rndy * rndy, 0.0f));
        TangentFrame f(N);
        out = f.get(rndx, rndy, cos_theta);
        pdf = cos_theta * float(M_1_PI);
    }
};

// Multiple Importance Sampling helper functions
struct MIS {
    // for the function below, enumerate the cases for:
    // the sampled function being a weight or eval,
    // the "other" function being a weight or eval
    enum MISMode {
        WEIGHT_WEIGHT,
        WEIGHT_EVAL,
        EVAL_WEIGHT
    };

    // Evaluates the weight factor for doing MIS when computing a product of two
    // functions such as light * brdf.
    // Provides options depending how the functions being multiplied together are
    // expressed (controlled by the enum above).
    // Centralizing the handling of the pdfs this way ensures that all numerical
    // cases can be enumerated and handled robustly without arbitrary epsilons.
    template <MISMode mode>
    static inline float power_heuristic(float sampled_pdf, float other_pdf)
    {
        // NOTE: inf is ok!
        assert(sampled_pdf >= 0);
        assert(  other_pdf >= 0);

        float r, mis;
        if (sampled_pdf > other_pdf) {
            r = other_pdf / sampled_pdf;
            mis = 1 / (1 + r * r);
        } else if (sampled_pdf < other_pdf) {
            r = sampled_pdf / other_pdf;
            mis = 1 - 1 / (1 + r * r);
        } else {
            // avoid (possible, but extremely rare) inf/inf cases
            assert(sampled_pdf == other_pdf);
            r = 1.0f;
            mis = 0.5f;
        }
        assert(r >= 0);
        assert(r <= 1);
        assert(mis >= 0);
        assert(mis <= 1);
        const float MAX = std::numeric_limits<float>::max();
        switch (mode) {
            case WEIGHT_WEIGHT: return std::min(other_pdf, MAX) * mis; // avoid inf * 0
            case WEIGHT_EVAL:   return mis;
            case EVAL_WEIGHT:   return mis * ((other_pdf > sampled_pdf) ? std::min(1 / r, MAX) : r); // NOTE: mis goes to 0 faster than 1/r goes to inf
        }
        return 0;
    }

    // Encapsulates the balance heuristic when evaluating a sum of functions
    // such as a BRDF mixture. This updates a (weight, pdf) pair with a new one
    // to represent the sum of both. b is the probability of choosing the provided
    // weight. A running sum should be started with a weight and pdf of 0.
    static inline void update_eval(Color3* w, float* pdf, Color3 ow, float opdf, float b)
    {
        // NOTE: inf is ok!
        assert(*pdf >= 0);
        assert(opdf >= 0);
        assert(b >= 0);
        assert(b <= 1);

        // make sure 1 / b is not inf
        // note that if the weight has components > 1 ow can still overflow, but
        // well designed BSDFs should keep weight <= 1
        if (b > std::numeric_limits<float>::min())
        {
            opdf *= b;
            ow *= 1 / b;
            float mis;
            if (*pdf < opdf)
                mis = 1 / (1 + *pdf / opdf);
            else if (opdf < *pdf)
                mis = 1 - 1 / (1 + opdf / *pdf);
            else
                mis = 0.5f; // avoid (rare) inf/inf

            *w = *w * (1 - mis) + ow * mis;
            *pdf += opdf;
        }

        assert(*pdf >= 0);
    }
};

// Simple stratified progressive sampling using owen scrambled sobol points.
// Code is written for clarity and simplicity over maximum speed.
struct Sampler {
    Sampler(int px, int py, int si) :
        seed(((px & 2047) << 22) | ((py & 2047) << 11)), si(si) { assert(si < (1 << 24)); }

    Vec3 get() {
        static const uint32_t zmatrix[24] = { // 2^24 precision
            0x800000u, 0xc00000u, 0x600000u, 0x900000u, 0xe80000u, 0x5c0000u, 0x8e0000u, 0xc50000u,
            0x688000u, 0x9cc000u, 0xee6000u, 0x559000u, 0x806800u, 0xc09c00u, 0x60ee00u, 0x905500u,
            0xe88080u, 0x5cc0c0u, 0x8e6060u, 0xc59090u, 0x6868e8u, 0x9c9c5cu, 0xeeee8eu, 0x5555c5u,
        };
        seed += 4; // advance depth for next call
        uint32_t index = progressive_permute(si, hash(seed - 4));
        uint32_t px = 0, py = 0, pz = 0, dx = 0x800000u, dy = 0x800000u;
        for (int c = 0; index; c++, index >>= 1) {
            if (index & 1) {
                px ^= dx;
                py ^= dy;
                pz ^= zmatrix[c];
            }
            dx >>= 1; dy ^= dy >> 1;
        } // scramble and scale by 2^-24
        return { owen_scramble(px, seed - 3) * 5.96046448e-08f,
                 owen_scramble(py, seed - 2) * 5.96046448e-08f,
                 owen_scramble(pz, seed - 1) * 5.96046448e-08f };
    }

private:
    uint32_t seed, si;

    static uint32_t hash(uint32_t s) {
        // https://nullprogram.com/blog/2018/07/31/
        s ^= s >> 16;
        s *= 0x7feb352du;
        s ^= s >> 15;
        s *= 0x846ca68bu;
        s ^= s >> 16;
        return s;
    }
    static uint32_t progressive_permute(uint32_t si, uint32_t p) {
        // shuffle order of points in power of 2 blocks
        if (si < 4) return cmj_permute(si, 4, p);
        uint32_t l = si;
        l = l | (l >> 1);
        l = l | (l >> 2);
        l = l | (l >> 4);
        l = l | (l >> 8);
        l = l | (l >> 16);
        l = l - (l >> 1);
        return cmj_permute(si - l, l, p) + l;
    }
    static inline uint32_t cmj_permute(uint32_t i, uint32_t l, uint32_t p) {
        // in-place random permutation (power of 2), see:
        // "Correlated Multi-Jittered Sampling" by "Andrew Kensler"
        const uint32_t w = l - 1; assert((l & w) == 0);
        i ^= p;
        i *= 0xe170893d;
        i ^= p >> 16;
        i ^= (i & w) >> 4;
        i ^= p >> 8;
        i *= 0x0929eb3f;
        i ^= p >> 23;
        i ^= (i & w) >> 1;
        i *= 1 | p >> 27;
        i *= 0x6935fa69;
        i ^= (i & w) >> 11;
        i *= 0x74dcb303;
        i ^= (i & w) >> 2;
        i *= 0x9e501cc3;
        i ^= (i & w) >> 2;
        i *= 0xc860a3df;
        i &= w;
        i ^= i >> 5;
        return (i + p) & w;
    }
    static uint32_t owen_scramble(uint32_t p, uint32_t s) {
        for (uint32_t m = 1u << 23; m; m >>= 1) {
            s = hash(s); // randomize state
            p ^= s & m;  // flip output (depending on state)
            s ^= p & m;  // flip state  (depending on output)
        }
        return p;
    }
};


OSL_NAMESPACE_EXIT
