#pragma once

#include "OSL/oslconfig.h"
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

// Simple deep stratified sampling using randomly shuffled LP points.
// There are better ways to apply such constructions, but this is
// reasonably compact and better than pure monte carlo sampling.
struct Sampler {
	Sampler(int px, int py, int si, int AA) : px(px), py(py), si(si), AA(AA), depth(0) {}

	Vec3 get() {
	    const uint32_t scramble_x = depth ? scramble(px, py, depth + 0) : 0;
	    const uint32_t scramble_y = depth ? scramble(px, py, depth + 1) : 0;
	    const uint32_t scramble_z = depth ? scramble(px, py, depth + 2) : 0;
	    const uint32_t sample_idx = depth ? cmj_permute(si, AA * AA,
	    									scramble(px, py, depth + 3)) : si;
	    depth += 4; // advance depth for next call
	    // fetch offset of scrambled LP pattern over the frame
	    const int sx = sample_idx % AA;
	    const int sy = sample_idx / AA;
	    const uint32_t ex = (px * AA + sx) & 65535;
	    const uint32_t ey = (py * AA + sy) & 65535;
	    const uint32_t upper = (ex ^ (scramble_x >> 16)) << 16;
	    const uint32_t lpUpper = ri_LP(upper) ^ scramble_y;
	    const uint32_t delta = (ey << 16) ^ (lpUpper & 0xFFFF0000u);
	    const uint32_t lower = ri_LP_inv(delta);
	    const uint32_t index = upper | lower;
	    const uint32_t x = index ^ scramble_x;
	    const uint32_t y = lpUpper ^ delta;
	    const float jx = (x - (ex << 16)) * (1 / 65536.0f);
	    const float jy = (y - (ey << 16)) * (1 / 65536.0f);
        uint32_t rz = scramble_z, ii = index;
        for (uint64_t v2 = uint64_t(3) << 62; ii; ii >>= 1, v2 ^= v2 >> 1)
            if (ii & 1)
                rz ^= uint32_t(v2 >> 31);

	    return Vec3((sx + jx) / AA, (sy + jy) / AA, rz * 2.3283063e-10f);
	}

private:
	int px, py, si, AA, depth;

	static inline uint32_t cmj_permute(uint32_t i, uint32_t l, uint32_t p) {
		// in-place permuation generator
		// "Correlated Multi-Jittered Sampling" by "Andrew Kensler"
	    uint32_t w = l - 1;
	    if ((l & w) == 0) {
	        /* l is a power of two (fast) */
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
	    } else {
	        /* l is not a power of two (slow) */
	        w |= w >> 1;
	        w |= w >> 2;
	        w |= w >> 4;
	        w |= w >> 8;
	        w |= w >> 16;
	        do {
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
	        } while (i >= l);
	        return (i + p) % l;
	    }
	}

	static inline uint32_t ri_LP(uint32_t i) {
	    uint32_t r = 0;
	    for (uint32_t v = 1U << 31; i; i >>= 1, v |= v >> 1)
	        if (i & 1)
	            r ^= v;
	    return r;
	}

	static inline uint32_t ri_LP_inv(uint32_t i) {
	    uint32_t r = 0;
	    for (uint32_t v = 3U << 30; i; i >>= 1, v >>= 1)
	        if (i & 1)
	            r ^= v;
	    return r;
	}

	static inline uint32_t scramble(uint32_t a, uint32_t b, uint32_t c) {
	    const int len = 3;
	    const int seed = (0xdeadbeef + (len << 2) + 13);
	    return OIIO::bjhash::bjfinal(a + seed, b + seed, c + seed);
	}
};


OSL_NAMESPACE_EXIT
