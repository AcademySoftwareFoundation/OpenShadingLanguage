// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/config.h>
#include <BSDL/spectrum_decl.h>

#include <tuple>

BSDL_ENTER_NAMESPACE

BSDL_INLINE float
MAX_ABS_XYZ(const Imath::V3f& v)
{
    return std::max(fabsf(v.x), std::max(fabsf(v.y), fabsf(v.z)));
}

BSDL_INLINE float
MAX_RGB(const Imath::C3f& c)
{
    return std::max(c.x, std::max(c.y, c.z));
}

BSDL_INLINE float
MAX_ABS_RGB(const Imath::C3f& c)
{
    return std::max(fabsf(c.x), std::max(fabsf(c.y), fabsf(c.z)));
}

BSDL_INLINE float
MIN_RGB(const Imath::C3f& c)
{
    return std::min(c.x, std::min(c.y, c.z));
}

BSDL_INLINE float
AVG_RGB(const Imath::C3f& c)
{
    return (c.x + c.y + c.z) * (1.0f / 3);
}

template<typename T>
BSDL_INLINE constexpr T
SQR(T x)
{
    return x * x;
}

BSDL_INLINE float
CLAMP(float x, float a, float b)
{
    return std::min(std::max(x, a), b);
}

BSDL_INLINE Imath::C3f
CLAMP(const Imath::C3f& c, float a, float b)
{
    return { CLAMP(c.x, a, b), CLAMP(c.y, a, b), CLAMP(c.z, a, b) };
}

template<typename T>
BSDL_INLINE T
LERP(float f, T a, T b)
{
    f = CLAMP(f, 0, 1);
    return (1 - f) * a + f * b;
}

// Hermite interpolation between 0 and 1 using 't' (0<=t<=1)
template<typename T>
BSDL_INLINE constexpr T
HERP01(T t)
{
    return t * t * (3 - 2 * t);
}

template<typename T>
constexpr T
LINEARSTEP(T lo, T hi, T t)
{
    return CLAMP((t - lo) / (hi - lo), T(0), T(1));
}

// RenderMan's smoothstep() function
// return 0 if (t < e0) or 1 if (t > e1) or
// a hermitian interpolation for (e0 < t < e1)
template<typename T>
BSDL_INLINE constexpr T
SMOOTHSTEP(T e0, T e1, T t)
{
    return (t <= e0)
               ? T(0)
               : ((t >= e1) ? T(1)
                            : CLAMP(HERP01((t - e0) / (e1 - e0)), T(0), T(1)));
}

// Return the sum of a and b but without exceeding smax, given that
// the two operands are in range. This is a smooth alternative to clamp
// and has the following properties:
//
//    sum_max(a, b, smax) == sum_max(b, a, smax)
//    sum_max(a, b, smax) >= a
//    sum_max(a, b, smax) >= b
//    sum_max(a, 0, smax) == a
//    sum_max(a, b, smax) <= smax
//
// When called like sum_max(x, x, 1.0) is equivalent to 2x - x^2
BSDL_INLINE float
sum_max(float a, float b, float smax)
{
    const float maxab = std::max(a, b), minab = std::min(a, b);
    return maxab + (smax - maxab) * (minab / smax);
}

// This parametric curve (k) is a tool for adjusting responses from a 0-1 input
// where the parameter k adjusts the shape.
//
//    Flat-zero   exp()-ish    Identity  Reverse exp()   Flat one
//                                                        _______
//           |         |           /          _,---      |
//           |         |          /          /           |
//    _______|    ____/          /          |            |
//
//     k = 0    k = 0.1       k = 0.5        k = 0.9       k = 1
//
// And it transitions smoothly from one extreme to the other with k in [0, 1]
// where k = 0.5 gives you the identity. Also gamma^-1(x, k) == gamma(x, 1 -k)
//
BSDL_INLINE float
bias_curve01(float x, float k)
{
    // From Christophe Schlick. “Fast Alternatives to Perlin’s Bias and Gain Functions”.
    // In Graphics Gems IV, Morgan Kaufmann, 1994, pages 401–403.
    return x / std::max((1 - x) * (1 / k - 2) + 1, FLOAT_MIN);
}

// Using the above bias curve, this is a simple sigmoid in [0, 1]
BSDL_INLINE float
gain_curve01(float x, float k)
{
    return x < 0.5f ? 0.5f * bias_curve01(2 * x, k)
                    : 0.5f * bias_curve01(2 * x - 1, 1 - k) + 0.5f;
}

// Function approximations below are from
// "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD" by Petrik Clarberg.
//
// http://fileadmin.cs.lth.se/graphics/research/papers/2008/simdmapping/clarberg_simdmapping08_preprint.pdf
//
// They are handy for the low distortion mapping to and from the unit disc/hemisphere/sphere
BSDL_INLINE float
fast_cos_quadrant(float x)
{
    assert(x >= -2);
    assert(x <= 2);
    // Coefficients for minimax approximation of cos(x*pi/4), x=[-2,2].
    constexpr float c1 = 0.99998736f;
    constexpr float c2 = -0.30837047f;
    constexpr float c3 = 0.01578646f;
    constexpr float c4 = -0.00029826362f;

    float x2 = x * x;
    float cp = c3 + c4 * x2;
    cp       = c2 + cp * x2;
    cp       = c1 + cp * x2;
    return cp;
}

BSDL_INLINE float
fast_sin_quadrant(float x)
{
    assert(x >= -2);
    assert(x <= 2);
    // Coefficients for minimax approximation of sin(x*pi/4), x=[0,2].
    const float s1 = 0.7853975892066955566406250000000000f;
    const float s2 = -0.0807407423853874206542968750000000f;
    const float s3 = 0.0024843954015523195266723632812500f;
    const float s4 = -0.0000341485538228880614042282104492f;

    float x2 = x * x;
    float sp = s3 + s4 * x2;
    sp       = s2 + sp * x2;
    sp       = s1 + sp * x2;
    return sp * x;
}

BSDL_INLINE float
fast_atan_quadrant(float x)
{
    assert(x >= -1);
    assert(x <= 1);
    // Coefficients for 6th degree minimax approximation of atan(x)*2/pi, x=[0,1].
    const float t1 = 0.406758566246788489601959989e-5f;
    const float t2 = 0.636226545274016134946890922156f;
    const float t3 = 0.61572017898280213493197203466e-2f;
    const float t4 = -0.247333733281268944196501420480f;
    const float t5 = 0.881770664775316294736387951347e-1f;
    const float t6 = 0.419038818029165735901852432784e-1f;
    const float t7 = -0.251390972343483509333252996350e-1f;

    // Polynomial approximation of atan(x)*2/pi
    float phi = t6 + t7 * x;
    phi       = t5 + phi * x;
    phi       = t4 + phi * x;
    phi       = t3 + phi * x;
    phi       = t2 + phi * x;
    phi       = t1 + phi * x;
    return phi;
}

BSDL_INLINE Imath::V3f
sample_cos_hemisphere(float randu, float randv)
{
    // stretch unit square + get quadrant
    const float a = 2 * randu - 1, qa = fabsf(a);  // (a,b) is now on [-1,1]^2
    const float b = 2 * randv - 1, qb = fabsf(b);
    // map to radius/angle
    const float rad = qa > qb ? qa : qb;
    const float phi = qa > qb ? qb / qa : ((qa == qb) ? 1.0f : 2 - qa / qb);
    // map to disk + flip back into right quadrant
    const float x = copysignf(rad * fast_cos_quadrant(phi), a);
    const float y = copysignf(rad * fast_sin_quadrant(phi), b);
    assert(rad <= 1);
    assert(fabsf(x) <= 1);
    assert(fabsf(y) <= 1);
    // map to cosine weighted hemisphere
    return { x, y, sqrtf(1 - rad * rad) };
}

BSDL_INLINE Imath::V3f
sample_uniform_hemisphere(float randu, float randv)
{
    // stretch unit square + get quadrant
    const float a = 2 * randu - 1, qa = fabsf(a);  // (a,b) is now on [-1,1]^2
    const float b = 2 * randv - 1, qb = fabsf(b);
    // map to radius/angle
    const float rad = qa > qb ? qa : qb;
    const float phi = qa > qb ? qb / qa : ((qa == qb) ? 1.0f : 2 - qa / qb);
    // map to disk + flip back into right quadrant
    const float x = copysignf(rad * fast_cos_quadrant(phi), a);
    const float y = copysignf(rad * fast_sin_quadrant(phi), b);
    assert(rad <= 1);
    assert(fabsf(x) <= 1);
    assert(fabsf(y) <= 1);
    // map to uniform hemisphere
    const float cos_theta = 1 - rad * rad;
    const float sin_theta = sqrtf(2 - rad * rad);
    return { sin_theta * x, sin_theta * y, cos_theta };
}

BSDL_INLINE Imath::V3f
reflect(const Imath::V3f& E, const Imath::V3f& N)
{
    return N * (2 * N.dot(E)) - E;
}

BSDL_INLINE Imath::V3f
refract(const Imath::V3f& E, const Imath::V3f& N, float eta)
{
    Imath::V3f R(0.0f);
    if (eta == 0)
        return R;

    Imath::V3f Nn;
    float cosi = E.dot(N), neta;
    // check which side of the surface we are on
    if (cosi > 0) {
        // we are on the outside of the surface, going in
        neta = 1 / eta;
        Nn   = N;
    } else {
        // we are inside the surface,
        cosi = -cosi;
        neta = eta;
        Nn   = -N;
    }

    float arg = 1 - (neta * neta * (1 - (cosi * cosi)));
    if (arg >= 0) {
        float dnp = sqrtf(arg);
        float nK  = (neta * cosi) - dnp;
        R         = (E * (-neta) + Nn * nK).normalized();
    }
    return R;
}

BSDL_INLINE Imath::V3f
rotate(const Imath::V3f& v, const Imath::V3f& axis, float angle)
{
    float s = BSDLConfig::Fast::sinf(angle), c = BSDLConfig::Fast::cosf(angle);
    return v * c + axis * v.dot(axis) * (1.f - c) + s * axis.cross(v);
}

BSDL_INLINE float
fresnel_dielectric(float cosi, float eta)
{
    if (eta == 0.0f)
        // Early exit for some reflectors that leave eta = 0
        // meaning no fresnel decay
        return 1.0f;
    // compute fresnel reflectance without explicitly computing
    // the refracted direction
    if (cosi < 0.0f)
        eta = 1.0f / eta;
    float c = fabsf(cosi);
    float g = eta * eta - 1 + c * c;
    if (g > 0) {
        g       = sqrtf(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1) / (c * (g - c) + 1);
        return 0.5f * A * A * (1 + B * B);
    }
    return 1.0f;  // TIR (no refracted component)
}

BSDL_INLINE float
avg_fresnel_dielectric(float eta)
{
#if 0
    // A quadrature is appliead to the interval [a, b] where a is the critical
    // angle of fresnel (its cosine). And b is always 1.0.
    const float a = eta < 1 ? sqrtf((1 - eta) * (1 + eta)) : 0;
    // now compute a quadrature for the [a, 1] interval
    // average fresnel (max error ~= 0.0009) for eta > 1, but about 0.04 under 1
    // This evaluates the integral F(eta)=2*integrate(fresnel(C, eta)*C,C,0,1)
    // using a 4 point gauss-legendre quadrature
    // http://www.efunda.com/math/num_integration/findgausslegendre.cfm
    const float h = (1 + a) / 2; // center of quadrature
    const float r = (1 - a) / 2; // radius
    const float xi[2] = { 0.861136311594f, 0.339981043585f };
    const float wi[2] = { 0.347854845137f, 0.652145154863f };
    float q = fresnel_dielectric(r * -xi[0] + h, eta) * ((r * -xi[0] + h) * wi[0]) +
              fresnel_dielectric(r * -xi[1] + h, eta) * ((r * -xi[1] + h) * wi[1]) +
              fresnel_dielectric(r *  xi[1] + h, eta) * ((r *  xi[1] + h) * wi[1]) +
              fresnel_dielectric(r *  xi[0] + h, eta) * ((r *  xi[0] + h) * wi[0]);
    // The average is cosine weighted. The average of F*x in [0 a) is 1, and q
    // is the avg in [a 1), now we need to compute the weighted average of the
    // two. Since we weight with cosine, the first interval weights the integral
    // of x from 0 to a (a^2/2) divided by the integral of x from 0 to 1 (1/2),
    // and the second interval weights the complement. Therefore:
    float avg = q * (1.0f - a * a) + a * a;
    assert(0 <= avg && avg <= 1.0f);
    return avg;
#elif 0
    // much simpler fit computed in mathematica:
    // max error for 0<=eta<=  1 is 0.003
    // max error for 1<=eta<=400 is 0.010
    if (eta < 1)
        return CLAMP((9.13734f - 0.00419542f * eta - 9.11295f * SQR(eta))
                         / (9.16567f - 0.974132f * eta),
                     0.0f, 1.0f);
    else
        return CLAMP((-2.78491f + 2.72524f * eta + 6.55885e-6f * SQR(eta))
                         / (10.9957f + 2.7292f * eta),
                     0.0f, 1.0f);
#else
    // even simpler fit with lower error (computed in mathematica)
    // max error for 0<=eta<=  1 is ~0.29%
    // max error for 1<=eta<=400 is ~0.65%
    if (eta < 1)
        return 0.997118f
               + eta * (0.1014f + eta * (-0.965241f - eta * 0.130607f));
    else
        return (eta - 1) / (4.08567f + 1.00071f * eta);
#endif
}

// Fast fresnel function for metals
//
// c is the angle cosine
// r is the reflectance
// g is the edge tint
BSDL_INLINE float
fresnel_metal(float c, float r, float g)
{
    // from: "Artist Friendly Metallic Fresnel", Ole Gulbrandsen
    // http://jcgt.org/published/0003/04/03/
    const float n = LERP(g, (1 + sqrtf(r)) / (1 - sqrtf(r)), (1 - r) / (1 + r));
    const float k2  = (SQR(n + 1) * r - SQR(n - 1)) / (1 - r);
    const float n2  = n * n;
    const float c2  = c * c;
    const float tnc = 2 * n * c;

    const float rs_num = (n2 + k2) - tnc + c2;
    const float rs_den = (n2 + k2) + tnc + c2;
    const float rp_num = (n2 + k2) * c2 - tnc + 1;
    const float rp_den = (n2 + k2) * c2 + tnc + 1;

    return 0.5f * (rs_num / rs_den + rp_num / rp_den);
}

BSDL_INLINE Power
fresnel_metal(float c, const Power r, const Power g, float lambda_0)
{
    return Power([&](int i) { return fresnel_metal(c, r[i], g[i]); }, lambda_0);
}

BSDL_INLINE Power
fresnel_schlick(float c, const Power r, const Power g, float p)
{
    constexpr auto fast_exp2 = BSDLConfig::Fast::exp2f;
    constexpr auto fast_log2 = BSDLConfig::Fast::log2f;

    c = std::min(c, 1 - 1e-4f);
    p = std::max(p, 1e-4f);
    return LERP(fast_exp2(fast_log2(1 - c) / p), r, g);
}

struct Frame {
    // Given a unit vector N, build two arbitrary orthogonal vectors U and V
    // The output is guarenteed to form a right handed orthonormal basis. (U x V = N)
    static BSDL_INLINE_METHOD std::tuple<Imath::V3f, Imath::V3f>
    ortho_build(const Imath::V3f& Z)
    {
        // http://jcgt.org/published/0006/01/01/
        //
        // Building an Orthonormal Basis, Revisited
        // Tom Duff, James Burgess, Per Christensen, Christophe Hery, Andrew Kensler, Max Liani, Ryusuke Villemin
        const float s = copysignf(1.0f, Z.z);
        const float a = -1.0f / (s + Z.z);
        const float b = Z.x * Z.y * a;
        Imath::V3f X  = { 1.0f + s * Z.x * Z.x * a, s * b, -s * Z.x };
        Imath::V3f Y  = { b, s + Z.y * Z.y * a, -Z.y };
        return { X, Y };
    }
    BSDL_INLINE_METHOD Frame(const Imath::V3f& Z) : Z(Z)
    {
        auto XY = ortho_build(Z);
        X       = std::get<0>(XY);
        Y       = std::get<1>(XY);
    }
    // frame with z axis pointing along n and x axis pointing in the same direction as u (but orthogonal)
    BSDL_INLINE_METHOD Frame(const Imath::V3f& Z, const Imath::V3f& _X)
        : X(_X), Z(Z)
    {
        if (MAX_ABS_XYZ(X) < 1e-4f) {
            // X not provided, pick arbitrary
            auto XY = ortho_build(Z);
            X       = std::get<0>(XY);
            Y       = std::get<1>(XY);
        } else {
            Y = Z.cross(X).normalized();
            X = Y.cross(Z);
        }
    }
    // take a world space vector and spin it around to be expressed in the
    // coordinate system of this frame
    BSDL_INLINE_METHOD Imath::V3f local(const Imath::V3f& a) const
    {
        return { a.dot(X), a.dot(Y), a.dot(Z) };
    }
    BSDL_INLINE_METHOD Imath::V3f world(const Imath::V3f& a) const
    {
        return { X.x * a.x + Y.x * a.y + Z.x * a.z,
                 X.y * a.x + Y.y * a.y + Z.y * a.z,
                 X.z * a.x + Y.z * a.y + Z.z * a.z };
    }

    Imath::V3f X, Y, Z;
};

// This transforms points on [0,1]^2 to points on unit disc centered at
// origin. Each "pie-slice" quadrant of square is handled as a separate
// case. The bad floating point cases are all handled appropriately.
// The regions for (a,b) are:
//
//                 phi = pi/2
//                -----*-----
//                |\       /|
//                |  \ 2 /  |
//                |   \ /   |
//         phi=pi * 3  *  1 * phi = 0
//                |   / \   |
//                |  / 4 \  |
//                |/       \|
//                -----*-----
//                phi = 3pi/2
//
// (rnd.x,rnd.y) is a point on [0,1]^2. (x,y) is point on radius 1 disc
//
BSDL_INLINE Imath::V2f
square_to_unit_disc(const Imath::V2f rnd)
{
    // assert(rnd.x >= 0);
    // assert(rnd.x <= 1);
    // assert(rnd.y >= 0);
    // assert(rnd.y <= 1);
    // stretch unit square + get quadrant
    const float a = 2 * rnd.x - 1, qa = fabsf(a);  // (a,b) is now on [-1,1]^2
    const float b = 2 * rnd.y - 1, qb = fabsf(b);
    // map to radius/angle
    const float rad = qa > qb ? qa : qb;
    const float phi = qa > qb ? qb / qa : ((qa == qb) ? 1.0f : 2 - qa / qb);
    // map to disk + flip back into right quadrant
    const float x = copysignf(rad * fast_cos_quadrant(phi), a);
    const float y = copysignf(rad * fast_sin_quadrant(phi), b);
    // assert(x >= -1);
    // assert(x <= 1);
    // assert(y >= -1);
    // assert(y <= 1);
    return { x, y };
}

// Inverse function of the above disk mapping
BSDL_INLINE Imath::V2f
disc_to_unit_square(const Imath::V2f& disc)
{
    const float r = sqrtf(std::min(SQR(disc.x) + SQR(disc.y), 1.0f));
    // compute on quadrant
    const float qa = fabsf(disc.x);
    const float qb = fabsf(disc.y);
    // figure out angle in [0,1]
    const float t   = qa > qb ? qb / qa : ((qa == qb) ? 1.0f : qa / qb);
    const float phi = fast_atan_quadrant(t) * 2;
    // Map back to unit square
    const float x = copysignf(qa > qb ? r : r * phi, disc.x) * 0.5f + 0.5f;
    const float y = copysignf(qa > qb ? r * phi : r, disc.y) * 0.5f + 0.5f;
    assert(x >= 0);
    assert(x <= 1);
    assert(y >= 0);
    assert(y <= 1);
    return { x, y };
}

BSDL_LEAVE_NAMESPACE
