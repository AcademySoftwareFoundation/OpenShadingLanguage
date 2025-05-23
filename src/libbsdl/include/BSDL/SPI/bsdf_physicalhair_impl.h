// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <array>

#include <BSDL/SPI/bsdf_physicalhair_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
PhysicalHairLobe<BSDF_ROOT>::PhysicalHairLobe(T* lobe,
                                              const BsdfGlobals& globals,
                                              const Data& data)
    : Base(lobe, data.T, globals.wo,
           std::min(globals.regularize_roughness(data.lroughness),
                    globals.regularize_roughness(data.troughness)),
           globals.lambda_0, true)
    , offset(data.offset)
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, false);

    const float input_ior = CLAMP(data.IOR, 1.001f, BIG);
    const float eta       = LERP(CLAMP(data.force_eta, 0.0f, 1.0f),
                                 globals.relative_eta(data.IOR), input_ior);
    lrough                = globals.regularize_roughness(data.lroughness);
    trough                = globals.regularize_roughness(data.troughness);
    arough                = CLAMP(data.aroughness, 0.01f, 1.0f);
    scattering            = CLAMP(SQR(data.scattering), 0.0f, 0.99f);
    const int debug       = globals.path_roughness > 0 ? 0 : data.flags;

    // Compute hair coordinate system terms related to _wo_
    const float sinThetaO = CLAMP(globals.wo.dot(Base::frame.Z), -1.0f, 1.0f);
    const float cosThetaO = sqrtf(1 - SQR(sinThetaO));

    constexpr auto fast_exp  = BSDLConfig::Fast::expf;
    constexpr auto fast_asin = BSDLConfig::Fast::asinf;
    gammaO                   = fast_asin(data.h);

    // Compute $\cos \thetat$ for refracted ray
    float sinThetaT = sinThetaO / eta;
    float cosThetaT = sqrtf(MAX(1 - SQR(sinThetaT), 0.0f));

    // Compute $\gammat$ for refracted ray
    float etap      = sqrtf(SQR(eta) - SQR(sinThetaO)) / cosThetaO;
    float sinGammaT = data.h / etap;
    float cosGammaT = sqrtf(1 - SQR(sinGammaT));
    gammaT          = fast_asin(sinGammaT);

    // Compute the transmittance _T_ of a single path through the cylinder
    float l          = 2 * cosGammaT / cosThetaT;
    Power absorption = globals.wave(data.absorption);
    Power tau([&](int i) { return fast_exp(-absorption[i] * l); },
              globals.lambda_0);

    ap = Ap(cosThetaO, eta, data.h, tau, globals.lambda_0);
    ap[0] *= globals.wave(data.R_tint).clamped(0.0f, 1.0f);
    ap[1] *= globals.wave(data.TT_tint).clamped(0.0f, 1.0f);
    ap[2] *= globals.wave(data.TRT_tint).clamped(0.0f, 1.0f);

    // If debugging isolate lobes
    if (debug) {
        for (int i = 0; i <= P_MAX; ++i)
            if (i != (debug - 1))
                ap[i] = Power::ZERO();
    }
    auto cdf = lobe_cdf();
    for (int i = 0; i <= P_MAX; ++i)
        ap[i] = ap[i].clamped(0.0f, cdf.pdf(i));
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD StaticCdf<P_MAX + 1>
PhysicalHairLobe<BSDF_ROOT>::lobe_cdf() const
{
    StaticCdf<P_MAX + 1> cdf;
    for (int i = 0; i <= P_MAX; ++i)
        cdf[i] = ap[i].max();
    const float total = cdf.build();
    // if lobes are all 0, just pretend like we have some chance of sampling the
    // last lobe (even though its 0 just to avoid special cases)
    if (total == 0)
        cdf[P_MAX] = 1;
    return cdf;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
PhysicalHairLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                       const Imath::V3f& wi) const
{
    const float sinThetaO = CLAMP(wo.z, -1.0f, 1.0f);
    const float cosThetaO = sqrtf(1 - SQR(sinThetaO));
    const float sinThetaI = CLAMP(wi.z, -1.0f, 1.0f);
    const float cosThetaI = sqrtf(1 - SQR(sinThetaI));
    const float phiI      = fast_atan2(wi.y, wi.x);
    std::array<float, P_MAX> sin2kAlpha, cos2kAlpha;
    // Compute alpha terms for hair scales
    auto sincosa = sincos_alpha(offset);
    sin2kAlpha   = sincosa.first;
    cos2kAlpha   = sincosa.second;

    const float phiO = 0;
    const float phi  = phiI - phiO;

    auto cdf      = lobe_cdf();
    Sample sample = { wi };
    BSDL_UNROLL()
    for (int p = 0; p <= P_MAX; ++p) {
        const float lobe_prob = cdf.pdf(p);
        if (lobe_prob <= PDF_MIN)
            continue;

        // Compute sin/cos theta to account for scales
        float sinThetaOp, cosThetaOp;
        switch (p) {
        case 0: {
            sinThetaOp = sinThetaO * cos2kAlpha[1] - cosThetaO * sin2kAlpha[1];
            cosThetaOp = cosThetaO * cos2kAlpha[1] + sinThetaO * sin2kAlpha[1];
            break;
        }
        case 1: {
            sinThetaOp = sinThetaO * cos2kAlpha[0] + cosThetaO * sin2kAlpha[0];
            cosThetaOp = cosThetaO * cos2kAlpha[0] - sinThetaO * sin2kAlpha[0];
            break;
        }
        case 2: {
            sinThetaOp = sinThetaO * cos2kAlpha[2] + cosThetaO * sin2kAlpha[2];
            cosThetaOp = cosThetaO * cos2kAlpha[2] - sinThetaO * sin2kAlpha[2];
            break;
        }
        default: {
            sinThetaOp = sinThetaO;
            cosThetaOp = cosThetaO;
            break;
        }
        }

        // handle out of range from scale adjustment
        cosThetaOp = fabsf(cosThetaOp);

        auto vs = variances(lrough, trough, arough, scattering, cosThetaO);
        std::array<float, P_MAX + 1> v = vs.first, s = vs.second;

        const float lobe_pdf = Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp,
                                  v[p])
                               * Np(phi, p, s[p], gammaO, gammaT);

        sample.update(ap[p], lobe_pdf, lobe_prob);

        // NOTE: this should almost always be the case, but because the pdf is nudged during cdf normalization, we can end up with
        // weights ever so slightly larger than 1.0 -- so just clamp
        sample.weight = sample.weight.clamped(0, 1);
        assert(sample.pdf >= 0);
    }

    sample.roughness = Base::roughness();
    return sample;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
PhysicalHairLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                         const Imath::V3f& _rnd) const
{
    const float sinThetaO = CLAMP(wo.z, -1.0f, 1.0f);
    const float cosThetaO = sqrtf(1 - SQR(sinThetaO));
    int p                 = 0;
    float pdf             = 0;
    Imath::V3f rnd        = _rnd;
    auto cdf              = lobe_cdf();
    rnd.x                 = cdf.sample(rnd.x, &p, &pdf);
    std::array<float, P_MAX> sin2kAlpha, cos2kAlpha;
    // Compute alpha terms for hair scales
    auto sincosa = sincos_alpha(offset);
    sin2kAlpha   = sincosa.first;
    cos2kAlpha   = sincosa.second;

    // Update sin/cos thetao to account for scales
    float sinThetaOp, cosThetaOp;
    switch (p) {
    case 0: {
        sinThetaOp = sinThetaO * cos2kAlpha[1] - cosThetaO * sin2kAlpha[1];
        cosThetaOp = cosThetaO * cos2kAlpha[1] + sinThetaO * sin2kAlpha[1];
        break;
    }
    case 1: {
        sinThetaOp = sinThetaO * cos2kAlpha[0] + cosThetaO * sin2kAlpha[0];
        cosThetaOp = cosThetaO * cos2kAlpha[0] - sinThetaO * sin2kAlpha[0];
        break;
    }
    case 2: {
        sinThetaOp = sinThetaO * cos2kAlpha[2] + cosThetaO * sin2kAlpha[2];
        cosThetaOp = cosThetaO * cos2kAlpha[2] - sinThetaO * sin2kAlpha[2];
        break;
    }
    default: {
        sinThetaOp = sinThetaO;
        cosThetaOp = cosThetaO;
        break;
    }
    }

    auto vs = variances(lrough, trough, arough, scattering, cosThetaO);
    std::array<float, P_MAX + 1> v = vs.first, s = vs.second;

    constexpr auto fast_exp    = BSDLConfig::Fast::expf;
    constexpr auto fast_log    = BSDLConfig::Fast::logf;
    constexpr auto fast_cos    = BSDLConfig::Fast::cosf;
    constexpr auto fast_sincos = BSDLConfig::Fast::sincosf;
    // Sample $M_p$ to compute $\thetai$
    rnd.x = MAX(rnd.x, 1e-5f);
    float cosTheta
        = 1 + v[p] * fast_log(rnd.x + (1 - rnd.x) * fast_exp(-2 / v[p]));
    float sinTheta  = sqrtf(MAX(1 - SQR(cosTheta), 0.0f));
    float cosPhi    = fast_cos(2 * PI * rnd.y);
    float sinThetaI = -cosTheta * sinThetaOp + sinTheta * cosPhi * cosThetaOp;
    float cosThetaI = sqrtf(MAX(1 - SQR(sinThetaI), 0.0f));

    // Sample N_p to compute delta_phi
    float dphi;
    if (p < P_MAX)
        dphi = Phi(p, gammaO, gammaT) + SampleTrimmedLogistic(rnd.z, s[p]);
    else
        dphi = 2 * PI * rnd.z;
    // Compute wi from sampled hair scattering angles
    const float phiO = 0;
    float phiI       = phiO + dphi;
    float cos_phi_i, sin_phi_i;
    fast_sincos(phiI, &sin_phi_i, &cos_phi_i);

    Imath::V3f wi = { cos_phi_i * cosThetaI, sin_phi_i * cosThetaI, sinThetaI };
    return eval_impl(wo, wi);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD std::array<Power, P_MAX + 1>
PhysicalHairLobe<BSDF_ROOT>::Ap(float cosThetaO, float eta, float h,
                                const Power T, float lambda_0)
{
    std::array<Power, P_MAX + 1> ap;
    // Compute $p=0$ attenuation at initial cylinder intersection
    float cosGammaO = sqrtf(MAX(1 - h * h, 0.0f));
    float cosTheta  = cosThetaO * cosGammaO;
    float f         = fresnel_dielectric(cosTheta, eta);
    ap[0]           = Power(f, 1);

    // Compute $p=1$ attenuation term
    ap[1] = SQR(1 - f) * T;

    // Compute attenuation terms up to $p=_pMax_$
    for (int p = 2; p < P_MAX; ++p)
        ap[p] = ap[p - 1] * T * f;

    // Compute attenuation term accounting for remaining orders of scattering
    ap[P_MAX] = ap[P_MAX - 1] * f * T;
    ap[P_MAX].update([&](int i,
                         float v) { return v / MAX(1.0f - T[i] * f, 1e-5f); },
                     lambda_0);
    for (int p = 0; p <= P_MAX; p++) {
        assert(ap[p].min(lambda_0) >= 0 && ap[p].max() <= 1);
    }
    return ap;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD std::pair<std::array<float, P_MAX>, std::array<float, P_MAX>>
PhysicalHairLobe<BSDF_ROOT>::sincos_alpha(float offset)
{
    constexpr auto fast_sin = BSDLConfig::Fast::sinf;
    std::array<float, P_MAX> sin2kAlpha, cos2kAlpha;
    sin2kAlpha[0] = fast_sin(offset);
    cos2kAlpha[0] = sqrtf(1 - SQR(sin2kAlpha[0]));
    for (int i = 1; i < P_MAX; ++i) {
        sin2kAlpha[i] = 2 * cos2kAlpha[i - 1] * sin2kAlpha[i - 1];
        cos2kAlpha[i] = SQR(cos2kAlpha[i - 1]) - SQR(sin2kAlpha[i - 1]);
    }
    return { sin2kAlpha, cos2kAlpha };
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD
    std::pair<std::array<float, P_MAX + 1>, std::array<float, P_MAX + 1>>
    PhysicalHairLobe<BSDF_ROOT>::variances(float lrough, float trough,
                                           float arough, float scattering,
                                           float cosThetaO)
{
    constexpr auto fast_exp   = BSDLConfig::Fast::expf;
    constexpr auto fast_log1p = BSDLConfig::Fast::log1pf;
    std::array<float, P_MAX + 1> v, s;
    const float sigma_s = -fast_log1p(-scattering);
    // This is the amount of light that goes unscattered through the hair
    // medium. We use this to boost exit roughness and fake scattering.
    const float unscattered = fast_exp(-sigma_s / cosThetaO);
    v[0] = RemapLongitudinalRoughness(lrough);  //   R lobe (primary spec)
    v[1] = 0.25f
           * RemapLongitudinalRoughness(
               sum_max(trough, 1 - unscattered,
                       1.0f));  //  TT lobe (transmission)
    v[2] = 4
           * RemapLongitudinalRoughness(
               sum_max(lrough, 1 - SQR(unscattered),
                       1.0f));  // TRT lobe (secondary spec)
    for (int p = 3; p <= P_MAX; ++p)
        v[p] = v[2];  // energy conservation lobe

    // Azimuthal roughness
    s[0] = RemapAzimuthalRoughness(arough);
    s[1] = RemapAzimuthalRoughness(sum_max(arough, 1 - unscattered, 1.0f));
    s[2] = RemapAzimuthalRoughness(sum_max(arough, 1 - SQR(unscattered), 1.0f));
    for (int p = 3; p <= P_MAX; ++p)
        s[p] = s[2];  // energy conservation lobe
    return { v, s };
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
PhysicalHairLobe<BSDF_ROOT>::log_bessi0(float x)
{
    constexpr auto fast_log   = BSDLConfig::Fast::logf;
    constexpr auto fast_log1p = BSDLConfig::Fast::log1pf;
    float ax                  = fabsf(x);
    if (ax < 3.75f) {
        float y = SQR(x / 3.75f);
        return fast_log1p(
            y
            * (3.5156229f
               + y
                     * (3.0899424f
                        + y
                              * (1.2067492f
                                 + y
                                       * (0.2659732f
                                          + y
                                                * (0.360768e-1f
                                                   + y * 0.45813e-2f))))));
    } else {
        float y = 3.75f / ax;
        return ax
               + fast_log(
                   1 / sqrtf(ax)
                   * (0.39894228f
                      + y
                            * (0.1328592e-1f
                               + y
                                     * (0.225319e-2f
                                        + y
                                              * (-0.157565e-2f
                                                 + y
                                                       * (0.916281e-2f
                                                          + y
                                                                * (-0.2057706e-1f
                                                                   + y
                                                                         * (0.2635537e-1f
                                                                            + y
                                                                                  * (-0.1647633e-1f
                                                                                     + y * 0.392377e-2f)))))))));
    }
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
PhysicalHairLobe<BSDF_ROOT>::bessi0_time_exp(float x, float exponent)
{
    constexpr auto fast_exp = BSDLConfig::Fast::expf;
    float ax                = fabsf(x);
    if (ax < 3.75f) {
        float y = SQR(x / 3.75f);
        return fast_exp(exponent)
               * (1.0f
                  + y
                        * (3.5156229f
                           + y
                                 * (3.0899424f
                                    + y
                                          * (1.2067492f
                                             + y
                                                   * (0.2659732f
                                                      + y
                                                            * (0.360768e-1f
                                                               + y * 0.45813e-2f))))));
    } else {
        float y = 3.75f / ax;
        return (fast_exp(ax + exponent) / sqrtf(ax))
               * (0.39894228f
                  + y
                        * (0.1328592e-1f
                           + y
                                 * (0.225319e-2f
                                    + y
                                          * (-0.157565e-2f
                                             + y
                                                   * (0.916281e-2f
                                                      + y
                                                            * (-0.2057706e-1f
                                                               + y
                                                                     * (0.2635537e-1f
                                                                        + y
                                                                              * (-0.1647633e-1f
                                                                                 + y * 0.392377e-2f))))))));
    }
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
PhysicalHairLobe<BSDF_ROOT>::Mp(float cosThetaI, float cosThetaO,
                                float sinThetaI, float sinThetaO, float v)
{
    constexpr auto fast_exp = BSDLConfig::Fast::expf;
    constexpr auto fast_log = BSDLConfig::Fast::logf;

    float a = cosThetaI * cosThetaO / v;
    float b = sinThetaI * sinThetaO / v;
    assert(!std::isnan(a));
    assert(!std::isnan(b));
    float mp = v <= .1f ? fast_exp(log_bessi0(a) - b - 1 / v + 0.6931f
                                   + fast_log(1 / (2 * v)))
                        : bessi0_time_exp(a, -b) / (fast_sinh(1 / v) * 2 * v);
    assert(mp == mp);
    return mp;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
PhysicalHairLobe<BSDF_ROOT>::Phi(int p, float gammaO, float gammaT)
{
    if (p % 2 == 0)
        return (2 * p) * gammaT - 2 * gammaO;
    else
        return (2 * p) * gammaT - 2 * gammaO - PI;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
PhysicalHairLobe<BSDF_ROOT>::TrimmedLogistic(float x, float s)
{
    constexpr auto fast_exp = BSDLConfig::Fast::expf;

    assert(x >= -PI);
    assert(x <= PI);
    const float t = std::min(fast_exp(PI / s), 1 / FLOAT_MIN);
    const float y = std::max(fast_exp(-fabsf(x) / s), FLOAT_MIN);
    return (t + 1) * y / ((t - 1) * s * SQR(1 + y));
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
PhysicalHairLobe<BSDF_ROOT>::Np(float phi, int p, float s, float gammaO,
                                float gammaT)
{
    if (p == P_MAX)
        return ONEOVERPI * 0.5f;
    float dphi = phi - Phi(p, gammaO, gammaT);
    if (dphi > PI)
        dphi -= 2 * PI;
    if (dphi < -PI)
        dphi += 2 * PI;
    return TrimmedLogistic(dphi, s);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
PhysicalHairLobe<BSDF_ROOT>::SampleTrimmedLogistic(float u, float s)
{
    constexpr auto fast_exp = BSDLConfig::Fast::expf;
    constexpr auto fast_log = BSDLConfig::Fast::logf;

    const float t = std::min(fast_exp(PI / s), 1 / FLOAT_MIN);
    const float x = -s * fast_log((1 + t) / (u * (1 - t) + t) - 1);
    assert(!std::isnan(x));
    return CLAMP(x, -PI, PI);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
PhysicalHairLobe<BSDF_ROOT>::RemapLongitudinalRoughness(float lr)
{
    // Roughness parametrization from http://pbrt.org/hair.pdf
    const float lr_2  = SQR(lr);
    const float lr_4  = SQR(lr_2);
    const float lr_20 = SQR(SQR(lr_4)) * lr_4;
    return SQR(0.726f * lr + 0.812f * lr_2 + 3.7f * lr_20);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
PhysicalHairLobe<BSDF_ROOT>::RemapAzimuthalRoughness(float ar)
{
    // Roughness parametrization from http://pbrt.org/hair.pdf
    const float ar_2  = SQR(ar);
    const float ar_4  = SQR(ar_2);
    const float ar_22 = SQR(SQR(ar_4)) * ar_4 * ar_2;
    return SQRT_PI_OVER_8 * (.265f * ar + 1.194f * ar_2 + 5.372f * ar_22);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
HairDiffuseLobe<BSDF_ROOT>::ecc2longrough(float ecc, float aniso)
{
    // We use d'Eon's M() longitudinal function and we map it to ecc this
    // way so as the azimuthal side shrinks, this shrinks too. We square to
    // match the visual linearity that ecc2s offers.
    return SQR(LERP(fabsf(ecc), 1.0f, sqrtf(MIN_ROUGH))) * (1 - aniso);
}

// Map eccentricity to logistic variance s
template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
HairDiffuseLobe<BSDF_ROOT>::ecc2s(float ecc, float aniso)
{
    // This mapping keeps it linear close to 0 without going too fast to
    // infinity. In practice it gives a linear resonse in appearance.
    const float roughness = LERP(fabsf(ecc) * (1 - aniso), 1.0f, MIN_ROUGH);
    return roughness / sqrtf(1 - roughness);
}

// These two functions map eccentricity to common BSDF roughness and back so
// we let roughness boost kick in modifying eccentricity.
template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
HairDiffuseLobe<BSDF_ROOT>::ecc2roughness(float ecc)
{
    return LERP(std::min(1.0f, fabsf(ecc)), 1.0f, MIN_ROUGH);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
HairDiffuseLobe<BSDF_ROOT>::roughness2ecc(float rough, float ecc)
{
    return copysign(LERP(LINEARSTEP(MIN_ROUGH, 1.0f, rough), 1.0f, 0.0f), ecc);
}

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
HairDiffuseLobe<BSDF_ROOT>::HairDiffuseLobe(T* lobe, const BsdfGlobals& globals,
                                            const Data& data)
    : Base(lobe, data.T, globals.wo,
           globals.regularize_roughness(ecc2roughness(data.eccentricity)),
           globals.lambda_0, true)
{
    constexpr auto fast_exp = BSDLConfig::Fast::expf;

    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, false);
    const float R       = Base::roughness();
    eccentricity        = roughness2ecc(R, data.eccentricity);
    // Fade out anisotropy with ecc so it also goes away with depth
    anisotropy = CLAMP(data.anisotropy, 0.0f, 1.0f) * fabsf(eccentricity);
    const float cos_o = CLAMP(data.T.dot(globals.wo), -1.0f, 1.0f);
    // Divide by IOR to get the refracted ray sine
    const float sin_t = sqrtf(1 - SQR(cos_o / std::max(data.IOR, 1.001f)));
    // We tried the sine weighted average for 1 / sin_t = 0.5f * PI but using
    // the actual value gives more color variation at grazing angles
    const float flatten = CLAMP(data.flatten_density, 0.0f, 1.0f);
    const float d       = LERP(flatten, 1 / sin_t, 1.0f);

    Power abs = globals.wave(data.absorption);
    color     = Power([&](int i) { return fast_exp(-abs[i] * d); },
                  globals.lambda_0);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
HairDiffuseLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                      const Imath::V3f& wi) const
{
    const float cos_o = CLAMP(wo.z, -1.0f, 1.0f);
    const float sin_o = sqrtf(1 - SQR(cos_o));
    const float cos_i = CLAMP(wi.z, -1.0f, 1.0f);
    const float sin_i = sqrtf(1 - SQR(cos_i));
    // Flip the lobe depending on eccentricity sign
    const float flip = eccentricity >= 0 ? -1 : 1;
    const float phi  = fast_atan2(wi.y, wi.x * flip);
    const float s    = ecc2s(eccentricity, anisotropy);
    const float v    = PhysicalHairLobe<BSDF_ROOT>::RemapLongitudinalRoughness(
        ecc2longrough(eccentricity, anisotropy));
    const float D_theta = PhysicalHairLobe<BSDF_ROOT>::Mp(sin_i, sin_o, cos_i,
                                                          cos_o, v);
    const float D_phi   = fabsf(eccentricity) > 0.001f
                              ? PhysicalHairLobe<BSDF_ROOT>::TrimmedLogistic(phi,
                                                                             s)
                              : 0.5f * ONEOVERPI;  // Isotropic
    return { wi, color, D_theta * D_phi, Base::roughness() };
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
HairDiffuseLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                        const Imath::V3f& rnd) const
{
    constexpr auto fast_exp = BSDLConfig::Fast::expf;
    constexpr auto fast_log = BSDLConfig::Fast::logf;
    constexpr auto fast_cos = BSDLConfig::Fast::cosf;

    const float cos_o = CLAMP(wo.z, -1.0f, 1.0f);
    const float sin_o = sqrtf(1 - SQR(cos_o));

    // Sample M
    const float v = PhysicalHairLobe<BSDF_ROOT>::RemapLongitudinalRoughness(
        ecc2longrough(eccentricity, anisotropy));
    const float x     = MAX(rnd.x, 1e-5f);
    float sin_theta_m = 1 + v * fast_log(x + (1 - x) * fast_exp(-2 / v));
    float cos_theta_m = sqrtf(MAX(1 - SQR(sin_theta_m), 0.0f));
    float cos_p       = fast_cos(2 * PI * rnd.y);
    float cos_theta   = -sin_theta_m * cos_o + cos_theta_m * cos_p * sin_o;
    float sin_theta   = sqrtf(MAX(1 - SQR(cos_theta), 0.0f));

    // Flip scattered phi depending on ecc sign
    const float flip = eccentricity >= 0 ? -1 : 1;
    const float s    = ecc2s(eccentricity, anisotropy);
    const float phi
        = fabsf(eccentricity) > 0.001f
              ? PhysicalHairLobe<BSDF_ROOT>::SampleTrimmedLogistic(rnd.z, s)
              : (rnd.z * 2 - 1) * PI;  // Isotropic
    float sin_phi, cos_phi;
    fast_sincos(phi, &sin_phi, &cos_phi);
    const Imath::V3f wi = { sin_theta * cos_phi * flip, sin_theta * sin_phi,
                            cos_theta };
    return eval_impl(wo, wi);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
HairDiffuseLobe<BSDF_ROOT>::albedo2absorption(float x, float g)
{
    // We simulate scattering in a hair cube and run it for 11 eccentricities from
    // -1 to 1 and 32 absorption values from 0 to 1 (non uniform, log^2), then
    // record the results tabulated as eccentricity, resulting albedo, absorption
    // which for gnuplot will be x, y, z in srf.txt. Then fit with this script:
    //
    //   set xrange[-1:1] # eccentricity
    //   set yrange[0.05:1] # ignore dark albedos, we don't care for accuracy there
    //   f(x, y) = ((log(y) + g * (x - 1)**2 * (1 - (2*y - 1)**2)) /
    //          abs(a + b * x + c * x**2 + d * x**3 + e * x**4 + f * x**5 + 1))**2
    //   a=b=c=d=e=f=g=1
    //   fit f(x, y) "srf.txt" using 1:2:3:(1) via a, b, c, d, e, f, g
    //   set view 60, 60, 1, 2
    //   splot "srf.txt", f(x, y)

    constexpr auto fast_log = BSDLConfig::Fast::logf;

    const float B
        = 1 - SQR(2 * x - 1);  // Useful shift to the log curve for fitting
    // Avoid returning inf for 0.0 albedo
    return MIN(
        BIG,
        SQR((fast_log(x) - 0.0621599f * SQR(g - 1) * B)
            / (2.3141f
               + g
                     * (0.740211f
                        + g
                              * (-0.252978f
                                 + g
                                       * (0.910376f
                                          + g * (2.79624f + g * 1.89986f)))))));
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Power
HairDiffuseLobe<BSDF_ROOT>::albedo2absorption(Power x, float lambda_0, float g)
{
    return Power([&](int i) { return albedo2absorption(x[i], g); }, lambda_0);
}

// This BSDF comes from PhysicalHairLobe but isolating either R or TRT
template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
HairSpecularLobe<BSDF_ROOT>::HairSpecularLobe(T* lobe,
                                              const BsdfGlobals& globals,
                                              const Data& data)
    : Base(lobe, data.T, globals.wo,
           globals.regularize_roughness(data.lroughness), globals.lambda_0,
           true)
    , trt(data.trt)
{
    constexpr auto fast_exp  = BSDLConfig::Fast::expf;
    constexpr auto fast_sin  = BSDLConfig::Fast::sinf;
    constexpr auto fast_asin = BSDLConfig::Fast::asinf;

    Base::sample_filter   = globals.get_sample_filter(Base::frame.Z, false);
    const float input_ior = CLAMP(data.IOR, 1.001f, BIG);
    const float eta       = LERP(CLAMP(data.force_eta, 0.0f, 1.0f),
                                 globals.relative_eta(input_ior), input_ior);
    const float lrough    = Base::roughness();
    const float arough    = CLAMP(data.aroughness, 0.01f, 1.0f);

    // Compute hair coordinate system terms related to _wo_
    const float sinThetaO = CLAMP(globals.wo.dot(Base::frame.Z), -1.0f, 1.0f);
    const float cosThetaO = sqrtf(1 - SQR(sinThetaO));

    long_v = (trt ? 4 : 1)
             * PhysicalHairLobe<BSDF_ROOT>::RemapLongitudinalRoughness(lrough);

    // Azimuthal roughness
    azim_s = PhysicalHairLobe<BSDF_ROOT>::RemapAzimuthalRoughness(arough);

    // Compute alpha terms for hair scales
    float sin2kAlpha_tmp[3], cos2kAlpha_tmp[3];
    sin2kAlpha_tmp[0] = fast_sin(data.offset);
    cos2kAlpha_tmp[0] = sqrtf(1 - SQR(sin2kAlpha_tmp[0]));
    for (int i = 1; i < 3; ++i) {
        sin2kAlpha_tmp[i] = 2 * cos2kAlpha_tmp[i - 1] * sin2kAlpha_tmp[i - 1];
        cos2kAlpha_tmp[i] = SQR(cos2kAlpha_tmp[i - 1])
                            - SQR(sin2kAlpha_tmp[i - 1]);
    }
    sin2kAlpha = sin2kAlpha_tmp[trt ? 2 : 1];
    cos2kAlpha = cos2kAlpha_tmp[trt ? 2 : 1];

    gammaO = fast_asin(data.h);

    // Compute $\cos \thetat$ for refracted ray
    float sinThetaT = sinThetaO / eta;

    // Compute $\gammat$ for refracted ray
    float etap      = sqrtf(SQR(eta) - SQR(sinThetaO)) / cosThetaO;
    float sinGammaT = data.h / etap;
    float cosGammaT = sqrtf(1 - SQR(sinGammaT));
    gammaT          = fast_asin(sinGammaT);

    // Compute $p=0$ attenuation at initial cylinder intersection
    const float cosGammaO = sqrtf(MAX(1 - SQR(data.h), 0.0f));
    const float cosTheta  = cosThetaO * cosGammaO;
    const float f         = fresnel_dielectric(cosTheta, eta);
    if (trt) {
        float cosThetaT = sqrtf(MAX(1 - SQR(sinThetaT), 0.0f));
        // Compute the transmittance _T_ of a single path through the cylinder
        const float flatten = CLAMP(data.flatten_density, 0.0f, 1.0f);
        float l             = LERP(flatten, 2 * cosGammaT / cosThetaT, 1.0f);

        Power absorption = globals.wave(data.absorption);
        Power tau([&](int i) { return fast_exp(-absorption[i] * l); },
                  globals.lambda_0);

        // First fresnel atten comes from layering if R is present. This makes
        // putting R on top of TRT sensible, otherwise this would be SQR(1 - f) * f
        color        = (1 - f) * f * SQR(tau) * globals.wave(data.tint);
        fresnel_term = (1 - f) * f;
    } else {
        color        = f * globals.wave(data.tint);
        fresnel_term = f;
    }
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
HairSpecularLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                       const Imath::V3f& wi) const
{
    constexpr auto fast_atan2 = BSDLConfig::Fast::atan2f;

    const float sinThetaI = CLAMP(wi.z, -1.0f, 1.0f);
    const float cosThetaI = sqrtf(1 - SQR(sinThetaI));
    const float sinThetaO = CLAMP(wo.z, -1.0f, 1.0f);
    const float cosThetaO = sqrtf(1 - SQR(sinThetaO));
    const float phiI      = fast_atan2(wi.y, wi.x);

    const float phiO = 0;
    const float phi  = phiI - phiO;

    Power weight = color;
    // Compute sin/cos theta to account for scales
    float sinThetaOp, cosThetaOp;
    if (trt) {
        sinThetaOp = sinThetaO * cos2kAlpha + cosThetaO * sin2kAlpha;
        cosThetaOp = cosThetaO * cos2kAlpha - sinThetaO * sin2kAlpha;
    } else {
        sinThetaOp = sinThetaO * cos2kAlpha - cosThetaO * sin2kAlpha;
        cosThetaOp = cosThetaO * cos2kAlpha + sinThetaO * sin2kAlpha;
    }
    // handle out of range from scale adjustment
    cosThetaOp = fabsf(cosThetaOp);
    int p      = trt ? 2 : 0;

    const float pdf
        = PhysicalHairLobe<BSDF_ROOT>::Mp(cosThetaI, cosThetaOp, sinThetaI,
                                          sinThetaOp, long_v)
          * PhysicalHairLobe<BSDF_ROOT>::Np(phi, p, azim_s, gammaO, gammaT);

    return { wi, weight, pdf, Base::roughness() };
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
HairSpecularLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                         const Imath::V3f& _rnd) const
{
    constexpr auto fast_log    = BSDLConfig::Fast::logf;
    constexpr auto fast_sincos = BSDLConfig::Fast::sincosf;

    const float sinThetaO = CLAMP(wo.z, -1.0f, 1.0f);
    const float cosThetaO = sqrtf(1 - SQR(sinThetaO));
    Imath::V3f rnd        = _rnd;
    // Update sin/cos thetao to account for scales
    float sinThetaOp, cosThetaOp;
    if (trt) {
        sinThetaOp = sinThetaO * cos2kAlpha + cosThetaO * sin2kAlpha;
        cosThetaOp = cosThetaO * cos2kAlpha - sinThetaO * sin2kAlpha;
    } else {
        sinThetaOp = sinThetaO * cos2kAlpha - cosThetaO * sin2kAlpha;
        cosThetaOp = cosThetaO * cos2kAlpha + sinThetaO * sin2kAlpha;
    }

    // Sample $M_p$ to compute $\thetai$
    rnd.x = std::max(rnd.x, 1e-5f);
    float cosTheta
        = 1 + long_v * fast_log(rnd.x + (1 - rnd.x) * fast_exp(-2 / long_v));
    float sinTheta  = sqrtf(MAX(1 - SQR(cosTheta), 0.0f));
    float cosPhi    = fast_cos(2 * PI * rnd.y);
    float sinThetaI = -cosTheta * sinThetaOp + sinTheta * cosPhi * cosThetaOp;
    float cosThetaI = sqrtf(MAX(1 - SQR(sinThetaI), 0.0f));
    int p           = trt ? 2 : 0;

    // Sample N_p to compute delta_phi
    float dphi = PhysicalHairLobe<BSDF_ROOT>::Phi(p, gammaO, gammaT)
                 + PhysicalHairLobe<BSDF_ROOT>::SampleTrimmedLogistic(rnd.z,
                                                                      azim_s);
    // Compute wi from sampled hair scattering angles
    const float phiO = 0;
    float phiI       = phiO + dphi;
    float cos_phi_i, sin_phi_i;
    fast_sincos(phiI, &sin_phi_i, &cos_phi_i);

    Imath::V3f wi = { cos_phi_i * cosThetaI, sin_phi_i * cosThetaI, sinThetaI };
    return eval_impl(wo, wi);
}

}  // namespace spi

BSDL_LEAVE_NAMESPACE
