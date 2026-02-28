// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <BSDL/MTX/bsdf_sheen_decl.h>

#include <BSDL/MTX/bsdf_zeltnersheen_param.h>
#ifndef BAKE_BSDL_TABLES
#    include <BSDL/MTX/bsdf_contysheen_luts.h>
#    include <BSDL/MTX/bsdf_zeltnersheen_luts.h>
#endif

BSDL_ENTER_NAMESPACE

namespace mtx {

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
SheenLobe<BSDF_ROOT>::SheenLobe(T* lobe, const BsdfGlobals& globals,
                                const Data& data)
    : Base(lobe, globals.visible_normal(data.N), globals.wo,
           globals.regularize_roughness(CLAMP(data.roughness, 0.0f, 1.0f)),
           globals.lambda_0, false)
    , tint(globals.wave(data.albedo))
    , Emiss(1.0f)
    , sheen_mode(data.mode)
    , is_backfacing(globals.backfacing)
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, true);
    sheen_alpha
        = use_zeltner()
              ? MAX(ZeltnerBurleySheen::MIN_ROUGHNESS, sqrtf(Base::roughness()))
              : MAX(ContyKullaDist<false>::MIN_ROUGHNESS, Base::roughness());

    const float cosNO = CLAMP(Base::frame.Z.dot(globals.wo), 0.0f, 1.0f);
    if (is_backfacing)
        Emiss = 1;
    else if (use_zeltner()) {
        ZeltnerBurleySheen sheen(sheen_alpha);
        Emiss = 1 - std::min(sheen.albedo(cosNO) * tint.max(), 1.0f);
    } else {
        ContyKullaSheenMTX sheen(sheen_alpha);
        Emiss = 1 - std::min(sheen.albedo(cosNO) * tint.max(), 1.0f);
    }
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
SheenLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                const Imath::V3f& wi) const
{
    const float cosNO        = wo.z;
    const float cosNI        = wi.z;
    const bool is_reflection = cosNI > 0 && cosNO >= 0;
    const bool do_self       = is_reflection && !is_backfacing;
    if (!do_self)
        return {};

    Sample s = use_zeltner() ? ZeltnerBurleySheen(sheen_alpha).eval(wo, wi)
                             : ContyKullaSheenMTX(sheen_alpha).eval(wo, wi);
    s.weight *= tint;
    s.roughness = Base::roughness();
    return s;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
SheenLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                  const Imath::V3f& rnd) const
{
    if (is_backfacing)
        return {};

    Sample s
        = use_zeltner()
              ? ZeltnerBurleySheen(sheen_alpha).sample(wo, rnd.x, rnd.y, rnd.z)
              : ContyKullaSheenMTX(sheen_alpha).sample(wo, rnd.x, rnd.y, rnd.z);
    s.weight *= tint;
    s.roughness = Base::roughness();
    return s;
}

// Reference:
// Conty Estevez and Kulla, "Production Friendly Microfacet Sheen BRDF", 2017.
// https://github.com/aconty/aconty/blob/main/pdf/s2017_pbs_imageworks_sheen.pdf

template<bool SHD_LEGACY>
BSDL_INLINE_METHOD float
ContyKullaDist<SHD_LEGACY>::D(const Imath::V3f& Hr) const
{
    // Eq. (2): D(m) = ((2 + 1/r) * sin(theta)^(1/r)) / (2*pi)
    float cos_theta = CLAMP(Hr.z, 0.0f, 1.0f);
    float sin_theta = sqrtf(1.0f - SQR(cos_theta));
    return BSDLConfig::Fast::powf(sin_theta, 1 / a) * (2 + 1 / a) * 0.5f
           * ONEOVERPI;
}

template<bool SHD_LEGACY>
BSDL_INLINE_METHOD float
ContyKullaDist<SHD_LEGACY>::get_lambda(float cosNO) const
{
    float rt = SQR(1 - a);

    // These params come from gnuplot fitting tool. Roughness = 1 and
    // roughness = 0. We interpolate in between with (1 - roughness)^2
    // to get the best match.
    // Eq. (3): fitted Lambda(theta) model.
    const float a = LERP(rt, 21.5473f, 25.3245f);
    const float b = LERP(rt, 3.82987f, 3.32435f);
    const float c = LERP(rt, 0.19823f, 0.16801f);
    const float d = LERP(rt, -1.97760f, -1.27393f);
    const float e = LERP(rt, -4.32054f, -4.85967f);
    // The curve is anti-symmetrical aroung 0.5
    const float x     = cosNO > 0.5f ? 1 - cosNO : cosNO;
    const float pivot = a / (1 + b * BSDLConfig::Fast::powf(0.5f, c)) + d * 0.5f
                        + e;

    float p = a / (1 + b * BSDLConfig::Fast::powf(x, c)) + d * x + e;
    if (cosNO > 0.5f)  // Mirror around 0.5f
        p = 2 * pivot - p;
    // That curve fits lambda in log scale, now exponentiate
    return BSDLConfig::Fast::expf(p);
}

template<bool SHD_LEGACY>
BSDL_INLINE_METHOD float
ContyKullaDist<SHD_LEGACY>::G2(const Imath::V3f& wo, const Imath::V3f& wi) const
{
    assert(wi.z > 0);
    assert(wo.z > 0);
    float cosNI = std::min(1.0f, wi.z);
    float cosNO = std::min(1.0f, wo.z);
    if constexpr (SHD_LEGACY) {
        // Original shadowing term from the paper
        float lambdaI = get_lambda(cosNI);
        float lambdaO = get_lambda(cosNO);
        // Eq. (4): lambda' light-side softening near the terminator.
        // This is intentionally non-reciprocal.
        lambdaI = BSDLConfig::Fast::powf(lambdaI,
                                         1.0f + 2 * SQR(SQR(SQR((1 - cosNI)))));
        // Correlated masking-shadowing term: G2 = 1 / (1 + Lambda(wo) + Lambda(wi)).
        return 1 / (1 + lambdaI + lambdaO);
    } else
        // The simpler shadowing/masking visibility term below is used. It also has less
        // darkening at low roughness on grazing angles.
        // https://dassaultsystemes-technology.github.io/EnterprisePBRShadingModel/spec-2022x.md.html#components/sheen
        return (cosNI * cosNO) / (cosNI + cosNO - cosNI * cosNO);
}

template<typename Dist>
BSDL_INLINE_METHOD Sample
SheenMicrofacet<Dist>::eval(const Imath::V3f& wo, const Imath::V3f& wi) const
{
    assert(wo.z >= 0);
    assert(wi.z >= 0);
    float cosNO = wo.z;
    float cosNI = wi.z;
    if (cosNI <= 1e-5f || cosNO <= 1e-5f)
        return {};

    const float D = d.D((wo + wi).normalized());
    if (D < 1e-6)
        return {};
    const float G2 = d.G2(wo, wi);
    // Eq. (1): microfacet BRDF form. For sheen, Fresnel F = 1.
    return { wi, Power(D * G2 * 0.5f * PI / cosNO, 1), 0.5f * ONEOVERPI, 0 };
}

template<typename Dist>
BSDL_INLINE_METHOD Sample
SheenMicrofacet<Dist>::sample(const Imath::V3f& wo, float randu, float randv,
                              float randw) const
{
    Imath::V3f wi = sample_uniform_hemisphere(randu, randv);
    return eval(wo, wi);
}

BSDL_INLINE_METHOD float
ContyKullaSheen::albedo(float cosNO) const
{
    return 1
           - TabulatedEnergyCurve<ContyKullaSheen>(roughness(), 0)
                 .Emiss_eval(cosNO);
}

BSDL_INLINE_METHOD float
ContyKullaSheenMTX::albedo(float cosNO) const
{
    // Rational fit from the Material X project
    // Ref: https://github.com/AcademySoftwareFoundation/MaterialX/blob/main/libraries/pbrlib/genglsl/lib/mx_microfacet_sheen.glsl
    const Imath::V2f r
        = Imath::V2f(13.67300f, 1.0f)
          + Imath::V2f(-68.78018f, 61.57746f) * cosNO
          + Imath::V2f(799.08825f, 442.78211f) * roughness()
          + Imath::V2f(-905.00061f, 2597.49308f) * cosNO * roughness()
          + Imath::V2f(60.28956f, 121.81241f) * cosNO * cosNO
          + Imath::V2f(1086.96473f, 3045.55075f) * roughness() * roughness();
    return CLAMP(r.x / r.y, 0.0f, 1.0f);
}

// Reference:
// Zeltner, Burley, Chiang, "Practical Multiple-Scattering Sheen Using
// Linearly Transformed Cosines", SIGGRAPH 2022.
// https://tizianzeltner.com/projects/Zeltner2022Practical/

BSDL_INLINE_METHOD Sample
ZeltnerBurleySheen::eval(Imath::V3f wo, Imath::V3f wi) const
{
    if (wo.z < 0 || wi.z <= 0)
        return {};

    return eval(wo, wi, fetch_coeffs(wo.z));
}

BSDL_INLINE_METHOD Sample
ZeltnerBurleySheen::eval(Imath::V3f wo, Imath::V3f wi, Imath::V3f ltc) const
{
    // Eq. (1): evaluate the sheen LTC with per-view coefficients
    // (A, B, R) = (a_inv, b_inv, r_coeff).
    const float a_inv   = ltc.x;
    const float b_inv   = ltc.y;
    const float r_coeff = ltc.z;
    Imath::V3f wi_orig  = { a_inv * wi.x + b_inv * wi.z, a_inv * wi.y, wi.z };

    // The (inverse) transform matrix `M^{-1}` is given by:
    //              [[aInv 0    bInv]
    //     M^{-1} =  [0    aInv 0   ]
    //               [0    0    1   ]]
    // with `aInv = ltcCoeffs[0]`, `bInv = ltcCoeffs[1]` fetched from the
    // table. The transformed direction `wiOriginal` is therefore:
    //                                [[aInv * wi.x + bInv * wi.z]
    //     wiOriginal = M^{-1} * wi =  [aInv * wi.y              ]
    //                                 [wi.z                     ]]
    // which is subsequently normalized. The determinant of the matrix is
    //     |M^{-1}| = aInv * aInv
    // which is used to compute the Jacobian determinant of the complete
    // mapping including the normalization.
    // See the original paper [Heitz et al. 2016] for details about the LTC
    // itself.
    const float jacobian = pown<2>(a_inv / wi_orig.length2());

    if (LTC_SAMPLING) {
        float pdf = jacobian * std::max(wi_orig.z, 0.0f) * ONEOVERPI;
        if (pdf > FLOAT_MIN)
            return { wi, Power(r_coeff, 1), pdf, roughness };
        else
            return {};
    } else {
        float pdf = 0.5f * ONEOVERPI;
        // Eq. (1): f_sheen = (2 * R / pi) * |M^{-1}| * max(0, z) / ||M^{-1} wi||^4.
        // NOTE: sheen closure has no Fresnel/masking terms.
        return { wi,
                 Power(2 * r_coeff * jacobian * std::max(wi_orig.z, 0.0f), 1),
                 pdf, roughness };
    }
}

BSDL_INLINE_METHOD Sample
ZeltnerBurleySheen::sample(Imath::V3f wo, float randu, float randv,
                           float randw) const
{
    if (wo.z < 0)
        return {};
    if (LTC_SAMPLING) {
        // Sample the cosine base distribution, then transform by M.
        const Imath::V3f ltc = fetch_coeffs(wo.z);
        const float a_inv    = ltc.x;
        const float b_inv    = ltc.y;
        Imath::V3f wi_orig   = sample_cos_hemisphere(randu, randv);
        const Imath::V3f wi  = { wi_orig.x - wi_orig.z * b_inv, wi_orig.y,
                                 a_inv * wi_orig.z };

        return eval(wo, wi.normalized(), ltc);
    } else {
        Imath::V3f wi = sample_uniform_hemisphere(randu, randv);
        return eval(wo, wi);
    }
}

// The following functions and LTC coefficient tables are translated from:
// https://github.com/tizian/ltc-sheen
// and the Zeltner et al. 2022 sheen paper.

// Fetch the LTC coefficients by bilinearly interpolating entries in a 32x32
// lookup table. */
BSDL_INLINE_METHOD Imath::V3f
ZeltnerBurleySheen::fetch_coeffs(float cosNO) const
{
    if (FITTED_LTC) {
        // To avoid look-up tables, we use a fit of the LTC coefficients derived by Stephen Hill
        // for the implementation in MaterialX:
        // https://github.com/AcademySoftwareFoundation/MaterialX/blob/main/libraries/pbrlib/genglsl/lib/mx_microfacet_sheen.glsl
        const float x = CLAMP(cosNO, 0.0f, 1.0f);
        const float y = std::max(roughness, 1e-3f);
        // Fitted approximation of the paper's tabulated LTC coefficients (A, B, R).
        const float A = ((2.58126f * x + 0.813703f * y) * y)
                        / (1.0f + 0.310327f * x * x + 2.60994f * x * y);
        const float B = sqrtf(1.0f - x) * (y - 1.0f) * y * y * y
                        / (0.0000254053f + 1.71228f * x - 1.71506f * x * y
                           + 1.34174f * y * y);
        const float invs = (0.0379424f + y * (1.32227f + y))
                           / (y * (0.0206607f + 1.58491f * y));
        const float m = y
                        * (-0.193854f
                           + y * (-1.14885 + y * (1.7932f - 0.95943f * y * y)))
                        / (0.046391f + y);
        const float o = y * (0.000654023f + (-0.0207818f + 0.119681f * y) * y)
                        / (1.26264f + y * (-1.92021f + y));
        float q                 = (x - m) * invs;
        const float inv_sqrt2pi = 0.39894228040143f;
        float R = BSDLConfig::Fast::expf(-0.5f * q * q) * invs * inv_sqrt2pi
                  + o;
        assert(isfinite(A));
        assert(isfinite(B));
        assert(isfinite(R));
        return { A, B, R };
    } else {
        // Bilinearly interpolate (A, B, R) from the 32x32 LUT over
        // (roughness, cos(theta_o)).
        float row = CLAMP(roughness, 0.0f, ALMOSTONE) * (ltc_res - 1);
        float col = CLAMP(cosNO, 0.0f, ALMOSTONE) * (ltc_res - 1);
        float r   = std::floor(row);
        float c   = std::floor(col);
        float rf  = row - r;
        float cf  = col - c;
        int ri    = (int)r;
        int ci    = (int)c;

        const V32_array param = param_ptr();
        // Bilinear interpolation
        Imath::V3f coeffs;
        const Imath::V3f v1 = param[ri][ci];
        const Imath::V3f v2 = param[ri][ci + 1];
        const Imath::V3f v3 = param[ri + 1][ci];
        const Imath::V3f v4 = param[ri + 1][ci + 1];
        coeffs              = LERP(rf, LERP(cf, v1, v2), LERP(cf, v3, v4));
        return coeffs;
    }
}

BSDL_INLINE_METHOD float
ZeltnerBurleySheen::albedo(float cosNO) const
{
    if constexpr (LTC_ALBEDO)
        return fetch_coeffs(cosNO).z;
    else
        return 1
               - TabulatedEnergyCurve<ZeltnerBurleySheen>(roughness, 0)
                     .Emiss_eval(cosNO);
}

}  // namespace mtx

BSDL_LEAVE_NAMESPACE
