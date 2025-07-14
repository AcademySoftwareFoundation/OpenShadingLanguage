// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/MTX/bsdf_dielectric_decl.h>
#include <BSDL/SPI/bsdf_dielectric_impl.h>

#ifndef BAKE_BSDL_TABLES
#    include <BSDL/MTX/bsdf_dielectric_bothback_luts.h>
#    include <BSDL/MTX/bsdf_dielectric_bothfront_luts.h>
#    include <BSDL/MTX/bsdf_dielectric_reflfront_luts.h>
#endif

BSDL_ENTER_NAMESPACE

namespace mtx {

BSDL_INLINE_METHOD
DielectricFresnel::DielectricFresnel(float _eta, bool backside)
{
    if (backside)
        _eta = 1 / _eta;
    eta = _eta >= 1 ? CLAMP(_eta, IOR_MIN, IOR_MAX)
                    : CLAMP(_eta, 1 / IOR_MAX, 1 / IOR_MIN);
}

BSDL_INLINE_METHOD Power
DielectricFresnel::eval(const float c) const
{
    assert(c >= 0);  // slightly above 1.0 is ok
    float g = (eta - 1.0f) * (eta + 1.0f) + c * c;
    if (g > 0) {
        g       = sqrtf(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1) / (c * (g - c) + 1);
        return Power(0.5f * A * A * (1 + B * B), 1);
    }
    return Power::UNIT();  // TIR (no refracted component)
}

BSDL_INLINE_METHOD DielectricFresnel
DielectricFresnel::from_table_index(float tx, bool backside)
{
    const float eta = LERP(SQR(tx), IOR_MIN, IOR_MAX);
    return DielectricFresnel(eta, backside);
}

// Note index of eta equals index of 1 / eta, so this function works for
// either side (both tables)
BSDL_INLINE_METHOD float
DielectricFresnel::table_index() const
{
    // turn the IOR value into something suitable for integrating
    // this is the reverse of the method above
    const float feta = eta;
    const float seta = CLAMP(feta < 1 ? 1 / feta : feta, IOR_MIN, IOR_MAX);
    const float x    = (seta - IOR_MIN) * (1 / (IOR_MAX - IOR_MIN));
    assert(x >= 0);
    assert(x <= 1);
    return sqrtf(x);
}

BSDL_INLINE_METHOD
DielectricRefl::DielectricRefl(float cosNO, float roughness_index,
                               float fresnel_index)
    : d(roughness_index, 0)
    , f(DielectricFresnel::from_table_index(fresnel_index, false))
{
    TabulatedEnergyCurve<spi::MiniMicrofacetGGX> curve(roughness_index,
                                                       fresnel_index);
    E_ms = curve.Emiss_eval(cosNO);
    assert(0 <= E_ms && E_ms <= 1);
}

BSDL_INLINE_METHOD
DielectricRefl::DielectricRefl(const GGXDist& dist,
                               const DielectricFresnel& fresnel, float cosNO,
                               float roughness)
    : d(dist), f(fresnel)
{
    TabulatedEnergyCurve<spi::MiniMicrofacetGGX> curve(roughness, 0.0f);
    E_ms = curve.Emiss_eval(cosNO);
    assert(0 <= E_ms && E_ms <= 1);
}

BSDL_INLINE_METHOD Sample
DielectricRefl::eval(Imath::V3f wo, Imath::V3f wi) const
{
    return eval_turquin_microms_reflection(d, f, E_ms, wo, wi);
}

BSDL_INLINE_METHOD Sample
DielectricRefl::sample(Imath::V3f wo, float randu, float randv,
                       float randw) const
{
    return sample_turquin_microms_reflection(d, f, E_ms, wo,
                                             { randu, randv, randw });
}

BSDL_INLINE_METHOD
DielectricReflFront::DielectricReflFront(float cosNO, float roughness_index,
                                         float fresnel_index)
    : DielectricRefl(cosNO, roughness_index, fresnel_index)
{
}

BSDL_INLINE_METHOD
DielectricBoth::DielectricBoth(float cosNO, float roughness_index,
                               float fresnel_index, bool backside)
    : d(roughness_index, 0)
    , f(DielectricFresnel::from_table_index(fresnel_index, backside))
{
}

BSDL_INLINE_METHOD
DielectricBoth::DielectricBoth(const GGXDist& dist,
                               const DielectricFresnel& fresnel)
    : d(dist), f(fresnel)
{
}

BSDL_INLINE_METHOD Sample
DielectricBoth::eval(Imath::V3f wo, Imath::V3f wi) const
{
    const float cosNO = wo.z;
    const float cosNI = wi.z;
    assert(cosNO >= 0);
    if (cosNI > 0) {
        const Imath::V3f m = (wo + wi).normalized();
        const float cosMO  = m.dot(wo);
        if (cosMO <= 0)
            return {};
        const float D = d.D(m);
        // Reflection optimized density
        const float D_refl_D = d.D_refl_D(wo, m);
        const float D_refl   = D_refl_D * D;
        const float G1       = d.G1(wo);
        const Power F        = f.eval(cosMO);
        const float out      = d.G2_G1(wi, wo) * G1 / D_refl_D;
        const float pdf      = D_refl / (4.0f * cosNO) * F[0];
        return { wi, Power(out, 1), pdf, 0 };
    } else if (cosNI < 0) {
        // flip to same side as N
        const Imath::V3f Ht = (f.eta * wi + wo).normalized()
                              * ((f.eta > 1) ? -1 : 1);
        // compute fresnel term
        const float cosHO = Ht.dot(wo);
        const float cosHI = Ht.dot(wi);
        if (cosHO <= 0 || cosHI >= 0)
            return {};
        const float Ft = 1.0f - f.eval(cosHO)[0];
        if (Ht.z <= 0 || cosHO <= 0 || cosHI >= 0 || Ft <= 0)
            return {};
        const float D = d.D(Ht);
        // Reflection optimized density
        const float D_refl_D = d.D_refl_D(wo, Ht);
        const float D_refl   = D_refl_D * D;
        const float G1       = d.G1(wo);
        float J              = (-cosHI * cosHO * SQR(f.eta))
                  / (wo.z * SQR(cosHI * f.eta + cosHO));
        float pdf       = D_refl * J * Ft;
        const float out = d.G2_G1({ wi.x, wi.y, -wi.z }, wo) * G1 / D_refl_D;
        return { wi, Power(out, 1), pdf, 0 };

    } else
        return {};
}

BSDL_INLINE_METHOD Sample
DielectricBoth::sample(Imath::V3f wo, float randu, float randv,
                       float randw) const
{
    // This skips micro normals not valid for reflection, but they
    // could be valid for refraction. Energy is ok because we renormalize
    // this lobe, but refraction will be biased for high roughness. We
    // trade that for reduced noise.
    const Imath::V3f m = d.sample_for_refl(wo, randu, randv);
    const float cosMO  = wo.dot(m);
    if (cosMO <= 0)
        return {};
    const float F       = f.eval(cosMO)[0];
    bool choose_reflect = randw < F;
    const Imath::V3f wi = choose_reflect ? reflect(wo, m)
                                         : refract(wo, m, f.eta);
    if ((choose_reflect && wi.z <= 0) || (!choose_reflect && wi.z >= 0))
        return {};
    return eval(wo, wi);
}

BSDL_INLINE_METHOD
DielectricBothFront::DielectricBothFront(float cosNO, float roughness_index,
                                         float fresnel_index)
    : DielectricBoth(cosNO, roughness_index, fresnel_index, false)
{
}

BSDL_INLINE_METHOD
DielectricBothBack::DielectricBothBack(float cosNO, float roughness_index,
                                       float fresnel_index)
    : DielectricBoth(cosNO, roughness_index, fresnel_index, true)
{
}

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
DielectricLobe<BSDF_ROOT>::DielectricLobe(T* lobe, const BsdfGlobals& globals,
                                          const Data& data)
    : Base(lobe, globals.visible_normal(data.N), data.U, 0.0f, globals.lambda_0,
           MAX_RGB(data.refr_tint) > 0)
    , refl_tint(globals.wave(data.refl_tint))
    , refr_tint(globals.wave(data.refr_tint))
    , wo_absorption(1.0f, globals.lambda_0)
    , dispersion(data.dispersion > 0 && globals.lambda_0 > 0)
{
    dorefl              = MAX_RGB(data.refl_tint) > 0;
    dorefr              = MAX_RGB(data.refr_tint) > 0;
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, false);
    // MaterialX expects the raw x/y roughness as input, but for albedo tables it
    // is better to use the roughness/anisotropy parametrization so we can
    // ignore roughness
    const float rx    = CLAMP(data.roughness_x, EPSILON, 2.0f);
    const float ry    = CLAMP(data.roughness_y, EPSILON, 2.0f);
    const float ax    = std::max(rx, ry);
    const float ay    = std::min(rx, ry);
    const float b     = ay / ax;
    const float aniso = (1 - b) / (1 + b);
    // Also assume we square the roughness for linearity
    const float roughness = globals.regularize_roughness(
        sqrtf(ax / (1 + aniso)));
    const float cosNO = Base::frame.Z.dot(globals.wo);
    const float IOR
        = CLAMP(dispersion
                    ? Spectrum::get_dispersion_ior(data.dispersion, data.IOR,
                                                   globals.lambda_0)
                    : data.IOR,
                DielectricFresnel::IOR_MIN, DielectricFresnel::IOR_MAX);

    assert(cosNO >= 0);
    if (dorefl && !dorefr) {
        spec.refl = DielectricRefl(GGXDist(roughness, aniso, rx < ry),
                                   DielectricFresnel(globals.relative_eta(IOR),
                                                     globals.backfacing),
                                   cosNO, roughness);
        E_ms      = TabulatedEnergyCurve<DielectricReflFront>(
                   roughness, spec.refl.fresnel().table_index())
                   .Emiss_eval(cosNO);
    } else if (dorefr) {
        spec.both = DielectricBoth(GGXDist(roughness, aniso, rx < ry),
                                   DielectricFresnel(globals.relative_eta(IOR),
                                                     globals.backfacing));
        if (spec.both.fresnel().eta >= 1.0f) {
            E_ms = TabulatedEnergyCurve<DielectricBothFront>(
                       roughness, spec.both.fresnel().table_index())
                       .Emiss_eval(cosNO);
        } else {
            E_ms = TabulatedEnergyCurve<DielectricBothBack>(
                       roughness, spec.both.fresnel().table_index())
                       .Emiss_eval(cosNO);
        }
    }

    Base::set_roughness(roughness);

    if (MAX_RGB(data.absorption) > 0) {
        constexpr auto fast_exp = BSDLConfig::Fast::expf;

        float cos_p = cosNO;
        // Take into account how the ray bends with the refraction to compute
        // the traveled distance through absorption.
        const float sinNO2  = 1 - SQR(cosNO);
        const float inveta2 = SQR(1 / spec.refl.fresnel().eta);
        cos_p               = sqrtf(1 - std::min(1.0f, inveta2 * sinNO2));
        const float dist    = 1 / std::max(cos_p, FLOAT_MIN);

        const Power sigma_a = globals.wave(data.absorption);
        wo_absorption
            = Power([&](int i) { return fast_exp(-sigma_a[i] * dist); },
                    globals.lambda_0);
    }
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
DielectricLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                     const Imath::V3f& wi) const
{
    if (!dorefl && !dorefr)
        return {};
    Sample s = {};
    if (dorefl && !dorefr)
        s = spec.refl.eval(wo, wi);
    else {
        s = spec.both.eval(wo, wi);
        s.weight *= 1 / std::max(0.01f, 1 - E_ms);
    }

    s.weight *= get_tint(s.wi.z);
    s.roughness = Base::roughness();
    return s;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
DielectricLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                       const Imath::V3f& rnd) const
{
    if (!dorefl && !dorefr)
        return {};

    Sample s = {};
    if (dorefl && !dorefr)
        s = spec.refl.sample(wo, rnd.x, rnd.y, rnd.z);
    else {
        s = spec.both.sample(wo, rnd.x, rnd.y, rnd.z);
        s.weight *= 1 / std::max(0.01f, 1 - E_ms);
    }

    if (MAX_ABS_XYZ(s.wi) < EPSILON)
        return {};

    s.weight *= get_tint(s.wi.z);
    s.roughness = Base::roughness();
    return s;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Power
DielectricLobe<BSDF_ROOT>::get_tint(float cosNI) const
{
    return cosNI > 0 ? refl_tint : refr_tint;
}

}  // namespace mtx

BSDL_LEAVE_NAMESPACE
