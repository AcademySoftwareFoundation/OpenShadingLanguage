// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/MTX/bsdf_dielectric_impl.h>
#include <BSDL/MTX/bsdf_schlick_decl.h>
#include <BSDL/microfacet_tools_impl.h>

BSDL_ENTER_NAMESPACE

namespace mtx {

SchlickFresnel::SchlickFresnel(Power F0, Power F90, float exponent, float _eta,
                               bool backfacing)
    : DielectricFresnel(_eta, backfacing)
    , F0(F0.clamped(0, 1))
    , F90(F90.clamped(0, 1))
    , exponent(exponent)
    , tir_cos(eta >= 1
                  ? 0
                  : sqrtf(1 - SQR(eta)))  // critical angle from Snell's law
{
}

BSDL_INLINE_METHOD Power
SchlickFresnel::eval(float c) const
{
    c = CLAMP(c, 0.0f, 1.0f);
    if (c < tir_cos)
        return Power::UNIT();
    return LERP(BSDLConfig::Fast::powf(1 - c, exponent), F0, F90);
}

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
SchlickLobe<BSDF_ROOT>::SchlickLobe(T* lobe, const BsdfGlobals& globals,
                                    const Data& data)
    : Base(lobe, globals.visible_normal(data.N), data.U, 0.0f, globals.lambda_0,
           MAX_RGB(data.refr_tint) > 0)
    , refl_tint(globals.wave(data.refl_tint))
    , refr_tint(globals.wave(data.refr_tint))
{
    dorefl              = MAX_RGB(data.refl_tint) > 0;
    dorefr              = MAX_RGB(data.refr_tint) > 0;
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, false);
    // MaterialX expects the raw x/y roughness as input, but for albedo tables it
    // is better to use the roughness/anisotropy parametrization so we can
    // ignore anisotropy
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

    // Derive a physical IOR from F0 using the inverse Schlick approximation.
    // Used for refraction direction and energy compensation table lookups.
    const float avg_F0         = CLAMP(AVG_RGB(data.F0), 0.0f, 0.99f);
    const float sqrt_F0        = sqrtf(avg_F0);
    const float refraction_ior = (1 + sqrt_F0) / (1 - sqrt_F0);
    const Power F0             = globals.wave(data.F0);
    const Power F90            = globals.wave(data.F90);
    SchlickFresnel fresnel(F0, F90, data.exponent, refraction_ior,
                           globals.backfacing);

    E_ms = 0;
    spec = DielectricBSDF<SchlickFresnel>(GGXDist(roughness, aniso, rx < ry),
                                          fresnel, cosNO, roughness, dorefr);
    if (dorefl && !dorefr) {
        // Energy compensation reuses the dielectric Fresnel albedo tables,
        // which assumes the Schlick curve matches the true dielectric Fresnel.
        // This fails with colored F0/F90 or exponents different from 5 in
        // the way that energy may be a bit off.
        E_ms = TabulatedEnergyCurve<DielectricReflFront>(roughness,
                                                         fresnel.table_index())
                   .Emiss_eval(cosNO);
    } else if (dorefr) {
        if (!globals.backfacing) {
            E_ms = TabulatedEnergyCurve<DielectricBothFront>(
                       roughness, fresnel.table_index())
                       .Emiss_eval(cosNO);
        } else {
            E_ms = TabulatedEnergyCurve<DielectricBothBack>(
                       roughness, fresnel.table_index())
                       .Emiss_eval(cosNO);
        }
    }

    Base::set_roughness(roughness);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
SchlickLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                  const Imath::V3f& wi) const
{
    if (!dorefl && !dorefr)
        return {};
    Sample s = spec.eval(wo, wi);
    if (dorefr)
        // Renormalize refraction+reflection to account for multi-scatter energy
        s.weight *= 1 / std::max(0.01f, 1 - E_ms);

    s.weight *= get_tint(s.wi.z);
    s.roughness = Base::roughness();
    return s;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
SchlickLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                    const Imath::V3f& rnd) const
{
    if (!dorefl && !dorefr)
        return {};

    Sample s = spec.sample(wo, rnd.x, rnd.y, rnd.z);
    if (dorefr)
        s.weight *= 1 / std::max(0.01f, 1 - E_ms);

    if (MAX_ABS_XYZ(s.wi) < EPSILON)
        return {};

    s.weight *= get_tint(s.wi.z);
    s.roughness = Base::roughness();
    return s;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Power
SchlickLobe<BSDF_ROOT>::get_tint(float cosNI) const
{
    return cosNI > 0 ? refl_tint : refr_tint;
}

}  // namespace mtx

BSDL_LEAVE_NAMESPACE
