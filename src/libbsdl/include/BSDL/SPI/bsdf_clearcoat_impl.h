// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/SPI/bsdf_clearcoat_decl.h>
#include <BSDL/microfacet_tools_impl.h>

#ifndef BAKE_BSDL_TABLES
#    include <BSDL/SPI/bsdf_clearcoat_luts.h>
#endif

BSDL_ENTER_NAMESPACE

namespace spi {

BSDL_INLINE_METHOD
PlasticFresnel::PlasticFresnel(float eta) : eta(CLAMP(eta, IOR_MIN, IOR_MAX)) {}

BSDL_INLINE_METHOD Power
PlasticFresnel::eval(const float c) const
{
    assert(c >= 0);   // slightly above 1.0 is ok
    assert(eta > 1);  // avoid singularity at eta==1
    // optimized for c in [0,1] and eta in (1,inf)
    const float g = sqrtf(eta * eta - 1 + c * c);
    const float A = (g - c) / (g + c);
    const float B = (c * (g + c) - 1) / (c * (g - c) + 1);
    return Power(0.5f * A * A * (1 + B * B), 1);
}

BSDL_INLINE_METHOD
Power
PlasticFresnel::avg() const
{
    // see avg_fresnel -- but we know that eta >= 1 here
    return Power((eta - 1) / (4.08567f + 1.00071f * eta), 1);
}

BSDL_INLINE_METHOD PlasticFresnel
PlasticFresnel::from_table_index(float tx)
{
    return PlasticFresnel(LERP(SQR(SQR(tx)), IOR_MIN, IOR_MAX));
}

BSDL_INLINE_METHOD float
PlasticFresnel::table_index() const
{
    // turn the IOR value into something suitable for integrating
    // this is the reverse of the method above
    assert(eta >= IOR_MIN);
    assert(eta <= IOR_MAX);
    float x = (eta - IOR_MIN) * (1.0f / (IOR_MAX - IOR_MIN));
    assert(x >= 0);
    assert(x <= 1);
    x = sqrtf(sqrtf(x));
    assert(x >= 0);
    assert(x <= 1);
    return x;
}

BSDL_INLINE_METHOD
PlasticGGX ::PlasticGGX(float cosNO, float roughness_index, float fresnel_index)
    : MicrofacetMS<PlasticFresnel>(cosNO, roughness_index, fresnel_index)
{
}

BSDL_INLINE_METHOD
PlasticGGX ::PlasticGGX(const bsdl::GGXDist& dist,
                        const PlasticFresnel& fresnel, float cosNO,
                        float roughness)
    : MicrofacetMS<PlasticFresnel>(dist, fresnel, cosNO, roughness)
{
}

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
ClearCoatLobe<BSDF_ROOT>::ClearCoatLobe(T* lobe, const BsdfGlobals& globals,
                                        const Data& data)
    : Base(lobe, globals.visible_normal(data.N), data.U,
           globals.regularize_roughness(data.roughness), globals.lambda_0,
           false)
    , spec(GGXDist(Base::roughness(), CLAMP(data.anisotropy, 0.0f, 1.0f)),
           PlasticFresnel(LERP(CLAMP(data.force_eta, 0.0f, 1.0f),
                               globals.relative_eta(data.IOR), data.IOR)),
           Base::frame.Z.dot(globals.wo), Base::roughness())
    , spec_color(globals.wave(data.spec_color).clamped(0, 1))
    , wo_absorption(1.0f, globals.lambda_0)
    , backside(data.doublesided ? false : globals.backfacing)
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, true);
    bsdl::TabulatedEnergyCurve<PlasticGGX> diff_curve(
        Base::roughness(), spec.getFresnel().table_index());

    const float cosNO = std::min(fabsf(Base::frame.Z.dot(globals.wo)), 1.0f);
    Eo                = !backside ? diff_curve.Emiss_eval(cosNO) : 1;
    if (!data.legacy_absorption && MAX_RGB(data.sigma_a) > 0) {
        float cos_p_artistic = cosNO;
        float cos_p          = cosNO;
        if (data.artistic_mix > 0.0f) {
            const float b = CLAMP(data.absorption_bias, 0.0f, 1.0f);
            const float g = CLAMP(data.absorption_gain, 0.0f, 1.0f);
            // First apply gamma curve
            cos_p_artistic = bias_curve01(cos_p_artistic, b);
            // Then apply sigma curve
            cos_p_artistic = gain_curve01(cos_p_artistic, g);
        }
        // Take into account how the ray bends with the refraction to compute
        // the traveled distance through absorption.
        const float sinNO2  = 1 - SQR(cosNO);
        const float inveta2 = SQR(1 / spec.getFresnel().get_ior());
        cos_p               = sqrtf(1 - std::min(1.0f, inveta2 * sinNO2));
        cos_p = CLAMP(LERP(data.artistic_mix, cos_p, cos_p_artistic), 0.0f,
                      1.0f);
        const float dist = 1 / std::max(cos_p, FLOAT_MIN);

        constexpr auto fast_exp = BSDLConfig::Fast::expf;

        const Power sigma_a = globals.wave(data.sigma_a);
        wo_absorption
            = Power([&](int i) { return fast_exp(-sigma_a[i] * dist); },
                    globals.lambda_0);
    } else
        wo_absorption = Power(1, globals.lambda_0);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
ClearCoatLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                    const Imath::V3f& wi) const
{
    const float cosNI = wi.z;
    const bool isrefl = cosNI > 0;
    const bool doself = isrefl && !backside;
    if (!doself)
        return {};

    Sample sample = spec.eval(wo, wi);
    sample.weight *= spec_color;
    sample.roughness = Base::roughness();
    return sample;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
ClearCoatLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                      const Imath::V3f& rnd) const
{
    const bool doself = !backside;
    if (!doself)
        return {};

    Sample sample = spec.sample(wo, rnd.x, rnd.y, rnd.z);
    sample.weight *= spec_color;
    sample.roughness = Base::roughness();
    return sample;
}

}  // namespace spi

BSDL_LEAVE_NAMESPACE
