// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/SPI/bsdf_backscatter_decl.h>
#include <BSDL/microfacet_tools_decl.h>
#include <BSDL/tools.h>
#ifndef BAKE_BSDL_TABLES
#    include <BSDL/SPI/bsdf_backscatter_luts.h>
#endif

BSDL_ENTER_NAMESPACE

namespace spi {

BSDL_INLINE_METHOD float
CharlieDist::common_roughness(float alpha)
{
    // Using the PDF we would have if we sampled the microfacet, one of
    // the 1/2 comes from the cosine avg of 1/(4 cosMO).
    //
    // (2 + 1 / alpha) / 4      = 1 / (2 pi roughness^4)
    // 1 / (pi (1 + 4 / alpha)) = roughness^4
    return sqrtf(sqrtf(ONEOVERPI / (1 + 0.5f / alpha)));
}

BSDL_INLINE_METHOD float
CharlieDist::D(const Imath::V3f& Hr) const
{
    float cos_theta = Hr.z;
    float sin_theta = sqrtf(1.0f - SQR(cos_theta));
    return BSDLConfig::Fast::powf(sin_theta, 1 / a) * (2 + 1 / a) * 0.5f
           * ONEOVERPI;
}

BSDL_INLINE_METHOD float
CharlieDist::get_lambda(float cosNv) const
{
    float rt = SQR(1 - a);

    // These params come from gnuplot fitting tool. Roughness = 1 and
    // roughness = 0. We interpolate in between with (1 - roughness)^2
    // to get the best match.
    const float a = LERP(rt, 21.5473f, 25.3245f);
    const float b = LERP(rt, 3.82987f, 3.32435f);
    const float c = LERP(rt, 0.19823f, 0.16801f);
    const float d = LERP(rt, -1.97760f, -1.27393f);
    const float e = LERP(rt, -4.32054f, -4.85967f);
    // The curve is anti-symmetrical aroung 0.5
    const float x     = cosNv > 0.5f ? 1 - cosNv : cosNv;
    const float pivot = a / (1 + b * BSDLConfig::Fast::powf(0.5f, c)) + d * 0.5f
                        + e;

    float p = a / (1 + b * BSDLConfig::Fast::powf(x, c)) + d * x + e;
    if (cosNv > 0.5f)  // Mirror around 0.5f
        p = 2 * pivot - p;
    // That curve fits lambda in log scale, now exponentiate
    return BSDLConfig::Fast::expf(p);
}

BSDL_INLINE_METHOD float
CharlieDist::G2(const Imath::V3f& wo, const Imath::V3f& wi) const
{
    assert(wi.z > 0);
    assert(wo.z > 0);
    float cosNI = std::min(1.0f, wi.z);
    float cosNO = std::min(1.0f, wo.z);
    float Li    = get_lambda(cosNI);
    float Lo    = get_lambda(cosNO);
    // This makes the BSDF non-reciprocal. Cheat to hide the terminator
    Li = BSDLConfig::Fast::powf(Li, 1.0f + 2 * SQR(SQR(SQR((1 - cosNI)))));
    return 1 / (1 + Li + Lo);
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

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
CharlieLobe<BSDF_ROOT>::CharlieLobe(T* lobe, const BsdfGlobals& globals,
                                    const CharlieLobe<BSDF_ROOT>::Data& data)
    : Base(lobe, globals.visible_normal(data.N),
           CharlieDist::common_roughness(
               globals.regularize_roughness(data.roughness)),
           globals.lambda_0, false)
    , sheen(0,
            CLAMP(globals.regularize_roughness(data.roughness),
                  CharlieDist::MIN_ROUGHNESS, 1.0f),
            0)
    , tint(globals.wave(data.tint))
    , back(data.doublesided ? false : globals.backfacing)
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, true);
    TabulatedEnergyCurve<CharlieSheen> curve(sheen.roughness(), 0);
    // Get energy compensation taking tint into account
    Eo = back ? 1
              : 1
                    - MIN((1 - curve.Emiss_eval(Base::frame.Z.dot(globals.wo)))
                              * tint.max(),
                          1.0f);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
CharlieLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                  const Imath::V3f& wi) const
{
    const float cosNO = wo.z;
    const float cosNI = wi.z;
    const bool isrefl = cosNI > 0 && cosNO >= 0;
    const bool doself = isrefl && !back;
    if (!doself)
        return {};

    Sample s = sheen.eval(wo, wi);
    s.weight *= tint;
    s.roughness = BSDF_ROOT::roughness();
    return s;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
CharlieLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                    const Imath::V3f& sample) const
{
    const bool doself = !back;
    if (!doself)
        return {};

    Sample ss = sheen.sample(wo, sample.x, sample.y, sample.z);
    ss.weight *= tint;
    ss.roughness = BSDF_ROOT::roughness();
    return ss;
}

}  // namespace spi

BSDL_LEAVE_NAMESPACE
