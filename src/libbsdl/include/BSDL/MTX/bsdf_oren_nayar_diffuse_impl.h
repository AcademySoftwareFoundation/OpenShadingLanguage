// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/MTX/bsdf_oren_nayar_diffuse_decl.h>

BSDL_ENTER_NAMESPACE

namespace mtx {

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
OrenNayarDiffuseLobe<BSDF_ROOT>::OrenNayarDiffuseLobe(
    T* lobe, const BsdfGlobals& globals, const Data& data)
    : Base(lobe, globals.visible_normal(data.N), 1.0f, globals.lambda_0, false)
    , diff_albedo(globals.wave(data.albedo))
    , diff_roughness(CLAMP(data.roughness, 0.0f, 1.0f))
    , do_energy_compensation(data.energy_compensation != 0)
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, true);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
OrenNayarDiffuseLobe<BSDF_ROOT>::E_FON_analytic(float mu) const
{
    const float sigma = diff_roughness;
    const float AF    = 1.0f / (1.0f + constant1_FON * sigma);
    const float BF    = sigma * AF;
    const float Si    = sqrtf(std::max(0.0f, 1.0f - mu * mu));
    const float G
        = Si * (BSDLConfig::Fast::acosf(mu) - Si * mu)
          + 2.0f * ((Si / std::max(mu, 1e-7f)) * (1.0f - Si * Si * Si) - Si)
                * (1.0f / 3.0f);
    return AF + (BF * ONEOVERPI) * G;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
OrenNayarDiffuseLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                           const Imath::V3f& wi) const
{
    const float cosNI = CLAMP(wi.z, 0.0f, 1.0f);
    const float cosNO = CLAMP(wo.z, 0.0f, 1.0f);
    if (cosNI <= 0.0f || cosNO <= 0.0f)
        return {};

    const float cosIO = CLAMP(wo.dot(wi), -1.0f, 1.0f);
    const float s     = cosIO - cosNI * cosNO;
    const float pdf   = cosNI * ONEOVERPI;

    if (!do_energy_compensation) {
        // Simplified math from: "A tiny improvement of Oren-Nayar reflectance model"
        // by Yasuhiro Fujii
        // http://mimosa-pudica.net/improved-oren-nayar.html
        // NOTE: This is using the math to match the original qualitative ON model
        // (QON in the paper above) and not the tweak proposed in the text which
        // is a slightly different BRDF (FON in the paper above). This is done for
        // backwards compatibility purposes only.
        const float s2    = SQR(diff_roughness);
        const float A     = 1.0f - 0.50f * s2 / (s2 + 0.33f);
        const float B     = 0.45f * s2 / (s2 + 0.09f);
        const float stinv = s > 0.0f ? s / std::max(cosNI, cosNO) : 0.0f;
        const float f_ss  = A + B * stinv;
        return { wi, diff_albedo * f_ss, pdf, 1.0f };
    } else {
        // Code below from Jamie Portsmouth's tech report on Energy conversion Oren-Nayar
        // See slack thread for whitepaper:
        // https://academysoftwarefdn.slack.com/files/U03SWQFPD08/F06S50CUKV1/oren_nayar.pdf
        // rho should be the albedo which is a parameter of the closure in the Mx parameters
        // This only matters for the color-saturation aspect of the BRDF which is rather subtle anyway
        // and not always desireable for artists. Hardcoding to 1 leaves the coloring entirely up to the
        // closure weight.
        const float AF    = 1.0f / (1.0f + constant1_FON * diff_roughness);
        const float stinv = s > 0.0f ? s / std::max(cosNI, cosNO) : s;
        const float f_ss  = AF * (1.0f + diff_roughness * stinv);
        const float EFo   = E_FON_analytic(cosNO);
        const float EFi   = E_FON_analytic(cosNI);
        const float avgEF = AF * (1.0f + constant2_FON * diff_roughness);

        const Power rho_ms(
            [&](int i) {
                return SQR(diff_albedo[i]) * avgEF
                       / (1 - diff_albedo[i] * std::max(0.0f, 1.0f - avgEF));
            },
            1);
        const float f_ms = std::max(1e-7f, 1.0f - EFo)
                           * std::max(1e-7f, 1.0f - EFi)
                           / std::max(1e-7f, 1.0f - avgEF);

        return { wi, diff_albedo * f_ss + rho_ms * f_ms, pdf, 1.0f };
    }
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
OrenNayarDiffuseLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                             const Imath::V3f& rnd) const
{
    if (wo.z <= 0.0f)
        return {};

    const Imath::V3f wi = sample_cos_hemisphere(rnd.x, rnd.y);
    return eval_impl(wo, wi);
}

}  // namespace mtx

BSDL_LEAVE_NAMESPACE
