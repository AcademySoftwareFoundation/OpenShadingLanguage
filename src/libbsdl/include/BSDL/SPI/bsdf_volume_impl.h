// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/SPI/bsdf_volume_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
VolumeLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                 const Imath::V3f& wi) const
{
    // TODO: are we missing a minus sign here?
    float OdotI = CLAMP(-wi.z, -1.0f, 1.0f);
    float pdf   = phase_func(OdotI, m_g1, m_g2, m_blend);
    return { wi, Power::UNIT(), pdf, 1.0f };
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
VolumeLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                   const Imath::V3f& sample) const
{
    Imath::V3f wi = sample_phase(m_g1, m_g2, m_blend, { sample.x, sample.y });
    return eval_impl(wo, wi);
}

// cos_theta is the angle between view and light vectors (both pointing away from shading point)
template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
VolumeLobe<BSDF_ROOT>::phase_func(float costheta, float g)
{
    // NOTE: HG phase function has the nice property of having an exact importance
    //       importance sampling, so it is equal to its pdf
    if (g == 0)
        return 0.25f * ONEOVERPI;
    else {
        assert(fabsf(g) < 1.0f);
        const float num = 0.25f * ONEOVERPI * (1 - g * g);
        const float den = 1 + g * g + 2.0f * g * costheta;
        assert(den > 0);
        const float r = num / sqrtf(den * den * den);  // ^1.5
        assert(r >= 0);
        return r;
    }
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
VolumeLobe<BSDF_ROOT>::phase_func(float costheta, float g1, float g2,
                                  float blend)
{
    const float p1 = phase_func(costheta, g1);
    const float p2 = phase_func(costheta, g2);
    return LERP(blend, p1, p2);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Imath::V3f
VolumeLobe<BSDF_ROOT>::sample_phase(float g1, float g2, float blend,
                                    const Imath::V2f& sample)
{
    float g, x;
    if (sample.x < blend) {
        g = g2;
        x = sample.x / blend;
    } else {
        g = g1;
        x = (sample.x - blend) / (1 - blend);
    }

    float cosTheta;
    if (fabsf(g) < 1e-3f) {
        // avoid the singularity at g=0 and just use uniform sampling
        cosTheta = 1 - 2 * x;
    } else {
        float k  = (1 - g * g) / (1 - g + 2 * g * x);
        cosTheta = (1 + g * g - k * k) / (2 * g);
    }
    float sinTheta = sqrtf(MAX(0.0f, 1.0f - cosTheta * cosTheta));
    float phi      = 2 * sample.y;
    float cosPhi   = BSDLConfig::Fast::cospif(phi);
    float sinPhi   = BSDLConfig::Fast::sinpif(phi);
    return { sinTheta * cosPhi, sinTheta * sinPhi, cosTheta };
}

}  // namespace spi

BSDL_LEAVE_NAMESPACE
