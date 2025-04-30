// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/SPI/bsdf_oren_nayar_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

template<typename BSDF_ROOT, bool TR>
template<typename T>
BSDL_INLINE_METHOD
OrenNayarLobeGen<BSDF_ROOT, TR>::OrenNayarLobeGen(T* lobe,
                                                  const BsdfGlobals& globals,
                                                  const Data& data)
    : Base(lobe, globals.visible_normal(data.N), 1.0f, globals.lambda_0, TR)
    , m_improved(data.improved)
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, !TR);
    if (m_improved) {
        m_A = 1 - 0.235f * data.sigma;
        m_B = data.sigma * m_A;
    } else {
        float s2 = SQR(data.sigma);
        m_A      = 1 - 0.50f * s2 / (s2 + 0.33f);
        m_B      = 0.45f * s2 / (s2 + 0.09f);
    }
}

template<typename BSDF_ROOT, bool TR>
BSDL_INLINE_METHOD Sample
OrenNayarLobeGen<BSDF_ROOT, TR>::eval_impl(const Imath::V3f& wo,
                                           const Imath::V3f& _wi) const
{
    // When translucent we mirror wi to the other side of the normal and perform
    // a regular oren-nayar BSDF. */
    const Imath::V3f N  = Base::frame.Z;
    const Imath::V3f wi = TR ? _wi - 2 * _wi.z * N : _wi;
    const float NL      = wi.z;
    const float NV      = wo.z;
    if (NL > 0 && NV > 0) {
        const float pdf = NL * ONEOVERPI;
        // Simplified math from: A tiny improvement of Oren-Nayar reflectance model - Yasuhiro Fujii
        // http://mimosa-pudica.net/improved-oren-nayar.html
        const float LV = wi.dot(wo);

        const float s     = LV - NL * NV;
        const float stinv = s > 0 ? s / MAX(NL, NV) : (m_improved ? s : 0);
        const float out   = MAX(m_A + m_B * stinv, 0.0f);
        return { _wi, Power(out, 1), pdf, 1.0f };
    }
    return {};
}

template<typename BSDF_ROOT, bool TR>
BSDL_INLINE_METHOD Sample
OrenNayarLobeGen<BSDF_ROOT, TR>::sample_impl(const Imath::V3f& wo,
                                             const Imath::V3f rnd) const
{
    Imath::V3f wi = sample_cos_hemisphere(rnd.x, rnd.y);
    if (TR)
        wi.z = -wi.z;
    return eval_impl(wo, wi);
}

}  // namespace spi

BSDL_LEAVE_NAMESPACE
