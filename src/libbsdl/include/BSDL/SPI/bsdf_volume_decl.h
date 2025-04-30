// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

template<typename BSDF_ROOT> struct VolumeLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data {
        float g1;
        float g2;
        float blend;
        using lobe_type = VolumeLobe<BSDF_ROOT>;
    };
    template<typename T>
    BSDL_INLINE_METHOD VolumeLobe(T* lobe, Imath::V3f wo, float g1, float g2,
                                  float l0, float blend)
        : Base(lobe, -wo, 1.0f, l0, true)
        , m_g1(CLAMP(g1, -0.99f, 0.99f))
        , m_g2(CLAMP(g2, -0.99f, 0.99f))
        , m_blend(blend)
    {
    }
    template<typename T>
    BSDL_INLINE_METHOD VolumeLobe(T* lobe, const BsdfGlobals& globals,
                                  const Data& data)
        : VolumeLobe(lobe, globals.wo, data.g1, data.g2, globals.lambda_0,
                     data.blend)
    {
    }

    const char* name() const { return "volume"; }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& sample) const;

    // cos_theta is the angle between view and light vectors (both pointing away from shading point)
    static BSDL_INLINE_METHOD float phase_func(float costheta, float g);
    static BSDL_INLINE_METHOD float phase_func(float costheta, float g1,
                                               float g2, float blend);
    static BSDL_INLINE_METHOD Imath::V3f
    sample_phase(float g1, float g2, float blend, const Imath::V2f& sample);

private:
    float m_g1, m_g2, m_blend;
};

}  // namespace spi

BSDL_LEAVE_NAMESPACE
