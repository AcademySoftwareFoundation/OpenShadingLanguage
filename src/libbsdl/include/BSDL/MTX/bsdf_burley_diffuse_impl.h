// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/MTX/bsdf_burley_diffuse_decl.h>

BSDL_ENTER_NAMESPACE

namespace mtx {

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
BurleyDiffuseLobe<BSDF_ROOT>::BurleyDiffuseLobe(T* lobe,
                                                const BsdfGlobals& globals,
                                                const Data& data)
    : Base(lobe, globals.visible_normal(data.N), 1.0f, globals.lambda_0, false)
    , diff_albedo(globals.wave(data.albedo))
    , diff_roughness(CLAMP(data.roughness, 0.0f, 1.0f))
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, true);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD float
BurleyDiffuseLobe<BSDF_ROOT>::fresnel(float cos_theta, float F90)
{
    const float x = CLAMP(1.0f - cos_theta, 0.0f, 1.0f);
    return LERP(pown<5>(x), 1.0f, F90);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
BurleyDiffuseLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const
{
    if (wo.z <= 0.0f || wi.z <= 0.0f)
        return {};

    const Imath::V3f H = wi + wo;
    if (MAX_ABS_XYZ(H) < EPSILON)
        return {};
    // From "Physically Based Shading at Disney" by Brent Burley, section 5.3
    const Imath::V3f Hn = H.normalized();
    const float cosHI   = CLAMP(wi.dot(Hn), 0.0f, 1.0f);
    const float cosNO   = CLAMP(wo.z, 0.0f, 1.0f);
    const float cosNI   = CLAMP(wi.z, 0.0f, 1.0f);
    // Loses energy with low roughness at grazing angles. Gains it at high roughness
    const float F90  = 0.5f + 2.0f * diff_roughness * SQR(cosHI);
    const float refL = fresnel(cosNI, F90);
    const float refV = fresnel(cosNO, F90);
    const float pdf  = cosNI * ONEOVERPI;

    return { wi, diff_albedo * (refL * refV), pdf, 1.0f };
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
BurleyDiffuseLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const
{
    if (wo.z <= 0.0f)
        return {};

    Imath::V3f wi = sample_cos_hemisphere(rnd.x, rnd.y);
    return eval_impl(wo, wi);
}

}  // namespace mtx

BSDL_LEAVE_NAMESPACE
