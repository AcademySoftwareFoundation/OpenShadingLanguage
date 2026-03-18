// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/MTX/bsdf_translucent_decl.h>

BSDL_ENTER_NAMESPACE

namespace mtx {

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
TranslucentLobe<BSDF_ROOT>::TranslucentLobe(T* lobe, const BsdfGlobals& globals,
                                            const Data& data)
    : Base(lobe, globals.visible_normal(data.N), 1.0f, globals.lambda_0, true)
    , tint(globals.wave(data.albedo))
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, false);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
TranslucentLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                      const Imath::V3f& wi) const
{
    (void)wo;
    if (wi.z >= 0.0f)
        return {};
    return { wi, tint, fabsf(wi.z) * ONEOVERPI, 1.0f };
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
TranslucentLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                        const Imath::V3f& rnd) const
{
    Imath::V3f wi = sample_cos_hemisphere(rnd.x, rnd.y);
    wi.z          = -wi.z;
    return eval_impl(wo, wi);
}

}  // namespace mtx

BSDL_LEAVE_NAMESPACE
