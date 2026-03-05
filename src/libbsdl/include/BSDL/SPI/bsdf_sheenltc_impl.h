// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/MTX/bsdf_sheen_impl.h>
#include <BSDL/SPI/bsdf_sheenltc_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
SheenLTCLobe<BSDF_ROOT>::SheenLTCLobe(T* lobe, const BsdfGlobals& globals,
                                      const Data& data)
    : Base(lobe, globals.visible_normal(data.N), globals.wo, data.roughness,
           globals.lambda_0, false)
    , sheen(CLAMP(globals.regularize_roughness(data.roughness), 0.0f, 1.0f))
    , tint(globals.wave(data.tint) * 1.333814f)  // Legacy scaling
    , is_backfacing(data.doublesided ? false : globals.backfacing)
{
    Base::sample_filter = globals.get_sample_filter(Base::frame.Z, true);
    const float cosNO   = CLAMP(Base::frame.Z.dot(globals.wo), 0.0f, 1.0f);
    // Get energy compensation taking tint into account
    Emiss = is_backfacing
                ? 1
                : 1 - std::min(sheen.albedo(cosNO) * tint.max(), 1.0f);
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
SheenLTCLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                   const Imath::V3f& wi) const
{
    const float cosNO        = wo.z;
    const float cosNI        = wi.z;
    const bool is_reflection = cosNI > 0 && cosNO >= 0;
    const bool do_self       = is_reflection && !is_backfacing;

    Sample s = {};
    if (do_self) {
        s = sheen.eval(wo, wi);  // Return a grayscale sheen.
        s.weight *= tint;
    }
    return s;
}

template<typename BSDF_ROOT>
BSDL_INLINE_METHOD Sample
SheenLTCLobe<BSDF_ROOT>::sample_impl(const Imath::V3f& wo,
                                     const Imath::V3f& sample) const
{
    const bool do_self = !is_backfacing;

    if (!do_self)
        return {};

    Sample s = sheen.sample(wo, sample.x, sample.y, sample.z);
    s.weight *= tint;
    return s;
}

}  // namespace spi

BSDL_LEAVE_NAMESPACE
