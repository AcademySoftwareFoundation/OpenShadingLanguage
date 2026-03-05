// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/MTX/bsdf_sheen_impl.h>
#include <BSDL/SPI/bsdf_backscatter_decl.h>
#include <BSDL/microfacet_tools_decl.h>
#include <BSDL/tools.h>

BSDL_ENTER_NAMESPACE

namespace spi {

template<typename BSDF_ROOT>
template<typename T>
BSDL_INLINE_METHOD
CharlieLobe<BSDF_ROOT>::CharlieLobe(T* lobe, const BsdfGlobals& globals,
                                    const CharlieLobe<BSDF_ROOT>::Data& data)
    : Base(lobe, globals.visible_normal(data.N),
           common_roughness(globals.regularize_roughness(data.roughness)),
           globals.lambda_0, false)
    , sheen(0,
            CLAMP(globals.regularize_roughness(data.roughness),
                  mtx::ContyKullaDist<true>::MIN_ROUGHNESS, 1.0f),
            0)
    , tint(globals.wave(data.tint))
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
CharlieLobe<BSDF_ROOT>::eval_impl(const Imath::V3f& wo,
                                  const Imath::V3f& wi) const
{
    const float cosNO        = wo.z;
    const float cosNI        = wi.z;
    const bool is_reflection = cosNI > 0 && cosNO >= 0;
    const bool do_self       = is_reflection && !is_backfacing;
    if (!do_self)
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
    const bool do_self = !is_backfacing;
    if (!do_self)
        return {};

    Sample ss = sheen.sample(wo, sample.x, sample.y, sample.z);
    ss.weight *= tint;
    ss.roughness = BSDF_ROOT::roughness();
    return ss;
}

}  // namespace spi

BSDL_LEAVE_NAMESPACE
