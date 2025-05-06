// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>
#include <BSDL/microfacet_tools_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

struct MetalFresnel {
    BSDL_INLINE_METHOD MetalFresnel() {}

    BSDL_INLINE_METHOD
    MetalFresnel(Power c, Power edge, float artist_blend, float artist_power);
    BSDL_INLINE_METHOD Power eval(float cosine) const;

    BSDL_INLINE_METHOD Power avg() const;

private:
    Power r, g;
    float blend;  // blend between physical and artistic
    float p;      // power of artistic mode
};

template<typename BSDF_ROOT> struct MetalLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data {
        // microfacet params
        Imath::V3f N, U;
        float roughness;
        float anisotropy;
        // fresnel params
        Imath::C3f color;
        Imath::C3f edge_tint;
        float artist_blend;
        float artist_power;
        float force_eta;
        using lobe_type = MetalLobe<BSDF_ROOT>;
    };
    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::N), R::param(&D::U), R::param(&D::roughness),
                   R::param(&D::anisotropy), R::param(&D::color),
                   R::param(&D::edge_tint), R::param(&D::artist_blend),
                   R::param(&D::artist_power), R::param(&D::force_eta),
                   R::close() } };
    }

    static BSDL_INLINE_METHOD float adjust_reflection(float r, float outer_ior);
    static BSDL_INLINE_METHOD Power adjust_reflection(float outer_ior, Power r,
                                                      float force_eta);

    template<typename T>
    BSDL_INLINE_METHOD MetalLobe(T*, const BsdfGlobals& globals,
                                 const Data& data);

    static const char* name() { return "metal"; }

    BSDL_INLINE_METHOD Power albedo_impl() const
    {
        return spec.getFresnel().avg();
    }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const;

private:
    typedef MicrofacetMS<MetalFresnel> GGX;
    Imath::V3f U;
    GGX spec;
};

}  // namespace spi

BSDL_LEAVE_NAMESPACE
