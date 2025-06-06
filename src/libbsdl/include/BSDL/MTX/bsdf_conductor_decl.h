// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>
#include <BSDL/microfacet_tools_decl.h>

BSDL_ENTER_NAMESPACE

namespace mtx {

struct ConductorFresnel {
    BSDL_INLINE_METHOD ConductorFresnel() {}

    BSDL_INLINE_METHOD
    ConductorFresnel(Power IOR, Power extinction, float lambda_0);
    BSDL_INLINE_METHOD Power eval(float cos_theta) const;
    BSDL_INLINE_METHOD Power F0() const;

    BSDL_INLINE_METHOD Power avg() const;

private:
    Power IOR, extinction;
    float lambda_0;
};

template<typename BSDF_ROOT> struct ConductorLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data {
        // microfacet params
        Imath::V3f N, U;
        float roughness_x;
        float roughness_y;
        // fresnel params
        Imath::C3f IOR;
        Imath::C3f extinction;
        const char* distribution;
        using lobe_type = ConductorLobe<BSDF_ROOT>;
    };
    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::N), R::param(&D::U), R::param(&D::roughness_x),
                   R::param(&D::roughness_y), R::param(&D::IOR),
                   R::param(&D::extinction), R::param(&D::distribution),
                   R::close() } };
    }

    template<typename T>
    BSDL_INLINE_METHOD ConductorLobe(T*, const BsdfGlobals& globals,
                                     const Data& data);

    static const char* name() { return "conductor_bsdf"; }

    BSDL_INLINE_METHOD Power albedo_impl() const { return fresnel.avg(); }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const;

private:
    GGXDist dist;
    ConductorFresnel fresnel;
    float E_ms;
};

}  // namespace mtx

BSDL_LEAVE_NAMESPACE
