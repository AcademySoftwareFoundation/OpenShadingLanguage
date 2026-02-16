// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/MTX/bsdf_dielectric_decl.h>
#include <BSDL/bsdf_decl.h>
#include <BSDL/microfacet_tools_decl.h>

BSDL_ENTER_NAMESPACE

namespace mtx {

// Generalized Schlick Fresnel: lerp(pow(1-c, exponent), F0, F90) with TIR.
// Inherits from DielectricFresnel to reuse its eta and table_index for
// energy compensation lookups.
struct SchlickFresnel : public DielectricFresnel {
    SchlickFresnel() = default;
    BSDL_INLINE_METHOD
    SchlickFresnel(Power F0, Power F90, float exponent, float eta,
                   bool backfacing);

    BSDL_INLINE_METHOD Power eval(float c) const;

    Power F0, F90;
    float exponent;
    float tir_cos;  // cosine at which TIR kicks in (0 when eta >= 1)
};

template<typename BSDF_ROOT> struct SchlickLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data : public LayeredData {
        Imath::V3f N;
        Imath::V3f U;
        Imath::C3f refl_tint;
        Imath::C3f refr_tint;
        float roughness_x;
        float roughness_y;
        Imath::C3f F0;
        Imath::C3f F90;
        float exponent;
        Stringhash distribution;
        using lobe_type = SchlickLobe<BSDF_ROOT>;
    };
    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::N), R::param(&D::U), R::param(&D::refl_tint),
                   R::param(&D::refr_tint), R::param(&D::roughness_x),
                   R::param(&D::roughness_y), R::param(&D::F0),
                   R::param(&D::F90), R::param(&D::exponent),
                   R::param(&D::distribution), R::close() } };
    }

    template<typename T>
    BSDL_INLINE_METHOD SchlickLobe(T*, const BsdfGlobals& globals,
                                   const Data& data);

    static constexpr const char* name() { return "generalized_schlick_bsdf"; }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const;

    BSDL_INLINE_METHOD Power filter_o(const Imath::V3f& wo) const
    {
        return !dorefr ? Power(E_ms, 1) : Power::ZERO();
    }

protected:
    BSDL_INLINE_METHOD Power get_tint(float cosNI) const;

    DielectricBSDF<SchlickFresnel> spec;
    Power refl_tint;
    Power refr_tint;
    float E_ms;
    bool dorefl;
    bool dorefr;
};

}  // namespace mtx

BSDL_LEAVE_NAMESPACE
