// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>

BSDL_ENTER_NAMESPACE

namespace mtx {

template<typename BSDF_ROOT> struct BurleyDiffuseLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;

    struct Data {
        Imath::V3f N;
        Imath::C3f albedo;
        float roughness;
        Stringhash label;
        using lobe_type = BurleyDiffuseLobe<BSDF_ROOT>;
    };

    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::N), R::param(&D::albedo),
                   R::param(&D::roughness), R::param(&D::label, "label"),
                   R::close() } };
    }

    template<typename T>
    BSDL_INLINE_METHOD BurleyDiffuseLobe(T*, const BsdfGlobals& globals,
                                         const Data& data);

    static constexpr const char* name() { return "burley_diffuse_bsdf"; }

    BSDL_INLINE_METHOD Power albedo_impl() const { return diff_albedo; }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const;

private:
    static BSDL_INLINE_METHOD float fresnel(float cos_theta, float F90);

    Power diff_albedo;
    float diff_roughness;
};

}  // namespace mtx

BSDL_LEAVE_NAMESPACE
