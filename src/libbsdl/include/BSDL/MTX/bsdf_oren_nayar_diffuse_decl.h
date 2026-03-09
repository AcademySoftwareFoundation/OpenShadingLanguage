// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>

BSDL_ENTER_NAMESPACE

namespace mtx {

template<typename BSDF_ROOT>
struct OrenNayarDiffuseLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;

    struct Data {
        Imath::V3f N;
        Imath::C3f albedo;
        float roughness;
        int energy_compensation;
        Stringhash label;
        using lobe_type = OrenNayarDiffuseLobe<BSDF_ROOT>;
    };

    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::N), R::param(&D::albedo),
                   R::param(&D::roughness),
                   R::param(&D::energy_compensation, "energy_compensation"),
                   R::param(&D::label, "label"), R::close() } };
    }

    template<typename T>
    BSDL_INLINE_METHOD OrenNayarDiffuseLobe(T*, const BsdfGlobals& globals,
                                            const Data& data);

    static constexpr const char* name() { return "oren_nayar_diffuse_bsdf"; }

    BSDL_INLINE_METHOD Power albedo_impl() const { return diff_albedo; }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const;

private:
    BSDL_INLINE_METHOD float E_FON_analytic(float mu) const;

    static constexpr float constant1_FON = 0.5f - 2.0f / (3.0f * PI);
    static constexpr float constant2_FON = 2.0f / 3.0f - 28.0f / (15.0f * PI);

    Power diff_albedo;
    float diff_roughness;
    bool do_energy_compensation;
};

}  // namespace mtx

BSDL_LEAVE_NAMESPACE
