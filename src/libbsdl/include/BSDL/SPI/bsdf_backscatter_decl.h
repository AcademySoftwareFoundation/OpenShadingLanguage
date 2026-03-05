// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/MTX/bsdf_sheen_decl.h>
#include <BSDL/bsdf_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

using mtx::ContyKullaSheen;

template<typename BSDF_ROOT> struct CharlieLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data : public LayeredData {
        Imath::V3f N;
        Imath::C3f tint;
        float roughness;
        int doublesided;
        using lobe_type = CharlieLobe;
    };
    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::closure), R::param(&D::N), R::param(&D::tint),
                   R::param(&D::roughness),
                   R::param(&D::doublesided, "doublesided"), R::close() } };
    }

    template<typename T>
    BSDL_INLINE_METHOD CharlieLobe(T*, const BsdfGlobals& globals,
                                   const Data& data);
    static const char* name() { return "sheen"; }

    BSDL_INLINE_METHOD Power albedo_impl() const { return Power(1 - Emiss, 1); }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& sample) const;

private:
    BSDL_INLINE_METHOD float common_roughness(float alpha)
    {
        // Using the PDF we would have if we sampled the microfacet, one of
        // the 1/2 comes from the cosine avg of 1/(4 cosMO).
        //
        // (2 + 1 / alpha) / 4      = 1 / (2 pi roughness^4)
        // 1 / (pi (1 + 4 / alpha)) = roughness^4
        return sqrtf(sqrtf(ONEOVERPI / (1 + 0.5f / alpha)));
    }

    ContyKullaSheen sheen;
    Power tint;
    float Emiss;
    bool is_backfacing;
};

}  // namespace spi

BSDL_LEAVE_NAMESPACE
