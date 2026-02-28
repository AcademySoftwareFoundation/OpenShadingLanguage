// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/MTX/bsdf_sheen_decl.h>
#include <BSDL/bsdf_decl.h>
#include <BSDL/microfacet_tools_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

using mtx::ZeltnerBurleySheen;

template<typename BSDF_ROOT> struct SheenLTCLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data : public LayeredData {
        Imath::V3f N;
        Imath::C3f tint;
        float roughness;
        int doublesided;
        using lobe_type = SheenLTCLobe<BSDF_ROOT>;
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
    BSDL_INLINE_METHOD SheenLTCLobe(T*, const BsdfGlobals& globals,
                                    const Data& data);

    static const char* name() { return "sheen_ltc"; }

    BSDL_INLINE_METHOD Power albedo_impl() const { return Power(1 - Emiss, 1); }
    BSDL_INLINE_METHOD Power filter_o(const Imath::V3f& wo) const
    {
        return Power(Emiss, 1);
    }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& sample) const;

private:
    ZeltnerBurleySheen sheen;
    Power tint;
    float Emiss;
    bool is_backfacing;
};

}  // namespace spi

BSDL_LEAVE_NAMESPACE
