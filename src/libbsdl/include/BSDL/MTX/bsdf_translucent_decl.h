// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>

BSDL_ENTER_NAMESPACE

namespace mtx {

template<typename BSDF_ROOT> struct TranslucentLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;

    struct Data {
        Imath::V3f N;
        Imath::C3f albedo;
        Stringhash label;
        using lobe_type = TranslucentLobe<BSDF_ROOT>;
    };

    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::N), R::param(&D::albedo),
                   R::param(&D::label, "label"), R::close() } };
    }

    template<typename T>
    BSDL_INLINE_METHOD TranslucentLobe(T*, const BsdfGlobals& globals,
                                       const Data& data);

    static constexpr const char* name() { return "translucent_bsdf"; }

    BSDL_INLINE_METHOD Power albedo_impl() const { return tint; }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const;

private:
    Power tint;
};

}  // namespace mtx

BSDL_LEAVE_NAMESPACE
