// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

template<typename BSDF_ROOT, bool TR = false>
struct OrenNayarLobeGen : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;

    struct Data {
        Imath::V3f N;
        float sigma;
        int improved;
        using lobe_type = OrenNayarLobeGen<BSDF_ROOT, TR>;
    };
    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::N), R::param(&D::sigma),
                   R::param(&D::improved, "improved"), R::close() } };
    }

    template<typename T>
    BSDL_INLINE_METHOD OrenNayarLobeGen(T*, const BsdfGlobals& globals,
                                        const Data& data);

    static const char* name()
    {
        return TR ? "oren_nayar_translucent" : "oren_nayar";
    }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& _wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f rnd) const;
    float m_A, m_B;
    bool m_improved;
};

template<typename BSDF_ROOT>
using OrenNayarLobe = OrenNayarLobeGen<BSDF_ROOT, false>;
template<typename BSDF_ROOT>
using OrenNayarTranslucentLobe = OrenNayarLobeGen<BSDF_ROOT, true>;

}  // namespace spi

BSDL_LEAVE_NAMESPACE
