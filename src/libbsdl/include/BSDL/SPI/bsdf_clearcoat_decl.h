// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>
#include <BSDL/microfacet_tools_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

struct PlasticFresnel {
    BSDL_INLINE_METHOD PlasticFresnel() {}
    BSDL_INLINE_METHOD PlasticFresnel(float eta);

    BSDL_INLINE_METHOD Power eval(const float c) const;
    BSDL_INLINE_METHOD Power avg() const;
    static BSDL_INLINE_METHOD PlasticFresnel from_table_index(float tx);
    BSDL_INLINE_METHOD float table_index() const;
    BSDL_INLINE_METHOD float get_ior() const { return eta; }

private:
    static constexpr float IOR_MIN = 1.00000012f;
    static constexpr float IOR_MAX = 5.0f;
    float eta;
};

struct PlasticGGX : public bsdl::MicrofacetMS<PlasticFresnel> {
    explicit BSDL_INLINE_METHOD PlasticGGX(float cosNO, float roughness_index,
                                           float fresnel_index);
    BSDL_INLINE_METHOD PlasticGGX(const bsdl::GGXDist& dist,
                                  const PlasticFresnel& fresnel, float cosNO,
                                  float roughness);
    struct Energy {
        float data[Nf * Nr * Nc];
    };
    static BSDL_INLINE_METHOD Energy& get_energy();

    static constexpr const char* NS = "spi";
    static const char* lut_header() { return "SPI/bsdf_clearcoat_luts.h"; }
    static const char* struct_name() { return "PlasticGGX"; }
};

template<typename BSDF_ROOT> struct ClearCoatLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data : public LayeredData {
        Imath::V3f N;
        Imath::V3f U;
        float IOR;
        float roughness;
        float anisotropy;
        Imath::C3f spec_color;
        Imath::C3f sigma_a;
        int doublesided;
        float force_eta;
        int legacy_absorption;
        float artistic_mix;
        float absorption_bias;
        float absorption_gain;
        using lobe_type = ClearCoatLobe<BSDF_ROOT>;
    };
    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::closure), R::param(&D::N), R::param(&D::U),
                   R::param(&D::IOR), R::param(&D::roughness),
                   R::param(&D::anisotropy), R::param(&D::spec_color),
                   R::param(&D::sigma_a), R::param(&D::doublesided),
                   R::param(&D::force_eta),
                   R::param(&D::artistic_mix, "artistic_mix"),
                   R::param(&D::absorption_bias, "absorption_bias"),
                   R::param(&D::absorption_gain, "absorption_gain"),
                   R::close() } };
    }

    template<typename T>
    BSDL_INLINE_METHOD ClearCoatLobe(T*, const BsdfGlobals& globals,
                                     const Data& data);

    static const char* name() { return "clearcoat"; }

    BSDL_INLINE_METHOD Power albedo_impl() const
    {
        return spec_color * (1 - Eo);
    }

    BSDL_INLINE_METHOD Power filter_o(const Imath::V3f& wo) const
    {
        if (backside)
            return Power::UNIT();
        // wo is the same as when constructed, Eo is cached
        return Eo * wo_absorption;
    }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const;

private:
    Imath::V3f U;
    PlasticGGX spec;
    Power spec_color;
    Power wo_absorption;
    float Eo;
    bool backside;
};

}  // namespace spi

BSDL_LEAVE_NAMESPACE
