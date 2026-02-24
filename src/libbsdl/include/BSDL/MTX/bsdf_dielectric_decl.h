// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>
#include <BSDL/microfacet_tools_decl.h>

BSDL_ENTER_NAMESPACE

namespace mtx {

struct DielectricFresnel {
    DielectricFresnel() = default;
    BSDL_INLINE_METHOD DielectricFresnel(float _eta, bool backside);
    BSDL_INLINE_METHOD Power eval(const float c) const;
    static BSDL_INLINE_METHOD DielectricFresnel from_table_index(float tx,
                                                                 bool backside);
    BSDL_INLINE_METHOD float table_index() const;
    BSDL_INLINE_METHOD float refraction_eta() const { return eta; }

    float eta;

    static constexpr float IOR_MIN = 1.001f;
    static constexpr float IOR_MAX = 5.0f;
};

// Microfacet dielectric, generic over the Fresnel model.
// Concrete types (Dielectric, Schlick) inherit from this.
template<typename Fresnel> struct DielectricBSDF {
    // describe how tabulation should be done
    static constexpr int Nc = 16;
    static constexpr int Nr = 16;
    static constexpr int Nf = 32;

    static constexpr float get_cosine(int i)
    {
        return std::max(float(i) * (1.0f / (Nc - 1)), 1e-6f);
    }

    BSDL_INLINE_METHOD
    DielectricBSDF(const GGXDist& dist, const Fresnel& fresnel, float cosNO,
                   float roughness, bool dorefr);

    DielectricBSDF() = default;

    BSDL_INLINE_METHOD Sample eval(Imath::V3f wo, Imath::V3f wi) const;
    BSDL_INLINE_METHOD Sample sample(Imath::V3f wo, float randu, float randv,
                                     float randw) const;

    BSDL_INLINE_METHOD Fresnel fresnel() const { return f; }

    struct Energy {
        float data[Nf * Nr * Nc];
    };
    static constexpr const char* NS = "mtx";

protected:
    GGXDist d;
    Fresnel f;
    float E_ms;
    bool dorefr;
};

// For table baking purposes, they can't be templated because of the baking
// logic in genluts.cpp
struct DielectricReflFront : public DielectricBSDF<DielectricFresnel> {
    explicit BSDL_INLINE_METHOD DielectricReflFront(float cosNO,
                                                    float roughness_index,
                                                    float fresnel_index);
    static const char* lut_header()
    {
        return "MTX/bsdf_dielectric_reflfront_luts.h";
    }
    static const char* struct_name() { return "DielectricReflFront"; }
    static BSDL_INLINE_METHOD Energy& get_energy();
};
struct DielectricBothFront : public DielectricBSDF<DielectricFresnel> {
    explicit BSDL_INLINE_METHOD DielectricBothFront(float cosNO,
                                                    float roughness_index,
                                                    float fresnel_index);
    static const char* lut_header()
    {
        return "MTX/bsdf_dielectric_bothfront_luts.h";
    }
    static const char* struct_name() { return "DielectricBothFront"; }
    static BSDL_INLINE_METHOD Energy& get_energy();
};
struct DielectricBothBack : public DielectricBSDF<DielectricFresnel> {
    explicit BSDL_INLINE_METHOD
    DielectricBothBack(float cosNO, float roughness_index, float fresnel_index);
    static const char* lut_header()
    {
        return "MTX/bsdf_dielectric_bothback_luts.h";
    }
    static const char* struct_name() { return "DielectricBothBack"; }
    static BSDL_INLINE_METHOD Energy& get_energy();
};

template<typename BSDF_ROOT> struct DielectricLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data : public LayeredData {
        Imath::V3f N;
        Imath::V3f U;
        Imath::C3f refl_tint;
        Imath::C3f refr_tint;
        float roughness_x;
        float roughness_y;
        float IOR;
        Stringhash distribution;
        float thinfilm_thickness;
        float thinfilm_ior;
        float dispersion;
        Imath::C3f absorption;
        using lobe_type = DielectricLobe<BSDF_ROOT>;
    };
    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::N), R::param(&D::U), R::param(&D::refl_tint),
                   R::param(&D::refr_tint), R::param(&D::roughness_x),
                   R::param(&D::roughness_y), R::param(&D::IOR),
                   R::param(&D::distribution),
                   R::param(&D::thinfilm_thickness, "thinfilm_thickness"),
                   R::param(&D::thinfilm_ior, "thinfilm_ior"),
                   R::param(&D::absorption, "absorption"),
                   R::param(&D::dispersion, "dispersion"), R::close() } };
    }

    template<typename T>
    BSDL_INLINE_METHOD DielectricLobe(T*, const BsdfGlobals& globals,
                                      const Data& data);

    static constexpr const char* name() { return "dielectric_bsdf"; }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const;

    BSDL_INLINE_METHOD Power filter_o(const Imath::V3f& wo) const
    {
        // wo is the same as when constructed, Eo is cached
        return !dorefr ? E_ms * wo_absorption : Power::ZERO();
    }

    BSDL_INLINE_METHOD bool single_wavelength() const { return dispersion; }

protected:
    BSDL_INLINE_METHOD Power get_tint(float cosNI) const;

    DielectricBSDF<DielectricFresnel> spec;
    Power refl_tint;
    Power refr_tint;
    Power wo_absorption;
    float E_ms;
    bool dorefl;
    bool dorefr;
    bool dispersion;
};

}  // namespace mtx

BSDL_LEAVE_NAMESPACE
