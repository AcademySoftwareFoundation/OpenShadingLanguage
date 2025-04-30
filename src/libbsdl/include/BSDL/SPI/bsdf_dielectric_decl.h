// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>
#include <BSDL/microfacet_tools_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

struct DielectricFresnel {
    BSDL_INLINE_METHOD DielectricFresnel(float _eta, bool backside);
    BSDL_INLINE_METHOD float eval(const float c) const;
    BSDL_INLINE_METHOD float avg() const;
    static BSDL_INLINE_METHOD DielectricFresnel from_table_index(float tx,
                                                                 int side);
    BSDL_INLINE_METHOD float table_index() const;

    float eta;

private:
    static constexpr float IOR_MIN = 1.001f;
    static constexpr float IOR_MAX = 5.0f;
};

// side template argument only matters for baking
template<typename Dist, int side> struct Dielectric {
    // describe how tabulation should be done
    static constexpr int Nc = 16;
    static constexpr int Nr = 16;
    static constexpr int Nf = 32;

    static constexpr float get_cosine(int i)
    {
        return std::max(float(i) * (1.0f / (Nc - 1)), 1e-6f);
    }

    explicit BSDL_INLINE_METHOD Dielectric(float, float roughness_index,
                                           float fresnel_index);

    BSDL_INLINE_METHOD
    Dielectric(const Dist& dist, const DielectricFresnel& fresnel,
               float prob_clamp)
        : d(dist), f(fresnel), prob_clamp(prob_clamp)
    {
    }

    BSDL_INLINE_METHOD Sample eval(Imath::V3f wo, Imath::V3f wi, bool doreflect,
                                   bool dorefract) const;
    BSDL_INLINE_METHOD Sample sample(Imath::V3f wo, float randu, float randv,
                                     float randw, bool doreflect,
                                     bool dorefract) const;

    BSDL_INLINE_METHOD Sample sample(Imath::V3f wo, float randu, float randv,
                                     float randw) const
    {
        return sample(wo, randu, randv, randw, true, true);
    }

    BSDL_INLINE_METHOD float fresnel_prob(float f) const;
    BSDL_INLINE_METHOD DielectricFresnel fresnel() const { return f; }
    BSDL_INLINE_METHOD float fresnel(float c) const { return f.eval(c); }
    BSDL_INLINE_METHOD float roughness() const { return d.roughness(); }
    BSDL_INLINE_METHOD float eta() const { return f.eta; }

private:
    Dist d;
    DielectricFresnel f;
    float prob_clamp;
};

struct DielectricFront : public Dielectric<GGXDist, 0> {
    explicit BSDL_INLINE_METHOD DielectricFront(float, float roughness_index,
                                                float fresnel_index);
    BSDL_INLINE_METHOD DielectricFront(const GGXDist& dist,
                                       const DielectricFresnel& fresnel,
                                       float prob_clamp);
    struct Energy {
        float data[Nf * Nr * Nc];
    };
    static BSDL_INLINE_METHOD Energy& get_energy();

    static const char* lut_header() { return "bsdf_dielectric_front_luts.h"; }
    static const char* struct_name() { return "DielectricFront"; }
};

struct DielectricBack : public Dielectric<GGXDist, 1> {
    explicit BSDL_INLINE_METHOD DielectricBack(float, float roughness_index,
                                               float fresnel_index);
    BSDL_INLINE_METHOD DielectricBack(const bsdl::GGXDist& dist,
                                      const DielectricFresnel& fresnel,
                                      float prob_clamp);
    struct Energy {
        float data[Nf * Nr * Nc];
    };
    static BSDL_INLINE_METHOD Energy& get_energy();

    static const char* lut_header() { return "bsdf_dielectric_back_luts.h"; }
    static const char* struct_name() { return "DielectricBack"; }
};

template<typename BSDF_ROOT> struct DielectricLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data {
        Imath::V3f N;
        Imath::V3f T;
        float IOR;
        float roughness;
        float anisotropy;
        Imath::C3f refl_tint;
        Imath::C3f refr_tint;
        float dispersion;
        float force_eta;
        float prob_clamp;  // Not exposed
        float wavelength;
        using lobe_type = DielectricLobe<BSDF_ROOT>;
    };
    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::N), R::param(&D::T), R::param(&D::IOR),
                   R::param(&D::roughness), R::param(&D::anisotropy),
                   R::param(&D::refl_tint), R::param(&D::refr_tint),
                   R::param(&D::dispersion), R::param(&D::force_eta),
                   R::close() } };
    }

    template<typename T>
    BSDL_INLINE_METHOD DielectricLobe(T*, const BsdfGlobals& globals,
                                      const Data& data);

    static constexpr const char* name() { return "dielectric"; }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi, bool doreflect,
                                        bool dorefract) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& _rnd,
                                          bool doreflect, bool dorefract) const;

protected:
    BSDL_INLINE_METHOD Power get_tint(float cosNI) const;
    BSDL_INLINE_METHOD Sample eval_ec_lobe(Sample s) const;

    BSDL_INLINE_METHOD Imath::V3f sample_ec_lobe(float randu, float randv,
                                                 bool back) const;

    DielectricFront spec;  // Also good for back eval
    Power refl_tint;
    Power refr_tint;
    // energy compensation data (R and T lobes)
    float RT_ratio, Eo;
    bool backside;
};

}  // namespace spi

BSDL_LEAVE_NAMESPACE
