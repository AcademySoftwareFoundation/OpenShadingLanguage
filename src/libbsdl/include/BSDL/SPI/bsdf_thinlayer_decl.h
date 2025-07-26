// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>
#include <BSDL/microfacet_tools_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

struct ThinFresnel {
    BSDL_INLINE_METHOD ThinFresnel(float eta);

    BSDL_INLINE_METHOD float eval(const float c) const;
    BSDL_INLINE_METHOD float eval_inv(const float c) const;
    BSDL_INLINE_METHOD Power avg() const;
    BSDL_INLINE_METHOD float avgf() const;
    BSDL_INLINE_METHOD float avg_invf() const;

    static BSDL_INLINE_METHOD ThinFresnel from_table_index(float tx);

    // Note index of eta equals index of 1 / eta, so this function works for
    // either side (both tables)
    BSDL_INLINE_METHOD float table_index() const;

    float eta;

private:
    static constexpr float IOR_MIN = 1.001f;
    static constexpr float IOR_MAX = 5.0f;
};

template<typename Dist> struct ThinMicrofacet {
    // describe how tabulation should be done
    static constexpr int Nc = 16;
    static constexpr int Nr = 16;
    static constexpr int Nf = 32;
    // If we assume the reflected cosine follows a cosine distribution for rough
    // microfacted, the average inverse cosine is 2 (analitically), but we run some
    // similations with GGX and it turned out to be closer to 2.2
    static constexpr float AVG_INV_COS = 2.2f;

    static BSDL_INLINE_METHOD float sum_refl_series(float Rout, float Tin,
                                                    float Rin, float A);
    static BSDL_INLINE_METHOD float sum_refr_series(float Rout, float Tin,
                                                    float Rin, float A);
    static constexpr float get_cosine(int i)
    {
        return std::max(float(i) * (1.0f / (Nc - 1)), 1e-6f);
    }
    static constexpr const char* name() { return "Thinlayer"; }

    explicit BSDL_INLINE_METHOD ThinMicrofacet(float, float roughness_index,
                                               float fresnel_index);
    BSDL_INLINE_METHOD ThinMicrofacet(float roughness, float aniso, float eta,
                                      float thickness, float prob_clamp,
                                      Power sigma_t);
    BSDL_INLINE_METHOD Sample eval(const Imath::V3f& wo, const Imath::V3f& wi,
                                   bool doreflect, bool dorefract) const;
    BSDL_INLINE_METHOD Sample sample(Imath::V3f wo, float randu, float randv,
                                     float randw, bool doreflect,
                                     bool dorefract) const;
    BSDL_INLINE_METHOD Sample sample(Imath::V3f wo, float randu, float randv,
                                     float randw) const
    {
        return sample(wo, randu, randv, randw, true, true);
    };
    BSDL_INLINE_METHOD ThinFresnel fresnel() const { return f; }
    BSDL_INLINE_METHOD float fresnel(float c) const { return f.eval(c); }
    BSDL_INLINE_METHOD float eta() const { return f.eta; }
    // Borrowed from dielectric
    BSDL_INLINE_METHOD float fresnel_prob(float f) const;
    BSDL_INLINE_METHOD Sample eval(const Imath::V3f& wo, const Imath::V3f& m,
                                   const Imath::V3f& wi, const bool both,
                                   const float refr_slope_scale,
                                   Power refl_atten, Power refr_atten) const;
    // This function computes the slope scale needed for a micronormal if we
    // are going to fake the refraction with a flipped reflection
    BSDL_INLINE_METHOD float refraction_slope_scale(float cosNO) const;
    BSDL_INLINE_METHOD Imath::V3f scale_slopes(const Imath::V3f& m,
                                               const float s) const;
    BSDL_INLINE_METHOD void attenuation(Imath::V3f wo, Imath::V3f m,
                                        Power* refl, Power* refr) const;

    Dist d;
    Power sigma_t;
    ThinFresnel f;
    float thickness;
    float roughness;
    float prob_clamp;
};

struct Thinlayer : public ThinMicrofacet<GGXDist> {
    explicit BSDL_INLINE_METHOD Thinlayer(float, float roughness_index,
                                          float fresnel_index);
    BSDL_INLINE_METHOD Thinlayer(float roughness, float aniso, float eta,
                                 float thickness, float prob_clamp,
                                 Power sigma_t);
    struct Energy {
        float data[Nf * Nr * Nc];
    };
    static BSDL_INLINE_METHOD Energy& get_energy();

    static constexpr const char* NS = "spi";
    static const char* lut_header() { return "SPI/bsdf_thinlayer_luts.h"; }
    static const char* struct_name() { return "Thinlayer"; }
};

template<typename BSDF_ROOT> struct ThinLayerLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data {
        Imath::V3f N;
        Imath::V3f T;
        float IOR;
        float roughness;
        float anisotropy;
        float thickness;
        float prob_clamp;      // Not exposed
        Imath::C3f refl_tint;  // overall tint on reflected rays
        Imath::C3f refr_tint;  // overall tint on refracted rays
        Imath::C3f sigma_t;
        using lobe_type = ThinLayerLobe<BSDF_ROOT>;
    };
    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::N), R::param(&D::T), R::param(&D::IOR),
                   R::param(&D::roughness), R::param(&D::anisotropy),
                   R::param(&D::thickness), R::param(&D::refl_tint),
                   R::param(&D::refr_tint), R::param(&D::sigma_t),
                   R::close() } };
    }

    template<typename T>
    BSDL_INLINE_METHOD ThinLayerLobe(T*, const BsdfGlobals& globals,
                                     const Data& data);

    static const char* name() { return "thinlayer"; }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi, bool doreflect,
                                        bool dorefract) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& _rnd,
                                          bool doreflect, bool dorefract) const;

protected:
    BSDL_INLINE_METHOD std::pair<Power, Power> get_diff_trans() const;
    BSDL_INLINE_METHOD Power get_tint(float cosNI) const;
    BSDL_INLINE_METHOD void eval_ec_lobe(Sample* s, Imath::V3f wi_l,
                                         const Power diff_tint,
                                         const Power trans_tint,
                                         float Tprob) const;
    BSDL_INLINE_METHOD Imath::V3f sample_lobe(float randu, float randv,
                                              bool back) const;

    Thinlayer spec;
    Power refl_tint;
    Power refr_tint;
    // energy compensation data (R and T lobes)
    float Eo;
};

}  // namespace spi

BSDL_LEAVE_NAMESPACE
