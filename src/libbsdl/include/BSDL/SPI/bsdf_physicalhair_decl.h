// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <array>

#include <BSDL/bsdf_decl.h>
#include <BSDL/cdf_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

constexpr int PHYSICAL_HAIR_DEBUG_R   = 1;
constexpr int PHYSICAL_HAIR_DEBUG_TT  = 2;
constexpr int PHYSICAL_HAIR_DEBUG_TRT = 3;

static constexpr int P_MAX = 3;

// From "An Energy-Conserving Hair Reflectance Model" (Weta)
// and  "Importance Sampling for Physically-Based Hair Fiber Models" d'Eon et al
// and  "A Practical and Controllable Hair and Fur Model for Production Path
//       Tracing" (Disney)
// A physically based hair BSDF with four lobes R, TT, TRT and a fourth
// simplified sum of the remaining bounces.
template<typename BSDF_ROOT> struct PhysicalHairLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data {
        Imath::V3f T;
        float IOR;
        float offset;
        float lroughness;
        float troughness;
        float aroughness;
        Imath::C3f R_tint;
        Imath::C3f TT_tint;
        Imath::C3f TRT_tint;
        Imath::C3f absorption;
        int flags;
        float force_eta;
        float scattering;
        float h;  // Offset in the curve
        using lobe_type = PhysicalHairLobe<BSDF_ROOT>;
    };

    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::T), R::param(&D::IOR), R::param(&D::offset),
                   R::param(&D::lroughness), R::param(&D::troughness),
                   R::param(&D::aroughness), R::param(&D::R_tint),
                   R::param(&D::TT_tint), R::param(&D::TRT_tint),
                   R::param(&D::absorption), R::param(&D::scattering),
                   R::param(&D::force_eta), R::param(&D::flags), R::close() } };
    }

    static constexpr float SQRT_PI_OVER_8 = 0.626657069f;

    template<typename T>
    BSDL_INLINE_METHOD PhysicalHairLobe(T*, const BsdfGlobals& globals,
                                        const Data& data);

    static const char* name() { return "physical_hair"; }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& _rnd) const;

    BSDL_INLINE_METHOD StaticCdf<P_MAX + 1> lobe_cdf() const;

    static BSDL_INLINE_METHOD std::array<Power, P_MAX + 1>
    Ap(float cosThetaO, float eta, float h, const Power T, float lambda_0);
    static BSDL_INLINE_METHOD
        std::pair<std::array<float, P_MAX>, std::array<float, P_MAX>>
        sincos_alpha(float offset);
    static BSDL_INLINE_METHOD
        std::pair<std::array<float, P_MAX + 1>, std::array<float, P_MAX + 1>>
        variances(float lrough, float trough, float arough, float scattering,
                  float cosThetaO);
    static BSDL_INLINE_METHOD float log_bessi0(float x);
    static BSDL_INLINE_METHOD float bessi0_time_exp(float x, float exponent);
    static BSDL_INLINE_METHOD float Mp(float cosThetaI, float cosThetaO,
                                       float sinThetaI, float sinThetaO,
                                       float v);
    static BSDL_INLINE_METHOD float Phi(int p, float gammaO, float gammaT);
    static BSDL_INLINE_METHOD float TrimmedLogistic(float x, float s);
    static BSDL_INLINE_METHOD float Np(float phi, int p, float s, float gammaO,
                                       float gammaT);
    static BSDL_INLINE_METHOD float SampleTrimmedLogistic(float u, float s);
    static BSDL_INLINE_METHOD float RemapLongitudinalRoughness(float lr);
    static BSDL_INLINE_METHOD float RemapAzimuthalRoughness(float ar);

    std::array<Power, P_MAX + 1> ap;
    float lrough;
    float trough;
    float arough;
    float scattering;
    // These are for angles from -pi/2 to pi/2
    float gammaO, gammaT;
    float offset;
};

template<typename BSDF_ROOT> struct HairDiffuseLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data {
        Imath::V3f T;
        float IOR;
        Imath::C3f absorption;
        float eccentricity;
        float anisotropy;
        float flatten_density;
        using lobe_type = HairDiffuseLobe<BSDF_ROOT>;
    };
    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::T), R::param(&D::IOR), R::param(&D::absorption),
                   R::param(&D::eccentricity), R::param(&D::anisotropy),
                   R::param(&D::flatten_density), R::close() } };
    }

    static constexpr float MIN_ROUGH = 0.025f;

    template<typename T>
    BSDL_INLINE_METHOD HairDiffuseLobe(T*, const BsdfGlobals& globals,
                                       const Data& data);

    BSDL_INLINE_METHOD Power albedo_impl() const { return color; }
    static const char* name() { return "hair_diffuse"; }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const;

private:
    static BSDL_INLINE_METHOD float ecc2longrough(float ecc, float aniso);
    static BSDL_INLINE_METHOD float ecc2s(float ecc, float aniso);
    static BSDL_INLINE_METHOD float ecc2roughness(float ecc);
    static BSDL_INLINE_METHOD float roughness2ecc(float rough, float ecc);
    static BSDL_INLINE_METHOD float albedo2absorption(float x, float g);
    static BSDL_INLINE_METHOD Power albedo2absorption(Power x, float lambda_0,
                                                      float g);

    Power color;
    float eccentricity;
    float anisotropy;
};

// This BSDF comes from PhysicalHairLobe but isolating either R or TRT
template<typename BSDF_ROOT> struct HairSpecularLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data : public LayeredData {
        Imath::V3f T;
        float IOR;
        float offset;
        float lroughness;
        float aroughness;
        int trt;
        Imath::C3f tint;
        Imath::C3f absorption;
        float force_eta;
        float flatten_density;
        float h;  // Offset in the curve
        using lobe_type = HairSpecularLobe<BSDF_ROOT>;
    };
    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::closure), R::param(&D::T), R::param(&D::IOR),
                   R::param(&D::offset), R::param(&D::lroughness),
                   R::param(&D::aroughness), R::param(&D::trt),
                   R::param(&D::tint), R::param(&D::absorption),
                   R::param(&D::flatten_density), R::param(&D::force_eta),
                   R::close() } };
    }

    template<typename T>
    BSDL_INLINE_METHOD HairSpecularLobe(T*, const BsdfGlobals& globals,
                                        const Data& data);

    static const char* name() { return "hair_specular"; }
    BSDL_INLINE_METHOD Power albedo_impl() const { return color; }

    // Filter the BSDF under it
    BSDL_INLINE_METHOD Power filter_o(const Imath::V3f& wo) const
    {
        return Power(1 - fresnel_term, 1);
    }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& _rnd) const;

    Power color;
    float long_v;                  // Variance arg for M distribution
    float azim_s;                  // Scale arg for azimuthal distribution
    float gammaO, gammaT;          // -pi/2 to pi/2 angle
    float sin2kAlpha, cos2kAlpha;  // Precomputed scale rotation
    float fresnel_term;
    bool trt;
};

}  // namespace spi

BSDL_LEAVE_NAMESPACE
