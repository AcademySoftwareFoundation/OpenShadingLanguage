// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <BSDL/bsdf_decl.h>

BSDL_ENTER_NAMESPACE

namespace mtx {

template<typename BSDF_ROOT> struct SheenLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;

    enum Mode { CONTY = 0, ZELTNER = 1 };

    struct Data : public LayeredData {
        Imath::V3f N;
        Imath::C3f albedo;
        float roughness;
        int mode;
        Stringhash label;
        using lobe_type = SheenLobe<BSDF_ROOT>;
    };

    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::N), R::param(&D::albedo),
                   R::param(&D::roughness), R::param(&D::label, "label"),
                   R::param(&D::mode, "mode"), R::close() } };
    }

    template<typename T>
    BSDL_INLINE_METHOD SheenLobe(T*, const BsdfGlobals& globals,
                                 const Data& data);

    static constexpr const char* name() { return "sheen_bsdf"; }

    BSDL_INLINE_METHOD Power albedo_impl() const { return tint * (1 - Emiss); }
    BSDL_INLINE_METHOD Power filter_o(const Imath::V3f& wo) const
    {
        return Power(Emiss, 1);
    }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const;

private:
    BSDL_INLINE_METHOD bool use_zeltner() const
    {
        return sheen_mode == ZELTNER;
    }

    Power tint;
    float sheen_alpha;
    float Emiss;
    int sheen_mode;
    bool is_backfacing;
};

// SHD_LEGACY tells the distribution to use the shadowing term from
// the original paper. Otherwise we use Dassault Systemes improvement.
template<bool SHD_LEGACY> struct ContyKullaDist {
    static constexpr float MIN_ROUGHNESS = 0.06f;

    BSDL_INLINE_METHOD ContyKullaDist(float rough)
        : a(CLAMP(rough, MIN_ROUGHNESS, 1.0f))
    {
    }

    BSDL_INLINE_METHOD float D(const Imath::V3f& Hr) const;
    BSDL_INLINE_METHOD float get_lambda(float cosNO) const;
    BSDL_INLINE_METHOD float G2(const Imath::V3f& wo,
                                const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD float roughness() const { return a; }

private:
    float a;
};

template<typename Dist> struct SheenMicrofacet {
    // describe how tabulation should be done
    static constexpr int Nc = 16;
    static constexpr int Nr = 16;
    static constexpr int Nf = 1;

    static constexpr float get_cosine(int i)
    {
        return std::max(float(i) * (1.0f / (Nc - 1)), 1e-6f);
    }
    explicit BSDL_INLINE_METHOD SheenMicrofacet(float rough) : d(rough) {}
    BSDL_INLINE_METHOD Sample eval(const Imath::V3f& wo,
                                   const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample(const Imath::V3f& wo, float randu,
                                     float randv, float) const;
    BSDL_INLINE_METHOD float roughness() const { return d.roughness(); }

private:
    Dist d;
};

template<bool SHD_LEGACY>
struct ContyKullaSheenGen : public SheenMicrofacet<ContyKullaDist<SHD_LEGACY>> {
    using SheenMicrofacet<ContyKullaDist<SHD_LEGACY>>::SheenMicrofacet;
};

struct ContyKullaSheen : public ContyKullaSheenGen<true> {
    using ContyKullaSheenGen<true>::ContyKullaSheenGen;
    explicit BSDL_INLINE_METHOD ContyKullaSheen(float, float rough, float)
        : ContyKullaSheenGen(rough)
    {
    }
    BSDL_INLINE_METHOD float albedo(float cosNO) const;
    struct Energy {
        float data[Nf * Nr * Nc];
    };
    static BSDL_INLINE_METHOD Energy& get_energy();

    static constexpr const char* NS = "mtx";
    static const char* lut_header() { return "MTX/bsdf_contysheen_luts.h"; }
    static const char* struct_name() { return "ContyKullaSheen"; }
};

struct ContyKullaSheenMTX : public ContyKullaSheenGen<false> {
    using ContyKullaSheenGen<false>::ContyKullaSheenGen;
    BSDL_INLINE_METHOD float albedo(float cosNO) const;
};

struct ZeltnerBurleySheen {
#if BAKE_BSDL_TABLES
    // Use uniform sampling, more reliable
    static constexpr bool LTC_SAMPLING = false;
#else
    static constexpr bool LTC_SAMPLING = true;
#endif
    // Skip albedo tables, use LTC coefficents. Disabling is useful for validation
    static constexpr bool LTC_ALBEDO = true;
    // This flattens the look a bit, so leaving it disabled for now
    static constexpr bool FITTED_LTC = false;

    // LTC sampling is weak at low roughness, gains energy, so we clamp it.
    static constexpr float MIN_ROUGHNESS = LTC_SAMPLING ? 0.02f : 0.0f;

    // describe how tabulation should be done
    static constexpr int Nc      = 16;
    static constexpr int Nr      = 16;
    static constexpr int Nf      = 1;
    static constexpr int ltc_res = 32;

    explicit BSDL_INLINE_METHOD ZeltnerBurleySheen(float rough)
        : roughness(CLAMP(rough, MIN_ROUGHNESS, 1.0f))
    {
    }
    // This constructor is just for baking albedo tables
    explicit ZeltnerBurleySheen(float, float rough, float) : roughness(rough) {}

    static constexpr float get_cosine(int i)
    {
        return std::max(float(i) * (1.0f / (Nc - 1)), 1e-6f);
    }

    BSDL_INLINE_METHOD Sample eval(Imath::V3f wo, Imath::V3f wi) const;
    BSDL_INLINE_METHOD Sample eval(Imath::V3f wo, Imath::V3f wi,
                                   Imath::V3f ltc) const;
    BSDL_INLINE_METHOD Sample sample(Imath::V3f wo, float randu, float randv,
                                     float randw) const;
    BSDL_INLINE_METHOD float albedo(float cosNO) const;

    struct Energy {
        float data[Nf * Nr * Nc];
    };
    struct Param {
        Imath::V3f data[32][32];
    };

    static BSDL_INLINE_METHOD Energy& get_energy();

    typedef const Imath::V3f (*V32_array)[32];
    static BSDL_INLINE_METHOD V32_array param_ptr();

    static constexpr const char* NS = "mtx";
    static const char* lut_header() { return "MTX/bsdf_zeltnersheen_luts.h"; }
    static const char* struct_name() { return "ZeltnerBurleySheen"; }

    // Fetch the LTC coefficients by bilinearly interpolating entries in a 32x32
    // lookup table or using a fit.
    BSDL_INLINE_METHOD Imath::V3f fetch_coeffs(float cosNO) const;

private:
    float roughness;
};

}  // namespace mtx

BSDL_LEAVE_NAMESPACE
