// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/config.h>
#include <BSDL/tools.h>
#include <Imath/ImathVec.h>

BSDL_ENTER_NAMESPACE

struct GGXDist {
    BSDL_INLINE_METHOD GGXDist() : ax(0), ay(0) {}

    BSDL_INLINE_METHOD GGXDist(float rough, float aniso)
        : ax(SQR(rough)), ay(ax)
    {
        assert(rough >= 0 && rough <= 1);
        assert(aniso >= 0 && aniso <= 1);
        constexpr float ALPHA_MIN = 1e-5f;
        ax                        = std::max(ax * (1 + aniso), ALPHA_MIN);
        ay                        = std::max(ay * (1 - aniso), ALPHA_MIN);
    }

    BSDL_INLINE_METHOD float D(const Imath::V3f& Hr) const;
    BSDL_INLINE_METHOD float G1(Imath::V3f w) const;
    BSDL_INLINE_METHOD float G2_G1(Imath::V3f wi, Imath::V3f wo) const;
    BSDL_INLINE_METHOD Imath::V3f sample(const Imath::V3f& wo, float randu,
                                         float randv) const;

    BSDL_INLINE_METHOD float roughness() const { return std::max(ax, ay); }

private:
    float ax, ay;
};

template<typename BSDF> struct TabulatedEnergyCurve {
    BSDL_INLINE_METHOD TabulatedEnergyCurve(const float roughness,
                                            const float fresnel_index)
        : roughness(roughness), fresnel_index(fresnel_index)
    {
    }

    BSDL_INLINE_METHOD float interpolate_emiss(int i) const;
    BSDL_INLINE_METHOD float get_Emiss_avg() const;
    BSDL_INLINE_METHOD float Emiss_eval(float c) const;

private:
    float roughness, fresnel_index;
};

// Not a full BxDF, just enough implemented to allow baking tables
template<typename Dist> struct MiniMicrofacet {
    // describe how tabulation should be done
    static constexpr int Nc = 16;
    static constexpr int Nr = 16;
    static constexpr int Nf = 1;

    static constexpr float get_cosine(int i)
    {
        // we don't use a uniform spacing of cosines because we want a bit more resolution near 0
        // where the energy compensation tables tend to vary more quickly
        return std::max(SQR(float(i) * (1.0f / (Nc - 1))), 1e-6f);
    }

    explicit MiniMicrofacet(float rough, float) : d(rough, 0.0f) {}

    BSDL_INLINE_METHOD Sample sample(Imath::V3f wo, float randu, float randv,
                                     float randw) const;

private:
    Dist d;
};

namespace spi {

struct MiniMicrofacetGGX : public MiniMicrofacet<GGXDist> {
    explicit MiniMicrofacetGGX(float, float rough, float)
        : MiniMicrofacet<GGXDist>(rough, 0.0f)
    {
    }
    struct Energy {
        float data[Nf * Nr * Nc];
    };
    static BSDL_INLINE_METHOD Energy& get_energy();

    static const char* lut_header() { return "microfacet_tools_luts.h"; }
    static const char* struct_name() { return "MiniMicrofacetGGX"; }
};

}  // namespace spi

template<typename Fresnel> struct MicrofacetMS {
    // describe how tabulation should be done
    static constexpr int Nc = 16;
    static constexpr int Nr = 16;
    static constexpr int Nf = 32;

    static constexpr float get_cosine(int i)
    {
        // we don't use a uniform spacing of cosines because we want a bit more resolution near 0
        // where the energy compensation tables tend to vary more quickly
        return std::max(SQR(float(i) * (1.0f / (Nc - 1))), 1e-6f);
    }
    static constexpr const char* name() { return "MicrofacetMS"; }

    BSDL_INLINE_METHOD MicrofacetMS() {}

    explicit BSDL_INLINE_METHOD MicrofacetMS(float cosNO, float roughness_index,
                                             float fresnel_index);
    BSDL_INLINE_METHOD MicrofacetMS(const GGXDist& dist, const Fresnel& fresnel,
                                    float cosNO, float roughness);
    BSDL_INLINE_METHOD Sample eval(Imath::V3f wo, Imath::V3f wi) const;
    BSDL_INLINE_METHOD Sample sample(Imath::V3f wo, float randu, float randv,
                                     float randw) const;

    BSDL_INLINE_METHOD const Fresnel& getFresnel() const { return f; }

private:
    BSDL_INLINE_METHOD Power computeFmiss() const;

    GGXDist d;
    Fresnel f;
    float Eo;
    float Eo_avg;
};

BSDL_LEAVE_NAMESPACE
