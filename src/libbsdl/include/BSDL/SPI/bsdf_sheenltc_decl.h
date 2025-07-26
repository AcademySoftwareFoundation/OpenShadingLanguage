// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>
#include <BSDL/microfacet_tools_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

struct SheenLTC {
    // describe how tabulation should be done
    static constexpr int Nc     = 16;
    static constexpr int Nr     = 16;
    static constexpr int Nf     = 1;
    static constexpr int ltcRes = 32;

    explicit BSDL_INLINE_METHOD SheenLTC(float rough)
        : roughness(CLAMP(rough, 0.0f, 1.0f))
    {
    }
    // This constructor is just for baking albedo tables
    explicit SheenLTC(float, float rough, float) : roughness(rough) {}

    static constexpr const char* name() { return "sheen_ltc"; }

    static constexpr float get_cosine(int i)
    {
        return std::max(float(i) * (1.0f / (Nc - 1)), 1e-6f);
    }

    BSDL_INLINE_METHOD Sample eval(Imath::V3f wo, Imath::V3f wi) const;
    BSDL_INLINE_METHOD Sample sample(Imath::V3f wo, float randu, float randv,
                                     float randw) const;

    struct Energy {
        float data[Nf * Nr * Nc];
    };
    struct Param {
        Imath::V3f data[32][32];
    };

    static BSDL_INLINE_METHOD Energy& get_energy();

    typedef const Imath::V3f (*V32_array)[32];
    static BSDL_INLINE_METHOD V32_array param_ptr();

    static constexpr const char* NS = "spi";
    static const char* lut_header() { return "SPI/bsdf_sheenltc_luts.h"; }
    static const char* struct_name() { return "SheenLTC"; }

    BSDL_INLINE_METHOD float calculate_phi(const Imath::V3f& v) const;
    BSDL_INLINE_METHOD bool same_hemisphere(const Imath::V3f& wo,
                                            const Imath::V3f& wi) const;

    BSDL_INLINE_METHOD float get_roughness() const { return roughness; }

    BSDL_INLINE_METHOD void albedo_range(float& min_albedo,
                                         float& max_albedo) const;
    BSDL_INLINE_METHOD void compute_scale(float& scale) const;

    // Fetch the LTC coefficients by bilinearly interpolating entries in a 32x32
    // lookup table.
    BSDL_INLINE_METHOD Imath::V3f fetchCoeffs(const Imath::V3f& wo) const;
    // Evaluate the LTC distribution in its local coordinate system.
    BSDL_INLINE_METHOD float evalLTC(const Imath::V3f& wi,
                                     const Imath::V3f& ltcCoeffs) const;
    // Sample from the LTC distribution in its local coordinate system.
    BSDL_INLINE_METHOD Imath::V3f sampleLTC(const Imath::V3f& ltcCoeffs,
                                            float randu, float randv) const;

private:
    float roughness;
};

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

    BSDL_INLINE_METHOD Power albedo_impl() const { return Power(1 - Eo, 1); }
    BSDL_INLINE_METHOD Power filter_o(const Imath::V3f& wo) const
    {
        return Power(Eo, 1);
    }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& sample) const;

private:
    SheenLTC sheenLTC;
    Power tint;
    float Eo;
    bool back;
};

}  // namespace spi

BSDL_LEAVE_NAMESPACE
