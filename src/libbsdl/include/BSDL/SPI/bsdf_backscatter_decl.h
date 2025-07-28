// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

struct CharlieDist {
    static constexpr float MIN_ROUGHNESS = 0.06f;

    static BSDL_INLINE_METHOD float common_roughness(float alpha);
    BSDL_INLINE_METHOD CharlieDist(float rough)
        : a(CLAMP(rough, MIN_ROUGHNESS, 1.0f))
    {
    }

    BSDL_INLINE_METHOD float D(const Imath::V3f& Hr) const;
    BSDL_INLINE_METHOD float get_lambda(float cosNv) const;
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

struct CharlieSheen : public SheenMicrofacet<CharlieDist> {
    explicit BSDL_INLINE_METHOD CharlieSheen(float, float rough, float)
        : SheenMicrofacet<CharlieDist>(rough)
    {
    }
    struct Energy {
        float data[Nf * Nr * Nc];
    };
    static BSDL_INLINE_METHOD Energy& get_energy();

    static constexpr const char* NS = "spi";
    static const char* lut_header() { return "SPI/bsdf_backscatter_luts.h"; }
    static const char* struct_name() { return "CharlieSheen"; }
};

template<typename BSDF_ROOT> struct CharlieLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data : public LayeredData {
        Imath::V3f N;
        Imath::C3f tint;
        float roughness;
        int doublesided;
        using lobe_type = CharlieLobe;
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
    BSDL_INLINE_METHOD CharlieLobe(T*, const BsdfGlobals& globals,
                                   const Data& data);
    static const char* name() { return "sheen"; }

    BSDL_INLINE_METHOD Power albedo_impl() const { return Power(1 - Eo, 1); }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& sample) const;

private:
    CharlieSheen sheen;
    Power tint;
    float Eo;
    bool back;
};

}  // namespace spi

BSDL_LEAVE_NAMESPACE
