// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/bsdf_decl.h>

BSDL_ENTER_NAMESPACE

namespace spi {

template<typename BSDF_ROOT, bool TR = false>
struct DiffuseLobeGen : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;

    struct Data {
        Imath::V3f N;
        using lobe_type = DiffuseLobeGen<BSDF_ROOT, TR>;
    };

    static const char* name() { return TR ? "translucent" : "diffuse"; }

    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(), { R::param(&D::N), R::close() } };
    }

    template<typename T>
    BSDL_INLINE_METHOD DiffuseLobeGen(T*, const BsdfGlobals& globals,
                                      const Data& data);

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const;
};

template<typename BSDF_ROOT>
using DiffuseLobe = DiffuseLobeGen<BSDF_ROOT, false>;
template<typename BSDF_ROOT>
using TranslucentLobe = DiffuseLobeGen<BSDF_ROOT, true>;

template<typename BSDF_ROOT> struct BasicDiffuseLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data {
        // microfacet params
        Imath::V3f N;
        Imath::C3f color;
        float diffuse_roughness;
        float translucent;
        using lobe_type = BasicDiffuseLobe<BSDF_ROOT>;
    };
    template<typename D> static typename LobeRegistry<D>::Entry entry()
    {
        static_assert(
            std::is_base_of<Data, D>::value);  // Make no other assumptions
        using R = LobeRegistry<D>;
        return { name(),
                 { R::param(&D::N), R::param(&D::color),
                   R::param(&D::diffuse_roughness), R::param(&D::translucent),
                   R::close() } };
    }

    template<typename T>
    BSDL_INLINE_METHOD BasicDiffuseLobe(T*, const BsdfGlobals& globals,
                                        const Data& data);

    static const char* name() { return "basic_diffuse"; }

    BSDL_INLINE_METHOD Power albedo_impl() const { return diff_color; }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const;

private:
    float diff_rough;
    float diff_trans;
    Power diff_color;
};

template<typename BSDF_ROOT> struct ChandrasekharLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data {
        Imath::V3f N;
        Imath::C3f albedo;

        using lobe_type = ChandrasekharLobe;
    };
    template<typename T>
    BSDL_INLINE_METHOD ChandrasekharLobe(T*, const BsdfGlobals& globals,
                                         const Data& data);

    const char* name() const { return "chandrasekhar"; }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;

    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const;

private:
    BSDL_INLINE_METHOD float H(float a, float u) const;
    BSDL_INLINE_METHOD Power H(float u) const;

    Power m_a;
};

// Diffusion Transport BRDF:
//   https://www.researchgate.net/publication/333325137_The_Albedo_Problem_in_Nonexponential_Radiative_Transfer
template<typename BSDF_ROOT> struct DiffuseTLobe : public Lobe<BSDF_ROOT> {
    using Base = Lobe<BSDF_ROOT>;
    struct Data {
        Imath::V3f N;
        Imath::C3f albedo;

        using lobe_type = DiffuseTLobe<BSDF_ROOT>;
    };

    template<typename T>
    BSDL_INLINE_METHOD DiffuseTLobe(T*, const BsdfGlobals& globals,
                                    const Data& data);

    const char* name() const { return "deon-diffusion"; }

    BSDL_INLINE_METHOD Sample eval_impl(const Imath::V3f& wo,
                                        const Imath::V3f& wi) const;
    BSDL_INLINE_METHOD Sample sample_impl(const Imath::V3f& wo,
                                          const Imath::V3f& rnd) const;
    // Solve for single-scattering albedo c given diffuse reflectance kD under uniform hemisphere illumination
    static BSDL_INLINE_METHOD Power albedoInvert(const Power in_R);

private:
    // H-function for Gamma-2 flights in 3D with isotropic scattering
    static BSDL_INLINE_METHOD float HFunctionGamma2(const float u,
                                                    const float sqrt1minusc);
    static BSDL_INLINE_METHOD Power HFunctionGamma2(const float u,
                                                    const Power sqrt1minusc);

    Power m_c;
};

}  // namespace spi

BSDL_LEAVE_NAMESPACE
