// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/config.h>
#include <BSDL/params.h>
#include <BSDL/spectrum_decl.h>
#include <BSDL/thinfilm_decl.h>
#include <BSDL/tools.h>

#include <Imath/ImathColor.h>
#include <Imath/ImathVec.h>

BSDL_ENTER_NAMESPACE

struct Sample {
    Imath::V3f wi   = Imath::V3f(0);
    Power weight    = Power::ZERO();
    float pdf       = 0;
    float roughness = 0;

    BSDL_INLINE_METHOD bool null() const { return pdf == 0; }
    BSDL_INLINE_METHOD void update(Power ow, float opdf, float cpdf);
    static BSDL_INLINE_METHOD float stretch(float x, float min, float length);
};

struct LayeredData {
    const void* closure;
};

struct BsdfGlobals {
    BSDL_INLINE_METHOD BsdfGlobals(const Imath::V3f& wo, const Imath::V3f& Nf,
                                   const Imath::V3f& Ngf, bool backfacing,
                                   float path_roughness, float outer_ior,
                                   float lambda_0)
        : wo(wo)
        , Nf(Nf)
        , Ngf(Ngf)
        , backfacing(backfacing)
        , path_roughness(path_roughness)
        , outer_ior(outer_ior)
        , lambda_0(lambda_0)
    {
    }
    struct Filter {
        template<typename T>
        BSDL_INLINE_METHOD Power eval(const T& lobe, const Imath::V3f& wo,
                                      const Imath::V3f& Nf,
                                      const Imath::V3f& wi) const;
        template<typename T>
        BSDL_INLINE_METHOD Power eval(const T& lobe, const Imath::V3f& wo,
                                      const Imath::V3f& Nf,
                                      const Imath::V3f& Ngf,
                                      const Imath::V3f& wi) const;

        float bump_alpha2 = 0;
        Power sigma_t     = Power::ZERO();
        ThinFilm thinfilm = {};
        float lambda_0    = 0;
    };

    BSDL_INLINE_METHOD Filter get_sample_filter(const Imath::V3f& N,
                                                bool bump_shadow) const;
    static BSDL_INLINE_METHOD Imath::V3f visible_normal(const Imath::V3f& wo,
                                                        const Imath::V3f& Ngf,
                                                        const Imath::V3f& N);
    BSDL_INLINE_METHOD Imath::V3f visible_normal(const Imath::V3f& N) const;
    BSDL_INLINE_METHOD float regularize_roughness(float roughness) const;
    BSDL_INLINE_METHOD float relative_eta(float IOR) const
    {
        return IOR / outer_ior;
    }
    BSDL_INLINE_METHOD Power wave(const Imath::C3f& c) const;

    Imath::V3f wo;
    Imath::V3f Nf;
    Imath::V3f Ngf;
    bool backfacing;
    float path_roughness;
    float outer_ior;
    float lambda_0;
};

template<typename BSDF_ROOT> struct Lobe : public BSDF_ROOT {
    using Sample = bsdl::Sample;  // Prevent leaking from BSDF_ROOT

    template<typename T>
    BSDL_INLINE_METHOD Lobe(T* child, const Imath::V3f& Z, float r, float l0,
                            bool tr);
    template<typename T>
    BSDL_INLINE_METHOD Lobe(T* child, const Imath::V3f& Z, const Imath::V3f& X,
                            float r, float l0, bool tr);

    BSDL_INLINE_METHOD void set_absorption(const Power a)
    {
        sample_filter.sigma_t = a;
    }
    BSDL_INLINE_METHOD void set_thinfilm(const ThinFilm& f)
    {
        sample_filter.thinfilm = f;
    }

    Frame frame;
    typename BsdfGlobals::Filter sample_filter;
};

BSDL_LEAVE_NAMESPACE
