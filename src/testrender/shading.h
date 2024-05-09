// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <OSL/dual_vec.h>
#include <OSL/hashes.h>
#include <OSL/oslclosure.h>
#include <OSL/oslconfig.h>
#include <OSL/oslexec.h>
#include "sampling.h"

// TODO: This used to be in the anonymous namespace ...
// unique identifier for each closure supported by testrender
enum ClosureIDs {
    COMPONENT_BASE_ID = 0, MUL = -1, ADD = -2,
    EMISSION_ID = 1,
    BACKGROUND_ID,
    DIFFUSE_ID,
    OREN_NAYAR_ID,
    TRANSLUCENT_ID,
    PHONG_ID,
    WARD_ID,
    MICROFACET_ID,
    REFLECTION_ID,
    FRESNEL_REFLECTION_ID,
    REFRACTION_ID,
    TRANSPARENT_ID,
    DEBUG_ID,
    HOLDOUT_ID,
    // See MATERIALX_CLOSURES in stdosl.h
    MX_OREN_NAYAR_DIFFUSE_ID,
    MX_BURLEY_DIFFUSE_ID,
    MX_DIELECTRIC_ID,
    MX_CONDUCTOR_ID,
    MX_GENERALIZED_SCHLICK_ID,
    MX_TRANSLUCENT_ID,
    MX_TRANSPARENT_ID,
    MX_SUBSURFACE_ID,
    MX_SHEEN_ID,
    MX_UNIFORM_EDF_ID,
    MX_ANISOTROPIC_VDF_ID,
    MX_MEDIUM_VDF_ID,
    MX_LAYER_ID,
    // TODO: adding vdfs would require extending testrender with volume support ...
    EMPTY_ID
};


// Closure params
namespace {

// these structures hold the parameters of each closure type
// they will be contained inside ClosureComponent
struct EmptyParams {
};
struct DiffuseParams {
    OSL::Vec3 N;
};
struct OrenNayarParams {
    OSL::Vec3 N;
    float sigma;
};
struct PhongParams {
    OSL::Vec3 N;
    float exponent;
};
struct WardParams {
    OSL::Vec3 N, T;
    float ax, ay;
};
struct ReflectionParams {
    OSL::Vec3 N;
    float eta;
};
struct RefractionParams {
    OSL::Vec3 N;
    float eta;
};
struct MicrofacetParams {
    OSL::ustringhash dist;
    OSL::Vec3 N, U;
    float xalpha, yalpha, eta;
    int refract;
};

// MATERIALX_CLOSURES

struct MxOrenNayarDiffuseParams {
    OSL::Vec3 N;
    OSL::Color3 albedo;
    float roughness;
    // optional
    OSL::ustringhash label;
};

struct MxBurleyDiffuseParams {
    OSL::Vec3 N;
    OSL::Color3 albedo;
    float roughness;
    // optional
    OSL::ustringhash label;
};

// common to all MaterialX microfacet closures
struct MxMicrofacetBaseParams {
    OSL::Vec3 N, U;
    float roughness_x;
    float roughness_y;
    OSL::ustringhash distribution;
    // optional
    OSL::ustringhash label;
};

struct MxDielectricParams : public MxMicrofacetBaseParams {
    OSL::Color3 reflection_tint;
    OSL::Color3 transmission_tint;
    float ior;
    // optional
    float thinfilm_thickness;
    float thinfilm_ior;

    OSL_HOSTDEVICE OSL::Color3 evalR(float cos_theta) const;
    OSL_HOSTDEVICE OSL::Color3 evalT(float cos_theta) const;
};

struct MxConductorParams : public MxMicrofacetBaseParams {
    OSL::Color3 ior;
    OSL::Color3 extinction;
    // optional
    float thinfilm_thickness;
    float thinfilm_ior;

    OSL_HOSTDEVICE OSL::Color3 evalR(float cos_theta) const;
    OSL_HOSTDEVICE OSL::Color3 evalT(float cos_theta) const;

    // Avoid function was declared but never referenced
    // float get_ior() const
    // {
    //     return 0;  // no transmission possible
    // }
};

struct MxGeneralizedSchlickParams : public MxMicrofacetBaseParams {
    OSL::Color3 reflection_tint;
    OSL::Color3 transmission_tint;
    OSL::Color3 f0;
    OSL::Color3 f90;
    float exponent;
    // optional
    float thinfilm_thickness;
    float thinfilm_ior;

    OSL_HOSTDEVICE OSL::Color3 evalR(float cos_theta) const;
    OSL_HOSTDEVICE OSL::Color3 evalT(float cos_theta) const;
};

struct MxTranslucentParams {
    OSL::Vec3 N;
    OSL::Color3 albedo;
    // optional
    OSL::ustringhash label;
};

struct MxSubsurfaceParams {
    OSL::Vec3 N;
    OSL::Color3 albedo;
    float transmission_depth;
    OSL::Color3 transmission_color;
    float anisotropy;
    // optional
    OSL::ustringhash label;
};

struct MxSheenParams {
    OSL::Vec3 N;
    OSL::Color3 albedo;
    float roughness;
    // optional
    OSL::ustringhash label;
};

struct MxUniformEdfParams {
    OSL::Color3 emittance;
    // optional
    OSL::ustringhash label;
};

struct MxLayerParams {
    OSL::ClosureColor* top;
    OSL::ClosureColor* base;
};

struct MxAnisotropicVdfParams {
    OSL::Color3 albedo;
    OSL::Color3 extinction;
    float anisotropy;
    // optional
    OSL::ustringhash label;
};

struct MxMediumVdfParams {
    OSL::Color3 albedo;
    float transmission_depth;
    OSL::Color3 transmission_color;
    float anisotropy;
    float ior;
    int priority;
    // optional
    OSL::ustringhash label;
};

}


OSL_NAMESPACE_ENTER


struct ShadingResult;


/// Individual BSDF (diffuse, phong, refraction, etc ...)
/// Actual implementations of this class are private
struct BSDF {
    struct Sample {
        OSL_HOSTDEVICE Sample() : wi(0.0f), weight(0.0f), pdf(0.0f), roughness(0.0f) {}
        OSL_HOSTDEVICE Sample(const Sample& o)
            : wi(o.wi), weight(o.weight), pdf(o.pdf), roughness(o.roughness)
        {
        }
        OSL_HOSTDEVICE Sample(Vec3 wi, Color3 w, float pdf, float r)
            : wi(wi), weight(w), pdf(pdf), roughness(r)
        {
        }
        Vec3 wi;
        Color3 weight;
        float pdf;
        float roughness;
    };
    OSL_HOSTDEVICE BSDF(ClosureIDs id=EMPTY_ID) : id(id) {}
    virtual OSL_HOSTDEVICE Color3 get_albedo(const Vec3& /*wo*/) const { return Color3(1); }
    virtual OSL_HOSTDEVICE Sample eval(const Vec3& wo, const Vec3& wi) const = 0;
    virtual OSL_HOSTDEVICE Sample sample(const Vec3& wo, float rx, float ry,
                                         float rz) const                     = 0;

#ifdef __CUDACC__
    OSL_HOSTDEVICE Color3 get_albedo_cuda(const Vec3& wo, ClosureIDs id) const;
#endif
    ClosureIDs id;
};

/// Represents a weighted sum of BSDFS
/// NOTE: no need to inherit from BSDF here because we use a "flattened" representation and therefore never nest these
///
struct CompositeBSDF {
    OSL_HOSTDEVICE CompositeBSDF() : num_bsdfs(0), num_bytes(0) {}

    OSL_HOSTDEVICE
    void prepare(const Vec3& wo, const Color3& path_weight, bool absorb)
    {
        float total = 0;
        for (int i = 0; i < num_bsdfs; i++) {
#ifndef __CUDACC__
            pdfs[i] = weights[i].dot(path_weight * bsdfs[i]->get_albedo(wo))
                      / (path_weight.x + path_weight.y + path_weight.z);
#else
            // NB: We're using a full-precision divide here to avoid going slightly
            //     out of range and triggering the following asserts.
            pdfs[i] = __fdiv_rn(weights[i].dot(path_weight * get_bsdf_albedo(bsdfs[i], wo)),
                                path_weight.x + path_weight.y + path_weight.z);
#endif

#ifndef __CUDACC__
            // TODO: Figure out what to do with weights/albedos with negative
            //       components (e.g., as might happen when bipolar noise is
            //       used as a color).

            // The PDF is out-of-range in some test scenes on the CPU path, but
            // these asserts are no-ops in release builds. The asserts are active
            // on the CUDA path, so we need to skip them.
            assert(pdfs[i] >= 0);
            assert(pdfs[i] <= 1);
#endif
            total += pdfs[i];
        }
        if ((!absorb && total > 0) || total > 1) {
            for (int i = 0; i < num_bsdfs; i++) {
#ifndef __CUDACC__
                pdfs[i] /= total;
#else
                pdfs[i] = __fdiv_rn(pdfs[i], total);
#endif
            }
        }
    }

    OSL_HOSTDEVICE
    Color3 get_albedo(const Vec3& wo) const
    {
        Color3 result(0, 0, 0);
        for (int i = 0; i < num_bsdfs; i++) {
#ifndef __CUDACC__
            result += weights[i] * bsdfs[i]->get_albedo(wo);
#else
            result += weights[i] * get_bsdf_albedo(bsdfs[i], wo);
#endif
        }
        return result;
    }

    OSL_HOSTDEVICE
    BSDF::Sample eval(const Vec3& wo, const Vec3& wi) const
    {
        BSDF::Sample s = {};
        for (int i = 0; i < num_bsdfs; i++) {
#ifndef __CUDACC__
            BSDF::Sample b = bsdfs[i]->eval(wo, wi);
#else
            BSDF::Sample b = eval_bsdf(bsdfs[i],wo, wi);
#endif
            b.weight *= weights[i];
            MIS::update_eval(&s.weight, &s.pdf, b.weight, b.pdf, pdfs[i]);
            s.roughness += b.roughness * pdfs[i];
        }
        return s;
    }

    OSL_HOSTDEVICE
    BSDF::Sample sample(const Vec3& wo, float rx, float ry, float rz) const
    {
        float accum = 0;
        for (int i = 0; i < num_bsdfs; i++) {
            if (rx < (pdfs[i] + accum)) {
                rx = (rx - accum) / pdfs[i];
                rx = std::min(rx, 0.99999994f);  // keep result in [0,1)
#ifndef __CUDACC__
                BSDF::Sample s = bsdfs[i]->sample(wo, rx, ry, rz);
#else
                BSDF::Sample s = sample_bsdf(bsdfs[i], wo, rx, ry, rz);
#endif
                s.weight *= weights[i] * (1 / pdfs[i]);
                s.pdf *= pdfs[i];
                if (s.pdf == 0.0f)
                    return {};
                // we sampled PDF i, now figure out how much the other bsdfs contribute to the chosen direction
                for (int j = 0; j < num_bsdfs; j++) {
                    if (i != j) {
#ifndef __CUDACC__
                        BSDF::Sample b = bsdfs[j]->eval(wo, s.wi);
#else
                        BSDF::Sample b = eval_bsdf(bsdfs[j], wo, s.wi);
#endif
                        b.weight *= weights[j];
                        MIS::update_eval(&s.weight, &s.pdf, b.weight, b.pdf,
                                         pdfs[j]);
                    }
                }
                return s;
            }
            accum += pdfs[i];
        }
        return {};
    }

    template<typename BSDF_Type, typename... BSDF_Args>
    OSL_HOSTDEVICE
    bool add_bsdf(const Color3& w, BSDF_Args&&... args)
    {
        if (num_bsdfs >= MaxEntries)
            return false;
        if (num_bytes + sizeof(BSDF_Type) > MaxSize)
            return false;
        weights[num_bsdfs] = w;
        bsdfs[num_bsdfs]   = new (pool + num_bytes)
            BSDF_Type(std::forward<BSDF_Args>(args)...);
        num_bsdfs++;
        num_bytes += sizeof(BSDF_Type);
        return true;
    }

#ifdef __CUDACC__
    OSL_HOSTDEVICE bool add_bsdf_cuda(const Color3& w, const ClosureComponent* comp, ShadingResult& result);

    // Helper functions to avoid virtual function calls
    OSL_HOSTDEVICE Color3 get_bsdf_albedo(OSL::BSDF* bsdf, const Vec3& wo) const;
    OSL_HOSTDEVICE BSDF::Sample sample_bsdf(OSL::BSDF* bsdf, const Vec3& wo, float rx, float ry, float rz) const;
    OSL_HOSTDEVICE BSDF::Sample eval_bsdf(OSL::BSDF* bsdf, const Vec3& wo, const Vec3& wi) const;
#endif

private:
    /// Never try to copy this struct because it would invalidate the bsdf pointers
    OSL_HOSTDEVICE CompositeBSDF(const CompositeBSDF& c);
    OSL_HOSTDEVICE CompositeBSDF& operator=(const CompositeBSDF& c);

    enum { MaxEntries = 8 };
    enum { MaxSize = 256 * sizeof(float) };

    Color3 weights[MaxEntries];
    float pdfs[MaxEntries];
    BSDF* bsdfs[MaxEntries];
    char pool[MaxSize];
    int num_bsdfs, num_bytes;
};

struct ShadingResult {
    Color3 Le          = Color3(0.0f);
    CompositeBSDF bsdf = {};
    // medium data
    Color3 sigma_s       = Color3(0.0f);
    Color3 sigma_t       = Color3(0.0f);
    float medium_g       = 0.0f;  // volumetric anisotropy
    float refraction_ior = 1.0f;
    int priority         = 0;
};

#if !defined(__CUDACC__)
void
register_closures(ShadingSystem* shadingsys);
void
process_closure(const OSL::ShaderGlobals& sg, ShadingResult& result,
                const ClosureColor* Ci, bool light_only);
Vec3
process_background_closure(const ClosureColor* Ci);
#endif // !defined(__CUDACC__)

OSL_NAMESPACE_EXIT
