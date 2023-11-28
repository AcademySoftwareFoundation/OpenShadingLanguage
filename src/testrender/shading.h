// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <OSL/dual_vec.h>
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

namespace {  // anonymous namespace

#ifdef __CUDACC__
static OSL_HOSTDEVICE const char* id_to_string(ClosureIDs id)
{
    switch(id) {
        case ClosureIDs::COMPONENT_BASE_ID: return "COMPONENT_BASE_ID"; break;
        case ClosureIDs::MUL: return "MUL"; break;
        case ClosureIDs::ADD: return "ADD"; break;
        case ClosureIDs::EMISSION_ID: return "EMISSION_ID"; break;
        case ClosureIDs::BACKGROUND_ID: return "BACKGROUND_ID"; break;
        case ClosureIDs::DIFFUSE_ID: return "DIFFUSE_ID"; break;
        case ClosureIDs::OREN_NAYAR_ID: return "OREN_NAYAR_ID"; break;
        case ClosureIDs::TRANSLUCENT_ID: return "TRANSLUCENT_ID"; break;
        case ClosureIDs::PHONG_ID: return "PHONG_ID"; break;
        case ClosureIDs::WARD_ID: return "WARD_ID"; break;
        case ClosureIDs::MICROFACET_ID: return "MICROFACET_ID"; break;
        case ClosureIDs::REFLECTION_ID: return "REFLECTION_ID"; break;
        case ClosureIDs::FRESNEL_REFLECTION_ID: return "FRESNEL_REFLECTION_ID"; break;
        case ClosureIDs::REFRACTION_ID: return "REFRACTION_ID"; break;
        case ClosureIDs::TRANSPARENT_ID: return "TRANSPARENT_ID"; break;
        case ClosureIDs::DEBUG_ID: return "DEBUG_ID"; break;
        case ClosureIDs::HOLDOUT_ID: return "HOLDOUT_ID"; break;
        default: break;
    };
    return "UNKNOWN_ID";
}
#endif

}  // anonymous namespace


#if 1 // Closure params
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
    OIIO::ustring dist;
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
    OIIO::ustring label;
};

struct MxBurleyDiffuseParams {
    OSL::Vec3 N;
    OSL::Color3 albedo;
    float roughness;
    // optional
    OIIO::ustring label;
};

// common to all MaterialX microfacet closures
struct MxMicrofacetBaseParams {
    OSL::Vec3 N, U;
    float roughness_x;
    float roughness_y;
    OIIO::ustring distribution;
    // optional
    OIIO::ustring label;
};

struct MxDielectricParams : public MxMicrofacetBaseParams {
    OSL::Color3 reflection_tint;
    OSL::Color3 transmission_tint;
    float ior;
    // optional
    float thinfilm_thickness;
    float thinfilm_ior;

    OSL::Color3 evalR(float cos_theta) const;
    OSL::Color3 evalT(float cos_theta) const;
};

struct MxConductorParams : public MxMicrofacetBaseParams {
    OSL::Color3 ior;
    OSL::Color3 extinction;
    // optional
    float thinfilm_thickness;
    float thinfilm_ior;

    OSL::Color3 evalR(float cos_theta) const;
    OSL::Color3 evalT(float cos_theta) const;

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

    OSL::Color3 evalR(float cos_theta) const;
    OSL::Color3 evalT(float cos_theta) const;
};

struct MxTranslucentParams {
    OSL::Vec3 N;
    OSL::Color3 albedo;
    // optional
    OIIO::ustring label;
};

struct MxSubsurfaceParams {
    OSL::Vec3 N;
    OSL::Color3 albedo;
    float transmission_depth;
    OSL::Color3 transmission_color;
    float anisotropy;
    // optional
    OIIO::ustring label;
};

struct MxSheenParams {
    OSL::Vec3 N;
    OSL::Color3 albedo;
    float roughness;
    // optional
    OIIO::ustring label;
};

struct MxUniformEdfParams {
    OSL::Color3 emittance;
    // optional
    OIIO::ustring label;
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
    OIIO::ustring label;
};

struct MxMediumVdfParams {
    OSL::Color3 albedo;
    float transmission_depth;
    OSL::Color3 transmission_color;
    float anisotropy;
    float ior;
    int priority;
    // optional
    OIIO::ustring label;
};

}
#endif // Closure params


OSL_NAMESPACE_ENTER


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
    BSDF(ClosureIDs id=EMPTY_ID) : id(id) {}
    virtual OSL_HOSTDEVICE Color3 get_albedo(const Vec3& /*wo*/) const { return Color3(1); }
    virtual OSL_HOSTDEVICE Sample eval(const Vec3& wo, const Vec3& wi) const = 0;
    virtual OSL_HOSTDEVICE Sample sample(const Vec3& wo, float rx, float ry,
                                         float rz) const                     = 0;

#ifdef __CUDACC__
    OSL_HOSTDEVICE Color3 get_albedo_gpu(const Vec3& wo, ClosureIDs id) const;
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
            pdfs[i] = weights[i].dot(path_weight * bsdfs[i]->get_albedo(wo))
                      / (path_weight.x + path_weight.y + path_weight.z);
            assert(pdfs[i] >= 0);
            assert(pdfs[i] <= 1);
            total += pdfs[i];
        }
        if ((!absorb && total > 0) || total > 1) {
            for (int i = 0; i < num_bsdfs; i++)
                pdfs[i] /= total;
        }
    }

    OSL_HOSTDEVICE
    Color3 get_albedo(const Vec3& wo) const
    {
        Color3 result(0, 0, 0);
        for (int i = 0; i < num_bsdfs; i++)
            result += weights[i] * bsdfs[i]->get_albedo(wo);
        return result;
    }

    OSL_HOSTDEVICE
    BSDF::Sample eval(const Vec3& wo, const Vec3& wi) const
    {
        BSDF::Sample s = {};
        for (int i = 0; i < num_bsdfs; i++) {
            BSDF::Sample b = bsdfs[i]->eval(wo, wi);
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
                BSDF::Sample s = bsdfs[i]->sample(wo, rx, ry, rz);
                s.weight *= weights[i] * (1 / pdfs[i]);
                s.pdf *= pdfs[i];
                if (s.pdf == 0.0f)
                    return {};
                // we sampled PDF i, now figure out how much the other bsdfs contribute to the chosen direction
                for (int j = 0; j < num_bsdfs; j++) {
                    if (i != j) {
                        BSDF::Sample b = bsdfs[j]->eval(wo, s.wi);
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
        // make sure we have enough space
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
    OSL_HOSTDEVICE void prepare_gpu(const Vec3& wo, const Color3& path_weight, bool absorb);
    OSL_HOSTDEVICE Color3 get_albedo_gpu(const Vec3& wo) const;
    OSL_HOSTDEVICE BSDF::Sample eval_gpu(const Vec3& wo, const Vec3& wi) const;
    OSL_HOSTDEVICE BSDF::Sample sample_gpu(const Vec3& wo, float rx, float ry, float rz) const;
    OSL_HOSTDEVICE bool add_bsdf_gpu(const Color3& w, const ClosureComponent* comp);
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
