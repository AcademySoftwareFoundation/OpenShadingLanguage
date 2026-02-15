// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <OSL/dual_vec.h>
#include <OSL/hashes.h>
#include <OSL/oslclosure.h>
#include <OSL/oslconfig.h>
#include <OSL/oslexec.h>

#include "bsdl_config.h"
#include <BSDL/static_virtual.h>
#include <queue>

#include "bvh.h"
#include "optics.h"
#include "raytracer.h"
#include "sampling.h"

OSL_NAMESPACE_BEGIN


enum ClosureIDs {
    ADD               = -2,
    MUL               = -1,
    COMPONENT_BASE_ID = 0,
    EMISSION_ID       = 1,
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
    // BSDL SPI closures
    SPI_THINLAYER,
    EMPTY_ID
};


// these structures hold the parameters of each closure type
// they will be contained inside ClosureComponent
struct EmptyParams {};
struct DiffuseParams {
    Vec3 N;
};
struct OrenNayarParams {
    Vec3 N;
    float sigma;
};
struct PhongParams {
    Vec3 N;
    float exponent;
};
struct WardParams {
    Vec3 N, T;
    float ax, ay;
};
struct ReflectionParams {
    Vec3 N;
    float eta;
};
struct RefractionParams {
    Vec3 N;
    float eta;
};
struct MicrofacetParams {
    ustringhash dist;
    Vec3 N, U;
    float xalpha, yalpha, eta;
    int refract;
};

// MATERIALX_CLOSURES

struct MxOrenNayarDiffuseParams {
    Vec3 N;
    Color3 albedo;
    float roughness;
    // optional
    ustringhash label;
    int energy_compensation;
};

struct MxBurleyDiffuseParams {
    Vec3 N;
    Color3 albedo;
    float roughness;
    // optional
    ustringhash label;
};

// common to all MaterialX microfacet closures
struct MxMicrofacetBaseParams {
    Vec3 N, U;
    float roughness_x;
    float roughness_y;
    ustringhash distribution;
    // optional
    ustringhash label;
};

struct MxGeneralizedSchlickParams : public MxMicrofacetBaseParams {
    Color3 reflection_tint;
    Color3 transmission_tint;
    Color3 f0;
    Color3 f90;
    float exponent;
    // optional
    float thinfilm_thickness;
    float thinfilm_ior;

    OSL_HOSTDEVICE Color3 evalR(float cos_theta) const
    {
        return reflection_tint
               * fresnel_generalized_schlick(cos_theta, f0, f90, exponent);
    }

    OSL_HOSTDEVICE Color3 evalT(float cos_theta) const
    {
        return transmission_tint
               * (Color3(1.0f)
                  - fresnel_generalized_schlick(cos_theta, f0, f90, exponent));
    }

    OSL_HOSTDEVICE Color3 dirAlbedoR(float cos_theta) const
    {
        // Rational quadratic fit for GGX directional albedo
        // https://github.com/AcademySoftwareFoundation/MaterialX/blob/main/libraries/pbrlib/genglsl/lib/mx_microfacet_specular.glsl
        float x  = OIIO::clamp(cos_theta, 0.0f, 1.0f);
        float y  = sqrtf(roughness_x * roughness_y);  // average alpha
        float x2 = x * x;
        float y2 = y * y;
        Vec2 num = Vec2(0.1003f, 0.9345f) + Vec2(-0.6303f, -2.323f) * x
                   + Vec2(9.748f, 2.229f) * y + Vec2(-2.038f, -3.748f) * x * y
                   + Vec2(29.34f, 1.424f) * x2 + Vec2(-8.245f, -0.7684f) * y2
                   + Vec2(-26.44f, 1.436f) * x2 * y
                   + Vec2(19.99f, 0.2913f) * x * y2
                   + Vec2(-5.448f, 0.6286f) * x2 * y2;
        Vec2 den = Vec2(1.0f, 1.0f) + Vec2(-1.765f, 0.2281f) * x
                   + Vec2(8.263f, 15.94f) * y + Vec2(11.53f, -55.83f) * x * y
                   + Vec2(28.96f, 13.08f) * x2 + Vec2(-7.507f, 41.26f) * y2
                   + Vec2(-36.11f, 54.9f) * x2 * y
                   + Vec2(15.86f, 300.2f) * x * y2
                   + Vec2(33.37f, -285.1f) * x2 * y2;
        float a = OIIO::clamp(num.x / den.x, 0.0f, 1.0f);
        float b = OIIO::clamp(num.y / den.y, 0.0f, 1.0f);
        return reflection_tint * (f0 * a + f90 * b);
    }
};

struct MxTranslucentParams {
    Vec3 N;
    Color3 albedo;
    // optional
    ustringhash label;
};

struct MxSubsurfaceParams {
    Vec3 N;
    Color3 albedo;
    Color3 radius;
    float anisotropy;
    // optional
    ustringhash label;
};

struct MxSheenParams {
    Vec3 N;
    Color3 albedo;
    float roughness;
    // optional
    int mode;
    ustringhash label;
};

struct MxUniformEdfParams {
    Color3 emittance;
    // optional
    ustringhash label;
};

struct MxLayerParams {
    OSL::ClosureColor* top;
    OSL::ClosureColor* base;
};

struct MxAnisotropicVdfParams {
    Color3 albedo;
    Color3 extinction;
    float anisotropy;
    // optional
    ustringhash label;
};

struct MxMediumVdfParams {
    Color3 albedo;
    float transmission_depth;
    Color3 transmission_color;
    float anisotropy;
    float ior;
    int priority;
    // optional
    ustringhash label;
};

struct GGXDist;
struct BeckmannDist;

template<int trans> struct Diffuse;

template<typename Distribution, int Refract> struct Microfacet;
using MicrofacetGGXRefl      = Microfacet<GGXDist, 0>;
using MicrofacetGGXRefr      = Microfacet<GGXDist, 1>;
using MicrofacetGGXBoth      = Microfacet<GGXDist, 2>;
using MicrofacetBeckmannRefl = Microfacet<BeckmannDist, 0>;
using MicrofacetBeckmannRefr = Microfacet<BeckmannDist, 1>;
using MicrofacetBeckmannBoth = Microfacet<BeckmannDist, 2>;

template<typename MxMicrofacetParams, typename Distribution,
         bool EnableTransmissionLobe>
struct MxMicrofacet;

using MxGeneralizedSchlick
    = MxMicrofacet<MxGeneralizedSchlickParams, GGXDist, true>;
using MxGeneralizedSchlickOpaque
    = MxMicrofacet<MxGeneralizedSchlickParams, GGXDist, false>;
struct MxConductor;
struct MxDielectric;

struct Transparent;
struct OrenNayar;
struct Phong;
struct Ward;
struct Reflection;
struct Refraction;
struct MxBurleyDiffuse;
struct EnergyCompensatedOrenNayar;
struct ZeltnerBurleySheen;
struct CharlieSheen;
struct SpiThinLayer;
struct HenyeyGreenstein;

// StaticVirtual generates a switch/case dispatch method for us given
// a list of possible subtypes. We just need to forward declare them.
using AbstractBSDF = bsdl::StaticVirtual<
    Diffuse<0>, Transparent, OrenNayar, Diffuse<1>, Phong, Ward, Reflection,
    Refraction, MicrofacetBeckmannRefl, MicrofacetBeckmannRefr,
    MicrofacetBeckmannBoth, MicrofacetGGXRefl, MicrofacetGGXRefr,
    MicrofacetGGXBoth, MxConductor, MxDielectric, MxBurleyDiffuse,
    EnergyCompensatedOrenNayar, ZeltnerBurleySheen, CharlieSheen,
    MxGeneralizedSchlickOpaque, MxGeneralizedSchlick, SpiThinLayer, HenyeyGreenstein>;

// Then we just need to inherit from AbstractBSDF or AbstractMedium

/// Individual BSDF (diffuse, phong, refraction, etc ...)
/// Actual implementations of this class are private
struct BSDF : public AbstractBSDF {
    struct Sample {
        OSL_HOSTDEVICE Sample()
            : wi(0.0f), weight(0.0f), pdf(0.0f), roughness(0.0f)
        {
        }
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
    // We get the specific BSDF type as a template parameter LOBE in
    // the constructor. We pass it to AbstractBSDF and it computes an
    // id internally to remember.
    template<typename LOBE> OSL_HOSTDEVICE BSDF(LOBE* lobe) : AbstractBSDF(lobe)
    {
    }
    // Default implementations, to be overriden by subclasses
    OSL_HOSTDEVICE Color3 get_albedo(const Vec3& /*wo*/) const
    {
        return Color3(1);
    }
    OSL_HOSTDEVICE Sample eval(const Vec3& wo, const Vec3& wi) const
    {
        return {};
    }
    OSL_HOSTDEVICE Sample sample(const Vec3& wo, float rx, float ry,
                                 float rz) const
    {
        return {};
    }
    // And the "virtual" versions of the above. They are implemented via
    // dispatch with a lambda, but it has to be written after subclasses
    // with their inline methods have been defined. See shading.cpp
    OSL_HOSTDEVICE Color3 get_albedo_vrtl(const Vec3& wo) const;
    OSL_HOSTDEVICE Sample eval_vrtl(const Vec3& wo, const Vec3& wi) const;
    OSL_HOSTDEVICE Sample sample_vrtl(const Vec3& wo, float rx, float ry,
                                      float rz) const;
#ifdef __CUDACC__
    // TODO: This is a total hack to avoid a misaligned address error
    // that sometimes occurs with the EnergyCompensatedOrenNayar BSDF.
    // It's not clear what the issue is or why this fixes it, but that
    // will take a bit of digging.
    int pad;
#endif
};

/// Represents a weighted sum of BSDFS
/// NOTE: no need to inherit from BSDF here because we use a "flattened" representation and therefore never nest these
///
struct CompositeBSDF {
    OSL_HOSTDEVICE CompositeBSDF() : num_bsdfs(0), num_bytes(0) {}

    OSL_HOSTDEVICE void prepare(const Vec3& wo, const Color3& path_weight,
                                bool absorb)
    {
        float total = 0;
        for (int i = 0; i < num_bsdfs; i++) {
            pdfs[i] = weights[i].dot(path_weight
                                     * bsdfs[i]->get_albedo_vrtl(wo))
                      / (path_weight.x + path_weight.y + path_weight.z);
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
                // TODO: This helps avoid NaNs, but it's not clear where the
                // NaNs are coming from.
                pdfs[i] = __fdiv_rz(pdfs[i], total);
#endif
            }
        }
    }

    OSL_HOSTDEVICE Color3 get_albedo(const Vec3& wo) const
    {
        Color3 result(0, 0, 0);
        for (int i = 0; i < num_bsdfs; i++)
            result += weights[i] * bsdfs[i]->get_albedo_vrtl(wo);
        return result;
    }

    OSL_HOSTDEVICE BSDF::Sample eval(const Vec3& wo, const Vec3& wi) const
    {
        BSDF::Sample s = {};
        for (int i = 0; i < num_bsdfs; i++) {
            BSDF::Sample b = bsdfs[i]->eval_vrtl(wo, wi);
            b.weight *= weights[i];
            MIS::update_eval(&s.weight, &s.pdf, b.weight, b.pdf, pdfs[i]);
            s.roughness += b.roughness * pdfs[i];
        }
        return s;
    }

    OSL_HOSTDEVICE BSDF::Sample sample(const Vec3& wo, float rx, float ry,
                                       float rz) const
    {
        float accum = 0;
        for (int i = 0; i < num_bsdfs; i++) {
            if (rx < (pdfs[i] + accum)) {
                rx = (rx - accum) / pdfs[i];
                rx = std::min(rx, 0.99999994f);  // keep result in [0,1)
                BSDF::Sample s = bsdfs[i]->sample_vrtl(wo, rx, ry, rz);
                s.weight *= weights[i] * (1 / pdfs[i]);
                s.pdf *= pdfs[i];
                if (s.pdf == 0.0f)
                    return {};
                // we sampled PDF i, now figure out how much the other bsdfs contribute to the chosen direction
                for (int j = 0; j < num_bsdfs; j++) {
                    if (i != j) {
                        BSDF::Sample b = bsdfs[j]->eval_vrtl(wo, s.wi);
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
    OSL_HOSTDEVICE bool add_bsdf(const Color3& w, BSDF_Args&&... args)
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

private:
    /// Never try to copy this struct because it would invalidate the bsdf pointers
    OSL_HOSTDEVICE CompositeBSDF(const CompositeBSDF& c);
    OSL_HOSTDEVICE CompositeBSDF& operator=(const CompositeBSDF& c);

    OSL_HOSTDEVICE BSDF::Sample eval(const BSDF* bsdf, const Vec3& wo,
                                     const Vec3& wi) const;

    enum { MaxEntries = 8 };
    enum { MaxSize = 256 * sizeof(float) };

    Color3 weights[MaxEntries];
    float pdfs[MaxEntries];
    BSDF* bsdfs[MaxEntries];
    char pool[MaxSize];
    int num_bsdfs, num_bytes;
};

struct MediumParams {
    Color3 sigma_t       = Color3(0.0f);  // extinction coefficient
    Color3 sigma_s       = Color3(0.0f);  // scattering
    float medium_g       = 0.0f;          // volumetric anisotropy
    float refraction_ior = 1.0f;
    int priority         = 0;
    bool is_empty        = true;

    OSL_HOSTDEVICE bool is_vaccum() const
    {
        return sigma_t.x <= 0.0f && sigma_t.y <= 0.0f && sigma_t.z <= 0.0f;
    }

    OSL_HOSTDEVICE bool is_special_priority() const { return priority == 0; }

    OSL_HOSTDEVICE BSDF::Sample sample_phase_func(const Vec3& wo, float rx,
                                                  float ry, float rz) const;
};

struct MediumStack {
    OSL_HOSTDEVICE MediumStack() : depth(0), pool_size(0) {}

    OSL_HOSTDEVICE const MediumParams* get_current_params() const
    {
        if (depth > 0) {
            return mediums[0];
        }
        return nullptr;
    }

    OSL_HOSTDEVICE bool in_medium() const { return depth > 0; }

    OSL_HOSTDEVICE int size() const { return depth; }

    OSL_HOSTDEVICE bool integrate(Ray& r, Sampler& sampler, Intersection& hit,
                                  Color3& path_weight, Color3& path_radiance,
                                  float& bsdf_pdf)
    {
        if (depth <= 0) {
            return false;
        }

        if (need_to_compute_current_params) {
            // only compute data for overlapping mediums
            // if a new medium was added to the stack
            MediumParams new_params;
            num_overlapping = 0;

            for (int i = 0; i < depth; i++) {
                const MediumParams& params_i = *mediums[i];

                if (i == 0) {
                    // current params has to have the same priority as mediums[0]
                    new_params.priority = params_i.priority;
                }
                if (params_i.priority != new_params.priority) {
                    continue;
                }
                overlapping_medium_indices[num_overlapping] = i;
                new_params.sigma_t += params_i.sigma_t;
                new_params.sigma_s += params_i.sigma_s;
                
                float avg_sigma_s_i = (params_i.sigma_s.x
                                       + params_i.sigma_s.y
                                       + params_i.sigma_s.z)
                                      / 3.0f;
                cdf[num_overlapping] = (num_overlapping > 0
                                            ? cdf[num_overlapping - 1]
                                            : 0.0f)
                                       + avg_sigma_s_i;
                new_params.is_empty &= params_i.is_empty;
                num_overlapping++;
            }

            if (num_overlapping > 1 && !new_params.is_empty) {
                float total_cdf = cdf[num_overlapping - 1];
                if (total_cdf > 0.0f) {
                    for (int i = 0; i < num_overlapping; i++) {
                        cdf[i] /= total_cdf;
                    }
                }
            }

            current_params                 = new_params;
            need_to_compute_current_params = false;
        }

        if (current_params.is_empty) {
            return false;
        }

        float total_sigma_s = current_params.sigma_s.x
                              + current_params.sigma_s.y
                              + current_params.sigma_s.z;
        
        if (total_sigma_s <= 0.0f) {
            // pure absorption: apply transmittance to the surface and continue
            Color3 tr = transmittance(current_params.sigma_t, hit.t);
            path_weight *= tr;
            return false;
        }

        Vec3 rand_vol     = sampler.get();
        Vec3 rand_channel = sampler.get();

        // Build per-channel selection weights from sigma_s
        float channel_weights[3] = { current_params.sigma_s.x,
                                     current_params.sigma_s.y,
                                     current_params.sigma_s.z };

        // total_sigma_s is guaranteed > 0 from the early-out above
        float inv_total = 1.0f / total_sigma_s;
        channel_weights[0] *= inv_total;
        channel_weights[1] *= inv_total;
        channel_weights[2] *= inv_total;

        // pick a channel using the cdf of channel_weights
        int channel;
        if (rand_channel.x < channel_weights[0]) {
            channel = 0;
        } else if (rand_channel.x < channel_weights[0] + channel_weights[1]) {
            channel = 1;
        } else {
            channel = 2;
        }

        // sample distance along the ray using the selected channel's sigma_t
        float sigma_t_channel = current_params.sigma_t[channel];
        if (sigma_t_channel <= 0.0f) {
            // channel has scattering but no extinction
            Color3 tr = transmittance(current_params.sigma_t, hit.t);
            path_weight *= tr;
            return false;
        }

        float t_volume = -logf(1.0f - rand_vol.x) / sigma_t_channel;

        bool scatter = t_volume < hit.t;
        float t      = scatter ? t_volume : hit.t;

        Color3 tr = transmittance(current_params.sigma_t, t);

        Color3 density = scatter ? (current_params.sigma_t * tr) : tr;
        float pdf      = density.x * channel_weights[0]
                         + density.y * channel_weights[1]
                         + density.z * channel_weights[2];

        if (pdf <= 0.0f) {
            return false;
        }

        if (scatter) {
            path_weight *= (tr * current_params.sigma_s) / pdf;
        } else {
            path_weight *= (tr / pdf);
            return false;
        }

        // scattering pick a new direction 
        r.origin = r.point(t_volume);

        Vec3 rand_phase = sampler.get();

        int index = 0;
        if (num_overlapping > 1) {
            Vec3 rand_index = sampler.get();

            for (index = 0; index < num_overlapping; ++index) {
                if (rand_index.x < cdf[index]) {
                    break;
                }
            }
        }

        index = fmin(index, num_overlapping - 1);

        int medium_index = overlapping_medium_indices[index];

        BSDF::Sample phase_sample
            = mediums[medium_index]->sample_phase_func(-r.direction,
                                                      rand_phase.x,
                                                      rand_phase.y,
                                                      rand_phase.z);

        if (phase_sample.pdf > 0.0f) {
            path_weight *= phase_sample.weight;
            r.direction = phase_sample.wi;
            bsdf_pdf    = phase_sample.pdf; 
            return true;
        }

        return false;
    }

    OSL_HOSTDEVICE bool add_medium(MediumParams new_params)
    {
        if (depth >= MaxEntries || pool_size >= MaxPool)
            return false;

        MediumParams* p = &pool[pool_size++];
        *p = new_params;

        // find insertion point (sorted by descending priority)
        int insert_pos = depth;
        for (int i = 0; i < depth; ++i) {
            if (new_params.priority > mediums[i]->priority) {
                insert_pos = i;
                break;
            }
        }

        // shift pointers to make room
        for (int j = depth; j > insert_pos; --j) {
            mediums[j] = mediums[j - 1];
        }

        mediums[insert_pos] = p;
        entry_order[depth]  = p;

        need_to_compute_current_params = true;
        depth++;

        return true;
    }

    OSL_HOSTDEVICE void pop_medium()
    {
        if (depth <= 0)
            return;

        depth--;

        // find the most recently added medium 
        MediumParams* p = entry_order[depth];
        int sorted_index = -1;
        for (int i = 0; i <= depth; ++i) {
            if (mediums[i] == p) {
                sorted_index = i;
                break;
            }
        }

        if (sorted_index < 0)
            return;  // shouldn't happen

        // shift pointers down to fill the gap
        for (int j = sorted_index; j < depth; ++j) {
            mediums[j] = mediums[j + 1];
        }

        // reclaim pool if this was the last allocated entry
        if (p == &pool[pool_size - 1])
            pool_size--;

        need_to_compute_current_params = true;
    }


    OSL_HOSTDEVICE bool
    false_intersection_with(const MediumParams& entrant_params)
    {
        const MediumParams* current = get_current_params();
        if (!current)
            return false;

        // special priority will always record an intersection
        if (entrant_params.is_special_priority()
            && current->is_special_priority())
            return false;

        // same non-special priority: never record an intersection
        if (entrant_params.priority == current->priority)
            return true;

        return entrant_params.priority > current->priority;
    }


    OSL_HOSTDEVICE Color3 transmittance(const Color3& sigma_t,
                                        float distance) const
    {  // Beer-Lambert law
        return Color3(expf(-sigma_t.x * distance), expf(-sigma_t.y * distance),
                      expf(-sigma_t.z * distance));
    }

private:
    /// Never try to copy this struct because it would invalidate the medium pointers
    OSL_HOSTDEVICE MediumStack(const MediumStack& c);
    OSL_HOSTDEVICE MediumStack& operator=(const MediumStack& c);

    enum { MaxEntries = 8 };
    enum { MaxPool = 32 };

    MediumParams pool[MaxPool];           // stable storage
    MediumParams* mediums[MaxEntries];    // sorted by descending priority
    MediumParams* entry_order[MaxEntries]; // LIFO order (pointer identity)

    float cdf[MaxEntries];
    int overlapping_medium_indices[MaxEntries];

    MediumParams current_params;

    int depth;
    int pool_size;
    int num_overlapping;

    bool need_to_compute_current_params;

};

struct ShadingResult {
    Color3 Le                = Color3(0.0f);
    CompositeBSDF bsdf       = {};
    MediumParams medium_data = {};
};


void
register_closures(ShadingSystem* shadingsys);
OSL_HOSTDEVICE void
process_closure(const OSL::ShaderGlobals& sg, float path_roughness,
                ShadingResult& result, MediumStack& medium_stack,
                const ClosureColor* Ci, bool light_only);
OSL_HOSTDEVICE Vec3
process_background_closure(const ClosureColor* Ci);

OSL_NAMESPACE_END
