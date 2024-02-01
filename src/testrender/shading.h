// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <OSL/dual_vec.h>
#include <OSL/oslclosure.h>
#include <OSL/oslconfig.h>
#include <OSL/oslexec.h>
#include "sampling.h"


OSL_NAMESPACE_ENTER

/// Individual BSDF (diffuse, phong, refraction, etc ...)
/// Actual implementations of this class are private
struct BSDF {
    struct Sample {
        Sample() : wi(0.0f), weight(0.0f), pdf(0.0f), roughness(0.0f) {}
        Sample(const Sample& o)
            : wi(o.wi), weight(o.weight), pdf(o.pdf), roughness(o.roughness)
        {
        }
        Sample(Vec3 wi, Color3 w, float pdf, float r)
            : wi(wi), weight(w), pdf(pdf), roughness(r)
        {
        }
        Vec3 wi;
        Color3 weight;
        float pdf;
        float roughness;
    };
    BSDF() {}
    virtual Color3 get_albedo(const Vec3& /*wo*/) const { return Color3(1); }
    virtual Sample eval(const Vec3& wo, const Vec3& wi) const = 0;
    virtual Sample sample(const Vec3& wo, float rx, float ry, float rz) const
        = 0;
};

/// Represents a weighted sum of BSDFS
/// NOTE: no need to inherit from BSDF here because we use a "flattened" representation and therefore never nest these
///
struct CompositeBSDF {
    CompositeBSDF() : num_bsdfs(0), num_bytes(0) {}

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

    Color3 get_albedo(const Vec3& wo) const
    {
        Color3 result(0, 0, 0);
        for (int i = 0; i < num_bsdfs; i++)
            result += weights[i] * bsdfs[i]->get_albedo(wo);
        return result;
    }

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

private:
    /// Never try to copy this struct because it would invalidate the bsdf pointers
    CompositeBSDF(const CompositeBSDF& c);
    CompositeBSDF& operator=(const CompositeBSDF& c);

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

void
register_closures(ShadingSystem* shadingsys);
void
process_closure(const OSL::ShaderGlobals& sg, ShadingResult& result,
                const ClosureColor* Ci, bool light_only);
Vec3
process_background_closure(const ClosureColor* Ci);

OSL_NAMESPACE_EXIT
