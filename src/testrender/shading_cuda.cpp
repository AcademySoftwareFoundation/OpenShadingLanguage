// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include "shading.h"
#include <OSL/genclosure.h>
#include "optics.h"
#include "sampling.h"

#ifdef __CUDACC__
#include "cuda/vec_math.h"
#endif


namespace {  // anonymous namespace
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
        case ClosureIDs::MX_OREN_NAYAR_DIFFUSE_ID: return "MX_OREN_NAYAR_DIFFUSE_ID"; break;
        case ClosureIDs::MX_BURLEY_DIFFUSE_ID: return "MX_BURLEY_DIFFUSE_ID"; break;
        case ClosureIDs::MX_DIELECTRIC_ID: return "MX_DIELECTRIC_ID"; break;
        case ClosureIDs::MX_CONDUCTOR_ID: return "MX_CONDUCTOR_ID"; break;
        case ClosureIDs::MX_GENERALIZED_SCHLICK_ID: return "MX_GENERALIZED_SCHLICK_ID"; break;
        case ClosureIDs::MX_TRANSLUCENT_ID: return "MX_TRANSLUCENT_ID"; break;
        case ClosureIDs::MX_TRANSPARENT_ID: return "MX_TRANSPARENT_ID"; break;
        case ClosureIDs::MX_SUBSURFACE_ID: return "MX_SUBSURFACE_ID"; break;
        case ClosureIDs::MX_SHEEN_ID: return "MX_SHEEN_ID"; break;
        case ClosureIDs::MX_UNIFORM_EDF_ID: return "MX_UNIFORM_EDF_ID"; break;
        case ClosureIDs::MX_ANISOTROPIC_VDF_ID: return "MX_ANISOTROPIC_VDF_ID"; break;
        case ClosureIDs::MX_MEDIUM_VDF_ID: return "MX_MEDIUM_VDF_ID"; break;
        case ClosureIDs::MX_LAYER_ID: return "MX_LAYER_ID"; break;
        case ClosureIDs::EMPTY_ID: return "EMPTY_ID"; break;
        default: break;
    };
    return "UNKNOWN_ID";
}
}  // anonymous namespace


using namespace OSL;

OSL_NAMESPACE_ENTER


typedef MxMicrofacet<MxConductorParams, GGXDist, false> MxConductor;
typedef MxMicrofacet<MxDielectricParams, GGXDist, true> MxDielectric;
typedef MxMicrofacet<MxDielectricParams, GGXDist, false> MxDielectricOpaque;
typedef MxMicrofacet<MxGeneralizedSchlickParams, GGXDist, true> MxGeneralizedSchlick;
typedef MxMicrofacet<MxGeneralizedSchlickParams, GGXDist, false> MxGeneralizedSchlickOpaque;


OSL_HOSTDEVICE
bool
CompositeBSDF::add_bsdf_gpu(const Color3& w, const ClosureComponent* comp, ShadingResult& result)
{
    auto sizeof_params = [](ClosureIDs id) {
        size_t sz = 0;
        switch (id) {
        case DIFFUSE_ID: sz = sizeof(Diffuse<0>); break;
        case OREN_NAYAR_ID: sz = sizeof(OrenNayar); break;
        case PHONG_ID: sz = sizeof(Phong); break;
        case WARD_ID: sz = sizeof(Ward); break;
        case REFLECTION_ID: sz = sizeof(Reflection); break;
        case FRESNEL_REFLECTION_ID: sz = sizeof(Reflection); break;
        case REFRACTION_ID: sz = sizeof(Refraction); break;
        case TRANSPARENT_ID: sz = sizeof(Transparent); break;
        case MICROFACET_ID: sz = sizeof(MicrofacetBeckmannRefl); break;
        case MX_OREN_NAYAR_DIFFUSE_ID: sz = sizeof(OrenNayar); break;
        case MX_BURLEY_DIFFUSE_ID: sz = sizeof(MxBurleyDiffuse); break;
        case MX_DIELECTRIC_ID: sz = sizeof(MxDielectric); break;
        case MX_CONDUCTOR_ID: sz = sizeof(MxConductor); break;
        case MX_GENERALIZED_SCHLICK_ID:
            sz = sizeof(MxGeneralizedSchlick);
            break;
        case MX_TRANSLUCENT_ID: sz = sizeof(Diffuse<1>); break;
        case MX_TRANSPARENT_ID: sz = sizeof(Transparent); break;
        case MX_SUBSURFACE_ID: sz = sizeof(Diffuse<0>); break;
        case MX_SHEEN_ID: sz = sizeof(MxSheen); break;
        default: assert(false); break;
        }
        return sz;
    };

    ClosureIDs id = static_cast<ClosureIDs>(comp->id);
    size_t sz     = sizeof_params(id);

    if (num_bsdfs >= MaxEntries)
        return false;
    if (num_bytes + sz > MaxSize)
        return false;

    Color3 weight = w;

    // OptiX doesn't support virtual function calls, so we need to manually
    // construct each of the BSDF sub-types.
    switch (id) {
    case DIFFUSE_ID: {
        // TODO: Do we need to handle trans=1?
        const DiffuseParams* params = comp->as<DiffuseParams>();
        bsdfs[num_bsdfs]                   = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id               = DIFFUSE_ID;
        ((Diffuse<0>*)bsdfs[num_bsdfs])->N = params->N;
        break;
    }
    case OREN_NAYAR_ID: {
        const OrenNayarParams* params = comp->as<OrenNayarParams>();
        bsdfs[num_bsdfs]                      = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                  = OREN_NAYAR_ID;
        ((OrenNayar*)bsdfs[num_bsdfs])->N     = params->N;
        ((OrenNayar*)bsdfs[num_bsdfs])->sigma = params->sigma;
        ((OrenNayar*)bsdfs[num_bsdfs])->calcAB();
        break;
    }
    case TRANSLUCENT_ID: {
        const DiffuseParams* params = comp->as<DiffuseParams>();
        bsdfs[num_bsdfs]                   = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id               = DIFFUSE_ID;
        ((Diffuse<1>*)bsdfs[num_bsdfs])->N = params->N;
        break;
    }
    case PHONG_ID: {
        const PhongParams* params            = comp->as<PhongParams>();
        bsdfs[num_bsdfs]                     = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                 = PHONG_ID;
        ((Phong*)bsdfs[num_bsdfs])->N        = params->N;
        ((Phong*)bsdfs[num_bsdfs])->exponent = params->exponent;
        break;
    }
    case WARD_ID: {
        const WardParams* params      = comp->as<WardParams>();
        bsdfs[num_bsdfs]              = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id          = WARD_ID;
        ((Ward*)bsdfs[num_bsdfs])->N  = params->N;
        ((Ward*)bsdfs[num_bsdfs])->T  = params->T;
        ((Ward*)bsdfs[num_bsdfs])->ax = params->ax;
        ((Ward*)bsdfs[num_bsdfs])->ay = params->ay;
        break;
    }
    case REFLECTION_ID:
    case FRESNEL_REFLECTION_ID: {
        const ReflectionParams* params       = comp->as<ReflectionParams>();
        bsdfs[num_bsdfs]                     = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                 = REFLECTION_ID;
        ((Reflection*)bsdfs[num_bsdfs])->N   = params->N;
        ((Reflection*)bsdfs[num_bsdfs])->eta = params->eta;
        break;
    }
    case REFRACTION_ID: {
        const RefractionParams* params       = comp->as<RefractionParams>();
        bsdfs[num_bsdfs]                     = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                 = REFRACTION_ID;
        ((Refraction*)bsdfs[num_bsdfs])->N   = params->N;
        ((Refraction*)bsdfs[num_bsdfs])->eta = params->eta;
        break;
    }
    case TRANSPARENT_ID: {
        bsdfs[num_bsdfs]     = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id = TRANSPARENT_ID;
        break;
    }
    case MICROFACET_ID: {
        const MicrofacetParams* params        = comp->as<MicrofacetParams>();
        bsdfs[num_bsdfs]                      = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                  = MICROFACET_ID;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->dist    = params->dist;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->N       = params->N;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->U       = params->U;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->xalpha  = params->xalpha;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->yalpha  = params->yalpha;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->eta     = params->eta;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->refract = params->refract;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->calcTangentFrame();

#if 0
        const char* mem  = (const char*)((OSL::ClosureComponent*)comp)->data();
        const char* dist = *(const char**)&mem[0];
        if (HDSTR(dist) == STRING_PARAMS(default))
            printf("default\n");
#endif
        break;
    }
    case MX_OREN_NAYAR_DIFFUSE_ID: {
        const MxOrenNayarDiffuseParams* params = comp->as<MxOrenNayarDiffuseParams>();
        bsdfs[num_bsdfs]                      = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                  = OREN_NAYAR_ID;
        ((OrenNayar*)bsdfs[num_bsdfs])->N     = params->N;
        ((OrenNayar*)bsdfs[num_bsdfs])->sigma = params->roughness;
        ((OrenNayar*)bsdfs[num_bsdfs])->calcAB();
        break;
    }
    case MX_BURLEY_DIFFUSE_ID: {
        const MxBurleyDiffuseParams* params = comp->as<MxBurleyDiffuseParams>();
        bsdfs[num_bsdfs]                    = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                = MX_BURLEY_DIFFUSE_ID;
        ((MxBurleyDiffuse*)bsdfs[num_bsdfs])->N         = params->N;
        ((MxBurleyDiffuse*)bsdfs[num_bsdfs])->albedo    = params->albedo;
        ((MxBurleyDiffuse*)bsdfs[num_bsdfs])->roughness = params->roughness;
        ((MxBurleyDiffuse*)bsdfs[num_bsdfs])->label     = params->label;
        break;
    }
    case MX_DIELECTRIC_ID: {
        const MxDielectricParams* params = comp->as<MxDielectricParams>();
        bsdfs[num_bsdfs]                = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id            = MX_DIELECTRIC_ID;
        // MxMicrofacetBaseParams
        ((MxDielectric*)bsdfs[num_bsdfs])->N            = params->N;
        ((MxDielectric*)bsdfs[num_bsdfs])->U            = params->U;
        ((MxDielectric*)bsdfs[num_bsdfs])->roughness_x  = params->roughness_x;
        ((MxDielectric*)bsdfs[num_bsdfs])->roughness_y  = params->roughness_y;
        ((MxDielectric*)bsdfs[num_bsdfs])->distribution = params->distribution;
        ((MxDielectric*)bsdfs[num_bsdfs])->label        = params->label;
        // MxDielectricParams
        ((MxDielectric*)bsdfs[num_bsdfs])->reflection_tint    = params->reflection_tint;
        ((MxDielectric*)bsdfs[num_bsdfs])->transmission_tint  = params->transmission_tint;
        ((MxDielectric*)bsdfs[num_bsdfs])->ior                = params->ior;
        ((MxDielectric*)bsdfs[num_bsdfs])->thinfilm_thickness = params->thinfilm_thickness;
        ((MxDielectric*)bsdfs[num_bsdfs])->thinfilm_ior       = params->thinfilm_ior;
        if (is_black(params->transmission_tint)) {
            ((MxDielectricOpaque*)bsdfs[num_bsdfs])->set_refraction_ior(1.0f);
        } else {
            ((MxDielectric*)bsdfs[num_bsdfs])->set_refraction_ior(result.refraction_ior);
        }
        ((MxDielectric*)bsdfs[num_bsdfs])->calcTangentFrame();
        break;
    }
    case MX_CONDUCTOR_ID: {
        const MxConductorParams* params = comp->as<MxConductorParams>();
        bsdfs[num_bsdfs]                = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id            = MX_CONDUCTOR_ID;
        // MxMicrofacetBaseParams
        ((MxConductor*)bsdfs[num_bsdfs])->N                  = params->N;
        ((MxConductor*)bsdfs[num_bsdfs])->U                  = params->U;
        ((MxConductor*)bsdfs[num_bsdfs])->roughness_x        = params->roughness_x;
        ((MxConductor*)bsdfs[num_bsdfs])->roughness_y        = params->roughness_y;
        ((MxConductor*)bsdfs[num_bsdfs])->distribution       = params->distribution;
        ((MxConductor*)bsdfs[num_bsdfs])->label              = params->label;
        // MxConductorParams
        ((MxConductor*)bsdfs[num_bsdfs])->ior                = params->ior;
        ((MxConductor*)bsdfs[num_bsdfs])->extinction         = params->extinction;
        ((MxConductor*)bsdfs[num_bsdfs])->thinfilm_thickness = params->thinfilm_thickness;
        ((MxConductor*)bsdfs[num_bsdfs])->thinfilm_ior       = params->thinfilm_ior;
        ((MxConductor*)bsdfs[num_bsdfs])->calcTangentFrame();
        ((MxConductor*)bsdfs[num_bsdfs])->set_refraction_ior(1.0f);
        break;
    }
    case MX_GENERALIZED_SCHLICK_ID: {
        const MxGeneralizedSchlickParams* params = comp->as<MxGeneralizedSchlickParams>();
        bsdfs[num_bsdfs]                = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id            = MX_GENERALIZED_SCHLICK_ID;
        // MxMicrofacetBaseParams
        ((MxGeneralizedSchlick*)bsdfs[num_bsdfs])->N            = params->N;
        ((MxGeneralizedSchlick*)bsdfs[num_bsdfs])->U            = params->U;
        ((MxGeneralizedSchlick*)bsdfs[num_bsdfs])->roughness_x  = params->roughness_x;
        ((MxGeneralizedSchlick*)bsdfs[num_bsdfs])->roughness_y  = params->roughness_y;
        ((MxGeneralizedSchlick*)bsdfs[num_bsdfs])->distribution = params->distribution;
        ((MxGeneralizedSchlick*)bsdfs[num_bsdfs])->label        = params->label;
        // MxGeneralizedSchlickParams
        ((MxGeneralizedSchlick*)bsdfs[num_bsdfs])->reflection_tint    = params->reflection_tint;
        ((MxGeneralizedSchlick*)bsdfs[num_bsdfs])->transmission_tint  = params->transmission_tint;
        ((MxGeneralizedSchlick*)bsdfs[num_bsdfs])->f0                 = params->f0;
        ((MxGeneralizedSchlick*)bsdfs[num_bsdfs])->f90                = params->f90;
        ((MxGeneralizedSchlick*)bsdfs[num_bsdfs])->exponent           = params->exponent;
        ((MxGeneralizedSchlick*)bsdfs[num_bsdfs])->thinfilm_thickness = params->thinfilm_thickness;
        ((MxGeneralizedSchlick*)bsdfs[num_bsdfs])->thinfilm_ior       = params->thinfilm_ior;
        if (is_black(params->transmission_tint)) {
            ((MxGeneralizedSchlickOpaque*)bsdfs[num_bsdfs])->set_refraction_ior(1.0f);
        } else {
            ((MxGeneralizedSchlick*)bsdfs[num_bsdfs])->set_refraction_ior(result.refraction_ior);
        }
        ((MxGeneralizedSchlick*)bsdfs[num_bsdfs])->calcTangentFrame();
        break;
    }
    case MX_SHEEN_ID: {
        const MxSheenParams* params = comp->as<MxSheenParams>();
        bsdfs[num_bsdfs]                        = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                    = MX_SHEEN_ID;
        ((MxSheen*)bsdfs[num_bsdfs])->N         = params->N;
        ((MxSheen*)bsdfs[num_bsdfs])->albedo    = params->albedo;
        ((MxSheen*)bsdfs[num_bsdfs])->roughness = params->roughness;
        ((MxSheen*)bsdfs[num_bsdfs])->label     = params->label;
        break;
    }
    case MX_TRANSLUCENT_ID: {
        const MxTranslucentParams* params  = comp->as<MxTranslucentParams>();
        bsdfs[num_bsdfs]                   = (OSL::BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id               = DIFFUSE_ID;
        ((Diffuse<1>*)bsdfs[num_bsdfs])->N = params->N;
        // TODO: Gotta do something with albedo?
        weight *= params->albedo;
        break;
    }
    default: printf("add unknown: %s (%d), sz: %d\n", id_to_string(id), (int)id, num_bytes); break;
    }
    weights[num_bsdfs] = weight;
    num_bsdfs++;
    num_bytes += sz;
    return true;
}


OSL_HOSTDEVICE void
CompositeBSDF::prepare_gpu(const Vec3& wo, const Color3& path_weight,
                           bool absorb)
{
    float total = 0;
    for (int i = 0; i < num_bsdfs; i++) {
        pdfs[i] = weights[i].dot(path_weight * get_bsdf_albedo(bsdfs[i], wo))
                  / (path_weight.x + path_weight.y + path_weight.z);

        // TODO: What is an acceptable range?
        assert(pdfs[i] >= (0.0f - 1e-6f));
        assert(pdfs[i] <= (1.0f + 1e-6f));

        // Clamp the PDF to [0,1]. The PDF can fall outside of this range due to
        // floating-point precision issues.
        pdfs[i] = (pdfs[i] < 0.0f) ? 0.0f : (pdfs[i] > 1.0f) ? 1.0f : pdfs[i];

        total += pdfs[i];
    }
    if ((!absorb && total > 0) || total > 1) {
        for (int i = 0; i < num_bsdfs; i++)
            pdfs[i] = __fdiv_rn(pdfs[i], total);
    }
}


OSL_HOSTDEVICE BSDF::Sample
CompositeBSDF::eval_gpu(const Vec3& wo, const Vec3& wi) const
{
    BSDF::Sample s = {};
    for (int i = 0; i < num_bsdfs; i++) {
        BSDF::Sample b = eval_bsdf(bsdfs[i],wo, wi);
        b.weight *= weights[i];
        MIS::update_eval(&s.weight, &s.pdf, b.weight, b.pdf, pdfs[i]);
        s.roughness += b.roughness * pdfs[i];
    }
    return s;
}


OSL_HOSTDEVICE BSDF::Sample
CompositeBSDF::sample_gpu(const Vec3& wo, float rx, float ry, float rz) const
{
    float accum = 0;
    for (int i = 0; i < num_bsdfs; i++) {
        if (rx < (pdfs[i] + accum)) {
            rx             = (rx - accum) / pdfs[i];
            rx             = std::min(rx, 0.99999994f);  // keep result in [0,1)
            BSDF::Sample s = sample_bsdf(bsdfs[i], wo, rx, ry, rz);
            s.weight *= weights[i] * (1 / pdfs[i]);
            s.pdf *= pdfs[i];
            if (s.pdf == 0.0f)
                return {};

            // we sampled PDF i, now figure out how much the other bsdfs contribute to the chosen direction
            for (int j = 0; j < num_bsdfs; j++) {
                if (i != j) {
                    BSDF::Sample b = eval_bsdf(bsdfs[j], wo, s.wi);
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


OSL_HOSTDEVICE Color3
CompositeBSDF::get_albedo_gpu(const Vec3& wo) const
{
    Color3 result(0, 0, 0);
    for (int i = 0; i < num_bsdfs; i++) {
        result += weights[i] * get_bsdf_albedo(bsdfs[i], wo);
    }
    return result;
}


//
// Helper functions to avoid virtual function calls
//


OSL_HOSTDEVICE Color3
CompositeBSDF::get_bsdf_albedo(OSL::BSDF* bsdf, const Vec3& wo) const
{
    Color3 albedo(0);
    switch (bsdf->id) {
    case DIFFUSE_ID:
        albedo = ((Diffuse<0>*)bsdf)->Diffuse<0>::get_albedo(wo);
        break;
    case OREN_NAYAR_ID:
    case MX_OREN_NAYAR_DIFFUSE_ID:
        albedo = ((OrenNayar*)bsdf)->OrenNayar::get_albedo(wo);
        break;
    case PHONG_ID: albedo = ((Phong*)bsdf)->Phong::get_albedo(wo); break;
    case WARD_ID: albedo = ((Ward*)bsdf)->Ward::get_albedo(wo); break;
    case REFLECTION_ID:
    case FRESNEL_REFLECTION_ID:
        albedo = ((Reflection*)bsdf)->Reflection::get_albedo(wo);
        break;
    case REFRACTION_ID:
        albedo = ((Refraction*)bsdf)->Refraction::get_albedo(wo);
        break;
    case MICROFACET_ID: {
        const int refract = ((MicrofacetBeckmannRefl*)bsdf)->refract;
        switch (refract) {
        case 0:
            albedo = ((MicrofacetBeckmannRefl*)bsdf)
                         ->MicrofacetBeckmannRefl::get_albedo(wo);
            break;
        case 1:
            albedo = ((MicrofacetBeckmannRefr*)bsdf)
                         ->MicrofacetBeckmannRefr::get_albedo(wo);
            break;
        case 2:
            albedo = ((MicrofacetBeckmannBoth*)bsdf)
                         ->MicrofacetBeckmannBoth::get_albedo(wo);
            break;
        }
        break;
    }
    case MX_CONDUCTOR_ID:
        albedo = ((MxConductor*)bsdf)->MxConductor::get_albedo(wo);
        break;
    case MX_DIELECTRIC_ID:
        albedo = ((MxDielectricOpaque*)bsdf)->MxDielectricOpaque::get_albedo(wo);
        break;
    case MX_BURLEY_DIFFUSE_ID:
        albedo = ((MxBurleyDiffuse*)bsdf)->MxBurleyDiffuse::get_albedo(wo);
        break;
    case MX_SHEEN_ID:
        albedo = ((MxSheen*)bsdf)->MxSheen::get_albedo(wo);
        break;
    case MX_GENERALIZED_SCHLICK_ID:
        if (is_black(((MxGeneralizedSchlick*)bsdf)->transmission_tint))
            albedo = ((MxGeneralizedSchlickOpaque*)bsdf)->MxGeneralizedSchlickOpaque::get_albedo(wo);
        else
            albedo = ((MxGeneralizedSchlick*)bsdf)->MxGeneralizedSchlick::get_albedo(wo);
        break;
    default: break;
    }
    return albedo;
}


OSL_HOSTDEVICE BSDF::Sample
CompositeBSDF::sample_bsdf(OSL::BSDF* bsdf, const Vec3& wo, float rx, float ry,
                           float rz) const
{
    BSDF::Sample sample = {};
    switch (bsdf->id) {
    case DIFFUSE_ID:
        sample = ((Diffuse<0>*)bsdf)->sample(wo, rx, ry, rz);
        break;
    case OREN_NAYAR_ID:
    case MX_OREN_NAYAR_DIFFUSE_ID:
        sample = ((OrenNayar*)bsdf)->sample(wo, rx, ry, rz);
        break;
    case PHONG_ID: sample = ((Phong*)bsdf)->sample(wo, rx, ry, rz); break;
    case WARD_ID: sample = ((Ward*)bsdf)->sample(wo, rx, ry, rz); break;
    case REFLECTION_ID:
    case FRESNEL_REFLECTION_ID:
        sample = ((Reflection*)bsdf)->sample(wo, rx, ry, rz);
        break;
    case REFRACTION_ID:
        sample = ((Refraction*)bsdf)->sample(wo, rx, ry, rz);
        break;
    case MICROFACET_ID: {
        const int refract = ((MicrofacetBeckmannRefl*)bsdf)->refract;
        switch (refract) {
        case 0:
            sample = ((MicrofacetBeckmannRefl*)bsdf)
                         ->MicrofacetBeckmannRefl::sample(wo, rx, ry, rz);
            break;
        case 1:
            sample = ((MicrofacetBeckmannRefr*)bsdf)
                         ->MicrofacetBeckmannRefr::sample(wo, rx, ry, rz);
            break;
        case 2:
            sample = ((MicrofacetBeckmannBoth*)bsdf)
                         ->MicrofacetBeckmannBoth::sample(wo, rx, ry, rz);
            break;
        }
        break;
    }
    case MX_CONDUCTOR_ID: sample = ((MxConductor*)bsdf)->sample(wo, rx, ry, rz); break;
    case MX_DIELECTRIC_ID: sample = ((MxDielectricOpaque*)bsdf)->sample(wo, rx, ry, rz); break;
    case MX_BURLEY_DIFFUSE_ID: sample = ((MxBurleyDiffuse*)bsdf)->sample(wo, rx, ry, rz); break;
    case MX_SHEEN_ID: sample = ((MxSheen*)bsdf)->sample(wo, rx, ry, rz); break;
    case MX_GENERALIZED_SCHLICK_ID:
        if (is_black(((MxGeneralizedSchlick*)bsdf)->transmission_tint))
            sample = ((MxGeneralizedSchlickOpaque*)bsdf)->sample(wo, rx, ry, rz);
        else
            sample = ((MxGeneralizedSchlick*)bsdf)->sample(wo, rx, ry, rz);
        break;
    default: break;
    }
    if (sample.pdf != sample.pdf)
    {
        uint3 launch_index = optixGetLaunchIndex();
        printf("sample_bsdf( %s ), PDF is NaN [%d, %d]\n",
               id_to_string(bsdf->id), launch_index.x, launch_index.y);
    }
    return sample;
}


OSL_HOSTDEVICE BSDF::Sample
CompositeBSDF::eval_bsdf(OSL::BSDF* bsdf, const Vec3& wo, const Vec3& wi) const
{
    BSDF::Sample sample = {};
    switch (bsdf->id) {
    case DIFFUSE_ID: sample = ((Diffuse<0>*)bsdf)->eval(wo, wi); break;
    case MX_OREN_NAYAR_DIFFUSE_ID:
    case OREN_NAYAR_ID: sample = ((OrenNayar*)bsdf)->eval(wo, wi); break;
    case PHONG_ID: sample = ((Phong*)bsdf)->eval(wo, wi); break;
    case WARD_ID: sample = ((Ward*)bsdf)->eval(wo, wi); break;
    case REFLECTION_ID:
    case FRESNEL_REFLECTION_ID:
        sample = ((Reflection*)bsdf)->eval(wo, wi);
        break;
    case REFRACTION_ID: sample = ((Refraction*)bsdf)->eval(wo, wi); break;
    case MICROFACET_ID: {
        const int refract = ((MicrofacetBeckmannRefl*)bsdf)->refract;
        switch (refract) {
        case 0:
            sample = ((MicrofacetBeckmannRefl*)bsdf)
                         ->MicrofacetBeckmannRefl::eval(wo, wi);
            break;
        case 1:
            sample = ((MicrofacetBeckmannRefr*)bsdf)
                         ->MicrofacetBeckmannRefr::eval(wo, wi);
            break;
        case 2:
            sample = ((MicrofacetBeckmannBoth*)bsdf)
                         ->MicrofacetBeckmannBoth::eval(wo, wi);
            break;
        }
        break;
    }
    case MX_CONDUCTOR_ID: sample = ((MxConductor*)bsdf)->eval(wo, wi); break;
    case MX_DIELECTRIC_ID: sample = ((MxDielectricOpaque*)bsdf)->eval(wo, wi); break;
    case MX_BURLEY_DIFFUSE_ID: sample = ((MxBurleyDiffuse*)bsdf)->eval(wo, wi); break;
    case MX_SHEEN_ID: sample = ((MxSheen*)bsdf)->eval(wo, wi); break;
    case MX_GENERALIZED_SCHLICK_ID:
        if (is_black(((MxGeneralizedSchlick*)bsdf)->transmission_tint))
            sample = ((MxGeneralizedSchlickOpaque*)bsdf)->eval(wo, wi);
        else
            sample = ((MxGeneralizedSchlick*)bsdf)->eval(wo, wi);
        break;
    default: break;
    }
    if (sample.pdf != sample.pdf)
    {
        uint3 launch_index = optixGetLaunchIndex();
        printf("eval_bsdf( %s ), PDF is NaN [%d, %d]\n",
               id_to_string(bsdf->id), launch_index.x, launch_index.y);
    }
    return sample;
}

OSL_NAMESPACE_EXIT
