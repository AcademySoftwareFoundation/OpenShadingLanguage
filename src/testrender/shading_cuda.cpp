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
static OSL_HOSTDEVICE const char* id_to_string(int id)
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


OSL_NAMESPACE_ENTER


typedef MxMicrofacet<MxConductorParams, GGXDist, false> MxConductor;
typedef MxMicrofacet<MxDielectricParams, GGXDist, true> MxDielectric;
typedef MxMicrofacet<MxDielectricParams, GGXDist, false> MxDielectricOpaque;
typedef MxMicrofacet<MxGeneralizedSchlickParams, GGXDist, true> MxGeneralizedSchlick;
typedef MxMicrofacet<MxGeneralizedSchlickParams, GGXDist, false> MxGeneralizedSchlickOpaque;


OSL_HOSTDEVICE bool
CompositeBSDF::add_bsdf_gpu(const Color3& w, const ClosureComponent* comp,
                            ShadingResult& result)
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
        bsdfs[num_bsdfs]                   = (BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id               = DIFFUSE_ID;
        ((Diffuse<0>*)bsdfs[num_bsdfs])->N = params->N;
        break;
    }
    case OREN_NAYAR_ID: {
        const OrenNayarParams* params = comp->as<OrenNayarParams>();
        bsdfs[num_bsdfs]                      = (BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                  = OREN_NAYAR_ID;
        ((OrenNayar*)bsdfs[num_bsdfs])->N     = params->N;
        ((OrenNayar*)bsdfs[num_bsdfs])->sigma = params->sigma;
        ((OrenNayar*)bsdfs[num_bsdfs])->calcAB();
        break;
    }
    case TRANSLUCENT_ID: {
        const DiffuseParams* params = comp->as<DiffuseParams>();
        bsdfs[num_bsdfs]                   = (BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id               = DIFFUSE_ID;
        ((Diffuse<1>*)bsdfs[num_bsdfs])->N = params->N;
        break;
    }
    case PHONG_ID: {
        const PhongParams* params            = comp->as<PhongParams>();
        bsdfs[num_bsdfs]                     = (BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                 = PHONG_ID;
        ((Phong*)bsdfs[num_bsdfs])->N        = params->N;
        ((Phong*)bsdfs[num_bsdfs])->exponent = params->exponent;
        break;
    }
    case WARD_ID: {
        const WardParams* params      = comp->as<WardParams>();
        bsdfs[num_bsdfs]              = (BSDF*)(pool + num_bytes);
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
        bsdfs[num_bsdfs]                     = (BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                 = REFLECTION_ID;
        ((Reflection*)bsdfs[num_bsdfs])->N   = params->N;
        ((Reflection*)bsdfs[num_bsdfs])->eta = params->eta;
        break;
    }
    case REFRACTION_ID: {
        const RefractionParams* params       = comp->as<RefractionParams>();
        bsdfs[num_bsdfs]                     = (BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                 = REFRACTION_ID;
        ((Refraction*)bsdfs[num_bsdfs])->N   = params->N;
        ((Refraction*)bsdfs[num_bsdfs])->eta = params->eta;
        break;
    }
    case TRANSPARENT_ID: {
        bsdfs[num_bsdfs]     = (BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id = TRANSPARENT_ID;
        break;
    }
    case MICROFACET_ID: {
        const MicrofacetParams* params        = comp->as<MicrofacetParams>();
        bsdfs[num_bsdfs]                      = (BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                  = MICROFACET_ID;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->dist    = params->dist;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->N       = params->N;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->U       = params->U;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->xalpha  = params->xalpha;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->yalpha  = params->yalpha;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->eta     = params->eta;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->refract = params->refract;
        ((MicrofacetBeckmannRefl*)bsdfs[num_bsdfs])->calcTangentFrame();
        break;
    }
    case MX_OREN_NAYAR_DIFFUSE_ID: {
        const MxOrenNayarDiffuseParams* params = comp->as<MxOrenNayarDiffuseParams>();
        bsdfs[num_bsdfs]                      = (BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                  = OREN_NAYAR_ID;
        ((OrenNayar*)bsdfs[num_bsdfs])->N     = params->N;
        ((OrenNayar*)bsdfs[num_bsdfs])->sigma = params->roughness;
        ((OrenNayar*)bsdfs[num_bsdfs])->calcAB();
        weight *= params->albedo;
        break;
    }
    case MX_BURLEY_DIFFUSE_ID: {
        const MxBurleyDiffuseParams* params = comp->as<MxBurleyDiffuseParams>();
        bsdfs[num_bsdfs]                    = (BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                = MX_BURLEY_DIFFUSE_ID;
        ((MxBurleyDiffuse*)bsdfs[num_bsdfs])->N         = params->N;
        ((MxBurleyDiffuse*)bsdfs[num_bsdfs])->albedo    = params->albedo;
        ((MxBurleyDiffuse*)bsdfs[num_bsdfs])->roughness = params->roughness;
        ((MxBurleyDiffuse*)bsdfs[num_bsdfs])->label     = params->label;
        break;
    }
    case MX_DIELECTRIC_ID: {
        const MxDielectricParams* params = comp->as<MxDielectricParams>();
        bsdfs[num_bsdfs]                = (BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id            = MX_OREN_NAYAR_DIFFUSE_ID; // MX_DIELECTRIC_ID;
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
        bsdfs[num_bsdfs]                = (BSDF*)(pool + num_bytes);
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
        bsdfs[num_bsdfs]                = (BSDF*)(pool + num_bytes);
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
        bsdfs[num_bsdfs]                        = (BSDF*)(pool + num_bytes);
        bsdfs[num_bsdfs]->id                    = MX_SHEEN_ID;
        ((MxSheen*)bsdfs[num_bsdfs])->N         = params->N;
        ((MxSheen*)bsdfs[num_bsdfs])->albedo    = params->albedo;
        ((MxSheen*)bsdfs[num_bsdfs])->roughness = params->roughness;
        ((MxSheen*)bsdfs[num_bsdfs])->label     = params->label;
        break;
    }
    case MX_TRANSLUCENT_ID: {
        const MxTranslucentParams* params  = comp->as<MxTranslucentParams>();
        bsdfs[num_bsdfs]                   = (BSDF*)(pool + num_bytes);
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
CompositeBSDF::get_bsdf_albedo(BSDF* bsdf, const Vec3& wo) const
{
    Color3 albedo(0);
    switch (bsdf->id) {
    case DIFFUSE_ID:
        albedo = ((Diffuse<0>*)bsdf)->Diffuse<0>::get_albedo(wo);
        break;
    case OREN_NAYAR_ID:
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
        const char* dist  = ((MicrofacetBeckmannRefl*)bsdf)->dist;
        if (HDSTR(dist) == STRING_PARAMS(default)
            || HDSTR(dist) == STRING_PARAMS(beckmann)) {
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
        } else if (HDSTR(dist) == STRING_PARAMS(ggx)) {
            switch (refract) {
            case 0:
                albedo = ((MicrofacetGGXRefl*)bsdf)
                             ->MicrofacetGGXRefl::get_albedo(wo);
                break;
            case 1:
                albedo = ((MicrofacetGGXRefr*)bsdf)
                             ->MicrofacetGGXRefr::get_albedo(wo);
                break;
            case 2:
                albedo = ((MicrofacetGGXBoth*)bsdf)
                             ->MicrofacetGGXBoth::get_albedo(wo);
                break;
            }
        }
        break;
    }
    case MX_CONDUCTOR_ID:
        albedo = ((MxConductor*)bsdf)->MxConductor::get_albedo(wo);
        break;
    case MX_DIELECTRIC_ID:
        albedo = ((MxDielectricOpaque*)bsdf)->MxDielectricOpaque::get_albedo(wo);
        break;
    case MX_OREN_NAYAR_DIFFUSE_ID:
        albedo = ((MxDielectric*)bsdf)->MxDielectric::get_albedo(wo);
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
CompositeBSDF::sample_bsdf(BSDF* bsdf, const Vec3& wo, float rx, float ry,
                           float rz) const
{
    BSDF::Sample sample = {};
    switch (bsdf->id) {
    case DIFFUSE_ID:
        sample = ((Diffuse<0>*)bsdf)->sample(wo, rx, ry, rz);
        break;
    case OREN_NAYAR_ID:
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
        const char* dist  = ((MicrofacetBeckmannRefl*)bsdf)->dist;
        if (HDSTR(dist) == STRING_PARAMS(default)
            || HDSTR(dist) == STRING_PARAMS(beckmann)) {
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
        } else if (HDSTR(dist) == STRING_PARAMS(ggx)) {
            switch (refract) {
            case 0:
                sample = ((MicrofacetGGXRefl*)bsdf)
                             ->MicrofacetGGXRefl::sample(wo, rx, ry, rz);
                break;
            case 1:
                sample = ((MicrofacetGGXRefr*)bsdf)
                             ->MicrofacetGGXRefr::sample(wo, rx, ry, rz);
                break;
            case 2:
                sample = ((MicrofacetGGXBoth*)bsdf)
                             ->MicrofacetGGXBoth::sample(wo, rx, ry, rz);
                break;
            }
        }
        break;
    }
    case MX_CONDUCTOR_ID: sample = ((MxConductor*)bsdf)->sample(wo, rx, ry, rz); break;
    case MX_DIELECTRIC_ID: sample = ((MxDielectricOpaque*)bsdf)->sample(wo, rx, ry, rz); break;
    case MX_BURLEY_DIFFUSE_ID: sample = ((MxBurleyDiffuse*)bsdf)->sample(wo, rx, ry, rz); break;
    case MX_OREN_NAYAR_DIFFUSE_ID: sample = ((MxDielectric*)bsdf)->sample(wo, rx, ry, rz); break;
    case MX_SHEEN_ID: sample = ((MxSheen*)bsdf)->sample(wo, rx, ry, rz); break;
    case MX_GENERALIZED_SCHLICK_ID:
        if (is_black(((MxGeneralizedSchlick*)bsdf)->transmission_tint))
            sample = ((MxGeneralizedSchlickOpaque*)bsdf)->sample(wo, rx, ry, rz);
        else
            sample = ((MxGeneralizedSchlick*)bsdf)->sample(wo, rx, ry, rz);
        break;
    default: break;
    }
        if (sample.pdf != sample.pdf) {
            uint3 launch_index = optixGetLaunchIndex();
            printf("sample_bsdf( %s ), PDF is NaN [%d, %d]\n",
                   id_to_string(bsdf->id), launch_index.x, launch_index.y);
        }
        return sample;
}


OSL_HOSTDEVICE BSDF::Sample
CompositeBSDF::eval_bsdf(BSDF* bsdf, const Vec3& wo, const Vec3& wi) const
{
    BSDF::Sample sample = {};
    switch (bsdf->id) {
    case DIFFUSE_ID: sample = ((Diffuse<0>*)bsdf)->eval(wo, wi); break;
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
        const char* dist  = ((MicrofacetBeckmannRefl*)bsdf)->dist;
        if (HDSTR(dist) == STRING_PARAMS(default)
            || HDSTR(dist) == STRING_PARAMS(beckmann)) {
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
        } else if (HDSTR(dist) == STRING_PARAMS(ggx)) {
            switch (refract) {
            case 0:
                sample = ((MicrofacetGGXRefl*)bsdf)
                             ->MicrofacetGGXRefl::eval(wo, wi);
                break;
            case 1:
                sample = ((MicrofacetGGXRefr*)bsdf)
                             ->MicrofacetGGXRefr::eval(wo, wi);
                break;
            case 2:
                sample = ((MicrofacetGGXBoth*)bsdf)
                             ->MicrofacetGGXBoth::eval(wo, wi);
                break;
            }
        }
        break;
    }
    case MX_CONDUCTOR_ID: sample = ((MxConductor*)bsdf)->eval(wo, wi); break;
    case MX_DIELECTRIC_ID: sample = ((MxDielectricOpaque*)bsdf)->eval(wo, wi); break;
    case MX_BURLEY_DIFFUSE_ID: sample = ((MxBurleyDiffuse*)bsdf)->eval(wo, wi); break;
    case MX_OREN_NAYAR_DIFFUSE_ID: sample = ((MxDielectric*)bsdf)->eval(wo, wi); break;
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


static __device__ Color3
evaluate_layer_opacity(const ShaderGlobalsType& sg, const ClosureColor* closure)
{
    // Null closure, the layer is fully transparent
    if (closure == nullptr)
        return Color3(0);

    // The depth of the closure tree must not exceed the stack size.
    // A stack size of 8 is probably quite generous for relatively
    // balanced trees.
    const int STACK_SIZE = 8;

    // Non-recursive traversal stack
    int stack_idx = 0;
    const ClosureColor* ptr_stack[STACK_SIZE];
    Color3 weight_stack[STACK_SIZE];

    // Shading accumulator
    Color3 weight = Color3(1.0f);

    while (closure) {
        switch (closure->id) {
        case ClosureIDs::MUL: {
            weight *= ((ClosureMul*)closure)->weight;
            closure = ((ClosureMul*)closure)->closure;
            break;
        }
        case ClosureIDs::ADD: {
            ptr_stack[stack_idx]      = ((ClosureAdd*)closure)->closureB;
            weight_stack[stack_idx++] = weight;
            closure                   = ((ClosureAdd*)closure)->closureA;
            break;
        }
        default: {
            const ClosureComponent* comp = closure->as_comp();
            Color3 w                     = comp->w;
            switch (comp->id) {
            case MX_LAYER_ID: {
                const MxLayerParams* srcparams = comp->as<MxLayerParams>();
                closure                   = srcparams->top;
                ptr_stack[stack_idx]      = srcparams->base;
                weight_stack[stack_idx++] = weight * w;
                break;
            }
            case REFLECTION_ID:
            case FRESNEL_REFLECTION_ID: {
                Reflection bsdf(*comp->as<ReflectionParams>());
                const Vec3& I = *reinterpret_cast<const Vec3*>(&sg.I);
                weight *= w * bsdf.get_albedo(-I);
                closure = nullptr;
                break;
            }
            case MX_DIELECTRIC_ID: {
                const MxDielectricParams& params = *comp->as<MxDielectricParams>();
                // Transmissive dielectrics are opaque
                if (!is_black(params.transmission_tint))
                    return Color3(1);
                MxMicrofacet<MxDielectricParams, GGXDist, false> mf(params, 1.0f);
                const Vec3& I = *reinterpret_cast<const Vec3*>(&sg.I);
                weight *= w * mf.get_albedo(-I);
                closure = nullptr;
                break;
            }
            case MX_GENERALIZED_SCHLICK_ID: {
                const MxGeneralizedSchlickParams& params
                    = *comp->as<MxGeneralizedSchlickParams>();
                // Transmissive dielectrics are opaque
                if (!is_black(params.transmission_tint))
                    return Color3(1);
                MxMicrofacet<MxGeneralizedSchlickParams, GGXDist, false> mf(params,
                                                                            1.0f);
                const Vec3& I = *reinterpret_cast<const Vec3*>(&sg.I);
                weight *= w * mf.get_albedo(-I);
                closure = nullptr;
            }
            case MX_SHEEN_ID: {
                MxSheen bsdf(*comp->as<MxSheenParams>());
                const Vec3& I = *reinterpret_cast<const Vec3*>(&sg.I);
                weight *= w * bsdf.get_albedo(-I);
                closure = nullptr;
                break;
            }
            default:  // Assume unhandled BSDFs are opaque
                closure = nullptr;
                break;
            }
        }
        }
        if (closure == nullptr && stack_idx > 0) {
            closure = ptr_stack[--stack_idx];
            weight  = weight_stack[stack_idx];
        }
    }
    // We should never reach this point
    assert(false);
    Color3(0);
}


static __device__ Color3
process_medium_closure(const ShaderGlobalsType& sg, ShadingResult& result,
                       const ClosureColor* closure, const Color3& w)
{
    Color3 color_result = Color3(0.0f);
    if (!closure) {
        return color_result;
    }

    // The depth of the closure tree must not exceed the stack size.
    // A stack size of 8 is probably quite generous for relatively
    // balanced trees.
    const int STACK_SIZE = 8;

    // Non-recursive traversal stack
    int stack_idx = 0;
    const ClosureColor* ptr_stack[STACK_SIZE];
    Color3 weight_stack[STACK_SIZE];

    // Shading accumulator
    Color3 weight = w; // Color3(1.0f);
    while (closure) {
        ClosureIDs id = static_cast<ClosureIDs>(closure->id);
        switch (id) {
        case ClosureIDs::ADD: {
            ptr_stack[stack_idx]      = ((ClosureAdd*)closure)->closureB;
            weight_stack[stack_idx++] = weight;
            closure                   = ((ClosureAdd*)closure)->closureA;
            break;
        }
        case ClosureIDs::MUL: {
            weight *= ((ClosureMul*)closure)->weight;
            closure = ((ClosureMul*)closure)->closure;
            break;
        }
        case MX_LAYER_ID: {
            const ClosureComponent* comp = closure->as_comp();
            const MxLayerParams* params  = comp->as<MxLayerParams>();
            Color3 base_w
                = w
                  * (Color3(1)
                     - clamp(evaluate_layer_opacity(sg, params->top), 0.f, 1.f));
            closure                   = params->top;
            ptr_stack[stack_idx]      = params->base;
            weight_stack[stack_idx++] = weight * w;
            break;
        }
        case MX_ANISOTROPIC_VDF_ID: {
            const ClosureComponent* comp = closure->as_comp();
            Color3 cw                    = w * comp->w;
            const auto& params           = *comp->as<MxAnisotropicVdfParams>();
            result.sigma_t               = cw * params.extinction;
            result.sigma_s               = params.albedo * result.sigma_t;
            result.medium_g              = params.anisotropy;
            result.refraction_ior        = 1.0f;
            result.priority = 0;  // TODO: should this closure have a priority?
            closure = nullptr;
            break;
        }
        case MX_MEDIUM_VDF_ID: {
            const ClosureComponent* comp = closure->as_comp();
            Color3 cw                    = w * comp->w;
            const auto& params           = *comp->as<MxMediumVdfParams>();
            result.sigma_t = { -OIIO::fast_log(params.transmission_color.x),
                               -OIIO::fast_log(params.transmission_color.y),
                               -OIIO::fast_log(params.transmission_color.z) };
            // NOTE: closure weight scales the extinction parameter
            result.sigma_t *= cw / params.transmission_depth;
            result.sigma_s  = params.albedo * result.sigma_t;
            result.medium_g = params.anisotropy;
            // TODO: properly track a medium stack here ...
            result.refraction_ior = sg.backfacing ? 1.0f / params.ior : params.ior;
            result.priority       = params.priority;
            closure = nullptr;
            break;
        }
        case MX_DIELECTRIC_ID: {
            const ClosureComponent* comp = closure->as_comp();
            const auto& params           = *comp->as<MxDielectricParams>();
            if (!is_black(w * comp->w * params.transmission_tint)) {
                // TODO: properly track a medium stack here ...
                result.refraction_ior = sg.backfacing ? 1.0f / params.ior
                                                      : params.ior;
            }
            closure = nullptr;
            break;
        }
        case MX_GENERALIZED_SCHLICK_ID: {
            const ClosureComponent* comp = closure->as_comp();
            const auto& params           = *comp->as<MxGeneralizedSchlickParams>();
            if (!is_black(w * comp->w * params.transmission_tint)) {
                // TODO: properly track a medium stack here ...
                float avg_F0  = clamp((params.f0.x + params.f0.y + params.f0.z)
                                          / 3.0f,
                                      0.0f, 0.99f);
                float sqrt_F0 = sqrtf(avg_F0);
                float ior     = (1 + sqrt_F0) / (1 - sqrt_F0);
                result.refraction_ior = sg.backfacing ? 1.0f / ior : ior;
            }
            closure = nullptr;
            break;
        }
        default:
            closure = nullptr;
            break;
        }
        if (closure == nullptr && stack_idx > 0) {
            closure = ptr_stack[--stack_idx];
            weight  = weight_stack[stack_idx];
        }
    }
    return weight;
}


static __device__ void
process_closure(const ShaderGlobalsType& sg, const ClosureColor* closure, ShadingResult& result,
                bool light_only)
{
    if (!closure) {
        return;
    }
    // The depth of the closure tree must not exceed the stack size.
    // A stack size of 8 is probably quite generous for relatively
    // balanced trees.
    const int STACK_SIZE = 8;

    // Non-recursive traversal stack
    int stack_idx = 0;
    const ClosureColor* ptr_stack[STACK_SIZE];
    Color3 weight_stack[STACK_SIZE];

    // Shading accumulator
    Color3 weight = Color3(1.0f);
    while (closure) {
        ClosureIDs id = static_cast<ClosureIDs>(closure->id);
        switch (id) {
        case ClosureIDs::ADD: {
            ptr_stack[stack_idx]      = ((ClosureAdd*)closure)->closureB;
            weight_stack[stack_idx++] = weight;
            closure                   = ((ClosureAdd*)closure)->closureA;
            break;
        }
        case ClosureIDs::MUL: {
            weight *= ((ClosureMul*)closure)->weight;
            closure = ((ClosureMul*)closure)->closure;
            break;
        }
        default: {
            const ClosureComponent* comp = closure->as_comp();
            Color3 cw                    = weight * comp->w;
            switch (id) {
            case ClosureIDs::EMISSION_ID: {
                result.Le += cw;
                closure = nullptr;
                break;
            }
            case ClosureIDs::MICROFACET_ID:
            case ClosureIDs::DIFFUSE_ID:
            case ClosureIDs::OREN_NAYAR_ID:
            case ClosureIDs::PHONG_ID:
            case ClosureIDs::WARD_ID:
            case ClosureIDs::REFLECTION_ID:
            case ClosureIDs::FRESNEL_REFLECTION_ID:
            case ClosureIDs::REFRACTION_ID:
            case ClosureIDs::MX_CONDUCTOR_ID:
            case ClosureIDs::MX_DIELECTRIC_ID:
            case ClosureIDs::MX_BURLEY_DIFFUSE_ID:
            case ClosureIDs::MX_OREN_NAYAR_DIFFUSE_ID:
            case ClosureIDs::MX_SHEEN_ID:
            case ClosureIDs::MX_GENERALIZED_SCHLICK_ID: {
                if (!result.bsdf.add_bsdf_gpu(cw, comp, result))
                    printf("unable to add BSDF\n");
                closure = nullptr;
                break;
            }
            case MX_LAYER_ID: {
                // TODO: The weight handling here is questionable ...
                const MxLayerParams* srcparams = comp->as<MxLayerParams>();
                Color3 base_w
                    = cw
                      * (Color3(1, 1, 1)
                         - clamp(evaluate_layer_opacity(sg, srcparams->top),
                                 0.f, 1.f));
                closure = srcparams->top;
                weight  = cw;
                if (!is_black(base_w)) {
                    ptr_stack[stack_idx]      = srcparams->base;
                    weight_stack[stack_idx++] = base_w;
                }
                break;
            }
            case ClosureIDs::MX_ANISOTROPIC_VDF_ID:
            case ClosureIDs::MX_MEDIUM_VDF_ID: {
                closure = nullptr;
                break;
            }
            default:
                printf("unhandled ID? %s (%d)\n", id_to_string(id), int(id));
                closure = nullptr;
                break;
            }
        }
        }
        if (closure == nullptr && stack_idx > 0) {
            closure = ptr_stack[--stack_idx];
            weight  = weight_stack[stack_idx];
        }
    }
}


static __device__ void
process_closure(const ShaderGlobalsType& sg, ShadingResult& result,
                const void* Ci, bool light_only)
{
    if (!light_only) {
        process_medium_closure(sg, result, (const ClosureColor*) Ci, Color3(1));
    }
    process_closure(sg, (const ClosureColor*)Ci, result, light_only);
}


static __device__ Color3
process_background_closure(const ShaderGlobalsType& sg, const ClosureColor* closure)
{
    if (!closure) {
        return Color3(0);
    }

    // The depth of the closure tree must not exceed the stack size.
    // A stack size of 8 is probably quite generous for relatively
    // balanced trees.
    const int STACK_SIZE = 8;

    // Non-recursive traversal stack
    int stack_idx = 0;
    const ClosureColor* ptr_stack[STACK_SIZE];
    Color3 weight_stack[STACK_SIZE];

    // Shading accumulator
    Color3 weight = Color3(1.0f);
    while (closure) {
        ClosureIDs id = static_cast<ClosureIDs>(closure->id);
        switch (id) {
        case ClosureIDs::ADD: {
            ptr_stack[stack_idx]      = ((ClosureAdd*)closure)->closureB;
            weight_stack[stack_idx++] = weight;
            closure                   = ((ClosureAdd*)closure)->closureA;
            break;
        }
        case ClosureIDs::MUL: {
            weight *= ((ClosureMul*)closure)->weight;
            closure = ((ClosureMul*)closure)->closure;
            break;
        }
        case ClosureIDs::BACKGROUND_ID: {
            const ClosureComponent* comp = closure->as_comp();
            weight *= comp->w;
            closure = nullptr;
            break;
        }
        default:
            // Should never get here
            assert(false);
        }
        if (closure == nullptr && stack_idx > 0) {
            closure = ptr_stack[--stack_idx];
            weight  = weight_stack[stack_idx];
        }
    }
    return weight;
}


OSL_NAMESPACE_EXIT
