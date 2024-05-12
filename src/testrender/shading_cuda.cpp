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
CompositeBSDF::add_bsdf_cuda(const Color3& w, const ClosureComponent* comp,
                             ShadingResult& result)
{
    ClosureIDs id = static_cast<ClosureIDs>(comp->id);
    size_t sz     = 0;
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
    case MX_GENERALIZED_SCHLICK_ID: sz = sizeof(MxGeneralizedSchlick); break;
    case MX_TRANSLUCENT_ID: sz = sizeof(Diffuse<1>); break;
    case MX_TRANSPARENT_ID: sz = sizeof(Transparent); break;
    case MX_SUBSURFACE_ID: sz = sizeof(Diffuse<0>); break;
    case MX_SHEEN_ID: sz = sizeof(MxSheen); break;
    default: assert(false); break;
    }

    if (num_bsdfs >= MaxEntries)
        return false;
    if (num_bytes + sz > MaxSize)
        return false;

    Color3 weight = w;

    switch (id) {
    case DIFFUSE_ID: {
        const DiffuseParams& params = *comp->as<DiffuseParams>();
        Diffuse<0>* bsdf            = new (pool + num_bytes) Diffuse<0>(params);
        bsdf->id                    = DIFFUSE_ID;
        bsdfs[num_bsdfs]            = (BSDF*)bsdf;
        break;
    }
    case OREN_NAYAR_ID: {
        const OrenNayarParams& params = *comp->as<OrenNayarParams>();
        OrenNayar* bsdf  = new (pool + num_bytes) OrenNayar(params);
        bsdf->id         = OREN_NAYAR_ID;
        bsdfs[num_bsdfs] = bsdf;
        break;
    }
    case TRANSLUCENT_ID: {
        const DiffuseParams& params = *comp->as<DiffuseParams>();
        Diffuse<1>* bsdf            = new (pool + num_bytes) Diffuse<1>(params);
        bsdf->id                    = TRANSLUCENT_ID;
        bsdfs[num_bsdfs]            = bsdf;
        break;
    }
    case PHONG_ID: {
        const PhongParams& params = *comp->as<PhongParams>();
        Phong* bsdf               = new (pool + num_bytes) Phong(params);
        bsdf->id                  = PHONG_ID;
        bsdfs[num_bsdfs]          = bsdf;
        break;
    }
    case WARD_ID: {
        const WardParams& params = *comp->as<WardParams>();
        Ward* bsdf               = new (pool + num_bytes) Ward(params);
        bsdf->id                 = WARD_ID;
        bsdfs[num_bsdfs]         = bsdf;
        break;
    }
    case REFLECTION_ID:
    case FRESNEL_REFLECTION_ID: {
        const ReflectionParams& params = *comp->as<ReflectionParams>();
        Reflection* bsdf = new (pool + num_bytes) Reflection(params);
        bsdf->id         = REFLECTION_ID;
        bsdfs[num_bsdfs] = bsdf;
        break;
    }
    case REFRACTION_ID: {
        const RefractionParams& params = *comp->as<RefractionParams>();
        Refraction* bsdf = new (pool + num_bytes) Refraction(params);
        bsdf->id         = REFRACTION_ID;
        bsdfs[num_bsdfs] = bsdf;
        break;
    }
    case TRANSPARENT_ID:
    case MX_TRANSPARENT_ID: {
        bsdfs[num_bsdfs]     = new (pool + num_bytes) BSDF(TRANSPARENT_ID);
        bsdfs[num_bsdfs]->id = TRANSPARENT_ID;
        break;
    }
    case MICROFACET_ID: {
        const MicrofacetParams& params = *comp->as<MicrofacetParams>();
        MicrofacetBeckmannRefl* bsdf   = new (pool + num_bytes)
            MicrofacetBeckmannRefl(params);
        bsdf->id         = MICROFACET_ID;
        bsdfs[num_bsdfs] = bsdf;
        break;
    }
    case MX_OREN_NAYAR_DIFFUSE_ID: {
        const MxOrenNayarDiffuseParams* src_params
            = comp->as<MxOrenNayarDiffuseParams>();
        const OrenNayarParams params = { src_params->N, src_params->roughness };
        OrenNayar* bsdf              = new (pool + num_bytes) OrenNayar(params);
        bsdf->id                     = OREN_NAYAR_ID;
        bsdfs[num_bsdfs]             = bsdf;
        weight *= src_params->albedo;
        break;
    }
    case MX_BURLEY_DIFFUSE_ID: {
        const MxBurleyDiffuseParams& params = *comp->as<MxBurleyDiffuseParams>();
        MxBurleyDiffuse* bsdf = new (pool + num_bytes) MxBurleyDiffuse(params);
        bsdf->id              = MX_BURLEY_DIFFUSE_ID;
        bsdfs[num_bsdfs]      = bsdf;
        break;
    }
    case MX_DIELECTRIC_ID: {
        const MxDielectricParams& params = *comp->as<MxDielectricParams>();
        if (is_black(params.transmission_tint)) {
            MxDielectricOpaque* bsdf = new (pool + num_bytes)
                MxDielectricOpaque(params, 1.0f);
            bsdfs[num_bsdfs] = bsdf;
        } else {
            MxDielectric* bsdf = new (pool + num_bytes)
                MxDielectric(params, result.refraction_ior);
            bsdfs[num_bsdfs] = bsdf;
        }
        bsdfs[num_bsdfs]->id = MX_DIELECTRIC_ID;
        break;
    }
    case MX_CONDUCTOR_ID: {
        const MxConductorParams& params = *comp->as<MxConductorParams>();
        MxConductor* bsdf = new (pool + num_bytes) MxConductor(params, 1.0f);
        bsdfs[num_bsdfs]  = bsdf;
        bsdf->id          = MX_CONDUCTOR_ID;
        break;
    }
    case MX_GENERALIZED_SCHLICK_ID: {
        const MxGeneralizedSchlickParams& params
            = *comp->as<MxGeneralizedSchlickParams>();
        if (is_black(params.transmission_tint)) {
            MxGeneralizedSchlickOpaque* bsdf = new (pool + num_bytes)
                MxGeneralizedSchlickOpaque(params, 1.0f);
            bsdfs[num_bsdfs] = bsdf;
        } else {
            MxGeneralizedSchlick* bsdf = new (pool + num_bytes)
                MxGeneralizedSchlick(params, result.refraction_ior);
            bsdfs[num_bsdfs] = bsdf;
        }
        bsdfs[num_bsdfs]->id = MX_GENERALIZED_SCHLICK_ID;
        break;
    }
    case MX_SHEEN_ID: {
        const MxSheenParams& params = *comp->as<MxSheenParams>();
        MxSheen* bsdf               = new (pool + num_bytes) MxSheen(params);
        bsdf->id                    = MX_SHEEN_ID;
        bsdfs[num_bsdfs]            = bsdf;
        break;
    }
    case MX_TRANSLUCENT_ID: {
        const MxTranslucentParams* src_params = comp->as<MxTranslucentParams>();
        const DiffuseParams params            = { src_params->N };
        Diffuse<1>* bsdf = new (pool + num_bytes) Diffuse<1>(params);
        bsdf->id         = DIFFUSE_ID;
        bsdfs[num_bsdfs] = bsdf;
        weight *= src_params->albedo;
        break;
    }
    default:
        printf("add unknown: %s (%d), sz: %d\n", id_to_string(id), (int)id,
               num_bytes);
        break;
    }
    weights[num_bsdfs] = weight;
    num_bsdfs++;
    num_bytes += sz;
    return true;
}


//
// Helper functions to avoid virtual function calls
//

template <class BSDF_TYPE>
__forceinline__ __device__ Color3
get_albedo_fn(BSDF_TYPE* bsdf, const Vec3& wo)
{
    return bsdf->BSDF_TYPE::get_albedo(wo);
}


template <class BSDF_TYPE>
__forceinline__ __device__ BSDF::Sample
sample_fn(BSDF_TYPE* bsdf, const Vec3& wo, float rx, float ry, float rz)
{
    return bsdf->BSDF_TYPE::sample(wo, rx, ry, rz);
}


template <class BSDF_TYPE>
__forceinline__ __device__ BSDF::Sample
eval_fn(BSDF_TYPE* bsdf, const Vec3& wo, const Vec3& wi)
{
    return bsdf->BSDF_TYPE::eval(wo, wi);
}


OSL_HOSTDEVICE Color3
CompositeBSDF::get_bsdf_albedo(BSDF* bsdf, const Vec3& wo) const
{
    static const ustringhash uh_ggx("ggx");
    static const ustringhash uh_beckmann("beckmann");
    static const ustringhash uh_default("default");

    Color3 albedo(0);
    switch (bsdf->id) {
    case DIFFUSE_ID: albedo = get_albedo_fn((Diffuse<0>*)bsdf, wo); break;
    case TRANSPARENT_ID:
    case MX_TRANSPARENT_ID:
        albedo = get_albedo_fn((Transparent*)bsdf, wo);
        break;
    case OREN_NAYAR_ID: albedo = get_albedo_fn((OrenNayar*)bsdf, wo); break;
    case TRANSLUCENT_ID: albedo = get_albedo_fn((Diffuse<1>*)bsdf, wo); break;
    case PHONG_ID: albedo = get_albedo_fn((Phong*)bsdf, wo); break;
    case WARD_ID: albedo = get_albedo_fn((Ward*)bsdf, wo); break;
    case REFLECTION_ID:
    case FRESNEL_REFLECTION_ID:
        albedo = get_albedo_fn((Reflection*)bsdf, wo);
        break;
    case REFRACTION_ID: albedo = get_albedo_fn((Refraction*)bsdf, wo); break;
    case MICROFACET_ID: {
        const int refract      = ((MicrofacetBeckmannRefl*)bsdf)->refract;
        const ustringhash dist = ((MicrofacetBeckmannRefl*)bsdf)->dist;
        if (dist == uh_default || dist == uh_beckmann) {
            switch (refract) {
            case 0:
                albedo = get_albedo_fn((MicrofacetBeckmannRefl*)bsdf, wo);
                break;
            case 1:
                albedo = get_albedo_fn((MicrofacetBeckmannRefr*)bsdf, wo);
                break;
            case 2:
                albedo = get_albedo_fn((MicrofacetBeckmannBoth*)bsdf, wo);
                break;
            }
        } else if (dist == uh_ggx) {
            switch (refract) {
            case 0: albedo = get_albedo_fn((MicrofacetGGXRefl*)bsdf, wo); break;
            case 1: albedo = get_albedo_fn((MicrofacetGGXRefr*)bsdf, wo); break;
            case 2: albedo = get_albedo_fn((MicrofacetGGXBoth*)bsdf, wo); break;
            }
        }
        break;
    }
    case MX_CONDUCTOR_ID: albedo = get_albedo_fn((MxConductor*)bsdf, wo); break;
    case MX_DIELECTRIC_ID:
        if (is_black(((MxDielectricOpaque*)bsdf)->transmission_tint))
            albedo = get_albedo_fn((MxDielectricOpaque*)bsdf, wo);
        else
            albedo = get_albedo_fn((MxDielectric*)bsdf, wo);
        break;
    case MX_OREN_NAYAR_DIFFUSE_ID:
        albedo = get_albedo_fn((MxDielectric*)bsdf, wo);
        break;
    case MX_BURLEY_DIFFUSE_ID:
        albedo = get_albedo_fn((MxBurleyDiffuse*)bsdf, wo);
        break;
    case MX_SHEEN_ID: albedo = get_albedo_fn((MxSheen*)bsdf, wo); break;
    case MX_GENERALIZED_SCHLICK_ID: {
        const Color3& tint = ((MxGeneralizedSchlick*)bsdf)->transmission_tint;
        if (is_black(tint))
            albedo = get_albedo_fn((MxGeneralizedSchlickOpaque*)bsdf, wo);
        else
            albedo = get_albedo_fn((MxGeneralizedSchlick*)bsdf, wo);
        break;
    }
    default: break;
    }
    return albedo;
}


OSL_HOSTDEVICE BSDF::Sample
CompositeBSDF::sample_bsdf(BSDF* bsdf, const Vec3& wo, float rx, float ry,
                           float rz) const
{
    static const ustringhash uh_ggx("ggx");
    static const ustringhash uh_beckmann("beckmann");
    static const ustringhash uh_default("default");

    BSDF::Sample sample = {};
    switch (bsdf->id) {
    case DIFFUSE_ID:
        sample = sample_fn((Diffuse<0>*)bsdf, wo, rx, ry, rz);
        break;
    case TRANSPARENT_ID:
    case MX_TRANSPARENT_ID:
        sample = sample_fn((Transparent*)bsdf, wo, rx, ry, rz);
        break;
    case OREN_NAYAR_ID:
        sample = sample_fn((OrenNayar*)bsdf, wo, rx, ry, rz);
        break;
    case TRANSLUCENT_ID:
        sample = sample_fn((Diffuse<1>*)bsdf, wo, rx, ry, rz);
        break;
    case PHONG_ID: sample = sample_fn((Phong*)bsdf, wo, rx, ry, rz); break;
    case WARD_ID: sample = sample_fn((Ward*)bsdf, wo, rx, ry, rz); break;
    case REFLECTION_ID:
    case FRESNEL_REFLECTION_ID:
        sample = sample_fn((Reflection*)bsdf, wo, rx, ry, rz);
        break;
    case REFRACTION_ID:
        sample = sample_fn((Refraction*)bsdf, wo, rx, ry, rz);
        break;
    case MICROFACET_ID: {
        const int refract      = ((MicrofacetBeckmannRefl*)bsdf)->refract;
        const ustringhash dist = ((MicrofacetBeckmannRefl*)bsdf)->dist;
        if (dist == uh_default || dist == uh_beckmann) {
            switch (refract) {
            case 0:
                sample = sample_fn((MicrofacetBeckmannRefl*)bsdf, wo, rx, ry,
                                   rz);
                break;
            case 1:
                sample = sample_fn((MicrofacetBeckmannRefr*)bsdf, wo, rx, ry,
                                   rz);
                break;
            case 2:
                sample = sample_fn((MicrofacetBeckmannBoth*)bsdf, wo, rx, ry,
                                   rz);
                break;
            }
        } else if (dist == uh_ggx) {
            switch (refract) {
            case 0:
                sample = sample_fn((MicrofacetGGXRefl*)bsdf, wo, rx, ry, rz);
                break;
            case 1:
                sample = sample_fn((MicrofacetGGXRefr*)bsdf, wo, rx, ry, rz);
                break;
            case 2:
                sample = sample_fn((MicrofacetGGXBoth*)bsdf, wo, rx, ry, rz);
                break;
            }
        }
        break;
    }
    case MX_CONDUCTOR_ID:
        sample = sample_fn((MxConductor*)bsdf, wo, rx, ry, rz);
        break;
    case MX_DIELECTRIC_ID:
        if (is_black(((MxDielectricOpaque*)bsdf)->transmission_tint))
            sample = sample_fn((MxDielectricOpaque*)bsdf, wo, rx, ry, rz);
        else
            sample = sample_fn((MxDielectric*)bsdf, wo, rx, ry, rz);
        break;
    case MX_BURLEY_DIFFUSE_ID:
        sample = sample_fn((MxBurleyDiffuse*)bsdf, wo, rx, ry, rz);
        break;
    case MX_OREN_NAYAR_DIFFUSE_ID:
        sample = sample_fn((MxDielectric*)bsdf, wo, rx, ry, rz);
        break;
    case MX_SHEEN_ID: sample = sample_fn((MxSheen*)bsdf, wo, rx, ry, rz); break;
    case MX_GENERALIZED_SCHLICK_ID: {
        const Color3& tint = ((MxGeneralizedSchlick*)bsdf)->transmission_tint;
        if (is_black(tint)) {
            sample = sample_fn((MxGeneralizedSchlickOpaque*)bsdf, wo, rx, ry,
                               rz);
        } else {
            sample = sample_fn((MxGeneralizedSchlick*)bsdf, wo, rx, ry, rz);
        }
        break;
    }
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
    static const ustringhash uh_ggx("ggx");
    static const ustringhash uh_beckmann("beckmann");
    static const ustringhash uh_default("default");

    BSDF::Sample sample = {};
    switch (bsdf->id) {
    case DIFFUSE_ID: sample = eval_fn((Diffuse<0>*)bsdf, wo, wi); break;
    case TRANSPARENT_ID:
    case MX_TRANSPARENT_ID: sample = eval_fn((Transparent*)bsdf, wo, wi); break;
    case OREN_NAYAR_ID: sample = eval_fn((OrenNayar*)bsdf, wo, wi); break;
    case TRANSLUCENT_ID: sample = eval_fn((Diffuse<1>*)bsdf, wo, wi); break;
    case PHONG_ID: sample = eval_fn((Phong*)bsdf, wo, wi); break;
    case WARD_ID: sample = eval_fn((Ward*)bsdf, wo, wi); break;
    case REFLECTION_ID:
    case FRESNEL_REFLECTION_ID:
        sample = eval_fn((Reflection*)bsdf, wo, wi);
        break;
    case REFRACTION_ID: sample = eval_fn((Refraction*)bsdf, wo, wi); break;
    case MICROFACET_ID: {
        const int refract      = ((MicrofacetBeckmannRefl*)bsdf)->refract;
        const ustringhash dist = ((MicrofacetBeckmannRefl*)bsdf)->dist;
        if (dist == uh_default || dist == uh_beckmann) {
            switch (refract) {
            case 0:
                sample = eval_fn((MicrofacetBeckmannRefl*)bsdf, wo, wi);
                break;
            case 1:
                sample = eval_fn((MicrofacetBeckmannRefr*)bsdf, wo, wi);
                break;
            case 2:
                sample = eval_fn((MicrofacetBeckmannBoth*)bsdf, wo, wi);
                break;
            }
        } else if (dist == uh_ggx) {
            switch (refract) {
            case 0: sample = eval_fn((MicrofacetGGXRefl*)bsdf, wo, wi); break;
            case 1: sample = eval_fn((MicrofacetGGXRefr*)bsdf, wo, wi); break;
            case 2: sample = eval_fn((MicrofacetGGXBoth*)bsdf, wo, wi); break;
            }
        }
        break;
    }
    case MX_CONDUCTOR_ID: sample = eval_fn((MxConductor*)bsdf, wo, wi); break;
    case MX_DIELECTRIC_ID:
        if (is_black(((MxDielectricOpaque*)bsdf)->transmission_tint))
            sample = eval_fn((MxDielectricOpaque*)bsdf, wo, wi);
        else
            sample = eval_fn((MxDielectric*)bsdf, wo, wi);
        break;
    case MX_BURLEY_DIFFUSE_ID:
        sample = eval_fn((MxBurleyDiffuse*)bsdf, wo, wi);
        break;
    case MX_OREN_NAYAR_DIFFUSE_ID:
        sample = eval_fn((MxDielectric*)bsdf, wo, wi);
        break;
    case MX_SHEEN_ID: sample = ((MxSheen*)bsdf)->MxSheen::eval(wo, wi); break;
    case MX_GENERALIZED_SCHLICK_ID: {
        const Color3& tint = ((MxGeneralizedSchlick*)bsdf)->transmission_tint;
        if (is_black(tint)) {
            sample = eval_fn((MxGeneralizedSchlickOpaque*)bsdf, wo, wi);
        } else {
            sample = eval_fn((MxGeneralizedSchlick*)bsdf, wo, wi);
        }
        break;
    }
    default: break;
    }
    if (sample.pdf != sample.pdf) {
        uint3 launch_index = optixGetLaunchIndex();
        printf("eval_bsdf( %s ), PDF is NaN [%d, %d]\n", id_to_string(bsdf->id),
               launch_index.x, launch_index.y);
    }
    return sample;
}

OSL_NAMESPACE_EXIT
