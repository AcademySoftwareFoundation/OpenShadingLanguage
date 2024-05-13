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


//
// Helper functions to avoid virtual function calls
//

template <class BSDF_TYPE>
__forceinline__ __device__ Color3
get_albedo_fn(const BSDF_TYPE* bsdf, const Vec3& wo)
{
    return bsdf->BSDF_TYPE::get_albedo(wo);
}


template <class BSDF_TYPE>
__forceinline__ __device__ BSDF::Sample
sample_fn(const BSDF_TYPE* bsdf, const Vec3& wo, float rx, float ry, float rz)
{
    return bsdf->BSDF_TYPE::sample(wo, rx, ry, rz);
}


template <class BSDF_TYPE>
__forceinline__ __device__ BSDF::Sample
eval_fn(const BSDF_TYPE* bsdf, const Vec3& wo, const Vec3& wi)
{
    return bsdf->BSDF_TYPE::eval(wo, wi);
}


OSL_HOSTDEVICE Color3
CompositeBSDF::get_bsdf_albedo(const BSDF* bsdf, const Vec3& wo) const
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
CompositeBSDF::sample_bsdf(const BSDF* bsdf, const Vec3& wo, float rx, float ry,
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
CompositeBSDF::eval_bsdf(const BSDF* bsdf, const Vec3& wo, const Vec3& wi) const
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
