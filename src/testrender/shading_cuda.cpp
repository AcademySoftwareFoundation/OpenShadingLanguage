// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <OSL/genclosure.h>
#include "optics.h"
#include "sampling.h"
#include "shading.h"

#ifdef __CUDACC__
#    include "cuda/vec_math.h"
#endif


namespace {  // anonymous namespace
static OSL_HOSTDEVICE const char*
id_to_string(int id)
{
    switch (id) {
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
    case ClosureIDs::FRESNEL_REFLECTION_ID:
        return "FRESNEL_REFLECTION_ID";
        break;
    case ClosureIDs::REFRACTION_ID: return "REFRACTION_ID"; break;
    case ClosureIDs::TRANSPARENT_ID: return "TRANSPARENT_ID"; break;
    case ClosureIDs::DEBUG_ID: return "DEBUG_ID"; break;
    case ClosureIDs::HOLDOUT_ID: return "HOLDOUT_ID"; break;
    case ClosureIDs::MX_OREN_NAYAR_DIFFUSE_ID:
        return "MX_OREN_NAYAR_DIFFUSE_ID";
        break;
    case ClosureIDs::MX_BURLEY_DIFFUSE_ID: return "MX_BURLEY_DIFFUSE_ID"; break;
    case ClosureIDs::MX_DIELECTRIC_ID: return "MX_DIELECTRIC_ID"; break;
    case ClosureIDs::MX_CONDUCTOR_ID: return "MX_CONDUCTOR_ID"; break;
    case ClosureIDs::MX_GENERALIZED_SCHLICK_ID:
        return "MX_GENERALIZED_SCHLICK_ID";
        break;
    case ClosureIDs::MX_TRANSLUCENT_ID: return "MX_TRANSLUCENT_ID"; break;
    case ClosureIDs::MX_TRANSPARENT_ID: return "MX_TRANSPARENT_ID"; break;
    case ClosureIDs::MX_SUBSURFACE_ID: return "MX_SUBSURFACE_ID"; break;
    case ClosureIDs::MX_SHEEN_ID: return "MX_SHEEN_ID"; break;
    case ClosureIDs::MX_UNIFORM_EDF_ID: return "MX_UNIFORM_EDF_ID"; break;
    case ClosureIDs::MX_ANISOTROPIC_VDF_ID:
        return "MX_ANISOTROPIC_VDF_ID";
        break;
    case ClosureIDs::MX_MEDIUM_VDF_ID: return "MX_MEDIUM_VDF_ID"; break;
    case ClosureIDs::MX_LAYER_ID: return "MX_LAYER_ID"; break;
    case ClosureIDs::EMPTY_ID: return "EMPTY_ID"; break;
    default: break;
    };
    return "UNKNOWN_ID";
}
}  // anonymous namespace


OSL_NAMESPACE_ENTER


typedef MxMicrofacet<MxConductorParams, GGXDist, MX_CONDUCTOR_ID, false>
    MxConductor;
typedef MxMicrofacet<MxDielectricParams, GGXDist, MX_DIELECTRIC_ID, true>
    MxDielectric;
typedef MxMicrofacet<MxDielectricParams, GGXDist, MX_DIELECTRIC_ID, false>
    MxDielectricOpaque;
typedef MxMicrofacet<MxGeneralizedSchlickParams, GGXDist,
                     MX_GENERALIZED_SCHLICK_ID, true>
    MxGeneralizedSchlick;
typedef MxMicrofacet<MxGeneralizedSchlickParams, GGXDist,
                     MX_GENERALIZED_SCHLICK_ID, false>
    MxGeneralizedSchlickOpaque;


// Cast a BSDF* to the specified sub-type
#define BSDF_CAST(BSDF_TYPE, bsdf) reinterpret_cast<const BSDF_TYPE*>(bsdf)


OSL_HOSTDEVICE Color3
CompositeBSDF::get_bsdf_albedo(const BSDF* bsdf, const Vec3& wo) const
{
    static const ustringhash uh_ggx("ggx");
    static const ustringhash uh_beckmann("beckmann");
    static const ustringhash uh_default("default");

    Color3 albedo(0);
    switch (bsdf->id) {
    case DIFFUSE_ID:
        albedo = BSDF_CAST(Diffuse<0>, bsdf)->get_albedo(wo);
        break;
    case TRANSPARENT_ID:
    case MX_TRANSPARENT_ID:
        albedo = BSDF_CAST(Transparent, bsdf)->get_albedo(wo);
        break;
    case OREN_NAYAR_ID:
        albedo = BSDF_CAST(OrenNayar, bsdf)->get_albedo(wo);
        break;
    case TRANSLUCENT_ID:
        albedo = BSDF_CAST(Diffuse<1>, bsdf)->get_albedo(wo);
        break;
    case PHONG_ID: albedo = BSDF_CAST(Phong, bsdf)->get_albedo(wo); break;
    case WARD_ID: albedo = BSDF_CAST(Ward, bsdf)->get_albedo(wo); break;
    case REFLECTION_ID:
    case FRESNEL_REFLECTION_ID:
        albedo = BSDF_CAST(Reflection, bsdf)->get_albedo(wo);
        break;
    case REFRACTION_ID:
        albedo = BSDF_CAST(Refraction, bsdf)->get_albedo(wo);
        break;
    case MICROFACET_ID: {
        const int refract      = ((MicrofacetBeckmannRefl*)bsdf)->refract;
        const ustringhash dist = ((MicrofacetBeckmannRefl*)bsdf)->dist;
        if (dist == uh_default || dist == uh_beckmann) {
            switch (refract) {
            case 0:
                albedo = BSDF_CAST(MicrofacetBeckmannRefl, bsdf)->get_albedo(wo);
                break;
            case 1:
                albedo = BSDF_CAST(MicrofacetBeckmannRefr, bsdf)->get_albedo(wo);
                break;
            case 2:
                albedo = BSDF_CAST(MicrofacetBeckmannBoth, bsdf)->get_albedo(wo);
                break;
            }
        } else if (dist == uh_ggx) {
            switch (refract) {
            case 0:
                albedo = BSDF_CAST(MicrofacetGGXRefl, bsdf)->get_albedo(wo);
                break;
            case 1:
                albedo = BSDF_CAST(MicrofacetGGXRefr, bsdf)->get_albedo(wo);
                break;
            case 2:
                albedo = BSDF_CAST(MicrofacetGGXBoth, bsdf)->get_albedo(wo);
                break;
            }
        }
        break;
    }
    case MX_CONDUCTOR_ID:
        albedo = BSDF_CAST(MxConductor, bsdf)->get_albedo(wo);
        break;
    case MX_DIELECTRIC_ID:
        if (is_black(((MxDielectricOpaque*)bsdf)->transmission_tint))
            albedo = BSDF_CAST(MxDielectricOpaque, bsdf)->get_albedo(wo);
        else
            albedo = BSDF_CAST(MxDielectric, bsdf)->get_albedo(wo);
        break;
    case MX_OREN_NAYAR_DIFFUSE_ID:
        albedo = BSDF_CAST(MxDielectric, bsdf)->get_albedo(wo);
        break;
    case MX_BURLEY_DIFFUSE_ID:
        albedo = BSDF_CAST(MxBurleyDiffuse, bsdf)->get_albedo(wo);
        break;
    case MX_SHEEN_ID: albedo = BSDF_CAST(MxSheen, bsdf)->get_albedo(wo); break;
    case MX_GENERALIZED_SCHLICK_ID: {
        const Color3& tint = ((MxGeneralizedSchlick*)bsdf)->transmission_tint;
        if (is_black(tint))
            albedo = BSDF_CAST(MxGeneralizedSchlickOpaque, bsdf)->get_albedo(wo);
        else
            albedo = BSDF_CAST(MxGeneralizedSchlick, bsdf)->get_albedo(wo);
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
        sample = BSDF_CAST(Diffuse<0>, bsdf)->sample(wo, rx, ry, rz);
        break;
    case TRANSPARENT_ID:
    case MX_TRANSPARENT_ID:
        sample = BSDF_CAST(Transparent, bsdf)->sample(wo, rx, ry, rz);
        break;
    case OREN_NAYAR_ID:
        sample = BSDF_CAST(OrenNayar, bsdf)->sample(wo, rx, ry, rz);
        break;
    case TRANSLUCENT_ID:
        sample = BSDF_CAST(Diffuse<1>, bsdf)->sample(wo, rx, ry, rz);
        break;
    case PHONG_ID:
        sample = BSDF_CAST(Phong, bsdf)->sample(wo, rx, ry, rz);
        break;
    case WARD_ID: sample = BSDF_CAST(Ward, bsdf)->sample(wo, rx, ry, rz); break;
    case REFLECTION_ID:
    case FRESNEL_REFLECTION_ID:
        sample = BSDF_CAST(Reflection, bsdf)->sample(wo, rx, ry, rz);
        break;
    case REFRACTION_ID:
        sample = BSDF_CAST(Refraction, bsdf)->sample(wo, rx, ry, rz);
        break;
    case MICROFACET_ID: {
        const int refract      = ((MicrofacetBeckmannRefl*)bsdf)->refract;
        const ustringhash dist = ((MicrofacetBeckmannRefl*)bsdf)->dist;
        if (dist == uh_default || dist == uh_beckmann) {
            switch (refract) {
            case 0:
                sample = BSDF_CAST(MicrofacetBeckmannRefl, bsdf)
                             ->sample(wo, rx, ry, rz);
                break;
            case 1:
                sample = BSDF_CAST(MicrofacetBeckmannRefr, bsdf)
                             ->sample(wo, rx, ry, rz);
                break;
            case 2:
                sample = BSDF_CAST(MicrofacetBeckmannBoth, bsdf)
                             ->sample(wo, rx, ry, rz);
                break;
            }
        } else if (dist == uh_ggx) {
            switch (refract) {
            case 0:
                sample
                    = BSDF_CAST(MicrofacetGGXRefl, bsdf)->sample(wo, rx, ry, rz);
                break;
            case 1:
                sample
                    = BSDF_CAST(MicrofacetGGXRefr, bsdf)->sample(wo, rx, ry, rz);
                break;
            case 2:
                sample
                    = BSDF_CAST(MicrofacetGGXBoth, bsdf)->sample(wo, rx, ry, rz);
                break;
            }
        }
        break;
    }
    case MX_CONDUCTOR_ID:
        sample = BSDF_CAST(MxConductor, bsdf)->sample(wo, rx, ry, rz);
        break;
    case MX_DIELECTRIC_ID:
        if (is_black(((MxDielectricOpaque*)bsdf)->transmission_tint))
            sample = BSDF_CAST(MxDielectricOpaque, bsdf)->sample(wo, rx, ry, rz);
        else
            sample = BSDF_CAST(MxDielectric, bsdf)->sample(wo, rx, ry, rz);
        break;
    case MX_BURLEY_DIFFUSE_ID:
        sample = BSDF_CAST(MxBurleyDiffuse, bsdf)->sample(wo, rx, ry, rz);
        break;
    case MX_OREN_NAYAR_DIFFUSE_ID:
        sample = BSDF_CAST(MxDielectric, bsdf)->sample(wo, rx, ry, rz);
        break;
    case MX_SHEEN_ID:
        sample = BSDF_CAST(MxSheen, bsdf)->sample(wo, rx, ry, rz);
        break;
    case MX_GENERALIZED_SCHLICK_ID: {
        const Color3& tint = ((MxGeneralizedSchlick*)bsdf)->transmission_tint;
        if (is_black(tint)) {
            sample = BSDF_CAST(MxGeneralizedSchlickOpaque, bsdf)
                         ->sample(wo, rx, ry, rz);
        } else {
            sample
                = BSDF_CAST(MxGeneralizedSchlick, bsdf)->sample(wo, rx, ry, rz);
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
    case DIFFUSE_ID: sample = BSDF_CAST(Diffuse<0>, bsdf)->eval(wo, wi); break;
    case TRANSPARENT_ID:
    case MX_TRANSPARENT_ID:
        sample = BSDF_CAST(Transparent, bsdf)->eval(wo, wi);
        break;
    case OREN_NAYAR_ID:
        sample = BSDF_CAST(OrenNayar, bsdf)->eval(wo, wi);
        break;
    case TRANSLUCENT_ID:
        sample = BSDF_CAST(Diffuse<1>, bsdf)->eval(wo, wi);
        break;
    case PHONG_ID: sample = BSDF_CAST(Phong, bsdf)->eval(wo, wi); break;
    case WARD_ID: sample = BSDF_CAST(Ward, bsdf)->eval(wo, wi); break;
    case REFLECTION_ID:
    case FRESNEL_REFLECTION_ID:
        sample = BSDF_CAST(Reflection, bsdf)->eval(wo, wi);
        break;
    case REFRACTION_ID:
        sample = BSDF_CAST(Refraction, bsdf)->eval(wo, wi);
        break;
    case MICROFACET_ID: {
        const int refract      = ((MicrofacetBeckmannRefl*)bsdf)->refract;
        const ustringhash dist = ((MicrofacetBeckmannRefl*)bsdf)->dist;
        if (dist == uh_default || dist == uh_beckmann) {
            switch (refract) {
            case 0:
                sample = BSDF_CAST(MicrofacetBeckmannRefl, bsdf)->eval(wo, wi);
                break;
            case 1:
                sample = BSDF_CAST(MicrofacetBeckmannRefr, bsdf)->eval(wo, wi);
                break;
            case 2:
                sample = BSDF_CAST(MicrofacetBeckmannBoth, bsdf)->eval(wo, wi);
                break;
            }
        } else if (dist == uh_ggx) {
            switch (refract) {
            case 0:
                sample = BSDF_CAST(MicrofacetGGXRefl, bsdf)->eval(wo, wi);
                break;
            case 1:
                sample = BSDF_CAST(MicrofacetGGXRefr, bsdf)->eval(wo, wi);
                break;
            case 2:
                sample = BSDF_CAST(MicrofacetGGXBoth, bsdf)->eval(wo, wi);
                break;
            }
        }
        break;
    }
    case MX_CONDUCTOR_ID:
        sample = BSDF_CAST(MxConductor, bsdf)->eval(wo, wi);
        break;
    case MX_DIELECTRIC_ID:
        if (is_black(((MxDielectricOpaque*)bsdf)->transmission_tint))
            sample = BSDF_CAST(MxDielectricOpaque, bsdf)->eval(wo, wi);
        else
            sample = BSDF_CAST(MxDielectric, bsdf)->eval(wo, wi);
        break;
    case MX_BURLEY_DIFFUSE_ID:
        sample = BSDF_CAST(MxBurleyDiffuse, bsdf)->eval(wo, wi);
        break;
    case MX_OREN_NAYAR_DIFFUSE_ID:
        sample = BSDF_CAST(MxDielectric, bsdf)->eval(wo, wi);
        break;
    case MX_SHEEN_ID: sample = ((MxSheen*)bsdf)->MxSheen::eval(wo, wi); break;
    case MX_GENERALIZED_SCHLICK_ID: {
        const Color3& tint = ((MxGeneralizedSchlick*)bsdf)->transmission_tint;
        if (is_black(tint)) {
            sample = BSDF_CAST(MxGeneralizedSchlickOpaque, bsdf)->eval(wo, wi);
        } else {
            sample = BSDF_CAST(MxGeneralizedSchlick, bsdf)->eval(wo, wi);
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
