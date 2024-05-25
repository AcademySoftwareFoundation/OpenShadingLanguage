// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include "shading.h"
#include <OSL/genclosure.h>
#include "optics.h"
#include "sampling.h"

using namespace OSL;

namespace {  // anonymous namespace

using OIIO::clamp;

Color3
clamp(const Color3& c, float min, float max)
{
    return Color3(clamp(c.x, min, max), clamp(c.y, min, max),
                  clamp(c.z, min, max));
}

bool
is_black(const Color3& c)
{
    return c.x == 0 && c.y == 0 && c.z == 0;
}



// unique identifier for each closure supported by testrender
enum ClosureIDs {
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
    ustring dist;
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
    ustring label;
    int energy_compensation;
};

struct MxBurleyDiffuseParams {
    Vec3 N;
    Color3 albedo;
    float roughness;
    // optional
    ustring label;
};

// common to all MaterialX microfacet closures
struct MxMicrofacetBaseParams {
    Vec3 N, U;
    float roughness_x;
    float roughness_y;
    ustring distribution;
    // optional
    ustring label;
};

struct MxDielectricParams : public MxMicrofacetBaseParams {
    Color3 reflection_tint;
    Color3 transmission_tint;
    float ior;
    // optional
    float thinfilm_thickness;
    float thinfilm_ior;

    Color3 evalR(float cos_theta) const
    {
        return reflection_tint * fresnel_dielectric(cos_theta, ior);
    }

    Color3 evalT(float cos_theta) const
    {
        return transmission_tint * (1.0f - fresnel_dielectric(cos_theta, ior));
    }
};

struct MxConductorParams : public MxMicrofacetBaseParams {
    Color3 ior;
    Color3 extinction;
    // optional
    float thinfilm_thickness;
    float thinfilm_ior;

    Color3 evalR(float cos_theta) const
    {
        return fresnel_conductor(cos_theta, ior, extinction);
    }

    Color3 evalT(float cos_theta) const { return Color3(0.0f); }

    // Avoid function was declared but never referenced
    // float get_ior() const
    // {
    //     return 0;  // no transmission possible
    // }
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

    Color3 evalR(float cos_theta) const
    {
        return reflection_tint
               * fresnel_generalized_schlick(cos_theta, f0, f90, exponent);
    }

    Color3 evalT(float cos_theta) const
    {
        return transmission_tint
               * (Color3(1.0f)
                  - fresnel_generalized_schlick(cos_theta, f0, f90, exponent));
    }
};

struct MxTranslucentParams {
    Vec3 N;
    Color3 albedo;
    // optional
    ustring label;
};

struct MxSubsurfaceParams {
    Vec3 N;
    Color3 albedo;
    float transmission_depth;
    Color3 transmission_color;
    float anisotropy;
    // optional
    ustring label;
};

struct MxSheenParams {
    Vec3 N;
    Color3 albedo;
    float roughness;
    // optional
    ustring label;
};

struct MxUniformEdfParams {
    Color3 emittance;
    // optional
    ustring label;
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
    ustring label;
};

struct MxMediumVdfParams {
    Color3 albedo;
    float transmission_depth;
    Color3 transmission_color;
    float anisotropy;
    float ior;
    int priority;
    // optional
    ustring label;
};

}  // anonymous namespace


OSL_NAMESPACE_ENTER


void
register_closures(OSL::ShadingSystem* shadingsys)
{
    // Describe the memory layout of each closure type to the OSL runtime
    constexpr int MaxParams = 32;
    struct BuiltinClosures {
        const char* name;
        int id;
        ClosureParam params[MaxParams];  // upper bound
    };
    BuiltinClosures builtins[] = {
        { "emission", EMISSION_ID, { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "background", BACKGROUND_ID, { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "diffuse",
          DIFFUSE_ID,
          { CLOSURE_VECTOR_PARAM(DiffuseParams, N),
            CLOSURE_FINISH_PARAM(DiffuseParams) } },
        { "oren_nayar",
          OREN_NAYAR_ID,
          { CLOSURE_VECTOR_PARAM(OrenNayarParams, N),
            CLOSURE_FLOAT_PARAM(OrenNayarParams, sigma),
            CLOSURE_FINISH_PARAM(OrenNayarParams) } },
        { "translucent",
          TRANSLUCENT_ID,
          { CLOSURE_VECTOR_PARAM(DiffuseParams, N),
            CLOSURE_FINISH_PARAM(DiffuseParams) } },
        { "phong",
          PHONG_ID,
          { CLOSURE_VECTOR_PARAM(PhongParams, N),
            CLOSURE_FLOAT_PARAM(PhongParams, exponent),
            CLOSURE_FINISH_PARAM(PhongParams) } },
        { "ward",
          WARD_ID,
          { CLOSURE_VECTOR_PARAM(WardParams, N),
            CLOSURE_VECTOR_PARAM(WardParams, T),
            CLOSURE_FLOAT_PARAM(WardParams, ax),
            CLOSURE_FLOAT_PARAM(WardParams, ay),
            CLOSURE_FINISH_PARAM(WardParams) } },
        { "microfacet",
          MICROFACET_ID,
          { CLOSURE_STRING_PARAM(MicrofacetParams, dist),
            CLOSURE_VECTOR_PARAM(MicrofacetParams, N),
            CLOSURE_VECTOR_PARAM(MicrofacetParams, U),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, xalpha),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, yalpha),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, eta),
            CLOSURE_INT_PARAM(MicrofacetParams, refract),
            CLOSURE_FINISH_PARAM(MicrofacetParams) } },
        { "reflection",
          REFLECTION_ID,
          { CLOSURE_VECTOR_PARAM(ReflectionParams, N),
            CLOSURE_FINISH_PARAM(ReflectionParams) } },
        { "reflection",
          FRESNEL_REFLECTION_ID,
          { CLOSURE_VECTOR_PARAM(ReflectionParams, N),
            CLOSURE_FLOAT_PARAM(ReflectionParams, eta),
            CLOSURE_FINISH_PARAM(ReflectionParams) } },
        { "refraction",
          REFRACTION_ID,
          { CLOSURE_VECTOR_PARAM(RefractionParams, N),
            CLOSURE_FLOAT_PARAM(RefractionParams, eta),
            CLOSURE_FINISH_PARAM(RefractionParams) } },
        { "transparent", TRANSPARENT_ID, { CLOSURE_FINISH_PARAM(EmptyParams) } },
        // See MATERIALX_CLOSURES in stdosl.h
        { "oren_nayar_diffuse_bsdf",
          MX_OREN_NAYAR_DIFFUSE_ID,
          { CLOSURE_VECTOR_PARAM(MxOrenNayarDiffuseParams, N),
            CLOSURE_COLOR_PARAM(MxOrenNayarDiffuseParams, albedo),
            CLOSURE_FLOAT_PARAM(MxOrenNayarDiffuseParams, roughness),
            CLOSURE_STRING_KEYPARAM(MxOrenNayarDiffuseParams, label, "label"),
            CLOSURE_INT_KEYPARAM(MxOrenNayarDiffuseParams, energy_compensation,
                                 "energy_compensation"),
            CLOSURE_FINISH_PARAM(MxOrenNayarDiffuseParams) } },
        { "burley_diffuse_bsdf",
          MX_BURLEY_DIFFUSE_ID,
          { CLOSURE_VECTOR_PARAM(MxBurleyDiffuseParams, N),
            CLOSURE_COLOR_PARAM(MxBurleyDiffuseParams, albedo),
            CLOSURE_FLOAT_PARAM(MxBurleyDiffuseParams, roughness),
            CLOSURE_STRING_KEYPARAM(MxBurleyDiffuseParams, label, "label"),
            CLOSURE_FINISH_PARAM(MxBurleyDiffuseParams) } },
        { "dielectric_bsdf",
          MX_DIELECTRIC_ID,
          { CLOSURE_VECTOR_PARAM(MxDielectricParams, N),
            CLOSURE_VECTOR_PARAM(MxDielectricParams, U),
            CLOSURE_COLOR_PARAM(MxDielectricParams, reflection_tint),
            CLOSURE_COLOR_PARAM(MxDielectricParams, transmission_tint),
            CLOSURE_FLOAT_PARAM(MxDielectricParams, roughness_x),
            CLOSURE_FLOAT_PARAM(MxDielectricParams, roughness_y),
            CLOSURE_FLOAT_PARAM(MxDielectricParams, ior),
            CLOSURE_STRING_PARAM(MxDielectricParams, distribution),
            CLOSURE_FLOAT_KEYPARAM(MxDielectricParams, thinfilm_thickness,
                                   "thinfilm_thickness"),
            CLOSURE_FLOAT_KEYPARAM(MxDielectricParams, thinfilm_ior,
                                   "thinfilm_ior"),
            CLOSURE_STRING_KEYPARAM(MxDielectricParams, label, "label"),
            CLOSURE_FINISH_PARAM(MxDielectricParams) } },
        { "conductor_bsdf",
          MX_CONDUCTOR_ID,
          { CLOSURE_VECTOR_PARAM(MxConductorParams, N),
            CLOSURE_VECTOR_PARAM(MxConductorParams, U),
            CLOSURE_FLOAT_PARAM(MxConductorParams, roughness_x),
            CLOSURE_FLOAT_PARAM(MxConductorParams, roughness_y),
            CLOSURE_COLOR_PARAM(MxConductorParams, ior),
            CLOSURE_COLOR_PARAM(MxConductorParams, extinction),
            CLOSURE_STRING_PARAM(MxConductorParams, distribution),
            CLOSURE_FLOAT_KEYPARAM(MxConductorParams, thinfilm_thickness,
                                   "thinfilm_thickness"),
            CLOSURE_FLOAT_KEYPARAM(MxConductorParams, thinfilm_ior,
                                   "thinfilm_ior"),
            CLOSURE_STRING_KEYPARAM(MxConductorParams, label, "label"),
            CLOSURE_FINISH_PARAM(MxConductorParams) } },
        { "generalized_schlick_bsdf",
          MX_GENERALIZED_SCHLICK_ID,
          { CLOSURE_VECTOR_PARAM(MxGeneralizedSchlickParams, N),
            CLOSURE_VECTOR_PARAM(MxGeneralizedSchlickParams, U),
            CLOSURE_COLOR_PARAM(MxGeneralizedSchlickParams, reflection_tint),
            CLOSURE_COLOR_PARAM(MxGeneralizedSchlickParams, transmission_tint),
            CLOSURE_FLOAT_PARAM(MxGeneralizedSchlickParams, roughness_x),
            CLOSURE_FLOAT_PARAM(MxGeneralizedSchlickParams, roughness_y),
            CLOSURE_COLOR_PARAM(MxGeneralizedSchlickParams, f0),
            CLOSURE_COLOR_PARAM(MxGeneralizedSchlickParams, f90),
            CLOSURE_FLOAT_PARAM(MxGeneralizedSchlickParams, exponent),
            CLOSURE_STRING_PARAM(MxGeneralizedSchlickParams, distribution),
            CLOSURE_FLOAT_KEYPARAM(MxGeneralizedSchlickParams,
                                   thinfilm_thickness, "thinfilm_thickness"),
            CLOSURE_FLOAT_KEYPARAM(MxGeneralizedSchlickParams, thinfilm_ior,
                                   "thinfilm_ior"),
            CLOSURE_STRING_KEYPARAM(MxGeneralizedSchlickParams, label, "label"),
            CLOSURE_FINISH_PARAM(MxGeneralizedSchlickParams) } },
        { "translucent_bsdf",
          MX_TRANSLUCENT_ID,
          { CLOSURE_VECTOR_PARAM(MxTranslucentParams, N),
            CLOSURE_COLOR_PARAM(MxTranslucentParams, albedo),
            CLOSURE_STRING_KEYPARAM(MxTranslucentParams, label, "label"),
            CLOSURE_FINISH_PARAM(MxTranslucentParams) } },
        { "transparent_bsdf",
          MX_TRANSPARENT_ID,
          { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "subsurface_bssrdf",
          MX_SUBSURFACE_ID,
          { CLOSURE_VECTOR_PARAM(MxSubsurfaceParams, N),
            CLOSURE_COLOR_PARAM(MxSubsurfaceParams, albedo),
            CLOSURE_FLOAT_PARAM(MxSubsurfaceParams, transmission_depth),
            CLOSURE_COLOR_PARAM(MxSubsurfaceParams, transmission_color),
            CLOSURE_FLOAT_PARAM(MxSubsurfaceParams, anisotropy),
            CLOSURE_STRING_KEYPARAM(MxSubsurfaceParams, label, "label"),
            CLOSURE_FINISH_PARAM(MxSubsurfaceParams) } },
        { "sheen_bsdf",
          MX_SHEEN_ID,
          { CLOSURE_VECTOR_PARAM(MxSheenParams, N),
            CLOSURE_COLOR_PARAM(MxSheenParams, albedo),
            CLOSURE_FLOAT_PARAM(MxSheenParams, roughness),
            CLOSURE_STRING_KEYPARAM(MxSheenParams, label, "label"),
            CLOSURE_FINISH_PARAM(MxSheenParams) } },
        { "uniform_edf",
          MX_UNIFORM_EDF_ID,
          { CLOSURE_COLOR_PARAM(MxUniformEdfParams, emittance),
            CLOSURE_STRING_KEYPARAM(MxUniformEdfParams, label, "label"),
            CLOSURE_FINISH_PARAM(MxUniformEdfParams) } },
        { "layer",
          MX_LAYER_ID,
          { CLOSURE_CLOSURE_PARAM(MxLayerParams, top),
            CLOSURE_CLOSURE_PARAM(MxLayerParams, base),
            CLOSURE_FINISH_PARAM(MxLayerParams) } },
        { "anisotropic_vdf",
          MX_ANISOTROPIC_VDF_ID,
          { CLOSURE_COLOR_PARAM(MxAnisotropicVdfParams, albedo),
            CLOSURE_COLOR_PARAM(MxAnisotropicVdfParams, extinction),
            CLOSURE_FLOAT_PARAM(MxAnisotropicVdfParams, anisotropy),
            CLOSURE_STRING_KEYPARAM(MxAnisotropicVdfParams, label, "label"),
            CLOSURE_FINISH_PARAM(MxAnisotropicVdfParams) } },
        { "medium_vdf",
          MX_MEDIUM_VDF_ID,
          { CLOSURE_COLOR_PARAM(MxMediumVdfParams, albedo),
            CLOSURE_FLOAT_PARAM(MxMediumVdfParams, transmission_depth),
            CLOSURE_COLOR_PARAM(MxMediumVdfParams, transmission_color),
            CLOSURE_FLOAT_PARAM(MxMediumVdfParams, anisotropy),
            CLOSURE_FLOAT_PARAM(MxMediumVdfParams, ior),
            CLOSURE_INT_PARAM(MxMediumVdfParams, priority),
            CLOSURE_STRING_KEYPARAM(MxMediumVdfParams, label, "label"),
            CLOSURE_FINISH_PARAM(MxMediumVdfParams) } },
    };

    for (const BuiltinClosures& b : builtins)
        shadingsys->register_closure(b.name, b.id, b.params, nullptr, nullptr);
}

OSL_NAMESPACE_EXIT

namespace {  // anonymous namespace

template<int trans> struct Diffuse final : public BSDF, DiffuseParams {
    Diffuse(const DiffuseParams& params) : BSDF(), DiffuseParams(params)
    {
        if (trans)
            N = -N;
    }
    Sample eval(const Vec3& /*wo*/, const OSL::Vec3& wi) const override
    {
        const float pdf = std::max(N.dot(wi), 0.0f) * float(M_1_PI);
        return { wi, Color3(1.0f), pdf, 1.0f };
    }
    Sample sample(const Vec3& /*wo*/, float rx, float ry,
                  float /*rz*/) const override
    {
        Vec3 out_dir;
        float pdf;
        Sampling::sample_cosine_hemisphere(N, rx, ry, out_dir, pdf);
        return { out_dir, Color3(1.0f), pdf, 1.0f };
    }
};

struct OrenNayar final : public BSDF, OrenNayarParams {
    OrenNayar(const OrenNayarParams& params) : BSDF(), OrenNayarParams(params)
    {
    }
    Sample eval(const Vec3& wo, const OSL::Vec3& wi) const override
    {
        float NL = N.dot(wi);
        float NV = N.dot(wo);
        if (NL > 0 && NV > 0) {
            float LV = wo.dot(wi);
            float s  = LV - NL * NV;
            // Simplified math from: "A tiny improvement of Oren-Nayar reflectance model"
            // by Yasuhiro Fujii
            // http://mimosa-pudica.net/improved-oren-nayar.html
            // NOTE: This is using the math to match the original qualitative ON model
            // (QON in the paper above) and not the tweak proposed in the text which
            // is a slightly different BRDF (FON in the paper above). This is done for
            // backwards compatibility purposes only.
            float s2    = sigma * sigma;
            float A     = 1 - 0.50f * s2 / (s2 + 0.33f);
            float B     = 0.45f * s2 / (s2 + 0.09f);
            float stinv = s > 0 ? s / std::max(NL, NV) : 0.0f;
            return { wi, Color3(A + B * stinv), NL * float(M_1_PI), 1.0f };
        }
        return {};
    }
    Sample sample(const Vec3& wo, float rx, float ry,
                  float /*rz*/) const override
    {
        Vec3 out_dir;
        float pdf;
        Sampling::sample_cosine_hemisphere(N, rx, ry, out_dir, pdf);
        return eval(wo, out_dir);
    }
};

struct EnergyCompensatedOrenNayar : public BSDF, MxOrenNayarDiffuseParams {
    EnergyCompensatedOrenNayar(const MxOrenNayarDiffuseParams& params)
        : BSDF(), MxOrenNayarDiffuseParams(params)
    {
    }
    Sample eval(const Vec3& wo, const OSL::Vec3& wi) const override
    {
        float NL = N.dot(wi);
        float NV = N.dot(wo);
        if (NL > 0 && NV > 0) {
            float LV = wo.dot(wi);
            float s  = LV - NL * NV;
            // Code below from Jamie Portsmouth's tech report on Energy conversion Oren-Nayar
            // See slack thread for whitepaper:
            // https://academysoftwarefdn.slack.com/files/U03SWQFPD08/F06S50CUKV1/oren_nayar.pdf

            // TODO: rho should be the albedo which is a parameter of the closure in the Mx parameters
            // This only matters for the color-saturation aspect of the BRDF which is rather subtle anyway
            // and not always desireable for artists. Hardcoding to 1 leaves the coloring entirely up to the
            // closure weight.

            const Color3 rho  = albedo;
            const float sigma = roughness;

            float AF      = 1.0f / (1.0f + constant1_FON * sigma);
            float stinv   = s > 0 ? s / std::max(NL, NV) : s;
            float f_ss    = AF * (1.0 + sigma * stinv);  // single-scatt. BRDF
            float EFo     = E_FON_analytic(NV);  // EFo at rho=1 (analytic)
            float EFi     = E_FON_analytic(NL);  // EFi at rho=1 (analytic)
            float avgEF   = AF * (1.0f + constant2_FON * sigma);  // avg. albedo
            Color3 rho_ms = (rho * rho) * avgEF
                            / (Color3(1.0f)
                               - rho * std::max(0.0f, 1.0f - avgEF));
            float f_ms = std::max(1e-7f, 1.0f - EFo)
                         * std::max(1e-7f, 1.0f - EFi)
                         / std::max(1e-7f, 1.0f - avgEF);  // multi-scatter lobe
            return { wi, Color3(rho * f_ss + rho_ms * f_ms), NL * float(M_1_PI),
                     1.0f };
        }
        return {};
    }

    Sample sample(const Vec3& wo, float rx, float ry,
                  float /*rz*/) const override
    {
        Vec3 out_dir;
        float pdf;
        Sampling::sample_cosine_hemisphere(N, rx, ry, out_dir, pdf);
        return eval(wo, out_dir);
    }

private:
    static constexpr float constant1_FON = float(0.5 - 2.0 / (3.0 * M_PI));
    static constexpr float constant2_FON = float(2.0 / 3.0
                                                 - 28.0 / (15.0 * M_PI));

    float E_FON_analytic(float mu) const
    {
        const float sigma = roughness;
        float AF          = 1.0f
                   / (1.0f
                      + constant1_FON * sigma);  // Fujii model A coefficient
        float BF = sigma * AF;                   // Fujii model B coefficient
        float Si = sqrtf(std::max(0.0f, 1.0f - mu * mu));
        float G  = Si * (OIIO::fast_acos(mu) - Si * mu)
                  + 2.0 * ((Si / mu) * (1.0 - Si * Si * Si) - Si) / 3.0f;
        float E = AF + (BF * float(M_1_PI)) * G;
        return E;
    }
};

struct Phong final : public BSDF, PhongParams {
    Phong(const PhongParams& params) : BSDF(), PhongParams(params) {}
    Sample eval(const Vec3& wo, const Vec3& wi) const override
    {
        float cosNI = N.dot(wi);
        float cosNO = N.dot(wo);
        if (cosNI > 0 && cosNO > 0) {
            // reflect the view vector
            Vec3 R      = (2 * cosNO) * N - wo;
            float cosRI = R.dot(wi);
            if (cosRI > 0) {
                const float pdf = (exponent + 1) * float(M_1_PI / 2)
                                  * OIIO::fast_safe_pow(cosRI, exponent);
                return { wi, Color3(cosNI * (exponent + 2) / (exponent + 1)),
                         pdf, 1 / (1 + exponent) };
            }
        }
        return {};
    }
    Sample sample(const Vec3& wo, float rx, float ry,
                  float /*rz*/) const override
    {
        float cosNO = N.dot(wo);
        if (cosNO > 0) {
            // reflect the view vector
            Vec3 R = (2 * cosNO) * N - wo;
            TangentFrame tf(R);
            float phi = 2 * float(M_PI) * rx;
            float sp, cp;
            OIIO::fast_sincos(phi, &sp, &cp);
            float cosTheta  = OIIO::fast_safe_pow(ry, 1 / (exponent + 1));
            float sinTheta2 = 1 - cosTheta * cosTheta;
            float sinTheta  = sinTheta2 > 0 ? sqrtf(sinTheta2) : 0;
            Vec3 wi         = tf.get(cp * sinTheta, sp * sinTheta, cosTheta);
            return eval(wo, wi);
        }
        return {};
    }
};

struct Ward final : public BSDF, WardParams {
    Ward(const WardParams& params) : BSDF(), WardParams(params) {}
    Sample eval(const Vec3& wo, const OSL::Vec3& wi) const override
    {
        float cosNO = N.dot(wo);
        float cosNI = N.dot(wi);
        if (cosNI > 0 && cosNO > 0) {
            // get half vector and get x,y basis on the surface for anisotropy
            Vec3 H = wi + wo;
            H.normalize();  // normalize needed for pdf
            TangentFrame tf(N, T);
            // eq. 4
            float dotx = tf.getx(H) / ax;
            float doty = tf.gety(H) / ay;
            float dotn = tf.getz(H);
            float oh   = H.dot(wi);
            float e    = OIIO::fast_exp(-(dotx * dotx + doty * doty)
                                        / (dotn * dotn));
            float c    = float(4 * M_PI) * ax * ay;
            float k    = oh * dotn * dotn * dotn;
            float pdf  = e / (c * k);
            return { wi, Color3(k * sqrtf(cosNI / cosNO)), pdf,
                     std::max(ax, ay) };
        }
        return {};
    }
    Sample sample(const Vec3& wo, float rx, float ry,
                  float /*rz*/) const override
    {
        float cosNO = N.dot(wo);
        if (cosNO > 0) {
            // get x,y basis on the surface for anisotropy
            TangentFrame tf(N, T);
            // generate random angles for the half vector
            float phi = 2 * float(M_PI) * rx;
            float sp, cp;
            OIIO::fast_sincos(phi, &sp, &cp);
            float cosPhi = ax * cp;
            float sinPhi = ay * sp;
            float k      = 1 / sqrtf(cosPhi * cosPhi + sinPhi * sinPhi);
            cosPhi *= k;
            sinPhi *= k;

            // eq. 6
            // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
            //                  and sin(atan(x)) == x/sqrt(1+x^2)
            float thetaDenom = (cosPhi * cosPhi) / (ax * ax)
                               + (sinPhi * sinPhi) / (ay * ay);
            float tanTheta2 = -OIIO::fast_log(1 - ry) / thetaDenom;
            float cosTheta  = 1 / sqrtf(1 + tanTheta2);
            float sinTheta  = cosTheta * sqrtf(tanTheta2);

            Vec3 h;  // already normalized because expressed from spherical coordinates
            h.x = sinTheta * cosPhi;
            h.y = sinTheta * sinPhi;
            h.z = cosTheta;
            // compute terms that are easier in local space
            float dotx = h.x / ax;
            float doty = h.y / ay;
            float dotn = h.z;
            // transform to world space
            h = tf.get(h.x, h.y, h.z);
            // generate the final sample
            float oh    = h.dot(wo);
            Vec3 wi     = 2 * oh * h - wo;
            float cosNI = N.dot(wi);
            if (cosNI > 0) {
                // eq. 9
                float e   = OIIO::fast_exp(-(dotx * dotx + doty * doty)
                                           / (dotn * dotn));
                float c   = float(4 * M_PI) * ax * ay;
                float k   = oh * dotn * dotn * dotn;
                float pdf = e / (c * k);
                return { wi, Color3(k * sqrtf(cosNI / cosNO)), pdf,
                         std::max(ax, ay) };
            }
        }
        return {};
    }
};

/* The anisotropic variant of GGX and Beckmann comes from
 * "Understanding the Masking-Shadowing Function in
 * Microfacet-Based BRDFs" by Eric Heitz, JCGT 2014 (section 5.4)
 *
 * We use the height correlated masking and shadowing function
 * instead of the separable form as it is more realistic and
 * reduces energy loss at grazing angles.
 *
 * The sampling method is derived from "Importance Sampling
 * Microfacet-Based BSDFs using the Distribution of Visible
 * Normals" by Eugene d'Eon and Eric Heitz, EGSR 2014
 *
 * The sampling method for GGX is simplified from the original
 * paper to be more numerically robust and more compact.
 *
 * The sampling method for Beckmann uses an improved variant of
 * "An Improved Visible Normal Sampling Routine for the Beckmann
 * Distribution" by Wenzel Jakob. The new formulation avoids
 * calls to inverse trigonometric functions and power functions
 * and does not require a loop for root refinement (a single step
 * is sufficient).
 */
struct GGXDist {
    static float F(const float tan_m2)
    {
        return 1 / (float(M_PI) * (1 + tan_m2) * (1 + tan_m2));
    }

    static float Lambda(const float a2)
    {
        return 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / a2));
    }

    static Vec2 sampleSlope(float cos_theta, float randu, float randv)
    {
        // GGX
        Vec2 slope;
        /* sample slope_x */

        float c   = cos_theta < 1e-6f ? 1e-6f : cos_theta;
        float Q   = (1 + c) * randu - c;
        float num = c * sqrtf((1 - c) * (1 + c)) - Q * sqrtf((1 - Q) * (1 + Q));
        float den = (Q - c) * (Q + c);
        float eps = 1.0f / 4294967296.0f;
        den       = fabsf(den) < eps ? copysignf(eps, den) : den;
        slope.x   = num / den;

        /* sample slope_y */
        float Ru = 1 - 2 * randv;
        float u2 = fabsf(Ru);
        float z  = (u2 * (u2 * (u2 * 0.27385f - 0.73369f) + 0.46341f))
                  / (u2 * (u2 * (u2 * 0.093073f + 0.309420f) - 1.0f)
                     + 0.597999f);
        slope.y = copysignf(1.0f, Ru) * z * sqrtf(1.0f + slope.x * slope.x);

        return slope;
    }
};

struct BeckmannDist {
    static float F(const float tan_m2)
    {
        return float(1 / M_PI) * OIIO::fast_exp(-tan_m2);
    }

    static float Lambda(const float a2)
    {
        const float a = sqrtf(a2);
        return a < 1.6f ? (1.0f - 1.259f * a + 0.396f * a2)
                              / (3.535f * a + 2.181f * a2)
                        : 0.0f;
    }

    static Vec2 sampleSlope(float cos_theta, float randu, float randv)
    {
        const float SQRT_PI_INV = 1 / sqrtf(float(M_PI));
        float ct                = cos_theta < 1e-6f ? 1e-6f : cos_theta;
        float tanThetaI         = sqrtf(1 - ct * ct) / ct;
        float cotThetaI         = 1 / tanThetaI;

        /* sample slope X */
        // compute a coarse approximation using the approximation:
        // exp(-ierf(x)^2) ~= 1 - x * x
        // solve y = 1 + b + K * (1 - b * b)
        float c       = OIIO::fast_erf(cotThetaI);
        float K       = tanThetaI * SQRT_PI_INV;
        float yApprox = randu * (1.0f + c + K * (1 - c * c));
        float yExact
            = randu * (1.0f + c + K * OIIO::fast_exp(-cotThetaI * cotThetaI));
        float b = K > 0 ? (0.5f - sqrtf(K * (K - yApprox + 1.0f) + 0.25f)) / K
                        : yApprox - 1.0f;

        // perform newton step to refine toward the true root
        float invErf = OIIO::fast_ierf(b);
        float value  = 1.0f + b + K * OIIO::fast_exp(-invErf * invErf) - yExact;

        // check if we are close enough already
        // this also avoids NaNs as we get close to the root
        Vec2 slope;
        if (fabsf(value) > 1e-6f) {
            b -= value / (1 - invErf * tanThetaI);  // newton step 1
            invErf = OIIO::fast_ierf(b);
            value  = 1.0f + b + K * OIIO::fast_exp(-invErf * invErf) - yExact;
            b -= value / (1 - invErf * tanThetaI);  // newton step 2
            // compute the slope from the refined value
            slope.x = OIIO::fast_ierf(b);
        } else {
            // we are close enough already
            slope.x = invErf;
        }

        /* sample slope Y */
        slope.y = OIIO::fast_ierf(2.0f * randv - 1.0f);

        return slope;
    }
};


template<typename Distribution, int Refract>
struct Microfacet final : public BSDF, MicrofacetParams {
    Microfacet(const MicrofacetParams& params)
        : BSDF()
        , MicrofacetParams(params)
        , tf(U == Vec3(0) || xalpha == yalpha ? TangentFrame(N)
                                              : TangentFrame(N, U))
    {
    }
    Color3 get_albedo(const Vec3& wo) const override
    {
        if (Refract == 2)
            return Color3(1.0f);
        // FIXME: this heuristic is not particularly good, and looses energy
        // compared to the reference solution
        float fr = fresnel_dielectric(N.dot(wo), eta);
        return Color3(Refract ? 1 - fr : fr);
    }
    Sample eval(const Vec3& wo, const OSL::Vec3& wi) const override
    {
        const Vec3 wo_l = tf.tolocal(wo);
        const Vec3 wi_l = tf.tolocal(wi);
        if (Refract == 0 || Refract == 2) {
            if (wo_l.z > 0 && wi_l.z > 0) {
                const Vec3 m         = (wi_l + wo_l).normalize();
                const float D        = evalD(m);
                const float Lambda_o = evalLambda(wo_l);
                const float Lambda_i = evalLambda(wi_l);
                const float G2       = evalG2(Lambda_o, Lambda_i);
                const float G1       = evalG1(Lambda_o);

                const float Fr = fresnel_dielectric(m.dot(wo_l), eta);
                float pdf      = (G1 * D * 0.25f) / wo_l.z;
                float out      = G2 / G1;
                if (Refract == 2) {
                    pdf *= Fr;
                    return { wi, Color3(out), pdf, std::max(xalpha, yalpha) };
                } else {
                    return { wi, Color3(out * Fr), pdf,
                             std::max(xalpha, yalpha) };
                }
            }
        }
        if (Refract == 1 || Refract == 2) {
            if (wi_l.z < 0 && wo_l.z > 0.0f) {
                // compute half-vector of the refraction (eq. 16)
                Vec3 ht = -(eta * wi_l + wo_l);
                if (eta < 1.0f)
                    ht = -ht;
                Vec3 Ht = ht.normalize();
                // compute fresnel term
                const float cosHO = Ht.dot(wo_l);
                const float Ft    = 1.0f - fresnel_dielectric(cosHO, eta);
                if (Ft > 0) {  // skip work in case of TIR
                    const float cosHI = Ht.dot(wi_l);
                    // eq. 33: first we calculate D(m) with m=Ht:
                    const float cosThetaM = Ht.z;
                    if (cosThetaM <= 0.0f)
                        return {};
                    const float Dt       = evalD(Ht);
                    const float Lambda_o = evalLambda(wo_l);
                    const float Lambda_i = evalLambda(wi_l);
                    const float G2       = evalG2(Lambda_o, Lambda_i);
                    const float G1       = evalG1(Lambda_o);

                    // probability
                    float invHt2 = 1 / ht.dot(ht);
                    float pdf = (fabsf(cosHI * cosHO) * (eta * eta) * (G1 * Dt)
                                 * invHt2)
                                / wo_l.z;
                    float out = G2 / G1;
                    if (Refract == 2) {
                        pdf *= Ft;
                        return { wi, Color3(out), pdf,
                                 std::max(xalpha, yalpha) };
                    } else {
                        return { wi, Color3(out * Ft), pdf,
                                 std::max(xalpha, yalpha) };
                    }
                }
            }
        }
        return {};
    }

    Sample sample(const Vec3& wo, float rx, float ry, float rz) const override
    {
        const Vec3 wo_l   = tf.tolocal(wo);
        const float cosNO = wo_l.z;
        if (!(cosNO > 0))
            return {};
        const Vec3 m      = sampleMicronormal(wo_l, rx, ry);
        const float cosMO = m.dot(wo_l);
        const float F     = fresnel_dielectric(cosMO, eta);
        if (Refract == 0 || (Refract == 2 && rz < F)) {
            // measure fresnel to decide which lobe to sample
            const Vec3 wi_l      = (2.0f * cosMO) * m - wo_l;
            const float D        = evalD(m);
            const float Lambda_o = evalLambda(wo_l);
            const float Lambda_i = evalLambda(wi_l);

            const float G2 = evalG2(Lambda_o, Lambda_i);
            const float G1 = evalG1(Lambda_o);

            Vec3 wi = tf.toworld(wi_l);

            float pdf = (G1 * D * 0.25f) / cosNO;
            float out = G2 / G1;
            if (Refract == 2) {
                pdf *= F;
                return { wi, Color3(out), pdf, std::max(xalpha, yalpha) };
            } else
                return { wi, Color3(F * out), pdf, std::max(xalpha, yalpha) };
        } else {
            const Vec3 M = tf.toworld(m);
            Vec3 wi;
            float Ft             = fresnel_refraction(-wo, M, eta, wi);
            const Vec3 wi_l      = tf.tolocal(wi);
            const float cosHO    = m.dot(wo_l);
            const float cosHI    = m.dot(wi_l);
            const float D        = evalD(m);
            const float Lambda_o = evalLambda(wo_l);
            const float Lambda_i = evalLambda(wi_l);

            const float G2 = evalG2(Lambda_o, Lambda_i);
            const float G1 = evalG1(Lambda_o);

            const Vec3 ht      = -(eta * wi_l + wo_l);
            const float invHt2 = 1.0f / ht.dot(ht);

            float pdf = (fabsf(cosHI * cosHO) * (eta * eta) * (G1 * D) * invHt2)
                        / fabsf(wo_l.z);
            float out = G2 / G1;
            if (Refract == 2) {
                pdf *= Ft;
                return { wi, Color3(out), pdf, std::max(xalpha, yalpha) };
            } else
                return { wi, Color3(Ft * out), pdf, std::max(xalpha, yalpha) };
        }
        return {};
    }

private:
    static float SQR(float x) { return x * x; }

    float evalLambda(const Vec3 w) const
    {
        float cosTheta2 = SQR(w.z);
        /* Have these two multiplied by sinTheta^2 for convenience */
        float cosPhi2st2 = SQR(w.x * xalpha);
        float sinPhi2st2 = SQR(w.y * yalpha);
        return Distribution::Lambda(cosTheta2 / (cosPhi2st2 + sinPhi2st2));
    }

    static float evalG2(float Lambda_i, float Lambda_o)
    {
        // correlated masking-shadowing
        return 1 / (Lambda_i + Lambda_o + 1);
    }

    static float evalG1(float Lambda_v) { return 1 / (Lambda_v + 1); }

    float evalD(const Vec3 Hr) const
    {
        float cosThetaM = Hr.z;
        if (cosThetaM > 0) {
            /* Have these two multiplied by sinThetaM2 for convenience */
            float cosPhi2st2 = SQR(Hr.x / xalpha);
            float sinPhi2st2 = SQR(Hr.y / yalpha);
            float cosThetaM2 = SQR(cosThetaM);
            float cosThetaM4 = SQR(cosThetaM2);

            float tanThetaM2 = (cosPhi2st2 + sinPhi2st2) / cosThetaM2;

            return Distribution::F(tanThetaM2) / (xalpha * yalpha * cosThetaM4);
        }
        return 0;
    }

    Vec3 sampleMicronormal(const Vec3 wo, float randu, float randv) const
    {
        /* Project wo and stretch by alpha values */
        Vec3 swo = wo;
        swo.x *= xalpha;
        swo.y *= yalpha;
        swo = swo.normalize();

        // figure out angles for the incoming vector
        float cos_theta = std::max(swo.z, 0.0f);
        float cos_phi   = 1;
        float sin_phi   = 0;
        /* Normal incidence special case gets phi 0 */
        if (cos_theta < 0.99999f) {
            float invnorm = 1 / sqrtf(SQR(swo.x) + SQR(swo.y));
            cos_phi       = swo.x * invnorm;
            sin_phi       = swo.y * invnorm;
        }

        Vec2 slope = Distribution::sampleSlope(cos_theta, randu, randv);

        /* Rotate and unstretch slopes */
        Vec2 s(cos_phi * slope.x - sin_phi * slope.y,
               sin_phi * slope.x + cos_phi * slope.y);
        s.x *= xalpha;
        s.y *= yalpha;

        float mlen = sqrtf(s.x * s.x + s.y * s.y + 1);
        Vec3 m(fabsf(s.x) < mlen ? -s.x / mlen : 1.0f,
               fabsf(s.y) < mlen ? -s.y / mlen : 1.0f, 1.0f / mlen);
        return m;
    }

    TangentFrame tf;
};

typedef Microfacet<GGXDist, 0> MicrofacetGGXRefl;
typedef Microfacet<GGXDist, 1> MicrofacetGGXRefr;
typedef Microfacet<GGXDist, 2> MicrofacetGGXBoth;
typedef Microfacet<BeckmannDist, 0> MicrofacetBeckmannRefl;
typedef Microfacet<BeckmannDist, 1> MicrofacetBeckmannRefr;
typedef Microfacet<BeckmannDist, 2> MicrofacetBeckmannBoth;


// We use the CRTP to inherit the parameters because each MaterialX closure uses a different set of parameters
template<typename MxMicrofacetParams, typename Distribution,
         bool EnableTransmissionLobe>
struct MxMicrofacet final : public BSDF, MxMicrofacetParams {
    MxMicrofacet(const MxMicrofacetParams& params, float refraction_ior)
        : BSDF()
        , MxMicrofacetParams(params)
        , tf(MxMicrofacetParams::U == Vec3(0)
                     || MxMicrofacetParams::roughness_x
                            == MxMicrofacetParams::roughness_y
                 ? TangentFrame(MxMicrofacetParams::N)
                 : TangentFrame(MxMicrofacetParams::N, MxMicrofacetParams::U))
        , refraction_ior(refraction_ior)
    {
    }

    float get_fresnel_angle(float cos_theta) const
    {
        if (EnableTransmissionLobe && refraction_ior < 1) {
            // handle TIR if we are on the backside
            const float cos_theta_t2 = 1.0f
                                       - (1.0f - cos_theta * cos_theta)
                                             * refraction_ior * refraction_ior;
            const float cos_theta_t = cos_theta_t2 > 0 ? sqrtf(cos_theta_t2)
                                                       : 0.0f;
            return cos_theta_t;
        }
        return cos_theta;
    }

    Color3 get_albedo(const Vec3& wo) const override
    {
        // if transmission is enabled, punt on
        if (EnableTransmissionLobe)
            return Color3(1.0f);
        // FIXME: this heuristic is not particularly good, and looses energy
        // compared to the reference solution

        return MxMicrofacetParams::evalR(
            get_fresnel_angle(MxMicrofacetParams::N.dot(wo)));
    }

    Sample eval(const Vec3& wo, const OSL::Vec3& wi) const override
    {
        const Vec3 wo_l = tf.tolocal(wo);
        const Vec3 wi_l = tf.tolocal(wi);

        // handle reflection lobe
        if (wo_l.z > 0 && wi_l.z > 0) {
            const Vec3 m         = (wi_l + wo_l).normalize();
            const float D        = evalD(m);
            const float Lambda_o = evalLambda(wo_l);
            const float Lambda_i = evalLambda(wi_l);
            const float G2       = evalG2(Lambda_o, Lambda_i);
            const float G1       = evalG1(Lambda_o);

            const float cosHO  = m.dot(wo_l);
            const float cosHOf = get_fresnel_angle(cosHO);
            const Color3 Fr    = MxMicrofacetParams::evalR(cosHOf);
            float pdf          = (G1 * D * 0.25f) / wo_l.z;
            float out          = G2 / G1;
            if (EnableTransmissionLobe) {
                const Color3 Ft      = MxMicrofacetParams::evalT(cosHOf);
                const float weight_t = Ft.x + Ft.y + Ft.z;
                const float weight_r = Fr.x + Fr.y + Fr.z;
                const float probT    = weight_t / (weight_t + weight_r + 1e-6f);
                const float probR    = 1.0f - probT;
                out /= probR;
                pdf *= probR;
            }
            return { wi, Fr * out, pdf,
                     std::max(MxMicrofacetParams::roughness_x,
                              MxMicrofacetParams::roughness_y) };
        }

        // handle refraction lobe
        if (EnableTransmissionLobe && wi_l.z < 0 && wo_l.z > 0.0f) {
            // compute half-vector of the refraction (eq. 16)
            Vec3 ht = -(refraction_ior * wi_l + wo_l);
            if (refraction_ior < 1.0f)
                ht = -ht;
            Vec3 Ht = ht.normalize();
            // compute fresnel term
            const float cosHO  = Ht.dot(wo_l);
            const float cosHOf = get_fresnel_angle(cosHO);
            const Color3 Ft    = MxMicrofacetParams::evalR(cosHOf);
            if (Ft.x + Ft.y + Ft.z > 0) {  // skip work in case of TIR
                const float cosHI = Ht.dot(wi_l);
                // eq. 33: first we calculate D(m) with m=Ht:
                const float cosThetaM = Ht.z;
                if (cosThetaM <= 0.0f)
                    return {};
                const float Dt       = evalD(Ht);
                const float Lambda_o = evalLambda(wo_l);
                const float Lambda_i = evalLambda(wi_l);
                const float G2       = evalG2(Lambda_o, Lambda_i);
                const float G1       = evalG1(Lambda_o);

                // probability
                float invHt2 = 1 / ht.dot(ht);
                float pdf    = (fabsf(cosHI * cosHO)
                             * (refraction_ior * refraction_ior) * (G1 * Dt)
                             * invHt2)
                            / wo_l.z;
                float out = G2 / G1;
                // figure out lobe probabilities
                const Color3 Fr      = MxMicrofacetParams::evalR(cosHOf);
                const float weight_t = Ft.x + Ft.y + Ft.z;
                const float weight_r = Fr.x + Fr.y + Fr.z;
                const float probT    = weight_t / (weight_t + weight_r + 1e-6f);
                pdf *= probT;
                return { wi, Ft * out / probT, pdf,
                         std::max(MxMicrofacetParams::roughness_x,
                                  MxMicrofacetParams::roughness_y) };
            }
        }

        return {};
    }


    Sample sample(const Vec3& wo, float rx, float ry, float rz) const override
    {
        const Vec3 wo_l   = tf.tolocal(wo);
        const float cosNO = wo_l.z;
        if (!(cosNO > 0))
            return {};
        const Vec3 m       = sampleMicronormal(wo_l, rx, ry);
        const float cosMO  = m.dot(wo_l);
        const float cosMOf = get_fresnel_angle(cosMO);
        {
            // reflection lobe
            const Vec3 wi_l      = (2.0f * cosMO) * m - wo_l;
            const float D        = evalD(m);
            const float Lambda_o = evalLambda(wo_l);
            const float Lambda_i = evalLambda(wi_l);

            const float G2 = evalG2(Lambda_o, Lambda_i);
            const float G1 = evalG1(Lambda_o);

            const Color3 Fr = MxMicrofacetParams::evalR(cosMOf);

            Vec3 wi = tf.toworld(wi_l);

            float pdf = (G1 * D * 0.25f) / cosNO;
            float out = G2 / G1;

            if (EnableTransmissionLobe) {
                const Color3 Ft      = MxMicrofacetParams::evalT(cosMOf);
                const float weight_t = Ft.x + Ft.y + Ft.z;
                const float weight_r = Fr.x + Fr.y + Fr.z;
                const float probT    = weight_t / (weight_t + weight_r + 1e-6f);
                if (rz < probT) {
                    // switch to transmitted vector instead
                    const Vec3 M = tf.toworld(m);
                    fresnel_refraction(-wo, M, refraction_ior, wi);
                    const Vec3 wi_l      = tf.tolocal(wi);
                    const float cosHO    = m.dot(wo_l);
                    const float cosHI    = m.dot(wi_l);
                    const float D        = evalD(m);
                    const float Lambda_o = evalLambda(wo_l);
                    const float Lambda_i = evalLambda(wi_l);

                    const float G2 = evalG2(Lambda_o, Lambda_i);
                    const float G1 = evalG1(Lambda_o);

                    const Vec3 ht      = -(refraction_ior * wi_l + wo_l);
                    const float invHt2 = 1.0f / ht.dot(ht);

                    pdf = (fabsf(cosHI * cosHO)
                           * (refraction_ior * refraction_ior) * (G1 * D)
                           * invHt2)
                          / fabsf(wo_l.z);

                    float out = G2 / G1;

                    pdf *= probT;
                    return { wi, Ft * out / probT, pdf,
                             std::max(MxMicrofacetParams::roughness_x,
                                      MxMicrofacetParams::roughness_y) };
                } else {
                    pdf *= probT;
                    return { wi, Fr * out / (1.0f - probT), pdf,
                             std::max(MxMicrofacetParams::roughness_x,
                                      MxMicrofacetParams::roughness_y) };
                }
            }

            //
            return { wi, Fr * out, pdf,
                     std::max(MxMicrofacetParams::roughness_x,
                              MxMicrofacetParams::roughness_y) };
        }
    }

private:
    static float SQR(float x) { return x * x; }

    float evalLambda(const Vec3 w) const
    {
        float cosTheta2 = SQR(w.z);
        /* Have these two multiplied by sinTheta^2 for convenience */
        float cosPhi2st2 = SQR(w.x * MxMicrofacetParams::roughness_x);
        float sinPhi2st2 = SQR(w.y * MxMicrofacetParams::roughness_y);
        return Distribution::Lambda(cosTheta2 / (cosPhi2st2 + sinPhi2st2));
    }

    static float evalG2(float Lambda_i, float Lambda_o)
    {
        // correlated masking-shadowing
        return 1 / (Lambda_i + Lambda_o + 1);
    }

    static float evalG1(float Lambda_v) { return 1 / (Lambda_v + 1); }

    float evalD(const Vec3 Hr) const
    {
        float cosThetaM = Hr.z;
        if (cosThetaM > 0) {
            /* Have these two multiplied by sinThetaM2 for convenience */
            float cosPhi2st2 = SQR(Hr.x / MxMicrofacetParams::roughness_x);
            float sinPhi2st2 = SQR(Hr.y / MxMicrofacetParams::roughness_y);
            float cosThetaM2 = SQR(cosThetaM);
            float cosThetaM4 = SQR(cosThetaM2);

            float tanThetaM2 = (cosPhi2st2 + sinPhi2st2) / cosThetaM2;

            return Distribution::F(tanThetaM2)
                   / (MxMicrofacetParams::roughness_x
                      * MxMicrofacetParams::roughness_y * cosThetaM4);
        }
        return 0;
    }

    Vec3 sampleMicronormal(const Vec3 wo, float randu, float randv) const
    {
        /* Project wo and stretch by alpha values */
        Vec3 swo = wo;
        swo.x *= MxMicrofacetParams::roughness_x;
        swo.y *= MxMicrofacetParams::roughness_y;
        swo = swo.normalize();

        // figure out angles for the incoming vector
        float cos_theta = std::max(swo.z, 0.0f);
        float cos_phi   = 1;
        float sin_phi   = 0;
        /* Normal incidence special case gets phi 0 */
        if (cos_theta < 0.99999f) {
            float invnorm = 1 / sqrtf(SQR(swo.x) + SQR(swo.y));
            cos_phi       = swo.x * invnorm;
            sin_phi       = swo.y * invnorm;
        }

        Vec2 slope = Distribution::sampleSlope(cos_theta, randu, randv);

        /* Rotate and unstretch slopes */
        Vec2 s(cos_phi * slope.x - sin_phi * slope.y,
               sin_phi * slope.x + cos_phi * slope.y);
        s.x *= MxMicrofacetParams::roughness_x;
        s.y *= MxMicrofacetParams::roughness_y;

        float mlen = sqrtf(s.x * s.x + s.y * s.y + 1);
        Vec3 m(fabsf(s.x) < mlen ? -s.x / mlen : 1.0f,
               fabsf(s.y) < mlen ? -s.y / mlen : 1.0f, 1.0f / mlen);
        return m;
    }

    TangentFrame tf;
    float refraction_ior;
};

struct Reflection final : public BSDF, ReflectionParams {
    Reflection(const ReflectionParams& params)
        : BSDF(), ReflectionParams(params)
    {
    }
    Color3 get_albedo(const Vec3& wo) const override
    {
        float cosNO = N.dot(wo);
        if (cosNO > 0)
            return Color3(fresnel_dielectric(cosNO, eta));
        return Color3(1);
    }
    Sample eval(const Vec3& /*wo*/, const OSL::Vec3& /*wi*/) const override
    {
        return {};
    }
    Sample sample(const Vec3& wo, float /*rx*/, float /*ry*/,
                  float /*rz*/) const override
    {
        // only one direction is possible
        float cosNO = dot(N, wo);
        if (cosNO > 0) {
            Vec3 wi   = (2 * cosNO) * N - wo;
            float pdf = std::numeric_limits<float>::infinity();
            return { wi, Color3(fresnel_dielectric(cosNO, eta)), pdf, 0 };
        }
        return {};
    }
};

struct Refraction final : public BSDF, RefractionParams {
    Refraction(const RefractionParams& params)
        : BSDF(), RefractionParams(params)
    {
    }
    Color3 get_albedo(const Vec3& wo) const override
    {
        float cosNO = N.dot(wo);
        return Color3(1 - fresnel_dielectric(cosNO, eta));
    }
    Sample eval(const Vec3& /*wo*/, const OSL::Vec3& /*wi*/) const override
    {
        return {};
    }
    Sample sample(const Vec3& wo, float /*rx*/, float /*ry*/,
                  float /*rz*/) const override
    {
        float pdf = std::numeric_limits<float>::infinity();
        Vec3 wi;
        float Ft = fresnel_refraction(-wo, N, eta, wi);
        return { wi, Color3(Ft), pdf, 0 };
    }
};

struct Transparent final : public BSDF {
    Transparent() : BSDF() {}
    Sample eval(const Vec3& /*wo*/, const Vec3& /*wi*/) const override
    {
        return {};
    }
    Sample sample(const Vec3& wo, float /*rx*/, float /*ry*/,
                  float /*rz*/) const override
    {
        Vec3 wi   = -wo;
        float pdf = std::numeric_limits<float>::infinity();
        return { wi, Color3(1.0f), pdf, 0 };
    }
};

struct MxBurleyDiffuse final : public BSDF, MxBurleyDiffuseParams {
    MxBurleyDiffuse(const MxBurleyDiffuseParams& params)
        : BSDF(), MxBurleyDiffuseParams(params)
    {
    }

    Color3 get_albedo(const Vec3& wo) const override { return albedo; }

    Sample eval(const Vec3& wo, const Vec3& wi) const override
    {
        const Vec3 L = wi, V = wo;
        const Vec3 H = (L + V).normalize();
        float LdotH  = clamp(dot(L, H), 0.0f, 1.0f);
        float NdotV  = clamp(dot(N, V), 0.0f, 1.0f);
        float NdotL  = clamp(dot(N, L), 0.0f, 1.0f);
        float F90    = 0.5f + (2.0f * roughness * LdotH * LdotH);
        float refL   = fresnel_schlick(NdotL, 1.0f, F90);
        float refV   = fresnel_schlick(NdotV, 1.0f, F90);
        float pdf    = NdotL * float(M_1_PI);
        return { wi, albedo * refL * refV, pdf, 1.0f };
    }

    Sample sample(const Vec3& wo, float rx, float ry, float rz) const override
    {
        Vec3 out_dir;
        float pdf;
        Sampling::sample_cosine_hemisphere(N, rx, ry, out_dir, pdf);
        return eval(wo, out_dir);
    }
};

struct MxSheen final : public BSDF, MxSheenParams {
    MxSheen(const MxSheenParams& params) : BSDF(), MxSheenParams(params) {}

    Color3 get_albedo(const Vec3& wo) const override
    {
        const float NdotV = clamp(N.dot(wo), 0.0f, 1.0f);
        // Rational fit from the Material X project
        // Ref: https://github.com/AcademySoftwareFoundation/MaterialX/blob/main/libraries/pbrlib/genglsl/lib/mx_microfacet_sheen.glsl
        const Vec2 r = Vec2(13.67300f, 1.0f)
                       + Vec2(-68.78018f, 61.57746f) * NdotV
                       + Vec2(799.08825f, 442.78211f) * roughness
                       + Vec2(-905.00061f, 2597.49308f) * NdotV * roughness
                       + Vec2(60.28956f, 121.81241f) * NdotV * NdotV
                       + Vec2(1086.96473f, 3045.55075f) * roughness * roughness;
        return clamp(albedo * (r.x / r.y), 0.0f, 1.0f);
    }

    Sample eval(const Vec3& wo, const Vec3& wi) const override
    {
        const Vec3 L = wi, V = wo;
        const Vec3 H       = (L + V).normalize();
        float NdotV        = clamp(dot(N, V), 0.0f, 1.0f);
        float NdotL        = clamp(dot(N, L), 0.0f, 1.0f);
        float NdotH        = clamp(dot(N, H), 0.0f, 1.0f);
        float invRoughness = 1.0f / std::max(roughness, 0.005f);

        float D = (2.0f + invRoughness)
                  * powf(1.0f - NdotH * NdotH, invRoughness * 0.5f)
                  / float(2 * M_PI);
        float pdf = float(0.5 * M_1_PI);
        // NOTE: sheen closure has no fresnel/masking
        return { wi,
                 Color3(float(2 * M_PI) * NdotL * albedo * D
                        / (4.0f * (NdotL + NdotV - NdotL * NdotV))),
                 pdf, 1.0f };
    }

    Sample sample(const Vec3& wo, float rx, float ry, float rz) const override
    {
        Vec3 out_dir;
        float pdf;
        Sampling::sample_uniform_hemisphere(N, rx, ry, out_dir, pdf);
        return eval(wo, out_dir);
    }
};

Color3
evaluate_layer_opacity(const OSL::ShaderGlobals& sg,
                       const ClosureColor* closure)
{
    // Null closure, the layer is fully transparent
    if (closure == nullptr)
        return Color3(0);

    switch (closure->id) {
    case ClosureColor::MUL:
        return closure->as_mul()->weight
               * evaluate_layer_opacity(sg, closure->as_mul()->closure);
    case ClosureColor::ADD:
        return evaluate_layer_opacity(sg, closure->as_add()->closureA)
               + evaluate_layer_opacity(sg, closure->as_add()->closureB);
    default: {
        const ClosureComponent* comp = closure->as_comp();
        Color3 w                     = comp->w;
        switch (comp->id) {
        case MX_LAYER_ID: {
            const MxLayerParams* srcparams = comp->as<MxLayerParams>();
            return w
                   * (evaluate_layer_opacity(sg, srcparams->top)
                      + evaluate_layer_opacity(sg, srcparams->base));
        }
        case REFLECTION_ID:
        case FRESNEL_REFLECTION_ID: {
            Reflection bsdf(*comp->as<ReflectionParams>());
            return w * bsdf.get_albedo(-sg.I);
        }
        case MX_DIELECTRIC_ID: {
            const MxDielectricParams& params = *comp->as<MxDielectricParams>();
            // Transmissive dielectrics are opaque
            if (!is_black(params.transmission_tint))
                return Color3(1);
            MxMicrofacet<MxDielectricParams, GGXDist, false> mf(params, 1.0f);
            return w * mf.get_albedo(-sg.I);
        }
        case MX_GENERALIZED_SCHLICK_ID: {
            const MxGeneralizedSchlickParams& params
                = *comp->as<MxGeneralizedSchlickParams>();
            // Transmissive dielectrics are opaque
            if (!is_black(params.transmission_tint))
                return Color3(1);
            MxMicrofacet<MxGeneralizedSchlickParams, GGXDist, false> mf(params,
                                                                        1.0f);
            return w * mf.get_albedo(-sg.I);
        }
        case MX_SHEEN_ID: {
            MxSheen bsdf(*comp->as<MxSheenParams>());
            return w * bsdf.get_albedo(-sg.I);
        }
        default:  // Assume unhandled BSDFs are opaque
            return Color3(1);
        }
    }
    }
    OSL_ASSERT(false && "Layer opacity evaluation failed");
    return Color3(0);
}

void
process_medium_closure(const OSL::ShaderGlobals& sg, ShadingResult& result,
                       const ClosureColor* closure, const Color3& w)
{
    if (!closure)
        return;
    switch (closure->id) {
    case ClosureColor::MUL: {
        process_medium_closure(sg, result, closure->as_mul()->closure,
                               w * closure->as_mul()->weight);
        break;
    }
    case ClosureColor::ADD: {
        process_medium_closure(sg, result, closure->as_add()->closureA, w);
        process_medium_closure(sg, result, closure->as_add()->closureB, w);
        break;
    }
    case MX_LAYER_ID: {
        const ClosureComponent* comp = closure->as_comp();
        const MxLayerParams* params  = comp->as<MxLayerParams>();
        Color3 base_w
            = w
              * (Color3(1)
                 - clamp(evaluate_layer_opacity(sg, params->top), 0.f, 1.f));
        process_medium_closure(sg, result, params->top, w);
        process_medium_closure(sg, result, params->base, base_w);
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
        break;
    }
    }
}

// recursively walk through the closure tree, creating bsdfs as we go
void
process_bsdf_closure(const OSL::ShaderGlobals& sg, ShadingResult& result,
                     const ClosureColor* closure, const Color3& w,
                     bool light_only)
{
    static const ustring u_ggx("ggx");
    static const ustring u_beckmann("beckmann");
    static const ustring u_default("default");
    if (!closure)
        return;
    switch (closure->id) {
    case ClosureColor::MUL: {
        Color3 cw = w * closure->as_mul()->weight;
        process_bsdf_closure(sg, result, closure->as_mul()->closure, cw,
                             light_only);
        break;
    }
    case ClosureColor::ADD: {
        process_bsdf_closure(sg, result, closure->as_add()->closureA, w,
                             light_only);
        process_bsdf_closure(sg, result, closure->as_add()->closureB, w,
                             light_only);
        break;
    }
    default: {
        const ClosureComponent* comp = closure->as_comp();
        Color3 cw                    = w * comp->w;
        if (comp->id == EMISSION_ID)
            result.Le += cw;
        else if (comp->id == MX_UNIFORM_EDF_ID)
            result.Le += cw * comp->as<MxUniformEdfParams>()->emittance;
        else if (!light_only) {
            bool ok = false;
            switch (comp->id) {
            case DIFFUSE_ID:
                ok = result.bsdf.add_bsdf<Diffuse<0>>(
                    cw, *comp->as<DiffuseParams>());
                break;
            case OREN_NAYAR_ID:
                ok = result.bsdf.add_bsdf<OrenNayar>(
                    cw, *comp->as<OrenNayarParams>());
                break;
            case TRANSLUCENT_ID:
                ok = result.bsdf.add_bsdf<Diffuse<1>>(
                    cw, *comp->as<DiffuseParams>());
                break;
            case PHONG_ID:
                ok = result.bsdf.add_bsdf<Phong>(cw, *comp->as<PhongParams>());
                break;
            case WARD_ID:
                ok = result.bsdf.add_bsdf<Ward>(cw, *comp->as<WardParams>());
                break;
            case MICROFACET_ID: {
                const MicrofacetParams* mp = comp->as<MicrofacetParams>();
                if (mp->dist == u_ggx) {
                    switch (mp->refract) {
                    case 0:
                        ok = result.bsdf.add_bsdf<MicrofacetGGXRefl>(cw, *mp);
                        break;
                    case 1:
                        ok = result.bsdf.add_bsdf<MicrofacetGGXRefr>(cw, *mp);
                        break;
                    case 2:
                        ok = result.bsdf.add_bsdf<MicrofacetGGXBoth>(cw, *mp);
                        break;
                    }
                } else if (mp->dist == u_beckmann || mp->dist == u_default) {
                    switch (mp->refract) {
                    case 0:
                        ok = result.bsdf.add_bsdf<MicrofacetBeckmannRefl>(cw,
                                                                          *mp);
                        break;
                    case 1:
                        ok = result.bsdf.add_bsdf<MicrofacetBeckmannRefr>(cw,
                                                                          *mp);
                        break;
                    case 2:
                        ok = result.bsdf.add_bsdf<MicrofacetBeckmannBoth>(cw,
                                                                          *mp);
                        break;
                    }
                }
                break;
            }
            case REFLECTION_ID:
            case FRESNEL_REFLECTION_ID:
                ok = result.bsdf.add_bsdf<Reflection>(
                    cw, *comp->as<ReflectionParams>());
                break;
            case REFRACTION_ID:
                ok = result.bsdf.add_bsdf<Refraction>(
                    cw, *comp->as<RefractionParams>());
                break;
            case TRANSPARENT_ID:
                ok = result.bsdf.add_bsdf<Transparent>(cw);
                break;
            case MX_OREN_NAYAR_DIFFUSE_ID: {
                const MxOrenNayarDiffuseParams* srcparams
                    = comp->as<MxOrenNayarDiffuseParams>();
                if (srcparams->energy_compensation) {
                    // energy compensation handled by its own BSDF
                    ok = result.bsdf.add_bsdf<EnergyCompensatedOrenNayar>(
                        cw, *srcparams);
                } else {
                    // translate MaterialX parameters into existing closure
                    OrenNayarParams params = {};
                    params.N               = srcparams->N;
                    params.sigma           = srcparams->roughness;
                    ok = result.bsdf.add_bsdf<OrenNayar>(cw * srcparams->albedo,
                                                         params);
                }
                break;
            }
            case MX_BURLEY_DIFFUSE_ID: {
                const MxBurleyDiffuseParams& params
                    = *comp->as<MxBurleyDiffuseParams>();
                ok = result.bsdf.add_bsdf<MxBurleyDiffuse>(cw, params);
                break;
            }
            case MX_DIELECTRIC_ID: {
                const MxDielectricParams& params
                    = *comp->as<MxDielectricParams>();
                if (is_black(params.transmission_tint))
                    ok = result.bsdf.add_bsdf<
                        MxMicrofacet<MxDielectricParams, GGXDist, false>>(
                        cw, params, 1.0f);
                else
                    ok = result.bsdf.add_bsdf<
                        MxMicrofacet<MxDielectricParams, GGXDist, true>>(
                        cw, params, result.refraction_ior);
                break;
            }
            case MX_CONDUCTOR_ID: {
                const MxConductorParams& params = *comp->as<MxConductorParams>();
                ok = result.bsdf.add_bsdf<
                    MxMicrofacet<MxConductorParams, GGXDist, false>>(cw, params,
                                                                     1.0f);
                break;
            };
            case MX_GENERALIZED_SCHLICK_ID: {
                const MxGeneralizedSchlickParams& params
                    = *comp->as<MxGeneralizedSchlickParams>();
                if (is_black(params.transmission_tint))
                    ok = result.bsdf.add_bsdf<MxMicrofacet<
                        MxGeneralizedSchlickParams, GGXDist, false>>(cw, params,
                                                                     1.0f);
                else
                    ok = result.bsdf.add_bsdf<
                        MxMicrofacet<MxGeneralizedSchlickParams, GGXDist, true>>(
                        cw, params, result.refraction_ior);
                break;
            };
            case MX_TRANSLUCENT_ID: {
                const MxTranslucentParams* srcparams
                    = comp->as<MxTranslucentParams>();
                DiffuseParams params = {};
                params.N             = srcparams->N;
                ok = result.bsdf.add_bsdf<Diffuse<1>>(cw * srcparams->albedo,
                                                      params);
                break;
            }
            case MX_TRANSPARENT_ID: {
                ok = result.bsdf.add_bsdf<Transparent>(cw);
                break;
            }
            case MX_SUBSURFACE_ID: {
                // TODO: implement BSSRDF support?
                const MxSubsurfaceParams* srcparams
                    = comp->as<MxSubsurfaceParams>();
                DiffuseParams params = {};
                params.N             = srcparams->N;
                ok = result.bsdf.add_bsdf<Diffuse<0>>(cw * srcparams->albedo,
                                                      params);
                break;
            }
            case MX_SHEEN_ID: {
                const MxSheenParams& params = *comp->as<MxSheenParams>();
                ok = result.bsdf.add_bsdf<MxSheen>(cw, params);
                break;
            }
            case MX_LAYER_ID: {
                const MxLayerParams* srcparams = comp->as<MxLayerParams>();
                Color3 base_w
                    = w
                      * (Color3(1, 1, 1)
                         - clamp(evaluate_layer_opacity(sg, srcparams->top),
                                 0.f, 1.f));
                process_bsdf_closure(sg, result, srcparams->top, w, light_only);
                if (!is_black(base_w))
                    process_bsdf_closure(sg, result, srcparams->base, base_w,
                                         light_only);
                ok = true;
                break;
            }
            case MX_ANISOTROPIC_VDF_ID:
            case MX_MEDIUM_VDF_ID: {
                // already processed by process_medium_closure
                ok = true;
                break;
            }
            }
            OSL_ASSERT(ok && "Invalid closure invoked in surface shader");
        }
        break;
    }
    }
}

}  // anonymous namespace

OSL_NAMESPACE_ENTER

void
process_closure(const OSL::ShaderGlobals& sg, ShadingResult& result,
                const ClosureColor* Ci, bool light_only)
{
    if (!light_only)
        process_medium_closure(sg, result, Ci, Color3(1));
    process_bsdf_closure(sg, result, Ci, Color3(1), light_only);
}

Vec3
process_background_closure(const ClosureColor* closure)
{
    if (!closure)
        return Vec3(0, 0, 0);
    switch (closure->id) {
    case ClosureColor::MUL: {
        return closure->as_mul()->weight
               * process_background_closure(closure->as_mul()->closure);
    }
    case ClosureColor::ADD: {
        return process_background_closure(closure->as_add()->closureA)
               + process_background_closure(closure->as_add()->closureB);
    }
    case BACKGROUND_ID: {
        return closure->as_comp()->w;
    }
    }
    // should never happen
    OSL_ASSERT(false && "Invalid closure invoked in background shader");
    return Vec3(0, 0, 0);
}


OSL_NAMESPACE_EXIT
