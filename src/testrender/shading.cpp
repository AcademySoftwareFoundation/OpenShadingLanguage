// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include "shading.h"
#include <OSL/genclosure.h>
#include "optics.h"
#include "sampling.h"

#include <BSDL/MTX/bsdf_conductor_impl.h>
#include <BSDL/MTX/bsdf_dielectric_impl.h>
#include <BSDL/SPI/bsdf_thinlayer_impl.h>
#include <BSDL/spectrum_impl.h>

using namespace OSL;


#ifndef __CUDACC__
using ShaderGlobalsType = OSL::ShaderGlobals;
#else
using ShaderGlobalsType = OSL_CUDA::ShaderGlobals;
#endif


namespace {  // anonymous namespace

using OIIO::clamp;
using OSL::dot;

OSL_HOSTDEVICE Color3
clamp(const Color3& c, float min, float max)
{
    return Color3(clamp(c.x, min, max), clamp(c.y, min, max),
                  clamp(c.z, min, max));
}

OSL_HOSTDEVICE bool
is_black(const Color3& c)
{
    return c.x == 0 && c.y == 0 && c.z == 0;
}

}  // anonymous namespace

OSL_NAMESPACE_BEGIN

// BSDL expects this minimum functionality. We could put it in BSDF but
// the roughness() method interferes with other BSDFs
struct BSDLLobe : public BSDF {
    template<typename LOBE>
    OSL_HOSTDEVICE BSDLLobe(LOBE* lobe, float rough, float l0, bool tr)
        : BSDF(lobe), m_roughness(rough)
    {
    }

    OSL_HOSTDEVICE void set_roughness(float r) { m_roughness = r; }
    OSL_HOSTDEVICE float roughness() const { return m_roughness; }

    float m_roughness;
};

// This is the thin wrapper to insert spi::ThinLayerLobe into testrender
struct SpiThinLayer : public bsdl::spi::ThinLayerLobe<BSDLLobe> {
    using Base = bsdl::spi::ThinLayerLobe<BSDLLobe>;

    static constexpr int closureid() { return SPI_THINLAYER; }

    OSL_HOSTDEVICE SpiThinLayer(const Data& data, const Vec3& wo,
                                float path_roughness)
        : Base(this,
               bsdl::BsdfGlobals(wo,
                                 data.N,  // Nf
                                 data.N,  // Ngf
                                 false,   // backfacing
                                 path_roughness,
                                 1.0f,  // outer_ior
                                 0),    // hero wavelength off
               data)
    {
    }

    OSL_HOSTDEVICE BSDF::Sample eval(const Vec3& wo, const Vec3& wi) const
    {
        bsdl::Sample s = Base::eval_impl(Base::frame.local(wo),
                                         Base::frame.local(wi), true, true);
        return { wi, s.weight.toRGB(0), s.pdf, s.roughness };
    }
    OSL_HOSTDEVICE BSDF::Sample sample(const Vec3& wo, float rx, float ry,
                                       float rz) const
    {
        bsdl::Sample s = Base::sample_impl(Base::frame.local(wo),
                                           { rx, ry, rz }, true, true);
        return { Base::frame.world(s.wi), s.weight.toRGB(0), s.pdf,
                 s.roughness };
    }
};

// This is the thin wrapper to insert mtx::ConductorLobe into testrender
struct MxConductor : public bsdl::mtx::ConductorLobe<BSDLLobe> {
    using Base = bsdl::mtx::ConductorLobe<BSDLLobe>;

    static constexpr int closureid() { return MX_CONDUCTOR_ID; }

    OSL_HOSTDEVICE MxConductor(const Data& data, const Vec3& wo,
                               float path_roughness)
        : Base(this,
               bsdl::BsdfGlobals(wo,
                                 data.N,  // Nf
                                 data.N,  // Ngf
                                 false,   // backfacing
                                 path_roughness,
                                 1.0f,  // outer_ior
                                 0),    // hero wavelength off
               data)
    {
    }

    OSL_HOSTDEVICE BSDF::Sample eval(const Vec3& wo, const Vec3& wi) const
    {
        bsdl::Sample s = Base::eval_impl(Base::frame.local(wo),
                                         Base::frame.local(wi));
        return { wi, s.weight.toRGB(0), s.pdf, s.roughness };
    }
    OSL_HOSTDEVICE BSDF::Sample sample(const Vec3& wo, float rx, float ry,
                                       float rz) const
    {
        bsdl::Sample s = Base::sample_impl(Base::frame.local(wo),
                                           { rx, ry, rz });
        return { Base::frame.world(s.wi), s.weight.toRGB(0), s.pdf,
                 s.roughness };
    }
};

// This is the thin wrapper to insert mtx::ConductorLobe into testrender
struct MxDielectric : public bsdl::mtx::DielectricLobe<BSDLLobe> {
    using Base = bsdl::mtx::DielectricLobe<BSDLLobe>;

    static constexpr int closureid() { return MX_DIELECTRIC_ID; }

    OSL_HOSTDEVICE MxDielectric(const Data& data, const Vec3& wo,
                                bool backfacing, float path_roughness)
        : Base(this,
               bsdl::BsdfGlobals(wo,
                                 data.N,  // Nf
                                 data.N,  // Ngf
                                 backfacing, path_roughness,
                                 1.0f,  // outer_ior
                                 0),    // hero wavelength off
               data)
    {
    }

    OSL_HOSTDEVICE BSDF::Sample eval(const Vec3& wo, const Vec3& wi) const
    {
        bsdl::Sample s = Base::eval_impl(Base::frame.local(wo),
                                         Base::frame.local(wi));
        return { wi, s.weight.toRGB(0), s.pdf, s.roughness };
    }
    OSL_HOSTDEVICE BSDF::Sample sample(const Vec3& wo, float rx, float ry,
                                       float rz) const
    {
        bsdl::Sample s = Base::sample_impl(Base::frame.local(wo),
                                           { rx, ry, rz });
        return { Base::frame.world(s.wi), s.weight.toRGB(0), s.pdf,
                 s.roughness };
    }
};

#ifndef __CUDACC__
// Helper to register BSDL closures
struct BSDLtoOSL {
    template<typename BSDF> void visit()
    {
        const auto e = BSDF::template entry<typename BSDF::Data>();
        std::vector<ClosureParam> params;
        for (int i = 0; true; ++i) {
            const bsdl::LobeParam& in = e.params[i];
            TypeDesc osltype;
            switch (in.type) {
            case bsdl::ParamType::NONE: osltype = TypeDesc(); break;
            case bsdl::ParamType::VECTOR: osltype = OSL::TypeVector; break;
            case bsdl::ParamType::INT: osltype = OSL::TypeInt; break;
            case bsdl::ParamType::FLOAT: osltype = OSL::TypeFloat; break;
            case bsdl::ParamType::COLOR: osltype = OSL::TypeColor; break;
            case bsdl::ParamType::STRING: osltype = OSL::TypeString; break;
            case bsdl::ParamType::CLOSURE: osltype = TypeDesc::PTR; break;
            }
            params.push_back({ osltype, in.offset, in.key, in.type_size });
            if (in.type == bsdl::ParamType::NONE)
                break;
        }
        shadingsys->register_closure(e.name, BSDF::closureid(), params.data(),
                                     nullptr, nullptr);
    }

    OSL::ShadingSystem* shadingsys;
};

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
            CLOSURE_COLOR_PARAM(MxSubsurfaceParams, radius),
            CLOSURE_FLOAT_PARAM(MxSubsurfaceParams, anisotropy),
            CLOSURE_STRING_KEYPARAM(MxSubsurfaceParams, label, "label"),
            CLOSURE_FINISH_PARAM(MxSubsurfaceParams) } },
        { "sheen_bsdf",
          MX_SHEEN_ID,
          { CLOSURE_VECTOR_PARAM(MxSheenParams, N),
            CLOSURE_COLOR_PARAM(MxSheenParams, albedo),
            CLOSURE_FLOAT_PARAM(MxSheenParams, roughness),
            CLOSURE_STRING_KEYPARAM(MxSheenParams, label, "label"),
            CLOSURE_INT_KEYPARAM(MxSheenParams, mode, "mode"),
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

    // BSDFs coming from BSDL
    using bsdl_set = bsdl::TypeList<SpiThinLayer, MxConductor, MxDielectric>;
    // Register them
    bsdl_set::apply(BSDLtoOSL { shadingsys });
}
#endif  // ifndef __CUDACC__

template<int trans> struct Diffuse final : public BSDF, DiffuseParams {
    OSL_HOSTDEVICE Diffuse(const DiffuseParams& params)
        : BSDF(this), DiffuseParams(params)
    {
        if (trans)
            N = -N;
    }
    OSL_HOSTDEVICE Sample eval(const Vec3& /*wo*/, const OSL::Vec3& wi) const
    {
        const float pdf = std::max(N.dot(wi), 0.0f) * float(M_1_PI);
        return { wi, Color3(1.0f), pdf, 1.0f };
    }
    OSL_HOSTDEVICE Sample sample(const Vec3& /*wo*/, float rx, float ry,
                                 float /*rz*/) const
    {
        Vec3 out_dir;
        float pdf;
        Sampling::sample_cosine_hemisphere(N, rx, ry, out_dir, pdf);
        return { out_dir, Color3(1.0f), pdf, 1.0f };
    }
};

struct OrenNayar final : public BSDF, OrenNayarParams {
    OSL_HOSTDEVICE OrenNayar(const OrenNayarParams& params)
        : BSDF(this), OrenNayarParams(params)
    {
    }
    OSL_HOSTDEVICE Sample eval(const Vec3& wo, const OSL::Vec3& wi) const
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
    OSL_HOSTDEVICE Sample sample(const Vec3& wo, float rx, float ry,
                                 float /*rz*/) const
    {
        Vec3 out_dir;
        float pdf;
        Sampling::sample_cosine_hemisphere(N, rx, ry, out_dir, pdf);
        return eval(wo, out_dir);
    }
};

struct EnergyCompensatedOrenNayar : public BSDF, MxOrenNayarDiffuseParams {
    OSL_HOSTDEVICE
    EnergyCompensatedOrenNayar(const MxOrenNayarDiffuseParams& params)
        : BSDF(this), MxOrenNayarDiffuseParams(params)
    {
    }
    OSL_HOSTDEVICE Sample eval(const Vec3& wo, const OSL::Vec3& wi) const
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

    OSL_HOSTDEVICE Sample sample(const Vec3& wo, float rx, float ry,
                                 float /*rz*/) const
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

    OSL_HOSTDEVICE float E_FON_analytic(float mu) const
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
    OSL_HOSTDEVICE Phong(const PhongParams& params)
        : BSDF(this), PhongParams(params)
    {
    }
    OSL_HOSTDEVICE Sample eval(const Vec3& wo, const Vec3& wi) const
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
    OSL_HOSTDEVICE Sample sample(const Vec3& wo, float rx, float ry,
                                 float /*rz*/) const
    {
        float cosNO = N.dot(wo);
        if (cosNO > 0) {
            // reflect the view vector
            Vec3 R    = (2 * cosNO) * N - wo;
            float phi = 2 * float(M_PI) * rx;
            float sp, cp;
            OIIO::fast_sincos(phi, &sp, &cp);
            float cosTheta  = OIIO::fast_safe_pow(ry, 1 / (exponent + 1));
            float sinTheta2 = 1 - cosTheta * cosTheta;
            float sinTheta  = sinTheta2 > 0 ? sqrtf(sinTheta2) : 0;
            Vec3 wi         = TangentFrame::from_normal(R).get(cp * sinTheta,
                                                               sp * sinTheta, cosTheta);
            return eval(wo, wi);
        }
        return {};
    }
};

struct Ward final : public BSDF, WardParams {
    OSL_HOSTDEVICE Ward(const WardParams& params)
        : BSDF(this), WardParams(params)
    {
    }
    OSL_HOSTDEVICE Sample eval(const Vec3& wo, const OSL::Vec3& wi) const
    {
        float cosNO = N.dot(wo);
        float cosNI = N.dot(wi);
        if (cosNI > 0 && cosNO > 0) {
            // get half vector and get x,y basis on the surface for anisotropy
            Vec3 H = wi + wo;
            H.normalize();  // normalize needed for pdf
            TangentFrame tf = TangentFrame::from_normal_and_tangent(N, T);
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
    OSL_HOSTDEVICE Sample sample(const Vec3& wo, float rx, float ry,
                                 float /*rz*/) const
    {
        float cosNO = N.dot(wo);
        if (cosNO > 0) {
            // get x,y basis on the surface for anisotropy
            TangentFrame tf = TangentFrame::from_normal_and_tangent(N, T);
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
    static OSL_HOSTDEVICE float F(const float tan_m2)
    {
        return 1 / (float(M_PI) * (1 + tan_m2) * (1 + tan_m2));
    }

    static OSL_HOSTDEVICE float Lambda(const float a2)
    {
        return 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / a2));
    }

    static OSL_HOSTDEVICE Vec2 sampleSlope(float cos_theta, float randu,
                                           float randv)
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
    static OSL_HOSTDEVICE float F(const float tan_m2)
    {
        return float(1 / M_PI) * OIIO::fast_exp(-tan_m2);
    }

    static OSL_HOSTDEVICE float Lambda(const float a2)
    {
        const float a = sqrtf(a2);
        return a < 1.6f ? (1.0f - 1.259f * a + 0.396f * a2)
                              / (3.535f * a + 2.181f * a2)
                        : 0.0f;
    }

    static OSL_HOSTDEVICE Vec2 sampleSlope(float cos_theta, float randu,
                                           float randv)
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
    OSL_HOSTDEVICE Microfacet(const MicrofacetParams& params)
        : BSDF(this)
        , MicrofacetParams(params)
        , tf(TangentFrame::from_normal_and_tangent(N, U))
    {
    }
    OSL_HOSTDEVICE Color3 get_albedo(const Vec3& wo) const
    {
        if (Refract == 2)
            return Color3(1.0f);
        // FIXME: this heuristic is not particularly good, and looses energy
        // compared to the reference solution
        float fr = fresnel_dielectric(N.dot(wo), eta);
        return Color3(Refract ? 1 - fr : fr);
    }
    OSL_HOSTDEVICE Sample eval(const Vec3& wo, const OSL::Vec3& wi) const
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

    OSL_HOSTDEVICE Sample sample(const Vec3& wo, float rx, float ry,
                                 float rz) const
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
    static OSL_HOSTDEVICE float SQR(float x) { return x * x; }

    OSL_HOSTDEVICE float evalLambda(const Vec3 w) const
    {
        float cosTheta2 = SQR(w.z);
        /* Have these two multiplied by sinTheta^2 for convenience */
        float cosPhi2st2 = SQR(w.x * xalpha);
        float sinPhi2st2 = SQR(w.y * yalpha);
        return Distribution::Lambda(cosTheta2 / (cosPhi2st2 + sinPhi2st2));
    }

    static OSL_HOSTDEVICE float evalG2(float Lambda_i, float Lambda_o)
    {
        // correlated masking-shadowing
        return 1 / (Lambda_i + Lambda_o + 1);
    }

    static OSL_HOSTDEVICE float evalG1(float Lambda_v)
    {
        return 1 / (Lambda_v + 1);
    }

    OSL_HOSTDEVICE float evalD(const Vec3 Hr) const
    {
        float cosThetaM = Hr.z;
        if (cosThetaM > 0) {
            /* Have these two multiplied by sinThetaM2 for convenience */
            float cosPhi2st2 = SQR(Hr.x / xalpha);
            float sinPhi2st2 = SQR(Hr.y / yalpha);
            float cosThetaM2 = SQR(cosThetaM);
            float cosThetaM4 = SQR(cosThetaM2);

            float tanThetaM2 = (cosPhi2st2 + sinPhi2st2) / cosThetaM2;

            const float val = Distribution::F(tanThetaM2)
                              / (xalpha * yalpha * cosThetaM4);
#ifndef __CUDACC__
            return val;
#else
            // Filter out NaNs that can be produced when cosThetaM is very small.
            return (val == val) ? val : 0;
#endif
        }
        return 0;
    }

    OSL_HOSTDEVICE Vec3 sampleMicronormal(const Vec3 wo, float randu,
                                          float randv) const
    {
        /* Project wo and stretch by alpha values */
        Vec3 swo = wo;
        swo.x *= xalpha;
        swo.y *= yalpha;
        swo = swo.normalize();

#ifdef __CUDACC__
        swo = swo.normalize();
#endif

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


// We use the CRTP to inherit the parameters because each MaterialX closure uses a different set of parameters
template<typename MxMicrofacetParams, typename Distribution,
         bool EnableTransmissionLobe>
struct MxMicrofacet final : public BSDF, MxMicrofacetParams {
    OSL_HOSTDEVICE MxMicrofacet(const MxMicrofacetParams& params,
                                float refraction_ior)
        : BSDF(this)
        , MxMicrofacetParams(params)
        , tf(TangentFrame::from_normal_and_tangent(MxMicrofacetParams::N,
                                                   MxMicrofacetParams::U))
        , refraction_ior(refraction_ior)
    {
    }

    OSL_HOSTDEVICE float get_fresnel_angle(float cos_theta) const
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

    OSL_HOSTDEVICE Color3 get_albedo(const Vec3& wo) const
    {
        // if transmission is enabled, punt on
        if (EnableTransmissionLobe)
            return Color3(1.0f);

        return MxMicrofacetParams::dirAlbedoR(
            get_fresnel_angle(MxMicrofacetParams::N.dot(wo)));
    }

    OSL_HOSTDEVICE Sample eval(const Vec3& wo, const OSL::Vec3& wi) const
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


    OSL_HOSTDEVICE Sample sample(const Vec3& wo, float rx, float ry,
                                 float rz) const
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
    static OSL_HOSTDEVICE float SQR(float x) { return x * x; }

    OSL_HOSTDEVICE float evalLambda(const Vec3 w) const
    {
        float cosTheta2 = SQR(w.z);
        /* Have these two multiplied by sinTheta^2 for convenience */
        float cosPhi2st2 = SQR(w.x * MxMicrofacetParams::roughness_x);
        float sinPhi2st2 = SQR(w.y * MxMicrofacetParams::roughness_y);
        return Distribution::Lambda(cosTheta2 / (cosPhi2st2 + sinPhi2st2));
    }

    static OSL_HOSTDEVICE float evalG2(float Lambda_i, float Lambda_o)
    {
        // correlated masking-shadowing
        return 1 / (Lambda_i + Lambda_o + 1);
    }

    static OSL_HOSTDEVICE float evalG1(float Lambda_v)
    {
        return 1 / (Lambda_v + 1);
    }

    OSL_HOSTDEVICE float evalD(const Vec3 Hr) const
    {
        float cosThetaM = Hr.z;
        if (cosThetaM > 0) {
            /* Have these two multiplied by sinThetaM2 for convenience */
            float cosPhi2st2 = SQR(Hr.x / MxMicrofacetParams::roughness_x);
            float sinPhi2st2 = SQR(Hr.y / MxMicrofacetParams::roughness_y);
            float cosThetaM2 = SQR(cosThetaM);
            float cosThetaM4 = SQR(cosThetaM2);

            float tanThetaM2 = (cosPhi2st2 + sinPhi2st2) / cosThetaM2;

            const float val = Distribution::F(tanThetaM2)
                              / (MxMicrofacetParams::roughness_x
                                 * MxMicrofacetParams::roughness_y
                                 * cosThetaM4);
#ifndef __CUDACC__
            return val;
#else
            // Filter out NaNs that can be produced when cosThetaM is very small.
            return (val == val) ? val : 0.0;
#endif
        }
        return 0;
    }

    OSL_HOSTDEVICE Vec3 sampleMicronormal(const Vec3 wo, float randu,
                                          float randv) const
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
    OSL_HOSTDEVICE Reflection(const ReflectionParams& params)
        : BSDF(this), ReflectionParams(params)
    {
    }
    OSL_HOSTDEVICE Color3 get_albedo(const Vec3& wo) const
    {
        float cosNO = N.dot(wo);
        if (cosNO > 0)
            return Color3(fresnel_dielectric(cosNO, eta));
        return Color3(1);
    }
    OSL_HOSTDEVICE Sample eval(const Vec3& /*wo*/,
                               const OSL::Vec3& /*wi*/) const
    {
        return {};
    }
    OSL_HOSTDEVICE Sample sample(const Vec3& wo, float /*rx*/, float /*ry*/,
                                 float /*rz*/) const
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
    OSL_HOSTDEVICE Refraction(const RefractionParams& params)
        : BSDF(this), RefractionParams(params)
    {
    }
    OSL_HOSTDEVICE Color3 get_albedo(const Vec3& wo) const
    {
        float cosNO = N.dot(wo);
        return Color3(1 - fresnel_dielectric(cosNO, eta));
    }
    OSL_HOSTDEVICE Sample eval(const Vec3& /*wo*/,
                               const OSL::Vec3& /*wi*/) const
    {
        return {};
    }
    OSL_HOSTDEVICE Sample sample(const Vec3& wo, float /*rx*/, float /*ry*/,
                                 float /*rz*/) const
    {
        float pdf = std::numeric_limits<float>::infinity();
        Vec3 wi;
        float Ft = fresnel_refraction(-wo, N, eta, wi);
        return { wi, Color3(Ft), pdf, 0 };
    }
};

struct Transparent final : public BSDF {
    OSL_HOSTDEVICE Transparent() : BSDF(this) {}
    OSL_HOSTDEVICE Sample eval(const Vec3& /*wo*/, const Vec3& /*wi*/) const
    {
        return {};
    }
    OSL_HOSTDEVICE Sample sample(const Vec3& wo, float /*rx*/, float /*ry*/,
                                 float /*rz*/) const
    {
        Vec3 wi   = -wo;
        float pdf = std::numeric_limits<float>::infinity();
        return { wi, Color3(1.0f), pdf, 0 };
    }
};

struct MxBurleyDiffuse final : public BSDF, MxBurleyDiffuseParams {
    OSL_HOSTDEVICE MxBurleyDiffuse(const MxBurleyDiffuseParams& params)
        : BSDF(this), MxBurleyDiffuseParams(params)
    {
    }

    OSL_HOSTDEVICE Color3 get_albedo(const Vec3& wo) const { return albedo; }

    OSL_HOSTDEVICE Sample eval(const Vec3& wo, const Vec3& wi) const
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

    OSL_HOSTDEVICE Sample sample(const Vec3& wo, float rx, float ry,
                                 float rz) const
    {
        Vec3 out_dir;
        float pdf;
        Sampling::sample_cosine_hemisphere(N, rx, ry, out_dir, pdf);
        return eval(wo, out_dir);
    }
};

// Implementation of the "Charlie Sheen" model [Conty & Kulla, 2017]
// https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_sheen.pdf
// To simplify the implementation, the simpler shadowing/masking visibility term below is used:
// https://dassaultsystemes-technology.github.io/EnterprisePBRShadingModel/spec-2022x.md.html#components/sheen
struct CharlieSheen final : public BSDF, MxSheenParams {
    OSL_HOSTDEVICE CharlieSheen(const MxSheenParams& params)
        : BSDF(this), MxSheenParams(params)
    {
    }

    OSL_HOSTDEVICE Color3 get_albedo(const Vec3& wo) const
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

    OSL_HOSTDEVICE Sample eval(const Vec3& wo, const Vec3& wi) const
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

    OSL_HOSTDEVICE Sample sample(const Vec3& wo, float rx, float ry,
                                 float rz) const
    {
        Vec3 out_dir;
        float pdf;
        Sampling::sample_uniform_hemisphere(N, rx, ry, out_dir, pdf);
        return eval(wo, out_dir);
    }
};

// Implement the sheen model proposed in:
//  "Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines"
// Tizian Zeltner, Brent Burley, Matt Jen-Yuan Chiang - Siggraph 2022
// https://tizianzeltner.com/projects/Zeltner2022Practical/
struct ZeltnerBurleySheen final : public BSDF, MxSheenParams {
    OSL_HOSTDEVICE ZeltnerBurleySheen(const MxSheenParams& params)
        : BSDF(this), MxSheenParams(params)
    {
    }

#define USE_LTC_SAMPLING 1

    OSL_HOSTDEVICE Color3 get_albedo(const Vec3& wo) const
    {
        const float NdotV = clamp(N.dot(wo), 1e-5f, 1.0f);
        return Color3(fetch_ltc(NdotV).z);
    }

    OSL_HOSTDEVICE Sample eval(const Vec3& wo, const Vec3& wi) const
    {
        const Vec3 L = wi, V = wo;
        const float NdotV = clamp(N.dot(V), 0.0f, 1.0f);
        const Vec3 ltc    = fetch_ltc(NdotV);

        const Vec3 localL = TangentFrame::from_normal_and_tangent(N, V).tolocal(
            L);

        const float aInv = ltc.x, bInv = ltc.y, R = ltc.z;
        Vec3 wiOriginal(aInv * localL.x + bInv * localL.z, aInv * localL.y,
                        localL.z);
        const float len2 = dot(wiOriginal, wiOriginal);

        float det      = aInv * aInv;
        float jacobian = det / (len2 * len2);

#if USE_LTC_SAMPLING == 1
        float pdf = jacobian * std::max(wiOriginal.z, 0.0f) * float(M_1_PI);
        return { wi, Color3(R), pdf, 1.0f };
#else
        float pdf = float(0.5 * M_1_PI);
        // NOTE: sheen closure has no fresnel/masking
        return { wi, Color3(2 * R * jacobian * std::max(wiOriginal.z, 0.0f)),
                 pdf, 1.0f };
#endif
    }

    OSL_HOSTDEVICE Sample sample(const Vec3& wo, float rx, float ry,
                                 float rz) const
    {
#if USE_LTC_SAMPLING == 1
        const Vec3 V      = wo;
        const float NdotV = clamp(N.dot(V), 0.0f, 1.0f);
        const Vec3 ltc    = fetch_ltc(NdotV);
        const float aInv = ltc.x, bInv = ltc.y, R = ltc.z;
        Vec3 wi;
        float pdf;
        Sampling::sample_cosine_hemisphere(Vec3(0, 0, 1), rx, ry, wi, pdf);

        const Vec3 w         = Vec3(wi.x - wi.z * bInv, wi.y, wi.z * aInv);
        const float len2     = dot(w, w);
        const float jacobian = len2 * len2 / (aInv * aInv);
        const Vec3 wn        = w / sqrtf(len2);

        const Vec3 L = TangentFrame::from_normal_and_tangent(N, V).toworld(wn);

        pdf = jacobian * std::max(wn.z, 0.0f) * float(M_1_PI);

        return { L, Color3(R), pdf, 1.0f };
#else
        // plain uniform-sampling for validation
        Vec3 out_dir;
        float pdf;
        Sampling::sample_uniform_hemisphere(N, rx, ry, out_dir, pdf);
        return eval(wo, out_dir);
#endif
    }

private:
    OSL_HOSTDEVICE Vec3 fetch_ltc(float NdotV) const
    {
        // To avoid look-up tables, we use a fit of the LTC coefficients derived by Stephen Hill
        // for the implementation in MaterialX:
        // https://github.com/AcademySoftwareFoundation/MaterialX/blob/main/libraries/pbrlib/genglsl/lib/mx_microfacet_sheen.glsl
        const float x = NdotV;
        const float y = std::max(roughness, 1e-3f);
        const float A = ((2.58126f * x + 0.813703f * y) * y)
                        / (1.0f + 0.310327f * x * x + 2.60994f * x * y);
        const float B = sqrtf(1.0f - x) * (y - 1.0f) * y * y * y
                        / (0.0000254053f + 1.71228f * x - 1.71506f * x * y
                           + 1.34174f * y * y);
        const float invs = (0.0379424f + y * (1.32227f + y))
                           / (y * (0.0206607f + 1.58491f * y));
        const float m = y
                        * (-0.193854f
                           + y * (-1.14885 + y * (1.7932f - 0.95943f * y * y)))
                        / (0.046391f + y);
        const float o = y * (0.000654023f + (-0.0207818f + 0.119681f * y) * y)
                        / (1.26264f + y * (-1.92021f + y));
        float q                 = (x - m) * invs;
        const float inv_sqrt2pi = 0.39894228040143f;
        float R                 = expf(-0.5f * q * q) * invs * inv_sqrt2pi + o;
        assert(isfinite(A));
        assert(isfinite(B));
        assert(isfinite(R));
        return Vec3(A, B, R);
    }
};


struct HenyeyGreenstein final : public BSDF {
    const float g;
    OSL_HOSTDEVICE HenyeyGreenstein(float g) : BSDF(this), g(g) {}

    static OSL_HOSTDEVICE float PhaseHG(float cos_theta, float g)
    {
        const float denom = 1 + g * g + 2 * g * cos_theta;
        return (1 - g * g) / (4 * M_PI * denom * sqrtf(denom));
    }

    OSL_HOSTDEVICE Sample eval(const Vec3& wo, const Vec3& wi) const
    {
        const float pdf = PhaseHG(dot(wo, wi), g);
        return { wi, Color3(pdf), pdf, 0.0f };
    }

    OSL_HOSTDEVICE Sample sample(const Vec3& wo, float rx, float ry,
                                 float rz) const
    {
        TangentFrame frame = TangentFrame::from_normal(wo);

        float cos_theta;
        if (abs(g) < 1e-3f) {
            cos_theta = 1.0f - 2.0f * rx;
        } else {
            float sqr_term = (1 - g * g) / (1 - g + 2 * g * rx);
            cos_theta      = (1 + g * g - sqr_term * sqr_term) / (2 * g);
            cos_theta      = OIIO::clamp(cos_theta, -1.0f, 1.0f);
        }

        float sin_theta = sqrtf(
            OIIO::clamp(1.0f - cos_theta * cos_theta, 0.0f, 1.0f));
        float phi     = 2 * M_PI * ry;
        Vec3 local_wi = Vec3(sin_theta * cosf(phi), sin_theta * sinf(phi),
                             cos_theta);

        Vec3 wi       = frame.toworld(local_wi);
        float pdf_val = PhaseHG(cos_theta, g);

        return { wi, Color3(1.0f), pdf_val, 0.0f };
    }
};

struct HomogeneousMedium final : public Medium {
    MediumParams params;
    HenyeyGreenstein phase_func;

    OSL_HOSTDEVICE HomogeneousMedium(const MediumParams& params)
        : Medium(this), params(params), phase_func(params.medium_g)
    {
    }

    OSL_HOSTDEVICE Medium::Sample sample(Ray& r, Sampler& sampler,
                                         Intersection& hit) const
    {
        Vec3 rand_vol = sampler.get();

        float t_volume = -logf(1.0f - rand_vol.x) / params.avg_sigma_t();

        Color3 weight;
        Color3 tr;

        if (t_volume < hit.t) {
            r.origin = r.point(t_volume);
            tr       = transmittance(t_volume);

            Color3 albedo = params.sigma_s / params.sigma_t;

            weight = albedo / tr;
        } else {
            tr     = transmittance(hit.t);
            weight = Color3(1.0 / tr.x, 1.0 / tr.y, 1.0 / tr.z);
        }

        return Medium::Sample { t_volume, tr, weight };
    }

    OSL_HOSTDEVICE BSDF::Sample sample_phase_func(const Vec3& wo, float rx,
                                                     float ry,
                                                     float rz) const
    {
        return phase_func.sample(wo, rx, ry, rz);
    }

    OSL_HOSTDEVICE const MediumParams* get_params() const { return &params; }

    OSL_HOSTDEVICE Color3 transmittance(float distance) const
    {  // Beer-Lambert law
        return Color3(expf(-params.sigma_t.x * distance),
                      expf(-params.sigma_t.y * distance),
                      expf(-params.sigma_t.z * distance));
    }
};

struct EmptyMedium final : public Medium {
    MediumParams params;

    OSL_HOSTDEVICE EmptyMedium(const MediumParams& params)
        : Medium(this), params(params)
    {
    }

    OSL_HOSTDEVICE const MediumParams* get_params() const { return &params; }

    OSL_HOSTDEVICE Medium::Sample sample(Ray& ray, Sampler& sampler,
                                         Intersection& hit) const
    {
        return { 0.0f, Color3(1.0f), Color3(1.0f) };
    }

    OSL_HOSTDEVICE BSDF::Sample sample_phase_func(const Vec3& wo, float rx,
                                                     float ry,
                                                     float rz) const
    {
        return { Vec3(1.0f), Color3(1.0f), 0.0f, 0.0f };
    }

};


OSL_HOSTDEVICE Color3
evaluate_layer_opacity(const ShaderGlobalsType& sg, float path_roughness,
                       const ClosureColor* closure)
{
    // Null closure, the layer is fully transparent
    if (closure == nullptr)
        return Color3(0);

    // Non-recursive traversal stack
    const int STACK_SIZE = 16;
    int stack_idx        = 0;
    const ClosureColor* ptr_stack[STACK_SIZE];
    Color3 weight_stack[STACK_SIZE];
    Color3 weight = Color3(1.0f);

    while (closure) {
        switch (closure->id) {
        case ClosureColor::MUL:
            weight *= closure->as_mul()->weight;
            closure = closure->as_mul()->closure;
            break;
        case ClosureColor::ADD:
            ptr_stack[stack_idx]      = closure->as_add()->closureB;
            weight_stack[stack_idx++] = weight;
            closure                   = closure->as_add()->closureA;
            break;
        default: {
            const ClosureComponent* comp = closure->as_comp();
            Color3 w                     = comp->w;
            switch (comp->id) {
            case MX_LAYER_ID: {
                const MxLayerParams* srcparams = comp->as<MxLayerParams>();
                closure                        = srcparams->top;
                ptr_stack[stack_idx]           = srcparams->base;
                weight_stack[stack_idx++]      = weight * w;
                break;
            }
            case REFLECTION_ID:
            case FRESNEL_REFLECTION_ID: {
                Reflection bsdf(*comp->as<ReflectionParams>());
                weight *= w * bsdf.get_albedo(-sg.I);
                closure = nullptr;
                break;
            }
            case MxDielectric::closureid(): {
                const MxDielectric::Data& params
                    = *comp->as<MxDielectric::Data>();
                MxDielectric d(params, -sg.I, sg.backfacing, path_roughness);
                weight *= w * (Color3(1) - d.filter_o(-sg.I).toRGB(0));
                closure = nullptr;
                break;
            }
            case MX_GENERALIZED_SCHLICK_ID: {
                const MxGeneralizedSchlickParams& params
                    = *comp->as<MxGeneralizedSchlickParams>();
                // Transmissive dielectrics are opaque
                if (!is_black(params.transmission_tint)) {
                    closure = nullptr;
                    break;
                }
                MxMicrofacet<MxGeneralizedSchlickParams, GGXDist, false> mf(
                    params, 1.0f);
                weight *= w * mf.get_albedo(-sg.I);
                closure = nullptr;
                break;
            }
            case MX_SHEEN_ID: {
                const MxSheenParams& params = *comp->as<MxSheenParams>();
                if (params.mode == 1) {
                    weight *= w * ZeltnerBurleySheen(params).get_albedo(-sg.I);
                } else {
                    // otherwise, default to old sheen model
                    weight *= w * CharlieSheen(params).get_albedo(-sg.I);
                }
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
    return weight;
}

OSL_HOSTDEVICE void
process_medium_closure(const ShaderGlobalsType& sg, float path_roughness,
                       ShadingResult& result, MediumStack& medium_stack,
                       const ClosureColor* closure, const Color3& w)
{
    if (!closure)
        return;

    // Non-recursive traversal stack
    const int STACK_SIZE = 16;
    int stack_idx        = 0;
    const ClosureColor* ptr_stack[STACK_SIZE];
    Color3 weight_stack[STACK_SIZE];
    Color3 weight = w;

    while (closure) {
        switch (closure->id) {
        case ClosureColor::MUL: {
            weight *= closure->as_mul()->weight;
            closure = closure->as_mul()->closure;
            break;
        }
        case ClosureColor::ADD: {
            weight_stack[stack_idx] = weight;
            ptr_stack[stack_idx++]  = closure->as_add()->closureB;
            closure                 = closure->as_add()->closureA;
            break;
        }
        case MX_LAYER_ID: {
            const ClosureComponent* comp = closure->as_comp();
            const MxLayerParams* params  = comp->as<MxLayerParams>();
            Color3 base_w                = weight
                            * (Color3(1)
                               - clamp(evaluate_layer_opacity(sg, path_roughness,
                                                              params->top),
                                       0.f, 1.f));
            closure                   = params->top;
            ptr_stack[stack_idx]      = params->base;
            weight_stack[stack_idx++] = weight * base_w;
            break;
        }
        case MX_ANISOTROPIC_VDF_ID: {
            const ClosureComponent* comp = closure->as_comp();
            Color3 cw                    = weight * comp->w;
            const auto& params           = *comp->as<MxAnisotropicVdfParams>();
            result.medium_data.sigma_t   = cw * params.extinction;
            result.medium_data.sigma_s   = params.albedo
                                         * result.medium_data.sigma_t;
            result.medium_data.medium_g = params.anisotropy;
            result.medium_data.priority = 0;

            if (!sg.backfacing) {  // if entering
                if (result.medium_data.is_vaccum()) {
                    medium_stack.add_medium<EmptyMedium>(result.medium_data);
                } else {
                    medium_stack.add_medium<HomogeneousMedium>(
                        result.medium_data);
                }
            }

            closure = nullptr;
            break;
        }
        case MX_MEDIUM_VDF_ID: {
            const ClosureComponent* comp = closure->as_comp();
            Color3 cw                    = weight * comp->w;
            const auto& params           = *comp->as<MxMediumVdfParams>();

            result.medium_data.sigma_t
                = Color3(-OIIO::fast_log(params.transmission_color.x),
                         -OIIO::fast_log(params.transmission_color.y),
                         -OIIO::fast_log(params.transmission_color.z));

            result.medium_data.sigma_t *= cw / params.transmission_depth;
            result.medium_data.sigma_s = params.albedo
                                         * result.medium_data.sigma_t;
            result.medium_data.medium_g = params.anisotropy;

            result.medium_data.refraction_ior = sg.backfacing
                                                    ? 1.0f / params.ior
                                                    : params.ior;
            result.medium_data.priority       = params.priority;

            if (!sg.backfacing) {  // if entering
                if (result.medium_data.is_vaccum()) {
                    medium_stack.add_medium<EmptyMedium>(result.medium_data);
                } else {
                    medium_stack.add_medium<HomogeneousMedium>(
                        result.medium_data);
                }
            }

            closure = nullptr;
            break;
        }
        case MxDielectric::closureid(): {
            const ClosureComponent* comp     = closure->as_comp();
            const MxDielectric::Data& params = *comp->as<MxDielectric::Data>();
            if (!is_black(weight * comp->w * params.refr_tint)) {
                float new_ior = sg.backfacing ? 1.0f / params.IOR : params.IOR;

                result.medium_data.refraction_ior = new_ior;

                const MediumParams* current_params
                    = medium_stack.current_params();
                if (current_params
                    && result.medium_data.priority
                           <= current_params->priority) {
                    result.medium_data.refraction_ior
                        = current_params->refraction_ior;
                }
            }
            closure = nullptr;
            break;
        }
        case MX_GENERALIZED_SCHLICK_ID: {
            const ClosureComponent* comp = closure->as_comp();
            const auto& params = *comp->as<MxGeneralizedSchlickParams>();
            if (!is_black(weight * comp->w * params.transmission_tint)) {
                float avg_F0  = clamp((params.f0.x + params.f0.y + params.f0.z)
                                          / 3.0f,
                                      0.0f, 0.99f);
                float sqrt_F0 = sqrtf(avg_F0);
                float ior     = (1 + sqrt_F0) / (1 - sqrt_F0);
                float new_ior = sg.backfacing ? 1.0f / ior : ior;

                result.medium_data.refraction_ior = new_ior;

                const MediumParams* current_params
                    = medium_stack.current_params();
                if (current_params
                    && result.medium_data.priority
                           <= current_params->priority) {
                    result.medium_data.refraction_ior
                        = current_params->refraction_ior;
                }
            }
            closure = nullptr;
            break;
        }
        default: closure = nullptr; break;
        }
        if (closure == nullptr && stack_idx > 0) {
            closure = ptr_stack[--stack_idx];
            weight  = weight_stack[stack_idx];
        }
    }
}

// recursively walk through the closure tree, creating bsdfs as we go
OSL_HOSTDEVICE void
process_bsdf_closure(const ShaderGlobalsType& sg, float path_roughness,
                     ShadingResult& result, MediumStack& medium_stack,
                     const ClosureColor* closure, const Color3& w,
                     bool light_only)
{
    static const ustringhash uh_ggx("ggx");
    static const ustringhash uh_beckmann("beckmann");
    static const ustringhash uh_default("default");
    if (!closure)
        return;

    // Non-recursive traversal stack
    const int STACK_SIZE = 16;
    int stack_idx        = 0;
    const ClosureColor* ptr_stack[STACK_SIZE];
    Color3 weight_stack[STACK_SIZE];
    Color3 weight = w;

    while (closure) {
        switch (closure->id) {
        case ClosureColor::MUL: {
            weight *= closure->as_mul()->weight;
            closure = closure->as_mul()->closure;
            break;
        }
        case ClosureColor::ADD: {
            ptr_stack[stack_idx]      = closure->as_add()->closureB;
            weight_stack[stack_idx++] = weight;
            closure                   = closure->as_add()->closureA;
            break;
        }
        default: {
            const ClosureComponent* comp = closure->as_comp();
            Color3 cw                    = weight * comp->w;
            closure                      = nullptr;
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
                    ok = result.bsdf.add_bsdf<Phong>(cw,
                                                     *comp->as<PhongParams>());
                    break;
                case WARD_ID:
                    ok = result.bsdf.add_bsdf<Ward>(cw,
                                                    *comp->as<WardParams>());
                    break;
                case MICROFACET_ID: {
                    const MicrofacetParams* mp = comp->as<MicrofacetParams>();
                    if (mp->dist == uh_ggx) {
                        switch (mp->refract) {
                        case 0:
                            ok = result.bsdf.add_bsdf<MicrofacetGGXRefl>(cw,
                                                                         *mp);
                            break;
                        case 1:
                            ok = result.bsdf.add_bsdf<MicrofacetGGXRefr>(cw,
                                                                         *mp);
                            break;
                        case 2:
                            ok = result.bsdf.add_bsdf<MicrofacetGGXBoth>(cw,
                                                                         *mp);
                            break;
                        }
                    } else if (mp->dist == uh_beckmann
                               || mp->dist == uh_default) {
                        switch (mp->refract) {
                        case 0:
                            ok = result.bsdf.add_bsdf<MicrofacetBeckmannRefl>(
                                cw, *mp);
                            break;
                        case 1:
                            ok = result.bsdf.add_bsdf<MicrofacetBeckmannRefr>(
                                cw, *mp);
                            break;
                        case 2:
                            ok = result.bsdf.add_bsdf<MicrofacetBeckmannBoth>(
                                cw, *mp);
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
                        ok = result.bsdf.add_bsdf<OrenNayar>(
                            cw * srcparams->albedo, params);
                    }
                    break;
                }
                case MX_BURLEY_DIFFUSE_ID: {
                    const MxBurleyDiffuseParams& params
                        = *comp->as<MxBurleyDiffuseParams>();
                    ok = result.bsdf.add_bsdf<MxBurleyDiffuse>(cw, params);
                    break;
                }
                case MxDielectric::closureid(): {
                    const MxDielectric::Data& params
                        = *comp->as<MxDielectric::Data>();

                    if (medium_stack.false_intersection_with(
                            result.medium_data)) {
                        ok = result.bsdf.add_bsdf<Transparent>(cw);
                    } else {
                        ok = result.bsdf.add_bsdf<MxDielectric>(cw, params,
                                                                -sg.I,
                                                                sg.backfacing,
                                                                path_roughness);
                    }
                    break;
                }
                case MxConductor::closureid(): {
                    const MxConductor::Data& params
                        = *comp->as<MxConductor::Data>();
                    ok = result.bsdf.add_bsdf<MxConductor>(cw, params, -sg.I,
                                                           path_roughness);
                    break;
                }
                case MX_GENERALIZED_SCHLICK_ID: {
                    const MxGeneralizedSchlickParams& params
                        = *comp->as<MxGeneralizedSchlickParams>();

                    if (medium_stack.false_intersection_with(
                            result.medium_data)) {
                        ok = result.bsdf.add_bsdf<Transparent>(cw);
                    } else {
                        if (is_black(params.transmission_tint)) {
                            ok = result.bsdf.add_bsdf<MxMicrofacet<
                                MxGeneralizedSchlickParams, GGXDist, false>>(
                                cw, params, 1.0f);
                        } else {
                            ok = result.bsdf.add_bsdf<MxMicrofacet<
                                MxGeneralizedSchlickParams, GGXDist, true>>(
                                cw, params, result.medium_data.refraction_ior);
                        }
                    }
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
                    if (params.mode == 1)
                        ok = result.bsdf.add_bsdf<ZeltnerBurleySheen>(cw,
                                                                      params);
                    else
                        ok = result.bsdf.add_bsdf<CharlieSheen>(
                            cw, params);  // default to legacy closure
                    break;
                }
                case MX_LAYER_ID: {
                    const MxLayerParams* srcparams = comp->as<MxLayerParams>();
                    Color3 base_w
                        = weight
                          * (Color3(1, 1, 1)
                             - clamp(evaluate_layer_opacity(sg, path_roughness,
                                                            srcparams->top),
                                     0.f, 1.f));
                    closure = srcparams->top;
                    weight  = cw;
                    if (!is_black(base_w)) {
                        ptr_stack[stack_idx]      = srcparams->base;
                        weight_stack[stack_idx++] = base_w;
                    }
                    ok = true;
                    break;
                }
                case MX_ANISOTROPIC_VDF_ID:
                case MX_MEDIUM_VDF_ID: {
                    // already processed by process_medium_closure
                    ok = true;
                    break;
                }
                case SpiThinLayer::closureid(): {
                    const SpiThinLayer::Data& params
                        = *comp->as<SpiThinLayer::Data>();
                    ok = result.bsdf.add_bsdf<SpiThinLayer>(cw, params, -sg.I,
                                                            path_roughness);
                    break;
                }
                }
#ifndef __CUDACC__
                OSL_ASSERT(ok && "Invalid closure invoked in surface shader");
#else
                // TODO: We should never get here, but we sometimes do, e.g. in
                // the render-material-layer test.
                if (false && !ok)
                    printf("Invalid closure invoked in surface shader\n");
#endif
            }
            break;
        }
        }
        if (closure == nullptr && stack_idx > 0) {
            closure = ptr_stack[--stack_idx];
            weight  = weight_stack[stack_idx];
        }
    }
}


OSL_HOSTDEVICE void
process_closure(const ShaderGlobalsType& sg, float path_roughness,
                ShadingResult& result, MediumStack& medium_stack,
                const ClosureColor* Ci, bool light_only)
{
    if (!light_only)
        process_medium_closure(sg, path_roughness, result, medium_stack, Ci,
                               Color3(1));
    process_bsdf_closure(sg, path_roughness, result, medium_stack, Ci,
                         Color3(1), light_only);
}

OSL_HOSTDEVICE Vec3
process_background_closure(const ClosureColor* closure)
{
    if (!closure)
        return Vec3(0, 0, 0);

    // Non-recursive traversal stack
    const int STACK_SIZE = 16;
    int stack_idx        = 0;
    const ClosureColor* ptr_stack[STACK_SIZE];
    Color3 weight_stack[STACK_SIZE];
    Color3 weight = Color3(1.0f);

    while (closure) {
        switch (closure->id) {
        case ClosureColor::MUL: {
            weight *= closure->as_mul()->weight;
            closure = closure->as_mul()->closure;
            break;
        }
        case ClosureColor::ADD: {
            ptr_stack[stack_idx]      = closure->as_add()->closureB;
            weight_stack[stack_idx++] = weight;
            closure                   = closure->as_add()->closureA;
            break;
        }
        case BACKGROUND_ID: {
            weight *= closure->as_comp()->w;
            closure = nullptr;
            break;
        }
        }
        if (closure == nullptr && stack_idx > 0) {
            closure = ptr_stack[--stack_idx];
            weight  = weight_stack[stack_idx];
        }
    }
    return weight;
}

OSL_HOSTDEVICE Color3
BSDF::get_albedo_vrtl(const Vec3& wo) const
{
    return dispatch([&](auto bsdf) { return bsdf.get_albedo(wo); });
}

OSL_HOSTDEVICE BSDF::Sample
BSDF::eval_vrtl(const Vec3& wo, const Vec3& wi) const
{
    return dispatch([&](auto bsdf) { return bsdf.eval(wo, wi); });
}

OSL_HOSTDEVICE BSDF::Sample
BSDF::sample_vrtl(const Vec3& wo, float rx, float ry, float rz) const
{
    return dispatch([&](auto bsdf) { return bsdf.sample(wo, rx, ry, rz); });
}

OSL_HOSTDEVICE Medium::Sample
Medium::sample_vrtl(Ray& ray, Sampler& sampler, Intersection& hit) const
{
    return dispatch(
        [&](const auto& medium) { return medium.sample(ray, sampler, hit); });
}

OSL_HOSTDEVICE BSDF::Sample
Medium::sample_phase_func_vrtl(const Vec3& wo, float rx, float ry, float rz) const
{
    return dispatch([&](auto medium) { return medium.sample_phase_func(wo, rx, ry, rz); });
}

OSL_HOSTDEVICE const MediumParams*
Medium::get_params_vrtl() const
{
    return dispatch([&](const auto& medium) { return medium.get_params(); });
}

OSL_NAMESPACE_END
