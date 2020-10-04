// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage


#include "shading.h"
#include "sampling.h"
#include <OSL/genclosure.h>
#include "optics.h"

using namespace OSL;

namespace { // anonymous namespace

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
};

// these structures hold the parameters of each closure type
// they will be contained inside ClosureComponent
struct EmptyParams      { };
struct DiffuseParams    { Vec3 N; };
struct OrenNayarParams  { Vec3 N; float sigma; };
struct PhongParams      { Vec3 N; float exponent; };
struct WardParams       { Vec3 N, T; float ax, ay; };
struct ReflectionParams { Vec3 N; float eta; };
struct RefractionParams { Vec3 N; float eta; };
struct MicrofacetParams { ustring dist; Vec3 N, U; float xalpha, yalpha, eta; int refract; };

} // anonymous namespace


OSL_NAMESPACE_ENTER


void register_closures(OSL::ShadingSystem* shadingsys)
{
    // Describe the memory layout of each closure type to the OSL runtime
    enum { MaxParams = 32 };
    struct BuiltinClosures {
        const char* name;
        int id;
        ClosureParam params[MaxParams]; // upper bound
    };
    BuiltinClosures builtins[] = {
        { "emission"   , EMISSION_ID,           { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "background" , BACKGROUND_ID,         { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "diffuse"    , DIFFUSE_ID,            { CLOSURE_VECTOR_PARAM(DiffuseParams, N),
                                                  CLOSURE_FINISH_PARAM(DiffuseParams) } },
        { "oren_nayar" , OREN_NAYAR_ID,         { CLOSURE_VECTOR_PARAM(OrenNayarParams, N),
                                                  CLOSURE_FLOAT_PARAM (OrenNayarParams, sigma),
                                                  CLOSURE_FINISH_PARAM(OrenNayarParams) } },
        { "translucent", TRANSLUCENT_ID,        { CLOSURE_VECTOR_PARAM(DiffuseParams, N),
                                                  CLOSURE_FINISH_PARAM(DiffuseParams) } },
        { "phong"      , PHONG_ID,              { CLOSURE_VECTOR_PARAM(PhongParams, N),
                                                  CLOSURE_FLOAT_PARAM (PhongParams, exponent),
                                                  CLOSURE_FINISH_PARAM(PhongParams) } },
        { "ward"       , WARD_ID,               { CLOSURE_VECTOR_PARAM(WardParams, N),
                                                  CLOSURE_VECTOR_PARAM(WardParams, T),
                                                  CLOSURE_FLOAT_PARAM (WardParams, ax),
                                                  CLOSURE_FLOAT_PARAM (WardParams, ay),
                                                  CLOSURE_FINISH_PARAM(WardParams) } },
        { "microfacet", MICROFACET_ID,          { CLOSURE_STRING_PARAM(MicrofacetParams, dist),
                                                  CLOSURE_VECTOR_PARAM(MicrofacetParams, N),
                                                  CLOSURE_VECTOR_PARAM(MicrofacetParams, U),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, xalpha),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, yalpha),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, eta),
                                                  CLOSURE_INT_PARAM   (MicrofacetParams, refract),
                                                  CLOSURE_FINISH_PARAM(MicrofacetParams) } },
        { "reflection" , REFLECTION_ID,         { CLOSURE_VECTOR_PARAM(ReflectionParams, N),
                                                  CLOSURE_FINISH_PARAM(ReflectionParams) } },
        { "reflection" , FRESNEL_REFLECTION_ID, { CLOSURE_VECTOR_PARAM(ReflectionParams, N),
                                                  CLOSURE_FLOAT_PARAM (ReflectionParams, eta),
                                                  CLOSURE_FINISH_PARAM(ReflectionParams) } },
        { "refraction" , REFRACTION_ID,         { CLOSURE_VECTOR_PARAM(RefractionParams, N),
                                                  CLOSURE_FLOAT_PARAM (RefractionParams, eta),
                                                  CLOSURE_FINISH_PARAM(RefractionParams) } },
        { "transparent", TRANSPARENT_ID,        { CLOSURE_FINISH_PARAM(EmptyParams) } },
        // mark end of the array
        { NULL, 0, {} }
    };

    for (int i = 0; builtins[i].name; i++) {
        shadingsys->register_closure(
            builtins[i].name,
            builtins[i].id,
            builtins[i].params,
            NULL, NULL);
    }
}

OSL_NAMESPACE_EXIT

namespace { // anonymous namespace

template <int trans>
struct Diffuse final : public BSDF, DiffuseParams {
    Diffuse(const DiffuseParams& params) : BSDF(), DiffuseParams(params) { if (trans) N = -N; }
    virtual float eval  (const OSL::ShaderGlobals& /*sg*/, const OSL::Vec3& wi, float& pdf) const {
        pdf = std::max(N.dot(wi), 0.0f) * float(M_1_PI);
        return 1.0f;
    }
    virtual float sample(const OSL::ShaderGlobals& /*sg*/, float rx, float ry, float /*rz*/, OSL::Dual2<OSL::Vec3>& wi, float& pdf) const {
        Vec3 out_dir;
        Sampling::sample_cosine_hemisphere(N, rx, ry, out_dir, pdf);
        wi = out_dir; // FIXME: leave derivs 0?
        return 1;
    }
};

struct OrenNayar final : public BSDF, OrenNayarParams {
   OrenNayar(const OrenNayarParams& params) : BSDF(), OrenNayarParams(params) {
      // precompute some constants
      float s2 = sigma * sigma;
      A = 1 - 0.50f * s2 / (s2 + 0.33f);
      B =     0.45f * s2 / (s2 + 0.09f);
   }
   virtual float eval  (const OSL::ShaderGlobals& sg, const OSL::Vec3& wi, float& pdf) const {
      float NL =  N.dot(wi);
      float NV = -N.dot(sg.I);
      if (NL > 0 && NV > 0) {
         pdf = NL * float(M_1_PI);

         // Simplified math from: "A tiny improvement of Oren-Nayar reflectance model"
         // by Yasuhiro Fujii
         // http://mimosa-pudica.net/improved-oren-nayar.html
         // NOTE: This is using the math to match the original ON model, not the tweak
         // proposed in the text which is a slightly different BRDF
         float LV = -sg.I.dot(wi);
         float s = LV - NL * NV;
         float stinv = s > 0 ? s / std::max(NL, NV) : 0.0f;
         return A + B * stinv;
      }
      return pdf = 0;
   }
   virtual float sample(const OSL::ShaderGlobals& sg, float rx, float ry, float /*rz*/, OSL::Dual2<OSL::Vec3>& wi, float& pdf) const {
       Vec3 out_dir;
       Sampling::sample_cosine_hemisphere(N, rx, ry, out_dir, pdf);
       wi = out_dir; // leave derivs 0?
       const float NL =  N.dot(wi.val());
       const float NV = -N.dot(sg.I);
       if (NL > 0 && NV > 0) {
           float LV = -sg.I.dot(wi.val());
           float s = LV - NL * NV;
           float stinv = s > 0 ? s / std::max(NL, NV) : 0.0f;
           return A + B * stinv;
       }
       return 0;
   }
private:
   float A, B;
};

struct Phong final : public BSDF, PhongParams {
    Phong(const PhongParams& params) : BSDF(), PhongParams(params) {}
    virtual float eval  (const OSL::ShaderGlobals& sg, const OSL::Vec3& wi, float& pdf) const {
        float cosNI =  N.dot(wi);
        float cosNO = -N.dot(sg.I);
        if (cosNI > 0 && cosNO > 0) {
           // reflect the view vector
           Vec3 R = (2 * cosNO) * N + sg.I;
           float cosRI = R.dot(wi);
           if (cosRI > 0) {
               pdf = (exponent + 1) * float(M_1_PI / 2) * OIIO::fast_safe_pow(cosRI, exponent);
               return cosNI * (exponent + 2) / (exponent + 1);
           }
        }
        return pdf = 0;
    }
    virtual float sample(const OSL::ShaderGlobals& sg, float rx, float ry, float /*rz*/, OSL::Dual2<OSL::Vec3>& wi, float& pdf) const {
        float cosNO = -N.dot(sg.I);
        if (cosNO > 0) {
            // reflect the view vector
            Vec3 R = (2 * cosNO) * N + sg.I;
            TangentFrame tf(R);
            float phi = 2 * float(M_PI) * rx;
            float sp, cp;
            OIIO::fast_sincos(phi, &sp, &cp);
            float cosTheta = OIIO::fast_safe_pow(ry, 1 / (exponent + 1));
            float sinTheta2 = 1 - cosTheta * cosTheta;
            float sinTheta = sinTheta2 > 0 ? sqrtf(sinTheta2) : 0;
            wi = tf.get(cp * sinTheta,
                        sp * sinTheta,
                        cosTheta); // leave derivs 0?
            float cosNI = N.dot(wi.val());
            if (cosNI > 0) {
                pdf = (exponent + 1) * float(M_1_PI / 2) * OIIO::fast_safe_pow(cosTheta, exponent);
                return cosNI * (exponent + 2) / (exponent + 1);
            }
        }
        return pdf = 0;
    }
};

struct Ward final : public BSDF, WardParams {
    Ward(const WardParams& params) : BSDF(), WardParams(params) {}
    virtual float eval  (const OSL::ShaderGlobals& sg, const OSL::Vec3& wi, float& pdf) const {
        float cosNO = -N.dot(sg.I);
        float cosNI =  N.dot(wi);
        if (cosNI > 0 && cosNO > 0) {
            // get half vector and get x,y basis on the surface for anisotropy
            Vec3 H = wi - sg.I;
            H.normalize();  // normalize needed for pdf
            TangentFrame tf(N, T);
            // eq. 4
            float dotx = tf.getx(H) / ax;
            float doty = tf.gety(H) / ay;
            float dotn = tf.getz(H);
            float oh = H.dot(wi);
            float e = OIIO::fast_exp(-(dotx * dotx + doty * doty) / (dotn * dotn));
            float c = float(4 * M_PI) * ax * ay;
            float k = oh * dotn * dotn * dotn;
            pdf = e / (c * k);
            return k * sqrtf(cosNI / cosNO);
        }
        return 0;
    }
    virtual float sample(const OSL::ShaderGlobals& sg, float rx, float ry, float /*rz*/, OSL::Dual2<OSL::Vec3>& wi, float& pdf) const {
        float cosNO = -N.dot(sg.I);
        if (cosNO > 0) {
            // get x,y basis on the surface for anisotropy
            TangentFrame tf(N, T);
            // generate random angles for the half vector
            float phi = 2 * float(M_PI) * rx;
            float sp, cp;
            OIIO::fast_sincos(phi, &sp, &cp);
            float cosPhi = ax * cp;
            float sinPhi = ay * sp;
            float k = 1 / sqrtf(cosPhi * cosPhi + sinPhi * sinPhi);
            cosPhi *= k;
            sinPhi *= k;

            // eq. 6
            // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
            //                  and sin(atan(x)) == x/sqrt(1+x^2)
            float thetaDenom = (cosPhi * cosPhi) / (ax * ax) + (sinPhi * sinPhi) / (ay * ay);
            float tanTheta2 = -OIIO::fast_log(1 - ry) / thetaDenom;
            float cosTheta  = 1 / sqrtf(1 + tanTheta2);
            float sinTheta  = cosTheta * sqrtf(tanTheta2);

            Vec3 h; // already normalized because expressed from spherical coordinates
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
            float oh = -h.dot(sg.I);
            wi = 2 * oh * h + sg.I; // TODO: leave derivs 0?
            if (sg.Ng.dot(wi.val()) > 0) {
                float cosNI = N.dot(wi.val());
                if (cosNI > 0) {
                    // eq. 9
                    float e = OIIO::fast_exp(-(dotx * dotx + doty * doty) / (dotn * dotn));
                    float c = float(4 * M_PI) * ax * ay;
                    float k = oh * dotn * dotn * dotn;
                    pdf = e / (c * k);
                    return k * sqrtf(cosNI / cosNO);
                }
            }
        }
        return 0;
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
	static float F(const float tan_m2) {
        return 1 / (float(M_PI) * (1 + tan_m2) * (1 + tan_m2));
    }

    static float Lambda(const float a2) {
        return 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / a2));
    }

    static Vec2 sampleSlope(float cos_theta, float randu, float randv) {
        // GGX
        Vec2 slope;
        /* sample slope_x */

        float c = cos_theta < 1e-6f ? 1e-6f : cos_theta;
        float Q = (1 + c) * randu - c;
        float num = c * sqrtf((1 - c) * (1 + c)) - Q * sqrtf((1 - Q) * (1 + Q));
        float den = (Q - c) * (Q + c);
        float eps = 1.0f / 4294967296.0f;
        den = fabsf(den) < eps ? copysignf(eps, den) : den;
        slope.x = num / den;

        /* sample slope_y */
        float Ru = 1 - 2 * randv;
        float u2 = fabsf(Ru);
        float z = (u2 * (u2 * (u2 * 0.27385f - 0.73369f) + 0.46341f)) /
                  (u2 * (u2 * (u2 * 0.093073f + 0.309420f) - 1.0f) + 0.597999f);
        slope.y = copysignf(1.0f, Ru) * z * sqrtf(1.0f + slope.x * slope.x);

        return slope;
    }
};

struct BeckmannDist {
	static float F(const float tan_m2) {
        return float(1 / M_PI) * OIIO::fast_exp(-tan_m2);
    }

    static float Lambda(const float a2) {
        const float a = sqrtf(a2);
        return a < 1.6f ? (1.0f - 1.259f * a + 0.396f * a2) / (3.535f * a + 2.181f * a2) : 0.0f;
    }

    static Vec2 sampleSlope(float cos_theta, float randu, float randv) {
        const float SQRT_PI_INV = 1 / sqrtf(float(M_PI));
        float ct = cos_theta < 1e-6f ? 1e-6f : cos_theta;
        float tanThetaI = sqrtf(1 - ct * ct) / ct;
        float cotThetaI = 1 / tanThetaI;

        /* sample slope X */
        // compute a coarse approximation using the approximation:
        // exp(-ierf(x)^2) ~= 1 - x * x
        // solve y = 1 + b + K * (1 - b * b)
        float c = OIIO::fast_erf(cotThetaI);
        float K = tanThetaI * SQRT_PI_INV;
        float yApprox = randu * (1.0f + c + K * (1 - c * c));
        float yExact  = randu * (1.0f + c + K * OIIO::fast_exp(-cotThetaI * cotThetaI));
        float b = K > 0 ? (0.5f - sqrtf(K * (K - yApprox + 1.0f) + 0.25f)) / K : yApprox - 1.0f;

        // perform newton step to refine toward the true root
        float invErf = OIIO::fast_ierf(b);
        float value  = 1.0f + b + K * OIIO::fast_exp(-invErf * invErf) - yExact;

        // check if we are close enough already
        // this also avoids NaNs as we get close to the root
        Vec2 slope;
        if (fabsf(value) > 1e-6f) {
            b -= value / (1 - invErf * tanThetaI); // newton step 1
            invErf = OIIO::fast_ierf(b);
            value  = 1.0f + b + K * OIIO::fast_exp(-invErf * invErf) - yExact;
            b -= value / (1 - invErf * tanThetaI); // newton step 2
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


template <typename Distribution, int Refract>
struct Microfacet final : public BSDF, MicrofacetParams {
    Microfacet(const MicrofacetParams& params) : BSDF(),
        MicrofacetParams(params),
        tf(U == Vec3(0) || xalpha == yalpha ? TangentFrame(N) : TangentFrame(N, U)) { }
    virtual float albedo(const ShaderGlobals& sg) const {
        if (Refract == 2) return 1.0f;
        // FIXME: this heuristic is not particularly good, and looses energy
        // compared to the reference solution
        float fr = fresnel_dielectric(-N.dot(sg.I), eta);
        return Refract ? 1 - fr : fr;
    }
    virtual float eval  (const OSL::ShaderGlobals& sg, const OSL::Vec3& wi, float& pdf) const {
        Vec3 wo = -sg.I;
    	const Vec3 wo_l = tf.tolocal(wo);
    	const Vec3 wi_l = tf.tolocal(wi);
        if (Refract == 0 || Refract == 2) {
            if (wo_l.z > 0 && wi_l.z > 0) {
            	const Vec3 m = (wi_l + wo_l).normalize();
                const float D = evalD(m);
                const float Lambda_o = evalLambda(wo_l);
                const float Lambda_i = evalLambda(wi_l);
                const float G2 = evalG2(Lambda_o, Lambda_i);
                const float G1 = evalG1(Lambda_o);

                const float Fr = fresnel_dielectric(m.dot(wo_l), eta);
                pdf = (G1 * D * 0.25f) / wo_l.z;
                float out = G2 / G1;
                if (Refract == 2) {
                    pdf *= Fr;
                    return out;
                } else {
                    return out * Fr;
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
               const float Ft = 1.0f - fresnel_dielectric(cosHO, eta);
               if (Ft > 0) { // skip work in case of TIR
                  const float cosHI = Ht.dot(wi_l);
                  // eq. 33: first we calculate D(m) with m=Ht:
                  const float cosThetaM = Ht.z;
                  if (cosThetaM <= 0.0f)
                     return 0;
                  const float Dt = evalD(Ht);
                  const float Lambda_o = evalLambda(wo_l);
                  const float Lambda_i = evalLambda(wi_l);
                  const float G2 = evalG2(Lambda_o, Lambda_i);
                  const float G1 = evalG1(Lambda_o);

                  // probability
                  float invHt2 = 1 / ht.dot(ht);
                  pdf =  (fabsf(cosHI * cosHO) * (eta * eta) * (G1 * Dt) * invHt2) / wo_l.z;
                  float out = G2 / G1;
                  if (Refract == 2) {
                      pdf *= Ft;
                      return out;
                  } else {
                      return out * Ft;
                  }
               }
           }
        }
        return pdf = 0;
    }

    virtual float sample(const OSL::ShaderGlobals& sg, float rx, float ry, float rz, OSL::Dual2<OSL::Vec3>& wi, float& pdf) const {
    	const Vec3 wo_l = tf.tolocal(-sg.I);
    	const float cosNO = wo_l.z;
    	if (!(cosNO > 0)) return pdf = 0;
        const Vec3 m = sampleMicronormal(wo_l, rx, ry);
        const float cosMO = m.dot(wo_l);
        const float F = fresnel_dielectric(cosMO, eta);
        if (Refract == 0 || (Refract == 2 && rz < F)) {
            // measure fresnel to decide which lobe to sample
            const Vec3 wi_l = (2.0f * cosMO) * m - wo_l;
            const float D = evalD(m);
            const float Lambda_o = evalLambda(wo_l);
            const float Lambda_i = evalLambda(wi_l);

            const float G2 = evalG2(Lambda_o, Lambda_i);
            const float G1 = evalG1(Lambda_o);

            wi = tf.toworld(wi_l);

            pdf = (G1 * D * 0.25f) / cosNO;
            float out = G2 / G1;
            if (Refract == 2) {
                pdf *= F;
                return out;
            } else
                return F * out;
        } else {
            const Vec3 M = tf.toworld(m);
            float Ft = fresnel_refraction (sg.I, M, eta, wi);
            const Vec3 wi_l = tf.tolocal(wi.val());
            const float cosHO = m.dot(wo_l);
            const float cosHI = m.dot(wi_l);
            const float D = evalD(m);
            const float Lambda_o = evalLambda(wo_l);
            const float Lambda_i = evalLambda(wi_l);

            const float G2 = evalG2(Lambda_o, Lambda_i);
            const float G1 = evalG1(Lambda_o);

            const Vec3 ht = -(eta * wi_l + wo_l);
            const float invHt2 = 1.0f / ht.dot(ht);

            pdf = (fabsf(cosHI * cosHO) * (eta * eta) * (G1 * D) * invHt2) / fabsf(wo_l.z);
            float out = G2 / G1;
            if (Refract == 2) {
                pdf *= Ft;
                return out;
            } else
                return Ft * out;
        }
        return pdf = 0;
    }

private:
    static float SQR(float x) {
    	return x * x;
    }

    float evalLambda(const Vec3 w) const {
        float cosTheta2  = SQR(w.z);
        /* Have these two multiplied by sinTheta^2 for convenience */
        float cosPhi2st2 = SQR(w.x * xalpha);
        float sinPhi2st2 = SQR(w.y * yalpha);
        return Distribution::Lambda(cosTheta2 / (cosPhi2st2 + sinPhi2st2));
    }

    static float evalG2(float Lambda_i, float Lambda_o) {
    	// correlated masking-shadowing
        return 1 / (Lambda_i + Lambda_o + 1);
    }

    static float evalG1(float Lambda_v) {
        return 1 / (Lambda_v + 1);
    }

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

    Vec3 sampleMicronormal(const Vec3 wo, float randu, float randv) const {
        /* Project wo and stretch by alpha values */
        Vec3 swo = wo;
        swo.x *= xalpha;
        swo.y *= yalpha;
        swo = swo.normalize();

        // figure out angles for the incoming vector
        float cos_theta = std::max(swo.z, 0.0f);
        float cos_phi = 1;
        float sin_phi = 0;
        /* Normal incidence special case gets phi 0 */
        if (cos_theta < 0.99999f)
        {
            float invnorm = 1 / sqrtf(SQR(swo.x) + SQR(swo.y));
            cos_phi = swo.x * invnorm;
            sin_phi = swo.y * invnorm;
        }

        Vec2 slope = Distribution::sampleSlope(cos_theta, randu, randv);

        /* Rotate and unstretch slopes */
        Vec2 s(cos_phi * slope.x - sin_phi * slope.y,
               sin_phi * slope.x + cos_phi * slope.y);
        s.x *= xalpha;
        s.y *= yalpha;

        float mlen = sqrtf(s.x * s.x + s.y * s.y + 1);
        Vec3 m(fabsf(s.x) < mlen ? -s.x / mlen : 1.0f,
        	   fabsf(s.y) < mlen ? -s.y / mlen : 1.0f,
               1.0f / mlen);
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

struct Reflection final : public BSDF, ReflectionParams {
    Reflection(const ReflectionParams& params) : BSDF(), ReflectionParams(params) {}
    virtual float albedo(const ShaderGlobals& sg) const {
        float cosNO = -N.dot(sg.I);
        if (cosNO > 0)
            return fresnel_dielectric(cosNO, eta);
        return 1;
    }
    virtual float eval  (const OSL::ShaderGlobals& /*sg*/, const OSL::Vec3& /*wi*/, float& pdf) const {
        return pdf = 0;
    }
    virtual float sample(const OSL::ShaderGlobals& sg, float /*rx*/, float /*ry*/, float /*rz*/, OSL::Dual2<OSL::Vec3>& wi, float& pdf) const {
        // only one direction is possible
        OSL::Dual2<OSL::Vec3> I = OSL::Dual2<OSL::Vec3>(sg.I, sg.dIdx, sg.dIdy);
        OSL::Dual2<float> cosNO = -dot(N, I);
        if (cosNO.val() > 0) {
            wi = (2 * cosNO) * N + I;
            pdf = std::numeric_limits<float>::infinity();
            return fresnel_dielectric(cosNO.val(), eta);
        }
        return pdf = 0;
    }
};

struct Refraction final : public BSDF, RefractionParams {
    Refraction(const RefractionParams& params) : BSDF(), RefractionParams(params) {}
    virtual float albedo(const ShaderGlobals& sg) const {
        float cosNO = -N.dot(sg.I);
        return 1 - fresnel_dielectric(cosNO, eta);
    }
    virtual float eval  (const OSL::ShaderGlobals& /*sg*/, const OSL::Vec3& /*wi*/, float& pdf) const {
        return pdf = 0;
    }
    virtual float sample(const OSL::ShaderGlobals& sg, float /*rx*/, float /*ry*/, float /*rz*/, OSL::Dual2<OSL::Vec3>& wi, float& pdf) const {
        OSL::Dual2<OSL::Vec3> I = OSL::Dual2<OSL::Vec3>(sg.I, sg.dIdx, sg.dIdy);
        pdf = std::numeric_limits<float>::infinity();
        return fresnel_refraction(I, N, eta, wi);
    }
};

struct Transparent final : public BSDF {
    Transparent(const int& /*dummy*/) : BSDF() {}
    virtual float eval  (const OSL::ShaderGlobals& /*sg*/, const OSL::Vec3& /*wi*/, float& pdf) const {
        return pdf = 0;
    }
    virtual float sample(const OSL::ShaderGlobals& sg, float /*rx*/, float /*ry*/, float /*rz*/, OSL::Dual2<OSL::Vec3>& wi, float& pdf) const {
        wi = OSL::Dual2<OSL::Vec3>(sg.I, sg.dIdx, sg.dIdy);
        pdf = std::numeric_limits<float>::infinity();
        return 1;
    }
};


// recursively walk through the closure tree, creating bsdfs as we go
void process_closure (ShadingResult& result, const ClosureColor* closure, const Color3& w, bool light_only) {
   static const ustring u_ggx("ggx");
   static const ustring u_beckmann("beckmann");
   static const ustring u_default("default");
   if (!closure)
       return;
   switch (closure->id) {
       case ClosureColor::MUL: {
           Color3 cw = w * closure->as_mul()->weight;
           process_closure(result, closure->as_mul()->closure, cw, light_only);
           break;
       }
       case ClosureColor::ADD: {
           process_closure(result, closure->as_add()->closureA, w, light_only);
           process_closure(result, closure->as_add()->closureB, w, light_only);
           break;
       }
       default: {
           const ClosureComponent* comp = closure->as_comp();
           Color3 cw = w * comp->w;
           if (comp->id == EMISSION_ID)
               result.Le += cw;
           else if (!light_only) {
               bool ok = false;
               switch (comp->id) {
                   case DIFFUSE_ID:            ok = result.bsdf.add_bsdf<Diffuse<0>, DiffuseParams   >(cw, *comp->as<DiffuseParams>  ()); break;
                   case OREN_NAYAR_ID:         ok = result.bsdf.add_bsdf<OrenNayar , OrenNayarParams >(cw, *comp->as<OrenNayarParams>()); break;
                   case TRANSLUCENT_ID:        ok = result.bsdf.add_bsdf<Diffuse<1>, DiffuseParams   >(cw, *comp->as<DiffuseParams>  ()); break;
                   case PHONG_ID:              ok = result.bsdf.add_bsdf<Phong     , PhongParams     >(cw, *comp->as<PhongParams>    ()); break;
                   case WARD_ID:               ok = result.bsdf.add_bsdf<Ward      , WardParams      >(cw, *comp->as<WardParams>     ()); break;
                   case MICROFACET_ID: {
                       const MicrofacetParams* mp = comp->as<MicrofacetParams>();
                       if (mp->dist == u_ggx) {
                           switch (mp->refract) {
                               case 0: ok = result.bsdf.add_bsdf<MicrofacetGGXRefl, MicrofacetParams>(cw, *mp); break;
                               case 1: ok = result.bsdf.add_bsdf<MicrofacetGGXRefr, MicrofacetParams>(cw, *mp); break;
                               case 2: ok = result.bsdf.add_bsdf<MicrofacetGGXBoth, MicrofacetParams>(cw, *mp); break;
                           }
                       } else if (mp->dist == u_beckmann || mp->dist == u_default) {
                           switch (mp->refract) {
                               case 0: ok = result.bsdf.add_bsdf<MicrofacetBeckmannRefl, MicrofacetParams>(cw, *mp); break;
                               case 1: ok = result.bsdf.add_bsdf<MicrofacetBeckmannRefr, MicrofacetParams>(cw, *mp); break;
                               case 2: ok = result.bsdf.add_bsdf<MicrofacetBeckmannBoth, MicrofacetParams>(cw, *mp); break;
                           }
                       }
                       break;
                   }
                   case REFLECTION_ID:
                   case FRESNEL_REFLECTION_ID: ok = result.bsdf.add_bsdf<Reflection , ReflectionParams>(cw, *comp->as<ReflectionParams>()); break;
                   case REFRACTION_ID:         ok = result.bsdf.add_bsdf<Refraction , RefractionParams>(cw, *comp->as<RefractionParams>()); break;
                   case TRANSPARENT_ID:        ok = result.bsdf.add_bsdf<Transparent, int             >(cw, 0); break;
               }
               OSL_ASSERT(ok && "Invalid closure invoked in surface shader");
           }
           break;
       }
   }
}

} // anonymous namespace

OSL_NAMESPACE_ENTER

void process_closure(ShadingResult& result, const ClosureColor* Ci, bool light_only) {
    ::process_closure(result, Ci, Color3(1, 1, 1), light_only);
}

Vec3 process_background_closure(const ClosureColor* closure) {
    if (!closure) return Vec3(0, 0, 0);
    switch (closure->id) {
           case ClosureColor::MUL: {
               return closure->as_mul()->weight * process_background_closure(closure->as_mul()->closure);
           }
           case ClosureColor::ADD: {
               return process_background_closure(closure->as_add()->closureA) +
                      process_background_closure(closure->as_add()->closureB);
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
