#include "shading.h"
#include "sampling.h"
#include "OSL/genclosure.h"
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
    MICROFACET_GGX_ID,
    MICROFACET_GGX_REFR_ID,
    MICROFACET_BECKMANN_ID,
    MICROFACET_BECKMANN_REFR_ID,
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
struct MicrofacetParams { Vec3 N; float alpha, eta; };

} // anonymous namespace

OSL_NAMESPACE_ENTER

void register_closures(OSL::ShadingSystem* shadingsys) {
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
        { "microfacet_ggx", MICROFACET_GGX_ID,  { CLOSURE_VECTOR_PARAM(MicrofacetParams, N),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, alpha),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, eta),
                                                  CLOSURE_FINISH_PARAM(MicrofacetParams) } },
        { "microfacet_ggx_refraction", MICROFACET_GGX_REFR_ID,
                                                { CLOSURE_VECTOR_PARAM(MicrofacetParams, N),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, alpha),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, eta),
                                                  CLOSURE_FINISH_PARAM(MicrofacetParams) } },
        { "microfacet_beckmann", MICROFACET_BECKMANN_ID,
                                                { CLOSURE_VECTOR_PARAM(MicrofacetParams, N),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, alpha),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, eta),
                                                  CLOSURE_FINISH_PARAM(MicrofacetParams) } },
        { "microfacet_beckmann_refraction", MICROFACET_BECKMANN_REFR_ID,
                                                { CLOSURE_VECTOR_PARAM(MicrofacetParams, N),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, alpha),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, eta),
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
struct Diffuse : public BSDF, DiffuseParams {
    Diffuse(const DiffuseParams& params) : BSDF(false), DiffuseParams(params) { if (trans) N = -N; }
    virtual float eval  (const OSL::ShaderGlobals& sg, const OSL::Vec3& wi, float& pdf) const {
        pdf = std::max(N.dot(wi), 0.0f) * float(M_1_PI);
        return pdf;
    }
    virtual float sample(const OSL::ShaderGlobals& sg, float rx, float ry, OSL::Dual2<OSL::Vec3>& wi, float& invpdf) const {
        Vec3 out_dir;
        Sampling::sample_cosine_hemisphere(N, rx, ry, out_dir, invpdf);
        wi = out_dir; // FIXME: leave derivs 0?
        return 1;
    }
};

struct OrenNayar : public BSDF, OrenNayarParams {
   OrenNayar(const OrenNayarParams& params) : BSDF(false), OrenNayarParams(params) {
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
         // project L and V down to the plane defined by N
         Vec3 Lproj = (   wi - NL * N).normalize();
         Vec3 Vproj = (-sg.I - NV * N).normalize();
         // cosine of angle between vectors
         float cos_phi_diff = Lproj.dot(Vproj);
         if (cos_phi_diff > 0) {
            // take advantage of function monoticity to save inverse trig ops
            //     theta_i = acos(N.L)
            //     theta_r = acos(N.V)
            //     alpha   = max(    theta_i,      theta_r )
            // sin_alpha   = max(sin(theta_i), sin(theta_r))
            //     beta    = min(    theta_i ,     theta_r )
            // tan_beta    = min(tan(theta_i), tan(theta)r))
            float sin_theta_i2 = 1 - NL * NL;
            float sin_theta_r2 = 1 - NV * NV;
            float sin_alpha, tan_beta;
            if (sin_theta_i2 > sin_theta_r2) {
               sin_alpha = sin_theta_i2 > 0 ? sqrtf(sin_theta_i2) : 0.0f;
               tan_beta  = sin_theta_r2 > 0 ? sqrtf(sin_theta_r2) / NV : 0.0f;
            } else {
               sin_alpha = sin_theta_r2 > 0 ? sqrtf(sin_theta_r2) : 0.0f;
               tan_beta  = sin_theta_i2 > 0 ? sqrtf(sin_theta_i2) / NL : 0.0f;
            }
            return pdf * (A + B * cos_phi_diff * sin_alpha * tan_beta);
         } else
            return pdf * A;
      }
      return pdf = 0;
   }
   virtual float sample(const OSL::ShaderGlobals& sg, float rx, float ry, OSL::Dual2<OSL::Vec3>& wi, float& invpdf) const {
       Vec3 out_dir; float pdf;
       Sampling::sample_cosine_hemisphere(N, rx, ry, out_dir, invpdf);
       wi = out_dir; // leave derivs 0?
       return eval(sg, out_dir, pdf) * invpdf;
   }
private:
   float A, B;
};

struct Phong : public BSDF, PhongParams {
    Phong(const PhongParams& params) : BSDF(false), PhongParams(params) {}
    virtual float eval  (const OSL::ShaderGlobals& sg, const OSL::Vec3& wi, float& pdf) const {
        float cosNI =  N.dot(wi);
        float cosNO = -N.dot(sg.I);
        if (cosNI > 0 && cosNO > 0) {
           // reflect the view vector
           Vec3 R = (2 * cosNO) * N + sg.I;
           float cosRI = R.dot(wi);
           if (cosRI > 0) {
               float common = 0.5f * float(M_1_PI) * powf(cosRI, exponent);
               float out = cosNI * (exponent + 2) * common;
               pdf = (exponent + 1) * common;
               return out;
           }
        }
        return pdf = 0;
    }
    virtual float sample(const OSL::ShaderGlobals& sg, float rx, float ry, OSL::Dual2<OSL::Vec3>& wi, float& invpdf) const {
        float cosNO = -N.dot(sg.I);
        if (cosNO > 0) {
            // reflect the view vector
            Vec3 R = (2 * cosNO) * N + sg.I;
            TangentFrame tf(R);
            float phi = 2 * float(M_PI) * rx;
            float cosTheta = powf(ry, 1 / (exponent + 1));
            float sinTheta2 = 1 - cosTheta * cosTheta;
            float sinTheta = sinTheta2 > 0 ? sqrtf(sinTheta2) : 0;
            wi = tf.get(cosf(phi) * sinTheta,
                        sinf(phi) * sinTheta,
                        cosTheta); // leave derivs 0?
            float cosNI = N.dot(wi.val());
            if (cosNI > 0) {
                float d = 1 / (exponent + 1);
                invpdf = 2 * float(M_PI) * powf(cosTheta, -exponent) * d;
                return cosNI * (exponent + 2) * d;
            }
        }
        return invpdf = 0;
    }
};

struct Ward : public BSDF, WardParams {
    Ward(const WardParams& params) : BSDF(false), WardParams(params) {}
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
            float exp_arg = (dotx * dotx + doty * doty) / (dotn * dotn);
            float denom = (4 * float(M_PI) * ax * ay * sqrtf(cosNO * cosNI));
            float exp_val = expf(-exp_arg);
            float out = cosNI * exp_val / denom;
            float oh = H.dot(wi);
            denom = 4 * float(M_PI) * ax * ay * oh * dotn * dotn * dotn;
            pdf = exp_val / denom;
            return out;
        }
        return 0;
    }
    virtual float sample(const OSL::ShaderGlobals& sg, float rx, float ry, OSL::Dual2<OSL::Vec3>& wi, float& invpdf) const {
        float cosNO = -N.dot(sg.I);
        if (cosNO > 0) {
            // get x,y basis on the surface for anisotropy
            TangentFrame tf(N, T);
            // generate random angles for the half vector
            float phi = 2 * float(M_PI) * rx;
            float cosPhi = ax * cosf(phi);
            float sinPhi = ay * sinf(phi);
            float k = 1 / sqrtf(cosPhi * cosPhi + sinPhi * sinPhi);
            cosPhi *= k;
            sinPhi *= k;

            // eq. 6
            // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
            //                  and sin(atan(x)) == x/sqrt(1+x^2)
            float thetaDenom = (cosPhi * cosPhi) / (ax * ax) + (sinPhi * sinPhi) / (ay * ay);
            float tanTheta2 = -logf(1 - ry) / thetaDenom;
            float cosTheta  = 1 / sqrtf(1 + tanTheta2);
            float sinTheta  = cosTheta * sqrtf(tanTheta2);

            Vec3 h; // already normalized becaused expressed from spherical coordinates
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
                    float e = expf(-(dotx * dotx + doty * doty) / (dotn * dotn));
                    float c = 4 * float(M_PI) * ax * ay;
                    float k = oh * dotn * dotn * dotn;
                    invpdf = (c * k) / e;
                    return k * sqrtf(cosNI / cosNO);
                }
            }
        }
        return 0;
    }
};

struct GGXDist {
    GGXDist(float alpha) : alpha2(alpha * alpha) {}

    float D(float cosThetaM) const {
        // eq. 33: calculate D(m) with m=Hr:
        float cosThetaM2 = cosThetaM * cosThetaM;
        float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;
        float cosThetaM4 = cosThetaM2 * cosThetaM2;
        return alpha2 / (float(M_PI) * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));
    }
    float G(float cosNx) const {
        // eq. 34: calculate G
        return 2 / (1 + sqrtf(1 + alpha2 * (1 - cosNx * cosNx) / (cosNx * cosNx)));
    }
    Vec3 sample(float rx, float ry) const {
        // generate a random microfacet normal m
        // eq. 35,36:
        // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
        //                  and sin(atan(x)) == x/sqrt(1+x^2)
        float tanThetaM2 = alpha2 * rx / (1 - rx);
        float cosThetaM  = 1 / sqrtf(1 + tanThetaM2);
        float sinThetaM  = cosThetaM * sqrtf(tanThetaM2);
        float phiM = 2 * float(M_PI) * ry;
        return Vec3(cosf(phiM) * sinThetaM,
                    sinf(phiM) * sinThetaM,
                    cosThetaM);
    }
private:
    float alpha2;
};

struct BeckmannDist {
    BeckmannDist(float alpha) : alpha2(alpha * alpha) {}
    float D(float cosThetaM) const {
        float cosThetaM2 = cosThetaM * cosThetaM;
        float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;
        float cosThetaM4 = cosThetaM2 * cosThetaM2;
        return expf(-tanThetaM2 / alpha2) / (float(M_PI) * alpha2 *  cosThetaM4);
    }
    float G(float cosNx) const {
        // eq. 26, 27: calculate G
        float ax = 1 / sqrtf(alpha2 * (1 - cosNx * cosNx) / (cosNx * cosNx));
        return ax < 1.6f ? (3.535f * ax + 2.181f * ax * ax) / (1 + 2.276f * ax + 2.577f * ax * ax) : 1.0f;
    }
    Vec3 sample(float rx, float ry) const {
        // eq. 35,36:
        // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
        //                  and sin(atan(x)) == x/sqrt(1+x^2)
        float tanThetaM = sqrtf(-alpha2 * logf(1 - rx));
        float cosThetaM = 1 / sqrtf(1 + tanThetaM * tanThetaM);
        float sinThetaM = cosThetaM * tanThetaM;
        float phiM = 2 * float(M_PI) * ry;
        return Vec3(cosf(phiM) * sinThetaM,
                    sinf(phiM) * sinThetaM,
                    cosThetaM);
    }
private:
    float alpha2;
};

template <typename Distribution, int Refract>
struct Microfacet : public BSDF, MicrofacetParams {
    Microfacet(const MicrofacetParams& params) : BSDF(false), MicrofacetParams(params), dist(params.alpha) {}
    virtual float albedo(const ShaderGlobals& sg) const {
        float fr = fresnel_dielectric(-N.dot(sg.I), eta);
        return Refract ? 1 - fr : fr;
    }
    virtual float eval  (const OSL::ShaderGlobals& sg, const OSL::Vec3& wi, float& pdf) const {
        pdf = 0;
        Vec3 wo = -sg.I;
        if (!Refract) {
            float cosNO = N.dot(wo);
            float cosNI = N.dot(wi);
            if (cosNI > 0 && cosNO > 0) {
                // get half vector
                Vec3 Hr = (wi + wo).normalize();
                // eq. 20: (F*G*D)/(4*in*on)
                float cosThetaM = N.dot(Hr);
                float Dr = dist.D(cosThetaM);
                // eq. 34: now calculate G1(i,m) and G1(o,m)
                float Gr = dist.G(cosNO) * dist.G(cosNI);
                // fresnel term between outgoing direction and microfacet
                float cosHO = Hr.dot(wo);
                float Fr = fresnel_dielectric(cosHO, eta);
                float out = (Fr * Gr * Dr) * 0.25f / cosNO;
                // eq. 24
                float pm = Dr * cosThetaM;
                // convert into pdf of the sampled direction
                // eq. 38 - but see also:
                // eq. 17 in http://www.graphics.cornell.edu/~bjw/wardnotes.pdf
                pdf = pm * 0.25f / cosHO;
                return out;
            }
        } else {
           Vec3 ht, Ht;
           float cosNO, cosNI;
           if (wi.dot(wo) <= 0 && (cosNO = N.dot(wo)) > 0.0f) {
               cosNI = N.dot(wi);
               // compute half-vector of the refraction (eq. 16)
               ht = -(eta * wi + wo);
               if (eta < 1.0f)
                  ht = -ht;
               Ht = ht.normalize();
               // compute fresnel term
               float cosHO = Ht.dot(wo);
               float Ft = 1.0f - fresnel_dielectric(cosHO, eta);
               if (Ft > 0) { // skip work in case of TIR
                  float cosHI = Ht.dot(wi);
                  // eq. 33: first we calculate D(m) with m=Ht:
                  float cosThetaM = N.dot(Ht);
                  if (cosThetaM <= 0.0f)
                     return 0;
                  float Dt = dist.D(cosThetaM);
                  // eq. 34: now calculate G1(i,m) and G1(o,m)
                  float Gt = dist.G(cosNO) * dist.G(cosNI);
                  // probability
                  float invHt2 = 1 / ht.dot(ht);
                  pdf = Dt * cosThetaM * (fabsf(cosHI) * (eta * eta)) * invHt2;
                  return (fabsf(cosHI * cosHO) * (eta * eta) * (Ft * Gt * Dt) * invHt2) / fabsf(cosNO);
               }
           }
        }
        return 0;
    }

    virtual float sample(const OSL::ShaderGlobals& sg, float rx, float ry, OSL::Dual2<OSL::Vec3>& wi, float& invpdf) const {
        // generate a random microfacet normal m
        TangentFrame tf(N);
        Vec3 m = dist.sample(rx, ry);
        m = tf.get(m.x, m.y, m.z);
        if (!Refract) {
            Vec3 wo = -sg.I;
            float cosMO = m.dot(wo);
            if (cosMO > 0) {
                // eq. 39 - compute actual reflected direction
                wi = 2 * cosMO * m - wo;
                float e = eval(sg, wi.val(), invpdf);
                invpdf = 1 / invpdf; // eval returned pdf, invert it
                return e * invpdf; // FIXME: simplify math here
           }
        } else {
            float Ft = fresnel_refraction (sg.I, m, eta, wi);
            if (Ft > 0) { // FIXME: find bug for refractive eval
                //float e = eval(sg, wi.val(), invpdf);
                //invpdf = 1 / invpdf; // eval returned pdf, invert it
                //return e * invpdf; // FIXME: simplify math here
                return invpdf = 1;
            }
        }
        return invpdf = 0;
    }

private:
    Distribution dist;
};

typedef Microfacet<GGXDist, 0> MicrofacetGGXRefl;
typedef Microfacet<GGXDist, 1> MicrofacetGGXRefr;
typedef Microfacet<BeckmannDist, 0> MicrofacetBeckmannRefl;
typedef Microfacet<BeckmannDist, 1> MicrofacetBeckmannRefr;

struct Reflection : public BSDF, ReflectionParams {
    Reflection(const ReflectionParams& params) : BSDF(true), ReflectionParams(params) {}
    virtual float albedo(const ShaderGlobals& sg) const {
        float cosNO = -N.dot(sg.I);
        if (cosNO > 0)
            return fresnel_dielectric(cosNO, eta);
        return 1;
    }
    virtual float eval  (const OSL::ShaderGlobals& sg, const OSL::Vec3& wi, float& pdf) const {
        return pdf = 0;
    }
    virtual float sample(const OSL::ShaderGlobals& sg, float rx, float ry, OSL::Dual2<OSL::Vec3>& wi, float& invpdf) const {
        // only one direction is possible
        OSL::Dual2<OSL::Vec3> I = OSL::Dual2<OSL::Vec3>(sg.I, sg.dIdx, sg.dIdy);
        OSL::Dual2<float> cosNO = -dot(N, I);
        if (cosNO.val() > 0) {
            wi = (2 * cosNO) * N + I;
            invpdf = 0;
            return fresnel_dielectric(cosNO.val(), eta);
        }
        return invpdf = 0;
    }
};

struct Refraction : public BSDF, RefractionParams {
    Refraction(const RefractionParams& params) : BSDF(true), RefractionParams(params) {}
    virtual float albedo(const ShaderGlobals& sg) const {
        float cosNO = -N.dot(sg.I);
        return 1 - fresnel_dielectric(cosNO, eta);
    }
    virtual float eval  (const OSL::ShaderGlobals& sg, const OSL::Vec3& wi, float& pdf) const {
        return pdf = 0;
    }
    virtual float sample(const OSL::ShaderGlobals& sg, float rx, float ry, OSL::Dual2<OSL::Vec3>& wi, float& invpdf) const {
        OSL::Dual2<OSL::Vec3> I = OSL::Dual2<OSL::Vec3>(sg.I, sg.dIdx, sg.dIdy);
        invpdf = 0;
        return fresnel_refraction(I, N, eta, wi);
    }
};

struct Transparent : public BSDF {
    Transparent(const int& dummy) : BSDF(true) {}
    virtual float eval  (const OSL::ShaderGlobals& sg, const OSL::Vec3& wi, float& pdf) const {
        return pdf = 0;
    }
    virtual float sample(const OSL::ShaderGlobals& sg, float rx, float ry, OSL::Dual2<OSL::Vec3>& wi, float& invpdf) const {
        wi = OSL::Dual2<OSL::Vec3>(sg.I, sg.dIdx, sg.dIdy);
        invpdf = 0;
        return 1;
    }
};


// recursively walk through the closure tree, creating bsdfs as we go
void process_closure (ShadingResult& result, const ClosureColor* closure, const Color3& w, bool light_only) {
   if (!closure)
       return;
   switch (closure->type) {
       case ClosureColor::MUL: {
           Color3 cw = w * ((const ClosureMul*) closure)->weight;
           process_closure(result, ((const ClosureMul*) closure)->closure, cw, light_only);
           break;
       }
       case ClosureColor::ADD: {
           process_closure(result, ((const ClosureAdd*) closure)->closureA, w, light_only);
           process_closure(result, ((const ClosureAdd*) closure)->closureB, w, light_only);
           break;
       }
       case ClosureColor::COMPONENT: {
           const ClosureComponent* comp = (const ClosureComponent*) closure;
           Color3 cw = w * comp->w;
           if (comp->id == EMISSION_ID)
               result.Le += float(M_1_PI) * cw;
           else if (!light_only) {
               bool ok = false;
               switch (comp->id) {
                   case DIFFUSE_ID:            ok = result.bsdf.add_bsdf<Diffuse<0>, DiffuseParams   >(cw, *(const DiffuseParams*   ) comp->data()); break;
                   case OREN_NAYAR_ID:         ok = result.bsdf.add_bsdf<OrenNayar , OrenNayarParams >(cw, *(const OrenNayarParams* ) comp->data()); break;
                   case TRANSLUCENT_ID:        ok = result.bsdf.add_bsdf<Diffuse<1>, DiffuseParams   >(cw, *(const DiffuseParams*   ) comp->data()); break;
                   case PHONG_ID:              ok = result.bsdf.add_bsdf<Phong     , PhongParams     >(cw, *(const PhongParams*     ) comp->data()); break;
                   case WARD_ID:               ok = result.bsdf.add_bsdf<Ward      , WardParams      >(cw, *(const WardParams*      ) comp->data()); break;
                   case MICROFACET_GGX_ID:           ok = result.bsdf.add_bsdf<MicrofacetGGXRefl     , MicrofacetParams>(cw, *(const MicrofacetParams*) comp->data()); break;
                   case MICROFACET_GGX_REFR_ID:      ok = result.bsdf.add_bsdf<MicrofacetGGXRefr     , MicrofacetParams>(cw, *(const MicrofacetParams*) comp->data()); break;
                   case MICROFACET_BECKMANN_ID:      ok = result.bsdf.add_bsdf<MicrofacetBeckmannRefl, MicrofacetParams>(cw, *(const MicrofacetParams*) comp->data()); break;
                   case MICROFACET_BECKMANN_REFR_ID: ok = result.bsdf.add_bsdf<MicrofacetBeckmannRefr, MicrofacetParams>(cw, *(const MicrofacetParams*) comp->data()); break;
                   case REFLECTION_ID:
                   case FRESNEL_REFLECTION_ID: ok = result.bsdf.add_bsdf<Reflection, ReflectionParams>(cw, *(const ReflectionParams*) comp->data()); break;
                   case REFRACTION_ID:         ok = result.bsdf.add_bsdf<Refraction, RefractionParams>(cw, *(const RefractionParams*) comp->data()); break;
                   case TRANSPARENT_ID:        ok = result.bsdf.add_bsdf<Transparent, int            >(cw, 0); break;
               }
               ASSERT(ok && "Invalid closure invoked in surface shader");
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
    switch (closure->type) {
           case ClosureColor::MUL: {
               Color3 cw = ((const ClosureMul*) closure)->weight;
               return cw * process_background_closure(((const ClosureMul*) closure)->closure);
           }
           case ClosureColor::ADD: {
               return process_background_closure(((const ClosureAdd*) closure)->closureA) +
                      process_background_closure(((const ClosureAdd*) closure)->closureB);
           }
           case ClosureColor::COMPONENT: {
               const ClosureComponent* comp = (const ClosureComponent*) closure;
               if (comp->id == BACKGROUND_ID)
                   return comp->w;
           }
    }
    // should never happen
    ASSERT(false && "Invalid closure invoked in background shader");
    return Vec3(0, 0, 0);
}


OSL_NAMESPACE_EXIT
