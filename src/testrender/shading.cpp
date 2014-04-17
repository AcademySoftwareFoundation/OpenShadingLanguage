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

/* The anisotropic variant of GGX and Beckmann comes from
 * Eric Heitz Understanding the Masking-Shadowing Function in
 * Microfacet-Based BRDFs, section 5.4.
 */
struct GGXDist {
    GGXDist(float ax, float ay) : ax(ax), ay(ay), ax2(ax * ax), ay2(ay * ay) {}

    float D(const Vec3 &M) const {
        float cosThetaM = M.z;
        float cosThetaM2 = cosThetaM * cosThetaM;
        float cosThetaM4 = cosThetaM2 * cosThetaM2;
        if (ax != ay) {
            float sinThetaM = sqrtf (std::max (1.0f - cosThetaM2, 0.0f));
            float invSinThetaM = sinThetaM > 0.0f ? 1.0f / sinThetaM : 0.0f;
            float cosPhi2 = M.x * invSinThetaM;
            float sinPhi2 = M.y * invSinThetaM;
            cosPhi2 *= cosPhi2;
            sinPhi2 *= sinPhi2;
            float tanThetaM2 = (sinThetaM * sinThetaM) / cosThetaM2;
            float tmp = 1 + tanThetaM2 * (cosPhi2 / ax2 + sinPhi2 / ay2);

            return 1.0f / (float(M_PI) * ax * ay * cosThetaM4 * tmp * tmp);
        }
        // eq. 33: calculate D(m) with m=Hr:
        float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;
        return ax2 / (float(M_PI) * cosThetaM4 * (ax2 + tanThetaM2) * (ax2 + tanThetaM2));
    }
    float G(const Vec3 &w) const {
        float cosTheta = fabsf(w.z);
        if (ax != ay) {
            float sinTheta = sqrtf (std::max (1.0f - cosTheta * cosTheta, 0.0f));
            float cosPhi2 = w.x / sinTheta;
            float sinPhi2 = w.y / sinTheta;
            cosPhi2 *= cosPhi2;
            sinPhi2 *= sinPhi2;

            float alpha = sqrtf(cosPhi2 * ax2 + sinPhi2 * ay2);
            float a = cosTheta / (alpha * sinTheta);
            float Lambda = (-1 + sqrtf(1 + 1 / (a * a))) * 0.5f;
            return 1.0f / (1 + Lambda);
        }
        // eq. 34: calculate G
        return 2 / (1 + sqrtf(1 + ax2 * (1 - cosTheta * cosTheta) / (cosTheta * cosTheta)));
    }
    Vec3 sample(float rx, float ry) const {
        if (ax != ay)
        {
            float cosPhi = cosf(2 * float(M_PI) * rx) * ax;
            float sinPhi = sinf(2 * float(M_PI) * rx) * ay;
            float invnorm = 1.0f / sqrtf(cosPhi * cosPhi + sinPhi * sinPhi);
            cosPhi *= invnorm;
            sinPhi *= invnorm;

            float C = (cosPhi / ax) * (cosPhi / ax) +
                      (sinPhi / ay) * (sinPhi / ay);
            float tanTheta2 = ry / ((1 - ry) * C);
            float cosTheta  = 1 / sqrtf(1 + tanTheta2);
            float sinTheta  = cosTheta * sqrtf(tanTheta2);

            return Vec3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
        }
        // generate a random microfacet normal m
        // eq. 35,36:
        // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
        //                  and sin(atan(x)) == x/sqrt(1+x^2)
        float tanThetaM2 = ax2 * rx / (1 - rx);
        float cosThetaM  = 1 / sqrtf(1 + tanThetaM2);
        float sinThetaM  = cosThetaM * sqrtf(tanThetaM2);
        float phiM = 2 * float(M_PI) * ry;
        return Vec3(cosf(phiM) * sinThetaM,
                    sinf(phiM) * sinThetaM,
                    cosThetaM);
    }
private:
    float ax, ay, ax2, ay2;
};

struct BeckmannDist {
    BeckmannDist(float ax, float ay) : ax(ax), ay(ay), ax2(ax * ax), ay2(ay * ay) {}

    float D(const Vec3 &M) const {
        float cosThetaM = M.z;
        float cosThetaM2 = cosThetaM * cosThetaM;
        float cosThetaM4 = cosThetaM2 * cosThetaM2;
        float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;
        if (ax != ay) {
            float sinThetaM = sqrtf (std::max(1.0f - cosThetaM2, 0.0f));
            float invSinThetaM = sinThetaM > 0.0f ? 1.0f / sinThetaM : 0.0f;
            float cosPhi2 = M.x * invSinThetaM;
            float sinPhi2 = M.y * invSinThetaM;
            cosPhi2 *= cosPhi2;
            sinPhi2 *= sinPhi2;

            return expf(-tanThetaM2 * (cosPhi2 / ax2 + sinPhi2 / ay2)) /
                   (float(M_PI) * ax * ay * cosThetaM4);
        }
        return expf(-tanThetaM2 / ax2) / (float(M_PI) * ax2 *  cosThetaM4);
    }
    float G(const Vec3 &w) const {
        float cosTheta = fabsf(w.z);
        if (ax != ay) {
            static const float SQRT_PI = sqrtf(float(M_PI));
            float sinTheta = sqrtf (std::max (1 - cosTheta * cosTheta, 0.0f));
            float cosPhi2 = w.x / sinTheta;
            float sinPhi2 = w.y / sinTheta;
            cosPhi2 *= cosPhi2;
            sinPhi2 *= sinPhi2;

            float alpha = sqrtf(cosPhi2 * ax2 + sinPhi2 * ay2);
            float a = cosTheta / (alpha * sinTheta);
            float Lambda = (erff(a) - 1) * 0.5f + expf(-(a * a)) /
                                                       (2 * a * SQRT_PI);
            return 1.0f / (1 + Lambda);
        }
        // eq. 26, 27: calculate G
        float a = 1 / sqrtf(ax2 * (1 - cosTheta * cosTheta) / (cosTheta * cosTheta));
        return a < 1.6f ? (3.535f * a + 2.181f * a * a) / (1 + 2.276f * a + 2.577f * a * a) : 1.0f;
    }
    Vec3 sample(float rx, float ry) const {
        if (ax != ay) {
            float cosPhi = cosf(2 * float(M_PI) * rx) * ax;
            float sinPhi = sinf(2 * float(M_PI) * rx) * ay;
            float invnorm = 1.0f / sqrtf(cosPhi * cosPhi + sinPhi * sinPhi);
            cosPhi *= invnorm;
            sinPhi *= invnorm;

            float C = (cosPhi / ax) * (cosPhi / ax) +
                      (sinPhi / ay) * (sinPhi / ay);
            float tanTheta2 = -logf(1 - ry) / C;
            float cosTheta  = 1 / sqrtf(1 + tanTheta2);
            float sinTheta  = cosTheta * sqrtf(tanTheta2);

            return Vec3(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
        }
        // eq. 35,36:
        // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
        //                  and sin(atan(x)) == x/sqrt(1+x^2)
        float tanThetaM = sqrtf(-ax2 * logf(1 - rx));
        float cosThetaM = 1 / sqrtf(1 + tanThetaM * tanThetaM);
        float sinThetaM = cosThetaM * tanThetaM;
        float phiM = 2 * float(M_PI) * ry;
        return Vec3(cosf(phiM) * sinThetaM,
                    sinf(phiM) * sinThetaM,
                    cosThetaM);
    }
private:
    float ax, ay, ax2, ay2;
};

template <typename Distribution, int Refract>
struct Microfacet : public BSDF, MicrofacetParams {
    Microfacet(const MicrofacetParams& params) : BSDF(false),
        MicrofacetParams(params), dist(params.xalpha, params.yalpha),
        tf(U == Vec3(0) || xalpha == yalpha ? TangentFrame(N) : TangentFrame(N, U)) { }
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
                float Dr = dist.D(tf.tolocal(Hr));
                // eq. 34: now calculate G1(i,m) and G1(o,m)
                float Gr = dist.G(tf.tolocal(wo)) * dist.G(tf.tolocal(wi));
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
           float cosNO;
           if (wi.dot(wo) <= 0 && (cosNO = N.dot(wo)) > 0.0f) {
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
                  float Dt = dist.D(tf.tolocal(Ht));
                  // eq. 34: now calculate G1(i,m) and G1(o,m)
                  float Gt = dist.G(tf.tolocal(wo)) * dist.G(tf.tolocal(wi));
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
        Vec3 m = dist.sample(rx, ry);
        m = tf.toworld(m);
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
            if (Ft > 0) {
                float e = eval(sg, wi.val(), invpdf);
                invpdf = 1 / invpdf; // eval returned pdf, invert it
                return e * invpdf; // FIXME: simplify math here
                return invpdf = 1;
            }
        }
        return invpdf = 0;
    }

private:
    Distribution dist;
    TangentFrame tf;
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
   static const ustring u_ggx("ggx");
   static const ustring u_beckmann("beckmann");
   static const ustring u_default("default");
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
                   case MICROFACET_ID:
                       if (((const MicrofacetParams*) comp->data())->dist == u_ggx) {
                           if (((const MicrofacetParams*) comp->data())->refract)
                               ok = result.bsdf.add_bsdf<MicrofacetGGXRefr, MicrofacetParams>(cw, *(const MicrofacetParams*) comp->data());
                           else
                               ok = result.bsdf.add_bsdf<MicrofacetGGXRefl, MicrofacetParams>(cw, *(const MicrofacetParams*) comp->data());
                       } else if (((const MicrofacetParams*) comp->data())->dist == u_beckmann ||
                                  ((const MicrofacetParams*) comp->data())->dist == u_default) {
                           if (((const MicrofacetParams*) comp->data())->refract)
                               ok = result.bsdf.add_bsdf<MicrofacetBeckmannRefr, MicrofacetParams>(cw, *(const MicrofacetParams*) comp->data());
                           else
                               ok = result.bsdf.add_bsdf<MicrofacetBeckmannRefl, MicrofacetParams>(cw, *(const MicrofacetParams*) comp->data());
                       }
                       break;
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
