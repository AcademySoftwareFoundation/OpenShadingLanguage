/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cmath>

#include "oslops.h"
#include "oslexec_pvt.h"

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {

// TODO: refactor these two classes so they share everything by the microfacet
//       distribution terms


// microfacet model with GGX facet distribution
// see http://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
template <int Refractive = 0>
class MicrofacetGGXClosure : public BSDFClosure {
    Vec3 m_N;
    float m_ag;   // width parameter (roughness)
    float m_eta;  // index of refraction (for fresnel term)
public:
    CLOSURE_CTOR (MicrofacetGGXClosure) : BSDFClosure(Refractive ? Both : side, Labels::GLOSSY, Refractive ? Back : Front)
    {
        CLOSURE_FETCH_ARG (m_N  , 1);
        CLOSURE_FETCH_ARG (m_ag , 2);
        CLOSURE_FETCH_ARG (m_eta, 3);
    }

    void print_on (std::ostream &out) const {
        out << ((Refractive == 0) ? "microfacet_ggx (" : "microfacet_ggx_refraction (");
        out << "(" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";
        out << m_ag << ", ";
        out << m_eta;
        out << ")";
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float& pdf) const
    {
        if (Refractive == 1) return Color3 (0, 0, 0);
        float cosNO = normal_sign * m_N.dot(omega_out);
        float cosNI = normal_sign * m_N.dot(omega_in);
        if (cosNI > 0 && cosNO > 0) {
            // get half vector
            Vec3 Hr = omega_in + omega_out;
            Hr.normalize();
            // eq. 20: (F*G*D)/(4*in*on)
            // eq. 33: first we calculate D(m) with m=Hr:
            float alpha2 = m_ag * m_ag;
            float cosThetaM = normal_sign * m_N.dot(Hr);
            float cosThetaM2 = cosThetaM * cosThetaM;
            float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;
            float cosThetaM4 = cosThetaM2 * cosThetaM2;
            float D = alpha2 / ((float) M_PI * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));
            // eq. 34: now calculate G1(i,m) and G1(o,m)
            float G1o = 2 / (1 + sqrtf(1 + alpha2 * (1 - cosNO * cosNO) / (cosNO * cosNO)));
            float G1i = 2 / (1 + sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI))); 
            float G = G1o * G1i;
            // fresnel term between outgoing direction and microfacet
            float F = fresnel_dielectric(Hr.dot(omega_out), m_eta);
            float out = (F * G * D) * 0.25f / cosNO;
            // eq. 24
            float pm = D * cosThetaM;
            // convert into pdf of the sampled direction
            // eq. 38 - but see also:
            // eq. 17 in http://www.graphics.cornell.edu/~bjw/wardnotes.pdf
            pdf = pm * 0.25f / Hr.dot(omega_out);
            return Color3 (out, out, out);
        }
        return Color3 (0, 0, 0);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float& pdf) const
    {
        if (Refractive == 0) return Color3 (0, 0, 0);
        float cosNO = normal_sign * m_N.dot(omega_out);
        float cosNI = normal_sign * m_N.dot(omega_in);
        float eta_I, eta_O;
        if (cosNO > 0 && cosNI < 0)
            eta_I = m_eta, eta_O = 1.0f;
        else if (cosNI > 0 && cosNO < 0)
            eta_I = 1.0f, eta_O = m_eta;
        else
            return Color3 (0, 0, 0); // vectors on same side -- not possible
        // compute half-vector of the refraction (eq. 16)
        Vec3 ht = -(eta_I * omega_in + eta_O * omega_out);
        Vec3 Ht = ht; Ht.normalize();
        // compute fresnel term
        float cosHO = Ht.dot(omega_out);
        float Ft = 1 - fresnel_dielectric(cosHO, m_eta);
        if (Ft > 0) { // skip work in case of TIR
            float cosHI = Ht.dot(omega_in);
            // eq. 33: first we calculate D(m) with m=Ht:
            float alpha2 = m_ag * m_ag;
            float cosThetaM = normal_sign * m_N.dot(Ht);
            float cosThetaM2 = cosThetaM * cosThetaM;
            float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;
            float cosThetaM4 = cosThetaM2 * cosThetaM2;
            float D = alpha2 / ((float) M_PI * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));
            // eq. 34: now calculate G1(i,m) and G1(o,m)
            float G1o = 2 / (1 + sqrtf(1 + alpha2 * (1 - cosNO * cosNO) / (cosNO * cosNO)));
            float G1i = 2 / (1 + sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI))); 
            float G = G1o * G1i;
            // probability
            float invHt2 = 1 / ht.dot(ht);
            pdf = D * fabsf(cosThetaM) * (fabsf(cosHI) * (eta_I * eta_I)) * invHt2;
            float out = (fabsf(cosHI * cosHO) * (eta_I * eta_I) * (Ft * G * D) * invHt2) / cosNO;
            return Color3 (out, out, out);
        }
        return Color3 (0, 0, 0);
    }

    ustring sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval) const
    {
        Vec3 Ngf, Nf;
        if (!faceforward (omega_out, Ng, m_N, Ngf, Nf))
            return Labels::NONE;
        float cosNO = Nf.dot(omega_out);
        if (cosNO > 0) {
            Vec3 X, Y, Z = Refractive ? m_N : Nf;
            make_orthonormals(Z, X, Y);
            // generate a random microfacet normal m
            // eq. 35,36:
            // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
            //                  and sin(atan(x)) == x/sqrt(1+x^2)
            float alpha2 = m_ag * m_ag;
            float tanThetaM2 = alpha2 * randu / (1 - randu);
            float cosThetaM  = 1 / sqrtf(1 + tanThetaM2);
            float sinThetaM  = cosThetaM * sqrtf(tanThetaM2);
            float phiM = 2 * float(M_PI) * randv;
            Vec3 m = (cosf(phiM) * sinThetaM) * X +
                     (sinf(phiM) * sinThetaM) * Y +
                                   cosThetaM  * Z;
            if (Refractive == 0) {
                float cosMO = m.dot(omega_out);
                if (cosMO > 0) {
                    // eq. 39 - compute actual reflected direction
                    omega_in = 2 * cosMO * m - omega_out;
                    if (Ngf.dot(omega_in) > 0) {
                        // microfacet normal is visible to this ray
                        // eq. 33
                        float cosThetaM2 = cosThetaM * cosThetaM;
                        float cosThetaM4 = cosThetaM2 * cosThetaM2;
                        float D = alpha2 / (float(M_PI) * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));
                        // eq. 24
                        float pm = D * cosThetaM;
                        // convert into pdf of the sampled direction
                        // eq. 38 - but see also:
                        // eq. 17 in http://www.graphics.cornell.edu/~bjw/wardnotes.pdf
                        pdf = pm * 0.25f / cosMO;
                        // eval BRDF*cosNI
                        float cosNI = Nf.dot(omega_in);
                        // eq. 34: now calculate G1(i,m) and G1(o,m)
                        float G1o = 2 / (1 + sqrtf(1 + alpha2 * (1 - cosNO * cosNO) / (cosNO * cosNO)));
                        float G1i = 2 / (1 + sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI))); 
                        float G = G1o * G1i;
                        // fresnel term between outgoing direction and microfacet
                        float F = fresnel_dielectric(m.dot(omega_out), m_eta);
                        // eq. 20: (F*G*D)/(4*in*on)
                        float out = (F * G * D) * 0.25f / cosNO;
                        eval.setValue(out, out, out);
                        domega_in_dx = (2 * m.dot(domega_out_dx)) * m - domega_out_dx;
                        domega_in_dy = (2 * m.dot(domega_out_dy)) * m - domega_out_dy;
                        // Since there is some blur to this reflection, make the
                        // derivatives a bit bigger. In theory this varies with the
                        // roughness but the exact relationship is complex and
                        // requires more ops than are practical.
                        domega_in_dx *= 10;
                        domega_in_dy *= 10;
                    }
                }
            } else {
                // CAUTION: the i and o variables are inverted relative to the paper
                // eq. 39 - compute actual refractive direction
                Vec3 R, dRdx, dRdy;
                Vec3 T, dTdx, dTdy;
                bool inside;
                float Ft = 1 - fresnel_dielectric(m_eta, m, omega_out, domega_out_dx, domega_out_dy,
                                                  R, dRdx, dRdy,
                                                  T, dTdx, dTdy,
                                                  inside);
                if (Ft > 0) {
                    omega_in = T;
                    domega_in_dx = dTdx;
                    domega_in_dy = dTdy;
                    // eq. 33
                    float cosThetaM2 = cosThetaM * cosThetaM;
                    float cosThetaM4 = cosThetaM2 * cosThetaM2;
                    float D = alpha2 / (float(M_PI) * cosThetaM4 * (alpha2 + tanThetaM2) * (alpha2 + tanThetaM2));
                    // eq. 24
                    float pm = D * cosThetaM;
                    // eval BRDF*cosNI
                    float cosNI = Nf.dot(omega_in);
                    // eq. 34: now calculate G1(i,m) and G1(o,m)
                    float G1o = 2 / (1 + sqrtf(1 + alpha2 * (1 - cosNO * cosNO) / (cosNO * cosNO)));
                    float G1i = 2 / (1 + sqrtf(1 + alpha2 * (1 - cosNI * cosNI) / (cosNI * cosNI))); 
                    float G = G1o * G1i;
                    // eq. 21
                    float cosHI = m.dot(omega_in);
                    float cosHO = m.dot(omega_out);
                    float eta_I = inside ? 1.0f : m_eta;
                    float eta_O = inside ? m_eta : 1.0f;
                    float Ht2 = eta_I * cosHI + eta_O * cosHO;
                    Ht2 *= Ht2;
                    float out = (fabsf(cosHI * cosHO) * (eta_I * eta_I) * (Ft * G * D)) / (cosNO * Ht2);
                    // eq. 38 and eq. 17
                    pdf = pm * (eta_I * eta_I) * fabsf(cosHI) / Ht2;
                    eval.setValue(out, out, out);
                    // Since there is some blur to this refraction, make the
                    // derivatives a bit bigger. In theory this varies with the
                    // roughness but the exact relationship is complex and
                    // requires more ops than are practical.
                    domega_in_dx *= 10;
                    domega_in_dy *= 10;
                }
            }
        }
        return Refractive ? Labels::TRANSMIT : Labels::REFLECT;
    }
};

// microfacet model with Beckmann facet distribution
// see http://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
template <int Refractive = 0>
class MicrofacetBeckmannClosure : public BSDFClosure {
    Vec3 m_N;
    float m_ab;   // width parameter (roughness)
    float m_eta;  // index of refraction (for fresnel term)
public:
    CLOSURE_CTOR (MicrofacetBeckmannClosure) : BSDFClosure(Refractive ? Both : side, Labels::GLOSSY, Refractive ? Back : Front)
    {
        CLOSURE_FETCH_ARG (m_N  , 1);
        CLOSURE_FETCH_ARG (m_ab , 2);
        CLOSURE_FETCH_ARG (m_eta, 3);
    }

    void print_on (std::ostream &out) const
    {
        out << (Refractive ? "microfacet_beckmann_refractive (" : "microfacet_beckmann (");
        out << "(" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";
        out << m_ab << ", ";
        out << m_eta;
        out << ")";
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float& pdf) const
    {
        if (Refractive == 1) return Color3 (0, 0, 0);
        float cosNO = normal_sign * m_N.dot(omega_out);
        float cosNI = normal_sign * m_N.dot(omega_in);
        if (cosNO > 0 && cosNI > 0) {
           // get half vector
           Vec3 Hr = omega_in + omega_out;
           Hr.normalize();
           // eq. 20: (F*G*D)/(4*in*on)
           // eq. 25: first we calculate D(m) with m=Hr:
           float alpha2 = m_ab * m_ab;
           float cosThetaM = normal_sign * m_N.dot(Hr);
           float cosThetaM2 = cosThetaM * cosThetaM;
           float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;
           float cosThetaM4 = cosThetaM2 * cosThetaM2;
           float D = expf(-tanThetaM2 / alpha2) / (float(M_PI) * alpha2 *  cosThetaM4);
           // eq. 26, 27: now calculate G1(i,m) and G1(o,m)
           float ao = 1 / (m_ab * sqrtf((1 - cosNO * cosNO) / (cosNO * cosNO)));
           float ai = 1 / (m_ab * sqrtf((1 - cosNI * cosNI) / (cosNI * cosNI)));
           float G1o = ao < 1.6f ? (3.535f * ao + 2.181f * ao * ao) / (1 + 2.276f * ao + 2.577f * ao * ao) : 1.0f;
           float G1i = ai < 1.6f ? (3.535f * ai + 2.181f * ai * ai) / (1 + 2.276f * ai + 2.577f * ai * ai) : 1.0f;
           float G = G1o * G1i;
           // fresnel term between outgoing direction and microfacet
           float F = fresnel_dielectric(Hr.dot(omega_out), m_eta);
           float out = (F * G * D) * 0.25f / cosNO;
           // eq. 24
           float pm = D * cosThetaM;
           // convert into pdf of the sampled direction
           // eq. 38 - but see also:
           // eq. 17 in http://www.graphics.cornell.edu/~bjw/wardnotes.pdf
           pdf = pm * 0.25f / Hr.dot(omega_out);
           return Color3 (out, out, out);
        }
        return Color3 (0, 0, 0);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float& pdf) const
    {
        if (Refractive == 0) return Color3 (0, 0, 0);
        float cosNO = normal_sign * m_N.dot(omega_out);
        float cosNI = normal_sign * m_N.dot(omega_in);
        float eta_I, eta_O;
        if (cosNO > 0 && cosNI < 0)
            eta_I = m_eta, eta_O = 1.0f;
        else if (cosNI > 0 && cosNO < 0)
            eta_I = 1.0f, eta_O = m_eta;
        else
            return Color3 (0, 0, 0); // vectors on same side -- not possible
        // compute half-vector of the refraction (eq. 16)
        Vec3 ht = -(eta_I * omega_in + eta_O * omega_out);
        Vec3 Ht = ht; Ht.normalize();
        // compute fresnel term
        float cosHO = Ht.dot(omega_out);
        float Ft = 1 - fresnel_dielectric(cosHO, m_eta);
        if (Ft > 0) { // skip work in case of TIR
            float cosHI = Ht.dot(omega_in);
            // eq. 33: first we calculate D(m) with m=Ht:
            float alpha2 = m_ab * m_ab;
            float cosThetaM = normal_sign * m_N.dot(Ht);
            float cosThetaM2 = cosThetaM * cosThetaM;
            float tanThetaM2 = (1 - cosThetaM2) / cosThetaM2;
            float cosThetaM4 = cosThetaM2 * cosThetaM2;
            float D = expf(-tanThetaM2 / alpha2) / (float(M_PI) * alpha2 *  cosThetaM4);
            // eq. 26, 27: now calculate G1(i,m) and G1(o,m)
            float ao = 1 / (m_ab * sqrtf((1 - cosNO * cosNO) / (cosNO * cosNO)));
            float ai = 1 / (m_ab * sqrtf((1 - cosNI * cosNI) / (cosNI * cosNI)));
            float G1o = ao < 1.6f ? (3.535f * ao + 2.181f * ao * ao) / (1 + 2.276f * ao + 2.577f * ao * ao) : 1.0f;
            float G1i = ai < 1.6f ? (3.535f * ai + 2.181f * ai * ai) / (1 + 2.276f * ai + 2.577f * ai * ai) : 1.0f;
            float G = G1o * G1i;
            // probability
            float invHt2 = 1 / ht.dot(ht);
            pdf = D * fabsf(cosThetaM) * (fabsf(cosHI) * (eta_I * eta_I)) * invHt2;
            float out = (fabsf(cosHI * cosHO) * (eta_I * eta_I) * (Ft * G * D) * invHt2) / cosNO;
            return Color3 (out, out, out);
        }
        return Color3 (0, 0, 0);
    }

    ustring sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval) const
    {
        Vec3 Ngf, Nf;
        if (!faceforward (omega_out, Ng, m_N, Ngf, Nf))
            return Labels::NONE;
        float cosNO = Nf.dot(omega_out);
        if (cosNO > 0) {
            Vec3 X, Y, Z = Refractive  ? m_N : Nf;
            make_orthonormals(Z, X, Y);
            // generate a random microfacet normal m
            // eq. 35,36:
            // we take advantage of cos(atan(x)) == 1/sqrt(1+x^2)
            //                  and sin(atan(x)) == x/sqrt(1+x^2)
            float alpha2 = m_ab * m_ab;
            float tanThetaM = sqrtf(-alpha2 * logf(1 - randu));
            float cosThetaM = 1 / sqrtf(1 + tanThetaM * tanThetaM);
            float sinThetaM = cosThetaM * tanThetaM;
            float phiM = 2 * float(M_PI) * randv;
            Vec3 m = (cosf(phiM) * sinThetaM) * X +
                     (sinf(phiM) * sinThetaM) * Y +
                                   cosThetaM  * Z;
            if (Refractive == 0) {
                float cosMO = m.dot(omega_out);
                if (cosMO > 0) {
                    // eq. 39 - compute actual reflected direction
                    omega_in = 2 * cosMO * m - omega_out;
                    if (Ngf.dot(omega_in) > 0) {
                        // microfacet normal is visible to this ray
                        // eq. 25
                        float cosThetaM2 = cosThetaM * cosThetaM;
                        float tanThetaM2 = tanThetaM * tanThetaM;
                        float cosThetaM4 = cosThetaM2 * cosThetaM2;
                        float D = expf(-tanThetaM2 / alpha2) / (float(M_PI) * alpha2 *  cosThetaM4);
                        // eq. 24
                        float pm = D * cosThetaM;
                        // convert into pdf of the sampled direction
                        // eq. 38 - but see also:
                        // eq. 17 in http://www.graphics.cornell.edu/~bjw/wardnotes.pdf
                        pdf = pm * 0.25f / cosMO;
                        // Eval BRDF*cosNI
                        float cosNI = Nf.dot(omega_in);
                        // eq. 26, 27: now calculate G1(i,m) and G1(o,m)
                        float ao = 1 / (m_ab * sqrtf((1 - cosNO * cosNO) / (cosNO * cosNO)));
                        float ai = 1 / (m_ab * sqrtf((1 - cosNI * cosNI) / (cosNI * cosNI)));
                        float G1o = ao < 1.6f ? (3.535f * ao + 2.181f * ao * ao) / (1 + 2.276f * ao + 2.577f * ao * ao) : 1.0f;
                        float G1i = ai < 1.6f ? (3.535f * ai + 2.181f * ai * ai) / (1 + 2.276f * ai + 2.577f * ai * ai) : 1.0f;
                        float G = G1o * G1i;
                        // fresnel term between outgoing direction and microfacet
                        float F = fresnel_dielectric(m.dot(omega_out), m_eta);
                        // eq. 20: (F*G*D)/(4*in*on)
                        float out = (F * G * D) * 0.25f / cosNO;
                        eval.setValue(out, out, out);
                        domega_in_dx = (2 * m.dot(domega_out_dx)) * m - domega_out_dx;
                        domega_in_dy = (2 * m.dot(domega_out_dy)) * m - domega_out_dy;
                        // Since there is some blur to this reflection, make the
                        // derivatives a bit bigger. In theory this varies with the
                        // roughness but the exact relationship is complex and
                        // requires more ops than are practical.
                        domega_in_dx *= 10;
                        domega_in_dy *= 10;
                    }
                }
            } else {
                // CAUTION: the i and o variables are inverted relative to the paper
                // eq. 39 - compute actual refractive direction
                Vec3 R, dRdx, dRdy;
                Vec3 T, dTdx, dTdy;
                bool inside;
                float Ft = 1 - fresnel_dielectric(m_eta, m, omega_out, domega_out_dx, domega_out_dy,
                                                  R, dRdx, dRdy,
                                                  T, dTdx, dTdy,
                                                  inside);
                if (Ft > 0) {
                    omega_in = T;
                    domega_in_dx = dTdx;
                    domega_in_dy = dTdy;
                    // eq. 33
                    float cosThetaM2 = cosThetaM * cosThetaM;
                    float tanThetaM2 = tanThetaM * tanThetaM;
                    float cosThetaM4 = cosThetaM2 * cosThetaM2;
                    float D = expf(-tanThetaM2 / alpha2) / (float(M_PI) * alpha2 *  cosThetaM4);
                    // eq. 24
                    float pm = D * cosThetaM;
                    // eval BRDF*cosNI
                    float cosNI = Nf.dot(omega_in);
                    // eq. 26, 27: now calculate G1(i,m) and G1(o,m)
                    float ao = 1 / (m_ab * sqrtf((1 - cosNO * cosNO) / (cosNO * cosNO)));
                    float ai = 1 / (m_ab * sqrtf((1 - cosNI * cosNI) / (cosNI * cosNI)));
                    float G1o = ao < 1.6f ? (3.535f * ao + 2.181f * ao * ao) / (1 + 2.276f * ao + 2.577f * ao * ao) : 1.0f;
                    float G1i = ai < 1.6f ? (3.535f * ai + 2.181f * ai * ai) / (1 + 2.276f * ai + 2.577f * ai * ai) : 1.0f;
                    float G = G1o * G1i;
                    // eq. 21
                    float cosHI = m.dot(omega_in);
                    float cosHO = m.dot(omega_out);
                    float eta_I = inside ? 1.0f : m_eta;
                    float eta_O = inside ? m_eta : 1.0f;
                    float Ht2 = eta_I * cosHI + eta_O * cosHO;
                    Ht2 *= Ht2;
                    float out = (fabsf(cosHI * cosHO) * (eta_I * eta_I) * (Ft * G * D)) / (cosNO * Ht2);
                    // eq. 38 and eq. 17
                    pdf = pm * (eta_I * eta_I) * fabsf(cosHI) / Ht2;
                    eval.setValue(out, out, out);
                    // Since there is some blur to this refraction, make the
                    // derivatives a bit bigger. In theory this varies with the
                    // roughness but the exact relationship is complex and
                    // requires more ops than are practical.
                    domega_in_dx *= 10;
                    domega_in_dy *= 10;
                }
            }
        }
        return Refractive ? Labels::TRANSMIT : Labels::REFLECT;
    }
};



DECLOP (OP_microfacet_ggx)
{
    closure_op_guts<MicrofacetGGXClosure<0>, 4> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}

DECLOP (OP_microfacet_ggx_refraction)
{
    closure_op_guts<MicrofacetGGXClosure<1>, 4> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}

DECLOP (OP_microfacet_beckmann)
{
    closure_op_guts<MicrofacetBeckmannClosure<0>, 4> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}

DECLOP (OP_microfacet_beckmann_refraction)
{
    closure_op_guts<MicrofacetBeckmannClosure<1>, 4> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
