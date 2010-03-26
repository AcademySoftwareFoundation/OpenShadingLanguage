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

class WestinBackscatterClosure : public BSDFClosure {
    Vec3 m_N;
    float m_roughness;
    float m_invroughness;
public:
    CLOSURE_CTOR (WestinBackscatterClosure) : BSDFClosure(side, Labels::GLOSSY)
    {
        CLOSURE_FETCH_ARG (m_N, 1);
        CLOSURE_FETCH_ARG (m_roughness, 2);
        m_invroughness = m_roughness > 0 ? 1 / m_roughness : 0;
    }

    bool mergeable (const ClosurePrimitive *other) const {
        const WestinBackscatterClosure *comp = (const WestinBackscatterClosure *)other;
        return m_N == comp->m_N && m_roughness == comp->m_roughness &&
            BSDFClosure::mergeable(other);
    }

    size_t memsize () const { return sizeof(*this); }

    const char *name () const { return "westin_backscatter"; }

    void print_on (std::ostream &out) const
    {
        out << name() << " (";
        out << "(" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";
        out << m_roughness;
        out << ")";
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float &pdf) const
    {
        // pdf is implicitly 0 (no indirect sampling)
        float cosNO = normal_sign * m_N.dot(omega_out);
        float cosNI = normal_sign * m_N.dot(omega_in);
        if (cosNO > 0 && cosNI > 0) {
            float cosine = omega_out.dot(omega_in);
            pdf = cosine > 0 ? (m_invroughness + 1) * powf(cosine, m_invroughness) : 0;
            pdf *= 0.5f * float(M_1_PI);
            return Color3 (pdf, pdf, pdf);
        }
        return Color3 (0, 0, 0);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float &pdf) const
    {
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
            domega_in_dx = domega_out_dx;
            domega_in_dy = domega_out_dy;
            Vec3 T, B;
            make_orthonormals (omega_out, T, B);
            float phi = 2 * (float) M_PI * randu;
            float cosTheta = powf(randv, 1 / (m_invroughness + 1));
            float sinTheta2 = 1 - cosTheta * cosTheta;
            float sinTheta = sinTheta2 > 0 ? sqrtf(sinTheta2) : 0;
            omega_in = (cosf(phi) * sinTheta) * T +
                       (sinf(phi) * sinTheta) * B +
                       (            cosTheta) * omega_out;
            if (Ngf.dot(omega_in) > 0)
            {
                // common terms for pdf and eval
                float cosNI = Nf.dot(omega_in);
                // make sure the direction we chose is still in the right hemisphere
                if (cosNI > 0)
                {
                    pdf = 0.5f * (float) M_1_PI * powf(cosTheta, m_invroughness);
                    pdf = (m_invroughness + 1) * pdf;
                    eval.setValue(pdf, pdf, pdf);
                    // Since there is some blur to this reflection, make the
                    // derivatives a bit bigger. In theory this varies with the
                    // exponent but the exact relationship is complex and
                    // requires more ops than are practical.
                    domega_in_dx *= 10;
                    domega_in_dy *= 10;
                }
            }
        }
        return Labels::REFLECT;
    }

};


class WestinSheenClosure : public BSDFClosure {
    Vec3 m_N;
    float m_edginess;
//    float m_normalization;
public:
    CLOSURE_CTOR (WestinSheenClosure) : BSDFClosure(side, Labels::DIFFUSE)
    {
        CLOSURE_FETCH_ARG (m_N, 1);
        CLOSURE_FETCH_ARG (m_edginess, 2);
    }

    bool mergeable (const ClosurePrimitive *other) const {
        const WestinSheenClosure *comp = (const WestinSheenClosure *)other;
        return m_N == comp->m_N && m_edginess == comp->m_edginess &&
            BSDFClosure::mergeable(other);
    }

    size_t memsize () const { return sizeof(*this); }

    const char *name () const { return "westin_sheen"; }

    void print_on (std::ostream &out) const
    {
        out << name() << " (";
        out << "(" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";
        out << m_edginess;
        out << ")";
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float &pdf) const
    {
        // pdf is implicitly 0 (no indirect sampling)
        float cosNO = normal_sign * m_N.dot(omega_out);
        float cosNI = normal_sign * m_N.dot(omega_in);
        if (cosNO > 0 && cosNI > 0) {
            float sinNO2 = 1 - cosNO * cosNO;
            pdf = cosNI * float(M_1_PI);
            float westin = sinNO2 > 0 ? powf(sinNO2, 0.5f * m_edginess) * pdf : 0;
            return Color3 (westin, westin, westin);
        }
        return Color3 (0, 0, 0);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float &pdf) const
    {
        return Color3 (0, 0, 0);
    }

    ustring sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval) const
    {
        Vec3 Ngf, Nf;
        if (faceforward (omega_out, Ng, m_N, Ngf, Nf)) {
           // we are viewing the surface from the right side - send a ray out with cosine
           // distribution over the hemisphere
           sample_cos_hemisphere (Nf, omega_out, randu, randv, omega_in, pdf);
           if (Ngf.dot(omega_in) > 0) {
               // TODO: account for sheen when sampling
               float cosNO = Nf.dot(omega_out);
               float sinNO2 = 1 - cosNO * cosNO;
               float westin = sinNO2 > 0 ? powf(sinNO2, 0.5f * m_edginess) * pdf : 0;
               eval.setValue(westin, westin, westin);
               // TODO: find a better approximation for the diffuse bounce
               domega_in_dx = (2 * Nf.dot(domega_out_dx)) * Nf - domega_out_dx;
               domega_in_dy = (2 * Nf.dot(domega_out_dy)) * Nf - domega_out_dy;
               domega_in_dx *= 125;
               domega_in_dy *= 125;
           } else
               pdf = 0;
        }
        return Labels::REFLECT;
    }
};


DECLOP (OP_westin_backscatter)
{
    closure_op_guts<WestinBackscatterClosure, 3> (exec, nargs, args);
}

DECLOP (OP_westin_sheen)
{
    closure_op_guts<WestinSheenClosure, 3> (exec, nargs, args);
}


}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif


