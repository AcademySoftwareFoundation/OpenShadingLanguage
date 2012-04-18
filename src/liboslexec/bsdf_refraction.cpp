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

#include <OpenImageIO/fmath.h>

#include "genclosure.h"
#include "oslexec_pvt.h"

OSL_NAMESPACE_ENTER

namespace pvt {

class RefractionClosure : public BSDFClosure {
public:
    Vec3  m_N;     // shading normal
    float m_eta;   // ratio of indices of refraction (inside / outside)
    RefractionClosure() : BSDFClosure(Labels::SINGULAR, Back) { }

    void setup() {}

    bool mergeable (const ClosurePrimitive *other) const {
        const RefractionClosure *comp = (const RefractionClosure *)other;
        return m_N == comp->m_N && m_eta == comp->m_eta &&
            BSDFClosure::mergeable(other);
    }

    size_t memsize () const { return sizeof(*this); }

    const char *name () const { return "refraction"; }

    void print_on (std::ostream &out) const {
        out << name() << " (";
        out << "(" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";
        out << m_eta;
        out << ")";
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
        return Color3 (0, 0, 0);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
        return Color3 (0, 0, 0);
    }

    float albedo (const Vec3 &omega_out) const
    {
        float cosNO = m_N.dot(omega_out);
        return 1.0f - fresnel_dielectric(cosNO, m_eta);
    }

    ustring sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval) const
    {
        Vec3 R, dRdx, dRdy;
        Vec3 T, dTdx, dTdy;
        bool inside;
        float Ft = 1 - fresnel_dielectric(m_eta, m_N,
                                          omega_out, domega_out_dx, domega_out_dy,
                                          R, dRdx, dRdy,
                                          T, dTdx, dTdy,
                                          inside);
        if (Ft > 0 && !inside) {
            pdf = 1;
            eval.setValue(Ft, Ft, Ft);
            omega_in = T;
            domega_in_dx = dTdx;
            domega_in_dy = dTdy;
        }
        return Labels::TRANSMIT;
    }
};



class DielectricClosure : public BSDFClosure {
public:
    Vec3  m_N;     // shading normal
    float m_eta;   // ratio of indices of refraction (inside / outside)
    DielectricClosure() : BSDFClosure(Labels::SINGULAR, Both) { }

    void setup() { }

    bool mergeable (const ClosurePrimitive *other) const {
        const DielectricClosure *comp = (const DielectricClosure *)other;
        return m_N == comp->m_N && m_eta == comp->m_eta &&
            BSDFClosure::mergeable(other);
    }

    size_t memsize () const { return sizeof(*this); }

    const char *name () const { return "dielectric"; }

    void print_on (std::ostream &out) const {
        out << name() << " (";
        out << "(" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";
        out << m_eta;
        out << ")";
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
        return Color3 (0, 0, 0);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
        return Color3 (0, 0, 0);
    }

    float albedo (const Vec3 &omega_out) const
    {
        return 1.0f;
    }

    ustring sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval) const
    {
        Vec3 R, dRdx, dRdy;
        Vec3 T, dTdx, dTdy;
        bool inside;
        // randomly choose between reflection/refraction
        float Fr = fresnel_dielectric(m_eta, m_N,
                                      omega_out, domega_out_dx, domega_out_dy,
                                      R, dRdx, dRdy,
                                      T, dTdx, dTdy,
                                      inside);
        if (!inside)
        {
            if (randu < Fr) {
                eval.setValue(Fr, Fr, Fr);
                pdf = Fr;
                omega_in = R;
                domega_in_dx = dRdx;
                domega_in_dy = dRdy;
                return Labels::REFLECT;
            } else {
                pdf = 1 - Fr;
                eval.setValue(pdf, pdf, pdf);
                omega_in = T;
                domega_in_dx = dTdx;
                domega_in_dy = dTdy;
                return Labels::TRANSMIT;
            }
        }
        else
            return Labels::NONE;
    }
};



ClosureParam bsdf_refraction_params[] = {
    CLOSURE_VECTOR_PARAM(RefractionClosure, m_N),
    CLOSURE_FLOAT_PARAM (RefractionClosure, m_eta),
    CLOSURE_STRING_KEYPARAM("label"),
    CLOSURE_FINISH_PARAM(RefractionClosure) };

ClosureParam bsdf_dielectric_params[] = {
    CLOSURE_VECTOR_PARAM(DielectricClosure, m_N),
    CLOSURE_FLOAT_PARAM (DielectricClosure, m_eta),
    CLOSURE_STRING_KEYPARAM("label"),
    CLOSURE_FINISH_PARAM(DielectricClosure) };

CLOSURE_PREPARE(bsdf_refraction_prepare, RefractionClosure)
CLOSURE_PREPARE(bsdf_dielectric_prepare, DielectricClosure)

}; // namespace pvt
OSL_NAMESPACE_EXIT
