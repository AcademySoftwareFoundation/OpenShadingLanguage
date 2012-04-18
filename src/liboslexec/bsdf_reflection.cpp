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


class ReflectionClosure : public BSDFClosure {
public:
    Vec3  m_N;    // shading normal
    ReflectionClosure() : BSDFClosure(Labels::SINGULAR) { }

    void setup() {};

    bool mergeable (const ClosurePrimitive *other) const {
        const ReflectionClosure *comp = (const ReflectionClosure *)other;
        return m_N == comp->m_N && BSDFClosure::mergeable(other);
    }

    size_t memsize () const { return sizeof(*this); }

    const char *name () const { return "reflection"; }

    void print_on (std::ostream &out) const {
        out << name() << " (";
        out << "(" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "))";
    }

    float albedo (const Vec3 &omega_out) const
    {
        return 1.0f;
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
        return Color3 (0, 0, 0);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
        return Color3 (0, 0, 0);
    }

    ustring sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval) const
    {
        // only one direction is possible
        float cosNO = m_N.dot(omega_out);
        if (cosNO > 0) {
            omega_in = (2 * cosNO) * m_N - omega_out;
            if (Ng.dot(omega_in) > 0) {
                domega_in_dx = 2 * m_N.dot(domega_out_dx) * m_N - domega_out_dx;
                domega_in_dy = 2 * m_N.dot(domega_out_dy) * m_N - domega_out_dy;
                pdf = 1;
                eval.setValue(1, 1, 1);
            }
        }
        return Labels::REFLECT;
    }
};




class FresnelReflectionClosure : public BSDFClosure {
public:
    Vec3  m_N;    // shading normal
    float m_eta;  // index of refraction (for fresnel term)
    FresnelReflectionClosure() : BSDFClosure(Labels::SINGULAR) { }

    void setup() {};

    bool mergeable (const ClosurePrimitive *other) const {
        const FresnelReflectionClosure *comp = (const FresnelReflectionClosure *)other;
        return m_N == comp->m_N && m_eta == comp->m_eta &&
            BSDFClosure::mergeable(other);
    }

    size_t memsize () const { return sizeof(*this); }

    const char *name () const { return "reflection_fresnel"; }

    void print_on (std::ostream &out) const {
        out << "reflection (";
        out << "(" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";
        out << m_eta;
        out << ")";
    }

    float albedo (const Vec3 &omega_out) const
    {
        if (m_eta > 0.0f)
            return fresnel_dielectric(m_N.dot(omega_out), m_eta);
        else
            return 1.0f;
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
        pdf = 0;
        return Color3 (0, 0, 0);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
        pdf = 0;
        return Color3 (0, 0, 0);
    }

    ustring sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval) const
    {
        // only one direction is possible
        float cosNO = m_N.dot(omega_out);
        if (cosNO > 0) {
            omega_in = (2 * cosNO) * m_N - omega_out;
            if (Ng.dot(omega_in) > 0) {
                domega_in_dx = 2 * m_N.dot(domega_out_dx) * m_N - domega_out_dx;
                domega_in_dy = 2 * m_N.dot(domega_out_dy) * m_N - domega_out_dy;
                pdf = 1;
                float value = m_eta > 0.0f ? fresnel_dielectric(cosNO, m_eta) : 1.0f;
                eval.setValue(value, value, value);
            }
        }
        return Labels::REFLECT;
    }
};



ClosureParam bsdf_reflection_params[] = {
    CLOSURE_VECTOR_PARAM(FresnelReflectionClosure, m_N),
    CLOSURE_FLOAT_PARAM (FresnelReflectionClosure, m_eta),
    CLOSURE_STRING_KEYPARAM("label"),
    CLOSURE_FINISH_PARAM(FresnelReflectionClosure) };

CLOSURE_PREPARE(bsdf_reflection_prepare, FresnelReflectionClosure)

}; // namespace pvt
OSL_NAMESPACE_EXIT
