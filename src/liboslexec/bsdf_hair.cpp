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

#include "genclosure.h"
#include "oslops.h"
#include "oslexec_pvt.h"

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {



class HairDiffuseClosure : public BSDFClosure {
public:
    Vec3 m_T;
    HairDiffuseClosure() : BSDFClosure(Labels::DIFFUSE, Both) { }

    void setup() {};

    bool mergeable (const ClosurePrimitive *other) const {
        const HairDiffuseClosure *comp = (const HairDiffuseClosure *)other;
        return m_T == comp->m_T && BSDFClosure::mergeable(other);
    }

    size_t memsize () const { return sizeof(*this); }

    const char *name () const { return "hair_diffuse"; }

    void print_on (std::ostream &out) const
    {
        out << name() << " ((" << m_T[0] << ", " << m_T[1] << ", " << m_T[2] << "))";
    }

    float albedo (const Vec3 &omega_out) const
    {
        return 1.0f;
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
        float cos_a = m_T.dot(omega_in);
        float bsdf = sqrtf(std::max(1 - cos_a*cos_a, 0.0f)) * (float) (M_1_PI * M_1_PI);
        pdf = (float) 1 / (4 * M_PI);
        return Color3 (bsdf, bsdf, bsdf);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
       return eval_reflect(omega_out, omega_in, pdf);
    }

    ustring sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval) const
    {
        float Z = 1.0f - randu * 2.0f;
        float sinTheta = sqrtf(1.0 - Z*Z);
        float X = cosf(2*M_PI*randv) * sinTheta;
        float Y = sinf(2*M_PI*randv) * sinTheta;
        Vec3 T, B;
        make_orthonormals (m_T, T, B);
        omega_in = X * T + Y * B + Z * m_T;
        pdf = (float) 1 / (4 * M_PI);

        sinTheta *= (float) (M_1_PI * M_1_PI);
        eval.setValue(sinTheta, sinTheta, sinTheta);

        if (Ng.dot(omega_in) > 0.0f)
            return Labels::REFLECT;
        else
            return Labels::TRANSMIT;
    }
};



class HairSpecularClosure : public BSDFClosure {
public:
    Vec3 m_T;
    float m_offset, m_cos_off, m_sin_off;
    float m_exp;
    HairSpecularClosure() : BSDFClosure(Labels::GLOSSY, Both) { }

    void setup()
    {
        m_cos_off = cosf(m_offset);
        m_sin_off = sinf(m_offset);
    }

    bool mergeable (const ClosurePrimitive *other) const {
        const HairSpecularClosure *comp = (const HairSpecularClosure *)other;
        return m_T == comp->m_T && m_offset == comp->m_offset &&
            m_exp == comp->m_exp && BSDFClosure::mergeable(other);
    }

    size_t memsize () const { return sizeof(*this); }

    const char *name () const { return "hair_specular"; }

    void print_on (std::ostream &out) const
    {
        out << name() << " ((" << m_T[0] << ", " << m_T[1] << ", " << m_T[2] << "), " << m_offset << ")";
    }

    float albedo (const Vec3 &omega_out) const
    {
        // we don't know how to sample this
        return 0.0f;
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
        //float angle_i = acosf(m_T.dot(omega_in));
        //float angle_o = M_PI - (acosf(m_T.dot(omega_out)) + m_offset);
        //float cos_diff = cosf(angle_i - angle_o);
        //
        // Optimized version of the above commented code
        float cos_i = m_T.dot(omega_in);
        float cos_o = m_T.dot(omega_out);
        float sin_i = sqrtf (std::max (1 - cos_i*cos_i, 0.0f));
        float sin_o = sqrtf (std::max (1 - cos_o*cos_o, 0.0f));
        float cos_diff = sin_i * sin_o * m_cos_off +
                         sin_i * cos_o * m_sin_off +
                         cos_i * sin_o * m_sin_off -
                         cos_i * cos_o * m_cos_off;
        // TODO: normalization? ha!
        float bsdf = cos_diff > 0.0f ? powf(cos_diff, m_exp) : 0.0f;
        bsdf *= (float) (M_1_PI * M_1_PI);
        pdf = 0.0f;
        return Color3 (bsdf, bsdf, bsdf);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float& pdf) const
    {
       return eval_reflect(omega_out, omega_in, pdf);
    }

    ustring sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval) const
    {
        // TODO: we don't know how to do this, sorry
        pdf = 0.0f;
        return Labels::NONE;
    }
};



ClosureParam bsdf_hair_diffuse_params[] = {
    CLOSURE_VECTOR_PARAM   (HairDiffuseClosure, m_T),
    CLOSURE_STRING_KEYPARAM("label"),
    CLOSURE_FINISH_PARAM   (HairDiffuseClosure) };

ClosureParam bsdf_hair_specular_params[] = {
    CLOSURE_VECTOR_PARAM(HairSpecularClosure, m_T),
    CLOSURE_FLOAT_PARAM (HairSpecularClosure, m_offset),
    CLOSURE_FLOAT_PARAM (HairSpecularClosure, m_exp),
    CLOSURE_STRING_KEYPARAM("label"),
    CLOSURE_FINISH_PARAM(HairSpecularClosure) };

CLOSURE_PREPARE(bsdf_hair_diffuse_prepare, HairDiffuseClosure)
CLOSURE_PREPARE(bsdf_hair_specular_prepare, HairSpecularClosure)

}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
