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



class HairDiffuseClosure : public BSDFClosure {
    Vec3 m_T;
public:
    CLOSURE_CTOR (HairDiffuseClosure) : BSDFClosure(Both, Labels::DIFFUSE, Both)
    {
        CLOSURE_FETCH_ARG (m_T, 1);
    }

    void print_on (std::ostream &out) const
    {
        out << "hair_diffuse ((" << m_T[0] << ", " << m_T[1] << ", " << m_T[2] << "))";
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float& pdf) const
    {
        float cos_a = m_T.dot(omega_in);
        float bsdf = sqrtf(std::max(1 - cos_a*cos_a, 0.0f)) * (float) (M_1_PI * M_1_PI);
        pdf = (float) 1 / (4 * M_PI);
        return Color3 (bsdf, bsdf, bsdf);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float& pdf) const
    {
       return eval_reflect(omega_out, omega_in, normal_sign, pdf);
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
    Vec3 m_T;
    float m_offset, m_cos_off, m_sin_off;
    float m_exp;
public:
    CLOSURE_CTOR (HairSpecularClosure) : BSDFClosure(Both, Labels::GLOSSY, Both)
    {
        // Tangent vector
        CLOSURE_FETCH_ARG (m_T, 1);
        // specular offset
        CLOSURE_FETCH_ARG (m_offset, 2);
        // roughness for the specular as used in spi shaders
        CLOSURE_FETCH_ARG (m_exp, 3);
        m_cos_off = cosf(m_offset);
        m_sin_off = sinf(m_offset);
    }

    void print_on (std::ostream &out) const
    {
        out << "hair_specular ((" << m_T[0] << ", " << m_T[1] << ", " << m_T[2] << "), " << m_offset << ")";
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float& pdf) const
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

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float& pdf) const
    {
       return eval_reflect(omega_out, omega_in, normal_sign, pdf);
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



DECLOP (OP_hair_diffuse)
{
    closure_op_guts<HairDiffuseClosure, 2> (exec, nargs, args);
}



DECLOP (OP_hair_specular)
{
    closure_op_guts<HairSpecularClosure, 4> (exec, nargs, args);
}




}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
