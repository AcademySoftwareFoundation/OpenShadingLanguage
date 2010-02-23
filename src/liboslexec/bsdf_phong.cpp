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

// vanilla phong - leaks energy at grazing angles
// see Global Illumination Compendium entry (66) 
class PhongClosure : public BSDFClosure {
    Vec3 m_N;
    float m_exponent;
public:
    CLOSURE_CTOR (PhongClosure) : BSDFClosure(side, Labels::GLOSSY)
    {
        CLOSURE_FETCH_ARG (m_N       , 1);
        CLOSURE_FETCH_ARG (m_exponent, 2);
    }

    void print_on (std::ostream &out) const {
        out << "phong ((";
        out << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";
        out << m_exponent << ")";
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float& pdf) const
    {
        float cosNI = normal_sign * m_N.dot(omega_in);
        float cosNO = normal_sign * m_N.dot(omega_out);
        if (cosNI > 0 && cosNO > 0) {
           // reflect the view vector
           Vec3 R = (2 * cosNO * normal_sign) * m_N - omega_out;
           float cosRI = R.dot(omega_in);
           if (cosRI > 0) {
               float common = 0.5f * (float) M_1_PI * powf(cosRI, m_exponent);
               float out = cosNI * (m_exponent + 2) * common;
               pdf = (m_exponent + 1) * common;
               return Color3 (out, out, out);
           }
        }
        return Color3 (0, 0, 0);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float& pdf) const
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
            // reflect the view vector
            Vec3 R = (2 * cosNO) * Nf - omega_out;
            domega_in_dx = (2 * Nf.dot(domega_out_dx)) * Nf - domega_out_dx;
            domega_in_dy = (2 * Nf.dot(domega_out_dy)) * Nf - domega_out_dy;
            Vec3 T, B;
            make_orthonormals (R, T, B);
            float phi = 2 * (float) M_PI * randu;
            float cosTheta = powf(randv, 1 / (m_exponent + 1));
            float sinTheta2 = 1 - cosTheta * cosTheta;
            float sinTheta = sinTheta2 > 0 ? sqrtf(sinTheta2) : 0;
            omega_in = (cosf(phi) * sinTheta) * T +
                       (sinf(phi) * sinTheta) * B +
                       (            cosTheta) * R;
            if (Ngf.dot(omega_in) > 0)
            {
                // common terms for pdf and eval
                float cosNI = Nf.dot(omega_in);
                // make sure the direction we chose is still in the right hemisphere
                if (cosNI > 0)
                {
                    float common = 0.5f * (float) M_1_PI * powf(cosTheta, m_exponent);
                    pdf = (m_exponent + 1) * common;
                    float out = cosNI * (m_exponent + 2) * common;
                    eval.setValue(out, out, out);
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

DECLOP (OP_phong)
{
    closure_op_guts<PhongClosure, 3> (exec, nargs, args);
}



class PhongRampClosure : public BSDFClosure {
    static const int MAXCOLORS = 8;
    Vec3 m_N;
    float m_exponent;
    Color3 m_colors[MAXCOLORS];
    int m_ncolors;
public:
    CLOSURE_CTOR (PhongRampClosure) : BSDFClosure(side, Labels::GLOSSY)
    {
        CLOSURE_FETCH_ARG (m_N       , 1);
        CLOSURE_FETCH_ARG (m_exponent, 2);
        DASSERT (3 < nargs);
        // Fill our internal color array with the one provided
        Symbol &Colors (exec->sym (args[3]));
        DASSERT (Colors.typespec().is_array() && Colors.typespec().elementtype().is_triple());
        Color3 *m = (Color3 *)((char *)Colors.data() + idx * Colors.step());
        TypeDesc colorstype = Colors.typespec().simpletype();
        DASSERT (0 < colorstype.arraylen && colorstype.arraylen <= MAXCOLORS);
        m_ncolors = colorstype.arraylen;
        for (int r = 0;  r < colorstype.arraylen;  ++r)
            m_colors[r] = m[r];
    }

    void print_on (std::ostream &out) const {
        out << "phong_ramp((";
        out << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";
        out << m_exponent << ")";
    }

    Color3 get_color (float pos) const
    {
        float npos = pos * (float)(m_ncolors - 1);
        int ipos = (int)npos;
        if (ipos >= (m_ncolors - 1))
            return m_colors[m_ncolors - 1];
        float offset = npos - (float)ipos;
        return m_colors[ipos] * (1.0f - offset) + m_colors[ipos+1] * offset;
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float& pdf) const
    {
        float cosNI = normal_sign * m_N.dot(omega_in);
        float cosNO = normal_sign * m_N.dot(omega_out);
        if (cosNI > 0 && cosNO > 0) {
            // reflect the view vector
            Vec3 R = (2 * cosNO * normal_sign) * m_N - omega_out;
            float cosRI = R.dot(omega_in);
            if (cosRI > 0) {
                float cosp = powf(cosRI, m_exponent);
                float common = 0.5f * (float) M_1_PI * cosp;
                float out = cosNI * (m_exponent + 2) * common;
                pdf = (m_exponent + 1) * common;
                return get_color(cosp) * out;
            }
        }
        return Color3 (0, 0, 0);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float normal_sign, float& pdf) const
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
            // reflect the view vector
            Vec3 R = (2 * cosNO) * Nf - omega_out;
            domega_in_dx = (2 * Nf.dot(domega_out_dx)) * Nf - domega_out_dx;
            domega_in_dy = (2 * Nf.dot(domega_out_dy)) * Nf - domega_out_dy;
            Vec3 T, B;
            make_orthonormals (R, T, B);
            float phi = 2 * (float) M_PI * randu;
            float cosTheta = powf(randv, 1 / (m_exponent + 1));
            float sinTheta2 = 1 - cosTheta * cosTheta;
            float sinTheta = sinTheta2 > 0 ? sqrtf(sinTheta2) : 0;
            omega_in = (cosf(phi) * sinTheta) * T +
                       (sinf(phi) * sinTheta) * B +
                       (            cosTheta) * R;
            if (Ngf.dot(omega_in) > 0)
            {
                // common terms for pdf and eval
                float cosNI = Nf.dot(omega_in);
                // make sure the direction we chose is still in the right hemisphere
                if (cosNI > 0)
                {
                    float cosp = powf(cosTheta, m_exponent);
                    float common = 0.5f * (float) M_1_PI * cosp;
                    pdf = (m_exponent + 1) * common;
                    float out = cosNI * (m_exponent + 2) * common;
                    eval = get_color(cosp) * out;
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

DECLOP (OP_phong_ramp)
{
    closure_op_guts<PhongRampClosure, 4> (exec, nargs, args);
}


}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
