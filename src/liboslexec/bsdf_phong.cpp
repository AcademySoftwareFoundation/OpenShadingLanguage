/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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
    CLOSURE_CTOR (PhongClosure)
    {
        CLOSURE_FETCH_ARG (m_N       , 1);
        CLOSURE_FETCH_ARG (m_exponent, 2);
    }

    void print_on (std::ostream &out) const {
        out << "phong ((";
        out << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";
        out << m_exponent << ")";
    }

    bool get_cone(const Vec3 &omega_out, Vec3 &axis, float &angle) const
    {
        float cosNO = m_N.dot(omega_out);
        if (cosNO > 0) {
            // we are viewing the surface from the same side as the normal
            axis = m_N;
            angle = (float) M_PI;
            return true;
        }
        // we are below the surface
        return false;
    }

    Color3 eval (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &omega_in, Labels &labels) const
    {
        float cosNO = m_N.dot(omega_out);
        float cosNI = m_N.dot(omega_in);
        // reflect the view vector
        Vec3 R = (2 * cosNO) * m_N - omega_out;
        float cosRI = R.dot(omega_in);
        float out = (cosRI > 0) ? cosNI * ((m_exponent + 2) * 0.5f * (float) M_1_PI * powf(cosRI, m_exponent)) : 0;
        labels.set (Labels::SURFACE, Labels::REFLECT, Labels::GLOSSY);
        return Color3 (out, out, out);
    }

    void sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval, Labels &labels) const
    {
        float cosNO = m_N.dot(omega_out);
        labels.set (Labels::SURFACE, Labels::REFLECT, Labels::GLOSSY);
        if (cosNO > 0) {
            // reflect the view vector
            Vec3 R = (2 * cosNO) * m_N - omega_out;
            domega_in_dx = (2 * m_N.dot(domega_out_dx)) * m_N - domega_out_dx;
            domega_in_dy = (2 * m_N.dot(domega_out_dy)) * m_N - domega_out_dy;
            Vec3 T, B;
            make_orthonormals (R, T, B);
            float phi = 2 * (float) M_PI * randu;
            float cosTheta = powf(randv, 1 / (m_exponent + 1));
            float sinTheta2 = 1 - cosTheta * cosTheta;
            float sinTheta = sinTheta2 > 0 ? sqrtf(sinTheta2) : 0;
            omega_in = (cosf(phi) * sinTheta) * T +
                       (sinf(phi) * sinTheta) * B +
                       (            cosTheta) * R;
            if ((Ng ^ omega_in) > 0)
            {
                // common terms for pdf and eval
                float common = 0.5f * (float) M_1_PI * powf(R.dot(omega_in), m_exponent);
                float cosNI = m_N.dot(omega_in);
                float power;
                // make sure the direction we chose is still in the right hemisphere
                if (cosNI > 0)
                {
                    pdf = (m_exponent + 1) * common;
                    power = cosNI * (m_exponent + 2) * common;
                    eval.setValue(power, power, power);
                    // Since there is some blur to this reflection, make the
                    // derivatives a bit bigger. In theory this varies with the
                    // exponent but the exact relationship is complex and
                    // requires more ops than are practical.
                    domega_in_dx *= 10;
                    domega_in_dy *= 10;
                }
            }
        }
    }

    float pdf (const Vec3 &Ng,
               const Vec3 &omega_out, const Vec3 &omega_in) const
    {
        float cosNO = m_N.dot(omega_out);
        Vec3 R = (2 * cosNO) * m_N - omega_out;
        float cosRI = R.dot(omega_in);
        return cosRI > 0 ? (m_exponent + 1) * 0.5f * (float) M_1_PI * powf(cosRI, m_exponent) : 0;
    }

};

DECLOP (OP_phong)
{
    closure_op_guts<PhongClosure> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}


}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
