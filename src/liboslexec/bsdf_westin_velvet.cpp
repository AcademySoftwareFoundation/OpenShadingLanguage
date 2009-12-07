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

class WestinVelvetClosure : public BSDFClosure {
    Vec3 m_N;
    float m_roughness;
    float m_edginess;
    float m_backscatter;

public:
    CLOSURE_CTOR (WestinVelvetClosure) : BSDFClosure(side, Labels::DIFFUSE)
    {
        CLOSURE_FETCH_ARG (m_N, 1);
        CLOSURE_FETCH_ARG (m_roughness, 2);
        CLOSURE_FETCH_ARG (m_edginess, 3);
        CLOSURE_FETCH_ARG (m_backscatter, 4);
    }

    void print_on (std::ostream &out) const
    {
        out << "westin_velvet (";
        out << "(" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";
        out << m_roughness << ", ";
        out << m_edginess  << ", ";
        out << m_backscatter;
        out << ")";
    }

    Color3 eval_reflect (const Vec3 &omega_out, const Vec3 &omega_in, float &pdf) const
    {
        float cosine, sine, westin;

        cosine = std::max(omega_out.dot(omega_in), 0.0f);
        westin = powf(cosine, 1.0f/m_roughness) * m_backscatter;
        
        cosine = fabsf(m_N.dot(omega_out));      
        sine = sqrtf(std::max(1.0f - cosine*cosine, 0.f));

        westin += powf(sine, m_edginess) * fabsf(m_N.dot(omega_in));
        
        return Color3 (westin, westin, westin);
    }

    Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float &pdf) const
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
            // we are viewing the surface from above - send a ray out with uniform
            // distribution over the hemisphere
            sample_uniform_hemisphere (Nf, omega_out, randu, randv, omega_in, pdf);
        
            if (Ngf.dot(omega_in) > 0) {           
                float cosine, sine, westin;

                cosine = std::max(omega_out.dot(omega_in), 0.0f);
                westin = powf(cosine, 1.0f/m_roughness) * m_backscatter;
        
                cosine = fabsf(Nf.dot(omega_out));      
                sine = sqrtf(std::max(1.0f - cosine*cosine, 0.f));

                float power = westin + powf(sine, m_edginess) * Nf.dot(omega_in);
            
                eval.setValue(power, power, power);
                // TODO: find a better approximation for the retroreflective bounce
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

DECLOP (OP_westin_velvet)
{
    closure_op_guts<WestinVelvetClosure, 5> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}

}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif


