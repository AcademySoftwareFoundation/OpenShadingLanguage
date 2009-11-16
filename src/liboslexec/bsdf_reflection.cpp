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

class ReflectionClosure : public BSDFClosure {
    Vec3  m_N;    // shading normal
    float m_R0;   // fresnel reflectance at incidence
public:
    CLOSURE_CTOR (ReflectionClosure)
    {
        CLOSURE_FETCH_ARG (m_N , 1);
        if (nargs >= 3 && exec->sym (args[2]).typespec().is_float())
            CLOSURE_FETCH_ARG (m_R0, 2);
        else
            m_R0 = 1;
    }

    void print_on (std::ostream &out) const {
        out << "reflection (";
        out << "(" << m_N[0] << ", " << m_N[1] << ", " << m_N[2] << "), ";
        out << m_R0;
        out << ")";
    }

    bool get_cone(const Vec3 &omega_out, Vec3 &axis, float &angle) const
    {
        // does not need to be integrated directly
        return false;
    }

    Color3 eval (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &omega_in, Labels &labels) const
    {
        // should never be called - because get_cone is empty
        return Color3 (0.0f, 0.0f, 0.0f);
    }

    void sample (const Vec3 &Ng,
                 const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                 float randu, float randv,
                 Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                 float &pdf, Color3 &eval, Labels &labels) const
    {
        // only one direction is possible
        labels.set (Labels::SURFACE, Labels::REFLECT, Labels::SINGULAR);
        float cosNO = m_N.dot(omega_out);
        if (cosNO > 0) {
            omega_in = (2 * cosNO) * m_N - omega_out;
            domega_in_dx = 2 * m_N.dot(domega_out_dx) * m_N - domega_out_dx;
            domega_in_dy = 2 * m_N.dot(domega_out_dy) * m_N - domega_out_dy;
            pdf = 1;
            float value = fresnel_shlick(cosNO, m_R0);
            eval.setValue(value, value, value);
        }
    }

    float pdf (const Vec3 &Ng,
               const Vec3 &omega_out, const Vec3 &omega_in) const
    {
        // the pdf for an arbitrary direction is 0 because only a single
        // direction is actually possible
        return 0;
    }
};



DECLOP (OP_reflection)
{
    closure_op_guts<ReflectionClosure> (exec, nargs, args,
            runflags, beginpoint, endpoint);
}

}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
