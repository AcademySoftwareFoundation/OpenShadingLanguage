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
#include "oslclosure.h"

#include <cmath>


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {

/// Variable cone emissive closure
///
/// This primitive emits in a cone having a configurable
/// penumbra area where the light decays to 0 reaching the
/// outer_angle limit. It can also behave as a lambertian emitter
/// if the provided angles are PI/2, which is the default
///
class GenericEmissiveClosure : public EmissiveClosure {
    // Two params, angles both
    // first is the outer_angle where penumbra starts
    float m_inner_angle; // must be between 0 and outer_angle
    // and second the angle where light ends
    float m_outer_angle;
public:
    CLOSURE_CTOR (GenericEmissiveClosure) : EmissiveClosure(side)
    {
        if (nargs >= 2 && exec->sym (args[1]).typespec().is_float())
            CLOSURE_FETCH_ARG (m_inner_angle, 1);
        else
            m_inner_angle = float(M_PI) * 0.5f;
        if (nargs >= 3 && exec->sym (args[2]).typespec().is_float())
            CLOSURE_FETCH_ARG (m_outer_angle, 2);
        else
            m_outer_angle = m_inner_angle;
    }

    bool mergeable (const ClosurePrimitive *other) const {
        const GenericEmissiveClosure *comp = (const GenericEmissiveClosure *)other;
        return m_inner_angle == comp->m_inner_angle &&
            m_outer_angle == comp->m_outer_angle && 
            m_sidedness == comp->m_sidedness &&
            EmissiveClosure::mergeable(other);
    }

    size_t memsize () const { return sizeof(*this); }

    const char *name () const { return "emission"; }

    void print_on (std::ostream &out) const {
        out << name() << " (" << m_inner_angle << ", " << m_outer_angle << ")";
    }

    Color3 eval (const Vec3 &Ng, const Vec3 &omega_out) const
    {
        float outer_angle = m_outer_angle < float(M_PI*0.5) ? m_outer_angle : float(M_PI*0.5);
        if (outer_angle < 0.0f)
            outer_angle = 0.0f;
        float inner_angle = m_inner_angle < outer_angle ? m_inner_angle : outer_angle;
        if (inner_angle < 0.0f)
            inner_angle = 0.0f;
        float cosNO = fabsf(Ng.dot(omega_out));
        float cosU  = cosf(inner_angle);
        float cosA  = cosf(outer_angle);
        float res;
        // Normalization factor
        float totalemit = ((1.0f - cosU*cosU) +
                           // The second term of this sum is just an
                           // approximation. The actual integral is of
                           // the "smooth step" we are using later is
                           // way more complicated. this will work as
                           // long as the penumbra is not too big
                           (cosU*cosU - cosA*cosA)*0.5f) * float(M_PI);
        if (cosNO > cosU) // Total light
            res = 1.0f / totalemit;
        else if (cosNO > cosA) { // penumbra, apply smooth step
            float x = 1.0f - (outer_angle - acosf(cosNO)) / (outer_angle - inner_angle);
            //res = (1.0 - 2*x*x + x*x*x*x) / totalemit;
            res = (1.0f - x*x*(3-2*x)) / totalemit;
        }
        else res = 0.0f; // out of cone
        return Color3(res, res, res);
    }

    void sample (const Vec3 &Ng, float randu, float randv,
                 Vec3 &omega_out, float &pdf) const
    {
        // We don't do anything sophisticated here for the step
        // We just sample the whole cone uniformly to the cosine
        Vec3 T, B;
        make_orthonormals(Ng, T, B);
        float outer_angle = m_outer_angle < M_PI*0.5 ? m_outer_angle : M_PI*0.5;
        if (outer_angle < 0.0f)
            outer_angle = 0.0f;
        float cosA  = cosf(outer_angle);
        float phi = 2 * (float) M_PI * randu;
        float cosTheta = sqrtf(1.0f - (1.0f - cosA*cosA) * randv);
        float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
        omega_out = (cosf(phi) * sinTheta) * T +
                    (sinf(phi) * sinTheta) * B +
                                 cosTheta  * Ng;
        pdf = 1.0f / ((1.0f - cosA*cosA) * float(M_PI));
    }

    /// Return the probability distribution function in the direction omega_out,
    /// given the parameters and the light's surface normal.  This MUST match
    /// the PDF computed by sample().
    float pdf (const Vec3 &Ng,
               const Vec3 &omega_out) const
    {
        float outer_angle = m_outer_angle < float(M_PI*0.5) ? m_outer_angle : float(M_PI*0.5);
        if (outer_angle < 0.0f)
            outer_angle = 0.0f;
        float cosNO = Ng.dot(omega_out);
        float cosA  = cosf(outer_angle);
        if (cosNO < cosA)
            return 0.0f;
        else
            return 1.0f / ((1.0f - cosA*cosA) * float(M_PI));
    }
};


DECLOP (OP_emission)
{
    if (nargs >= 3 && exec->sym (args[2]).typespec().is_float())
        closure_op_guts<GenericEmissiveClosure, 3> (exec, nargs, args);
    else if (nargs >= 2 && exec->sym (args[1]).typespec().is_float())
        closure_op_guts<GenericEmissiveClosure, 2> (exec, nargs, args);
    else
        closure_op_guts<GenericEmissiveClosure, 1> (exec, nargs, args);

}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
