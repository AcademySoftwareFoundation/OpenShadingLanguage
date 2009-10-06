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

#include "oslclosure.h"

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {

class UniformEmissiveClosure : public EmissiveClosure {
public:
    UniformEmissiveClosure () : EmissiveClosure ("emission", "") { }

    Color3 eval (const void *paramsptr, const Vec3 &N, 
                 const Vec3 &omega_out) const
    {
        // make sure the outgoing direction is on the right side of the surface
        const float invpi = N.dot(omega_out) > 0 ? (float) (0.5f * M_1_PI) : 0.0f;
        return Color3(invpi, invpi, invpi);
    }

    void sample (const void *paramsptr, const Vec3 &N, float randu, float randv,
                 Vec3 &omega_out, float &pdf) const
    {
        // sample a random direction uniformly on the hemisphere
        Vec3 T, B;
        make_orthonormals(N, T, B);
        float phi = 2 * (float) M_PI * randu;
        float cosTheta = randv;
        float sinTheta = sqrtf(1 - cosTheta * cosTheta);
        omega_out = (cosf(phi) * sinTheta) * T +
                    (sinf(phi) * sinTheta) * B +
                                 cosTheta  * N;
        pdf = (float) (0.5f * M_1_PI);
    }

    /// Return the probability distribution function in the direction omega_out,
    /// given the parameters and the light's surface normal.  This MUST match
    /// the PDF computed by sample().
    float pdf (const void *paramsptr, const Vec3 &N,
               const Vec3 &omega_out) const
    {
        // make sure the outgoing direction is on the right side of the surface
        return N.dot(omega_out) > 0 ? (float) (0.5f * M_1_PI) : 0.0f;
    }
};

// these are all singletons
UniformEmissiveClosure uniform_emissive_closure_primitive;

}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
