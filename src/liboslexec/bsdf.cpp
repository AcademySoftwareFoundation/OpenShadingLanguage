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
#include "oslclosure.h"
#include "oslexec_pvt.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {



/// Given values x and y on [0,1], convert them in place to values on
/// [-1,1] uniformly distributed over a unit sphere.  This code is
/// derived from Peter Shirley, "Realistic Ray Tracing", p. 103.
static void
to_unit_disk (float &x, float &y)
{
    float r, phi;
    float a = 2.0f * x - 1.0f;
    float b = 2.0f * y - 1.0f;
    if (a > -b) {
        if (a > b) {
	    r = a;
	    phi = M_PI_4 * (b/a);
	} else {
	    r = b;
	    phi = M_PI_4 * (2.0f - a/b);
	}
    } else {
	if (a < b) {
	    r = -a;
	    phi = M_PI_4 * (4.0f + b/a);
	} else {
	    r = -b;
	    if (b != 0.0f)
		phi = M_PI_4 * (6.0f - a/b);
	    else phi = 0.0f;
	}
    }
    x = r * cosf (phi);
    y = r * sinf (phi);
}



void
ClosurePrimitive::sample (const ClosureColorComponent &comp, const Vec3 &P,
                          const Vec3 &N, const Vec3 &T, const Vec3 &B,
                          const Vec3 &I, float randu, float randv,
                          Vec3 &R, float &pdf) const
{
    // Default closure BSDF implementation: uniformly sample
    // cosine-weighted hemisphere above the point.
    to_unit_disk (randu, randv);
    float costheta = sqrtf (1.0f - randu*randu - randv*randv);
    R = randu * T + randv * B + costheta * N;
    pdf = costheta / M_PI;
}



namespace pvt {


class DiffuseClosure : public ClosurePrimitive {
public:
    DiffuseClosure () : ClosurePrimitive (Strings::diffuse, 0, ustring()) { }

    bool eval (const ClosureColorComponent &comp, const Vec3 &P,
               const Vec3 &N, const Vec3 &T, const Vec3 &B,
               const Vec3 &L, const Color3 &El,
               const Vec3 &R, Color3 &Er) const
    {
        if (N.dot(L) > 0.0f) {
            Er.setValue (1.0f, 1.0f, 1.0f);
            return true;
        } else {
            Er.setValue (0.0f, 0.0f, 0.0f);
            return false;
        }
    }

};



DiffuseClosure diffuse_closure_primitive;



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
