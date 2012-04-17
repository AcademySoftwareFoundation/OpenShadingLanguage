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

#include "oslexec_pvt.h"


OSL_NAMESPACE_ENTER



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
            else
                phi = 0.0f;
        }
    }
    x = r * cosf (phi);
    y = r * sinf (phi);
}



/// Make two unit vectors that are orthogonal to N and each other.  This
/// assumes that N is already normalized.  We get the first orthonormal
/// by taking the cross product of N and (1,1,1), unless N is 1,1,1, in
/// which case we cross with (-1,1,1).  Either way, we get something
/// orthogonal.  Then N x a is mutually orthogonal to the other two.
void
ClosurePrimitive::make_orthonormals (const Vec3 &N, Vec3 &a, Vec3 &b)
{
    if (N[0] != N[1] || N[0] != N[2])
        a = Vec3 (N[2]-N[1], N[0]-N[2], N[1]-N[0]);  // (1,1,1) x N
    else
        a = Vec3 (N[2]-N[1], N[0]+N[2], -N[1]-N[0]);  // (-1,1,1) x N
    a.normalize ();
    b = N.cross (a);
}

void
ClosurePrimitive::make_orthonormals (const Vec3 &N, const Vec3& T, Vec3 &x, Vec3& y)
{
    y = N.cross(T);
    x = y.cross(N);
}
    
float
ClosurePrimitive::fresnel_dielectric (float eta, const Vec3 &N,
        const Vec3 &I, const Vec3 &dIdx, const Vec3 &dIdy,
        Vec3 &R, Vec3 &dRdx, Vec3 &dRdy,
        Vec3& T, Vec3 &dTdx, Vec3 &dTdy,
        bool &is_inside)
{
    float cos = N.dot(I), neta;
    Vec3 Nn;
    // compute reflection
    R = (2 * cos) * N - I;
    dRdx = (2 * N.dot(dIdx)) * N - dIdx;
    dRdy = (2 * N.dot(dIdy)) * N - dIdy;
    // check which side of the surface we are on
    if (cos > 0) {
        // we are on the outside of the surface, going in
        neta = 1 / eta;
        Nn   = N;
        is_inside = false;
    } else {
        // we are inside the surface, 
        cos  = -cos;
        neta = eta;
        Nn   = -N;
        is_inside = true;
    }
    R = (2 * cos) * Nn - I;
    float arg = 1 - (neta * neta * (1 - (cos * cos)));
    if (arg < 0) {
        T.setValue(0, 0, 0);
        dTdx.setValue(0, 0, 0);
        dTdy.setValue(0, 0, 0);
        return 1; // total internal reflection
    } else {
        float dnp = std::sqrt(arg);
        float nK = (neta * cos) - dnp;
        T = -(neta * I) + (nK * Nn);
        dTdx = -(neta * dIdx) + ((neta - neta * neta * cos / dnp) * dIdx.dot(Nn)) * Nn;
        dTdy = -(neta * dIdy) + ((neta - neta * neta * cos / dnp) * dIdy.dot(Nn)) * Nn;
        // compute Fresnel terms
        float cosTheta1 = cos; // N.R
        float cosTheta2 = -Nn.dot(T);
        float pPara = (cosTheta1 - eta * cosTheta2) / (cosTheta1 + eta * cosTheta2);
        float pPerp = (eta * cosTheta1 - cosTheta2) / (eta * cosTheta1 + cosTheta2);
        return 0.5f * (pPara * pPara + pPerp * pPerp);
    }
}

float
ClosurePrimitive::fresnel_dielectric(float cosi, float eta)
{
    // compute fresnel reflectance without explicitly computing
    // the refracted direction
    float c = fabsf(cosi);
    float g = eta * eta - 1 + c * c;
    if (g > 0) {
        g = sqrtf(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1) / (c * (g - c) + 1);
        return 0.5f * A * A * (1 + B * B);
    }
    return 1.0f; // TIR (no refracted component)
}

float
ClosurePrimitive::fresnel_conductor (float cosi, float eta, float k)
{
    float tmp_f = eta * eta + k * k;
    float tmp = tmp_f * cosi * cosi;
    float Rparl2 = (tmp - (2.0f * eta * cosi) + 1) /
                   (tmp + (2.0f * eta * cosi) + 1);
    float Rperp2 = (tmp_f - (2.0f * eta * cosi) + cosi * cosi) /
                   (tmp_f + (2.0f * eta * cosi) + cosi * cosi);
    return (Rparl2 + Rperp2) * 0.5f;
}



void
ClosurePrimitive::sample_cos_hemisphere (const Vec3 &N, const Vec3 &omega_out,
                                         float randu, float randv,
                                         Vec3 &omega_in, float &pdf)
{
    // Default closure BSDF implementation: uniformly sample
    // cosine-weighted hemisphere above the point.
    to_unit_disk (randu, randv);
    float costheta = sqrtf (std::max(1 - randu * randu - randv * randv, 0.0f));
    Vec3 T, B;
    make_orthonormals (N, T, B);
    omega_in = randu * T + randv * B + costheta * N;
    pdf = costheta * (float) M_1_PI;
}



void 
ClosurePrimitive::sample_uniform_hemisphere (const Vec3 &N, const Vec3 &omega_out,
                                             float randu, float randv, 
                                             Vec3 &omega_in, float &pdf)
{
    float z = randu;
    float r = sqrtf(std::max(0.f, 1.f - z*z));
    float phi = 2.f * M_PI * randv;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    
    Vec3 T, B;
    make_orthonormals (N, T, B);
    omega_in = x * T + y * B + z * N;
    pdf = 0.5f * (float) M_1_PI;
    
}

OSL_NAMESPACE_EXIT
