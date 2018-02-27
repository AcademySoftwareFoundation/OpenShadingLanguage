/*
Copyright (c) 2009-2018 Sony Pictures Imageworks Inc., et al.
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


#include <optix.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>

#include <OSL/oslconfig.h>
#include <OSL/shaderglobals.h>
#include <OSL/dual.h>
#include <OSL/dual_vec.h>
#include <OpenImageIO/fmath.h>


using namespace optix;
using OSL::Dual2;
using OSL::Vec3;


rtDeclareVariable (float4, sphere, , );
rtDeclareVariable (float,  r2, , );
rtDeclareVariable (float,  a, , );

rtDeclareVariable (float3, texcoord,         attribute texcoord, );
rtDeclareVariable (float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable (float3, shading_normal,   attribute shading_normal, );
rtDeclareVariable (float,  surface_area,     attribute surface_area, );

rtDeclareVariable (float3, dPdu, attribute dPdu, );
rtDeclareVariable (float3, dPdv, attribute dPdv, );

rtDeclareVariable (optix::Ray, ray, rtCurrentRay, );


static __device__ __inline__
void calc_uv()
{
    Dual2<Vec3> n (Vec3 (shading_normal.x, shading_normal.y, shading_normal.z));

    Dual2<float> nx(n.val().x, n.dx().x, n.dy().x);
    Dual2<float> ny(n.val().y, n.dx().y, n.dy().y);
    Dual2<float> nz(n.val().z, n.dx().z, n.dy().z);
    Dual2<float> u = (fast_atan2(nx, nz) + Dual2<float>(M_PI)) * 0.5f * float(M_1_PI);
    Dual2<float> v = fast_acos(ny) * float(M_1_PI);
    float xz2 = nx.val() * nx.val() + nz.val() * nz.val();
    if (xz2 > 0.0f) {
        const float PI = float(M_PI);
        const float TWOPI = float(2 * M_PI);
        float xz = sqrtf(xz2);
        float inv = 1.0f / xz;
        dPdu.x = -TWOPI * nx.val();
        dPdu.y = TWOPI * nz.val();
        dPdu.z = 0.0f;
        dPdv.x = -PI * nz.val() * inv * ny.val();
        dPdv.y = -PI * nx.val() * inv * ny.val();
        dPdv.z = PI * xz;
    } else {
        // pick arbitrary axes for poles to avoid division by 0
        if (ny.val() > 0.0f) {
            dPdu = make_float3 (0.0f, 0.0f, 1.0f);
            dPdv = make_float3 (1.0f, 0.0f, 0.0f);
        } else {
            dPdu = make_float3 ( 0.0f, 0.0f, 1.0f);
            dPdv = make_float3 (-1.0f, 0.0f, 0.0f);
        }
    }
    texcoord = make_float3 (u.val(), v.val(), 0.0f);
}


// Intersection adapted from testrender/raytracer.h
RT_PROGRAM void intersect (void)
{
    float3 c   = make_float3(sphere);
    float3 oc  = c - ray.origin;
    float  b   = dot(oc, ray.direction);
    float  det = b * b - dot(oc, oc) + r2;

    if (det >= 0.0f) {
        det = sqrtf(det);
        float x = b - det;
        float y = b + det;

        // NB: this does not included the 'self' check from
        // the testrender sphere intersection
        float t = (x > 0) ? x : ((y > 0) ? y : 0);

        if (rtPotentialIntersection(t)) {
            float3 P = ray.origin + ray.direction * t;
            float3 N = normalize (P - c);
            shading_normal = geometric_normal = N;
            surface_area = a;

            // Calcuate the texture coordinates and derivatives
            calc_uv();

            rtReportIntersection(0);
        }
    }
}


RT_PROGRAM void bounds (int, float result[6])
{
    const float3 center = make_float3(sphere);
    const float3 radius = make_float3(sphere.w);

    optix::Aabb* aabb = reinterpret_cast<optix::Aabb*>(result);

    if (radius.x > 0.0f && !isinf(radius.x)) {
        aabb->m_min = center - radius;
        aabb->m_max = center + radius;
    } else {
        aabb->invalidate();
    }
}
