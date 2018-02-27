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


using namespace optix;

rtDeclareVariable (float4, sphere, , );
rtDeclareVariable (float,  r2, , );

rtDeclareVariable (float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable (float3, shading_normal, attribute shading_normal, );
rtDeclareVariable (optix::Ray, ray, rtCurrentRay, );


// intersection adapted from testrender/raytracer.h
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
            float3 N = P - c;
            shading_normal = geometric_normal = N;
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
