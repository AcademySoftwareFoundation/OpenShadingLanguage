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

rtDeclareVariable (float3, p,  , );
rtDeclareVariable (float3, ex, , );
rtDeclareVariable (float3, ey, , );
rtDeclareVariable (float3, n,  , );
rtDeclareVariable (float,  eu, , );
rtDeclareVariable (float,  ev, , );
rtDeclareVariable (float,  a, ,  );

rtDeclareVariable (float3, texcoord,         attribute texcoord, );
rtDeclareVariable (float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable (float3, shading_normal,   attribute shading_normal, );
rtDeclareVariable (float,  surface_area,     attribute surface_area, );

rtDeclareVariable (float3, dPdu, attribute dPdu, );
rtDeclareVariable (float3, dPdv, attribute dPdv, );

rtDeclareVariable (optix::Ray, ray, rtCurrentRay, );


RT_PROGRAM void intersect (void)
{
    float dn = dot(ray.direction, n);
    float en = dot(p - ray.origin, n);
    if (dn * en > 0) {
        float  t  = en / dn;
        float3 h  = (ray.origin + ray.direction * t) - p;
        float  dx = dot(h, ex) * eu;
        float  dy = dot(h, ey) * ev;

        if (dx >= 0 && dx < 1.0f && dy >= 0 && dy < 1.0f && rtPotentialIntersection(t)) {
            shading_normal = geometric_normal = n;
            texcoord = make_float3(dot (h, ex) * eu, dot (h, ey) * ev, 0.0f);
            dPdu = ey;
            dPdv = ex;
            surface_area = a;
            rtReportIntersection(0);
        }
    }
}


RT_PROGRAM void bounds (int, float result[6])
{
    const float3 p00  = p;
    const float3 p01  = p + ex;
    const float3 p10  = p + ey;
    const float3 p11  = p + ex + ey;
    const float  area = length(cross(ex, ey));

    optix::Aabb* aabb = reinterpret_cast<optix::Aabb*>(result);

    if (area > 0.0f && !isinf(area)) {
        aabb->m_min = fminf (fminf (p00, p01), fminf (p10, p11));
        aabb->m_max = fmaxf (fmaxf (p00, p01), fmaxf (p10, p11));
    } else {
        aabb->invalidate();
    }
}
