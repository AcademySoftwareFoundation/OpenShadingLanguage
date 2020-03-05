// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage


#include <optix.h>

#if (OPTIX_VERSION < 70000)
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

#else //#if (OPTIX_VERSION < 70000)

#include "wrapper.h"
#include "rend_lib.h"
#include "render_params.h"

extern "C" __device__
void __direct_callable__quad_shaderglobals (const unsigned int idx,
                                            const float        t_hit,
                                            const float3       ray_origin,
                                            const float3       ray_direction,
                                            ShaderGlobals     *sg)
{
    const GenericData *g_data  = reinterpret_cast<const GenericData *>(optixGetSbtDataPointer());
    const QuadParams *g_quads  = reinterpret_cast<const QuadParams *>(g_data->data);
    const QuadParams &quad     = g_quads[idx];
    const float3 P = ray_origin + t_hit * ray_direction;

    float3 h  = P - quad.p;

    sg->N = sg->Ng = quad.n;
    sg->u    = dot (h, quad.ex) * quad.eu;
    sg->v    = dot (h, quad.ey) * quad.ev;
    sg->dPdu = quad.ey;
    sg->dPdv = quad.ex;
    sg->surfacearea = quad.a;
    sg->shaderID    = quad.shaderID;
}


extern "C" __global__
void __intersection__quad ()
{
    const GenericData *g_data  = reinterpret_cast<const GenericData *>(optixGetSbtDataPointer());
    const QuadParams *g_quads  = reinterpret_cast<const QuadParams *>(g_data->data);
    const unsigned int idx     = optixGetPrimitiveIndex();
    const QuadParams &quad     = g_quads[idx];
    const float3 ray_origin    = optixGetObjectRayOrigin();
    const float3 ray_direction = optixGetObjectRayDirection();

    float dn = dot(ray_direction, quad.n);
    float en = dot(quad.p - ray_origin, quad.n);
    if (dn * en > 0) {
        float  t  = en / dn;
        float3 h  = (ray_origin + ray_direction * t) - quad.p;
        float  dx = dot(h, quad.ex) * quad.eu;
        float  dy = dot(h, quad.ey) * quad.ev;

        if (dx >= 0 && dx < 1.0f && dy >= 0 && dy < 1.0f && t < optixGetRayTmax())
            optixReportIntersection (t, RAYTRACER_HIT_QUAD);
    }
}

#endif //#if (OPTIX_VERSION < 70000)
