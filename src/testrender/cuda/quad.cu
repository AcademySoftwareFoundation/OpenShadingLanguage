// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <optix.h>

#include "optix_raytracer.h"
#include "rend_lib.h"
#include "vec_math.h"
#include "wrapper.h"


extern "C" __device__ void
__direct_callable__quad_shaderglobals(const unsigned int idx, const float t_hit,
                                      const float3 ray_origin,
                                      const float3 ray_direction,
                                      OSL_CUDA::ShaderGlobals* sg)
{
    const GenericData* g_data = reinterpret_cast<const GenericData*>(
        optixGetSbtDataPointer());
    const QuadParams* g_quads = reinterpret_cast<const QuadParams*>(
        g_data->data);
    const QuadParams& quad = g_quads[idx];
    const float3 P         = ray_origin + t_hit * ray_direction;

    float3 h = P - quad.p;

    sg->I = ray_direction;
    sg->N = sg->Ng  = quad.n;
    sg->u           = dot(h, quad.ex) * quad.eu;
    sg->v           = dot(h, quad.ey) * quad.ev;
    sg->dPdu        = quad.ey;
    sg->dPdv        = quad.ex;
    sg->surfacearea = quad.a;
    sg->shaderID    = quad.shaderID;
    sg->backfacing   = dot(sg->N, sg->I) > 0.0f;

    if (sg->backfacing) {
        sg->N  = -sg->N;
        sg->Ng = -sg->Ng;
    }
}


extern "C" __global__ void
__intersection__quad()
{
    const unsigned int idx = optixGetPrimitiveIndex();

    // Check for self-intersection
    Payload payload;
    payload.get();
    OSL_CUDA::ShaderGlobals* sg_ptr = (OSL_CUDA::ShaderGlobals*)payload.sg_ptr;
    uint32_t* trace_data            = (uint32_t*)sg_ptr->tracedata;
    const int hit_idx               = ((int*)trace_data)[2];
    const int hit_kind              = ((int*)trace_data)[3];
    const bool self = hit_idx == idx && hit_kind == RAYTRACER_HIT_QUAD;
    if (self) {
        return;
    }

    const GenericData* g_data = reinterpret_cast<const GenericData*>(
        optixGetSbtDataPointer());
    const QuadParams* g_quads = reinterpret_cast<const QuadParams*>(
        g_data->data);
    const QuadParams& quad     = g_quads[idx];
    const float3 ray_origin    = optixGetObjectRayOrigin();
    const float3 ray_direction = optixGetObjectRayDirection();

    float dn = dot(ray_direction, quad.n);
    float en = dot(quad.p - ray_origin, quad.n);
    if (dn * en > 0) {
        float t  = en / dn;
        float3 h = (ray_origin + ray_direction * t) - quad.p;
        float dx = dot(h, quad.ex) * quad.eu;
        float dy = dot(h, quad.ey) * quad.ev;

        if (dx >= 0 && dx < 1.0f && dy >= 0 && dy < 1.0f
            && t < optixGetRayTmax())
            optixReportIntersection(t, RAYTRACER_HIT_QUAD);
    }
}
