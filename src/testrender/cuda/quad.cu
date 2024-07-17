// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <optix.h>

#include "../raytracer.h"
#include "optix_raytracer.h"
#include "rend_lib.h"
#include "vec_math.h"


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
    const QuadParams& params = g_quads[idx];
    const float3 P           = ray_origin + t_hit * ray_direction;

    sg->I = ray_direction;
    sg->N = sg->Ng  = params.n;
    sg->surfacearea = params.a;
    sg->shaderID    = params.shaderID;
    sg->backfacing  = dot(V3_TO_F3(sg->N), V3_TO_F3(sg->I)) > 0.0f;

    if (sg->backfacing) {
        sg->N  = -sg->N;
        sg->Ng = -sg->Ng;
    }

    const OSL::Quad quad(F3_TO_V3(params.p), F3_TO_V3(params.ex),
                         F3_TO_V3(params.ey), 0, false);
    OSL::Vec3 dPdu, dPdv;
    OSL::Dual2<OSL::Vec2> uv = quad.uv(F3_TO_V3(P), F3_TO_V3(sg->N), dPdu,
                                       dPdv);
    sg->u                    = uv.val().x;
    sg->v                    = uv.val().y;
    sg->dPdu                 = V3_TO_F3(dPdu);
    sg->dPdv                 = V3_TO_F3(dPdv);
}


extern "C" __global__ void
__intersection__quad()
{
    const GenericData* g_data = reinterpret_cast<const GenericData*>(
        optixGetSbtDataPointer());
    const QuadParams* g_quads = reinterpret_cast<const QuadParams*>(
        g_data->data);

    Payload payload;
    payload.get();
    const OSL_CUDA::ShaderGlobals* sg_ptr
        = reinterpret_cast<OSL_CUDA::ShaderGlobals*>(payload.sg_ptr);
    const TraceData* tracedata = reinterpret_cast<TraceData*>(
        sg_ptr->tracedata);
    const int obj_id           = tracedata->obj_id;
    const unsigned int idx     = optixGetPrimitiveIndex();
    const QuadParams& params   = g_quads[idx];
    const float3 ray_origin    = optixGetObjectRayOrigin();
    const float3 ray_direction = optixGetObjectRayDirection();
    const bool self            = obj_id == params.objID;

    if (self)
        return;

    const OSL::Quad quad(F3_TO_V3(params.p), F3_TO_V3(params.ex),
                         F3_TO_V3(params.ey), 0, false);
    const OSL::Ray ray(F3_TO_V3(ray_origin), F3_TO_V3(ray_direction),
                       payload.radius, payload.spread, payload.raytype);
    const OSL::Dual2<float> t = quad.intersect(ray, self);

    if (t.val() != 0.0f && t.val() < optixGetRayTmax())
        optixReportIntersection(t.val(), RAYTRACER_HIT_QUAD);
}
