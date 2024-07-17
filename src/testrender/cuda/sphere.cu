// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <optix.h>


#include "../raytracer.h"
#include "optix_raytracer.h"
#include "rend_lib.h"
#include "vec_math.h"


static __device__ __inline__ void
calc_uv(OSL::Vec3 n, float& u, float& v, OSL::Vec3& dPdu, OSL::Vec3& dPdv, float r)
{
    u = (atan2(n.x, n.z) + float(M_PI)) * 0.5f
        * float(M_1_PI);
    v = acos(n.y) * float(M_1_PI);
    const float pi = float(M_PI);
    float twopiu   = 2.0f * pi * u;
    float sin2piu, cos2piu;
    OIIO::sincos(twopiu, &sin2piu, &cos2piu);
    float sinpiv, cospiv;
    OIIO::sincos(pi * v, &sinpiv, &cospiv);
    float pir = pi * r;
    dPdu.x    = -2.0f * pir * sinpiv * cos2piu;
    dPdu.y    = 0.0f;
    dPdu.z    = 2.0f * pir * sinpiv * sin2piu;
    dPdv.x    = -pir * cospiv * sin2piu;
    dPdv.y    = -pir * sinpiv;
    dPdv.z    = -pir * cospiv * cos2piu;
}


extern "C" __device__ void
__direct_callable__sphere_shaderglobals(const unsigned int idx,
                                        const float t_hit,
                                        const float3 ray_origin,
                                        const float3 ray_direction,
                                        OSL_CUDA::ShaderGlobals* sg)
{
    const GenericData* g_data = reinterpret_cast<const GenericData*>(
        optixGetSbtDataPointer());
    const SphereParams* g_spheres = reinterpret_cast<const SphereParams*>(
        g_data->data);
    const SphereParams& sphere = g_spheres[idx];
    const float3 P             = ray_origin + t_hit * ray_direction;

    sg->N = sg->Ng  = normalize(P - sphere.c);
    sg->surfacearea = sphere.a;
    sg->shaderID    = sphere.shaderID;

    calc_uv(sg->N, sg->u, sg->v, sg->dPdu, sg->dPdv, sphere.r);
}


extern "C" __global__ void
__intersection__sphere()
{
    const GenericData* g_data = reinterpret_cast<const GenericData*>(
        optixGetSbtDataPointer());
    const SphereParams* g_spheres = reinterpret_cast<const SphereParams*>(
        g_data->data);

    Payload payload;
    payload.get();
    const OSL_CUDA::ShaderGlobals* sg_ptr
        = reinterpret_cast<OSL_CUDA::ShaderGlobals*>(payload.sg_ptr);
    const TraceData* tracedata = reinterpret_cast<TraceData*>(
        sg_ptr->tracedata);
    const int obj_id           = tracedata->obj_id;
    const unsigned int idx     = optixGetPrimitiveIndex();
    const SphereParams& params = g_spheres[idx];
    const float3 ray_origin    = optixGetObjectRayOrigin();
    const float3 ray_direction = optixGetObjectRayDirection();
    const bool self            = obj_id == params.objID;

    const OSL::Sphere sphere(F3_TO_V3(params.c), params.r, 0, false);
    const OSL::Ray ray(F3_TO_V3(ray_origin), F3_TO_V3(ray_direction),
                       payload.radius, payload.spread, payload.raytype);
    const OSL::Dual2<float> t = sphere.Sphere::intersect(ray, self);

    if (t.val() != 0.0f && t.val() < optixGetRayTmax())
        optixReportIntersection(t.val(), RAYTRACER_HIT_SPHERE);
}
