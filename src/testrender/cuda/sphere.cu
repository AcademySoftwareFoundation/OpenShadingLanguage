// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <optix.h>


#include "rend_lib.h"
#include "render_params.h"
#include "vec_math.h"
#include "wrapper.h"


static __device__ __inline__ void
calc_uv(float3 shading_normal, float& u, float& v, float3& dPdu, float3& dPdv, float r)
{
    const float3 n = shading_normal;
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
    const unsigned int idx     = optixGetPrimitiveIndex();
    const SphereParams& sphere = g_spheres[idx];
    const float3 ray_origin    = optixGetObjectRayOrigin();
    const float3 ray_direction = optixGetObjectRayDirection();

    Payload payload;
    payload.get();

    OSL_CUDA::ShaderGlobals* sg_ptr = (OSL_CUDA::ShaderGlobals*)payload.ptr.ptr;
    uint32_t* trace_data            = (uint32_t*)sg_ptr->tracedata;
    const int hit_idx               = ((int*)trace_data)[2];
    const int hit_kind              = ((int*)trace_data)[3];
    const bool self                 = hit_idx == idx && hit_kind == 1;

    float3 oc = sphere.c - ray_origin;
    float b   = dot(oc, ray_direction);
    float det = b * b - dot(oc, oc) + sphere.r2;

    if (det >= 0.0f) {
        det     = sqrtf(det);
        float x = b - det;
        float y = b + det;
        float t = self ? (fabsf(x) > fabsf(y) ? (x > 0 ? x : 0)
                                              : (y > 0 ? y : 0))
                       : (x > 0 ? x : (y > 0 ? y : 0));

        if (t < optixGetRayTmax())
            optixReportIntersection(t, RAYTRACER_HIT_SPHERE);
    }
}


extern "C" __global__ void
__intersection__sphere_precise()
{
    const GenericData* g_data = reinterpret_cast<const GenericData*>(
        optixGetSbtDataPointer());
    const SphereParams* g_spheres = reinterpret_cast<const SphereParams*>(
        g_data->data);
    const unsigned int idx     = optixGetPrimitiveIndex();
    const SphereParams& sphere = g_spheres[idx];
    const float3 ray_origin    = optixGetObjectRayOrigin();
    const float3 ray_direction = optixGetObjectRayDirection();

    Payload payload;
    payload.get();

    OSL_CUDA::ShaderGlobals* sg_ptr = (OSL_CUDA::ShaderGlobals*)payload.ptr.ptr;
    uint32_t* trace_data            = (uint32_t*)sg_ptr->tracedata;
    const int hit_idx               = ((int*)trace_data)[2];
    const int hit_kind              = ((int*)trace_data)[3];
    const bool self                 = hit_idx == idx && hit_kind == 1;

    // Using the "round up" single-precision intrinsics helps match the results
    // from the CPU path more closely when fast-math is enabled.
    float3 oc = sub_ru(sphere.c, ray_origin);
    float b   = dot_ru(oc, ray_direction);
    float det = __fmul_ru(b, b);
    det       = __fsub_ru(det, dot_ru(oc, oc));
    det       = __fadd_ru(det, sphere.r2);

    if (det > 0.0f) {
        det     = __fsqrt_ru(det);
        float x = __fsub_ru(b, det);
        float y = __fadd_ru(b, det);
        float t = self ? (fabsf(x) > fabsf(y) ? (x > 0 ? x : 0)
                                              : (y > 0 ? y : 0))
                       : (x > 0 ? x : (y > 0 ? y : 0));

        if (t < optixGetRayTmax())
            optixReportIntersection(t, RAYTRACER_HIT_SPHERE);
    }
}
