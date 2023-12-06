// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <optix.h>


#include "rend_lib.h"
#include "render_params.h"
#include "vec_math.h"
#include "wrapper.h"


static __device__ __inline__ void
calc_uv(float3 shading_normal, float& u, float& v, float3& dPdu, float3& dPdv)
{
    const float3 n = shading_normal;

    const float nx = n.x;
    const float ny = n.y;
    const float nz = n.z;

    u = (atan2(nx, nz) + M_PI) * 0.5f * float(M_1_PI);
    v = acos(ny) * float(M_1_PI);

    float xz2 = nx * nx + nz * nz;
    if (xz2 > 0.0f) {
        const float PI    = float(M_PI);
        const float TWOPI = float(2 * M_PI);
        float xz          = sqrtf(xz2);
        float inv         = 1.0f / xz;
        dPdu              = make_float3(-TWOPI * nx, TWOPI * nz, 0.0f);
        dPdv = make_float3(-PI * nz * inv * ny, -PI * nx * inv * ny, PI * xz);
    } else {
        // pick arbitrary axes for poles to avoid division by 0
        if (ny > 0.0f) {
            dPdu = make_float3(0.0f, 0.0f, 1.0f);
            dPdv = make_float3(1.0f, 0.0f, 0.0f);
        } else {
            dPdu = make_float3(0.0f, 0.0f, 1.0f);
            dPdv = make_float3(-1.0f, 0.0f, 0.0f);
        }
    }
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

    calc_uv(sg->N, sg->u, sg->v, sg->dPdu, sg->dPdv);
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

    float3 oc = sphere.c - ray_origin;
    float b   = dot(oc, ray_direction);
    float det = b * b - dot(oc, oc) + sphere.r2;
    if (det >= 0.0f) {
        det     = sqrtf(det);
        float x = b - det;
        float y = b + det;

        // NB: this does not included the 'self' check from
        // the testrender sphere intersection
        float t = (x > 0) ? x : ((y > 0) ? y : 0);

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

        // NB: this does not included the 'self' check from
        // the testrender sphere intersection
        float t = (x > 0) ? x : ((y > 0) ? y : 0);
        if (t < optixGetRayTmax())
            optixReportIntersection(t, RAYTRACER_HIT_SPHERE);
    }
}
