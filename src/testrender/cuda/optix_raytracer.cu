// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <optix.h>
#include <optix_device.h>

#include <OSL/hashes.h>

#include "optix_raytracer.h"
#include "rend_lib.h"
#include "vec_math.h"

#include "../background.h"
#include "../raytracer.h"
#include "../render_params.h"
#include "../sampling.h"

// clang-format off
// These files must be included in this specific order
#include "../shading.h"
#include "../shading.cpp"
// clang-format on

#include <cstdint>


OSL_NAMESPACE_ENTER
namespace pvt {
__device__ CUdeviceptr s_color_system          = 0;
__device__ CUdeviceptr osl_printf_buffer_start = 0;
__device__ CUdeviceptr osl_printf_buffer_end   = 0;
__device__ uint64_t test_str_1                 = 0;
__device__ uint64_t test_str_2                 = 0;
__device__ uint64_t num_named_xforms           = 0;
__device__ CUdeviceptr xform_name_buffer       = 0;
__device__ CUdeviceptr xform_buffer            = 0;
}  // namespace pvt
OSL_NAMESPACE_EXIT


extern "C" {
__device__ __constant__ RenderParams render_params;
}


static __device__ void
globals_from_hit(ShaderGlobalsType& sg, float radius = 0.0f, float spread = 0.0f,
                 Ray::RayType raytype = Ray::RayType::CAMERA)
{
    ShaderGlobalsType local_sg;
    // hit-kind 0: quad hit
    //          1: sphere hit
    optixDirectCall<void, unsigned int, float, float3, float3, ShaderGlobalsType*>(
        optixGetHitKind(), optixGetPrimitiveIndex(), optixGetRayTmax(),
        optixGetWorldRayOrigin(), optixGetWorldRayDirection(), &local_sg);
    // Setup the ShaderGlobals
    const float3 ray_direction = optixGetWorldRayDirection();
    const float3 ray_origin    = optixGetWorldRayOrigin();
    const float t_hit          = optixGetRayTmax();

    // Construct a Ray in order to calculate P and its derivatives
    Ray ray(F3_TO_V3(ray_origin), F3_TO_V3(ray_direction), radius, spread,
            Ray::RayType::CAMERA);
    Dual2<float> t(t_hit);
    Dual2<Vec3> P = ray.point(t);

    sg.I  = ray_direction;
    sg.N  = normalize(optixTransformNormalFromObjectToWorldSpace(V3_TO_F3(local_sg.N)));
    sg.Ng = normalize(optixTransformNormalFromObjectToWorldSpace(V3_TO_F3(local_sg.Ng)));
    sg.P  = V3_TO_F3(P.val());
    sg.dPdx        = V3_TO_F3(P.dx());
    sg.dPdy        = V3_TO_F3(P.dy());
    sg.dPdu        = local_sg.dPdu;
    sg.dPdv        = local_sg.dPdv;
    sg.u           = local_sg.u;
    sg.v           = local_sg.v;
    sg.Ci          = nullptr;
    sg.surfacearea = local_sg.surfacearea;
    sg.backfacing  = dot(V3_TO_F3(sg.N), V3_TO_F3(sg.I)) > 0.0f;
    sg.shaderID    = local_sg.shaderID;

    if (sg.backfacing) {
        sg.N  = -sg.N;
        sg.Ng = -sg.Ng;
    }

    sg.raytype        = raytype;
    sg.flipHandedness = dot(V3_TO_F3(sg.N), cross(V3_TO_F3(sg.dPdx), V3_TO_F3(sg.dPdy))) < 0.0f;
}


static inline __device__ void
execute_shader(ShaderGlobalsType& sg, char* closure_pool)
{
    if (sg.shaderID < 0) {
        // TODO: should probably never get here ...
        return;
    }

    // Pack the "closure pool" into one of the ShaderGlobals pointers
    *(int*)&closure_pool[0] = 0;
    sg.renderstate          = &closure_pool[0];

    // Create some run-time options structs. The OSL shader fills in the structs
    // as it executes, based on the options specified in the shader source.
    NoiseOptCUDA noiseopt;
    TextureOptCUDA textureopt;
    TraceOptCUDA traceopt;

    // Pack the pointers to the options structs in a faux "context",
    // which is a rough stand-in for the host ShadingContext.
    ShadingContextCUDA shading_context = { &noiseopt, &textureopt, &traceopt };
    sg.context                         = &shading_context;

    // Run the OSL callable
    void* interactive_ptr = reinterpret_cast<void**>(
        render_params.interactive_params)[sg.shaderID];
    const unsigned int shaderIdx = 2u + sg.shaderID + 0u;
    optixDirectCall<void, ShaderGlobalsType*, void*, void*, void*, int, void*>(
        shaderIdx, &sg /*shaderglobals_ptr*/,
        nullptr /*groupdata_ptr*/,
        nullptr /*userdata_base_ptr*/,
        nullptr /*output_base_ptr*/,
        0 /*shadeindex - unused*/,
        interactive_ptr /*interactive_params_ptr*/
    );
}


static inline __device__ void
trace_ray(OptixTraversableHandle handle, const Payload& payload, const float3& origin,
          const float3& direction)
{
    uint32_t p0 = payload.raw[0];
    uint32_t p1 = payload.raw[1];
    uint32_t p2 = __float_as_uint(payload.radius);
    uint32_t p3 = __float_as_uint(payload.spread);
    uint32_t p4 = payload.raytype;

    optixTrace(handle,                         // handle
               origin,                         // origin
               direction,                      // direction
               1e-3f,                          // tmin
               1e13f,                          // tmax
               0,                              // ray time
               OptixVisibilityMask(1),         // visibility mask
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // ray flags
               0,                              // SBT offset
               1,                              // SBT stride
               0,                              // miss SBT offset
               p0, p1, p2, p3, p4);
};

//
// CudaScene
//

OSL_HOSTDEVICE bool
CudaScene::intersect(const Ray& r, Dual2<float>& t, int& primID, void* sg) const
{
    Payload payload;
    payload.sg_ptr  = reinterpret_cast<ShaderGlobalsType*>(sg);
    payload.radius  = r.radius;
    payload.spread  = r.spread;
    payload.raytype = *reinterpret_cast<const Ray::RayType*>(&r.raytype);
    TraceData tracedata(*payload.sg_ptr, primID);
    trace_ray(handle, payload, V3_TO_F3(r.origin), V3_TO_F3(r.direction));
    primID = tracedata.hit_id;
    t      = tracedata.hit_t;
    return (payload.sg_ptr->shaderID >= 0);
}


OSL_HOSTDEVICE float
CudaScene::shapepdf(int primID, const Vec3& x, const Vec3& p) const
{
    SphereParams* spheres = (SphereParams*)spheres_buffer;
    QuadParams* quads     = (QuadParams*)quads_buffer;
    if (primID < num_spheres) {
        const SphereParams& params = spheres[primID];
        const OSL::Sphere sphere(F3_TO_V3(params.c), params.r, 0, false);
        return sphere.shapepdf(x, p);
    } else {
        const QuadParams& params = quads[primID - num_spheres];
        const OSL::Quad quad(F3_TO_V3(params.p), F3_TO_V3(params.ex),
                             F3_TO_V3(params.ey), 0, false);
        return quad.shapepdf(x, p);
    }
}


OSL_HOSTDEVICE bool
CudaScene::islight(int primID) const
{
    SphereParams* spheres = (SphereParams*)spheres_buffer;
    QuadParams* quads     = (QuadParams*)quads_buffer;

    if (primID < num_spheres)
        return spheres[primID].isLight;
    return quads[primID - num_spheres].isLight;
}


OSL_HOSTDEVICE Vec3
CudaScene::sample(int primID, const Vec3& x, float xi, float yi,
                  float& pdf) const
{
    SphereParams* spheres = (SphereParams*)spheres_buffer;
    QuadParams* quads     = (QuadParams*)quads_buffer;
    if (primID < num_spheres) {
        const SphereParams& params = spheres[primID];
        const OSL::Sphere sphere(F3_TO_V3(params.c), params.r, 0, false);
        return sphere.sample(x, xi, yi, pdf);
    } else {
        const QuadParams& params = quads[primID - num_spheres];
        const OSL::Quad quad(F3_TO_V3(params.p), F3_TO_V3(params.ex),
                             F3_TO_V3(params.ey), 0, false);
        return quad.sample(x, xi, yi, pdf);
    }
}


OSL_HOSTDEVICE int
CudaScene::num_prims() const
{
    return num_spheres + num_quads;
}

//------------------------------------------------------------------------------

// Because clang++ 9.0 seems to have trouble with some of the texturing "intrinsics"
// let's do the texture look-ups in this file.
extern "C" __device__ float4
osl_tex2DLookup(void* handle, float s, float t, float dsdx, float dtdx, float dsdy, float dtdy)
{
    const float2 dx = {dsdx, dtdx};
    const float2 dy = {dsdy, dtdy};
    cudaTextureObject_t texID = cudaTextureObject_t(handle);
    return tex2DGrad<float4>(texID, s, t, dx, dy);
}


//
// OptiX Programs
//


extern "C" __global__ void
__miss__()
{
    uint3 launch_dims  = optixGetLaunchDimensions();
    uint3 launch_index = optixGetLaunchIndex();

    float3* output_buffer = reinterpret_cast<float3*>(
        render_params.output_buffer);

    int pixel            = launch_index.y * launch_dims.x + launch_index.x;
    output_buffer[pixel] = make_float3(0, 0, 1);
}


extern "C" __global__ void
__raygen__setglobals()
{
    uint3 launch_dims  = optixGetLaunchDimensions();
    uint3 launch_index = optixGetLaunchIndex();

    // Set global variables
    if (launch_index.x == 0 && launch_index.y == 0) {
        OSL::pvt::osl_printf_buffer_start
            = render_params.osl_printf_buffer_start;
        OSL::pvt::osl_printf_buffer_end = render_params.osl_printf_buffer_end;
        OSL::pvt::s_color_system        = render_params.color_system;
        OSL::pvt::test_str_1            = render_params.test_str_1;
        OSL::pvt::test_str_2            = render_params.test_str_2;
    }

    Background background;
    background.set_variables((Vec3*)render_params.bg_values,
                             (float*)render_params.bg_rows,
                             (float*)render_params.bg_cols,
                             render_params.bg_res);

    SimpleRaytracer raytracer;
    raytracer.background           = background;
    raytracer.backgroundResolution = render_params.bg_id >= 0
        ? render_params.bg_res
        : 0;
    raytracer.backgroundShaderID   = render_params.bg_id;
    raytracer.max_bounces          = render_params.max_bounces;
    raytracer.rr_depth             = 5;
    raytracer.show_albedo_scale    = render_params.show_albedo_scale;

    if (render_params.bg_id < 0)
        return;

    auto evaler = [&](const Dual2<Vec3>& dir) {
        return raytracer.eval_background(dir, nullptr);
    };

    // Background::prepare_cuda must run on a single warp
    assert(launch_index.x < 32 && launch_index.y == 0);
    background.prepare_cuda(launch_dims.x, launch_index.x, evaler);
}


extern "C" __global__ void
__miss__setglobals()
{
}


extern "C" __global__ void
__closesthit__deferred()
{
    Payload payload;
    payload.get();
    ShaderGlobalsType* sg_ptr = payload.sg_ptr;
    TraceData* tracedata      = reinterpret_cast<TraceData*>(sg_ptr->tracedata);
    globals_from_hit(*sg_ptr, payload.radius, payload.spread, payload.raytype);

    const unsigned int hit_idx  = optixGetPrimitiveIndex();
    const unsigned int hit_kind = optixGetHitKind();
    if (hit_kind == 0) {
        const QuadParams* quads = reinterpret_cast<const QuadParams*>(
            render_params.quads_buffer);
        tracedata->hit_id = quads[hit_idx].objID;
    } else if (hit_kind == 1) {
        const SphereParams* spheres = reinterpret_cast<const SphereParams*>(
            render_params.spheres_buffer);
        tracedata->hit_id = spheres[hit_idx].objID;
    }
    const float hit_t = optixGetRayTmax();
    tracedata->hit_t  = *(uint32_t*)&hit_t;
}


extern "C" __global__ void
__raygen__deferred()
{
    Background background;
    background.set_variables((Vec3*)render_params.bg_values,
                             (float*)render_params.bg_rows,
                             (float*)render_params.bg_cols,
                             render_params.bg_res);

    Color3 result(0, 0, 0);
    const int aa = render_params.aa;
    for (int si = 0, n = aa * aa; si < n; si++) {
        uint3 launch_index = optixGetLaunchIndex();
        Sampler sampler(launch_index.x, launch_index.y, si);
        Vec3 j = sampler.get();
        // warp distribution to approximate a tent filter [-1,+1)^2
        j.x *= 2;
        j.x = j.x < 1 ? sqrtf(j.x) - 1 : 1 - sqrtf(2 - j.x);
        j.y *= 2;
        j.y = j.y < 1 ? sqrtf(j.y) - 1 : 1 - sqrtf(2 - j.y);

        if (render_params.no_jitter) {
            j *= 0.0f;
        }

        // Compute the pixel coordinates
        const float2 d
            = make_float2(static_cast<float>(launch_index.x) + 0.5f + j.x,
                          static_cast<float>(launch_index.y) + 0.5f + j.y);

        SimpleRaytracer raytracer;
        raytracer.background           = background;
        raytracer.backgroundResolution = render_params.bg_id >= 0
                                             ? render_params.bg_res
                                             : 0;
        raytracer.backgroundShaderID   = render_params.bg_id;
        raytracer.max_bounces          = render_params.max_bounces;
        raytracer.rr_depth             = 5;
        raytracer.show_albedo_scale    = render_params.show_albedo_scale;

        const Vec3 eye  = F3_TO_V3(render_params.eye);
        const Vec3 dir  = F3_TO_V3(render_params.dir);
        const Vec3 up   = F3_TO_V3(render_params.up);
        const float fov = render_params.fov;

        uint3 launch_dims = optixGetLaunchDimensions();
        raytracer.camera.resolution(launch_dims.x, launch_dims.y);
        raytracer.camera.lookat(eye, dir, up, fov);
        raytracer.camera.finalize();

        raytracer.scene = { render_params.num_spheres, render_params.num_quads,
                            render_params.spheres_buffer,
                            render_params.quads_buffer,
                            render_params.traversal_handle };

        Color3 r = raytracer.subpixel_radiance(d.x, d.y, sampler, nullptr);

        result = OIIO::lerp(result, r, 1.0f / (si + 1));
    }

    uint3 launch_dims     = optixGetLaunchDimensions();
    uint3 launch_index    = optixGetLaunchIndex();
    float3* output_buffer = reinterpret_cast<float3*>(
        render_params.output_buffer);
    int pixel            = launch_index.y * launch_dims.x + launch_index.x;
    output_buffer[pixel] = C3_TO_F3(result);
}

//------------------------------------------------------------------------------

// We need to pull in the definition of SimpleRaytracer::subpixel_radiance(),
// which is shared between the host and CUDA renderers.
#include "../simpleraytracer.cpp"

//------------------------------------------------------------------------------
