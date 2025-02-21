// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <optix.h>
#include <optix_device.h>

#include <OSL/hashes.h>

#include "optix_raytracer.h"
#include "rend_lib.h"

#include "../background.h"
#include "../bvh.h"
#include "../raytracer.h"
#include "../render_params.h"
#include "../sampling.h"
#include "../shading.cpp"

#include <cstdint>


// Conversion macros for casting between vector types
#define F3_TO_V3(f3) (*reinterpret_cast<const OSL::Vec3*>(&f3))
#define F3_TO_C3(f3) (*reinterpret_cast<const OSL::Color3*>(&f3))
#define V3_TO_F3(v3) (*reinterpret_cast<const float3*>(&v3))
#define C3_TO_F3(c3) (*reinterpret_cast<const float3*>(&c3))


OSL_NAMESPACE_BEGIN
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
OSL_NAMESPACE_END


extern "C" {
__device__ __constant__ RenderParams render_params;
}


static inline __device__ void
execute_shader(ShaderGlobalsType& sg, const int shader_id,
               StackClosurePool& closure_pool)
{
    if (shader_id < 0) {
        // TODO: should probably never get here ...
        return;
    }

    closure_pool.reset();
    RenderState renderState;
    // TODO: renderState.context = ...
    renderState.closure_pool = &closure_pool;
    sg.renderstate           = &renderState;

    // Pack the pointers to the options structs in a faux "context",
    // which is a rough stand-in for the host ShadingContext.
    ShadingContextCUDA shading_context;
    sg.context = &shading_context;

    // Run the OSL callable
    void* interactive_ptr = reinterpret_cast<void**>(
        render_params.interactive_params)[shader_id];
    const unsigned int shaderIdx = shader_id + 0u;
    optixDirectCall<void, ShaderGlobalsType*, void*, void*, void*, int, void*>(
        shaderIdx, &sg /*shaderglobals_ptr*/, nullptr /*groupdata_ptr*/,
        nullptr /*userdata_base_ptr*/, nullptr /*output_base_ptr*/,
        0 /*shadeindex - unused*/, interactive_ptr /*interactive_params_ptr*/
    );
}


static inline __device__ void
trace_ray(OptixTraversableHandle handle, Payload& payload, const float3& origin,
          const float3& direction, const float tmin)
{
    uint32_t p0 = static_cast<uint32_t>(payload.hit_id);
    uint32_t p1 = __float_as_uint(payload.hit_t);
    uint32_t p2 = __float_as_uint(payload.hit_u);
    uint32_t p3 = __float_as_uint(payload.hit_v);
    optixTrace(handle,                         // handle
               origin,                         // origin
               direction,                      // direction
               tmin,                           // tmin
               1e13f,                          // tmax
               0,                              // ray time
               OptixVisibilityMask(1),         // visibility mask
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // ray flags
               0,                              // SBT offset
               1,                              // SBT stride
               0,                              // miss SBT offset
               p0, p1, p2, p3);
    payload.hit_id = static_cast<int32_t>(p0);
    payload.hit_t  = __uint_as_float(p1);
    payload.hit_u  = __uint_as_float(p2);
    payload.hit_v  = __uint_as_float(p3);
};


Intersection
Scene::intersect(const Ray& r, const float tmax, const unsigned skipID1,
                 const unsigned /*skipID2*/) const
{
    // Trace the ray against the scene. If the ID for the hit matches skipID1,
    // "nudge" the ray forward by adjusting tmin to exclude the hit interval
    // and try again.
    const int num_attempts = 2;
    float tmin             = 0.0f;
    for (int attempt = 0; attempt < num_attempts; ++attempt) {
        Payload payload;
        payload.hit_id = ~0u;
        trace_ray(handle, payload, V3_TO_F3(r.origin), V3_TO_F3(r.direction),
                  tmin);
        if (payload.hit_id == skipID1) {
            tmin = __uint_as_float(__float_as_uint(payload.hit_t) + 1u);
        } else if (payload.hit_id != ~0u) {
            return { payload.hit_t, payload.hit_u, payload.hit_v,
                     payload.hit_id };
        }
    }
    return { std::numeric_limits<float>::infinity() };
}


static inline __device__ void
setupRaytracer(SimpleRaytracer& raytracer, const bool bg_only)
{
    // Background
    raytracer.background = {};
    raytracer.background.set_variables((Vec3*)render_params.bg_values,
                                       (float*)render_params.bg_rows,
                                       (float*)render_params.bg_cols,
                                       render_params.bg_res);

    raytracer.backgroundResolution = render_params.bg_id >= 0
                                         ? render_params.bg_res
                                         : 0;
    raytracer.backgroundShaderID   = render_params.bg_id;

    if (bg_only)
        return;

    // Parameters
    raytracer.aa                = render_params.aa;
    raytracer.no_jitter         = render_params.no_jitter;
    raytracer.max_bounces       = render_params.max_bounces;
    raytracer.rr_depth          = 5;
    raytracer.show_albedo_scale = render_params.show_albedo_scale;
    raytracer.show_globals      = render_params.show_globals;

    // Pointers
    raytracer.lightprims_size = render_params.lightprims_size;
    raytracer.m_lightprims    = reinterpret_cast<unsigned int*>(
        render_params.lightprims);
    raytracer.m_mesh_surfacearea = reinterpret_cast<const float*>(
        render_params.surfacearea);
    raytracer.m_meshids = reinterpret_cast<const int*>(render_params.mesh_ids);
    raytracer.m_shader_is_light = reinterpret_cast<int*>(
        render_params.shader_is_light);

    // Scene
    raytracer.scene       = {};
    raytracer.scene.verts = reinterpret_cast<const OSL::Vec3*>(
        render_params.verts);
    raytracer.scene.normals = reinterpret_cast<const OSL::Vec3*>(
        render_params.normals);
    raytracer.scene.uvs = reinterpret_cast<const OSL::Vec2*>(render_params.uvs);
    raytracer.scene.triangles = reinterpret_cast<const TriangleIndices*>(
        render_params.triangles);
    raytracer.scene.uv_triangles = reinterpret_cast<const TriangleIndices*>(
        render_params.uv_indices);
    raytracer.scene.n_triangles = reinterpret_cast<const TriangleIndices*>(
        render_params.normal_indices);
    raytracer.scene.shaderids = reinterpret_cast<int*>(
        render_params.shader_ids);
    raytracer.scene.handle = render_params.traversal_handle;

    // Camera
    const Vec3 eye          = F3_TO_V3(render_params.eye);
    const Vec3 dir          = F3_TO_V3(render_params.dir);
    const Vec3 up           = F3_TO_V3(render_params.up);
    const float fov         = render_params.fov;
    const uint3 launch_dims = optixGetLaunchDimensions();
    raytracer.camera.resolution(launch_dims.x, launch_dims.y);
    raytracer.camera.lookat(eye, dir, up, fov);
    raytracer.camera.finalize();
}

//------------------------------------------------------------------------------

// Because clang++ 9.0 seems to have trouble with some of the texturing "intrinsics"
// let's do the texture look-ups in this file.
extern "C" __device__ float4
osl_tex2DLookup(void* handle, float s, float t, float dsdx, float dtdx,
                float dsdy, float dtdy)
{
    const float2 dx           = { dsdx, dtdx };
    const float2 dy           = { dsdy, dtdy };
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

    if (render_params.bg_id < 0)
        return;

    SimpleRaytracer raytracer;
    setupRaytracer(raytracer, /*bg_only=*/true);

    auto evaler = [&](const Dual2<Vec3>& dir) {
        return raytracer.eval_background(dir, nullptr);
    };

    // Background::prepare_cuda must run on a single warp
    assert(launch_index.x < 32 && launch_index.y == 0);
    raytracer.background.prepare_cuda(launch_dims.x, launch_index.x, evaler);
}


extern "C" __global__ void
__miss__setglobals()
{
}


extern "C" __global__ void
__closesthit__deferred()
{
    const unsigned int hit_idx = optixGetPrimitiveIndex();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float3 ray_origin    = optixGetWorldRayOrigin();
    const float hit_t          = optixGetRayTmax();
    const float2 barycentrics  = optixGetTriangleBarycentrics();
    const float b1             = barycentrics.x;
    const float b2             = barycentrics.y;

    Payload payload;
    payload.hit_t  = hit_t;
    payload.hit_u  = b1;
    payload.hit_v  = b2;
    payload.hit_id = hit_idx;
    payload.set();
}


extern "C" __global__ void
__raygen__deferred()
{
    SimpleRaytracer raytracer;
    setupRaytracer(raytracer, /*bg_only=*/false);

    const uint3 launch_index = optixGetLaunchIndex();
    Color3 result = raytracer.antialias_pixel(launch_index.x, launch_index.y,
                                              nullptr);

    // Write the output
    {
        uint3 launch_dims     = optixGetLaunchDimensions();
        uint3 launch_index    = optixGetLaunchIndex();
        float3* output_buffer = reinterpret_cast<float3*>(
            render_params.output_buffer);
        int pixel            = launch_index.y * launch_dims.x + launch_index.x;
        output_buffer[pixel] = C3_TO_F3(result);
    }
}

//------------------------------------------------------------------------------

// We need to pull in the definition of SimpleRaytracer::subpixel_radiance(),
// which is shared between the host and CUDA renderers.
#include "../simpleraytracer.cpp"

//------------------------------------------------------------------------------
