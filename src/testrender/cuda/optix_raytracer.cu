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
#include "../shading_cuda.cpp"
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


static __device__ Color3
eval_background(const Dual2<Vec3>& dir, int bounce = -1)
{
    ShaderGlobalsType sg;
    memset((char*)&sg, 0, sizeof(ShaderGlobalsType));
    sg.I    = V3_TO_F3(dir.val());
    sg.dIdx = V3_TO_F3(dir.dx());
    sg.dIdy = V3_TO_F3(dir.dy());
    if (bounce >= 0)
        sg.raytype = bounce > 0 ? Ray::DIFFUSE : Ray::CAMERA;
    sg.shaderID = render_params.bg_id;

    alignas(8) char closure_pool[256];
    execute_shader(sg, closure_pool);
    return process_background_closure(sg, (const ClosureColor*)sg.Ci);
}


// Return a direction towards a point on the sphere
static __device__ float3
sample_sphere(const float3& x, const SphereParams& sphere, float xi, float yi,
              float& pdf)
{
    const float TWOPI = float(2 * M_PI);
    float cmax2       = 1 - sphere.r2 / dot(sphere.c - x, sphere.c - x);
    float cmax        = cmax2 > 0 ? sqrtf(cmax2) : 0;
    float cos_a       = 1 - xi + xi * cmax;
    float sin_a       = sqrtf(1 - cos_a * cos_a);
    float phi         = TWOPI * yi;
    float sp, cp;
    OIIO::fast_sincos(phi, &sp, &cp);
    float3 sw = normalize(sphere.c - x), su, sv;
    ortho(sw, su, sv);
    pdf = 1 / (TWOPI * (1 - cmax));
    return normalize(su * (cp * sin_a) + sv * (sp * sin_a) + sw * cos_a);
}


// Return a direction towards a point on the quad
static __device__ float3
sample_quad(const float3& x, const QuadParams& quad, float xi, float yi,
            float& pdf)
{
    float3 l   = (quad.p + xi * quad.ex + yi * quad.ey) - x;
    float  d2  = dot(l, l); // l.length2();
    float3 dir = normalize(l);
    pdf        = d2 / (quad.a * fabsf(dot(dir, quad.n)));
    return dir;
}


static inline __device__ Color3
subpixel_radiance(Ray r, Sampler& sampler, Background& background)
{
    Color3 path_weight(1, 1, 1);
    Color3 path_radiance(0, 0, 0);
    float bsdf_pdf = std::numeric_limits<
        float>::infinity();  // camera ray has only one possible direction

    // TODO: Should these be separate?
    alignas(8) char closure_pool[256];
    alignas(8) char light_closure_pool[256];

    int max_bounces = render_params.max_bounces;
    int rr_depth    = 5;

    int prev_hit_idx  = -1;
    int prev_hit_kind = -1;

    for (int bounce = 0; bounce <= max_bounces; bounce++) {
        const bool last_bounce = bounce == max_bounces;
        int hit_idx  = prev_hit_idx;
        int hit_kind = prev_hit_kind;

        //
        // Trace camera/bounce ray
        //

        ShaderGlobalsType sg;
        sg.shaderID = -1;

        uint32_t trace_data[4] = { UINT32_MAX, UINT32_MAX,
                                   *(unsigned int*)&hit_idx,
                                   *(unsigned int*)&hit_kind };
        sg.tracedata           = (void*)&trace_data[0];

        Payload payload;
        payload.sg_ptr = (uint64_t)&sg;

        uint32_t p0 = payload.raw[0];
        uint32_t p1 = payload.raw[1];
        uint32_t p2 = __float_as_uint(r.radius);
        uint32_t p3 = __float_as_uint(r.spread);
        uint32_t p4 = r.raytype;

        // Trace the camera ray against the scene
        optixTrace(render_params.traversal_handle, // handle
                   V3_TO_F3(r.origin),             // origin
                   V3_TO_F3(r.direction),          // direction
                   1e-3f,                          // tmin
                   1e13f,                          // tmax
                   0,                              // ray time
                   OptixVisibilityMask(1),         // visibility mask
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // ray flags
                   0,                              // SBT offset
                   1,                              // SBT stride
                   0,                              // miss SBT offset
                   p0,
                   p1,
                   p2,
                   p3,
                   p4
            );

        //
        // Execute the shader
        //

        ShadingResult result;
        if (sg.shaderID >= 0) {
            hit_idx  = trace_data[0];
            hit_kind = trace_data[1];
            execute_shader(sg, closure_pool);
        } else {
            // ray missed
            if (render_params.bg_id >= 0) {
                if (bounce > 0 && render_params.bg_res > 0) {
                    float bg_pdf = 0;
                    Vec3 bg      = background.eval(r.direction, bg_pdf);
                    path_radiance
                        += path_weight * bg
                           * MIS::power_heuristic<MIS::WEIGHT_WEIGHT>(bsdf_pdf,
                                                                      bg_pdf);
                } else {
                    // we aren't importance sampling the background - so just run it directly
                    path_radiance += path_weight * eval_background(r.direction, bounce);
                }
            }
            break;
        }

        const float t      = ((float*)trace_data)[2];
        const float radius = r.radius + r.spread * t;

        //
        // Process the output closure
        //

        process_closure(sg, result, (void*) sg.Ci, last_bounce);

        //
        // Helpers
        //

        auto is_light = [&](unsigned int idx, unsigned int hit_kind) {
            QuadParams* quads     = (QuadParams*)render_params.quads_buffer;
            SphereParams* spheres = (SphereParams*)render_params.spheres_buffer;

            return (hit_kind == 0)
                       ? quads[idx].isLight
                       : spheres[idx].isLight;
        };

        auto shape_pdf = [&](unsigned int idx, unsigned int hit_kind,
                             const Vec3& x, const Vec3& p) {
            QuadParams* quads     = (QuadParams*)render_params.quads_buffer;
            SphereParams* spheres = (SphereParams*)render_params.spheres_buffer;
            return (hit_kind == 0)
                       ? quads[idx].shapepdf(x, p)
                       : spheres[idx].shapepdf(x, p);
        };

        //
        // Add self-emission
        //

        float k = 1;
        if (is_light(hit_idx, hit_kind)) {
            // figure out the probability of reaching this point
            float light_pdf = shape_pdf(hit_idx, hit_kind, r.origin, F3_TO_C3(sg.P));
            k = MIS::power_heuristic<MIS::WEIGHT_EVAL>(bsdf_pdf, light_pdf);
        }
        path_radiance += path_weight * k * result.Le;

        if (last_bounce) {
            break;
        }

        //
        // Build PDF
        //

        result.bsdf.prepare(-F3_TO_C3(sg.I), path_weight, bounce >= rr_depth);

        if (render_params.show_albedo_scale > 0) {
            // Instead of path tracing, just visualize the albedo
            // of the bsdf. This can be used to validate the accuracy of
            // the get_albedo method for a particular bsdf.
            path_radiance = path_weight
                             * result.bsdf.get_albedo(-F3_TO_V3(sg.I))
                             * render_params.show_albedo_scale;
            break;
        }

        // get three random numbers
        Vec3 s   = sampler.get();
        float xi = std::max(0.0f, std::min(1.0f, s.x));
        float yi = std::max(0.0f, std::min(1.0f, s.y));
        float zi = std::max(0.0f, std::min(1.0f, s.z));

        //
        // Trace background ray
        //

        // trace one ray to the background
        if (render_params.bg_id >= 0) {
            const float3 origin = sg.P;
            Dual2<Vec3> bg_dir;
            float bg_pdf   = 0;
            Vec3 bg        = background.sample(xi, yi, bg_dir, bg_pdf);
            BSDF::Sample b = result.bsdf.eval(-F3_TO_V3(sg.I), bg_dir.val());
            Color3 contrib = path_weight * b.weight * bg
                * MIS::power_heuristic<MIS::WEIGHT_WEIGHT>(bg_pdf,
                                                           b.pdf);

            if ((contrib.x + contrib.y + contrib.z) > 0) {
                ShaderGlobalsType light_sg;
                light_sg.shaderID = -1;

                Payload payload;
                payload.sg_ptr = (uint64_t)&light_sg;

                uint32_t trace_data[4] = { UINT32_MAX, UINT32_MAX,
                    *(unsigned int*)&hit_idx,
                    *(unsigned int*)&hit_kind };
                light_sg.tracedata           = (void*)&trace_data[0];

                const float3 dir = V3_TO_F3(bg_dir.val());

                // Trace the camera ray against the scene
                optixTrace(render_params.traversal_handle,  // handle
                           origin,                          // origin
                           dir,                             // direction
                           1e-3f,                           // tmin
                           1e13f,                           // tmax
                           0,                               // ray time
                           OptixVisibilityMask(1),          // visibility mask
                           OPTIX_RAY_FLAG_DISABLE_ANYHIT,   // ray flags
                           0,                               // SBT offset
                           1,                               // SBT stride
                           0,                               // miss SBT offset
                           payload.raw[0],
                           payload.raw[1]);

                const uint32_t prim_idx = trace_data[0];
                if (prim_idx == UINT32_MAX) {
                    path_radiance += contrib;
                }
            }
        }

        //
        // Trace light rays
        //

        auto sample_light = [&](const float3& light_dir, float light_pdf, int idx) {
            const float3 origin = sg.P;
            BSDF::Sample b      = result.bsdf.eval(-F3_TO_V3(sg.I),
                                                   F3_TO_V3(light_dir));
            Color3 contrib      = path_weight * b.weight
                             * MIS::power_heuristic<MIS::EVAL_WEIGHT>(light_pdf,
                                                                      b.pdf);

            if ((contrib.x + contrib.y + contrib.z) > 0) {
                ShaderGlobalsType light_sg;
                light_sg.shaderID = -1;

                Payload payload;
                payload.sg_ptr = (uint64_t)&light_sg;

                uint32_t trace_data[4] = { UINT32_MAX, UINT32_MAX,
                    *(unsigned int*)&hit_idx,
                    *(unsigned int*)&hit_kind };
                light_sg.tracedata           = (void*)&trace_data[0];

                uint32_t p2 = __float_as_uint(r.radius);
                uint32_t p3 = __float_as_uint(r.spread);

                // Trace the camera ray against the scene
                optixTrace(render_params.traversal_handle,  // handle
                           origin,                          // origin
                           light_dir,                       // direction
                           1e-3f,                           // tmin
                           1e13f,                           // tmax
                           0,                               // ray time
                           OptixVisibilityMask(1),          // visibility mask
                           OPTIX_RAY_FLAG_DISABLE_ANYHIT,   // ray flags
                           0,                               // SBT offset
                           1,                               // SBT stride
                           0,                               // miss SBT offset
                           payload.raw[0],
                           payload.raw[1]);

                // TODO: Make sure that the primitive indexing is correct
                const uint32_t prim_idx = trace_data[0];
                if (prim_idx == idx && light_sg.shaderID >= 0) {
                    // execute the light shader (for emissive closures only)
                    execute_shader(light_sg, light_closure_pool);

                    ShadingResult light_result;
                    process_closure(light_sg, light_result, (void*)light_sg.Ci,
                                    true);

                    // accumulate contribution
                    path_radiance += contrib * light_result.Le;
                }
            }
        };

        // Trace one ray to each quad light
        for (size_t idx = 0; idx < render_params.num_quads; ++idx) {
            QuadParams* quads = (QuadParams*)render_params.quads_buffer;
            if (hit_kind == 0 && hit_idx == idx)
                continue;  // skip self
            if (is_light(idx, 0)) {
                float light_pdf        = 0.0f;
                const float3 light_dir = sample_quad(sg.P, quads[idx], xi, yi,
                                                     light_pdf);
                sample_light(light_dir, light_pdf, idx);
            }
        }

        // Trace one ray to each sphere light
        for (size_t idx = 0; idx < render_params.num_spheres; ++idx) {
            SphereParams* spheres = (SphereParams*)render_params.spheres_buffer;
            if (hit_kind == 1 && hit_idx == idx)
                continue;  // skip self
            if (is_light(idx, 1)) {
                float light_pdf        = 0.0f;
                const float3 light_dir = sample_sphere(sg.P, spheres[idx], xi,
                                                       yi, light_pdf);
                sample_light(light_dir, light_pdf, idx);
            }
        }

        //
        // Setup bounce ray
        //

        BSDF::Sample p = result.bsdf.sample(-F3_TO_V3(sg.I), xi, yi, zi);
        path_weight *= p.weight;
        bsdf_pdf  = p.pdf;
        r.raytype = Ray::DIFFUSE;  // FIXME? Use DIFFUSE for all indiirect rays
        r.direction = p.wi;
        r.radius    = radius;
        // Just simply use roughness as spread slope
        r.spread = std::max(r.spread, p.roughness);
        if (!(path_weight.x > 0) && !(path_weight.y > 0)
            && !(path_weight.z > 0)) {
            break;  // filter out all 0's or NaNs
        }

        // TODO: Keep track of object IDs for self-intersection avoidance, etc.
        prev_hit_idx  = hit_idx;
        prev_hit_kind = hit_kind;
        r.origin = sg.P;
    }

    return path_radiance;
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

    Ray ray(F3_TO_V3(ray_origin), F3_TO_V3(ray_direction), radius, spread,
            Ray::RayType::CAMERA);
    Dual2<float> t(t_hit);
    Dual2<Vec3> P = ray.point(t);

    sg.I  = ray_direction;
    sg.N  = normalize(optixTransformNormalFromObjectToWorldSpace(local_sg.N));
    sg.Ng = normalize(optixTransformNormalFromObjectToWorldSpace(local_sg.Ng));
    sg.P  = V3_TO_F3(P.val());
    sg.dPdx        = V3_TO_F3(P.dx());
    sg.dPdy        = V3_TO_F3(P.dy());
    sg.dPdu        = local_sg.dPdu;
    sg.dPdv        = local_sg.dPdv;
    sg.u           = local_sg.u;
    sg.v           = local_sg.v;
    sg.Ci          = nullptr;
    sg.surfacearea = local_sg.surfacearea;
    sg.backfacing  = dot(sg.N, sg.I) > 0.0f;
    sg.shaderID    = local_sg.shaderID;

    if (sg.backfacing) {
        sg.N  = -sg.N;
        sg.Ng = -sg.Ng;
    }

    sg.raytype        = raytype;
    sg.flipHandedness = dot(sg.N, cross(sg.dPdx, sg.dPdy)) < 0.0f;
}


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

    if (render_params.bg_id < 0)
        return;

    // TODO: Paralellize Background::prepare()
    auto evaler = [](const Dual2<Vec3>& dir) {
        return eval_background(dir);
    };

    // Background::prepare_gpu must run on a single warp
    assert(launch_index.x < 32 && launch_index.y == 0);
    background.prepare_gpu(launch_dims.x, launch_index.x, evaler);
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
    ShaderGlobalsType* sg_ptr = (ShaderGlobalsType*)payload.sg_ptr;
    uint32_t* trace_data      = (uint32_t*)sg_ptr->tracedata;
    const float t_hit         = optixGetRayTmax();
    trace_data[0]             = optixGetPrimitiveIndex();
    trace_data[1]             = optixGetHitKind();
    trace_data[2]             = *(uint32_t*)&t_hit;
    globals_from_hit(*sg_ptr, payload.radius, payload.spread, payload.raytype);
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

        const Vec3& eye  = F3_TO_V3(render_params.eye);
        const Vec3& dir  = F3_TO_V3(render_params.dir);
        const Vec3& cx   = F3_TO_V3(render_params.cx);
        const Vec3& cy   = F3_TO_V3(render_params.cy);
        const float invw = render_params.invw;
        const float invh = render_params.invh;

        // Generate the camera ray
        const Vec3 v = (cx * (d.x * invw - 0.5f) + cy * (0.5f - d.y * invh)
                        + dir)
                           .normalize();
        const float cos_a = dir.dot(v);
        float spread = sqrtf(invw * invh * cx.length() * cy.length() * cos_a)
                       * cos_a;
        Ray ray(eye, v, 0.0f, spread, Ray::RayType::CAMERA);

        Color3 r = subpixel_radiance(ray, sampler, background);
        result   = OIIO::lerp(result, r, 1.0f / (si + 1));
    }

    uint3 launch_dims     = optixGetLaunchDimensions();
    uint3 launch_index    = optixGetLaunchIndex();
    float3* output_buffer = reinterpret_cast<float3*>(
        render_params.output_buffer);
    int pixel            = launch_index.y * launch_dims.x + launch_index.x;
    output_buffer[pixel] = C3_TO_F3(result);
}
