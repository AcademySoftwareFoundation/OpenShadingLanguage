// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <optix.h>

#include "util.h"

#include <optix_device.h>

#include <OSL/hashes.h>

#include "optix_raytracer.h"
#include "rend_lib.h"
#include "render_params.h"
#include "vec_math.h"

#include "../render_params.h"
#include "../sampling.h"
#include "../shading.h"
#include "../shading.cpp"

#include <cstdint>

using OSL_CUDA::ShaderGlobals;


// Conversion macros for casting between vector types
#define F3_TO_V3(f3) (*reinterpret_cast<const Vec3*>(&f3))
#define F3_TO_C3(f3) (*reinterpret_cast<const Color3*>(&f3))
#define V3_TO_F3(v3) (*reinterpret_cast<const float3*>(&v3))
#define C3_TO_F3(c3) (*reinterpret_cast<const float3*>(&c3))


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
__miss__occlusion()
{
    // printf("__miss__occlusion\n");
}


extern "C" __global__ void
__raygen__setglobals()
{
    // Set global variables
    OSL::pvt::osl_printf_buffer_start = render_params.osl_printf_buffer_start;
    OSL::pvt::osl_printf_buffer_end   = render_params.osl_printf_buffer_end;
    OSL::pvt::s_color_system          = render_params.color_system;
    OSL::pvt::test_str_1              = render_params.test_str_1;
    OSL::pvt::test_str_2              = render_params.test_str_2;
}


extern "C" __global__ void
__miss__setglobals()
{
}


extern "C" __global__ void
__raygen__()
{
    uint3 launch_dims  = optixGetLaunchDimensions();
    uint3 launch_index = optixGetLaunchIndex();
    const float3 eye   = render_params.eye;
    const float3 dir   = render_params.dir;
    const float3 cx    = render_params.cx;
    const float3 cy    = render_params.cy;
    const float invw   = render_params.invw;
    const float invh   = render_params.invh;

    // Compute the pixel coordinates
    const float2 d = make_float2(static_cast<float>(launch_index.x) + 0.5f,
                                 static_cast<float>(launch_index.y) + 0.5f);

    // Make the ray for the current pixel
    RayGeometry r;
    r.origin    = eye;
    r.direction = normalize(cx * (d.x * invw - 0.5f) + cy * (0.5f - d.y * invh)
                            + dir);
    optixTrace(render_params.traversal_handle, r.origin, r.direction, 1e-3f,
               1e13f, 0, OptixVisibilityMask(1), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               0, 1, 0);
}


// Because clang++ 9.0 seems to have trouble with some of the texturing "intrinsics"
// let's do the texture look-ups in this file.
extern "C" __device__ float4
osl_tex2DLookup(void* handle, float s, float t)
{
    cudaTextureObject_t texID = cudaTextureObject_t(handle);
    return tex2D<float4>(texID, s, t);
}

//--------------------------------------------------------------------------------
//
// Pathtracing
//
//--------------------------------------------------------------------------------

struct t_ab {
    uint32_t a, b;
};


struct t_ptr {
    uint64_t ptr;
};


struct Payload {
    union {
        t_ab  ab;
        t_ptr ptr;
    };

    __forceinline__ __device__ void set()
    {
        optixSetPayload_0( ab.a );
        optixSetPayload_1( ab.b );
    }

    __forceinline__ __device__ void get()
    {
        ab.a = optixGetPayload_0();
        ab.b = optixGetPayload_1();
    }
};


inline __device__
float3 cross(const float3& a, const float3& b)
{
  return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}


inline __device__
void ortho(const float3& n, float3& x, float3& y)
{
    x = normalize(fabsf(n.x) > .01f ? make_float3(n.z, 0, -n.x) : make_float3(0, -n.z, n.y));
    y = cross(n, x);
}


// return a direction towards a point on the sphere
static __device__
float3 sample_sphere(const float3& x, const SphereParams& sphere,
                     float xi, float yi, float& pdf)
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


// return a direction towards a point on the quad
static __device__
float3 sample_quad(const float3& x, const QuadParams& quad,
                   float xi, float yi, float& pdf)
{
    float3 l   = (quad.p + xi * quad.ex + yi * quad.ey) - x;
    float  d2  = dot(l, l); // l.length2();
    float3 dir = normalize(l);
    pdf        = d2 / (quad.a * fabsf(dot(dir, quad.n)));
    return dir;
}


static __device__ void
globals_from_hit(OSL_CUDA::ShaderGlobals& sg)
{
    // const GenericRecord* record = reinterpret_cast<GenericRecord*>(
    //     optixGetSbtDataPointer());

    OSL_CUDA::ShaderGlobals local_sg;
    // hit-kind 0: quad hit
    //          1: sphere hit
    optixDirectCall<void, unsigned int, float, float3, float3, OSL_CUDA::ShaderGlobals*>(
        optixGetHitKind(), optixGetPrimitiveIndex(), optixGetRayTmax(),
        optixGetWorldRayOrigin(), optixGetWorldRayDirection(), &local_sg);
    // Setup the ShaderGlobals
    const float3 ray_direction = optixGetWorldRayDirection();
    const float3 ray_origin    = optixGetWorldRayOrigin();
    const float t_hit          = optixGetRayTmax();

    sg.I  = ray_direction;
    sg.N  = normalize(optixTransformNormalFromObjectToWorldSpace(local_sg.N));
    sg.Ng = normalize(optixTransformNormalFromObjectToWorldSpace(local_sg.Ng));
    sg.P  = ray_origin + t_hit * ray_direction;
    sg.dPdu        = local_sg.dPdu;
    sg.dPdv        = local_sg.dPdv;
    sg.u           = local_sg.u;
    sg.v           = local_sg.v;
    sg.Ci          = NULL;
    sg.surfacearea = local_sg.surfacearea;
    sg.backfacing  = dot(sg.N, sg.I) > 0.0f;
    sg.shaderID    = local_sg.shaderID;

    if (sg.backfacing) {
        sg.N  = -sg.N;
        sg.Ng = -sg.Ng;
    }

    // NB: These variables are not used in the current iteration of the sample
    sg.raytype        = CAMERA;
    sg.flipHandedness = 0;
}


static __device__ float3
process_closure(const OSL::ClosureColor* closure_tree, ShadingResult& result)
{
    OSL::Color3 color_result = OSL::Color3(0.0f);

    if (!closure_tree) {
        return C3_TO_F3(color_result);
    }

    // The depth of the closure tree must not exceed the stack size.
    // A stack size of 8 is probably quite generous for relatively
    // balanced trees.
    const int STACK_SIZE = 8;

    // Non-recursive traversal stack
    int stack_idx = 0;
    const OSL::ClosureColor* ptr_stack[STACK_SIZE];
    OSL::Color3 weight_stack[STACK_SIZE];

    // Shading accumulator
    OSL::Color3 weight = OSL::Color3(1.0f);
    const void* cur    = closure_tree;
    while (cur) {
        ClosureIDs id = static_cast<ClosureIDs>(
            ((OSL::ClosureColor*)cur)->id);

        switch (id) {
        case ClosureIDs::ADD: {
            ptr_stack[stack_idx]      = ((OSL::ClosureAdd*)cur)->closureB;
            weight_stack[stack_idx++] = weight;
            cur                       = ((OSL::ClosureAdd*)cur)->closureA;
            break;
        }
        case ClosureIDs::MUL: {
            weight *= ((OSL::ClosureMul*)cur)->weight;
            cur = ((OSL::ClosureMul*)cur)->closure;
            break;
        }
        default: {
            const ClosureComponent* comp = reinterpret_cast<const ClosureColor*>(cur)->as_comp();
            Color3 cw                    = weight * comp->w;
            switch (id) {
            case ClosureIDs::EMISSION_ID: {
                result.Le += ((OSL::ClosureComponent*)cur)->w * weight;;
                cur = NULL;
                break;
            }
            case ClosureIDs::MICROFACET_ID:
            case ClosureIDs::DIFFUSE_ID:
            case ClosureIDs::OREN_NAYAR_ID:
            case ClosureIDs::PHONG_ID:
            case ClosureIDs::WARD_ID:
            case ClosureIDs::REFLECTION_ID:
            case ClosureIDs::REFRACTION_ID:
            case ClosureIDs::FRESNEL_REFLECTION_ID: {
                if (!result.bsdf.add_bsdf_gpu(cw, comp))
                    printf("unable to add BSDF\n");
                cur = nullptr;
                break;
            }

            default: cur = NULL; break;
            }
        }
        }
        if (cur == NULL && stack_idx > 0) {
            cur    = ptr_stack[--stack_idx];
            weight = weight_stack[stack_idx];
        }
    }

    return C3_TO_F3(color_result);
}


static __device__ void
process_closure(const ShaderGlobalsType& sg, ShadingResult& result,
                const void* Ci, bool light_only)
{
    // TODO: GPU media?
    // if (!light_only)
    //     process_medium_closure(sg, result, Ci, Color3(1));
    process_closure((const ClosureColor*)Ci, result);
}


extern "C" __global__ void
__closesthit__deferred()
{
    Payload payload;
    payload.get();
    OSL_CUDA::ShaderGlobals* sg_ptr = (OSL_CUDA::ShaderGlobals*) payload.ptr.ptr;
    globals_from_hit(*sg_ptr);
    uint32_t* trace_data = (uint32_t*) sg_ptr->tracedata;
    trace_data[0] = optixGetPrimitiveIndex();
    trace_data[1] = optixGetHitKind();
}


extern "C" __global__ void
__closesthit__occlusion()
{
    Payload payload;
    payload.get();
    uint32_t* vals_ptr = (uint32_t*) payload.ptr.ptr;

    vals_ptr[0] = optixGetPrimitiveIndex();
    vals_ptr[1] = optixGetHitKind();
}


static inline __device__ Color3 subpixel_radiance(float2 d, Sampler& sampler);


extern "C" __global__ void
__raygen__deferred()
{
    Color3 result(0, 0, 0);
    const int aa = render_params.aa;
    for (int si = 0, n = aa * aa; si < n; si++) {
        // uint3 launch_dims  = optixGetLaunchDimensions();
        uint3 launch_index = optixGetLaunchIndex();

        Sampler sampler(launch_index.x, launch_index.y, si);

        Vec3 j = sampler.get();
        // warp distribution to approximate a tent filter [-1,+1)^2
        j.x *= 2;
        j.x = j.x < 1 ? sqrtf(j.x) - 1 : 1 - sqrtf(2 - j.x);
        j.y *= 2;
        j.y = j.y < 1 ? sqrtf(j.y) - 1 : 1 - sqrtf(2 - j.y);

        // Compute the pixel coordinates
        const float2 d
            = make_float2(static_cast<float>(launch_index.x) + 0.5f + j.x,
                          static_cast<float>(launch_index.y) + 0.5f + j.y);

        Color3 r = subpixel_radiance(d, sampler);
        result   = OIIO::lerp(result, r, 1.0f / (si + 1));
    }

    uint3 launch_dims     = optixGetLaunchDimensions();
    uint3 launch_index    = optixGetLaunchIndex();
    float3* output_buffer = reinterpret_cast<float3*>(
        render_params.output_buffer);
    int pixel            = launch_index.y * launch_dims.x + launch_index.x;
    output_buffer[pixel] = C3_TO_F3(result);
}


static inline __device__ Color3 subpixel_radiance(float2 d, Sampler& sampler)
{
    uint3 launch_dims  = optixGetLaunchDimensions();
    uint3 launch_index = optixGetLaunchIndex();
    const float3 eye   = render_params.eye;
    const float3 dir   = render_params.dir;
    const float3 cx    = render_params.cx;
    const float3 cy    = render_params.cy;
    const float invw   = render_params.invw;
    const float invh   = render_params.invh;

    //
    // Generate camera ray
    //

    // Make the ray for the current pixel
    RayGeometry r;
    r.origin    = eye;
    r.direction = normalize(cx * (d.x * invw - 0.5f) + cy * (0.5f - d.y * invh)
                            + dir);

    Color3 path_weight(1, 1, 1);
    Color3 path_radiance(0, 0, 0);
    float bsdf_pdf = std::numeric_limits<
        float>::infinity();  // camera ray has only one possible direction

    // TODO: How many bounces is reasonable?
    int max_bounces = 10;
    for (int bounce = 0; bounce <= max_bounces; bounce++) {
        const bool last_bounce = bounce == max_bounces;

        //
        // Trace camera/bounce ray
        //

        ShaderGlobalsType sg;
        sg.shaderID = -1;

        Payload payload;
        payload.ptr.ptr = (uint64_t)&sg;

        uint32_t trace_data[2] = { UINT32_MAX, UINT32_MAX };
        sg.tracedata           = (void*)&trace_data[0];

        // Trace the camera ray against the scene
        optixTrace(render_params.traversal_handle, // handle
                   r.origin,                       // origin
                   r.direction,                    // direction
                   1e-3f,                          // tmin
                   1e13f,                          // tmax
                   0,                              // ray time
                   OptixVisibilityMask(1),         // visibility mask
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // ray flags
                   RAY_TYPE_RADIANCE,              // SBT offset
                   RAY_TYPE_COUNT,                 // SBT stride
                   RAY_TYPE_RADIANCE,              // miss SBT offset
                   payload.ab.a,
                   payload.ab.b
            );

        const uint32_t hit_idx  = trace_data[0];
        const uint32_t hit_kind = trace_data[1];

        //
        // Execute the shader
        //

        auto execute_shader = [](OSL_CUDA::ShaderGlobals& sg) {
            if(sg.shaderID < 0) {
                // TODO: should probably never get here ...
                return;
            }

            alignas(8) char closure_pool[256];

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
            sg.context = &shading_context;

            // Run the OSL callable
            void* interactive_ptr = reinterpret_cast<void**>(
                render_params.interactive_params)[sg.shaderID];
            const unsigned int shaderIdx = 2u + sg.shaderID + 0u;
            optixDirectCall<void, OSL_CUDA::ShaderGlobals*, void*, void*, void*, int, void*>(
                shaderIdx,
                &sg /*shaderglobals_ptr*/,
                nullptr /*groupdata_ptr*/,
                nullptr /*userdata_base_ptr*/,
                nullptr /*output_base_ptr*/,
                0 /*shadeindex - unused*/,
                interactive_ptr /*interactive_params_ptr*/
                );
        };

        ShadingResult result;
        if(sg.shaderID >= 0) {
            execute_shader(sg);
        }
        else {
            // Ray missed
            break;
        }

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
                       ? quads[idx - render_params.num_spheres].isLight
                       : spheres[idx].isLight;
        };

        auto shape_pdf = [&](unsigned int idx, unsigned int hit_kind,
                             const Vec3& x, const Vec3& p) {
            QuadParams* quads     = (QuadParams*)render_params.quads_buffer;
            SphereParams* spheres = (SphereParams*)render_params.spheres_buffer;
            return (hit_kind == 0)
                       ? quads[idx - render_params.num_spheres].shapepdf(x, p)
                       : spheres[idx].shapepdf(x, p);
        };

        //
        // Add self-emission
        //

        float k = 1;
        if (is_light(hit_idx, hit_kind)) {
            // figure out the probability of reaching this point
            float light_pdf = shape_pdf(hit_idx, hit_kind, F3_TO_C3(r.origin), F3_TO_C3(sg.P));
            k = MIS::power_heuristic<MIS::WEIGHT_EVAL>(bsdf_pdf, light_pdf);
        }
        path_radiance += path_weight * k * result.Le;

        if (last_bounce)
            break;

        //
        // Build PDF
        //

        result.bsdf.prepare_gpu(-F3_TO_C3(sg.I), path_weight, last_bounce);

        if (render_params.show_albedo_scale > 0) {
            // Instead of path tracing, just visualize the albedo
            // of the bsdf. This can be used to validate the accuracy of
            // the get_albedo method for a particular bsdf.
            path_radiance = path_weight
                             * result.bsdf.get_albedo_gpu(-F3_TO_V3(sg.I))
                             * render_params.show_albedo_scale;
            break;
        }

        // get three random numbers
        Vec3 s   = sampler.get();
        float xi = s.x;
        float yi = s.y;
        float zi = s.z;

        //
        // TODO: Trace background ray
        //

        //
        // Trace light rays
        //

        // Trace one ray to each light
        const size_t num_prims = render_params.num_quads + render_params.num_spheres;
        for (size_t idx = 0; idx < num_prims; ++idx) {
            QuadParams* quads     = (QuadParams*)render_params.quads_buffer;
            SphereParams* spheres = (SphereParams*)render_params.spheres_buffer;
            const int prim_kind   = idx >= render_params.num_quads;

            if (is_light(idx, prim_kind)) {
                float light_pdf        = 0.0f;
                const float3 light_dir = (prim_kind == 0)
                    ? sample_quad(sg.P, quads[idx], xi, yi, light_pdf)
                    : sample_sphere(sg.P, spheres[idx], xi, yi, light_pdf);

                const float3 origin = sg.P + sg.N * 1e-6f;  // offset the ray origin
                BSDF::Sample b      = result.bsdf.eval_gpu(-F3_TO_V3(sg.I), F3_TO_V3(light_dir));
                Color3 contrib
                    = path_weight * b.weight
                      * MIS::power_heuristic<MIS::EVAL_WEIGHT>(light_pdf,
                                                               b.pdf);

                if ((contrib.x + contrib.y + contrib.z) > 0) {
                    ShaderGlobalsType light_sg;
                    uint32_t trace_data[2] = { UINT32_MAX, UINT32_MAX };
                    light_sg.shaderID      = -1;
                    light_sg.tracedata     = (void*)&trace_data[0];

                    Payload payload;
                    payload.ptr.ptr = (uint64_t)&light_sg;

                    // Trace the camera ray against the scene
                    optixTrace(render_params.traversal_handle, // handle
                               origin,                         // origin
                               light_dir,                      // direction
                               1e-3f,                          // tmin
                               1e13f,                          // tmax
                               0,                              // ray time
                               OptixVisibilityMask(1),         // visibility mask
                               OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // ray flags
                               RAY_TYPE_RADIANCE,              // SBT offset
                               RAY_TYPE_COUNT,                 // SBT stride
                               RAY_TYPE_RADIANCE,              // miss SBT offset
                               payload.ab.a,
                               payload.ab.b
                        );

                    // TODO: Make sure that the primitive indexing is correct
                    const uint32_t prim_idx = trace_data[0];
                    if (prim_idx == idx && light_sg.shaderID >= 0) {
                        // execute the light shader (for emissive closures only)
                        execute_shader(light_sg);

                        ShadingResult light_result;
                        process_closure(light_sg, light_result,
                                        (void*)light_sg.Ci, true);

                        // accumulate contribution
                        path_radiance += contrib * light_result.Le;
                    }
                }
            }
        }

        //
        // Setup bounce ray
        //

        BSDF::Sample p = result.bsdf.sample_gpu(-F3_TO_V3(sg.I), xi, yi, zi);
        path_weight *= p.weight;
        bsdf_pdf = p.pdf;
        // r.raytype = Ray::DIFFUSE;  // FIXME? Use DIFFUSE for all indiirect rays
        r.direction = C3_TO_F3(p.wi);
        // r.radius    = radius;
        // Just simply use roughness as spread slope
        // r.spread = std::max(r.spread, p.roughness);
        if (!(path_weight.x > 0) && !(path_weight.y > 0)
            && !(path_weight.z > 0))
            break;  // filter out all 0's or NaNs
        // prev_id  = id;
        r.origin = V3_TO_F3(sg.P) + sg.N * 1e-6f;
    }

    return path_radiance;
}
