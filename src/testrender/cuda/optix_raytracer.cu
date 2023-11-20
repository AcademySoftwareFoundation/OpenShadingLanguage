// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <optix.h>

#include "util.h"

#include <optix_device.h>

#include <OSL/hashes.h>

#include "rend_lib.h"
#include "render_params.h"

#include "../render_params.h"
#include "../sampling.h"
#include "../shading.h"
#include "../shading.cpp"

using OSL_CUDA::ShaderGlobals;


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

//------------------------------------------------------------------------------
//
// EXPERIMENTAL
//
//------------------------------------------------------------------------------

#if 1

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

    __forceinline__ __device__ void setPtr()
    {
        optixSetPayload_0( ab.a );
        optixSetPayload_1( ab.b );
    }

    __forceinline__ __device__ void getPtr()
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
    const GenericRecord* record = reinterpret_cast<GenericRecord*>(
        optixGetSbtDataPointer());

    OSL_CUDA::ShaderGlobals local_sg;
    // hit-kind 0: quad hit
    //          1: sphere hit
    optixDirectCall<void, unsigned int, float, float3, float3, OSL_CUDA::ShaderGlobals*>(
        optixGetHitKind(), optixGetPrimitiveIndex(), optixGetRayTmax(),
        optixGetWorldRayOrigin(), optixGetWorldRayDirection(), &local_sg);
    // Setup the ShaderGlobals
    const float3 ray_direction = optixGetWorldRayDirection();
    const float3 ray_origin    = optixGetWorldRayOrigin();
    const float t_hit          = optixGetRayTmin();

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
process_closure(const OSL::ClosureColor* closure_tree)
{
    OSL::Color3 result = OSL::Color3(0.0f);

    if (!closure_tree) {
        return make_float3(result.x, result.y, result.z);
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
    const void* cur = closure_tree;
    while (cur) {
        MyClosureIDs id = static_cast<MyClosureIDs>(((OSL::ClosureColor*)cur)->id);
        switch (id) {
        case MyClosureIDs::ADD: {
            ptr_stack[stack_idx]      = ((OSL::ClosureAdd*)cur)->closureB;
            weight_stack[stack_idx++] = weight;
            cur                       = ((OSL::ClosureAdd*)cur)->closureA;
            break;
        }

        case MyClosureIDs::MUL: {
            weight *= ((OSL::ClosureMul*)cur)->weight;
            cur = ((OSL::ClosureMul*)cur)->closure;
            break;
        }

        case MyClosureIDs::EMISSION_ID: {
            cur = NULL;
            break;
        }

        case MyClosureIDs::DIFFUSE_ID:
        case MyClosureIDs::OREN_NAYAR_ID:
        case MyClosureIDs::PHONG_ID:
        case MyClosureIDs::WARD_ID:
        case MyClosureIDs::REFLECTION_ID:
        case MyClosureIDs::REFRACTION_ID:
        case MyClosureIDs::FRESNEL_REFLECTION_ID: {
            result += ((OSL::ClosureComponent*)cur)->w * weight;
            cur = NULL;
            break;
        }

        case MyClosureIDs::MICROFACET_ID: {
            const char* mem = (const char*)((OSL::ClosureComponent*)cur)->data();
            const char* dist_str = *(const char**)&mem[0];

            if (HDSTR(dist_str) == STRING_PARAMS(default))
                return make_float3(0.0f, 1.0f, 1.0f);
            else
                return make_float3(1.0f, 0.0f, 1.0f);

            break;
        }

        default: cur = NULL; break;
        }

        if (cur == NULL && stack_idx > 0) {
            cur    = ptr_stack[--stack_idx];
            weight = weight_stack[stack_idx];
        }
    }
    return make_float3(result.x, result.y, result.z);
}


extern "C" __global__ void
__closesthit__deferred()
{
    Payload payload;
    payload.getPtr();
    OSL_CUDA::ShaderGlobals* sg_ptr = (OSL_CUDA::ShaderGlobals*) payload.ptr.ptr;
    globals_from_hit(*sg_ptr);
}


extern "C" __global__ void
__raygen__deferred()
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

    OSL_CUDA::ShaderGlobals sg;
    // globals_from_hit(sg);

    Payload payload;
    payload.ptr.ptr = (uint64_t)&sg;

    optixTrace(render_params.traversal_handle, // handle
               r.origin,                       // origin
               r.direction,                    // direction
               1e-3f,                          // tmin
               1e13f,                          // tmax
               0,                              // ray time
               OptixVisibilityMask(1),         // visibility mask
               OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // ray flags
               0,                              // SBT offset
               1,                              // SBT stride
               0,                              // miss SBT offset
               payload.ab.a,
               payload.ab.b
        );

    {
        alignas(8) char closure_pool[256];

        // OSL_CUDA::ShaderGlobals sg;
        // globals_from_hit(sg);

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
            shaderIdx, &sg /*shaderglobals_ptr*/, nullptr /*groupdata_ptr*/,
            nullptr /*userdata_base_ptr*/, nullptr /*output_base_ptr*/,
            0 /*shadeindex - unused*/, interactive_ptr /*interactive_params_ptr*/);

        float3 result         = process_closure((OSL::ClosureColor*)sg.Ci);
        float3* output_buffer = reinterpret_cast<float3*>(
            render_params.output_buffer);
        int pixel            = launch_index.y * launch_dims.x + launch_index.x;
        output_buffer[pixel] = make_float3(result.x, result.y, result.z);
    }

    if (launch_index.x == launch_dims.x / 2 && launch_index.y == launch_dims.y / 2)
    {
        printf("num_quads: %zu, quads_buffer: %p\n", render_params.num_quads, render_params.quads_buffer);
        printf("num_spheres: %zu, spheres_buffer: %p\n", render_params.num_spheres, render_params.spheres_buffer);

        for( size_t idx = 0; idx < render_params.num_spheres; ++idx )
        {
            SphereParams* spheres = (SphereParams*) render_params.spheres_buffer;
            if( spheres[idx].isLight ) {
                printf("let there be light!\n");

                int sx = launch_index.x;
                int sy = launch_index.y;
                int si = 0;

                Sampler sampler(sx, sy, si);
                Vec3 s   = sampler.get();
                float xi = s.x;
                float yi = s.y;
                // float zi = s.z;

                printf("xi: %6.3f, yi: %6.3f\n", xi, yi);

                float3 x  = sg.P; // hit point
                // float xi  = 0.0f;
                // float yi  = 0.0f;
                float pdf = 0.0f;
                float3 light_dir = sample_sphere(x, spheres[idx], xi, yi, pdf);
                printf("light_dir: %6.3f, %6.3f, %6.3f\n", light_dir.x, light_dir.y, light_dir.z );
            }
        }
    }
}
#endif
