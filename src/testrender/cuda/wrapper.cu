// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <optix.h>

#include <cuda_runtime.h>
#include <optix.h>

#include <OSL/device_string.h>
#include <OSL/oslclosure.h>

#include "rend_lib.h"
#include "util.h"

#include "../render_params.h"


extern "C" {
__device__ __constant__ RenderParams render_params;
}


extern "C" __global__ void
__anyhit__any_hit_shadow()
{
    optixTerminateRay();
}



static __device__ void
globals_from_hit(ShaderGlobals& sg)
{
    const GenericRecord* record = reinterpret_cast<GenericRecord*>(
        optixGetSbtDataPointer());

    ShaderGlobals local_sg;
    // hit-kind 0: quad hit
    //          1: sphere hit
    optixDirectCall<void, unsigned int, float, float3, float3, ShaderGlobals*>(
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
        switch (((OSL::ClosureColor*)cur)->id) {
        case OSL::ClosureColor::ADD: {
            ptr_stack[stack_idx]      = ((OSL::ClosureAdd*)cur)->closureB;
            weight_stack[stack_idx++] = weight;
            cur                       = ((OSL::ClosureAdd*)cur)->closureA;
            break;
        }

        case OSL::ClosureColor::MUL: {
            weight *= ((OSL::ClosureMul*)cur)->weight;
            cur = ((OSL::ClosureMul*)cur)->closure;
            break;
        }

        case EMISSION_ID: {
            cur = NULL;
            break;
        }

        case DIFFUSE_ID:
        case OREN_NAYAR_ID:
        case PHONG_ID:
        case WARD_ID:
        case REFLECTION_ID:
        case REFRACTION_ID:
        case FRESNEL_REFLECTION_ID: {
            result += ((OSL::ClosureComponent*)cur)->w * weight;
            cur = NULL;
            break;
        }

        case MICROFACET_ID: {
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
__closesthit__closest_hit_osl()
{
    // TODO: Fixed-sized allocations can easily be exceeded by arbitrary shader
    //       networks, so there should be (at least) some mechanism to issue a
    //       warning or error if the closure or param storage can possibly be
    //       exceeded.
    alignas(8) char closure_pool[256];

    ShaderGlobals sg;
    globals_from_hit(sg);

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
    optixDirectCall<void, ShaderGlobals*, void*, void*, void*, int, void*>(
        shaderIdx, &sg /*shaderglobals_ptr*/, nullptr /*groupdata_ptr*/,
        nullptr /*userdata_base_ptr*/, nullptr /*output_base_ptr*/,
        0 /*shadeindex - unused*/, interactive_ptr /*interactive_params_ptr*/);

    float3 result      = process_closure((OSL::ClosureColor*)sg.Ci);
    uint3 launch_dims  = optixGetLaunchDimensions();
    uint3 launch_index = optixGetLaunchIndex();

    float3* output_buffer = reinterpret_cast<float3*>(
        render_params.output_buffer);
    int pixel            = launch_index.y * launch_dims.x + launch_index.x;
    output_buffer[pixel] = make_float3(result.x, result.y, result.z);
}
