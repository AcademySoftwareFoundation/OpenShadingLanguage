// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage


#include <optix.h>

#if (OPTIX_VERSION < 70000) 
#include <optixu/optixu_math_namespace.h>
#else
#include <optix.h>
#include <cuda_runtime.h>
#endif

#include <OSL/device_string.h>
#include <OSL/oslclosure.h>

#include "rend_lib.h"
#include "util.h"

#if (OPTIX_VERSION < 70000)

// Ray payload
rtDeclareVariable (PRD_radiance, prd_radiance, rtPayload, );

// ray/hit variables
rtDeclareVariable (float3, shading_normal,   attribute shading_normal,  );
rtDeclareVariable (float3, geometric_normal, attribute geometric_normal,);
rtDeclareVariable (float3, texcoord,         attribute texcoord,        );
rtDeclareVariable (float,  surface_area,     attribute surface_area,    );
rtDeclareVariable (float3, dPdu,             attribute dPdu,            );
rtDeclareVariable (float3, dPdv,             attribute dPdv,            );
rtDeclareVariable (int,    obj_id,           attribute obj_id,          );
rtDeclareVariable (int,    lgt_idx,          attribute lgt_idx,         );

// ray/hit variables
rtDeclareVariable (uint2,      launch_index, rtLaunchIndex,          );
rtDeclareVariable (uint2,      launch_dim,   rtLaunchDim,            );
rtDeclareVariable (optix::Ray, ray,          rtCurrentRay,           );
rtDeclareVariable (float,      t_hit,        rtIntersectionDistance, );

// Buffers
rtBuffer<float3,2> output_buffer;

// Function pointers for the OSL shader
rtDeclareVariable (rtCallableProgramId<void (void*, void*)>, osl_init_func, , );
rtDeclareVariable (rtCallableProgramId<void (void*, void*)>, osl_group_func, ,);

RT_PROGRAM void any_hit_shadow()
{
    rtTerminateRay();
}


static __device__
void globals_from_hit(ShaderGlobals& sg)
{
    // Setup the ShaderGlobals
    sg.I           = ray.direction;
    sg.N           = normalize(rtTransformNormal (RT_OBJECT_TO_WORLD, shading_normal));
    sg.Ng          = normalize(rtTransformNormal (RT_OBJECT_TO_WORLD, geometric_normal));
    sg.P           = ray.origin + t_hit * ray.direction;
    sg.dPdu        = dPdu;
    sg.u           = texcoord.x;
    sg.v           = texcoord.y;
    sg.Ci          = NULL;
    sg.surfacearea = surface_area;
    sg.backfacing  = (dot(sg.N, sg.I) > 0.0f);

    if (sg.backfacing) {
        sg.N  = -sg.N;
        sg.Ng = -sg.Ng;
    }

    // NB: These variables are not used in the current iteration of the sample
    sg.raytype = CAMERA;
    sg.flipHandedness = 0;
}


static __device__
float3 process_closure(const OSL::ClosureColor* closure_tree)
{
    OSL::Color3 result = OSL::Color3 (0.0f);

    if (!closure_tree) {
        return make_float3(result.x, result.y, result.z);
    }

    // The depth of the closure tree must not exceed the stack size.
    // A stack size of 8 is probably quite generous for relatively
    // balanced trees.
    const int STACK_SIZE = 8;

    // Non-recursive traversal stack
    int    stack_idx = 0;
    const OSL::ClosureColor* ptr_stack[STACK_SIZE];
    OSL::Color3 weight_stack[STACK_SIZE];

    // Shading accumulator
    OSL::Color3 weight = OSL::Color3(1.0f);

    const void* cur = closure_tree;
    while (cur) {
        switch (((OSL::ClosureColor*)cur)->id) {
        case OSL::ClosureColor::ADD: {
            ptr_stack   [stack_idx  ] = ((OSL::ClosureAdd*) cur)->closureB;
            weight_stack[stack_idx++] = weight;
            cur = ((OSL::ClosureAdd*) cur)->closureA;
            break;
        }

        case OSL::ClosureColor::MUL: {
            weight *= ((OSL::ClosureMul*) cur)->weight;
            cur     = ((OSL::ClosureMul*) cur)->closure;
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
            result += ((OSL::ClosureComponent*) cur)->w * weight;
            cur = NULL;
            break;
        }

        case MICROFACET_ID: {
            const char* mem = (const char*)((OSL::ClosureComponent*) cur)->data();
            const char* dist_str = *(const char**) &mem[0];
#if 0
            if (launch_index.x == launch_dim.x / 2 && launch_index.y == launch_dim.y / 2)
                printf ("microfacet, dist: %s\n", HDSTR(dist_str).c_str());
#endif

            if (HDSTR(dist_str) == OSL::DeviceStrings::default_)
                return make_float3(0.0f, 1.0f, 1.0f);

            return make_float3(1.0f, 0.0f, 1.0f);
        }

        default:
            cur = NULL;
            break;
        }

        if (cur == NULL && stack_idx > 0) {
            cur    = ptr_stack   [--stack_idx];
            weight = weight_stack[  stack_idx];
        }
    }

    return make_float3(result.x, result.y, result.z);
}


RT_PROGRAM void closest_hit_osl()
{
    // TODO: Fixed-sized allocations can easily be exceeded by arbitrary shader
    //       networks, so there should be (at least) some mechanism to issue a
    //       warning or error if the closure or param storage can possibly be
    //       exceeded.
    alignas(8) char closure_pool[256];
    alignas(8) char params      [256];

    ShaderGlobals sg;
    globals_from_hit (sg);

    // Pack the "closure pool" into one of the ShaderGlobals pointers
    *(int*) &closure_pool[0] = 0;
    sg.renderstate = &closure_pool[0];

    // Create some run-time options structs. The OSL shader fills in the structs
    // as it executes, based on the options specified in the shader source.
    NoiseOptCUDA   noiseopt;
    TextureOptCUDA textureopt;
    TraceOptCUDA   traceopt;

    // Pack the pointers to the options structs in a faux "context",
    // which is a rough stand-in for the host ShadingContext.
    ShadingContextCUDA shading_context = {
        &noiseopt, &textureopt, &traceopt
    };

    sg.context = &shading_context;

    // Run the OSL group and init functions
    osl_init_func (&sg, params);
    osl_group_func(&sg, params);

    prd_radiance.result = process_closure ((OSL::ClosureColor*) sg.Ci);
}

#else //#if (OPTIX_VERSION < 70000)


#include "../render_params.h"

extern "C" {
__device__ __constant__ RenderParams render_params;
}

extern"C" __global__ void __anyhit__any_hit_shadow ()
{
    optixTerminateRay();
}


static __device__
void globals_from_hit (ShaderGlobals& sg)
{
    const GenericRecord *record = reinterpret_cast<GenericRecord *> (optixGetSbtDataPointer());

    ShaderGlobals local_sg;
    // hit-kind 0: quad hit
    //          1: sphere hit
    optixDirectCall<void, unsigned int, float, float3, float3, ShaderGlobals *>(
                                     optixGetHitKind(),
                                     optixGetPrimitiveIndex(),
                                     optixGetRayTmax(),
                                     optixGetWorldRayOrigin(),
                                     optixGetWorldRayDirection(),
                                     &local_sg);
    // Setup the ShaderGlobals
    const float3 ray_direction = optixGetWorldRayDirection();
    const float3 ray_origin    = optixGetWorldRayOrigin();
    const float  t_hit         = optixGetRayTmin();

    sg.I           = ray_direction;
    sg.N           = normalize (optixTransformNormalFromObjectToWorldSpace (local_sg.N));
    sg.Ng          = normalize (optixTransformNormalFromObjectToWorldSpace (local_sg.Ng));
    sg.P           = ray_origin + t_hit * ray_direction;
    sg.dPdu        = local_sg.dPdu;
    sg.dPdv        = local_sg.dPdv;
    sg.u           = local_sg.u;
    sg.v           = local_sg.v;
    sg.Ci          = NULL;
    sg.surfacearea = local_sg.surfacearea;
    sg.backfacing  = dot (sg.N, sg.I) > 0.0f;
    sg.shaderID    = local_sg.shaderID;

    if (sg.backfacing) {
        sg.N  = -sg.N;
        sg.Ng = -sg.Ng;
    }

    // NB: These variables are not used in the current iteration of the sample
    sg.raytype = CAMERA;
    sg.flipHandedness = 0;
}


static __device__
float3 process_closure (const OSL::ClosureColor* closure_tree)
{
    OSL::Color3 result = OSL::Color3 (0.0f);

    if (!closure_tree) {
        return make_float3 (result.x, result.y, result.z);
    }

    // The depth of the closure tree must not exceed the stack size.
    // A stack size of 8 is probably quite generous for relatively
    // balanced trees.
    const int STACK_SIZE = 8;

    // Non-recursive traversal stack
    int    stack_idx = 0;
    const OSL::ClosureColor* ptr_stack[STACK_SIZE];
    OSL::Color3 weight_stack[STACK_SIZE];

    // Shading accumulator
    OSL::Color3 weight = OSL::Color3 (1.0f);

    const void* cur = closure_tree;
    while (cur) {
        switch (((OSL::ClosureColor*)cur)->id) {
        case OSL::ClosureColor::ADD: {
            ptr_stack   [stack_idx  ] = ((OSL::ClosureAdd*) cur)->closureB;
            weight_stack[stack_idx++] = weight;
            cur = ((OSL::ClosureAdd*) cur)->closureA;
            break;
        }

        case OSL::ClosureColor::MUL: {
            weight *= ((OSL::ClosureMul*) cur)->weight;
            cur     = ((OSL::ClosureMul*) cur)->closure;
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
            result += ((OSL::ClosureComponent*) cur)->w * weight;
            cur = NULL;
            break;
        }

        case MICROFACET_ID: {
            const char* mem = (const char*)((OSL::ClosureComponent*) cur)->data();
            const char* dist_str = *(const char**) &mem[0];

            if (HDSTR(dist_str) == OSL::DeviceStrings::default_)
                return make_float3(0.0f, 1.0f, 1.0f);
            else
                return make_float3(1.0f, 0.0f, 1.0f);

            break;
        }

        default:
            cur = NULL;
            break;
        }

        if (cur == NULL && stack_idx > 0) {
            cur    = ptr_stack   [--stack_idx];
            weight = weight_stack[  stack_idx];
        }
    }

    return make_float3(result.x, result.y, result.z);
}


extern "C" __global__  void __closesthit__closest_hit_osl()
{
    // TODO: Fixed-sized allocations can easily be exceeded by arbitrary shader
    //       networks, so there should be (at least) some mechanism to issue a
    //       warning or error if the closure or param storage can possibly be
    //       exceeded.
    alignas(8) char closure_pool[256];
    alignas(8) char params      [256];

    ShaderGlobals sg;
    globals_from_hit (sg);

    // Pack the "closure pool" into one of the ShaderGlobals pointers
    *(int*) &closure_pool[0] = 0;
    sg.renderstate = &closure_pool[0];

    // Create some run-time options structs. The OSL shader fills in the structs
    // as it executes, based on the options specified in the shader source.
    NoiseOptCUDA   noiseopt;
    TextureOptCUDA textureopt;
    TraceOptCUDA   traceopt;

    // Pack the pointers to the options structs in a faux "context",
    // which is a rough stand-in for the host ShadingContext.
    ShadingContextCUDA shading_context = {
        &noiseopt, &textureopt, &traceopt
    };

    sg.context = &shading_context;

    // Run the OSL group and init functions
    const unsigned int shaderInitOpIdx = 2u + 2u * sg.shaderID + 0u;
    const unsigned int shaderGroupIdx  = 2u + 2u * sg.shaderID + 1u;
    optixDirectCall<void, ShaderGlobals*, void *>(shaderInitOpIdx, &sg, params); // call osl_init_func
    optixDirectCall<void, ShaderGlobals*, void *>(shaderGroupIdx , &sg, params); // call osl_group_func

    float3 result = process_closure ((OSL::ClosureColor*) sg.Ci);
    uint3 launch_dims  = optixGetLaunchDimensions();
    uint3 launch_index = optixGetLaunchIndex();

    float3* output_buffer = reinterpret_cast<float3 *>(render_params.output_buffer);
    int pixel = launch_index.y * launch_dims.x + launch_index.x;
    output_buffer[pixel] = make_float3(result.x, result.y, result.z);

}

#endif //#if (OPTIX_VERSION < 70000)
