// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage


#include <optix.h>

#if (OPTIX_VERSION < 70000)
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>
#else
#include <optix_device.h>
#endif

#include "rend_lib.h"


#if (OPTIX_VERSION < 70000)
using namespace optix;

// Launch variables
rtDeclareVariable (uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable (uint2, launch_dim,   rtLaunchDim,   );

// Scene/Shading variables
rtDeclareVariable (float, invw, , );
rtDeclareVariable (float, invh, , );
rtDeclareVariable (int,   flipv, , );

// Buffers
rtBuffer<float3,2> output_buffer;

rtDeclareVariable (rtCallableProgramId<void (void*, void*)>, osl_init_func, , );
rtDeclareVariable (rtCallableProgramId<void (void*, void*)>, osl_group_func, ,);

RT_PROGRAM void raygen()
{
    // Compute the pixel coordinates
    float2 d = make_float2 (static_cast<float>(launch_index.x) + 0.5f,
                            static_cast<float>(launch_index.y) + 0.5f);

    // TODO: Fixed-sized allocations can easily be exceeded by arbitrary shader
    //       networks, so there should be (at least) some mechanism to issue a
    //       warning or error if the closure or param storage can possibly be
    //       exceeded.
    alignas(8) char closure_pool[256];
    alignas(8) char params      [256];

    ShaderGlobals sg;
    // Setup the ShaderGlobals
    sg.I           = make_float3(0,0,1);
    sg.N           = make_float3(0,0,1);
    sg.Ng          = make_float3(0,0,1);
    sg.P           = make_float3(d.x, d.y, 0);
    sg.u           = d.x * invw;
    sg.v           = d.y * invh;
    if (flipv)
         sg.v      = 1.f - sg.v;

    sg.dudx        = invw;
    sg.dudy        = 0;
    sg.dvdx        = 0;
    sg.dvdy        = invh;
    sg.dPdu        = make_float3(d.x, 0, 0);
    sg.dPdv        = make_float3(0, d.y, 0);

    sg.dPdu        = make_float3(1.f / invw, 0.f , 0.f);
    sg.dPdv        = make_float3(0.0f, 1.f / invh, 0.f);

    sg.dPdx        = make_float3(1.f, 0.f, 0.f);
    sg.dPdy        = make_float3(0.f, 1.f, 0.f);
    sg.dPdz        = make_float3(0.f, 0.f, 0.f);

    sg.Ci          = NULL;
    sg.surfacearea = 0;
    sg.backfacing  = 0;

    // NB: These variables are not used in the current iteration of the sample
    sg.raytype = CAMERA;
    sg.flipHandedness = 0;

    // Pack the "closure pool" into one of the ShaderGlobals pointers
    *(int*) &closure_pool[0] = 0;
    sg.renderstate = &closure_pool[0];

    // Run the OSL group and init functions
    osl_init_func (&sg, params);
    osl_group_func(&sg, params);

    float* output = (float*)params;
    output_buffer[launch_index] = {output[1], output[2], output[3]};
}

#else //#if (OPTIX_VERSION < 70000)

#include "render_params.h"
#include <optix_device.h>

extern "C" {
__device__ __constant__ RenderParams render_params;
}

extern "C" __global__ void __miss__()
{
    // do nothing
}

extern "C" __global__ void __closesthit__()
{
    // do nothing
}

extern "C" __global__ void __anyhit__()
{
    // do nothing
}

extern "C" __global__ void __raygen__()
{
    uint3 launch_dims  = optixGetLaunchDimensions();
    uint3 launch_index = optixGetLaunchIndex();

    void *p = reinterpret_cast<void *>(optixGetSbtDataPointer());

    // Compute the pixel coordinates
    float2 d = make_float2 (static_cast<float>(launch_index.x) + 0.5f,
                            static_cast<float>(launch_index.y) + 0.5f);

    // TODO: Fixed-sized allocations can easily be exceeded by arbitrary shader
    //       networks, so there should be (at least) some mechanism to issue a
    //       warning or error if the closure or param storage can possibly be
    //       exceeded.
    alignas(8) char closure_pool[256];
    alignas(8) char params      [256];

    const float invw = render_params.invw;
    const float invh = render_params.invh;
    bool       flipv = render_params.flipv;
    float3* output_buffer = reinterpret_cast<float3 *>(render_params.output_buffer);

    ShaderGlobals sg;
    // Setup the ShaderGlobals
    sg.I           = make_float3(0,0,1);
    sg.N           = make_float3(0,0,1);
    sg.Ng          = make_float3(0,0,1);
    sg.P           = make_float3(d.x, d.y, 0);
    sg.u           = d.x * invw;
    sg.v           = d.y * invh;
    if (flipv)
         sg.v      = 1.f - sg.v;

    sg.dudx        = invw;
    sg.dudy        = 0;
    sg.dvdx        = 0;
    sg.dvdy        = invh;
    sg.dPdu        = make_float3(d.x, 0, 0);
    sg.dPdv        = make_float3(0, d.y, 0);

    sg.dPdu        = make_float3(1.f / invw, 0.f , 0.f);
    sg.dPdv        = make_float3(0.0f, 1.f / invh, 0.f);

    sg.dPdx        = make_float3(1.f, 0.f, 0.f);
    sg.dPdy        = make_float3(0.f, 1.f, 0.f);
    sg.dPdz        = make_float3(0.f, 0.f, 0.f);

    sg.Ci          = NULL;
    sg.surfacearea = 0;
    sg.backfacing  = 0;

    // NB: These variables are not used in the current iteration of the sample
    sg.raytype = CAMERA;
    sg.flipHandedness = 0;

    // Pack the "closure pool" into one of the ShaderGlobals pointers
    *(int*) &closure_pool[0] = 0;
    sg.renderstate = &closure_pool[0];

    // Run the OSL group and init functions
    optixDirectCall<void, ShaderGlobals*, void *>(0u, &sg, params); // call osl_init_func
    optixDirectCall<void, ShaderGlobals*, void *>(1u, &sg, params); // call osl_group_func

    float* f_output = (float*)params;
    int pixel = launch_index.y * launch_dims.x + launch_index.x;
    output_buffer[pixel] = {f_output[1], f_output[2], f_output[3]};
}

// Because clang++ 9.0 seems to have trouble with some of the texturing "intrinsics"
// let's do the texture look-ups in this file.
extern "C"
__device__ float4 osl_tex2DLookup(void *handle, float s, float t)
{
    cudaTextureObject_t texID = cudaTextureObject_t(handle);
    return tex2D<float4>(texID, s, t);
}


#endif //#if (OPTIX_VERSION < 70000)
