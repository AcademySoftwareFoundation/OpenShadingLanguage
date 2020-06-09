// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage


#include <optix.h>

#if (OPTIX_VERSION < 70000)
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>
#endif

#include "util.h"

#if (OPTIX_VERSION < 70000)
using namespace optix;

// Launch variables
rtDeclareVariable (uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable (uint2, launch_dim,   rtLaunchDim,   );

// Scene/Shading variables
rtDeclareVariable (float3,   bad_color, ,  );
rtDeclareVariable (float3,   bg_color, ,   );
rtDeclareVariable (rtObject, top_object, , );

// Ray payload
rtDeclareVariable (PRD_radiance, prd_radiance, rtPayload, );

// Geometry/Intersection attributes
rtDeclareVariable (float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable (float3, shading_normal,   attribute shading_normal, );

// Camera variables
rtDeclareVariable (float3, eye, , );
rtDeclareVariable (float3, dir, , );
rtDeclareVariable (float3, cx,  , );
rtDeclareVariable (float3, cy,  , );

rtDeclareVariable (float, invw, , );
rtDeclareVariable (float, invh, , );

// Buffers
rtBuffer<float3,2> output_buffer;


RT_PROGRAM void raygen()
{
    // Compute the pixel coordinates
    float2 d = make_float2 (static_cast<float>(launch_index.x) + 0.5f,
                            static_cast<float>(launch_index.y) + 0.5f);

    // Make the ray for the current pixel
    RayGeometry r;
    r.origin = eye;
    r.direction = optix::normalize(cx * (d.x * invw - 0.5f) + cy * (0.5f - d.y * invh) + dir);

    Ray ray = optix::make_Ray (r.origin, r.direction, 0, 1e-3f, RT_DEFAULT_MAX);

    // Create a struct to hold the shading result
    PRD_radiance prd;
    prd.result = make_float3 (0.0f);

    // Trace the ray against the scene. The hit/miss program is called before
    // this call returns.
    rtTrace (top_object, ray, prd);

    // Write the shading result to the output buffer
    output_buffer[launch_index] = prd.result;
}


RT_PROGRAM void miss()
{
    prd_radiance.result = bg_color;
}


RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
    output_buffer[launch_index] = bad_color;
}

#else //#if (OPTIX_VERSION < 70000)

#include <optix.h>
#include <optix_device.h>

#include <OSL/device_string.h>

#include "rend_lib.h"
#include "render_params.h"


extern "C" {
__device__ __constant__ RenderParams render_params;
}


extern "C" __global__ void __miss__()
{
    uint3 launch_dims  = optixGetLaunchDimensions();
    uint3 launch_index = optixGetLaunchIndex();

    float3* output_buffer = reinterpret_cast<float3 *>(render_params.output_buffer);

    int pixel = launch_index.y * launch_dims.x + launch_index.x;
    output_buffer[pixel] = make_float3(0,0,1);
}


extern "C" __global__ void __raygen__()
{
    uint3 launch_dims  = optixGetLaunchDimensions();
    uint3 launch_index = optixGetLaunchIndex();

    const float3 eye = render_params.eye;
    const float3 dir = render_params.dir;
    const float3 cx  = render_params.cx ;
    const float3 cy  = render_params.cy ;
    const float invw = render_params.invw;
    const float invh = render_params.invh;

    // Compute the pixel coordinates
    const float2 d = make_float2 (static_cast<float>(launch_index.x) + 0.5f,
                                  static_cast<float>(launch_index.y) + 0.5f);

    // Make the ray for the current pixel
    RayGeometry r;
    r.origin = eye;
    r.direction = normalize (cx * (d.x * invw - 0.5f) + cy * (0.5f - d.y * invh) + dir);
    optixTrace (render_params.traversal_handle,
                r.origin,
                r.direction,
                1e-3f,
                1e13f,
                0,
                OptixVisibilityMask(1),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                0,
                1,
                0);
}


extern __device__ char *test_str_1;
extern __device__ char *test_str_2;


// Because clang++ 9.0 seems to have trouble with some of the texturing "intrinsics"
// let's do the texture look-ups in this file.
extern "C"
__device__ float4 osl_tex2DLookup (void *handle, float s, float t)
{
    cudaTextureObject_t texID = cudaTextureObject_t (handle);
    return tex2D<float4>(texID, s, t);
}

#endif //#if (OPTIX_VERSION < 70000)

