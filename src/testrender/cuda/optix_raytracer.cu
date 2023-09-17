// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <optix.h>

#include "util.h"

#include <optix_device.h>

#include <OSL/device_string.h>

#include "rend_lib.h"
#include "render_params.h"


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
