// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <optix.h>

#include <optix_device.h>

#include "rend_lib.h"


#include "render_params.h"
#include <optix_device.h>


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
__device__ __constant__ testshade::RenderParams render_params;
}

extern "C" __global__ void
__miss__()
{
    // do nothing
}

extern "C" __global__ void
__closesthit__()
{
    // do nothing
}

extern "C" __global__ void
__anyhit__()
{
    // do nothing
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
    OSL::pvt::num_named_xforms        = render_params.num_named_xforms;
    OSL::pvt::xform_name_buffer       = render_params.xform_name_buffer;
    OSL::pvt::xform_buffer            = render_params.xform_buffer;
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

    auto sbtdata = reinterpret_cast<GenericData*>(optixGetSbtDataPointer());

    const float invw      = render_params.invw;
    const float invh      = render_params.invh;
    bool flipv            = render_params.flipv;
    float3* output_buffer = reinterpret_cast<float3*>(
        render_params.output_buffer);


    // Compute the pixel coordinates
    // Matching testshade's setup_shaderglobals for !pixelcenters
    float2 d = make_float2((launch_dims.x == 1) ? 0.5f : invw * launch_index.x,
                           (launch_dims.y == 1) ? 0.5f : invh * launch_index.y);

    // TODO: Fixed-sized allocations can easily be exceeded by arbitrary shader
    //       networks, so there should be (at least) some mechanism to issue a
    //       warning or error if the closure or param storage can possibly be
    //       exceeded.
    StackClosurePool closure_pool;
    alignas(8) char params[256];

    OSL_CUDA::ShaderGlobals sg;
    // Setup the ShaderGlobals
    sg.I  = make_float3(0, 0, 1);
    sg.N  = make_float3(0, 0, 1);
    sg.Ng = make_float3(0, 0, 1);
    sg.P  = make_float3(d.x, d.y, 0);
    sg.u  = d.x;
    sg.v  = d.y;
    if (flipv)
        sg.v = 1.f - sg.v;

    sg.dudx = invw;
    sg.dudy = 0;
    sg.dvdx = 0;
    sg.dvdy = invh;

    // Matching testshade's setup_shaderglobals
    sg.dPdu = make_float3(1.f, 0.f, 0.f);
    sg.dPdv = make_float3(0.f, 1.f, 0.f);

    sg.dPdx = make_float3(1.f, 0.f, 0.f);
    sg.dPdy = make_float3(0.f, 1.f, 0.f);
    sg.dPdz = make_float3(0.f, 0.f, 0.f);

    sg.Ci          = NULL;
    sg.surfacearea = 0;
    sg.backfacing  = 0;

    // NB: These variables are not used in the current iteration of the sample
    sg.raytype        = OSL::Ray::CAMERA;
    sg.flipHandedness = 0;

    sg.shader2common = reinterpret_cast<void*>(render_params.shader2common);
    sg.object2common = reinterpret_cast<void*>(render_params.object2common);

    // Pack the "closure pool" into one of the ShaderGlobals pointers
    closure_pool.reset();
    RenderState renderState;
    // TODO: renderState.context = ...
    renderState.closure_pool = &closure_pool;
    sg.renderstate           = &renderState;

    // Run the OSL group and init functions
    if (render_params.fused_callable)
        // call osl_init_func
        optixDirectCall<void, OSL_CUDA::ShaderGlobals*, void*, void*, void*,
                        int, void*>(0u, &sg /*shaderglobals_ptr*/,
                                    params /*groupdata_ptr*/,
                                    nullptr /*userdata_base_ptr*/,
                                    nullptr /*output_base_ptr*/,
                                    0 /*shadeindex - unused*/,
                                    sbtdata->data /*interactive_params_ptr*/);
    else {
        // call osl_init_func
        optixDirectCall<void, OSL_CUDA::ShaderGlobals*, void*, void*, void*,
                        int, void*>(0u, &sg /*shaderglobals_ptr*/,
                                    params /*groupdata_ptr*/,
                                    nullptr /*userdata_base_ptr*/,
                                    nullptr /*output_base_ptr*/,
                                    0 /*shadeindex - unused*/,
                                    sbtdata->data /*interactive_params_ptr*/);
        // call osl_group_func
        optixDirectCall<void, OSL_CUDA::ShaderGlobals*, void*, void*, void*,
                        int, void*>(1u, &sg /*shaderglobals_ptr*/,
                                    params /*groupdata_ptr*/,
                                    nullptr /*userdata_base_ptr*/,
                                    nullptr /*output_base_ptr*/,
                                    0 /*shadeindex - unused*/,
                                    sbtdata->data /*interactive_params_ptr*/);
    }

    float* f_output      = (float*)params;
    int pixel            = launch_index.y * launch_dims.x + launch_index.x;
    output_buffer[pixel] = { f_output[1], f_output[2], f_output[3] };
}

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
