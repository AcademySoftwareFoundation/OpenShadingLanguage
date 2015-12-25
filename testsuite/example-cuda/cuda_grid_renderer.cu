// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include "rend_lib.h"
#include <OSL/oslclosure.h>

// these trampolines will be linked in by the renderer
extern "C" __device__ void
__osl__init(ShaderGlobals*, void*);
extern "C" __device__ void
__osl__entry(ShaderGlobals*, void*);

using OSL::ClosureAdd;
using OSL::ClosureColor;
using OSL::ClosureComponent;
using OSL::ClosureMul;

#include "closures.h"

// Recursive function to expore the closure tree as an example. You wouldn't want
// to do this in a real renderer
__device__ void
process_closure(const ClosureColor* closure, int depth = 0)
{
    ClosureMul* cmul;
    ClosureAdd* cadd;
    ClosureComponent* ccomp;
    for (int i = 0; i < depth; ++i) {
        printf("    ");
    }
    switch (closure->id) {
    case ClosureColor::MUL:
        cmul = (ClosureMul*)closure;
        printf("MUL %f %f %f\n", cmul->weight.x, cmul->weight.y,
               cmul->weight.z);
        process_closure(cmul->closure, depth + 1);
        break;
    case ClosureColor::ADD:
        cadd = (ClosureAdd*)closure;
        printf("ADD\n");
        process_closure(cadd->closureA, depth + 1);
        process_closure(cadd->closureB, depth + 1);
        break;
    default:
        ccomp = (ClosureComponent*)closure;
        printf("COMP %d (%f %f %f)\n", ccomp->id, ccomp->w.x, ccomp->w.y,
               ccomp->w.z);
        break;
    }
}

extern "C" __global__ void
shade(float3* Cout, int w, int h)
{
    int block_id  = blockIdx.x + blockIdx.y * gridDim.x;
    int thread_id = block_id * (blockDim.x * blockDim.y)
                    + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (thread_id >= w * h) {
        return;
    }

    int x = thread_id % w;
    int y = thread_id / h;

    float u = float(x) / w;
    float v = float(y) / h;

    float2 d = make_float2(static_cast<float>(x) + 0.5f,
                           static_cast<float>(y) + 0.5f);

    // TODO: Fixed-sized allocations can easily be exceeded by arbitrary shader
    //       networks, so there should be (at least) some mechanism to issue a
    //       warning or error if the closure or param storage can possibly be
    //       exceeded.
    alignas(8) char closure_pool[256];
    alignas(8) char params[256];

    const float invw = 1.0 / w;
    const float invh = 1.0 / h;
    const bool flipv = false;

    ShaderGlobals sg;

    sg.I  = make_float3(0, 0, 1);
    sg.N  = make_float3(0, 0, 1);
    sg.Ng = make_float3(0, 0, 1);
    sg.P  = make_float3(d.x, d.y, 0);
    sg.u  = d.x * invw;
    sg.v  = d.y * invh;
    if (flipv)
        sg.v = 1.f - sg.v;

    sg.dudx = invw;
    sg.dudy = 0;
    sg.dvdx = 0;
    sg.dvdy = invh;
    sg.dPdu = make_float3(d.x, 0, 0);
    sg.dPdv = make_float3(0, d.y, 0);

    sg.dPdu = make_float3(1.f / invw, 0.f, 0.f);
    sg.dPdv = make_float3(0.0f, 1.f / invh, 0.f);

    sg.dPdx = make_float3(1.f, 0.f, 0.f);
    sg.dPdy = make_float3(0.f, 1.f, 0.f);
    sg.dPdz = make_float3(0.f, 0.f, 0.f);

    sg.Ci          = NULL;
    sg.surfacearea = 0;
    sg.backfacing  = 0;

    // NB: These variables are not used in the current iteration of the sample
    sg.raytype        = CAMERA;
    sg.flipHandedness = 0;

    // Pack the "closure pool" into one of the ShaderGlobals pointers
    *(int*)&closure_pool[0] = 0;
    sg.renderstate          = &closure_pool[0];

    // Run the shader
    __osl__init(&sg, params);
    __osl__entry(&sg, params);

    // the test shader creates a single diffuse closure, weighted by a colour
    // noise pattern. Extract the weight from the mul closure and write it to
    // the output buffer
    if (sg.Ci) {
        auto cmul       = (ClosureMul*)sg.Ci;
        Cout[thread_id] = { cmul->weight.x, cmul->weight.y, cmul->weight.z };
        // Recursively traverse the closure tree for a single thread as an
        // example
        if (x == 10 && y == 10) {
            process_closure((const ClosureColor*)sg.Ci);
        }
    } else {
        Cout[thread_id] = { 0.0f, 0.0f, 0.0f };
    }
}

// Because clang++ 9.0 seems to have trouble with some of the texturing
// "intrinsics" let's do the texture look-ups in this file.
extern "C" __device__ float4
osl_tex2DLookup(void* handle, float s, float t)
{
    cudaTextureObject_t texID = cudaTextureObject_t(handle);
    return tex2D<float4>(texID, s, t);
}
