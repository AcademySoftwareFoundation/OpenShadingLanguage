/*
Copyright (c) 2009-2018 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>

#include "rend_lib.h"

using namespace optix;

// Launch variables
rtDeclareVariable (uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable (uint2, launch_dim,   rtLaunchDim,   );

// Scene/Shading variables
rtDeclareVariable (float, invw, , );
rtDeclareVariable (float, invh, , );

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
    __attribute__((aligned(8))) char closure_pool[256];
    __attribute__((aligned(8))) char params      [256];

    ShaderGlobals sg;
    // Setup the ShaderGlobals
    sg.I           = make_float3(0,0,1);
    sg.N           = make_float3(0,0,1);
    sg.Ng          = make_float3(0,0,1);
    sg.P           = make_float3(d.x, d.y, 0);
    sg.u           = d.x * invw;
    sg.v           = d.y * invh;

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
