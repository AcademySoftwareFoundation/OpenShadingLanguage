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

#include <OSL/device_string.h>

#include "rend_lib.h"
#include "util.h"

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
float3 process_closure(const ClosureColor* closure_tree)
{
    float3 result = make_float3 (0.0f);

    if (!closure_tree) {
        return result;
    }

    // The depth of the closure tree must not exceed the stack size.
    // A stack size of 8 is probably quite generous for relatively
    // balanced trees.
    const int STACK_SIZE = 8;

    // Non-recursive traversal stack
    int    stack_idx = 0;
    void*  ptr_stack   [STACK_SIZE];
    float3 weight_stack[STACK_SIZE];

    // Shading accumlator
    float3 weight = make_float3(1.0f);

    const void* cur = closure_tree;
    while (cur) {
        switch (((ClosureColor*)cur)->id) {
        case ClosureColor::ADD: {
            ptr_stack   [stack_idx  ] = ((ClosureAdd*) cur)->closureB;
            weight_stack[stack_idx++] = weight;
            cur = ((ClosureAdd*) cur)->closureA;
            break;
        }

        case ClosureColor::MUL: {
            weight *= ((ClosureMul*) cur)->weight;
            cur     = ((ClosureMul*) cur)->closure;
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
            result += ((ClosureComponent*) cur)->w * weight;
            cur = NULL;
            break;
        }

        case MICROFACET_ID: {
#if 0
            const char* mem = ((ClosureComponent*) cur)->mem;
            const char* dist_str = *(const char**) &mem[0];
            if (launch_index.x == launch_dim.x / 2 && launch_index.y == launch_dim.y / 2) {
                printf ("microfacet, dist: %s\n", HDSTR(dist_str).c_str());
                // Comparisons between the closure variable and "standard"
                // strings are possible
                if (HDSTR(dist_str) == OSLDeviceStrings::default_)
                    printf("dist is default\n");
            }
#endif

            result += ((ClosureComponent*) cur)->w * weight;
            cur = NULL;
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

    return result;
}


RT_PROGRAM void closest_hit_osl()
{
    // TODO: Fixed-sized allocations can easily be exceeded by arbitrary shader
    //       networks, so there should be (at least) some mechanism to issue a
    //       warning or error if the closure or param storage can possibly be
    //       exceeded.
    __attribute__((aligned(8))) char closure_pool[256];
    __attribute__((aligned(8))) char params      [256];

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

    prd_radiance.result = process_closure ((ClosureColor*) sg.Ci);
}
