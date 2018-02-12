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

#include "util.h"


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


RT_PROGRAM void closest_hit()
{
    prd_radiance.result = normalize (rtTransformNormal (RT_OBJECT_TO_WORLD, shading_normal)) * 0.5f + 0.5f;
}
