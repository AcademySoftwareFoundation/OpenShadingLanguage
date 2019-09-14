// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage


#pragma once

#include <optix.h>

#if (OPTIX_VERSION < 70000)
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>
typedef optix::float3 float3;
typedef optix::uchar4 uchar4;
#endif

#include <stdint.h>


struct PRD_radiance
{
    float3 result;
};


struct RayGeometry {
    float3 origin;
    float3 direction;
};


static __device__ __inline__
uchar4 make_color (const float3& c)
{
    return make_uchar4 (static_cast<unsigned char>(__saturatef(c.z)*255.99f),  /* B */
                        static_cast<unsigned char>(__saturatef(c.y)*255.99f),  /* G */
                        static_cast<unsigned char>(__saturatef(c.x)*255.99f),  /* R */
                        255u);                                                 /* A */
}

