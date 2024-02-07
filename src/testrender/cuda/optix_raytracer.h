#pragma once

#include <OSL/oslconfig.h>
#include <optix.h>

#include "../raytracer.h"

#include <cstdint>

#ifdef __CUDACC__

struct Payload {
    union {
        uint32_t raw[2];
        uint64_t sg_ptr;
    };
    float radius;
    float spread;
    OSL::Ray::RayType raytype;

    __forceinline__ __device__ void set()
    {
        optixSetPayload_0(raw[0]);
        optixSetPayload_1(raw[1]);
        optixSetPayload_2(__float_as_uint(radius));
        optixSetPayload_3(__float_as_uint(spread));
        optixSetPayload_4((uint32_t)raytype);
    }

    __forceinline__ __device__ void get()
    {
        raw[0]  = optixGetPayload_0();
        raw[1]  = optixGetPayload_1();
        radius  = __uint_as_float(optixGetPayload_2());
        spread  = __uint_as_float(optixGetPayload_3());
        raytype = (OSL::Ray::RayType)optixGetPayload_4();
    }
};


static __device__ __inline__ uchar4
make_color(const float3& c)
{
    return make_uchar4(
        static_cast<unsigned char>(__saturatef(c.z) * 255.99f), /* B */
        static_cast<unsigned char>(__saturatef(c.y) * 255.99f), /* G */
        static_cast<unsigned char>(__saturatef(c.x) * 255.99f), /* R */
        255u);                                                  /* A */
}

#endif  // #ifdef __CUDACC__
