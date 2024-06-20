#pragma once

#include <OSL/oslconfig.h>
#include <optix.h>

#include "rend_lib.h"
#include "../background.h"
#include "../raytracer.h"
#include "../sampling.h"

#include <cstdint>

#ifdef __CUDACC__

struct Payload {
    union {
        uint32_t raw[2];
        OSL_CUDA::ShaderGlobals* sg_ptr;
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

OSL_NAMESPACE_ENTER

struct CudaScene {
    OSL_HOSTDEVICE bool intersect(const Ray& r, Dual2<float>& t, int& primID,
                                  void* sg = nullptr) const;
    OSL_HOSTDEVICE float shapepdf(int primID, const Vec3& x,
                                  const Vec3& p) const;
    OSL_HOSTDEVICE bool islight(int primID) const;
    OSL_HOSTDEVICE Vec3 sample(int primID, const Vec3& x, float xi, float yi,
                               float& pdf) const;
    OSL_HOSTDEVICE int num_prims() const;

    uint64_t num_spheres;
    uint64_t num_quads;
    CUdeviceptr spheres_buffer;
    CUdeviceptr quads_buffer;
    OptixTraversableHandle handle;
};

struct SimpleRaytracer {
    using ShadingContext = ShadingContextCUDA;

    Background background;
    Camera camera;
    CudaScene scene;
    int aa                   = 1;
    int backgroundResolution = 1024;
    int backgroundShaderID   = -1;
    int max_bounces          = 1000000;
    int rr_depth             = 5;
    float show_albedo_scale  = 0.0f;

    OSL_HOSTDEVICE Vec3 eval_background(const Dual2<Vec3>& dir,
                                        ShadingContext* ctx, int bounce = -1);
    OSL_HOSTDEVICE Color3 subpixel_radiance(float x, float y, Sampler& sampler,
                                            ShadingContext* ctx = nullptr);
};

OSL_NAMESPACE_EXIT

#endif  // #ifdef __CUDACC__
