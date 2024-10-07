#pragma once

#include <OSL/oslconfig.h>
#include <optix.h>

#include "../background.h"
#include "../raytracer.h"
#include "../sampling.h"
#include "rend_lib.h"

#include <cstdint>

#ifdef __CUDACC__

struct TraceData {
    int32_t hit_id;
    float hit_t;
    float hit_u;
    float hit_v;
};

struct Payload {
    union {
        uint32_t tracedata_raw[2];
        TraceData* tracedata_ptr;
    };

    __forceinline__ __device__ void set()
    {
        optixSetPayload_0(tracedata_raw[0]);
        optixSetPayload_1(tracedata_raw[1]);
    }

    __forceinline__ __device__ void get()
    {
        tracedata_raw[0] = optixGetPayload_0();
        tracedata_raw[1] = optixGetPayload_1();
    }
};

OSL_NAMESPACE_ENTER

struct SimpleRaytracer {
    using ShadingContext = ShadingContextCUDA;

    Background background;
    Camera camera;
    Scene scene;
    int aa                          = 1;
    bool no_jitter                  = false;
    int backgroundResolution        = 1024;
    int backgroundShaderID          = -1;
    int max_bounces                 = 1000000;
    int rr_depth                    = 5;
    float show_albedo_scale         = 0.0f;
    int show_globals                = 0;
    const int* m_shader_is_light    = nullptr;
    const unsigned* m_lightprims    = nullptr;
    size_t lightprims_size          = 0;
    const int* m_shaderids          = nullptr;
    const int* m_meshids            = nullptr;
    const float* m_mesh_surfacearea = nullptr;

    OSL_HOSTDEVICE void globals_from_hit(OSL_CUDA::ShaderGlobals& sg,
                                         const Ray& r, const Dual2<float>& t,
                                         int id, float u, float v);
    OSL_HOSTDEVICE Vec3 eval_background(const Dual2<Vec3>& dir,
                                        ShadingContext* ctx, int bounce = -1);
    OSL_HOSTDEVICE Color3 subpixel_radiance(float x, float y, Sampler& sampler,
                                            ShadingContext* ctx = nullptr);
    OSL_HOSTDEVICE Color3 antialias_pixel(int x, int y,
                                          ShadingContext* ctx = nullptr);
};

OSL_NAMESPACE_EXIT

#endif  // #ifdef __CUDACC__
