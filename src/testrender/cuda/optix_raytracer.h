#pragma once

#include <OSL/oslconfig.h>
#include <optix.h>

#include "../background.h"
#include "../raytracer.h"
#include "../sampling.h"
#include "rend_lib.h"

#include <cstdint>

#ifdef __CUDACC__

struct Payload {
    uint32_t hit_id;
    float hit_t;
    float hit_u;
    float hit_v;

    __forceinline__ __device__ void set()
    {
        optixSetPayload_0(hit_id);
        optixSetPayload_1(__float_as_uint(hit_t));
        optixSetPayload_2(__float_as_uint(hit_u));
        optixSetPayload_3(__float_as_uint(hit_v));
    }

    __forceinline__ __device__ void get()
    {
        hit_id = static_cast<int32_t>(optixGetPayload_0());
        hit_t  = __uint_as_float(optixGetPayload_1());
        hit_u  = __uint_as_float(optixGetPayload_2());
        hit_v  = __uint_as_float(optixGetPayload_3());
    }
};

OSL_NAMESPACE_BEGIN

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

OSL_NAMESPACE_END

#endif  // #ifdef __CUDACC__
