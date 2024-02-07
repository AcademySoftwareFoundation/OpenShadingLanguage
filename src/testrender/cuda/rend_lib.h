// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>

#if defined(__has_include) && __has_include(<Imath/half.h>)
#    include <Imath/half.h>
#elif OSL_USING_IMATH >= 3
#    include <Imath/half.h>
#else
#    include <OpenEXR/half.h>
#endif

#include <OSL/hashes.h>
#include <OSL/oslexec.h>


#include "../raytracer.h"

OSL_NAMESPACE_ENTER


// TODO: update types from char * to ustringhash or ustringhash_pod
//       and remove uses of HDSTR
#define HDSTR(cstr) (*((OSL::ustringhash*)&cstr))

namespace pvt {
extern __device__ CUdeviceptr s_color_system;
extern __device__ CUdeviceptr osl_printf_buffer_start;
extern __device__ CUdeviceptr osl_printf_buffer_end;
extern __device__ uint64_t test_str_1;
extern __device__ uint64_t test_str_2;
extern __device__ uint64_t num_named_xforms;
extern __device__ CUdeviceptr xform_name_buffer;
extern __device__ CUdeviceptr xform_buffer;
}  // namespace pvt

OSL_NAMESPACE_EXIT

namespace {  // anonymous namespace

// These are CUDA variants of various OSL options structs. Their layouts and
// default values are identical to the host versions, but they might differ in
// how they are constructed. They are duplicated here as a convenience and to
// avoid including additional host headers.

struct NoiseOptCUDA {
    int anisotropic;
    int do_filter;
    float3 direction;
    float bandwidth;
    float impulses;

    __device__ NoiseOptCUDA()
        : anisotropic(0)
        , do_filter(true)
        , direction(make_float3(1.0f, 0.0f, 0.0f))
        , bandwidth(1.0f)
        , impulses(16.0f)
    {
    }
};


struct TextureOptCUDA {
    // TO BE IMPLEMENTED
};


struct TraceOptCUDA {
    // TO BE IMPLEMENTED
};


// This isn't really a CUDA version of the host-side ShadingContext class;
// instead, it is used as a container for a handful of pointers accessed during
// shader executions that are accessed via the ShadingContext.
struct ShadingContextCUDA {
    NoiseOptCUDA* m_noiseopt;
    TextureOptCUDA* m_textureopt;
    TraceOptCUDA* m_traceopt;

    __device__ void* noise_options_ptr() { return m_noiseopt; }
    __device__ void* texture_options_ptr() { return m_textureopt; }
    __device__ void* trace_options_ptr() { return m_traceopt; }
};


namespace OSL_CUDA {
struct ShaderGlobals {
    float3 P, dPdx, dPdy;
    float3 dPdz;
    float3 I, dIdx, dIdy;
    float3 N;
    float3 Ng;
    float u, dudx, dudy;
    float v, dvdx, dvdy;
    float3 dPdu, dPdv;
    float time;
    float dtime;
    float3 dPdtime;
    float3 Ps, dPsdx, dPsdy;
    void* renderstate;
    void* tracedata;
    void* objdata;
    void* context;
    void* shadingStateUniform;
    int thread_index;
    int shade_index;
    void* renderer;
    void* object2common;
    void* shader2common;
    void* Ci;
    float surfacearea;
    int raytype;
    int flipHandedness;
    int backfacing;
    int shaderID;
};
}

}  // anonymous namespace
