// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>

#include <OSL/hashes.h>
#include <OSL/oslexec.h>

#include "../raytracer.h"


#define RAYTRACER_HIT_QUAD   0
#define RAYTRACER_HIT_SPHERE 1


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

// This isn't really a CUDA version of the host-side ShadingContext class;
// instead, it is used as a container for a handful of pointers accessed during
// shader executions that are accessed via the ShadingContext.
struct ShadingContextCUDA {};


namespace OSL_CUDA {
struct ShaderGlobals {
    OSL::Vec3 P, dPdx, dPdy;
    OSL::Vec3 dPdz;
    OSL::Vec3 I, dIdx, dIdy;
    OSL::Vec3 N;
    OSL::Vec3 Ng;
    float u, dudx, dudy;
    float v, dvdx, dvdy;
    OSL::Vec3 dPdu, dPdv;
    float time;
    float dtime;
    OSL::Vec3 dPdtime;
    OSL::Vec3 Ps, dPsdx, dPsdy;
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
};
}  // namespace OSL_CUDA

}  // anonymous namespace
