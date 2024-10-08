// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/device_string.h>

OSL_NAMESPACE_ENTER
// Create an OptiX variable for each of the 'standard' strings declared in
// <OSL/strdecls.h>.
namespace DeviceStrings {
#define STRDECL(str, var_name) \
    extern __device__ OSL_NAMESPACE::DeviceString var_name;

#include <OSL/strdecls.h>
#undef STRDECL
}  // namespace DeviceStrings
OSL_NAMESPACE_EXIT

#include "closures.h"

namespace {  // anonymous namespace

// This isn't really a CUDA version of the host-side ShadingContext class;
// instead, it is used as a container for a handful of pointers accessed during
// shader executions that are accessed via the ShadingContext.
struct ShadingContextCUDA {};

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

enum RayType {
    CAMERA       = 1,
    SHADOW       = 2,
    REFLECTION   = 4,
    REFRACTION   = 8,
    DIFFUSE      = 16,
    GLOSSY       = 32,
    SUBSURFACE   = 64,
    DISPLACEMENT = 128
};

// ========================================
//
// Some helper vector functions
//
static __forceinline__ __device__ float3
operator*(const float a, const float3& b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}

static __forceinline__ __device__ float3
operator*(const float3& a, const float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

static __forceinline__ __device__ float3
operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3
operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __device__ float3
operator-(const float3& a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

static __forceinline__ __device__ float
dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float3
normalize(const float3& v)
{
    float invLen = 1.0f / sqrtf(dot(v, v));
    return invLen * v;
}
//
// ========================================

}  // anonymous namespace
