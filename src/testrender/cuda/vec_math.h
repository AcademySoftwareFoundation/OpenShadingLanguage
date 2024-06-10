// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <OSL/dual_vec.h>

#include <vector_functions.h>
#include <vector_types.h>

#if !defined(__CUDACC_RTC__)
#include <cmath>
#include <cstdlib>
#endif

namespace {  // anonymous namespace

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

static __forceinline__ __device__
float3 cross(const float3& a, const float3& b)
{
  return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}


static __forceinline__ __device__
float length(const float3& v)
{
    return __fsqrt_rn((v.x * v.x) + (v.y * v.y) + (v.z * v.z));
}


static __forceinline__ __device__
void ortho(const float3& n, float3& x, float3& y)
{
    x = normalize(fabsf(n.x) > .01f ? make_float3(n.z, 0, -n.x) : make_float3(0, -n.z, n.y));
    y = cross(n, x);
}


}  // anonymous namespace

// Conversion macros for casting between vector types
#define F3_TO_V3(f3) (*reinterpret_cast<const OSL::Vec3*>(&f3))
#define F3_TO_C3(f3) (*reinterpret_cast<const OSL::Color3*>(&f3))
#define V3_TO_F3(v3) (*reinterpret_cast<const float3*>(&v3))
#define C3_TO_F3(c3) (*reinterpret_cast<const float3*>(&c3))
