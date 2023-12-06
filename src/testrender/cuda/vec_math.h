// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

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
//
// ========================================

// Define some vector operations using the single-precision "round up"
// intrinsics.
//
// In some cases (e.g., the sphere intersection), using these
// intrinsics can help match the CPU results more closely, especially
// when fast-math is used for the GPU code.

static __forceinline__ __device__ float
dot_ru(const float3& a, const float3& b)
{
    float val = __fadd_ru(__fmul_ru(a.x, b.x), __fmul_ru(a.y, b.y));
    return __fadd_ru(val, __fmul_ru(a.z, b.z));
}

static __forceinline__ __device__ float3
add_ru(const float3& a, const float3& b)
{
    return make_float3(__fadd_ru(a.x, b.x), __fadd_ru(a.y, b.y),
                       __fadd_ru(a.z, b.z));
}

static __forceinline__ __device__ float3
sub_ru(const float3& a, const float3& b)
{
    return make_float3(__fsub_ru(a.x, b.x), __fsub_ru(a.y, b.y),
                       __fsub_ru(a.z, b.z));
}

}  // anonymous namespace
