// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once


#include "optix_compat.h"

#if OSL_USE_OPTIX || defined(__CUDA_ARCH__)

struct RenderParams {
    float3 bad_color;
    float3 bg_color;

    float3 eye;
    float3 dir;
    float3 cx;
    float3 cy;
    int    aa;
    float  show_albedo_scale;

    float invw;
    float invh;

    CUdeviceptr traversal_handle;
    CUdeviceptr output_buffer;
    CUdeviceptr osl_printf_buffer_start;
    CUdeviceptr osl_printf_buffer_end;
    CUdeviceptr color_system;
    CUdeviceptr interactive_params;

    // for transforms
    CUdeviceptr object2common;
    CUdeviceptr shader2common;
    uint64_t num_named_xforms;
    CUdeviceptr xform_name_buffer;
    CUdeviceptr xform_buffer;

    // for used-data tests
    uint64_t test_str_1;
    uint64_t test_str_2;

    uint64_t    num_quads;
    uint64_t    num_spheres;
    CUdeviceptr quads_buffer;
    CUdeviceptr spheres_buffer;
};



struct PrimitiveParams {
    float a;  // area
    unsigned int shaderID;
    bool isLight;
};



struct SphereParams : PrimitiveParams {
    float3 c;  // center
    float r;   // radius
    float r2;  // radius ^2

    OSL_HOSTDEVICE float shapepdf(const OSL::Vec3& x, const OSL::Vec3& /*p*/) const
    {
        const float TWOPI = float(2 * M_PI);
        OSL::Vec3 C(c.x, c.y, c.z);
        float cmax2       = 1 - r2 / (C - x).length2();
        float cmax        = cmax2 > 0 ? sqrtf(cmax2) : 0;
        return 1 / (TWOPI * (1 - cmax));
    }
};



struct QuadParams : PrimitiveParams {
    float3 p;
    float3 ex;
    float3 ey;
    float3 n;
    float eu;
    float ev;

    OSL_HOSTDEVICE float shapepdf(const OSL::Vec3& x, const OSL::Vec3& p) const
    {
        OSL::Vec3 l = OSL::Vec3(p.x, p.y, p.z) - OSL::Vec3(x.x, x.y, x.z);
        float d2    = l.length2();
        OSL::Vec3 dir = l.normalize();
        OSL::Vec3 N   = OSL::Vec3(n.x, n.y, n.z);
        return d2 / (a * fabsf(dir.dot(N)));
    }
};



struct GenericData {
    // For geometry hit callables, data is the pointer to the array of
    // primitive params for that primitive type, and sbtGeoIndex is the index
    // for this primitive.
    void* data;
    unsigned int sbtGeoIndex;
};



struct GenericRecord {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // What follows should duplicate GenericData
    void* data;
    unsigned int sbtGeoIndex;
};

#endif
