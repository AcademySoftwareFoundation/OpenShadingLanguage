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
    float3 up;
    float fov;
    int aa;
    int max_bounces;
    float show_albedo_scale;
    bool no_jitter;

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

    uint64_t num_spheres;
    uint64_t num_quads;
    CUdeviceptr spheres_buffer;
    CUdeviceptr quads_buffer;

    // for the background
    int bg_res;
    int bg_id;
    CUdeviceptr bg_values;
    CUdeviceptr bg_rows;
    CUdeviceptr bg_cols;
};



struct PrimitiveParams {
    float a;  // area
    int32_t shaderID;
    int32_t objID;
    bool isLight;
};



struct SphereParams : PrimitiveParams {
    float3 c;  // center
    float r;   // radius
    float r2;  // radius ^2
};



struct QuadParams : PrimitiveParams {
    float3 p;
    float3 ex;
    float3 ey;
    float3 n;
    float eu;
    float ev;
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
