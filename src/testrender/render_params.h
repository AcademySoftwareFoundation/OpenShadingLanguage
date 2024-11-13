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
    int show_globals;

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

    // geometry data
    CUdeviceptr triangles;
    CUdeviceptr verts;
    CUdeviceptr uvs;
    CUdeviceptr uv_indices;
    CUdeviceptr normals;
    CUdeviceptr normal_indices;
    CUdeviceptr shader_ids;
    CUdeviceptr shader_is_light;
    CUdeviceptr mesh_ids;
    CUdeviceptr surfacearea;
    CUdeviceptr lightprims;
    size_t lightprims_size;

    // for the background
    int bg_res;
    int bg_id;
    CUdeviceptr bg_values;
    CUdeviceptr bg_rows;
    CUdeviceptr bg_cols;
};



struct GenericData {
    // NB: This used to point to the geometry data for spheres and quads,
    //     but it is currently unused.
    void* data;
};



struct GenericRecord {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // What follows should duplicate GenericData
    void* data;
};

#endif
