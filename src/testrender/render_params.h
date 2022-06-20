#pragma once


#include "optix_compat.h"

#if defined(OSL_USE_OPTIX) || defined(__CUDA_ARCH__)

struct RenderParams {
    float3 bad_color;
    float3 bg_color;

    float3 eye;
    float3 dir;
    float3 cx;
    float3 cy;

    float invw;
    float invh;

    CUdeviceptr traversal_handle;
    CUdeviceptr output_buffer;
    CUdeviceptr osl_printf_buffer_start;
    CUdeviceptr osl_printf_buffer_end;
    CUdeviceptr color_system;

    // for transforms
    CUdeviceptr object2common;
    CUdeviceptr shader2common;
    uint64_t num_named_xforms;
    CUdeviceptr xform_name_buffer;
    CUdeviceptr xform_buffer;

    // for used-data tests
    uint64_t test_str_1;
    uint64_t test_str_2;
};



struct PrimitiveParams {
    float a;  // area
    unsigned int shaderID;
};



struct SphereParams : PrimitiveParams {
    float3 c;  // center
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
    void* data;
    unsigned int sbtGeoIndex;
};



struct GenericRecord {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

    void* data;
    unsigned int sbtGeoIndex;
};

#endif
