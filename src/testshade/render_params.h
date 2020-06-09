#pragma once

#if (OPTIX_VERSION >= 70000)
struct RenderParams
{
    float invw;
    float invh;
    CUdeviceptr output_buffer;
    bool flipv;
    CUdeviceptr osl_printf_buffer_start;
    CUdeviceptr osl_printf_buffer_end;
    CUdeviceptr color_system;                                                
    // for used-data tests
    uint64_t test_str_1;
    uint64_t test_str_2;
};
#endif

