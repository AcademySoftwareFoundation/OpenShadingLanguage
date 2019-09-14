#pragma once

#if (OPTIX_VERSION >= 70000)
struct RenderParams
{
    float invw;
    float invh;
    CUdeviceptr output_buffer;
    bool flipv;
};
#endif

