// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/hashes.h>
#include <OSL/oslconfig.h>

// All the the state free functions in rs_simplerend.cpp will need to do their job
// NOTE:  Additional data is here that will be used by rs_simplerend.cpp in future PR's
//        procedurally generating ShaderGlobals.
struct RenderContext {
    int xres;
    int yres;
    OSL::Matrix44 world_to_camera;
    OSL::ustringhash projection;
    float pixelaspect;
    float screen_window[4];
    float shutter[2];
    float fov;
    float hither;
    float yon;
    void* journal_buffer;
};

class StackClosurePool {
    static constexpr size_t capacity = 1024;
    alignas(8) char buffer[capacity];
    void* ptr;

public:
    OSL_HOSTDEVICE
    StackClosurePool() { reset(); }

    OSL_HOSTDEVICE
    void reset()
    {
        ptr        = &buffer[0];
        *(int*)ptr = 0;
    }

    OSL_HOSTDEVICE
    void* allocate(size_t size, size_t alignment)
    {
        uintptr_t p = OIIO::round_to_multiple_of_pow2((uintptr_t)ptr,
                                                      alignment);
        ptr         = (void*)(p + size);
        if (ptr <= &buffer[capacity])
            return (void*)p;
        assert(false);
        return nullptr;
    }
};

struct RenderState {
    RenderContext* context;
    StackClosurePool* closure_pool;
};
