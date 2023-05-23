// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <OSL/oslconfig.h>
#include <OSL/oslversion.h>

#if OSL_USE_OPTIX
#    include <cuda_runtime_api.h>
#    include <optix.h>
#    ifdef _WIN32
#        define NOMINMAX
#    endif
#else
#    include <stdlib.h>
#endif

#include <OSL/device_ptr.h>

#if !OSL_USE_OPTIX && !defined(__CUDA_ARCH__)
using CUdeviceptr = void*;
using float3      = OSL::Vec3;
#endif



OSL_NAMESPACE_ENTER

#if OSL_USE_OPTIX

////////////////////////////////////////////////////////////////////////
// If OptiX is available, alias everything in optix:: namespace into
// OSL::optix::


// TODO clean this up once OptiX6 support is dropped
namespace optix {
typedef OptixDeviceContext Context;
typedef cudaTextureObject_t TextureSampler;
using ::CUdeviceptr;
using ::float3;
}  // namespace optix

using ::cudaFree;
using ::cudaMalloc;


#else

////////////////////////////////////////////////////////////////////////
// If OptiX is not available, make reasonable proxies just so that we
// don't have to litter the code with quite as many #ifdef's.
// Mostly under OSL::optix::



namespace optix {

typedef void* Context;
typedef void* Program;
typedef void* TextureSampler;
typedef void* CUdeviceptr;
using float3 = OSL::Vec3;


struct Exception {
    static const char* what() { return "OSL compiled without Optix."; }
};

}  // end namespace optix

inline void
cudaMalloc(void** p, size_t s)
{
    *p = malloc(s);
}
inline void
cudaFree(void* p)
{
    free(p);
}


#endif

OSL_NAMESPACE_EXIT
