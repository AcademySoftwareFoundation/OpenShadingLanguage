// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <OSL/oslversion.h>

#ifdef OSL_USE_OPTIX
#    include <cuda_runtime_api.h>
#    include <optix.h>
#    ifdef _WIN32
#        define NOMINMAX
#    endif
#    if (OPTIX_VERSION < 70000)
#        include <optix_world.h>
#    endif
#else
#    include <stdlib.h>
#endif


OSL_NAMESPACE_ENTER

#ifdef OSL_USE_OPTIX

////////////////////////////////////////////////////////////////////////
// If OptiX is available, alias everything in optix:: namespace into
// OSL::optix::


// TODO clean this up once OptiX6 support is dropped
#    if (OPTIX_VERSION < 70000)
namespace optix = ::optix;
#    else
namespace optix {
typedef OptixDeviceContext Context;
typedef cudaTextureObject_t TextureSampler;
}  // namespace optix
#    endif

using ::cudaFree;
using ::cudaMalloc;


#else

////////////////////////////////////////////////////////////////////////
// If OptiX is not available, make reasonable proxies just so that we
// don't have to litter the code with quite as many #ifdef's.
// OSL::optix::

namespace optix {

typedef void* Context;
typedef void* Program;
typedef void* TextureSampler;
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
