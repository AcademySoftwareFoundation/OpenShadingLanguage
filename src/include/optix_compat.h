// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage


#pragma once

#include <OSL/oslversion.h>

#ifdef OSL_USE_OPTIX
#  include <cuda_runtime_api.h>
#  ifdef _WIN32
#    define NOMINMAX
#  endif
#  include <optix_world.h>
#else
#  include <stdlib.h>
#endif


OSL_NAMESPACE_ENTER

#ifdef OSL_USE_OPTIX

////////////////////////////////////////////////////////////////////////
// If OptiX is available, alias everything in optix:: namespace into
// OSL::optix::

namespace optix = ::optix;
using ::cudaMalloc;
using ::cudaFree;


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

inline void cudaMalloc (void** p, size_t s) { *p = malloc(s); }
inline void cudaFree (void* p) { free(p); }


#endif

OSL_NAMESPACE_EXIT
