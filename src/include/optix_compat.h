/*
Copyright (c) 2019 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOTSS
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#pragma once

#include <OSL/oslversion.h>

#ifdef OSL_USE_OPTIX
#  include <cuda_runtime_api.h>
#  ifdef WIN32
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
