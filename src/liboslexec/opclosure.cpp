/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
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

#include <iostream>
#include <cmath>

#include "oslexec_pvt.h"
#include <OSL/genclosure.h>


OSL_NAMESPACE_ENTER
namespace pvt {



OSL_SHADEOP const ClosureColor *
OSL_LLVM_PREFIXED_TOKEN(osl_add_closure_closure) (ShaderGlobals *sg,
                                                  const ClosureColor *a,
                                                  const ClosureColor *b)
{
    if (a == NULL) return b;
    if (b == NULL) return a;
    return sg->context->closure_add_allot (a, b);
}


OSL_SHADEOP const ClosureColor *
OSL_LLVM_PREFIXED_TOKEN(osl_mul_closure_color) (ShaderGlobals *sg,
                                                ClosureColor *a,
                                                const Color3 *w)
{
    if (a == NULL) return NULL;
    if (w->x == 0.0f &&
        w->y == 0.0f &&
        w->z == 0.0f) return NULL;
    if (w->x == 1.0f &&
        w->y == 1.0f &&
        w->z == 1.0f) return a;
    return sg->context->closure_mul_allot (*w, a);
}


OSL_SHADEOP const ClosureColor *
OSL_LLVM_PREFIXED_TOKEN(osl_mul_closure_float) (ShaderGlobals *sg,
                                                ClosureColor *a,
                                                float w)
{
    if (a == NULL) return NULL;
    if (w == 0.0f) return NULL;
    if (w == 1.0f) return a;
    return sg->context->closure_mul_allot (w, a);
}


OSL_SHADEOP ClosureComponent *
OSL_LLVM_PREFIXED_TOKEN(osl_allocate_closure_component) (ShaderGlobals *sg,
                                                         int id,
                                                         int size)
{
    return sg->context->closure_component_allot(id, size, Color3(1.0f));
}



OSL_SHADEOP ClosureColor *
OSL_LLVM_PREFIXED_TOKEN(osl_allocate_weighted_closure_component) (ShaderGlobals *sg,
                                                                  int id,
                                                                  int size,
                                                                  const Color3 *w)
{
    if (w->x == 0.0f && w->y == 0.0f && w->z == 0.0f)
        return NULL;
    return sg->context->closure_component_allot(id, size, *w);
}

OSL_SHADEOP const char *
OSL_LLVM_PREFIXED_TOKEN(osl_closure_to_string) (ShaderGlobals *sg,
                                                ClosureColor *c)
{
    // Special case for printing closures
    std::stringstream stream;
    print_closure(stream, c, &sg->context->shadingsys());
    return ustring(stream.str ()).c_str();
}


} // namespace pvt
OSL_NAMESPACE_EXIT
