// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include <iostream>
#include <cmath>

#include "oslexec_pvt.h"
#include <OSL/genclosure.h>


OSL_NAMESPACE_ENTER
namespace pvt {



OSL_SHADEOP const ClosureColor *
osl_add_closure_closure (ShaderGlobals *sg,
                         const ClosureColor *a, const ClosureColor *b)
{
    if (a == NULL) return b;
    if (b == NULL) return a;
    return sg->context->closure_add_allot (a, b);
}


OSL_SHADEOP const ClosureColor *
osl_mul_closure_color (ShaderGlobals *sg, ClosureColor *a, const Color3 *w)
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
osl_mul_closure_float (ShaderGlobals *sg, ClosureColor *a, float w)
{
    if (a == NULL) return NULL;
    if (w == 0.0f) return NULL;
    if (w == 1.0f) return a;
    return sg->context->closure_mul_allot (w, a);
}


OSL_SHADEOP ClosureComponent *
osl_allocate_closure_component (ShaderGlobals *sg, int id, int size)
{
    return sg->context->closure_component_allot(id, size, Color3(1.0f));
}



OSL_SHADEOP ClosureColor *
osl_allocate_weighted_closure_component (ShaderGlobals *sg, int id, int size, const Color3 *w)
{
    if (w->x == 0.0f && w->y == 0.0f && w->z == 0.0f)
        return NULL;
    return sg->context->closure_component_allot(id, size, *w);
}

OSL_SHADEOP const char *
osl_closure_to_string (ShaderGlobals *sg, ClosureColor *c)
{
    // Special case for printing closures
    std::ostringstream stream;
    stream.imbue (std::locale::classic());  // force C locale
    print_closure(stream, c, &sg->context->shadingsys());
    return ustring(stream.str ()).c_str();
}


} // namespace pvt
OSL_NAMESPACE_EXIT
