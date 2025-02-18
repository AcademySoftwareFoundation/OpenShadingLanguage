// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <cmath>
#include <iostream>

#include "oslexec_pvt.h"
#include <OSL/genclosure.h>
#include <OSL/rs_free_function.h>


OSL_NAMESPACE_BEGIN
namespace pvt {



OSL_SHADEOP OSL_HOSTDEVICE const ClosureColor*
osl_add_closure_closure(ShaderGlobals* sg, const ClosureColor* a,
                        const ClosureColor* b)
{
    if (a == NULL)
        return b;
    if (b == NULL)
        return a;
    ClosureAdd* add = (ClosureAdd*)rs_allocate_closure(sg, sizeof(ClosureAdd),
                                                       alignof(ClosureAdd));
    if (add) {
        add->id       = ClosureColor::ADD;
        add->closureA = a;
        add->closureB = b;
    }
    return add;
}


OSL_SHADEOP OSL_HOSTDEVICE const ClosureColor*
osl_mul_closure_color(ShaderGlobals* sg, const ClosureColor* a, const Color3* w)
{
    if (a == NULL)
        return NULL;
    if (w->x == 0.0f && w->y == 0.0f && w->z == 0.0f)
        return NULL;
    if (w->x == 1.0f && w->y == 1.0f && w->z == 1.0f)
        return a;
    ClosureMul* mul = (ClosureMul*)rs_allocate_closure(sg, sizeof(ClosureMul),
                                                       alignof(ClosureMul));
    if (mul) {
        mul->id      = ClosureColor::MUL;
        mul->weight  = *w;
        mul->closure = a;
    }
    return mul;
}


OSL_SHADEOP OSL_HOSTDEVICE const ClosureColor*
osl_mul_closure_float(ShaderGlobals* sg, const ClosureColor* a, float w)
{
    if (a == NULL)
        return NULL;
    if (w == 0.0f)
        return NULL;
    if (w == 1.0f)
        return a;
    ClosureMul* mul = (ClosureMul*)rs_allocate_closure(sg, sizeof(ClosureMul),
                                                       alignof(ClosureMul));
    if (mul) {
        mul->id      = ClosureColor::MUL;
        mul->weight  = Color3(w);
        mul->closure = a;
    }
    return mul;
}


OSL_SHADEOP OSL_HOSTDEVICE ClosureComponent*
osl_allocate_closure_component(ShaderGlobals* sg, int id, int size)
{
    // Allocate the component and the mul back to back
    const size_t needed = sizeof(ClosureComponent) + size;
    ClosureComponent* comp
        = (ClosureComponent*)rs_allocate_closure(sg, needed,
                                                 alignof(ClosureComponent));
    if (comp) {
        comp->id = id;
        comp->w  = Color3(1.0f);
    }
    return comp;
}



OSL_SHADEOP OSL_HOSTDEVICE ClosureColor*
osl_allocate_weighted_closure_component(ShaderGlobals* sg, int id, int size,
                                        const Color3* w)
{
    if (w->x == 0.0f && w->y == 0.0f && w->z == 0.0f)
        return NULL;
    // Allocate the component and the mul back to back
    const size_t needed = sizeof(ClosureComponent) + size;
    ClosureComponent* comp
        = (ClosureComponent*)rs_allocate_closure(sg, needed,
                                                 alignof(ClosureComponent));
    if (comp) {
        comp->id = id;
        comp->w  = *w;
    }
    return comp;
}

// Deprecated, remove when conversion from ustring to ustringhash is finished
OSL_SHADEOP const char*
osl_closure_to_string(ShaderGlobals* sg, ClosureColor* c)
{
    // Special case for printing closures
    std::ostringstream stream;
    stream.imbue(std::locale::classic());  // force C locale
    print_closure(stream, c, &sg->context->shadingsys(),
                  /*treat_ustrings_as_hash*/ false);
    return ustring(stream.str()).c_str();
}

OSL_SHADEOP ustringhash_pod
osl_closure_to_ustringhash(ShaderGlobals* sg, ClosureColor* c)
{
    // Special case for printing closures
    std::ostringstream stream;
    stream.imbue(std::locale::classic());  // force C locale
    print_closure(stream, c, &sg->context->shadingsys(),
                  /*treat_ustrings_as_hash*/ true);
    return ustring(stream.str()).hash();
}



}  // namespace pvt
OSL_NAMESPACE_END
