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



OSL_SHADEOP OSL_HOSTDEVICE const void*
osl_add_closure_closure(OpaqueExecContextPtr oec, const void* a_,
                        const void* b_)
{
    const ClosureColor* a = (const ClosureColor*)a_;
    const ClosureColor* b = (const ClosureColor*)b_;
    if (a == NULL)
        return b;
    if (b == NULL)
        return a;
    ClosureAdd* add = (ClosureAdd*)rs_allocate_closure(oec, sizeof(ClosureAdd),
                                                       alignof(ClosureAdd));
    if (add) {
        add->id       = ClosureColor::ADD;
        add->closureA = a;
        add->closureB = b;
    }
    return add;
}


OSL_SHADEOP OSL_HOSTDEVICE const void*
osl_mul_closure_color(OpaqueExecContextPtr oec, const void* a_, const void* w_)
{
    const ClosureColor* a = (const ClosureColor*)a_;
    const Color3* w       = (const Color3*)w_;
    if (a == NULL)
        return NULL;
    if (w->x == 0.0f && w->y == 0.0f && w->z == 0.0f)
        return NULL;
    if (w->x == 1.0f && w->y == 1.0f && w->z == 1.0f)
        return a;
    ClosureMul* mul = (ClosureMul*)rs_allocate_closure(oec, sizeof(ClosureMul),
                                                       alignof(ClosureMul));
    if (mul) {
        mul->id      = ClosureColor::MUL;
        mul->weight  = *w;
        mul->closure = a;
    }
    return mul;
}


OSL_SHADEOP OSL_HOSTDEVICE const void*
osl_mul_closure_float(OpaqueExecContextPtr oec, const void* a_, float w)
{
    const ClosureColor* a = (const ClosureColor*)a_;
    if (a == NULL)
        return NULL;
    if (w == 0.0f)
        return NULL;
    if (w == 1.0f)
        return a;
    ClosureMul* mul = (ClosureMul*)rs_allocate_closure(oec, sizeof(ClosureMul),
                                                       alignof(ClosureMul));
    if (mul) {
        mul->id      = ClosureColor::MUL;
        mul->weight  = Color3(w);
        mul->closure = a;
    }
    return mul;
}


OSL_SHADEOP OSL_HOSTDEVICE void*
osl_allocate_closure_component(OpaqueExecContextPtr oec, int id, int size)
{
    // Allocate the component and the mul back to back
    const size_t needed = sizeof(ClosureComponent) + size;
    ClosureComponent* comp
        = (ClosureComponent*)rs_allocate_closure(oec, needed,
                                                 alignof(ClosureComponent));
    if (comp) {
        comp->id = id;
        comp->w  = Color3(1.0f);
    }
    return comp;
}



OSL_SHADEOP OSL_HOSTDEVICE void*
osl_allocate_weighted_closure_component(OpaqueExecContextPtr oec, int id,
                                        int size, const void* w_)
{
    const Color3* w = (const Color3*)w_;
    if (w->x == 0.0f && w->y == 0.0f && w->z == 0.0f)
        return NULL;
    // Allocate the component and the mul back to back
    const size_t needed = sizeof(ClosureComponent) + size;
    ClosureComponent* comp
        = (ClosureComponent*)rs_allocate_closure(oec, needed,
                                                 alignof(ClosureComponent));
    if (comp) {
        comp->id = id;
        comp->w  = *w;
    }
    return comp;
}

// Deprecated, remove when conversion from ustring to ustringhash is finished
OSL_SHADEOP const char*
osl_closure_to_string(OpaqueExecContextPtr oec, const void* c_)
{
    ShaderGlobals* sg     = (ShaderGlobals*)oec;
    const ClosureColor* c = (const ClosureColor*)c_;
    // Special case for printing closures
    std::ostringstream stream;
    stream.imbue(std::locale::classic());  // force C locale
    print_closure(stream, c, &sg->context->shadingsys(),
                  /*treat_ustrings_as_hash*/ false);
    return ustring(stream.str()).c_str();
}

OSL_SHADEOP ustringhash_pod
osl_closure_to_ustringhash(OpaqueExecContextPtr oec, const void* c_)
{
    ShaderGlobals* sg     = (ShaderGlobals*)oec;
    const ClosureColor* c = (const ClosureColor*)c_;
    // Special case for printing closures
    std::ostringstream stream;
    stream.imbue(std::locale::classic());  // force C locale
    print_closure(stream, c, &sg->context->shadingsys(),
                  /*treat_ustrings_as_hash*/ true);
    return ustring(stream.str()).hash();
}



}  // namespace pvt
OSL_NAMESPACE_END
