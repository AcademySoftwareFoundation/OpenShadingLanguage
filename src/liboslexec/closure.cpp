// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include <vector>
#include <string>
#include <cstdio>

#include <OpenImageIO/sysutil.h>

#include <OSL/oslconfig.h>
#include <OSL/oslclosure.h>
#include <OSL/genclosure.h>
#include "oslexec_pvt.h"



OSL_NAMESPACE_ENTER

const ustring Labels::NONE       = ustring(NULL);
const ustring Labels::CAMERA     = ustring("C");
const ustring Labels::LIGHT      = ustring("L");
const ustring Labels::BACKGROUND = ustring("B");
const ustring Labels::VOLUME     = ustring("V");
const ustring Labels::OBJECT     = ustring("O");
const ustring Labels::TRANSMIT   = ustring("T");
const ustring Labels::REFLECT    = ustring("R");
const ustring Labels::DIFFUSE    = ustring("D");
const ustring Labels::GLOSSY     = ustring("G");
const ustring Labels::SINGULAR   = ustring("S");
const ustring Labels::STRAIGHT   = ustring("s");
const ustring Labels::STOP       = ustring("__stop__");



namespace pvt {



static void
print_component_value(std::ostream &out, ShadingSystemImpl *ss,
                      TypeDesc type, const void *data)

{
    if (type == TypeDesc::TypeInt)
        out << *(int *)data;
    else if (type == TypeDesc::TypeFloat)
        out << *(float *)data;
    else if (type == TypeDesc::TypeColor)
        out << "(" << ((Color3 *)data)->x << ", " << ((Color3 *)data)->y << ", " << ((Color3 *)data)->z << ")";
    else if (type == TypeDesc::TypeVector)
        out << "(" << ((Vec3 *)data)->x << ", " << ((Vec3 *)data)->y << ", " << ((Vec3 *)data)->z << ")";
    else if (type == TypeDesc::TypeString)
        out << "\"" << *((ustring *)data) << "\"";
    else if (type == TypeDesc::PTR)  // this only happens for closures
        print_closure (out, *(const ClosureColor **)data, ss);
}



static void
print_component (std::ostream &out, const ClosureComponent *comp, ShadingSystemImpl *ss, const Color3 &weight)
{
    const ClosureRegistry::ClosureEntry *clentry = ss->find_closure(comp->id);
    OSL_ASSERT(clentry);
    out << "(" << weight.x*comp->w.x << ", " << weight.y*comp->w.y << ", " << weight.z*comp->w.z << ") * ";
    out << clentry->name.c_str() << " (";
    for (int i = 0, nparams = clentry->params.size() - 1; i < nparams; ++i) {
        if (i) out << ", ";
        const ClosureParam& param = clentry->params[i];
        if (param.key != 0)
            out << "\"" << param.key << "\", ";
        if (param.type.numelements() > 1) out << "[";
        for (size_t j = 0; j < param.type.numelements(); ++j) {
            if (j) out << ", ";
            print_component_value(out, ss, param.type.elementtype(),
                                  (const char *)comp->data() + param.offset
                                                             + param.type.elementsize() * j);
        }
        if (clentry->params[i].type.numelements() > 1) out << "]";
    }
    out << ")";
}



static void
print_closure (std::ostream &out, const ClosureColor *closure, ShadingSystemImpl *ss, const Color3 &w, bool &first)
{
    if (closure == NULL)
        return;

    switch (closure->id) {
        case ClosureColor::MUL:
            print_closure(out, closure->as_mul()->closure, ss, closure->as_mul()->weight * w, first);
            break;
        case ClosureColor::ADD:
            print_closure(out, closure->as_add()->closureA, ss, w, first);
            print_closure(out, closure->as_add()->closureB, ss, w, first);
            break;
        default:
            if (!first)
                out << "\n\t+ ";
            print_component (out, closure->as_comp(), ss, w);
            first = false;
            break;
    }
}



void
print_closure (std::ostream &out, const ClosureColor *closure, ShadingSystemImpl *ss)
{
    bool first = true;
    print_closure(out, closure, ss, Color3(1, 1, 1), first);
}



} // namespace pvt



OSL_NAMESPACE_EXIT
