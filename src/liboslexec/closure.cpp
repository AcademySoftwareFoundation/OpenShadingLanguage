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

#include <vector>
#include <string>
#include <cstdio>

#include <OpenImageIO/dassert.h>
#include <OpenImageIO/sysutil.h>

#include "OSL/oslconfig.h"
#include "OSL/oslclosure.h"
#include "OSL/genclosure.h"
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
        out << "\"" << ((ustring *)data)->string() << "\"";
    else if (type == TypeDesc::PTR)  // this only happens for closures
        print_closure (out, *(const ClosureColor **)data, ss);
}



static void
print_component (std::ostream &out, const ClosureComponent *comp, ShadingSystemImpl *ss, const Color3 &weight)
{
    out << "(" << weight[0]*comp->w[0] << ", " << weight[1]*comp->w[1] << ", " << weight[2]*comp->w[2] << ") * ";
    const ClosureRegistry::ClosureEntry *clentry = ss->find_closure(comp->id);
    ASSERT(clentry);
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
    ClosureComponent *comp;
    if (closure == NULL)
        return;

    switch (closure->type) {
        case ClosureColor::MUL:
            print_closure(out, ((ClosureMul *)closure)->closure, ss, ((ClosureMul *)closure)->weight * w, first);
            break;
        case ClosureColor::ADD:
            print_closure(out, ((ClosureAdd *)closure)->closureA, ss, w, first);
            print_closure(out, ((ClosureAdd *)closure)->closureB, ss, w, first);
            break;
        case ClosureColor::COMPONENT:
            comp = (ClosureComponent *)closure;
            if (!first)
                out << "\n\t+ ";
            print_component (out, comp, ss, w);
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
