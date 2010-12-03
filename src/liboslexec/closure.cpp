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

#include "oslconfig.h"
#include "oslclosure.h"
#include "genclosure.h"
#include "oslexec_pvt.h"
#include "oslops.h"



#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {


std::ostream &
operator<< (std::ostream &out, const ClosurePrimitive &prim)
{
    // http://www.parashift.com/c++-faq-lite/input-output.html#faq-15.11
    prim.print_on(out);
    return out;
}

/*
void
ClosureColor::flatten (ClosureColor *closure, const Color3 &w, ShadingSystemImpl *ss)
{
    ClosureComponent *comp;

    if (closure == NULL)
        return;

    switch (closure->type) {
        case ClosureColor::CLOSURE_MUL:
            flatten((ClosureColor *)((ClosureMul *)closure)->closure, ((ClosureMul *)closure)->weight * w, ss);
            break;
        case ClosureColor::CLOSURE_ADD:
            flatten((ClosureColor *)((ClosureAdd *)closure)->closureA, w, ss);
            flatten((ClosureColor *)((ClosureAdd *)closure)->closureB, w, ss);
            break;
        case ClosureColor::CLOSURE_COMPONENT:
            comp = (ClosureComponent *)closure;
            comp->weight *= w;
            if (comp->weight[0] != 0.0f || comp->weight[1] != 0.0f || comp->weight[2] != 0.0f)
            {
                for (int i = 0; i < ncomponents(); ++i)
                {
                    ClosureComponent *existing = m_components[i];
                    if (existing->id != comp->id) continue;
                    const ClosureRegistry::ClosureEntry *closure = ss->find_closure(comp->id);
                    DASSERT(closure != NULL);
                    CompareClosureFunc compare = closure->compare;
                    if (compare ? compare(comp->id, comp->mem, existing->mem) : !memcmp(comp->mem, existing->mem, closure->struct_size))
                    {
                        existing->weight += comp->weight;
                        comp = NULL;
                        break;
                    }
                }
                if (comp)
                    push_component(comp);
            }
            break;
    }
}
*/




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
print_component_value(std::ostream &out, TypeDesc type, const void *data)
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
        out << "\"" << ((ustring *)data)->c_str() << "\"";
}



static void
print_component (std::ostream &out, const ClosureComponent *comp, ShadingSystemImpl *ss, const Color3 &weight)
{
    out << "(" << weight[0] << ", " << weight[1] << ", " << weight[2] << ") * ";
    const ClosureRegistry::ClosureEntry *clentry = ss->find_closure(comp->id);
    ASSERT(clentry);
    out << clentry->name.c_str() << " (";
    int i;
    for (i = 0; i < clentry->nformal; ++i) {
        if (i) out << ", ";
        if (clentry->params[i].type.numelements() > 1) out << "[";
        for (size_t j = 0; j < clentry->params[i].type.numelements(); ++j) {
            if (j) out << ", ";
            print_component_value(out, clentry->params[i].type.elementtype(),
                                  (const char *)comp->data() + clentry->params[i].offset
                                                             + clentry->params[i].type.elementsize() * j);
        }
        if (clentry->params[i].type.numelements() > 1) out << "]";
    }
    if (comp->nattrs) {
        const ClosureComponent::Attr * attrs = comp->attrs();
        for (int j = 0; j < comp->nattrs; ++j) {
            if (i || j) out << ", ";
            // find the type
            TypeDesc td;
            for (int p = 0; p < clentry->nkeyword; ++p)
                if (!strcmp(clentry->params[clentry->nformal + p].key, attrs[j].key.c_str()))
                    td = clentry->params[clentry->nformal + p].type;
            if (td != TypeDesc()) {
                out << "\"" << attrs[j].key.c_str() << "\", ";
                print_component_value(out, td, &attrs[j].value);
            }
        }
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



} // namespace pvt



void
print_closure (std::ostream &out, const ClosureColor *closure, ShadingSystemImpl *ss)
{
    bool first = true;
    print_closure(out, closure, ss, Color3(1, 1, 1), first);
}



}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
