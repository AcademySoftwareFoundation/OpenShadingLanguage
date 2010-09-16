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

void
print_primitive (std::ostream &out, const ClosurePrimitive *cprim, const Color3 &weight)
{
    out << "(" << weight[0] << ", " << weight[1] << ", " << weight[2] << ") * ";
    out << *cprim;
}

void
print_closure (std::ostream &out, const ClosureColor *closure, const Color3 &w, bool &first)
{
    ClosureComponent *comp;
    if (closure == NULL)
        return;

    switch (closure->type) {
        case ClosureColor::CLOSURE_MUL:
            print_closure(out, ((ClosureMul *)closure)->closure, ((ClosureMul *)closure)->weight * w, first);
            break;
        case ClosureColor::CLOSURE_ADD:
            print_closure(out, ((ClosureAdd *)closure)->closureA, w, first);
            print_closure(out, ((ClosureAdd *)closure)->closureB, w, first);
            break;
        case ClosureColor::CLOSURE_COMPONENT:
            comp = (ClosureComponent *)closure;
            if (comp->id < NBUILTIN_CLOSURES)
            {
                const ClosurePrimitive *cprim = (const ClosurePrimitive *)comp->mem;
                if (!first)
                    out << "\n\t+ ";
                print_primitive (out, cprim, w);
                first = false;
            }
            break;
    }
}

std::ostream &
operator<< (std::ostream &out, const ClosureColor &closure)
{
    bool first = true;
    print_closure(out, &closure, Color3(1, 1, 1), first);
    return out;
}



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

bool write_closure_param(const TypeDesc &typedesc, void *data, int offset, int argidx, int idx,
                         ShadingExecution *exec, int nargs, const int *args)
{
    char *p = (char *)data + offset;
    size_t size = typedesc.size();
    if (argidx < nargs)
    {
        Symbol &sym = exec->sym (args[argidx]);
        TypeDesc t = sym.typespec().simpletype();
        // Treat both NORMAL and POINT as VECTOR for closure parameters
        if (t.vecsemantics == TypeDesc::NORMAL || t.vecsemantics == TypeDesc::POINT)
            t.vecsemantics = TypeDesc::VECTOR;
        if (!sym.typespec().is_closure() && !sym.typespec().is_structure() && t == typedesc)
        {
            char *source = (char *)sym.data() + sym.step() * idx;
            memcpy(p, source, size);
            return true;
        }
        else
            return false;
    }
    else // The compiler had already checked that this arg was optional
        return true;
}

} // namespace pvt

}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
