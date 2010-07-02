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

#include <boost/foreach.hpp>

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


char*
ClosureColor::allocate_component (int id, size_t num_bytes)
{
    // FIXME: alignment ??

    // Resize memory to fit size of the new component
    m_mem.resize (num_bytes);

    // Make a new component
    m_components.clear();
    m_components.push_back (Component (id, Color3 (1, 1, 1), 0));

    // Return the block of memory for the caller to new the ClosurePrimitive into
    return &m_mem[0];
}


void
ClosureColor::add (const ClosureColor &A)
{
    // Look at all of A's components, decide which can be merged with our
    // own (just summing weights) and which need to be appended as new
    // closure primitives.
    int my_ncomponents = ncomponents();  // how many components I have now
    int num_unmerged = 0;                // how many more I'll need
    size_t new_bytes = 0;                // how much more mem I'll need
    int *unmerged = ALLOCA (int, A.ncomponents());  // temp index list
    for (int ac = 0;  ac < A.ncomponents();  ++ac) {
        const ClosurePrimitive *aprim (A.prim (ac));
        const Component &acomp (A.component (ac));
        if (acomp.weight[0] == 0.0f && acomp.weight[1] == 0.0f &&
                acomp.weight[2] == 0.0f)
            continue;   // don't bother adding a 0-weighted component
        bool merged = false;
        for (int c = 0;  c < my_ncomponents;  ++c) {
            if (prim(c)->name() == aprim->name() &&
                    prim(c)->mergeable (aprim)) {
                // We can merge with an existing component -- just add the
                // weights
                m_components[c].weight += acomp.weight;
                merged = true;
                break;
            }
        }
        if (! merged) {
            // Not a duplicate that can be merged.  Remember this component
            // index and how much memory it'll need.
            unmerged[num_unmerged++] = ac;
            new_bytes += aprim->memsize();
        }
    }

    // If we've merged everything and don't need to append, we're done
    if (! num_unmerged)
        return;

    // Grow our memory
    size_t oldmemsize = m_mem.size ();
    m_mem.resize (oldmemsize + new_bytes);

    // Append the components of A that we couldn't merge.
    for (int i = 0;  i < num_unmerged;  ++i) {
        int c = unmerged[i];   // next unmerged component index within A
        const Component &acomp (A.component (c));
        const ClosurePrimitive *aprim (A.prim (c));
        size_t asize = aprim->memsize();
        memcpy (&m_mem[oldmemsize], &A.m_mem[acomp.memoffset], asize);
        m_components.push_back (acomp);
        m_components.back().memoffset = oldmemsize;
        oldmemsize += asize;
    }
}



void
ClosureColor::add (const ClosureColor &A, const ClosureColor &B)
{
    if (this != &A)
        *this = A;
    add (B);
}


void
ClosureColor::mul (const Color3 &w)
{
    // Handle scale by 0 trivially
    if (w[0] == 0.0f && w[1] == 0.0f && w[2] == 0.0f) {
        clear();
        return;
    }

    // For every component, scale it
    BOOST_FOREACH (Component &c, m_components)
        c.weight *= w;
}



void
ClosureColor::mul (float w)
{
    // Handle scale by 0 trivially
    if (w == 0.0f) {
        clear();
        return;
    }

    // For every component, scale it
    BOOST_FOREACH (Component &c, m_components)
        c.weight *= w;
}



std::ostream &
operator<< (std::ostream &out, const ClosureColor &closure)
{
    for (int c = 0;  c < closure.ncomponents(); c++) {
        const Color3 &weight = closure.weight (c);
        const ClosurePrimitive *cprim = closure.prim (c);
        if (c)
            out << "\n\t+ ";
        out << "(" << weight[0] << ", " << weight[1] << ", " << weight[2] << ") * ";
        out << *cprim;
    }
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
