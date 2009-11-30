/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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

#include "oslconfig.h"
#include "oslclosure.h"
#include "oslexec_pvt.h"



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
ClosureColor::allocate_component (size_t num_bytes)
{
    // FIXME: alignment ??

    // Resize memory to fit size of the new component
    m_mem.resize (num_bytes);

    // Make a new component
    m_components.clear();
    m_components.push_back (Component (Color3 (1, 1, 1), 0));

    // Return the block of memory for the caller to new the ClosurePrimitive into
    return &m_mem[0];
}


void
ClosureColor::add (const ClosureColor &A)
{
    // Grow our memory
    size_t num_bytes = A.m_mem.size ();
    size_t oldmemsize = m_mem.size ();
    m_mem.resize (oldmemsize + num_bytes);

    // Copy A's memory at the end of ours
    memcpy(&m_mem[oldmemsize], &A.m_mem[0], num_bytes);

    // Copy A's components and adjust memory offsets to refer to new position
    BOOST_FOREACH (const Component &c, A.m_components) {
        m_components.push_back(c);
        m_components.back().memoffset += oldmemsize;
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
    // For every component, scale it
    BOOST_FOREACH (Component &c, m_components)
        c.weight *= w;
}



void
ClosureColor::mul (float w)
{
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
const ustring Labels::TRANSMIT   = ustring("T");
const ustring Labels::REFLECT    = ustring("R");
const ustring Labels::DIFFUSE    = ustring("D");
const ustring Labels::GLOSSY     = ustring("G");
const ustring Labels::SINGULAR   = ustring("S");
const ustring Labels::STRAIGHT   = ustring("s");
const ustring Labels::STOP       = ustring("__stop__");


}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
