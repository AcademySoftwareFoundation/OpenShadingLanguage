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
#include <OpenImageIO/hash.h>
#include <OpenImageIO/thread.h>

#include "oslconfig.h"
#include "oslclosure.h"
#include "oslexec_pvt.h"



namespace {

typedef hash_map<ustring, const OSL::ClosurePrimitive *, ustringHash> ClosurePrimMap;
ClosurePrimMap prim_map;
mutex closure_mutex;

};


namespace OSL {

//namespace pvt {   // OSL::pvt


// Define a null primitive used for error conditions
class NullClosure : public ClosurePrimitive {
public:
    NullClosure () : ClosurePrimitive (Strings::null, 0, ustring()) { }
};

static NullClosure nullclosure;



ClosurePrimitive::ClosurePrimitive (ustring name, int nargs, ustring argtypes)
    : m_name(name), m_nargs(nargs), m_argtypes(argtypes)
{
    ASSERT (name.length());
    // Base class ctr of a closure primitive registers it
    lock_guard guard (closure_mutex);
    ClosurePrimMap::const_iterator found = prim_map.find (m_name);
    ASSERT (found == prim_map.end());
    prim_map[m_name] = this;
    std::cerr << "Registered closure primitive '" << m_name << "'\n";
}



ClosurePrimitive::~ClosurePrimitive ()
{
    // Base class of a closure primitive registers it
    lock_guard guard (closure_mutex);
    ClosurePrimMap::iterator todelete = prim_map.find (m_name);
    ASSERT (todelete != prim_map.end() && todelete->second == this);
    prim_map.erase (todelete);
    std::cerr << "De-registered closure primitive '" << m_name << "'\n";
}



ClosureColor::compref_t
ClosureColor::primitive (ustring name)
{
    ClosurePrimMap::const_iterator found;
    {
        lock_guard guard (closure_mutex);
        found = prim_map.find (name);
    }
    if (found != prim_map.end())
        return new ClosureColorComponent (*found->second);
    // Oh no, not found!  Return a null primitive.
    return new ClosureColorComponent (nullclosure);
}



std::ostream &
operator<< (std::ostream &out, const ClosureColorComponent &comp)
{
    out << comp.m_cprim->name() << " (";
    ustring argtypes = comp.argtypes();
    int nextf = 0, nexts = 0;
    for (int i = 0;  i < comp.m_nargs;  ++i) {
        if (i)
            out << ", ";
        switch (argtypes[i]) {
        case 'f' :
            out << comp.m_fparams[nextf++];
            break;
        case 'p' :
        case 'v' :
        case 'n' :
        case 'c' :
            out << "(";
            out << comp.m_fparams[nextf++] << ", ";
            out << comp.m_fparams[nextf++] << ", ";
            out << comp.m_fparams[nextf++] << ")";
            break;
        case 'm' :
            out << "(";
            for (int m = 0;  m < 16;  ++m) {
                if (m)
                    out << ", ";
                out << comp.m_fparams[nextf++];
            }
            out << ")";
        case 's' :
            out << '\"' << comp.m_sparams[nexts++] << '\"';
            break;
        }
    }
    out << ")";
    return out;
}



void
ClosureColor::add (const compref_t &comp, const Color3 &weight)
{
    // See if this component is already present in us
    for (int m = 0;  m < m_ncomps;  ++m) {
        if (m_components[m] == comp || *m_components[m] == *comp) {
            // same primitive closure and same args
            m_weight[m] += weight;
            return;
        }
    }
    // But if we aren't adding to an existing component, add it now
    m_components.push_back (comp);
    m_weight.push_back (weight);
    ++m_ncomps;
}



void
ClosureColor::add (const ClosureColor &A)
{
    for (int a = 0;  a < A.m_ncomps;  ++a)
        add (A.m_components[a], A.m_weight[a]);
}



void
ClosureColor::add (const ClosureColor &A, const ClosureColor &B)
{
    if (this != &A)
        *this = A;
    add (B);
}



void
ClosureColor::sub (const ClosureColor &A)
{
    for (int a = 0;  a < A.m_ncomps;  ++a)
        add (A.m_components[a], -A.m_weight[a]);
}



void
ClosureColor::sub (const ClosureColor &A, const ClosureColor &B)
{
    if (this != &A)
        *this = A;
    sub (B);
}



void
ClosureColor::mul (const Color3 &w)
{
    // For every component, scale it
    for (int a = 0;  a < m_ncomps;  ++a)
        m_weight[a] *= w;
}



void
ClosureColor::mul (float w)
{
    // For every component, scale it
    for (int a = 0;  a < m_ncomps;  ++a)
        m_weight[a] *= w;
}



std::ostream &
operator<< (std::ostream &out, const ClosureColor &c)
{
    for (int i = 0;  i < c.m_ncomps;  ++i) {
        if (i)
            out << "\n\t+ ";
        out << "(" << c.m_weight[i][0] << ", "
            << c.m_weight[i][1] << ", " << c.m_weight[i][2] << ") * " 
            << *c.m_components[i];
    }
    return out;
}



//}; // namespace pvt
}; // namespace OSL
