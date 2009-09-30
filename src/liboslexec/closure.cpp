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



/// Grab the first type from 'code' (an encoded arg type string), return
/// the TypeDesc corresponding to it, and if advance the code pointer to
/// the next type.
static TypeDesc
typedesc_from_code (const char * &codestart)
{
    const char *code = codestart;
    TypeDesc t;
    switch (*code) {
    case 'i' : t = TypeDesc::TypeInt;          break;
    case 'f' : t = TypeDesc::TypeFloat;        break;
    case 'c' : t = TypeDesc::TypeColor;        break;
    case 'p' : t = TypeDesc::TypePoint;        break;
    case 'v' : t = TypeDesc::TypeVector;       break;
    case 'n' : t = TypeDesc::TypeNormal;       break;
    case 'm' : t = TypeDesc::TypeMatrix;       break;
    case 's' : t = TypeDesc::TypeString;       break;
    case 'x' : t = TypeDesc (TypeDesc::NONE);  break;
    default:
        std::cerr << "Don't know how to decode type code '" 
                  << code << "' " << (int)(*code) << "\n";
        ASSERT (0);   // FIXME
        ++codestart;
        return TypeDesc();
    }
    ++code;

    if (*code == '[') {
        ++code;
        t.arraylen = -1;   // signal arrayness, unknown length
        if (isdigit (*code)) {
            t.arraylen = atoi (code);
            while (isdigit (*code))
                ++code;
            if (*code == ']')
                ++code;
        }
    }

    codestart = code;
    return t;
}




ClosurePrimitive::ClosurePrimitive (const char *name, const char *argtypes,
                                    int category)
    : m_name(name), m_category((Category)category),
      m_nargs(0), m_argcodes(argtypes)
{
    ASSERT (m_name.length());
    // Base class ctr of a closure primitive registers it
    lock_guard guard (closure_mutex);
    ClosurePrimMap::const_iterator found = prim_map.find (m_name);
    ASSERT (found == prim_map.end());
    prim_map[m_name] = this;


    m_argmem = 0;
    for (const char *code = m_argcodes.c_str();  code && *code; ) {
        // Grab the next type code.  This automatically advances code!
        TypeDesc t = typedesc_from_code (code);

        // Add that type to our type list
        m_argtypes.push_back (t);

        // Round up the mem used if this type needs particular alignment
        if (t.basetype == TypeDesc::STRING) {
            // strings are really pointers that need to be aligned
            m_argmem = (m_argmem + sizeof(char *) - 1) & (~ sizeof(char *));
        }

        // Add the offset for this argument = mem used so far
        m_argoffsets.push_back (m_argmem);

        // Account for mem used for this argument
        m_argmem += t.size ();

        ++m_nargs;
    }

    std::cerr << "Registered closure primitive '" << m_name << "'\n";
    std::cerr << "   " << m_nargs << " arguments : " << m_argcodes << "\n";
    std::cerr << "   needs " << m_argmem << " bytes for arguments\n";
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



const ClosurePrimitive *
ClosurePrimitive::primitive (ustring name)
{
    ClosurePrimMap::const_iterator found;
    {
        lock_guard guard (closure_mutex);
        found = prim_map.find (name);
    }
    if (found != prim_map.end())
        return found->second;
    // Oh no, not found!  Return NULL;
    return NULL;
}



void
ClosureColor::add_component (const ClosurePrimitive *cprim,
                             const Color3 &weight, const void *params)
{
    // Make a new component
    m_components.push_back (Component (cprim, weight));
    Component &newcomp (m_components.back ());

    // Grow our memory
    size_t oldmemsize = m_mem.size ();
    newcomp.memoffset = oldmemsize;
    m_mem.resize (oldmemsize + cprim->argmem ());

    // Copy the params, if supplied
    if (params)
        memcpy (&m_mem[oldmemsize], params, cprim->argmem ());
}



void
ClosureColor::add (const ClosureColor &A)
{
    BOOST_FOREACH (const Component &Acomp, A.m_components)
        add_component (Acomp.cprim, Acomp.weight, &A.m_mem[Acomp.memoffset]);
}



void
ClosureColor::add (const ClosureColor &A, const ClosureColor &B)
{
    if (this != &A)
        *this = A;
    add (B);
}



#if 0
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
#endif



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
    for (size_t c = 0;  c < closure.m_components.size();  ++c) {
        const ClosureColor::Component &comp (closure.m_components[c]);
        const ClosurePrimitive *cprim = comp.cprim;
        if (c)
            out << "\n\t+ ";
        out << "(" << comp.weight[0] << ", "
            << comp.weight[1] << ", " << comp.weight[2] << ") * ";
        out << cprim->name() << " (";

        for (int a = 0;  a < comp.nargs;  ++a) {
            const char *data = &closure.m_mem[comp.memoffset] + cprim->argoffset(a);
            TypeDesc t = cprim->argtype (a);
            if (a)
                out << ", ";
            if (t.aggregate != TypeDesc::SCALAR || t.arraylen)
                out << "(";
            int n = t.numelements() * (int)t.aggregate;
            for (int i = 0;  i < n;  ++i) {
                if (i)
                    out << ", ";
                if (t.basetype == TypeDesc::FLOAT) {
                    out << ((const float *)data)[i];
                } else if (t.basetype == TypeDesc::INT) {
                    out << ((const int *)data)[i];
                } else if (t.basetype == TypeDesc::STRING) {
                    out << '\"' << ((const char **)data)[i] << '\"';
                }
            }
            if (t.aggregate != TypeDesc::SCALAR || t.arraylen)
                out << ")";
        }

        out << ")";
    }
    return out;
}



//}; // namespace pvt
}; // namespace OSL
