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
#include <memory>

#include <OpenImageIO/strutil.h>
#include <OpenImageIO/thread.h>

#include "oslexec_pvt.h"



OSL_NAMESPACE_ENTER

namespace pvt {   // OSL::pvt



vector<std::shared_ptr<StructSpec> > &
TypeSpec::struct_list ()
{
    static vector<std::shared_ptr<StructSpec> > m_structs;
    return m_structs;
}



TypeSpec::TypeSpec (const char *name, int structid, int arraylen)
    : m_simple(TypeDesc::UNKNOWN, arraylen), m_structure((short)structid),
      m_closure(false)
{
    if (m_structure == 0)
        m_structure = structure_id (name, true);
}



std::string
TypeSpec::string () const
{
    std::string str;
    if (is_closure() || is_closure_array()) {
        str += "closure color";
        if (is_unsized_array())
            str += "[]";
        else if (arraylength() > 0)
            str += Strutil::sprintf ("[%d]", arraylength());
    }
    else if (structure() > 0) {
        StructSpec *ss = structspec();
        if (ss)
            str += Strutil::sprintf ("struct %s", structspec()->name());
        else
            str += Strutil::sprintf ("struct %d", structure());
        if (is_unsized_array())
            str += "[]";
        else if (arraylength() > 0)
            str += Strutil::sprintf ("[%d]", arraylength());
    } else {
        str += simpletype().c_str();
    }
    return str;
}



const char *
TypeSpec::c_str () const
{
    ustring s (this->string());
    return s.c_str ();
}



int
TypeSpec::structure_id (const char *name, bool add)
{
    vector<std::shared_ptr<StructSpec> > & m_structs (struct_list());
    ustring n (name);
    for (int i = (int)m_structs.size()-1;  i > 0;  --i) {
        if (m_structs[i] && m_structs[i]->name() == n)
            return i;
    }
    if (add) {
        if (m_structs.size() >= 0x8000) {
            OSL_ASSERT(0 && "more struct id's than fit in a short!");
            return 0;
        }
        int id = new_struct (new StructSpec (n, 0));
        return id;
    }
    return 0;   // Not found, not added
}



int
TypeSpec::new_struct (StructSpec *n)
{
    vector<std::shared_ptr<StructSpec> > & m_structs (struct_list());
    if (m_structs.size() == 0)
        m_structs.resize (1);   // Allocate an empty one
    m_structs.push_back (std::shared_ptr<StructSpec>(n));
    return (int)m_structs.size()-1;
}



bool
equivalent (const StructSpec *a, const StructSpec *b)
{
    OSL_DASSERT (a && b);
    if (a->numfields() != b->numfields())
        return false;
    for (size_t i = 0;  i < (size_t)a->numfields();  ++i)
        if (! equivalent (a->field(i).type, b->field(i).type))
            return false;
    return true;
}



bool
equivalent (const TypeSpec &a, const TypeSpec &b)
{
    // The two complex types are equivalent if...
    // they are actually identical (duh)
    if (a == b)
        return true;
    // or if they are structs, and the structs are equivalent
    if (a.is_structure() || b.is_structure()) {
        return a.is_structure() && b.is_structure() &&
               a.structspec()->name() == b.structspec()->name() &&
               equivalent(a.structspec(), b.structspec());
    }
    // or if the underlying simple types are equivalent
    return
        ((a.is_vectriple_based() && b.is_vectriple_based()) || equivalent(a.m_simple, b.m_simple))
         // ... and either both or neither are closures
         && a.is_closure() == b.is_closure()
         // ... and, if arrays, they are the same length, or both unsized,
         //     or one is unsized and the other isn't
         && (a.m_simple.arraylen == b.m_simple.arraylen ||
             a.is_unsized_array() != b.is_unsized_array());
}



}; // namespace pvt
OSL_NAMESPACE_EXIT
