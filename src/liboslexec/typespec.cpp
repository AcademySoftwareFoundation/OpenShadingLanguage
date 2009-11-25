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

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"

#include "oslexec_pvt.h"




#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {

namespace pvt {   // OSL::pvt



std::vector<shared_ptr<StructSpec> > &
TypeSpec::struct_list ()
{
    static std::vector<shared_ptr<StructSpec> > m_structs;
    return m_structs;
}



TypeSpec::TypeSpec (const char *name, int structid, int arraylen)
    : m_simple(TypeDesc::UNKNOWN, arraylen), m_structure((short)structid),
      m_closure(false)
{
    if (m_structure == 0)
        m_structure = structure_id (name, true);
}



int
TypeSpec::structure_id (const char *name, bool add)
{
    std::vector<shared_ptr<StructSpec> > & m_structs (struct_list());
    ustring n (name);
    for (int i = (int)m_structs.size()-1;  i > 0;  --i) {
        ASSERT ((int)m_structs.size() > i);
        if (m_structs[i] && m_structs[i]->name() == n)
            return i;
    }
    if (add) {
        ASSERT (m_structs.size() < 0x8000 && "more struct id's than fit in a short!");
        int id = new_struct (new StructSpec (n, 0));
        return id;
    }
    return 0;   // Not found, not added
}



int
TypeSpec::new_struct (StructSpec *n)
{
    std::vector<shared_ptr<StructSpec> > & m_structs (struct_list());
    if (m_structs.size() == 0)
        m_structs.resize (1);   // Allocate an empty one
    m_structs.push_back (shared_ptr<StructSpec>(n));
    return (int)m_structs.size()-1;
}



bool
equivalent (const StructSpec *a, const StructSpec *b)
{
    ASSERT (a && b);
    if (a->numfields() != b->numfields())
        return false;
    for (size_t i = 0;  i < a->numfields();  ++i)
        if (! equivalent (a->field(i).type, b->field(i).type))
            return false;
    return true;
}



bool
equivalent (const TypeSpec &a, const TypeSpec &b)
{
    return (a == b) || 
        (a.is_vectriple_based() && b.is_vectriple_based() &&
         a.is_closure() == b.is_closure() &&
         a.arraylength() == b.arraylength()) ||
        (a.is_structure() && b.is_structure() &&
         equivalent(a.structspec(), b.structspec()));
}



}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
