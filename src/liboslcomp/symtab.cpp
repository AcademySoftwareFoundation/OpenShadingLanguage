/*****************************************************************************
 *
 *             Copyright (c) 2009 Sony Pictures Imageworks, Inc.
 *                            All rights reserved.
 *
 *  This material contains the confidential and proprietary information
 *  of Sony Pictures Imageworks, Inc. and may not be disclosed, copied or
 *  duplicated in any form, electronic or hardcopy, in whole or in part,
 *  without the express prior written consent of Sony Pictures Imageworks,
 *  Inc. This copyright notice does not imply publication.
 *
 *****************************************************************************/

#include <vector>
#include <string>
#include <fstream>
#include <cstdio>
#include <streambuf>

#include <boost/foreach.hpp>

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"

#include "oslcomp_pvt.h"
#include "ast.h"


namespace OSL {
namespace pvt {   // OSL::pvt


std::string
Symbol::mangled () const
{
    // FIXME: De-alias
    return scope() ? Strutil::format ("___%d_%s", scope(), m_name.c_str())
        : m_name.string();
}



std::string
StructSpec::mangled () const
{
    return scope() ? Strutil::format ("___%d_%s", scope(), m_name.c_str())
        : m_name.string();
}



Symbol *
SymbolTable::find (ustring name, Symbol *last) const
{
    recursive_lock_guard guard (m_mutex);  // thread safety
    ScopeTableStack::const_reverse_iterator scopelevel;
    scopelevel = m_scopetables.rbegin();
    if (last) {
        // We only want to match OUTSIDE the scope of 'last'.  So first
        // search for last.  Then advance to the next outer scope.
        for ( ;  scopelevel != m_scopetables.rend();  ++scopelevel) {
            ScopeTable::const_iterator s = scopelevel->find (name);
            if (s != scopelevel->end() && s->second == last) {
                ++scopelevel;
                break;
            }
        }
    }
    for ( ;  scopelevel != m_scopetables.rend();  ++scopelevel) {
        ScopeTable::const_iterator s = scopelevel->find (name);
        if (s != scopelevel->end())
            return s->second;
    }
    return NULL;  // not found
}



Symbol * 
SymbolTable::clash (ustring name) const
{
    recursive_lock_guard guard (m_mutex);  // thread safety
    Symbol *s = find (name);
    return (s && s->scope() == scopeid()) ? s : NULL;
}



void
SymbolTable::insert (Symbol *sym)
{
    recursive_lock_guard guard (m_mutex);  // thread safety
    DASSERT (sym != NULL);
    sym->scope (scopeid ());
    m_scopetables.back()[sym->name()] = sym;
    m_allsyms.push_back (sym);
}



int
SymbolTable::new_struct (ustring name)
{
    recursive_lock_guard guard (m_mutex);  // thread safety
    m_structs.push_back (new StructSpec (name, scopeid()));
    int structid = (int) m_structs.size() - 1;
    insert (new Symbol (name, TypeSpec ("",structid), Symbol::SymTypeType));
    return structid;
}



void
SymbolTable::add_struct_field (const TypeSpec &type, ustring name)
{
    recursive_lock_guard guard (m_mutex);  // thread safety
    m_structs.back()->add_field (type, name);
}



void
SymbolTable::push ()
{
    recursive_lock_guard guard (m_mutex);     // thread safety
    m_scopestack.push (m_scopeid);  // push old scope id on the scope stack
    m_scopeid = m_nextscopeid++;    // set to new scope id
    m_scopetables.resize (m_scopetables.size()+1); // push scope table
}



void
SymbolTable::pop ()
{
    recursive_lock_guard guard (m_mutex);  // thread safety
    m_scopetables.resize (m_scopetables.size()-1);
    ASSERT (! m_scopestack.empty());
    m_scopeid = m_scopestack.top ();
    m_scopestack.pop ();
}



void
SymbolTable::delete_syms ()
{
    recursive_lock_guard guard (m_mutex);  // thread safety
    for (SymbolList::iterator i = m_allsyms.begin(); i != m_allsyms.end(); ++i)
        delete (*i);
    m_allsyms.clear ();
    for (StructList::iterator i = m_structs.begin(); i != m_structs.end(); ++i)
        delete (*i);
    m_structs.clear ();
}




void
SymbolTable::print ()
{
    recursive_lock_guard guard (m_mutex);  // thread safety
    if (m_structs.size()) {
        std::cout << "Structure table:\n";
        int structid = 0;
        BOOST_FOREACH (const StructSpec * s, m_structs) {
            if (! s)
                continue;
            std::cout << "    " << structid << ": struct " << s->mangled();
            if (s->scope())
                std::cout << " (" << s->name() 
                          << " in scope " << s->scope() << ")";
            std::cout << " :\n";
            for (size_t i = 0;  i < s->numfields();  ++i) {
                const StructSpec::FieldSpec & f (s->field(i));
                std::cout << "\t" << f.name << " : " 
                          << f.type.string() << "\n";
            }
            ++structid;
        }
        std::cout << "\n";
    }
    std::cout << "Symbol table:\n";
    BOOST_FOREACH (const Symbol *s, m_allsyms) {
        if (s->is_structure())
            continue;
        std::cout << "\t" << s->mangled() << " : ";
        if (s->typespec().is_structure()) {
            std::cout << "struct " << s->typespec().structure() << " "
                      << m_structs[s->typespec().structure()]->name();
        } else {
            std::cout << s->typespec().string();
        }
        if (s->scope())
            std::cout << " (" << s->name() << " in scope " 
                      << s->scope() << ")";
        std::cout << "\n";
    }
    std::cout << "\n";
}



}; // namespace pvt
}; // namespace OSL
