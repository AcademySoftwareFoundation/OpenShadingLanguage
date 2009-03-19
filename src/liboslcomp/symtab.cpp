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

#include <boost/foreach.hpp>

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"

#include "oslcomp_pvt.h"
#include "ast.h"


namespace OSL {
namespace pvt {   // OSL::pvt


std::string
TypeSpec::string () const
{
    std::string str;
    if (is_structure())
        str = Strutil::format ("struct %d", structure());
    else {
        // Substitute some special names
        if (m_simple == TypeDesc::TypeColor)
            str = "color";
        else if (m_simple == TypeDesc::TypePoint)
            str = "point";
        else if (m_simple == TypeDesc::TypeVector)
            str = "vector";
        else if (m_simple == TypeDesc::TypeNormal)
            str = "normal";
        else if (m_simple == TypeDesc::TypeMatrix)
            str = "matrix";
        else
            str = simpletype().c_str();
    }
    if (is_closure())
        str += " closure";
    return str;
}



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
    Symbol *s = find (name);
    return (s && s->scope() == scopeid()) ? s : NULL;
}



void
SymbolTable::insert (Symbol *sym)
{
    DASSERT (sym != NULL);
    sym->scope (scopeid ());
    m_scopetables.back()[sym->name()] = sym;
    m_allsyms.push_back (sym);
}



int
SymbolTable::new_struct (ustring name)
{
    m_structs.push_back (new StructSpec (name, scopeid()));
    int structid = (int) m_structs.size() - 1;
    insert (new Symbol (name, TypeSpec ("",structid), Symbol::SymTypeType));
    return structid;
}



void
SymbolTable::add_struct_field (const TypeSpec &type, ustring name)
{
    m_structs.back()->add_field (type, name);
}



void
SymbolTable::push ()
{
    m_scopestack.push (m_scopeid);  // push old scope id on the scope stack
    m_scopeid = m_nextscopeid++;    // set to new scope id
    m_scopetables.resize (m_scopetables.size()+1); // push scope table
}



void
SymbolTable::pop ()
{
    m_scopetables.resize (m_scopetables.size()-1);
    ASSERT (! m_scopestack.empty());
    m_scopeid = m_scopestack.top ();
    m_scopestack.pop ();
}



void
SymbolTable::delete_syms ()
{
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
        if (s->is_structure()) {
            std::cout << "struct " << s->typespec().structure() << " "
                      << m_structs[s->typespec().structure()]->name();
        } else {
            std::cout << s->typespec().string();
        }
        if (s->scope())
            std::cout << " (" << s->name() << " in scope " 
                      << s->scope() << ")";
        if (s->is_function()) {
            const FunctionSymbol *f = (const FunctionSymbol *) s;
            const char *args = f->argcodes().c_str();
            int advance = 0;
            TypeSpec rettype = m_comp.type_from_code (args, &advance);
            args += advance;
            std::cout << " function (" << m_comp.typelist_from_code(args) << ") ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}



}; // namespace pvt
}; // namespace OSL
