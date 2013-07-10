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

#include "oslcomp_pvt.h"

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
namespace Strutil = OIIO::Strutil;

#include <boost/foreach.hpp>


OSL_NAMESPACE_ENTER

namespace pvt {   // OSL::pvt


std::string
TypeSpec::string () const
{
    std::string str;
    if (is_closure() || is_closure_array()) {
        str += "closure color";
        if (arraylength() > 0)
            str += Strutil::format ("[%d]", arraylength());
        else if (arraylength() < 0)
            str += "[]";
    }
    else if (structure() > 0) {
        str += Strutil::format ("struct %d", structure());
        if (arraylength() > 0)
            str += Strutil::format ("[%d]", arraylength());
        else if (arraylength() < 0)
            str += "[]";
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



std::string
Symbol::mangled () const
{
    // FIXME: De-alias
    return scope() ? Strutil::format ("___%d_%s", scope(), m_name.c_str())
        : m_name.string();
}



const char *
Symbol::symtype_shortname (SymType s)
{
    ASSERT ((int)s >= 0 && (int)s < (int)SymTypeType);
    static const char *names[] = { "param", "oparam", "local", "temp",
                                   "global", "const", "func" };
    return names[(int)s];
}



std::string
StructSpec::mangled () const
{
    return scope() ? Strutil::format ("___%d_%s", scope(), m_name.c_str())
        : m_name.string();
}



const char *
Symbol::valuesourcename (ValueSource v)
{
    switch (v) {
    case DefaultVal   : return "default";
    case InstanceVal  : return "instance";
    case GeomVal      : return "geom";
    case ConnectedVal : return "connected";
    }
    ASSERT(0 && "unknown valuesource");
    return NULL;
}



const char *
Symbol::valuesourcename () const
{
    return valuesourcename (valuesource());
}



std::ostream &
Symbol::print_vals (std::ostream &out, int maxvals) const
{
    if (! data())
        return out;
    TypeDesc t = typespec().simpletype();
    int n = std::min (int(t.aggregate * t.numelements()), maxvals);
    if (t.basetype == TypeDesc::FLOAT) {
        for (int j = 0;  j < n;  ++j)
            out << (j ? " " : "") << ((float *)data())[j];
    } else if (t.basetype == TypeDesc::INT) {
        for (int j = 0;  j < n;  ++j)
            out << (j ? " " : "") << ((int *)data())[j];
    } else if (t.basetype == TypeDesc::STRING) {
        for (int j = 0;  j < n;  ++j)
            out << (j ? " " : "") << "\"" 
                << Strutil::escape_chars(((ustring *)data())[j].string())
                << "\"";
    }
    if (int(t.aggregate * t.numelements()) > maxvals)
        out << "...";
    return out;
}



std::ostream &
Symbol::print (std::ostream &out) const
{
    out << Symbol::symtype_shortname(symtype())
        << " " << typespec().string() << " " << name();
    if (everused())
        out << " (used " << firstuse() << ' ' << lastuse() 
            << " read " << firstread() << ' ' << lastread() 
            << " write " << firstwrite() << ' ' << lastwrite();
    else
        out << " (unused";
    out << (has_derivs() ? " derivs" : "") << ")";
    if (symtype() == SymTypeParam || symtype() == SymTypeOutputParam) {
        if (has_init_ops())
            out << " init [" << initbegin() << ',' << initend() << ")";
        if (connected())
            out << " connected";
        if (connected_down())
            out << " down-connected";
        if (!connected() && !connected_down())
            out << " unconnected";
        if (symtype() == SymTypeParam && ! lockgeom())
            out << " lockgeom=0";
    }
    out << "\n";
    if (symtype() == SymTypeConst) {
        out << "\tconst: ";
        print_vals(out);
        out << "\n";
    } else if (symtype() == SymTypeParam || symtype() == SymTypeOutputParam) {
        if (valuesource() == Symbol::DefaultVal && !has_init_ops()) {
            out << "\tdefault: ";
            print_vals(out);
            out << "\n";
        } else if (valuesource() == Symbol::InstanceVal) {
            out << "\tvalue: ";
            print_vals(out);
            out << "\n";
        }
    }
    return out;
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
SymbolTable::find_exact (ustring mangled_name) const
{
    ScopeTable::const_iterator s = m_allmangled.find (mangled_name);
    return (s != m_allmangled.end()) ? s->second : NULL;
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
    m_allmangled[ustring(sym->mangled())] = sym;
}



int
SymbolTable::new_struct (ustring name)
{
    int structid = TypeSpec::new_struct (new StructSpec (name, scopeid()));
    insert (new Symbol (name, TypeSpec ("",structid), SymTypeType));
    return structid;
}



StructSpec *
SymbolTable::current_struct ()
{
    return TypeSpec::struct_list().back().get();
}



void
SymbolTable::add_struct_field (const TypeSpec &type, ustring name)
{
    StructSpec *s = current_struct();
    ASSERT (s && "add_struct_field couldn't find a current struct");
    s->add_field (type, name);
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
    for (SymbolPtrVec::iterator i = m_allsyms.begin(); i != m_allsyms.end(); ++i)
        delete (*i);
    m_allsyms.clear ();
    TypeSpec::struct_list().clear ();
}




void
SymbolTable::print ()
{
    if (TypeSpec::struct_list().size()) {
        std::cout << "Structure table:\n";
        int structid = 1;
        BOOST_FOREACH (shared_ptr<StructSpec> &s, TypeSpec::struct_list()) {
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
                      << s->typespec().structspec()->name();
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
            args += advance;
            std::cout << " function (" << m_comp.typelist_from_code(args) << ") ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}



}; // namespace pvt

OSL_NAMESPACE_EXIT
