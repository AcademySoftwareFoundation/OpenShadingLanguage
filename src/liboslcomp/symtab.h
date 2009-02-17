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

#ifndef SYMTAB_H
#define SYMTAB_H

#include <vector>
#include <stack>

#ifdef __GNUC__
# include <ext/hash_map>
# include <ext/hash_set>
using __gnu_cxx::hash_map;
using __gnu_cxx::hash_set;
#else
# include <hash_map>
# include <hash_set>
using std::hash_map;
using std::hash_set;
#endif

#include <boost/foreach.hpp>

#include "OpenImageIO/typedesc.h"
#include "OpenImageIO/ustring.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/strutil.h"


namespace OSL {
namespace pvt {


class ASTNode;  // forward declaration



/// Light-weight way to describe types for the compiler -- simple types,
/// closures, or the ID of a structure.
class TypeSpec {
public:
    TypeSpec (TypeDesc simple=TypeDesc::UNKNOWN, bool closure=false)
        : m_simple(simple), m_structure(0), m_closure(closure)
    { }

    TypeSpec (int structid)
        : m_simple(TypeDesc::UNKNOWN), m_structure((short)structid),
          m_closure(false)
    { }

    bool is_closure () const { return m_closure; }
    bool is_structure () const { return m_structure > 0; }
    TypeDesc type () const { return m_simple; }
    int structure () const { return m_structure; }

    void make_array (int len) { m_simple.arraylen = len; }

    /// Express the type as a string
    ///
    std::string string () const {
        std::string s;
        if (is_structure())
            s = Strutil::format ("struct %d", structure());
        else s = type().c_str();
        if (is_closure())
            s += " closure";
        return s;
    }

    bool operator== (const TypeSpec &x) const {
        return (m_simple == x.m_simple && m_structure == x.m_structure &&
                m_closure == x.m_closure);
    }
    bool operator!= (const TypeSpec &x) const { return ! (*this == x); }

    bool is_array () const { return m_simple.arraylen != 0; }

    bool is_int () const {
        return m_simple == TypeDesc::INT && !is_structure() && !is_closure();
    }

private:
    TypeDesc m_simple;     ///< Data if it's a simple type
    short m_structure;     ///< 0 is not a structure, >=1 for structure id
    bool  m_closure;       ///< Is it a closure? (m_simple also used)
};



class StructSpec {
public:
    StructSpec (ustring name, int scope) : m_name(name), m_scope(scope) { }

    struct FieldSpec {
        FieldSpec (const TypeSpec &t, ustring n) : type(t), name(n) { }
        TypeSpec type;
        ustring name;
    };

    void add_field (const TypeSpec &t, ustring n) {
        m_fields.push_back (FieldSpec (t, n));
    }

    ustring name () const { return m_name; }

    std::string mangled () const {
        return scope() ? Strutil::format ("___%d_%s", scope(), m_name.c_str())
                       : m_name.string();
    }

    size_t numfields () const { return m_fields.size(); }

    const FieldSpec & field (size_t i) const { return m_fields[i]; }

    int scope () const { return m_scope; }

private:
    ustring m_name;
    int m_scope;
    std::vector<FieldSpec> m_fields;
};


typedef std::vector<StructSpec *> StructList;



/// The compiler record of a single symbol (identifier) and all relevant
/// information about it.
class Symbol {
public:
    Symbol (ustring n, const TypeSpec &t) 
        : m_name(n), m_typespec(t), m_scope(0), m_isfunction(false), 
          m_node(NULL)
    { }
    ~Symbol () { }

    ustring name () const { return m_name; }

    std::string mangled () const {
        // FIXME: De-alias
        return scope() ? Strutil::format ("___%d_%s", scope(), m_name.c_str())
                       : m_name.string();
    }

    const TypeSpec &type () const { return m_typespec; }

    int scope () const { return m_scope; }

    void scope (int s) { m_scope = s; }

    bool is_function () const { return m_isfunction; }

    void is_function (ASTNode *def) {
        m_node = def;
        m_isfunction = true;
    }

private:
    ustring m_name;
    TypeSpec m_typespec;
    int m_scope;
    bool m_isfunction;
    ASTNode *m_node;
};



typedef std::vector<Symbol *> SymbolList;



/// SymbolTable maintains a list of Symbol records for the compiler.
/// Symbol lookups only work when done while parsing -- in other words,
/// the lookups are based on a "live" scope stack.  After parsing is
/// over, everybody who needs a symbol reference better already have it,
/// otherwise, lookups by name aren't going to work (and how could they,
/// considering that name resolution is based on lexical scope).
///
/// Implementation details: For each scope (within which symbol names
/// are unique), there's a hash_map of symbols.  There's also a stack
/// of such maps representing the current scope hierarchy, so a symbol
/// search proceeds from innermost scope (top of stack) to outermost
/// (bottom of stack).  

//template <class S>
class SymbolTable {
public:
//    typedef S Symbol;
    typedef hash_map<ustring, Symbol *,ustringHash> ScopeTable;
    typedef std::vector<ScopeTable> ScopeTableStack;

    SymbolTable ()
        : m_scopeid(-1), m_nextscopeid(0)
    {
        m_scopetables.reserve (20);  // So unlikely to ever copy tables
        push ();                     // Create scope 0 -- global scope
    }
    ~SymbolTable () {
        delete_syms ();
    }

    void lock () { m_mutex.lock (); }
    void unlock () { m_mutex.unlock (); }

    /// Look up the symbol, starting with the innermost active scope and
    /// proceeding to successively outer scopes.  Return a pointer to
    /// the symbol record if found, NULL if not found in any active
    /// scopes.  If 'last' is non-NULL, we're already found that one and
    /// are looking for another symbol of the same name in a farther-out
    /// scope.
    Symbol * find (ustring name, Symbol *last=NULL) const {
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

    /// If there is already a symbol with that name in the current scope,
    /// return it, otherwise return NULL.
    Symbol * clash (ustring name) const {
        recursive_lock_guard guard (m_mutex);  // thread safety
        Symbol *s = find (name);
        return (s && s->scope() == scopeid()) ? s : NULL;
    }

    /// Insert the symbol into the current inner scope.  
    ///
    void insert (Symbol *sym) {
        recursive_lock_guard guard (m_mutex);  // thread safety
        DASSERT (sym != NULL);
        DASSERT (! find (sym->name()));
        sym->scope (scopeid ());
        m_scopetables.back()[sym->name()] = sym;
        m_allsyms.push_back (sym);
    }

    /// Make a new structure type and name it.  Return the index of the
    /// new structure.
    int new_struct (ustring name) {
        recursive_lock_guard guard (m_mutex);  // thread safety
        m_structs.push_back (new StructSpec (name, scopeid()));
        return (int) m_structs.size();
    }

    void add_struct_field (const TypeSpec &type, ustring name) {
        recursive_lock_guard guard (m_mutex);  // thread safety
        m_structs.back()->add_field (type, name);
    }

    /// Return the current scope ID
    ///
    int scopeid () const {
        recursive_lock_guard guard (m_mutex);  // thread safety
        return m_scopeid;
    }

    /// Create a new unique scope, inner to the previous current scope.
    ///
    void push () {
        recursive_lock_guard guard (m_mutex);     // thread safety
        m_scopestack.push (m_scopeid);  // push old scope id on the scope stack
        m_scopeid = m_nextscopeid++;    // set to new scope id
        m_scopetables.resize (m_scopetables.size()+1); // push scope table
    }

    /// Restore to the next outermost scope.
    ///
    void pop () {
        recursive_lock_guard guard (m_mutex);  // thread safety
        m_scopetables.resize (m_scopetables.size()-1);
        ASSERT (! m_scopestack.empty());
        m_scopeid = m_scopestack.top ();
        m_scopestack.pop ();
    }

    /// delete all symbols that have ever been entered into the table.
    /// After doing this, beware following any Symbol pointers left over!
    void delete_syms () {
        recursive_lock_guard guard (m_mutex);  // thread safety
        for (SymbolList::iterator i = m_allsyms.begin(); i != m_allsyms.end(); ++i)
            delete (*i);
        m_allsyms.clear ();
        for (StructList::iterator i = m_structs.begin(); i != m_structs.end(); ++i)
            delete (*i);
        m_structs.clear ();
    }

    void print () {
        recursive_lock_guard guard (m_mutex);  // thread safety
        if (m_structs.size()) {
            std::cout << "Structure table:\n";
            BOOST_FOREACH (const StructSpec * s, m_structs) {
                std::cout << "    struct " << s->mangled();
                if (s->scope())
                    std::cout << " (" << s->name() 
                              << " in scope " << s->scope() << ")";
                std::cout << " :\n";
                for (size_t i = 0;  i < s->numfields();  ++i) {
                    const StructSpec::FieldSpec & f (s->field(i));
                    std::cout << "\t" << f.name << " : " 
                              << f.type.type().c_str() << "\n";
                }
            }
            std::cout << "\n";
        }
        std::cout << "Symbol table:\n";
        BOOST_FOREACH (const Symbol *s, m_allsyms) {
            std::cout << "\t" << s->mangled() << " : " 
                      << s->type().type().c_str();
            if (s->scope())
                std::cout << " (" << s->name() << " in scope " 
                          << s->scope() << ")";
            std::cout << "\n";
        }
        std::cout << "\n";
    }

private:
    SymbolList m_allsyms;            ///< Master list of all symbols
    StructList m_structs;            ///< All the structures we use
    ScopeTableStack m_scopetables;   ///< Stack of symbol scopes
    std::stack<int> m_scopestack;    ///< Stack of current scope IDs
    int m_scopeid;                   ///< Current scope ID
    int m_nextscopeid;               ///< Next unique scope ID
    mutable recursive_mutex m_mutex; ///< Mutex for thread-safety
};



}; // namespace pvt
}; // namespace OSL


#endif /* SYMTAB_H */
