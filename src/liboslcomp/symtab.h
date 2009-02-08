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

#include <boost/intrusive/list.hpp>

#include "OpenImageIO/typedesc.h"
#include "OpenImageIO/ustring.h"
#include "OpenImageIO/dassert.h"


namespace OSL {
namespace pvt {



/// Light-weight way to describe types for the compiler -- simple types,
/// closures, or the ID of a structure.
class TypeSpec {
public:
    TypeSpec (TypeDesc simple, bool closure=false)
        : m_structure(0), m_closure(closure), m_simple(simple)
    { }

    bool is_closure () const { return m_closure; }
    bool is_structure () const { return m_structure > 0; }
    TypeDesc type () const { return m_simple; }

private:
    short m_structure;     ///< 0 is not a structure, >=1 for structure id
    bool  m_closure;       ///< Is it a closure? (m_simple also used)
    TypeDesc m_simple;     ///< Data if it's a simple type
};



/// The compiler record of a single symbol (identifier) and all relevant
/// information about it.
class Symbol {
public:
    Symbol (ustring n, const TypeSpec &t) : m_name(n), m_typespec(t) { }
    ~Symbol () { }

    ustring name () const { return m_name; }

    const TypeSpec &type () const { return m_typespec; }

    int scope () const { return m_scope; }

private:
    ustring m_name;
    TypeSpec m_typespec;
    int m_scope;
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
    ~SymbolTable () { }

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

    /// Insert the symbol into the current inner scope.  
    ///
    void insert (Symbol *sym) {
        recursive_lock_guard guard (m_mutex);  // thread safety
        DASSERT (sym != NULL);
        DASSERT (! find (sym->name()));
        m_scopetables.back()[sym->name()] = sym;
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
        for (SymbolList::iterator i = m_allsyms.begin(); i != m_allsyms.end(); ++i)
            delete (*i);
        m_allsyms.clear ();
    }

private:
    SymbolList m_allsyms;          ///< Master list of all symbols
    ScopeTableStack m_scopetables; ///< Stack of symbol scopes
    std::stack<int> m_scopestack;  ///< Stack of current scope IDs
    int m_scopeid;                 ///< Current scope ID
    int m_nextscopeid;             ///< Next unique scope ID
    mutable recursive_mutex m_mutex;  ///< Mutex to make the table thread-safe
};



}; // namespace pvt
}; // namespace OSL


#endif /* SYMTAB_H */
