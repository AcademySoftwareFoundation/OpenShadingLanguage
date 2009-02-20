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
    TypeSpec ()
        : m_simple(TypeDesc::UNKNOWN), m_structure(0), m_closure(false)
    { }

    TypeSpec (TypeDesc simple)
        : m_simple(simple), m_structure(0), m_closure(false)
    { }

    TypeSpec (TypeDesc simple, bool closure)
        : m_simple(simple), m_structure(0), m_closure(closure)
    { }

    TypeSpec (const char *name, int structid, int arraylen=0)
        : m_simple(TypeDesc::UNKNOWN, arraylen), m_structure((short)structid),
          m_closure(false)
    { }

    /// Express the type as a string
    ///
    std::string string () const {
        std::string s;
        if (is_structure())
            s = Strutil::format ("struct %d", structure());
        else s = simpletype().c_str();
        if (is_closure())
            s += " closure";
        return s;
    }

    TypeSpec & operator= (const TypeDesc simple) {
        m_simple = simple;
        m_structure = 0;
        m_closure = false;
    }

    bool operator== (const TypeSpec &x) const {
        return (m_simple == x.m_simple && m_structure == x.m_structure &&
                m_closure == x.m_closure);
    }
    bool operator!= (const TypeSpec &x) const { return ! (*this == x); }

    TypeDesc simpletype () const { return m_simple; }

    bool is_closure () const { return m_closure; }
    bool is_structure () const { return m_structure > 0 && !is_array(); }
    int structure () const { return m_structure; }
    bool is_array () const { return m_simple.arraylen != 0; }
    void make_array (int len) { m_simple.arraylen = len; }
    TypeSpec elementtype () const { TypeSpec t; t.make_array (0); return t; }
    bool is_aggregate () const {
        return !is_structure() && !is_closure() && 
               !is_array() && m_simple.aggregate != TypeDesc::SCALAR;
    }
    TypeDesc::AGGREGATE aggregate () const { return (TypeDesc::AGGREGATE)m_simple.aggregate; }

    /// Is it an int?
    ///
    bool is_int () const {
        return m_simple == TypeDesc::INT && !is_structure() && !is_closure();
    }

    /// Is it a float?
    ///
    bool is_float () const {
        return m_simple == TypeDesc::TypeFloat && !is_structure() && !is_closure();
    }

    /// Is it a string?
    ///
    bool is_string () const {
        return m_simple == TypeDesc::TypeString && !is_structure() && !is_closure();
    }

    /// Is it a triple (color, point, vector, or normal)?
    ///
    bool is_triple () const {
        return ! is_structure() && ! is_closure() && 
            (m_simple == TypeDesc::TypeColor ||
             m_simple == TypeDesc::TypePoint ||
             m_simple == TypeDesc::TypeVector ||
             m_simple == TypeDesc::TypeNormal);
    }

    /// Is it based on floats (even if an aggregate?)
    bool is_floatbased () const {
        return ! is_structure() && ! is_closure() && ! is_array() &&
            m_simple.basetype == TypeDesc::FLOAT;
    }

    /// Is it a vector-like triple (point, vector, or normal)?
    ///
    bool is_vectriple () const {
        return ! is_structure() && ! is_closure() && 
            (m_simple == TypeDesc::TypePoint ||
             m_simple == TypeDesc::TypeVector ||
             m_simple == TypeDesc::TypeNormal);
    }

    /// Types are equivalent if they are identical, or if both are
    /// vector-like.
    friend bool equivalent (const TypeSpec &a, const TypeSpec &b) {
        return (a == b) || (a.is_vectriple() && b.is_vectriple());
    }

    /// Is type b is assignable to a?  It is if they are the equivalent(),
    /// or if a is a float or float-aggregate and b is a float or int.
    friend bool assignable (const TypeSpec &a, const TypeSpec &b) {
        return equivalent (a, b) || 
            (a.is_floatbased() && (b.is_float() || b.is_int()));
    }

private:
    TypeDesc m_simple;     ///< Data if it's a simple type
    int m_arraylen;        ///< 0 if not array, nonzero for array length
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
    enum SymType {
        SymTypeParam, SymTypeLocal, SymTypeTemp, SymTypeFunction, 
        SymTypeType
    };

    Symbol (ustring n, const TypeSpec &t, SymType s, ASTNode *node=NULL) 
        : m_name(n), m_typespec(t), m_symtype(s), m_scope(0), m_node(node)
    { }
    ~Symbol () { }

    ustring name () const { return m_name; }

    std::string mangled () const {
        // FIXME: De-alias
        return scope() ? Strutil::format ("___%d_%s", scope(), m_name.c_str())
                       : m_name.string();
    }

    const TypeSpec &typespec () const { return m_typespec; }

    SymType symtype () const { return m_symtype; }

    int scope () const { return m_scope; }

    void scope (int s) { m_scope = s; }

    bool is_function () const { return m_symtype == Symbol::SymTypeFunction; }

    bool is_structure () const { return m_symtype == Symbol::SymTypeType; }

private:
    ustring m_name;
    TypeSpec m_typespec;
    SymType m_symtype;
    int m_scope;
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
        m_structs.push_back (NULL);  // Create dummy struct
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
        sym->scope (scopeid ());
        m_scopetables.back()[sym->name()] = sym;
        m_allsyms.push_back (sym);
    }

    /// Make a new structure type and name it.  Return the index of the
    /// new structure.
    int new_struct (ustring name) {
        recursive_lock_guard guard (m_mutex);  // thread safety
        m_structs.push_back (new StructSpec (name, scopeid()));
        int structid = (int) m_structs.size() - 1;
        insert (new Symbol (name, TypeSpec ("",structid), Symbol::SymTypeType));
        return structid;
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
                              << f.type.simpletype().c_str() << "\n";
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
                std::cout << s->typespec().simpletype().c_str();
            }
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
