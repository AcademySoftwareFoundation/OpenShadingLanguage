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

#ifndef OSL_SYMTAB_H
#define OSL_SYMTAB_H

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

#include "OpenImageIO/typedesc.h"
#include "OpenImageIO/ustring.h"

#include "osl_pvt.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {


class OSLCompilerImpl;
class ASTNode;  // forward declaration
class ASTfunction_definition;



/// Describe the layout of an OSL 'struct'.
/// Basically it's just a list of all the individual fields' names and
/// types.
class StructSpec {
public:
    /// Construct a new struct with the given name, in the given scope.
    ///
    StructSpec (ustring name, int scope) : m_name(name), m_scope(scope) { }

    /// Description of a single structure field -- just a type and name.
    ///
    struct FieldSpec {
        FieldSpec (const TypeSpec &t, ustring n) : type(t), name(n) { }
        TypeSpec type;
        ustring name;
    };

    /// Append a new field (with type and name) to this struct.
    ///
    void add_field (const TypeSpec &type, ustring name) {
        m_fields.push_back (FieldSpec (type, name));
    }

    /// The name of this struct (may not be unique across all scopes).
    ///
    ustring name () const { return m_name; }

    /// The unique mangled name (with scope embedded) of this struct.
    ///
    std::string mangled () const;

    /// The scope number where this struct was defined.
    ///
    int scope () const { return m_scope; }

    /// Number of fields in the struct.
    ///
    size_t numfields () const { return m_fields.size(); }

    /// Return a reference to an individual FieldSpec for one field
    /// of the struct, indexed numerically (starting with 0).
    const FieldSpec & field (size_t i) const { return m_fields[i]; }

private:
    ustring m_name;                    ///< Structure name (unmangled)
    int m_scope;                       ///< Structure's scope id
    std::vector<FieldSpec> m_fields;   ///< List of fields of the struct
};


/// Handy typedef for a vector of pointers to StructSpec's.
///
typedef std::vector<StructSpec *> StructList;



/// Subclass of Symbol used just for functions, which are different
/// because they can be polymorphic, and also need to carry around more
/// information than other symbols.
class FunctionSymbol : public Symbol {
public:
    FunctionSymbol (ustring n, TypeSpec type, ASTNode *node=NULL)
        : Symbol(n, type, SymTypeFunction, node), m_nextpoly(NULL)
    { }

    void nextpoly (FunctionSymbol *nextpoly) { m_nextpoly = nextpoly; }
    FunctionSymbol *nextpoly () const { return m_nextpoly; }
    const ASTfunction_definition *funcdef () const {
        return (const ASTfunction_definition *) m_node;
    }
    void argcodes (ustring args) { m_argcodes = args; }
    ustring argcodes () const { return m_argcodes; }

private:
    ustring m_argcodes;              ///< Encoded arg types
    FunctionSymbol *m_nextpoly;      ///< Next polymorphic version
};



/// Subclass of Symbol used just for constants, which are different
/// because they need to carry around their value.
class ConstantSymbol : public Symbol {
public:
    ConstantSymbol (ustring n, ustring val)
        : Symbol(n, TypeDesc::TypeString, SymTypeConst), m_s(val) { }
    ConstantSymbol (ustring n, int val)
        : Symbol(n, TypeDesc::TypeInt, SymTypeConst), m_i(val) { }
    ConstantSymbol (ustring n, float val)
        : Symbol(n, TypeDesc::TypeFloat, SymTypeConst), m_f(val) { }

    ustring strval () const { return m_s; }
    int intval () const { return m_i; }
    float floatval () const { return m_typespec.is_int() ? (float)m_i : m_f; }

private:
    ustring m_s;
    int m_i;
    float m_f;
};



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
///
class SymbolTable {
public:
    typedef hash_map<ustring, Symbol *,ustringHash> ScopeTable;
    typedef std::vector<ScopeTable> ScopeTableStack;

    SymbolTable (OSLCompilerImpl &comp)
        : m_comp(comp), m_scopeid(-1), m_nextscopeid(0)
    {
        m_scopetables.reserve (20);  // So unlikely to ever copy tables
        push ();                     // Create scope 0 -- global scope
        m_structs.push_back (NULL);  // Create dummy struct
    }
    ~SymbolTable () {
        delete_syms ();
    }

    /// Look up the symbol, starting with the innermost active scope and
    /// proceeding to successively outer scopes.  Return a pointer to
    /// the symbol record if found, NULL if not found in any active
    /// scopes.  If 'last' is non-NULL, we're already found that one and
    /// are looking for another symbol of the same name in a farther-out
    /// scope.
    Symbol * find (ustring name, Symbol *last=NULL) const;

    /// If there is already a symbol with that name in the current scope,
    /// return it, otherwise return NULL.
    Symbol * clash (ustring name) const;

    /// Insert the symbol into the current inner scope.  
    ///
    void insert (Symbol *sym);

    /// Make a new structure type and name it.  Return the index of the
    /// new structure.
    int new_struct (ustring name);

    void add_struct_field (const TypeSpec &type, ustring name);

    /// Return the current scope ID
    ///
    int scopeid () const {
        return m_scopeid;
    }

    /// Create a new unique scope, inner to the previous current scope.
    ///
    void push ();

    /// Restore to the next outermost scope.
    ///
    void pop ();

    /// delete all symbols that have ever been entered into the table.
    /// After doing this, beware following any Symbol pointers left over!
    void delete_syms ();

    /// Dump the whole symbol table to stdout for debugging purposes.
    ///
    void print ();

    SymbolPtrVec::iterator symbegin () { return m_allsyms.begin(); }
    const SymbolPtrVec::const_iterator symbegin () const { return m_allsyms.begin(); }
    SymbolPtrVec::iterator symend () { return m_allsyms.end(); }
    const SymbolPtrVec::const_iterator symend () const { return m_allsyms.end(); }

private:
    OSLCompilerImpl &m_comp;         ///< Back-reference to compiler
    SymbolPtrVec m_allsyms;          ///< Master list of all symbols
    StructList m_structs;            ///< All the structures we use
    ScopeTableStack m_scopetables;   ///< Stack of symbol scopes
    std::stack<int> m_scopestack;    ///< Stack of current scope IDs
    int m_scopeid;                   ///< Current scope ID
    int m_nextscopeid;               ///< Next unique scope ID
};



}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif


#endif /* OSL_SYMTAB_H */
