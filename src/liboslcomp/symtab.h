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

#include "OpenImageIO/typedesc.h"
#include "OpenImageIO/ustring.h"

#include "osl_pvt.h"


namespace OSL {
namespace pvt {


class OSLCompilerImpl;
class ASTNode;  // forward declaration
class ASTfunction_definition;



/// Light-weight way to describe types for the compiler -- simple types,
/// closures, or the ID of a structure.
class TypeSpec {
public:
    /// Default ctr of TypeSpec (unknown type)
    ///
    TypeSpec ()
        : m_simple(TypeDesc::UNKNOWN), m_structure(0), m_closure(false)
    { }

    /// Construct a TypeSpec that represents an ordinary simple type
    /// (including arrays of simple types).
    TypeSpec (TypeDesc simple)
        : m_simple(simple), m_structure(0), m_closure(false)
    { }

    /// Construct a TypeSpec representing a closure (pass closure=true)
    /// of a simple type.
    TypeSpec (TypeDesc simple, bool closure)
        : m_simple(simple), m_structure(0), m_closure(closure)
    { }

    /// Construct a TypeSpec describing a struct or array of structs,
    /// by supplying the struct name, structure id, and array length
    /// (if it's an array of structures).
    TypeSpec (const char *name, int structid, int arraylen=0)
        : m_simple(TypeDesc::UNKNOWN, arraylen), m_structure((short)structid),
          m_closure(false)
    { }

    /// Express the type as a string
    ///
    std::string string () const;

    /// Assignment of a simple TypeDesc to a full TypeSpec.
    ///
    const TypeSpec & operator= (const TypeDesc simple) {
        m_simple = simple;
        m_structure = 0;
        m_closure = false;
        return *this;
    }

    /// Are two TypeSpec's identical?
    ///
    bool operator== (const TypeSpec &x) const {
        return (m_simple == x.m_simple && m_structure == x.m_structure &&
                m_closure == x.m_closure);
    }
    /// Are two TypeSpec's different?
    ///
    bool operator!= (const TypeSpec &x) const { return ! (*this == x); }

    /// Return just the simple type underlying this TypeSpec -- only works
    /// reliable if it's not a struct, a struct will return an UNKNOWN type.
    TypeDesc simpletype () const { return m_simple; }

    /// Is this typespec a closure?  (N.B. if so, you can find out what
    /// kind of closure it is with simpletype()).
    bool is_closure () const { return m_closure; }

    /// Is this typespec a single structure?  Caveat: Returns false if
    /// it's an array of structs.  N.B. You can find out which struct
    /// with structure().
    bool is_structure () const { return m_structure > 0 && !is_array(); }

    /// Return the structure ID of this typespec, or 0 if it's not a
    /// struct.
    int structure () const { return m_structure; }

    /// Is this an array (either a simple array, or an array of structs)?
    ///
    bool is_array () const { return m_simple.arraylen != 0; }

    /// Returns the length of the array, or 0 if not an array.
    ///
    int arraylength () const { return m_simple.arraylen; }

    /// Alter this typespec to make it into an array of the given length
    /// (including 0 -> make it not be an array).  The basic type (not
    /// counting its array length) is unchanged.
    void make_array (int len) { m_simple.arraylen = len; }

    /// For an array, return the TypeSpec of an individual element of the
    /// array.  For a non-array, just return the type.
    TypeSpec elementtype () const { TypeSpec t; t.make_array (0); return t; }

    /// Is it an "aggregate" type, meaning a structure, closure, array,
    /// or a simple type that isn't a scalar (such as a vector/point)?
    bool is_aggregate () const {
        return !is_structure() && !is_closure() && 
               !is_array() && m_simple.aggregate != TypeDesc::SCALAR;
    }

    /// Return the aggregateness of the underlying simple type (SCALAR,
    /// VEC3, or MATRIX44).
    TypeDesc::AGGREGATE aggregate () const { return (TypeDesc::AGGREGATE)m_simple.aggregate; }

    /// Is it a simple scalar int?
    ///
    bool is_int () const {
        return m_simple == TypeDesc::TypeInt && !is_structure() && !is_closure();
    }

    /// Is it a simple scalar float?
    ///
    bool is_float () const {
        return m_simple == TypeDesc::TypeFloat && !is_structure() && !is_closure();
    }

    /// Is it a simple scalar float?
    ///
    bool is_color () const {
        return m_simple == TypeDesc::TypeColor && !is_structure() && !is_closure();
    }

    /// Is it a simple string?
    ///
    bool is_string () const {
        return m_simple == TypeDesc::TypeString && !is_structure() && !is_closure();
    }

    /// Is it a simple triple (color, point, vector, or normal)?
    ///
    bool is_triple () const {
        return ! is_structure() && ! is_closure() && 
            (m_simple == TypeDesc::TypeColor ||
             m_simple == TypeDesc::TypePoint ||
             m_simple == TypeDesc::TypeVector ||
             m_simple == TypeDesc::TypeNormal);
    }

    /// Is this a simple type based on floats (including color/vector/etc)?  
    /// This will return false for a closure or array (even if of floats)
    /// or struct.
    bool is_floatbased () const {
        return ! is_structure() && ! is_closure() && ! is_array() &&
            m_simple.basetype == TypeDesc::FLOAT;
    }

    /// Is it a simple numeric type (based on float or int, even if an
    /// aggregate)?  This is false for a closure or array (even if of
    /// an underlying numeric type) or struct.
    bool is_numeric () const {
        return ! is_structure() && ! is_closure() && ! is_array() &&
            (m_simple.basetype == TypeDesc::FLOAT || m_simple.basetype == TypeDesc::INT);
    }

    bool is_scalarnum () const {
        return is_numeric() && m_simple.aggregate == TypeDesc::SCALAR;
    }

    /// Is it a simple straight-up single int or float)?
    ///
    bool is_int_or_float () const { return is_scalarnum(); }

    /// Is it a simple vector-like triple (point, vector, or normal, but
    /// not an array or closure)?
    bool is_vectriple () const {
        return ! is_structure() && ! is_closure() && 
            (m_simple == TypeDesc::TypePoint ||
             m_simple == TypeDesc::TypeVector ||
             m_simple == TypeDesc::TypeNormal);
    }

    /// Is it based on a vector-like triple (point, vector, or normal)?
    /// (It's ok for it to be an array or closure.)
    bool is_vectriple_based () const {
        return ! is_structure() && 
            (m_simple.elementtype() == TypeDesc::TypePoint ||
             m_simple.elementtype() == TypeDesc::TypeVector ||
             m_simple.elementtype() == TypeDesc::TypeNormal);
    }

    /// Is it a simple matrix (but not an array or closure)?
    ///
    bool is_matrix () const {
        return ! is_structure() && ! is_closure() && 
            m_simple == TypeDesc::TypeMatrix;
    }

    /// Is it a color closure?
    ///
    bool is_color_closure () const {
        return is_closure() && (m_simple == TypeDesc::TypeColor);
    }

    /// Types are equivalent if they are identical, or if both are
    /// vector-like (and match their array-ness and closure-ness).
    friend bool equivalent (const TypeSpec &a, const TypeSpec &b) {
        return (a == b) || 
            (a.is_vectriple_based() && b.is_vectriple_based() &&
             a.is_closure() == b.is_closure() &&
             a.arraylength() == b.arraylength());
    }

    /// Is type b is assignable to a?  It is if they are the equivalent(),
    /// or if a is a float or float-aggregate and b is a float or int.
    friend bool assignable (const TypeSpec &a, const TypeSpec &b) {
        return equivalent (a, b) || 
            (a.is_floatbased() && (b.is_float() || b.is_int()));
    }

private:
    TypeDesc m_simple;     ///< Data if it's a simple type
    short m_structure;     ///< 0 is not a structure, >=1 for structure id
    bool  m_closure;       ///< Is it a closure? (m_simple also used)
};



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



/// The compiler record of a single symbol (identifier) and all relevant
/// information about it.
class Symbol {
public:
    Symbol (ustring name, const TypeSpec &datatype, SymType symtype,
            ASTNode *declaration_node=NULL) 
        : m_name(name), m_typespec(datatype), m_symtype(symtype),
          m_scope(0), m_node(declaration_node), m_alias(NULL)
    { }
    virtual ~Symbol () { }

    /// The symbol's (unmangled) name, guaranteed unique only within the
    /// symbol's declaration scope.
    ustring name () const { return m_name; }

    /// The symbol's name, mangled to incorporate the scope so it will be
    /// a globally unique name.
    std::string mangled () const;

    /// Data type of this symbol.
    ///
    const TypeSpec &typespec () const { return m_typespec; }

    /// Kind of symbol this is (param, local, etc.)
    ///
    SymType symtype () const { return m_symtype; }

    /// Numerical ID of the scope in which this symbol was declared.
    ///
    int scope () const { return m_scope; }

    /// Set the scope of this symbol to s.
    ///
    void scope (int s) { m_scope = s; }

    /// Return teh AST node containing the declaration of this symbol.
    /// Use with care!
    ASTNode *node () const { return m_node; }

    /// Is this symbol a function?
    ///
    bool is_function () const { return m_symtype == SymTypeFunction; }

    /// Is this symbol a structure?
    ///
    bool is_structure () const { return m_symtype == SymTypeType; }

    /// Return a ptr to the symbol that this really refers to, tracing
    /// aliases back all the way until it finds a symbol that isn't an
    /// alias for anything else.
    Symbol *dealias () const {
        Symbol *s = const_cast<Symbol *>(this);
        while (s->m_alias)
            s = s->m_alias;
        return s;
    }

    /// Return a string representation ("param", "global", etc.) of the
    /// SymType s.
    static const char *symtype_shortname (SymType s);

    /// Return a string representation ("param", "global", etc.) of this
    /// symbol.
    const char *symtype_shortname () const {
        return symtype_shortname(m_symtype);
    }

protected:
    ustring m_name;             ///< Symbol name (unmangled)
    TypeSpec m_typespec;        ///< Data type of the symbol
    SymType m_symtype;          ///< Kind of symbol (param, local, etc.)
    int m_scope;                ///< Scope where this symbol was declared
    ASTNode *m_node;            ///< Ptr to the declaration of this symbol
    Symbol *m_alias;            ///< Another symbol that this is an alias for
    bool m_const_initializer;   ///< initializer is a constant expression
};



typedef std::vector<Symbol *> SymbolList;



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

    SymbolList::iterator symbegin () { return m_allsyms.begin(); }
    const SymbolList::const_iterator symbegin () const { return m_allsyms.begin(); }
    SymbolList::iterator symend () { return m_allsyms.end(); }
    const SymbolList::const_iterator symend () const { return m_allsyms.end(); }

private:
    OSLCompilerImpl &m_comp;         ///< Back-reference to compiler
    SymbolList m_allsyms;            ///< Master list of all symbols
    StructList m_structs;            ///< All the structures we use
    ScopeTableStack m_scopetables;   ///< Stack of symbol scopes
    std::stack<int> m_scopestack;    ///< Stack of current scope IDs
    int m_scopeid;                   ///< Current scope ID
    int m_nextscopeid;               ///< Next unique scope ID
};



}; // namespace pvt
}; // namespace OSL


#endif /* SYMTAB_H */
