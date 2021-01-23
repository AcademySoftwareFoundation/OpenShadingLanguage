// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <memory>
#include <stack>
#include <unordered_map>
#include <vector>

#include <OpenImageIO/typedesc.h>
#include <OpenImageIO/ustring.h>

#include "osl_pvt.h"


OSL_NAMESPACE_ENTER

namespace pvt {


class OSLCompilerImpl;
class ASTNode;  // forward declaration



/// Handy typedef for a vector of pointers to StructSpec's.
///
typedef std::vector<std::shared_ptr<StructSpec>> StructList;



/// Subclass of Symbol used just for functions, which are different
/// because they can be polymorphic, and also need to carry around more
/// information than other symbols.
class FunctionSymbol final : public Symbol {
public:
    FunctionSymbol(ustring n, TypeSpec type, ASTNode* node = NULL)
        : Symbol(n, type, SymTypeFunction, node)
        , m_nextpoly(NULL)
        , m_return_location(NULL)
        , m_number_of_returns(0)
        , m_readwrite_special_case(false)
        , m_texture_args(false)
        , m_printf_args(false)
        , m_takes_derivs(false)
    {
    }

    void nextpoly(FunctionSymbol* nextpoly) { m_nextpoly = nextpoly; }
    FunctionSymbol* nextpoly() const { return m_nextpoly; }
    void argcodes(ustring args) { m_argcodes = args; }
    ustring argcodes() const { return m_argcodes; }

    Symbol* return_location() const { return m_return_location; }
    void return_location(Symbol* r) { m_return_location = r; }

    void encountered_return() { ++m_number_of_returns; }
    int number_of_returns() const { return m_number_of_returns; }

    void push_nesting(bool isloop)
    {
        ++m_function_total_nesting;
        if (isloop)
            ++m_function_loop_nesting;
    }
    void pop_nesting(bool isloop)
    {
        --m_function_total_nesting;
        if (isloop)
            --m_function_loop_nesting;
    }
    int nesting_level() const { return m_function_total_nesting; }
    void init_nesting()
    {
        m_function_total_nesting = 0;
        m_function_loop_nesting  = 0;
    }

    void readwrite_special_case(bool s) { m_readwrite_special_case = s; }
    bool readwrite_special_case() const { return m_readwrite_special_case; }

    void texture_args(bool s) { m_texture_args = s; }
    bool texture_args() const { return m_texture_args; }

    void printf_args(bool s) { m_printf_args = s; }
    bool printf_args() const { return m_printf_args; }

    void takes_derivs(bool s) { m_takes_derivs = s; }
    bool takes_derivs() const { return m_takes_derivs; }

private:
    ustring m_argcodes;          ///< Encoded arg types
    FunctionSymbol* m_nextpoly;  ///< Next polymorphic version
    // Below, temporary storage used during code generation
    Symbol* m_return_location;      ///< Store return value location
    int m_number_of_returns;        ///< How many returns?
    int m_function_loop_nesting;    ///< Loop nesting level within the func
    int m_function_total_nesting;   ///< Total nesting level within the func
    bool m_readwrite_special_case;  ///< Unusual in how it r/w's its args
    bool m_texture_args;            ///< Has texture-like token/value args
    bool m_printf_args;             ///< Has printf-like varargs
    bool m_takes_derivs;            ///< Takes derivatives of its args
};



/// Subclass of Symbol used just for constants, which are different
/// because they need to carry around their value.  For small basic
/// types (int, float, Vec3, string), the storage is local.  For the
/// general case (like arrays), it allocates memory and uses the
/// parent class's fields m_data and m_data_free, which will also cause
/// the parent class to properly free it upon destruction.
class ConstantSymbol final : public Symbol {
public:
    ConstantSymbol(ustring n, ustring val)
        : Symbol(n, TypeDesc::TypeString, SymTypeConst)
    {
        m_val.s = val.c_str();
        m_data  = &m_val.s;
    }
    ConstantSymbol(ustring n, int val)
        : Symbol(n, TypeDesc::TypeInt, SymTypeConst)
    {
        m_val.i = val;
        m_data  = &m_val.i;
    }
    ConstantSymbol(ustring n, float val)
        : Symbol(n, TypeDesc::TypeFloat, SymTypeConst)
    {
        m_val.f = val;
        m_data  = &m_val.f;
    }
    ConstantSymbol(ustring n, TypeDesc type, float x, float y, float z)
        : Symbol(n, type, SymTypeConst)
    {
        m_val.v[0] = x;
        m_val.v[1] = y;
        m_val.v[2] = z;
        m_data     = &m_val.v;
    }
    ConstantSymbol(ustring n, TypeDesc type) : Symbol(n, type, SymTypeConst)
    {
        if (type == TypeDesc::FLOAT)
            m_data = &m_val.f;
        else if (type == TypeDesc::INT)
            m_data = &m_val.i;
        else if (type == TypeDesc::STRING)
            m_data = &m_val.s;
        else if (equivalent(type, TypeDesc::TypeVector))
            m_data = &m_val.v;
        else {
            OSL_DASSERT(m_data == nullptr);
            m_data      = new char[type.size()];
            m_free_data = true;
        }
    }

    // Return the index'th float, automatically cast from int if necessary.
    float floatval(int index = 0) const { return coerce_float(index); }

private:
    union {
        const char* s;  // actually a ustring
        int i;
        float f;
        float v[3];  // actually a Vec3
    } m_val;
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
    typedef std::unordered_map<ustring, Symbol*, ustringHash> ScopeTable;
    typedef std::vector<ScopeTable> ScopeTableStack;
    typedef SymbolPtrVec::iterator iterator;
    typedef SymbolPtrVec::const_iterator const_iterator;

    SymbolTable(OSLCompilerImpl& comp)
        : m_comp(comp), m_scopeid(-1), m_nextscopeid(0)
    {
        m_scopetables.reserve(20);  // So unlikely to ever copy tables
        push();                     // Create scope 0 -- global scope
        //        m_structs.resize (1);        // Create dummy struct
    }
    ~SymbolTable() { delete_syms(); }

    /// Look up the symbol, starting with the innermost active scope and
    /// proceeding to successively outer scopes.  Return a pointer to
    /// the symbol record if found, NULL if not found in any active
    /// scopes.  If 'last' is non-NULL, we're already found that one and
    /// are looking for another symbol of the same name in a farther-out
    /// scope.
    Symbol* find(ustring name, Symbol* last = NULL) const;

    /// If there is already a symbol with that name in the current scope,
    /// return it, otherwise return NULL.
    Symbol* clash(ustring name) const;

    /// Insert the symbol into the current inner scope.
    ///
    void insert(Symbol* sym);

    /// Find the full mangled lookup from the full list of symbols.
    /// (No scope stack involved.)
    Symbol* find_exact(ustring mangled_name) const;

    /// Make a new structure type and name it.  Return the index of the
    /// new structure.
    int new_struct(ustring name);

    void add_struct_field(const TypeSpec& type, ustring name);

    StructSpec* current_struct();

    /// Return the current scope ID
    ///
    int scopeid() const { return m_scopeid; }

    /// Create a new unique scope, inner to the previous current scope.
    ///
    void push();

    /// Restore to the next outermost scope.
    ///
    void pop();

    /// delete all symbols that have ever been entered into the table.
    /// After doing this, beware following any Symbol pointers left over!
    void delete_syms();

    /// Dump the whole symbol table to stdout for debugging purposes.
    ///
    void print();

    SymbolPtrVec::iterator begin() { return m_allsyms.begin(); }
    const SymbolPtrVec::const_iterator begin() const
    {
        return m_allsyms.begin();
    }
    SymbolPtrVec::iterator end() { return m_allsyms.end(); }
    const SymbolPtrVec::const_iterator end() const { return m_allsyms.end(); }

    SymbolPtrVec& allsyms() { return m_allsyms; }

private:
    OSLCompilerImpl& m_comp;        ///< Back-reference to compiler
    SymbolPtrVec m_allsyms;         ///< Master list of all symbols
    ScopeTableStack m_scopetables;  ///< Stack of symbol scopes
    std::stack<int> m_scopestack;   ///< Stack of current scope IDs
    ScopeTable m_allmangled;        ///< All syms, mangled, in a hash table
    int m_scopeid;                  ///< Current scope ID
    int m_nextscopeid;              ///< Next unique scope ID
};



};  // namespace pvt

OSL_NAMESPACE_EXIT
