// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#pragma once

#include <OpenImageIO/refcnt.h>

#include <OSL/oslcomp.h>
#include "symtab.h"


class oslFlexLexer;


OSL_NAMESPACE_ENTER

namespace pvt {


// Forward declarations
class OSLCompilerImpl;
class Symbol;
class TypeSpec;



/// Base node for an abstract syntax tree for the OSL parser.
///
class ASTNode : public OIIO::RefCnt {
public:
    ///< Ref-counted pointer to an ASTNode
    typedef OIIO::intrusive_ptr<ASTNode> ref;

    /// List of all the types of AST nodes.
    enum NodeType {
        unknown_node,
        shader_declaration_node,
        function_declaration_node,
        variable_declaration_node,
        compound_initializer_node,
        variable_ref_node,
        preincdec_node,
        postincdec_node,
        index_node,
        structselect_node,
        conditional_statement_node,
        loop_statement_node,
        loopmod_statement_node,
        return_statement_node,
        binary_expression_node,
        unary_expression_node,
        assign_expression_node,
        ternary_expression_node,
        comma_operator_node,
        typecast_expression_node,
        type_constructor_node,
        function_call_node,
        literal_node,
        _last_node
    };

    enum Operator {
        Nothing = 0,
        Decr,
        Incr,
        Assign,
        Mul,
        Div,
        Add,
        Sub,
        Mod,
        Equal,
        NotEqual,
        Greater,
        Less,
        GreaterEqual,
        LessEqual,
        BitAnd,
        BitOr,
        Xor,
        Compl,
        And,
        Or,
        Not,
        ShiftLeft,
        ShiftRight
    };

    ASTNode(NodeType nodetype, OSLCompilerImpl* compiler);

    ASTNode(NodeType nodetype, OSLCompilerImpl* compiler, int op);
    ASTNode(NodeType nodetype, OSLCompilerImpl* compiler, int op, ASTNode* a);
    ASTNode(NodeType nodetype, OSLCompilerImpl* compiler, int op, ASTNode* a,
            ASTNode* b);
    ASTNode(NodeType nodetype, OSLCompilerImpl* compiler, int op, ASTNode* a,
            ASTNode* b, ASTNode* c);
    ASTNode(NodeType nodetype, OSLCompilerImpl* compiler, int op, ASTNode* a,
            ASTNode* b, ASTNode* c, ASTNode* d);

    virtual ~ASTNode();

    /// Print a text description of this node (and its children) to the
    /// console, for debugging.
    virtual void print(std::ostream& out, int indentlevel = 0) const;

    /// What type of node is this?
    ///
    NodeType nodetype() const { return m_nodetype; }

    /// Name of this type of node
    ///
    virtual const char* nodetypename() const = 0;

    /// What data type is this node?
    ///
    const TypeSpec& typespec() const { return m_typespec; }

    /// Name of the op, if any, or NULL.
    ///
    virtual const char* opname() const { return NULL; }

    /// Return the number of child nodes.
    ///
    size_t nchildren() const { return m_children.size(); }

    /// Name of the child node
    ///
    virtual const char* childname(size_t i) const = 0;

    /// Type check the node, return its type.  Optionally an "expected
    /// type" may be passed down, conveying any requirements or
    /// coercion.  The default (base class) implementation just type
    /// checks all the child nodes and makes this node's type be the
    /// expected if it is unknown, but doesn't change it if it's not
    /// unknown.
    virtual TypeSpec typecheck(TypeSpec expected = TypeSpec());

    /// Generate IR code for this node and its children, return the
    /// symbol where the result is stored (if applicable, otherwise
    /// NULL).  The optional 'dest' is a request for the caller to store
    /// the results in a particular place (which it can't always do, of
    /// course).
    virtual Symbol* codegen(Symbol* dest = NULL);

    /// Generate IR code for this node make sure it's boiled down to an
    /// int (i.e. if not already an int, generate one that's 1 if the
    /// original code was non-zero or non-empty string).  The optional
    /// 'dest' is a request for the caller to store the results in a
    /// particular place (which it can't always do, of course).  If
    /// 'boolify' is true and the normal code generates an int, convert
    /// it to a 0 or 1 value.  If 'invert' is true, invert the result.
    Symbol* codegen_int(Symbol* dest = NULL, bool boolify = false,
                        bool invert = false);

    /// Append a new node (specified by raw pointer) onto the end of the
    /// sequence that *this belongs to.  Return *this.
    ASTNode* append(ASTNode* newnode)
    {
        ASTNode* n = this;
        while (n->nextptr())
            n = n->nextptr();
        // Now n points to the last in the sequence
        n->m_next = newnode;
        return this;
    }

    /// Append a new node (specified by a reference-counted pointer)
    /// onto the end of the sequence that *this belongs to.  Return a
    /// reference-counted pointer to *this.
    ref append(ref& x)
    {
        append(x.get());
        return this;
    }

    /// Detach any 'next' nodes.
    void detach_next() { m_next.reset(); }

    /// Concatenate ASTNode sequences A and B, returning a raw pointer to
    /// the concatenated sequence.  This is robust to either A or B or
    /// both being NULL.
    friend ASTNode* concat(ASTNode* A, ASTNode* B)
    {
        if (A)  // A is valid, B may or may not be
            return B ? A->append(B) : A;
        else  // A not valid, so just go with B
            return B;
    }

    /// Reverse the order of the list.
    friend ASTNode::ref reverse(ASTNode::ref list);

    /// What source file was this parse node created from?
    ///
    ustring sourcefile() const { return m_sourcefile; }

    /// What line of the source file was this parse node created from?
    ///
    int sourceline() const { return m_sourceline; }

    void sourceline(int line) { m_sourceline = line; }

    template<typename... Args>
    void errorf(const char* format, const Args&... args) const
    {
        OSL_DASSERT(format && format[0]);
        error_impl(OIIO::Strutil::sprintf(format, args...));
    }

    /// Warning reporting
    template<typename... Args>
    void warningf(const char* format, const Args&... args) const
    {
        OSL_DASSERT(format && format[0]);
        warning_impl(OIIO::Strutil::sprintf(format, args...));
    }

    /// info reporting
    template<typename... Args>
    void infof(const char* format, const Args&... args) const
    {
        OSL_DASSERT(format && format[0]);
        info_impl(OIIO::Strutil::sprintf(format, args...));
    }

    /// message reporting
    template<typename... Args>
    void messagef(const char* format, const Args&... args) const
    {
        OSL_DASSERT(format && format[0]);
        message_impl(OIIO::Strutil::sprintf(format, args...));
    }

    bool is_lvalue() const { return m_is_lvalue; }

    /// Return a reference-counted pointer to the next node in the sequence.
    ///
    ref next() const { return m_next; }

    /// Return the raw pointer to the next node in the sequence.  Use
    /// with caution!
    ASTNode* nextptr() const { return m_next.get(); }

protected:
    void indent(std::ostream& out, int indentlevel = 0) const
    {
        while (indentlevel--)
            out << "    ";
    }

    /// A is the head of a list of nodes, traverse the list and call
    /// the print() method for each node in the list.
    static void printlist(std::ostream& out, const ref& A, int indentlevel)
    {
        for (const ASTNode* n = A.get(); n; n = n->nextptr())
            n->print(out, indentlevel);
    }

    /// A is the head of a list of nodes, traverse the list and compute
    /// the length of the list.
    static size_t listlength(const ref& A)
    {
        size_t len = 0;
        for (const ASTNode* n = A.get(); n; n = n->nextptr())
            ++len;
        return len;
    }

    /// A is the head of a list of nodes, return a pointer to the n-th
    /// node in the list (or NULL if the list isn't long enough).
    static ASTNode* list_nth(const ref& A, int n)
    {
        for (ASTNode* node = A.get(); node; node = node->nextptr(), --n)
            if (n == 0)
                return node;
        return NULL;
    }

    /// Flatten a list of nodes (headed by A) into a vector of node refs
    /// (vec).
    static void list_to_vec(const ref& A, std::vector<ref>& vec);

    /// Turn a vector of node refs into a list of nodes, returning its
    /// head.
    static ref vec_to_list(std::vector<ref>& vec);

    /// Generate a comma-separated string of data types of the list headed
    /// by node (for example, "float, int, string").
    static std::string list_to_types_string(const ASTNode* node);

    /// Return a pointer for the i-th child node, or nullptr if there is no
    /// such node.
    ASTNode* child(size_t i) const
    {
        return (i < m_children.size()) ? m_children[i].get() : NULL;
    }

    /// Add a new node to the list of children.
    void addchild(ASTNode* n) { m_children.emplace_back(n); }

    /// Add a new node to the list of children.
    void addchild(ref& n) { m_children.emplace_back(n); }

    /// Call the print() method of all the children of this node.
    ///
    void printchildren(std::ostream& out, int indentlevel = 0) const;

    /// Clear the children. Use with caution!
    void clearchildren() { m_children.clear(); }

    /// Follow a list of nodes, type checking each in turn, and return
    /// the type of the last one.
    static TypeSpec typecheck_list(ref node, TypeSpec expected = TypeSpec());

    /// Type check all the children of this node.
    ///
    void typecheck_children(TypeSpec expected = TypeSpec());

    /// Helper for check_arglist: simple case of checking a single arg type
    /// against the front of the formals list (which will be advanced).
    bool check_simple_arg(const TypeSpec& argtype, const char*& formals,
                          bool coerce);

    /// Type check a list (whose head is given by 'arg' against the list
    /// of expected types given in encoded form by 'formals'.
    bool check_arglist(const char* funcname, ref arg, const char* formals,
                       bool coerce = false, bool bind = true);

    /// Follow a list of nodes, generating code for each in turn, and return
    /// the Symbol* for the last thing generated.
    static Symbol* codegen_list(ref node, Symbol* dest = NULL);

    /// Generate code for all the children of this node.
    ///
    void codegen_children();

    /// Emit a single IR opcode -- append one op to the list of
    /// intermediate code, returning the label (address) of the new op.
    int emitcode(const char* opname, Symbol* arg0 = NULL, Symbol* arg1 = NULL,
                 Symbol* arg2 = NULL, Symbol* arg3 = NULL);

    /// Emit a single IR opcode -- append one op to the list of
    /// intermediate code, returning the label (address) of the new op.
    int emitcode(const char* opname, size_t nargs, Symbol** args);

    /// Coerce sym into being the desired type.  Maybe it already is, or
    /// maybe a temporary needs to be created.  Only do float->triple
    /// coercion if acceptfloat is false.
    Symbol* coerce(Symbol* sym, const TypeSpec& type, bool acceptfloat = false);

    /// Return the c_str giving a human-readable name of a type, fully
    /// accounting for exotic types like structs, etc.
    /// N.B.: just conveniently wraps the compiler's identical method.
    const char* type_c_str(const TypeSpec& type) const;

    /// Assign the struct variable named by srcsym to the struct
    /// variable named by dstsym by assigning each field individually.
    /// In the case of dstsym naming an array of structs, arrayindex
    /// should be a symbol holding the index of the individual array
    /// element that should be copied into.  If 'copywholearrays' is
    /// true, we are (perhaps recursively) copying entire arrays, of
    /// or within the struct, and intindex is the element number if we
    /// know it -- these two items let us take some interesting shortcuts
    /// with whole arrays (copyarray versus assigning elements). Pass
    /// paraminit=true if we're doing the assignment as init ops of a
    /// shader param.
    void codegen_assign_struct(StructSpec* structspec, ustring dstsym,
                               ustring srcsym, Symbol* arrayindex,
                               bool copywholearrays, int intindex,
                               bool paraminit);

    // Helper: generate code for an initializer list -- either a single
    // item to a scalar, or a list to an array.
    void codegen_initlist(ref init, TypeSpec type, Symbol* sym);

    // Special code generation for structure initializers.
    // It's in the ASTNode base class because it's used from mutiple
    // subclasses.
    Symbol* codegen_struct_initializers(ref init, Symbol* sym,
                                        bool is_constructor = false,
                                        Symbol* arrayindex  = nullptr);

    // Codegen an array assignment: lval[index] = src
    // If no index is provided the constant i is used.
    // Will return either src or a temporary that was codegened.
    Symbol* codegen_aassign(TypeSpec elemtype, Symbol* src, Symbol* lval,
                            Symbol* index, int i = 0);

    // Check whether the node's symbol destination is ok to write. Return
    // true if ok, return false and issue error or warning if not. (Used to
    // check for writing to read-only variables like non-output parameters.)
    bool check_symbol_writeability(ASTNode* var);

    // Helper for param_default_literals: generate the string that gives
    // the initialization of the literal value (and/or the default, if
    // init==NULL) and append it to 'out'.  Return whether the full
    // initialization is comprised only of literals (no init ops needed).
    bool one_default_literal(const Symbol* sym, ASTNode* init, std::string& out,
                             string_view separator = " ") const;

    void error_impl(string_view msg) const;
    void warning_impl(string_view msg) const;
    void info_impl(string_view msg) const;
    void message_impl(string_view msg) const;

protected:
    NodeType m_nodetype;          ///< Type of node this is
    ref m_next;                   ///< Next node in the list
    OSLCompilerImpl* m_compiler;  ///< Back-pointer to the compiler
    ustring m_sourcefile;  ///< Filename of source where the node came from
    int m_sourceline;      ///< Line number in source where the node came from
    std::vector<ref> m_children;  ///< Child nodes
    int m_op;                     ///< Operator selection
    TypeSpec m_typespec;          ///< Data type of this node
    bool m_is_lvalue;             ///< Is it an lvalue (assignable?)

private:
};



class ASTshader_declaration : public ASTNode {
public:
    ASTshader_declaration(OSLCompilerImpl* comp, int stype, ustring name,
                          ASTNode* form, ASTNode* stmts, ASTNode* meta);
    const char* nodetypename() const { return "shader_declaration"; }
    const char* childname(size_t i) const;
    void print(std::ostream& out, int indentlevel = 0) const;
    // TypeSpec typecheck (TypeSpec expected); // Use the default
    Symbol* codegen(Symbol* dest = NULL);

    ref metadata() const { return child(0); }
    ref formals() const { return child(1); }
    ref statements() const { return child(2); }
    ustring shadername() const { return m_shadername; }
    string_view shadertypename() const;

private:
    ustring m_shadername;
};



class ASTfunction_declaration : public ASTNode {
public:
    ASTfunction_declaration(OSLCompilerImpl* comp, TypeSpec type, ustring name,
                            ASTNode* form, ASTNode* stmts, ASTNode* meta,
                            int sourceline_start = -1);
    const char* nodetypename() const { return "function_declaration"; }
    const char* childname(size_t i) const;
    void print(std::ostream& out, int indentlevel = 0) const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* /*dest*/)
    {
        return nullptr;  // generates no code on its own
    }

    ref metadata() const { return child(0); }
    ref formals() const { return child(1); }
    ref statements() const { return child(2); }
    FunctionSymbol* func() const { return (FunctionSymbol*)m_sym; }

    bool is_builtin() const { return m_is_builtin; }
    void add_meta(ref meta);

private:
    ustring m_name;
    Symbol* m_sym;
    bool m_is_builtin;
};



class ASTvariable_declaration : public ASTNode {
public:
    ASTvariable_declaration(OSLCompilerImpl* comp, const TypeSpec& type,
                            ustring name, ASTNode* init, bool isparam,
                            bool ismeta, bool isoutput, bool initlist,
                            int sourceline_start = -1);
    const char* nodetypename() const;
    const char* childname(size_t i) const;
    void print(std::ostream& out, int indentlevel = 0) const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);

    ref init() const { return child(0); }
    ref meta() const { return child(1); }

    void add_meta(ref meta)
    {
        while (nchildren() < 2)
            addchild(NULL);
        m_children[1] = meta;  // beware changing the order!
    }

    Symbol* sym() const { return m_sym; }
    ustring name() const { return m_name; }

    bool is_output() const { return m_isoutput; }

    /// For shader params, generate the string that gives the
    /// initialization of literal values and place it in 'out'.
    /// Return whether the full initialization is comprised only of
    /// literals (and no init ops are needed).
    bool param_default_literals(const Symbol* sym, ASTNode* init,
                                std::string& out,
                                string_view separator = " ") const;

    void codegen_initializer(ref init, Symbol* sym);

    void register_struct_init(ustring name, ASTNode* init)
    {
        m_struct_field_inits.emplace_back(name, init);
    }

private:
    ustring m_name;     ///< Name of the symbol (unmangled)
    Symbol* m_sym;      ///< Ptr to the symbol this declares
    bool m_isparam;     ///< Is this a parameter?
    bool m_isoutput;    ///< Is this an output parameter?
    bool m_ismetadata;  ///< Is this declaration a piece of metadata?
    bool m_initlist;    ///< Was initialized with a list (versus just an expr)
    // For structures, map the field names (even recursively!) to the
    // specific initializers.
    typedef std::pair<ustring, ASTNode*> NamedInit;
    std::vector<NamedInit> m_struct_field_inits;
};



class ASTvariable_ref : public ASTNode {
public:
    ASTvariable_ref(OSLCompilerImpl* comp, ustring name);
    const char* nodetypename() const { return "variable_ref"; }
    const char* childname(size_t /*i*/) const { return ""; }  // no children
    void print(std::ostream& out, int indentlevel = 0) const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);
    ustring name() const { return m_name; }
    std::string mangled() const { return m_sym->mangled(); }
    Symbol* sym() const { return m_sym; }

private:
    ustring m_name;
    Symbol* m_sym;
};



class ASTpreincdec : public ASTNode {
public:
    ASTpreincdec(OSLCompilerImpl* comp, int op, ASTNode* expr);
    const char* nodetypename() const
    {
        return m_op == Incr ? "preincrement" : "predecrement";
    }
    const char* childname(size_t i) const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);

    ref var() const { return child(0); }
};



class ASTpostincdec : public ASTNode {
public:
    ASTpostincdec(OSLCompilerImpl* comp, int op, ASTNode* expr);
    const char* nodetypename() const
    {
        return m_op == Incr ? "postincrement" : "postdecrement";
    }
    const char* childname(size_t i) const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);

    ref var() const { return child(0); }
};



class ASTindex : public ASTNode {
public:
    // ASTindex (OSLCompilerImpl *comp, ASTNode *expr, ASTNode *index);
    // ASTindex (OSLCompilerImpl *comp, ASTNode *expr, ASTNode *index, ASTNode *index2);
    ASTindex(OSLCompilerImpl* comp, ASTNode* expr, ASTNode* index,
             ASTNode* index2 = nullptr, ASTNode* index3 = nullptr);
    const char* nodetypename() const { return "index"; }
    const char* childname(size_t i) const;
    TypeSpec typecheck(TypeSpec expected = TypeSpec());
    Symbol* codegen(Symbol* dest = NULL);

    /// Special code generation that, when it generates the code for the
    /// indices, stores those in the extra variables.
    Symbol* codegen(Symbol* dest, Symbol*& ind, Symbol*& ind2, Symbol*& ind3);

    /// Special code generation of assignment of src to this indexed
    /// location. The ind1, ind2, ind3 symbols are overrides; if not
    /// provided, it will use index(), index2(), index3().
    void codegen_assign(Symbol* src, Symbol* ind = NULL, Symbol* ind2 = NULL,
                        Symbol* ind3 = NULL);

    /// Copy one element of the struct array named by srcname into the
    /// struct destname.
    void codegen_copy_struct_array_element(StructSpec* structspec,
                                           ustring destname, ustring srcname,
                                           Symbol* index);

    ref lvalue() const { return child(0); }
    ref index() const { return child(1); }
    ref index2() const { return child(2); }
    ref index3() const { return child(3); }
};



class ASTstructselect : public ASTNode {
public:
    ASTstructselect(OSLCompilerImpl* comp, ASTNode* expr, ustring field);
    const char* nodetypename() const { return "structselect"; }
    const char* childname(size_t i) const;
    void print(std::ostream& out, int indentlevel = 0) const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);

    /// Special code generation of assignment of src to this structure
    /// field.
    void codegen_assign(Symbol* dest, Symbol* src);

    ref lvalue() const { return child(0); }
    ustring field() const { return m_field; }
    ustring fieldname() const { return m_fieldname; }
    Symbol* fieldsym() const { return m_fieldsym; }
    int fieldid() const { return m_fieldid; }
    ASTindex* compindex() const { return (ASTindex*)m_compindex.get(); }

private:
    Symbol* find_fieldsym(int& structid, int& fieldid);
    static void find_structsym(ASTNode* structnode, ustring& structname,
                               TypeSpec& structtype);
    Symbol* codegen_index();

    ustring m_field;      ///< Name of the field
    int m_structid;       ///< index of the structure
    int m_fieldid;        ///< index of the field within the structure
    ustring m_fieldname;  ///< Name of the field variable
    Symbol* m_fieldsym;   ///< Symbol of the field variable
    ref m_compindex;      ///< Redirection of named component index
};



class ASTconditional_statement : public ASTNode {
public:
    ASTconditional_statement(OSLCompilerImpl* comp, ASTNode* cond,
                             ASTNode* truestmt, ASTNode* falsestmt = NULL)
        : ASTNode(conditional_statement_node, comp, 0, cond, truestmt,
                  falsestmt)
    {
    }

    const char* nodetypename() const { return "conditional_statement"; }
    const char* childname(size_t i) const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);

    ref cond() const { return child(0); }
    ref truestmt() const { return child(1); }
    ref falsestmt() const { return child(2); }
};



class ASTloop_statement : public ASTNode {
public:
    enum LoopType { LoopWhile, LoopDo, LoopFor };

    ASTloop_statement(OSLCompilerImpl* comp, LoopType looptype, ASTNode* init,
                      ASTNode* cond, ASTNode* iter, ASTNode* stmt);

    const char* nodetypename() const { return "loop_statement"; }
    const char* childname(size_t i) const;
    const char* opname() const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);

    ref init() const { return child(0); }
    ref cond() const { return child(1); }
    ref iter() const { return child(2); }
    ref stmt() const { return child(3); }
};



class ASTloopmod_statement : public ASTNode {
public:
    enum LoopModType { LoopModBreak, LoopModContinue };

    ASTloopmod_statement(OSLCompilerImpl* comp, LoopModType loopmodtype)
        : ASTNode(loopmod_statement_node, comp, loopmodtype)
    {
    }

    const char* nodetypename() const { return "loopmod_statement"; }
    const char* childname(size_t i) const;
    const char* opname() const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);
};



class ASTreturn_statement : public ASTNode {
public:
    ASTreturn_statement(OSLCompilerImpl* comp, ASTNode* expr)
        : ASTNode(return_statement_node, comp, 0, expr)
    {
    }

    const char* nodetypename() const { return "return_statement"; }
    const char* childname(size_t i) const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);

    ref expr() const { return child(0); }
};



class ASTtype_constructor : public ASTNode {
protected:
    ASTtype_constructor(NodeType n, OSLCompilerImpl* c, TypeSpec t, ASTNode* a)
        : ASTNode(n, c, Nothing, a)
    {
        m_typespec = t;
    }

public:
    ASTtype_constructor(OSLCompilerImpl* comp, TypeSpec typespec, ASTNode* args)
        : ASTtype_constructor(type_constructor_node, comp, typespec, args)
    {
    }

    const char* nodetypename() const { return "type_constructor"; }
    const char* childname(size_t i) const;
    Symbol* codegen(Symbol* dest = NULL);

    ref args() const { return child(0); }

    // Typecheck construction of expected against args()
    // Optionally ignoring errors and binding any init-list arguments to the
    // required type.
    TypeSpec typecheck(TypeSpec expected, bool error, bool bind = true);

    // Typecheck construction of m_typespec against args()
    TypeSpec typecheck(TypeSpec /*expected*/)
    {
        return typecheck(m_typespec, true, true);
    }
};


class ASTcompound_initializer : public ASTtype_constructor {
    bool m_ctor;

    TypeSpec typecheck(TypeSpec expected, unsigned mode);

public:
    ASTcompound_initializer(OSLCompilerImpl* comp, ASTNode* exprlist);
    const char* nodetypename() const { return "compound_initializer"; }
    const char* childname(size_t i) const;
    Symbol* codegen(Symbol* dest = NULL);

    ref initlist() const { return child(0); }

    bool canconstruct() const { return m_ctor; }
    void canconstruct(bool b) { m_ctor = b; }

    TypeSpec typecheck(TypeSpec expected) { return typecheck(expected, 0); }

    // Helper for typechecking an initlist or structure.
    class TypeAdjuster;
};



class ASTassign_expression : public ASTNode {
public:
    ASTassign_expression(OSLCompilerImpl* comp, ASTNode* var, Operator op,
                         ASTNode* expr);
    const char* nodetypename() const { return "assign_expression"; }
    const char* childname(size_t i) const;
    const char* opname() const;
    const char* opword() const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);

    ref var() const { return child(0); }
    ref expr() const { return child(1); }
};



class ASTunary_expression : public ASTNode {
public:
    ASTunary_expression(OSLCompilerImpl* comp, int op, ASTNode* expr);

    const char* nodetypename() const { return "unary_expression"; }
    const char* childname(size_t i) const;
    const char* opname() const;
    const char* opword() const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);

    ref expr() const { return child(0); }

private:
    FunctionSymbol* m_function_overload = nullptr;
};



class ASTbinary_expression : public ASTNode {
public:
    ASTbinary_expression(OSLCompilerImpl* comp, Operator op, ASTNode* left,
                         ASTNode* right);

    const char* nodetypename() const { return "binary_expression"; }
    const char* childname(size_t i) const;
    const char* opname() const;
    const char* opword() const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);

    ref left() const { return child(0); }
    ref right() const { return child(1); }

private:
    FunctionSymbol* m_function_overload = nullptr;

    // Special code generation for short-circuiting logical ops
    Symbol* codegen_logic(Symbol* dest);
};



class ASTternary_expression : public ASTNode {
public:
    ASTternary_expression(OSLCompilerImpl* comp, ASTNode* cond,
                          ASTNode* trueexpr, ASTNode* falseexpr)
        : ASTNode(ternary_expression_node, comp, 0, cond, trueexpr, falseexpr)
    {
    }

    const char* nodetypename() const { return "ternary_expression"; }
    const char* childname(size_t i) const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);

    ref cond() const { return child(0); }
    ref trueexpr() const { return child(1); }
    ref falseexpr() const { return child(2); }
};



class ASTcomma_operator : public ASTNode {
public:
    ASTcomma_operator(OSLCompilerImpl* comp, ASTNode* exprlist)
        : ASTNode(comma_operator_node, comp, Nothing, exprlist)
    {
    }

    const char* nodetypename() const { return "comma_operator"; }
    const char* childname(size_t /*i*/) const { return "expression_list"; }
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);

    ref expr() const { return child(0); }
};



class ASTtypecast_expression : public ASTNode {
public:
    ASTtypecast_expression(OSLCompilerImpl* comp, TypeSpec typespec,
                           ASTNode* expr)
        : ASTNode(typecast_expression_node, comp, 0, expr)
    {
        m_typespec = typespec;
    }

    const char* nodetypename() const { return "typecast_expression"; }
    const char* childname(size_t i) const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);

    ref expr() const { return child(0); }
};



class ASTfunction_call : public ASTNode {
public:
    ASTfunction_call(OSLCompilerImpl* comp, ustring name, ASTNode* args,
                     FunctionSymbol* funcsym = nullptr);
    const char* nodetypename() const { return "function_call"; }
    const char* childname(size_t i) const;
    const char* opname() const;
    void print(std::ostream& out, int indentlevel = 0) const;
    TypeSpec typecheck(TypeSpec expected);
    Symbol* codegen(Symbol* dest = NULL);

    FunctionSymbol* func() const { return (FunctionSymbol*)m_sym; }
    ref args() const { return child(0); }

    /// Is it a struct constructor?
    bool is_struct_ctr() const { return m_sym && m_sym->is_structure(); }

    /// Is it a user-defined function (as opposed to an OSL built-in)?
    ///
    bool is_user_function() const
    {
        return !is_struct_ctr() && user_function()
               && !user_function()->is_builtin();
    }

    /// Pointer to the ASTfunction_declaration node that defines the user
    /// function, or NULL if it's not a user-defined function.
    ASTfunction_declaration* user_function() const
    {
        return func() ? (ASTfunction_declaration*)func()->node() : nullptr;
    }

private:
    /// Handle all the special cases for built-ins.  This includes
    /// irregular patterns of which args are read vs written, special
    /// checks for printf- and texture-like, etc.
    void typecheck_builtin_specialcase();

    /// Make sure the printf-like format string matches the list of
    /// arguments pointed to by arg.  If ok, return true, otherwise
    /// return false and call an appropriate error().
    bool typecheck_printf_args(const char* format, ASTNode* arg);

    /// Handle "funcion calls" that are really struct ctrs.
    TypeSpec typecheck_struct_constructor();

    /// Is the argument number 'arg' read by the op?
    ///
    bool argread(int arg) const;
    /// Is the argument number 'arg' written by the op?
    ///
    bool argwrite(int arg) const;
    /// Declare that argument number 'arg' is read by this op.
    ///
    void argread(int arg, bool val)
    {
        if (arg < 32) {
            if (val)
                m_argread |= (1 << arg);
            else
                m_argread &= ~(1 << arg);
        }
    }
    /// Declare that argument number 'arg' is written by this op.
    ///
    void argwrite(int arg, bool val)
    {
        if (arg < 32) {
            if (val)
                m_argwrite |= (1 << arg);
            else
                m_argwrite &= ~(1 << arg);
        }
    }
    /// Declare that argument number 'arg' is only written (not read!) by
    /// this op.
    void argwriteonly(int arg)
    {
        argread(arg, false);
        argwrite(arg, true);
    }
    /// Declare optional arguments as outputs (write only) by this op.
    ///
    void mark_optional_output(int firstopt, const char** tags);
    /// Declare that argument number 'arg' takes derivatives.
    ///
    void argtakesderivs(int arg, bool val)
    {
        if (arg < 32) {
            if (val)
                m_argtakesderivs |= (1 << arg);
            else
                m_argtakesderivs &= ~(1 << arg);
        }
    }

    void codegen_arg(SymbolPtrVec& argdest, SymbolPtrVec& index,
                     SymbolPtrVec& index1, SymbolPtrVec& index2, int argnum,
                     ASTNode* arg, ASTNode* form, const TypeSpec& formaltype,
                     bool writearg, bool& indexed_output_params);

    /// Call compiler->struct_field_pair for each field in the struct.
    ///
    void struct_pair_all_fields(StructSpec* structspec, ustring formal,
                                ustring actual, Symbol* arrayindex = NULL);

    ustring m_name;                 ///< Name of the function being called
    Symbol* m_sym;                  ///< Symbol of the function
    FunctionSymbol* m_poly;         ///< The specific polymorphic variant
    unsigned int m_argread;         ///< Bit field - which args are read
    unsigned int m_argwrite;        ///< Bit field - which args are written
    unsigned int m_argtakesderivs;  ///< Bit field - which args take derivs
};



class ASTliteral : public ASTNode {
public:
    ASTliteral(OSLCompilerImpl* comp, int i)
        : ASTNode(literal_node, comp), m_i(i)
    {
        m_typespec = TypeDesc::TypeInt;
    }

    ASTliteral(OSLCompilerImpl* comp, float f)
        : ASTNode(literal_node, comp), m_f(f)
    {
        m_typespec = TypeDesc::TypeFloat;
    }

    ASTliteral(OSLCompilerImpl* comp, ustring s)
        : ASTNode(literal_node, comp), m_s(s)
    {
        m_typespec = TypeDesc::TypeString;
    }

    const char* nodetypename() const { return "literal"; }
    const char* childname(size_t i) const;
    void print(std::ostream& out, int indentlevel) const;
    TypeSpec typecheck(TypeSpec /*expected*/) { return m_typespec; }
    Symbol* codegen(Symbol* dest = NULL);

    const char* strval() const { return m_s.c_str(); }
    int intval() const { return m_i; }
    float floatval() const { return m_typespec.is_int() ? (float)m_i : m_f; }
    ustring ustrval() const { return m_s; }

    void negate()
    {
        m_i = -m_i;
        m_f = -m_f;
    }

private:
    ustring m_s;
    int m_i;
    float m_f;
};



};  // namespace pvt

OSL_NAMESPACE_EXIT
