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

#ifndef OSL_AST_H
#define OSL_AST_H

#include "OpenImageIO/ustring.h"
#include "OpenImageIO/thread.h"
#include "OpenImageIO/refcnt.h"
#include "oslcomp.h"
#include "symtab.h"


class oslFlexLexer;
extern int oslparse ();


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {


// Forward declarations
class OSLCompilerImpl;
class Symbol;
class TypeSpec;



/// Base node for an abstract syntax tree for the OSL parser.
///
class ASTNode : public RefCnt {
public:
    typedef intrusive_ptr<ASTNode> ref;  ///< Ref-counted pointer to an ASTNode

    /// List of all the types of AST nodes.
    ///
    enum NodeType {
        unknown_node, shader_declaration_node, function_declaration_node,
        variable_declaration_node,
        variable_ref_node, preincdec_node, postincdec_node,
        index_node, structselect_node,
        conditional_statement_node,
        loop_statement_node, loopmod_statement_node, return_statement_node,
        binary_expression_node, unary_expression_node,
        assign_expression_node, ternary_expression_node, 
        typecast_expression_node, type_constructor_node,
        function_call_node,
        literal_node,
        _last_node
    };

    enum Operator { Nothing=0, Decr, Incr, 
                    Assign, Mul, Div, Add, Sub, Mod,
                    Equal, NotEqual, Greater, Less, GreaterEqual, LessEqual, 
                    BitAnd, BitOr, Xor, Compl,
                    And, Or, Not, ShiftLeft, ShiftRight };

    ASTNode (NodeType nodetype, OSLCompilerImpl *compiler);

    ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op);
    ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op,
             ASTNode *a);
    ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op,
             ASTNode *a, ASTNode *b);
    ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op,
             ASTNode *a, ASTNode *b, ASTNode *c);
    ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op,
             ASTNode *a, ASTNode *b, ASTNode *c, ASTNode *d);

    virtual ~ASTNode () { }

    /// Print a text description of this node (and its children) to the
    /// console, for debugging.
    virtual void print (std::ostream &out, int indentlevel = 0) const;

    /// What type of node is this?
    ///
    NodeType nodetype () const { return m_nodetype; }

    /// Name of this type of node
    ///
    virtual const char *nodetypename () const = 0;

    /// What data type is this node?
    ///
    const TypeSpec &typespec () const { return m_typespec; }

    /// Name of the op, if any, or NULL.
    ///
    virtual const char *opname () const { return NULL; }

    /// Name of the child node
    ///
    virtual const char *childname (size_t i) const = 0;

    /// Type check the node, return its type.  Optionally an "expected
    /// type" may be passed down, conveying any requirements or
    /// coercion.  The default (base class) implementation just type
    /// checks all the child nodes and makes this node's type be the
    /// expected if it is unknown, but doens't change it if it's not
    /// unknown.
    virtual TypeSpec typecheck (TypeSpec expected = TypeSpec());

    /// Generate IR code for this node and its children, return the
    /// symbol where the result is stored (if applicable, otherwise
    /// NULL).  The optional 'dest' is a request for the caller to store
    /// the results in a particular place (which it can't always do, of
    /// course).
    virtual Symbol *codegen (Symbol *dest = NULL);

    /// Generate IR code for this node make sure it's boiled down to an
    /// int (i.e. if not already an int, generate one that's 1 if the
    /// original code was non-zero or non-empty string).  The optional
    /// 'dest' is a request for the caller to store the results in a
    /// particular place (which it can't always do, of course).  If
    /// 'boolify' is true and the normal code generates an int, convert
    /// it to a 0 or 1 value.  If 'invert' is true, invert the result.
    Symbol *codegen_int (Symbol *dest = NULL,
                         bool boolify=false, bool invert=false);

    /// Append a new node (specified by raw pointer) onto the end of the
    /// sequence that *this belongs to.  Return *this.
    ASTNode *append (ASTNode *newnode) {
        ASTNode *n = this;
        while (n->nextptr())
            n = n->nextptr();
        // Now n points to the last in the sequence
        n->m_next = newnode;
        return this;
    }

    /// Append a new node (specified by a reference-counted pointer)
    /// onto the end of the sequence that *this belongs to.  Return an
    /// reference-counted pointer to *this.
    ref append (ref &x) { append (x.get()); return this; }

    /// Concatenate ASTNode sequences A and B, returning a raw pointer to
    /// the concatenated sequence.  This is robust to either A or B or
    /// both being NULL.
    friend ASTNode *concat (ASTNode *A, ASTNode *B) {
        if (A)  // A is valid, B may or may not be
            return B ? A->append (B) : A;
        else    // A not valid, so just go with B
            return B;
    }

    /// What source file was this parse node created from?
    ///
    ustring sourcefile () const { return m_sourcefile; }

    /// What line of the source file was this parse node created from?
    ///
    int sourceline () const { return m_sourceline; }

    void sourceline (int line) { m_sourceline = line; }

    void error (const char *format, ...);

    bool is_lvalue () const { return m_is_lvalue; }

    /// Return a reference-counted pointer to the next node in the sequence.
    ///
    ref next () const { return m_next; }

    /// Return the raw pointer to the next node in the sequence.  Use
    /// with caution!
    ASTNode *nextptr () const { return m_next.get(); }

protected:
    void indent (std::ostream &out, int indentlevel=0) const {
        while (indentlevel--)
            out << "    ";
    }

    /// A is the head of a list of nodes, traverse the list and call
    /// the print() method for each node in the list.
    static void printlist (std::ostream &out, const ref &A, int indentlevel) {
        for (const ASTNode *n = A.get();  n;  n = n->nextptr())
            n->print (out, indentlevel);
    }

    /// A is the head of a list of nodes, traverse the list and call
    /// the print() method for each node in the list.
    static size_t listlength (const ref &A) {
        size_t len = 0;
        for (const ASTNode *n = A.get();  n;  n = n->nextptr())
            ++len;
        return len;
    }

    /// Return the number of child nodes.
    ///
    size_t nchildren () const { return m_children.size(); }

    /// Return the i-th child node, or NULL if there is no such node
    ///
    ASTNode *child (size_t i) const {
        return (i < m_children.size()) ? m_children[i].get() : NULL;
    }

    /// Add a new node to the list of children.
    ///
    void addchild (ASTNode *n) { m_children.push_back (n); }

    /// Call the print() method of all the children of this node.
    ///
    void printchildren (std::ostream &out, int indentlevel = 0) const;

    /// Follow a list of nodes, type checking each in turn, and return
    /// the type of the last one.
    static TypeSpec typecheck_list (ref node, TypeSpec expected = TypeSpec());

    /// Type check all the children of this node.
    ///
    void typecheck_children (TypeSpec expected = TypeSpec());

    /// Type check a list (whose head is given by 'arg' against the list
    /// of expected types given in encoded form by 'formals'.
    bool check_arglist (const char *funcname, ref arg,
                        const char *formals, bool coerce=false);

    /// Follow a list of nodes, generating code for each in turn.
    ///
    static void codegen_list (ref node);

    /// Generate code for all the children of this node.
    ///
    void codegen_children ();

    /// Emit a single IR opcode -- append one op to the list of
    /// intermediate code, returning the label (address) of the new op.
    int emitcode (const char *opname, Symbol *arg0=NULL, 
                  Symbol *arg1=NULL, Symbol *arg2=NULL, Symbol *arg3=NULL);

    /// Emit a single IR opcode -- append one op to the list of
    /// intermediate code, returning the label (address) of the new op.
    int emitcode (const char *opname, size_t nargs, Symbol **args);

    /// Coerce sym into being the desired type.  Maybe it already is, or
    /// maybe a temporary needs to be created.  Only do float->triple
    /// coercion if acceptfloat is false.
    Symbol *coerce (Symbol *sym, const TypeSpec &type, bool acceptfloat=false);

protected:
    NodeType m_nodetype;          ///< Type of node this is
    ref m_next;                   ///< Next node in the list
    OSLCompilerImpl *m_compiler;  ///< Back-pointer to the compiler
    ustring m_sourcefile;   ///< Filename of source where the node came from
    int m_sourceline;       ///< Line number in source where the node came from
    std::vector<ref> m_children;  ///< Child nodes
    int m_op;                     ///< Operator selection
    TypeSpec m_typespec;          ///< Data type of this node
    bool m_is_lvalue;             ///< Is it an lvalue (assignable?)

private:
};



class ASTshader_declaration : public ASTNode
{
public:
    ASTshader_declaration (OSLCompilerImpl *comp, int stype, ustring name,
                           ASTNode *form, ASTNode *stmts, ASTNode *meta) 
        : ASTNode (shader_declaration_node, comp, stype, meta, form, stmts),
          m_shadername(name)
    { }
    const char *nodetypename () const { return "shader_declaration"; }
    const char *childname (size_t i) const;
    void print (std::ostream &out, int indentlevel=0) const;
    // TypeSpec typecheck (TypeSpec expected); // Use the default
    Symbol *codegen (Symbol *dest = NULL);

    ref metadata () const { return child (0); }
    ref formals () const { return child (1); }
    ref statements () const { return child (2); }
    ustring shadername () const { return m_shadername; }
    const char *shadertypename () const;

private:
    ustring m_shadername;
};



class ASTfunction_declaration : public ASTNode
{
public:
    ASTfunction_declaration (OSLCompilerImpl *comp, TypeSpec type, ustring name,
                             ASTNode *form, ASTNode *stmts, ASTNode *meta=NULL);
    const char *nodetypename () const { return "function_declaration"; }
    const char *childname (size_t i) const;
    void print (std::ostream &out, int indentlevel=0) const;
    TypeSpec typecheck (TypeSpec expected);

    ref metadata () const { return child (0); }
    ref formals () const { return child (1); }
    ref statements () const { return child (2); }
    FunctionSymbol *func () const { return (FunctionSymbol *)m_sym; }

private:
    ustring m_name;
    Symbol *m_sym;
};



class ASTvariable_declaration : public ASTNode
{
public:
    ASTvariable_declaration (OSLCompilerImpl *comp, const TypeSpec &type,
                             ustring name, ASTNode *init, bool isparam=false,
                             bool ismeta=false);
    const char *nodetypename () const;
    const char *childname (size_t i) const;
    void print (std::ostream &out, int indentlevel=0) const;
    TypeSpec typecheck (TypeSpec expected);
    Symbol *codegen (Symbol *dest = NULL);

    ref init () const { return child (0); }
    ref meta () const { return child (1); }

    void make_param (bool param=true) { m_isparam = param; }
    void make_output (bool out=true) {
        m_isoutput = out;
        if (out && m_sym->symtype() == SymTypeParam)
            m_sym->symtype (SymTypeOutputParam);
    }
    void make_meta (bool meta=true) { m_ismetadata = meta; }

    void add_meta (ASTNode *meta) {
        while (nchildren() < 2)
            addchild (NULL);
        m_children[1] = meta;  // beware changing the order!
    }

    Symbol *sym () const { return m_sym; }
    ustring name () const { return m_name; }

    bool is_output () const { return m_isoutput; }

private:
    ustring m_name;
    Symbol *m_sym;
    bool m_isparam;
    bool m_isoutput;
    bool m_ismetadata;
};



class ASTvariable_ref : public ASTNode
{
public:
    ASTvariable_ref (OSLCompilerImpl *comp, ustring name);
    const char *nodetypename () const { return "variable_ref"; }
    const char *childname (size_t i) const { return ""; } // no children
    void print (std::ostream &out, int indentlevel=0) const;
    TypeSpec typecheck (TypeSpec expected);
    Symbol *codegen (Symbol *dest = NULL);
private:
    ustring m_name;
    Symbol *m_sym;
};



class ASTpreincdec : public ASTNode
{
public:
    ASTpreincdec (OSLCompilerImpl *comp, int op, ASTNode *expr)
        : ASTNode (preincdec_node, comp, op, expr)
    { }
    const char *nodetypename () const { return m_op==Incr ? "preincrement" : "predecrement"; }
    const char *childname (size_t i) const;
    TypeSpec typecheck (TypeSpec expected);
    Symbol *codegen (Symbol *dest = NULL);

    ref var () const { return child (0); }
};



class ASTpostincdec : public ASTNode
{
public:
    ASTpostincdec (OSLCompilerImpl *comp, int op, ASTNode *expr)
        : ASTNode (postincdec_node, comp, op, expr)
    { }
    const char *nodetypename () const { return m_op==Incr ? "postincrement" : "postdecrement"; }
    const char *childname (size_t i) const;
    TypeSpec typecheck (TypeSpec expected);
    Symbol *codegen (Symbol *dest = NULL);

    ref var () const { return child (0); }
};



class ASTindex : public ASTNode
{
public:
    ASTindex (OSLCompilerImpl *comp, ASTNode *expr, ASTNode *index)
        : ASTNode (index_node, comp, 0, expr, index)
    { }
    ASTindex (OSLCompilerImpl *comp, ASTNode *expr, ASTNode *index, ASTNode *index2)
        : ASTNode (index_node, comp, 0, expr, index, index2)
    { }
    ASTindex (OSLCompilerImpl *comp, ASTNode *expr, ASTNode *index,
              ASTNode *index2, ASTNode *index3)
        : ASTNode (index_node, comp, 0, expr, index, index2, index3)
    { }
    const char *nodetypename () const { return "index"; }
    const char *childname (size_t i) const;
    TypeSpec typecheck (TypeSpec expected = TypeSpec());
    Symbol *codegen (Symbol *dest = NULL);

    /// Special code generation that, when it generates the code for the
    /// indices, stores those in the extra variables.
    Symbol *codegen (Symbol *dest, Symbol * &ind,
                     Symbol * &ind2, Symbol *&ind3);

    /// Special code generation of assignment of src to this indexed location
    ///
    void codegen_assign (Symbol *src, Symbol *ind = NULL,
                         Symbol *ind2 = NULL, Symbol *ind3 = NULL);

    ref lvalue () const { return child (0); }
    ref index () const { return child (1); }
    ref index2 () const { return child (2); }
    ref index3 () const { return child (3); }
};



class ASTstructselect : public ASTNode
{
public:
    ASTstructselect (OSLCompilerImpl *comp, ASTNode *expr, ustring field)
        : ASTNode (structselect_node, comp, 0, expr), m_field(field)
    { }
    const char *nodetypename () const { return "structselect"; }
    const char *childname (size_t i) const;
    void print (std::ostream &out, int indentlevel=0) const;
    TypeSpec typecheck (TypeSpec expected);

    ref lvalue () const { return child (0); }
    ustring field () const { return m_field; }
private:
    ustring m_field;
};



class ASTconditional_statement : public ASTNode
{
public:
    ASTconditional_statement (OSLCompilerImpl *comp, ASTNode *cond,
                              ASTNode *truestmt, ASTNode *falsestmt=NULL)
        : ASTNode (conditional_statement_node, comp, 0, 
                   cond, truestmt, falsestmt)
    { }

    const char *nodetypename () const { return "conditional_statement"; }
    const char *childname (size_t i) const;
    TypeSpec typecheck (TypeSpec expected);
    Symbol *codegen (Symbol *dest = NULL);

    ref cond () const { return child (0); }
    ref truestmt () const { return child (1); }
    ref falsestmt () const { return child (2); }
};



class ASTloop_statement : public ASTNode
{
public:
    enum LoopType {
        LoopWhile, LoopDo, LoopFor
    };

    ASTloop_statement (OSLCompilerImpl *comp, LoopType looptype, ASTNode *init,
                       ASTNode *cond, ASTNode *iter, ASTNode *stmt)
        : ASTNode (loop_statement_node, comp, looptype, init, cond, iter, stmt)
    { }

    const char *nodetypename () const { return "loop_statement"; }
    const char *childname (size_t i) const;
    const char *opname () const;
    TypeSpec typecheck (TypeSpec expected);
    Symbol *codegen (Symbol *dest = NULL);

    ref init () const { return child (0); }
    ref cond () const { return child (1); }
    ref iter () const { return child (2); }
    ref stmt () const { return child (3); }
};



class ASTloopmod_statement : public ASTNode
{
public:
    enum LoopModType {
        LoopModBreak, LoopModContinue
    };

    ASTloopmod_statement (OSLCompilerImpl *comp, LoopModType loopmodtype)
        : ASTNode (loopmod_statement_node, comp, loopmodtype)
    { }

    const char *nodetypename () const { return "loopmod_statement"; }
    const char *childname (size_t i) const;
    const char *opname () const;
    TypeSpec typecheck (TypeSpec expected) { return ASTNode::typecheck(expected); /* FIXME */ }
};



class ASTreturn_statement : public ASTNode
{
public:
    ASTreturn_statement (OSLCompilerImpl *comp, ASTNode *expr)
        : ASTNode (return_statement_node, comp, 0, expr)
    { }

    const char *nodetypename () const { return "return_statement"; }
    const char *childname (size_t i) const;
    TypeSpec typecheck (TypeSpec expected);
    Symbol *codegen (Symbol *dest = NULL);

    ref expr () const { return child (0); }
};



class ASTassign_expression : public ASTNode
{
public:
    ASTassign_expression (OSLCompilerImpl *comp, ASTNode *var, Operator op,
                          ASTNode *expr);
    const char *nodetypename () const { return "assign_expression"; }
    const char *childname (size_t i) const;
    const char *opname () const;
    const char *opword () const;
    TypeSpec typecheck (TypeSpec expected);
    Symbol *codegen (Symbol *dest = NULL);

    ref var () const { return child (0); }
    ref expr () const { return child (1); }
};



class ASTunary_expression : public ASTNode
{
public:
    ASTunary_expression (OSLCompilerImpl *comp, int op, ASTNode *expr)
        : ASTNode (unary_expression_node, comp, op, expr)
    { }

    const char *nodetypename () const { return "unary_expression"; }
    const char *childname (size_t i) const;
    const char *opname () const;
    const char *opword () const;
    TypeSpec typecheck (TypeSpec expected);
    Symbol *codegen (Symbol *dest = NULL);

    ref expr () const { return child (0); }
};



class ASTbinary_expression : public ASTNode
{
public:
    ASTbinary_expression (OSLCompilerImpl *comp, Operator op,
                          ASTNode *left, ASTNode *right)
        : ASTNode (binary_expression_node, comp, op, left, right)
    { }

    const char *nodetypename () const { return "binary_expression"; }
    const char *childname (size_t i) const;
    const char *opname () const;
    const char *opword () const;
    TypeSpec typecheck (TypeSpec expected);
    Symbol *codegen (Symbol *dest = NULL);

    ref left () const { return child (0); }
    ref right () const { return child (1); }
private:
    // Special code generation for short-circuiting logical ops
    Symbol *codegen_logic (Symbol *dest);
};



class ASTternary_expression : public ASTNode
{
public:
    ASTternary_expression (OSLCompilerImpl *comp, ASTNode *cond,
                           ASTNode *trueexpr, ASTNode *falseexpr)
        : ASTNode (ternary_expression_node, comp, 0, 
                   cond, trueexpr, falseexpr)
    { }

    const char *nodetypename () const { return "ternary_expression"; }
    const char *childname (size_t i) const;
    TypeSpec typecheck (TypeSpec expected);

    ref cond () const { return child (0); }
    ref trueexpr () const { return child (1); }
    ref falseexpr () const { return child (2); }
};



class ASTtypecast_expression : public ASTNode
{
public:
    ASTtypecast_expression (OSLCompilerImpl *comp, TypeSpec typespec,
                            ASTNode *expr)
        : ASTNode (typecast_expression_node, comp, 0, expr)
    {
        m_typespec = typespec;
    }

    const char *nodetypename () const { return "typecast_expression"; }
    const char *childname (size_t i) const;
    TypeSpec typecheck (TypeSpec expected);
    Symbol *codegen (Symbol *dest = NULL);

    ref expr () const { return child (0); }
};



class ASTtype_constructor : public ASTNode
{
public:
    ASTtype_constructor (OSLCompilerImpl *comp, TypeSpec typespec,
                         ASTNode *args)
        : ASTNode (type_constructor_node, comp, 0, args)
    {
        m_typespec = typespec;
    }

    const char *nodetypename () const { return "type_constructor"; }
    const char *childname (size_t i) const;
    TypeSpec typecheck (TypeSpec expected);
    Symbol *codegen (Symbol *dest = NULL);

    ref args () const { return child (0); }
};



class ASTfunction_call : public ASTNode
{
public:
    ASTfunction_call (OSLCompilerImpl *comp, ustring name, ASTNode *args);
    const char *nodetypename () const { return "function_call"; }
    const char *childname (size_t i) const;
    const char *opname () const;
    TypeSpec typecheck (TypeSpec expected);
    Symbol *codegen (Symbol *dest = NULL);

    FunctionSymbol *func () const { return (FunctionSymbol *)m_sym; }
    ref args () const { return child (0); }
    bool is_user_function () const {
        return func()->node() != NULL;
    }
    ASTfunction_declaration * user_function () const {
        return (ASTfunction_declaration *) func()->node();
    }

private:
    /// Typecheck all polymorphic versions, return UNKNOWN if no match was
    /// found, or a real type if there was a match.  Also, upon matching,
    /// re-jigger m_sym to point to the specific polymorphic match.
    TypeSpec typecheck_all_poly (TypeSpec expected, bool coerce);

    ustring m_name;
    Symbol *m_sym;
    FunctionSymbol *m_poly;
};



class ASTliteral : public ASTNode
{
public:
    ASTliteral (OSLCompilerImpl *comp, int i)
        : ASTNode (literal_node, comp), m_i(i)
    { m_typespec = TypeDesc::TypeInt; }

    ASTliteral (OSLCompilerImpl *comp, float f)
        : ASTNode (literal_node, comp), m_f(f)
    { m_typespec = TypeDesc::TypeFloat; }

    ASTliteral (OSLCompilerImpl *comp, ustring s)
        : ASTNode (literal_node, comp), m_s(s)
    { m_typespec = TypeDesc::TypeString; }

    const char *nodetypename () const { return "literal"; }
    const char *childname (size_t i) const;
    void print (std::ostream &out, int indentlevel) const;
    TypeSpec typecheck (TypeSpec expected) { return m_typespec; }
    Symbol *codegen (Symbol *dest = NULL);

    const char *strval () const { return m_s.c_str(); }
    int intval () const { return m_i; }
    float floatval () const { return m_typespec.is_int() ? (float)m_i : m_f; }

    void negate () { m_i = -m_i;  m_f = -m_f; }

private:
    ustring m_s;
    int m_i;
    float m_f;
};



}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif


#endif /* OSL_AST_H */
