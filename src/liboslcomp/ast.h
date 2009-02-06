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

#ifndef AST_H
#define AST_H

#include "OpenImageIO/ustring.h"
#include "OpenImageIO/thread.h"
#include "OpenImageIO/refcnt.h"
#include "oslcomp.h"


class oslFlexLexer;
extern int oslparse ();


namespace OSL {
namespace pvt {


class OSLCompilerImpl;  // Forward decl



/// Base node for an abstract syntax tree for the OSL parser.
///
class ASTNode : public RefCnt {
public:
    typedef intrusive_ptr<ASTNode> ref;  ///< Ref-counted pointer to an ASTNode

    /// List of all the types of AST nodes.
    ///
    enum NodeType {
        unknown_node, shader_declaration_node,
        conditional_statement_node, loop_statement_node,
        binary_expression_node, unary_expression_node,
        assign_expression_node, ternary_expression_node, 
        typecast_expression_node,
        _last_node
    };

    ASTNode (NodeType nodetype, OSLCompilerImpl *compiler);

    virtual ~ASTNode () { }

    virtual void print (int indentlevel = 0) const = 0;

    /// What type of node is this?
    ///
    NodeType nodetype () const { return m_nodetype; }

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

protected:
    /// Return a reference-counted pointer to the next node in the sequence.
    ///
    ref next () const { return m_next; }

    /// Return the raw pointer to the next node in the sequence.  Use
    /// with caution!
    ASTNode *nextptr () const { return m_next.get(); }

    void indent (int indentlevel=0) const {
        while (indentlevel--)
            std::cout << "    ";
    }

    static void printlist (const ref &A, int indentlevel) {
        for (const ASTNode *n = A.get();  n;  n = n->nextptr())
            n->print (indentlevel);
    }


protected:
    NodeType m_nodetype;    ///< Type of node this is
    ref m_next;             ///< Next node in the list
    OSLCompilerImpl *m_compiler;  ///< Back-pointer to the compiler
    ustring m_sourcefile;   ///< Filename of source where the node came from
    int m_sourceline;       ///< Line number in source where the node came from

private:
};



class ASTshader_declaration : public ASTNode
{
public:
    ASTshader_declaration (OSLCompilerImpl *comp, int stype, ustring name,
                           ASTNode *form, ASTNode *stmts, ASTNode *meta) 
        : ASTNode (shader_declaration_node, comp),
          m_shadertype(stype), m_shadername(name),
          m_formals(form), m_statements(stmts), m_metadata(meta)
    { }
    void print (int indentlevel=0) const;

private:
    int m_shadertype;
    ustring m_shadername;
    ref m_formals;
    ref m_statements;
    ref m_metadata;
};



class ASTconditional_statement : public ASTNode
{
public:
    ASTconditional_statement (OSLCompilerImpl *comp, ASTNode *cond,
                              ASTNode *truestmt, ASTNode *falsestmt=NULL)
        : ASTNode (conditional_statement_node, comp), 
          m_cond(cond), m_truestmt(truestmt), m_falsestmt(falsestmt)
    { }

    void print (int indentlevel = 0) const;

private:
    ref m_cond, m_truestmt, m_falsestmt;
};



class ASTloop_statement : public ASTNode
{
public:
    enum LoopType {
        LoopWhile, LoopDo, LoopFor
    };

    ASTloop_statement (OSLCompilerImpl *comp, LoopType looptype, ASTNode *init,
                       ASTNode *cond, ASTNode *iter, ASTNode *stmt)
        : ASTNode (loop_statement_node, comp), m_looptype(looptype),
          m_init(init), m_cond(cond), m_iter(iter), m_stmt(stmt)
    { }

    void print (int indentlevel = 0) const;

    LoopType looptype () const { return m_looptype; }

private:
    LoopType m_looptype;
    ref m_init, m_cond, m_iter, m_stmt;
};



class ASTassign_expression : public ASTNode
{
public:
    enum Assignment { Assign, MulAssign, DivAssign, AddAssign, SubAssign,
                      BitwiseAndAssign, BitwiseOrAssign, BitwiseXorAssign,
                      ShiftLeftAssign, ShiftRightAssign };

    ASTassign_expression (OSLCompilerImpl *comp, ASTNode *var, Assignment op,
                          ASTNode *expr)
        : ASTNode (assign_expression_node, comp), 
              m_var(var), m_op(op), m_expr(expr)
    { }

    void print (int indentlevel = 0) const;

    const char *opsymbol () const;

private:
    Assignment m_op;
    ref m_var, m_expr;
};



class ASTbinary_expression : public ASTNode
{
public:
    enum Binop { Mul, Div, Add, Sub, Mod, 
                 Equal, NotEqual, Greater, Less, GreaterEqual, LessEqual, 
                 BitwiseAnd, BitwiseOr, BitwiseXor, LogicalAnd, LogicalOr,
                 ShiftLeft, ShiftRight };

    ASTbinary_expression (OSLCompilerImpl *comp, Binop op,
                          ASTNode *left, ASTNode *right)
        : ASTNode (binary_expression_node, comp), 
          m_left(left), m_right(right), m_op(op)
    { }

    void print (int indentlevel = 0) const;

    const char *opsymbol () const;

private:
    ref m_left, m_right;
    Binop m_op;
};



#if 0
class ASTsubclass : public ASTNode
{
public:
    ASTsubclass (OSLCompilerImpl *comp) : ASTNode (unknown_node, comp) { }
    ~ASTsubclass () { }
    void print (int indentlevel = 0) const { }
private:
};
#endif



}; // namespace pvt
}; // namespace OSL


#endif /* AST_H */
