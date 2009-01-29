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


/// Base node for an abstract syntax tree for the OSL parser.
///
class ASTNode : public RefCnt {
public:
    typedef intrusive_ptr<ASTNode> ref;  ///< Ref-counted pointer to an ASTNode

    /// List of all the types of AST nodes.
    ///
    enum NodeType {
        UnknownNode, ShaderDeclarationNode
    };

    ASTNode (NodeType nodetype, OSLCompilerImpl *compiler) 
        : m_nodetype(nodetype), m_compiler(compiler),
          m_sourcefile(compiler->filename()),
          m_sourceline(compiler->lineno())
    {
    }

    virtual ~ASTNode () { }

    virtual void print (int indent = 0) const = 0;

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
        : ASTNode (ShaderDeclarationNode, comp),
          m_shadertype(stype), m_shadername(name),
          m_formals(form), m_statements(stmts), m_metadata(meta)
    { }
    virtual void print (int indentlevel=0) const {
        indent (indentlevel);
        std::cout << "Shader declaration:\n";
        indent (indentlevel);
        std::cout << "  Type: " << m_shadertype << "\n";
        indent (indentlevel);
        std::cout << "  Name: " << m_shadername << "\n";
        indent (indentlevel);
        std::cout << "  Formals:\n";
        printlist (m_formals, indentlevel+1);
        indent (indentlevel);
        std::cout << "  Statements:\n";
        printlist (m_statements, indentlevel+1);
    }
private:
    int m_shadertype;
    ustring m_shadername;
    ref m_formals;
    ref m_statements;
    ref m_metadata;
};


class ASTsubclass : public ASTNode
{
public:
    ASTsubclass (OSLCompilerImpl *comp) : ASTNode (UnknownNode, comp) { }
    virtual ~ASTsubclass () { }
    virtual void print (int indent = 0) const { }
private:
};



}; // namespace pvt
}; // namespace OSL


#endif /* AST_H */
