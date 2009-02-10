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

#include <vector>
#include <string>
#include <fstream>
#include <cstdio>
#include <streambuf>

#include "oslcomp_pvt.h"
#include "ast.h"


namespace OSL {
namespace pvt {   // OSL::pvt


ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler) 
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(0)
{
}



ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op,
                  ASTNode *a)
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(op)
{
    addchild (a);
}



ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op)
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(op)
{
}



ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op,
                  ASTNode *a, ASTNode *b)
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(op)
{
    addchild (a);
    addchild (b);
}



ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op,
                  ASTNode *a, ASTNode *b, ASTNode *c)
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(op)
{
    addchild (a);
    addchild (b);
    addchild (c);
}



ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op,
                  ASTNode *a, ASTNode *b, ASTNode *c, ASTNode *d)
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(op)
{
    addchild (a);
    addchild (b);
    addchild (c);
    addchild (d);
}



void
ASTNode::print (int indentlevel) const 
{
    indent (indentlevel);
    std::cout << nodetypename() << " : " << (opname() ? opname() : "") << "\n";
    printchildren (indentlevel);
}



void
ASTNode::printchildren (int indentlevel) const 
{
    for (size_t i = 0;  i < m_children.size();  ++i) {
        if (! child(i))
            continue;
        indent (indentlevel);
        if (childname(i))
            std::cout << "  " << childname(i) << " :\n";
        else
            std::cout << "  child " << i << " :\n";
        printlist (child(i), indentlevel+1);
    }
}



const char *
ASTshader_declaration::childname (size_t i) const
{
    static const char *name[] = { "metadata", "formals", "statements" };
    return name[i];
}



void
ASTshader_declaration::print (int indentlevel) const
{
    indent (indentlevel);
    std::cout << nodetypename() << " \"" << m_shadername << "\"\n";
    printchildren (indentlevel);
}



ASTvariable_declaration::ASTvariable_declaration (OSLCompilerImpl *comp,
                                                  const TypeSpec &type,
                                                  ustring name, ASTNode *init,
                                                  bool isparam)
    : ASTNode (variable_declaration_node, comp, 0, init),
      m_name(name), m_sym(NULL), 
      m_isparam(isparam), m_isoutput(false), m_ismetadata(false)
{
    Symbol *f = comp->symtab().find (name);
    if (f && f->scope() == comp->symtab().scopeid()) {
        comp->error (sourcefile(), sourceline(), 
                     "\"%s\" already declared in this scope",
                     name.c_str());
        // FIXME -- print the file and line of the other definition
    }
    m_sym = new Symbol (name, type);
    oslcompiler->symtab().insert (m_sym);
}



const char *
ASTvariable_declaration::nodetypename () const
{
    return m_isparam ? "parameter" : "variable_declaration";
}



const char *
ASTvariable_declaration::childname (size_t i) const
{
    static const char *name[] = { "initializer" };
    return name[i];
}



void
ASTvariable_declaration::print (int indentlevel) const
{
    indent (indentlevel);
    std::cout << nodetypename() << " " 
              << m_sym->type().type().c_str() << " " 
              << m_name << "\n";
    printchildren (indentlevel);
}



ASTvariable_ref::ASTvariable_ref (OSLCompilerImpl *comp, ustring name,
                                  ASTNode *array_index, ASTNode *comp1_index,
                                  ASTNode *comp2_index)
    : ASTNode (variable_ref_node, comp, 0, array_index, comp1_index, comp2_index),
      m_name(name), m_sym(NULL), m_preop(0), m_postop(0)
{
    m_sym = comp->symtab().find (name);
    if (! m_sym) {
        comp->error (sourcefile(), sourceline(), 
                     "\"%s\" not found", name.c_str());
        // FIXME -- would be fun to troll through the symtab and try to
        // find the things that almost matched and offer suggestions.
    }
}



const char *
ASTvariable_ref::childname (size_t i) const
{
    static const char *name[] = { "arrayindex", "componentindex1", "componentindex2" };
    return name[i];
}



void
ASTvariable_ref::print (int indentlevel) const
{
    indent (indentlevel);
    std::cout << nodetypename() << " " 
              << m_sym->type().type().c_str() << " " 
              << m_name;
    if (m_preop == 1)
        std::cout << " PRE-INCREMENT";
    else if (m_preop == -1)
        std::cout << " PRE-DECREMENT";
    if (m_postop == 1)
        std::cout << " POST-INCREMENT";
    else if (m_postop == -1)
        std::cout << " POST-DECREMENT";
    std::cout << "\n";
    printchildren (indentlevel);
}



const char *
ASTconditional_statement::childname (size_t i) const
{
    static const char *name[] = { "condition",
                                  "truestatement", "falsestatement" };
    return name[i];
}



const char *
ASTloop_statement::childname (size_t i) const
{
    static const char *name[] = { "initializer", "condition",
                                  "iteration", "bodystatement" };
    return name[i];
}



const char *
ASTloop_statement::opname () const
{
    switch (m_op) {
    case LoopWhile : return "while";
    case LoopDo    : return "do";
    case LoopFor   : return "for";
    default: ASSERT(0);
    }
}



const char *
ASTloopmod_statement::childname (size_t i) const
{
    return NULL;  // no children
}



const char *
ASTloopmod_statement::opname () const
{
    switch (m_op) {
    case LoopModBreak    : return "break";
    case LoopModContinue : return "continue";
    default: ASSERT(0);
    }
}



const char *
ASTreturn_statement::childname (size_t i) const
{
    return "expression";  // only child
}



const char *
ASTassign_expression::childname (size_t i) const
{
    static const char *name[] = { "variable", "expression" };
    return name[i];
}



const char *
ASTassign_expression::opname () const
{
    switch (m_op) {
    case Assign           : return "=";
    case MulAssign        : return "*=";
    case DivAssign        : return "/=";
    case AddAssign        : return "+=";
    case SubAssign        : return "-=";
    case BitwiseAndAssign : return "&=";
    case BitwiseOrAssign  : return "|=";
    case BitwiseXorAssign : return "^=";
    case ShiftLeftAssign  : return "<<=";
    case ShiftRightAssign : return ">>=";
    default: ASSERT(0);
    }
}



const char *
ASTunary_expression::childname (size_t i) const
{
    static const char *name[] = { "expression" };
    return name[i];
}



const char *
ASTunary_expression::opname () const
{
    switch (m_op) {
    case Pos        : return "+";
    case Neg        : return "-";
    case LogicalNot : return "!";
    case BitwiseNot : return "~";
    default: ASSERT(0);
    }
}



const char *
ASTbinary_expression::childname (size_t i) const
{
    static const char *name[] = { "left", "right" };
    return name[i];
}



const char *
ASTbinary_expression::opname () const
{
    switch (m_op) {
    case Mul          : return "*";
    case Div          : return "/";
    case Add          : return "+";
    case Sub          : return "-";
    case Mod          : return "%";
    case Equal        : return "==";
    case NotEqual     : return "!=";
    case Greater      : return ">";
    case GreaterEqual : return ">=";
    case Less         : return "<";
    case LessEqual    : return "<=";
    case BitwiseAnd   : return "&";
    case BitwiseOr    : return "|";
    case BitwiseXor   : return "^";
    case LogicalAnd   : return "&&";
    case LogicalOr    : return "||";
    case ShiftLeft    : return "<<";
    case ShiftRight   : return ">>";
    default: ASSERT(0);
    }
}



const char *
ASTternary_expression::childname (size_t i) const
{
    static const char *name[] = { "condition",
                                  "trueexpression", "falseexpression" };
    return name[i];
}



const char *
ASTtypecast_expression::childname (size_t i) const
{
    static const char *name[] = { "expr" };
    return name[i];
}



ASTfunction_call::ASTfunction_call (OSLCompilerImpl *comp, ustring name,
                                    ASTNode *args)
    : ASTNode (function_call_node, comp, 0, args)
{
    m_sym = comp->symtab().find (name);
    if (! m_sym) {
        comp->error (sourcefile(), sourceline(), 
                     "function \"%s\" not found", name.c_str());
        // FIXME -- would be fun to troll through the symtab and try to
        // find the things that almost matched and offer suggestions.
    }
}



const char *
ASTfunction_call::childname (size_t i) const
{
    static const char *name[] = { "parameters" };
    return name[i];
}



const char *
ASTliteral::childname (size_t i) const
{
    return NULL;
}



void
ASTliteral::print (int indentlevel) const
{
    indent (indentlevel);
    std::cout << nodetypename() << " " 
              << m_typespec.type().c_str() << " ";
    if (m_typespec.type() == TypeDesc::TypeInt)
        std::cout << m_i;
    else if (m_typespec.type() == TypeDesc::TypeFloat)
        std::cout << m_f;
    if (m_typespec.type() == TypeDesc::TypeString)
        std::cout << "\"" << m_s << "\"";
    std::cout << "\n";
}


}; // namespace pvt
}; // namespace OSL
