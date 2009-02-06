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
      m_sourceline(compiler->lineno())
{
}



void
ASTshader_declaration::print (int indentlevel) const
{
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



void
ASTconditional_statement::print (int indentlevel) const 
{
    indent (indentlevel);
    std::cout << "Conditional\n";
    indent (indentlevel);
    std::cout << "  Condition:\n";
    printlist (m_cond, indentlevel+1);
    indent (indentlevel);
    std::cout << "  True statements:\n";
    printlist (m_truestmt, indentlevel+1);
    if (m_falsestmt) {
        indent (indentlevel);
        std::cout << "  False statements:\n";
        printlist (m_falsestmt, indentlevel+1);
    }
}



void
ASTloop_statement::print (int indentlevel) const 
{
    indent (indentlevel);
    std::cout << "Loop: " << (looptype() == LoopWhile ? "while" 
                             : looptype() == LoopDo ? "do" : "for") << "\n";
    if (m_init) {
        indent (indentlevel);
        std::cout << "  Initialization:\n";
        printlist (m_init, indentlevel+1);
    }
    indent (indentlevel);
    std::cout << "  Condition:\n";
    printlist (m_cond, indentlevel+1);
    if (m_iter) {
        indent (indentlevel);
        std::cout << "  Iteration:\n";
        printlist (m_iter, indentlevel+1);
    }
    indent (indentlevel);
    std::cout << "  Statements:\n";
    printlist (m_stmt, indentlevel+1);
}



void
ASTassign_expression::print (int indentlevel) const 
{
    indent (indentlevel);
    std::cout << "Assignment: operator " << opsymbol() << "\n";
    indent (indentlevel);
    std::cout << "  Variable:\n";
    printlist (m_var, indentlevel+1);
    indent (indentlevel);
    std::cout << "  Expression:\n";
    printlist (m_expr, indentlevel+1);
}



const char *
ASTassign_expression::opsymbol () const
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



void
ASTbinary_expression::print (int indentlevel) const 
{
    indent (indentlevel);
    std::cout << "Binary expresion: operator " << opsymbol() << "\n";
    indent (indentlevel);
    std::cout << "  Left:\n";
    printlist (m_left, indentlevel+1);
    indent (indentlevel);
    std::cout << "  Right:\n";
    printlist (m_right, indentlevel+1);
}



const char *
ASTbinary_expression::opsymbol () const
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



}; // namespace pvt
}; // namespace OSL
