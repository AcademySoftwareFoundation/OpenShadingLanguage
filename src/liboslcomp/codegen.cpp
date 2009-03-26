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
#include <iostream>

#include <boost/foreach.hpp>

#include "OpenImageIO/dassert.h"
#include "OpenImageIO/strutil.h"

#include "oslcomp_pvt.h"
#include "symtab.h"
#include "ast.h"


namespace OSL {
namespace pvt {   // OSL::pvt



void
OSLCompilerImpl::emitcode (const char *opname, size_t nargs, Symbol **args,
                           ASTNode *node)
{
//    std::cout << "\temit " << opname;
    m_ircode.push_back (IROpcode (ustring (opname), node));
    for (size_t i = 0;  i < nargs;  ++i) {
        if (args[i])
            m_ircode.back().add_arg (args[i]);
//        std::cout << " " << (args[i] ? args[i]->name() : ustring("<null>"));
    }
//    std::cout << "\n";
}



Symbol *
OSLCompilerImpl::make_temporary (const TypeSpec &type)
{
    ustring name = ustring::format ("$tmp%d", ++m_next_temp);
    Symbol *s = new Symbol (name, type, Symbol::SymTypeTemp);
    symtab().insert (s);
    return s;
}



void
ASTNode::emitcode (const char *opname, Symbol *arg0, 
                   Symbol *arg1, Symbol *arg2)
{
    Symbol *args[3] = { arg0, arg1, arg2 };
    size_t nargs = (arg0 != NULL) + (arg1 != NULL) + (arg2 != NULL);
    m_compiler->emitcode (opname, nargs, args, this);
}



void
ASTNode::emitcode (const char *opname, size_t nargs, Symbol **args)
{
    m_compiler->emitcode (opname, nargs, args, this);
}



Symbol *
ASTNode::codegen (Symbol *dest)
{
    codegen_children ();
    // FIXME -- nobody should ever call this
    std::cout << "codegen " << nodetypename() << " : " 
              << (opname() ? opname() : "") << "\n";
    return NULL;
}



void
ASTNode::codegen_children ()
{
    BOOST_FOREACH (ref &c, m_children) {
        codegen_list (c);
    }
}



void
ASTNode::codegen_list (ref node)
{
    while (node) {
        node->codegen ();
        node = node->next ();
    }
}



Symbol *
ASTshader_declaration::codegen (Symbol *dest)
{
    // FIXME -- formals

    codegen_list (statements());
    return NULL;
}



Symbol *
ASTassign_expression::codegen (Symbol *dest)
{
    dest = var()->codegen();
    if (m_op == Assign) {
        Symbol *operand = expr()->codegen (dest);
        // FIXME -- what about coerced types, do we need a temp and copy here?
        if (operand != dest)
            emitcode ("assign", dest, operand);
    } else {
        Symbol *operand = expr()->codegen ();
        // FIXME -- what about coerced types, do we need a temp and copy here?
        emitcode (opword(), dest, dest, operand);
    }

    return dest;
}



Symbol *
ASTvariable_declaration::codegen (Symbol *)
{
    if (init()) {
        Symbol *dest = init()->codegen (m_sym);
        if (dest != m_sym)
            emitcode ("assign", m_sym, dest);
    }        
    return m_sym;
}



Symbol *
ASTvariable_ref::codegen (Symbol *)
{
    return m_sym;
}



Symbol *
ASTbinary_expression::codegen (Symbol *dest)
{
    Symbol *lsym = left()->codegen ();
    Symbol *rsym = right()->codegen ();
    if (dest == NULL || ! equivalent (dest->typespec(), typespec()))
        dest = m_compiler->make_temporary (typespec());

    // FIXME -- what about coerced types, do we need a temp and copy here?

    emitcode (opword(), dest, lsym, rsym);
    return dest;
}



Symbol *
ASTfunction_call::codegen (Symbol *dest)
{
    // FIXME -- this is very wrong, just a placeholder

    if (dest == NULL || ! equivalent (dest->typespec(), typespec()))
        dest = m_compiler->make_temporary (typespec());
    std::vector<Symbol *> argdest;
    argdest.push_back (dest);
    for (ref a = args();  a;  a = a->next()) {
        argdest.push_back (a->codegen());
    }
    emitcode (m_name.c_str(), argdest.size(), &argdest[0]);
    return dest;
}



}; // namespace pvt
}; // namespace OSL
