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

#include <boost/foreach.hpp>

#include "oslcomp_pvt.h"
#include "ast.h"


namespace OSL {
namespace pvt {   // OSL::pvt


TypeSpec
ASTNode::typecheck (TypeSpec expected)
{
    typecheck_children (expected);
    if (m_typespec == TypeSpec())
        m_typespec = expected;
    return m_typespec;
}



TypeSpec
ASTNode::typecheck_children (TypeSpec expected)
{
    bool first = true;
    TypeSpec firsttype;
    BOOST_FOREACH (ref &c, m_children) {
        TypeSpec t = typecheck_list (c, expected);
        if (first) {
            firsttype = t;
            first = false;
        }
    }
    return firsttype;
}



TypeSpec
ASTNode::typecheck_list (ref node, TypeSpec expected)
{
    TypeSpec t;
    while (node) {
        t = node->typecheck (expected);
        node = node->next ();
    }
    return t;
}



TypeSpec
ASTvariable_ref::typecheck (TypeSpec expected)
{
    m_is_lvalue = true;             // A var ref is an lvalue
    return m_typespec;
}


 
TypeSpec
ASTindex::typecheck (TypeSpec expected)
{
   typecheck_children ();
    const char *indextype = "";
    TypeSpec t = lvalue()->typespec();
    if (t.is_structure()) {
        error ("Cannot use [] indexing on a struct");
        return TypeSpec();
    }
    if (t.is_closure()) {
        error ("Cannot use [] indexing on a closure");
        return TypeSpec();
    }
    if (t.is_array()) {
        indextype = "array";
        m_typespec = t.elementtype();
        if (index2())
            error ("can't use [][] on a simple array");
    } else if (t.aggregate() == TypeDesc::VEC3) {
        indextype = "component";
        TypeDesc tnew = t.simpletype();
        tnew.aggregate = TypeDesc::SCALAR;
        m_typespec = tnew;
        if (index2())
            error ("can't use [][] on a %s", t.string().c_str());
    } else if (t.aggregate() == TypeDesc::MATRIX44) {
        indextype = "component";
        TypeDesc tnew = t.simpletype();
        tnew.aggregate = TypeDesc::SCALAR;
        m_typespec = tnew;
        if (! index2())
            error ("must use [][] on a matrix, not just []");
    } else {
        error ("can only use [] indexing for arrays or multi-component types");
        return TypeSpec();
    }

    // Make sure the indices (children 1+) are integers
    for (size_t c = 1;  c < nchildren();  ++c)
        if (! child(c)->typespec().is_int())
            error ("%s index must be an integer, not a %s", 
                   indextype, index()->typespec().string().c_str());

    // If the thing we're indexing is an lvalue, so is the indexed element
    m_is_lvalue = lvalue()->is_lvalue();

    return m_typespec;
}



TypeSpec
ASTstructselect::typecheck (TypeSpec expected)
{
    m_is_lvalue = lvalue()->is_lvalue();
    return ASTNode::typecheck (expected);
    // FIXME -- this is totally wrong
}



TypeSpec
ASTassign_expression::typecheck (TypeSpec expected)
{
    typecheck_children (expected);
    TypeSpec vt = var()->typespec();
    TypeSpec et = expr()->typespec();

    if (! var()->is_lvalue()) {
        error ("Can't assign via %s to something that isn't an lvalue", opname());
        return TypeSpec();
    }
    
    // We don't currently support assignment of whole arrays
    if (vt.is_array() || et.is_array()) {
        error ("Can't assign entire arrays");
        return TypeSpec();
    }

    // Bitwise and shift can only apply to int
    if (m_op == BitwiseAnd || m_op == BitwiseOr || m_op == BitwiseXor ||
        m_op == ShiftLeft || m_op == ShiftRight) {
        if (! vt.is_int()) {
            error ("Operator %s can only be used on int, not %s",
                   opname(), vt.string().c_str());
            return TypeSpec();
        }
    }

    // Expression must be of a type assignable to the lvalue
    if (! assignable (vt, et)) {
        error ("Cannot assign '%s' to '%s'",
               et.string().c_str(), vt.string().c_str());
        // FIXME - can we print the variable in question?
        return TypeSpec();
    }

    return m_typespec = vt;
}


}; // namespace pvt
}; // namespace OSL
