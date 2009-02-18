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
ASTindex::typecheck (TypeSpec expected)
{
    typecheck_list (lvalue());
    typecheck_list (index(), TypeSpec(TypeDesc::INT));
    const char *indextype = "";
    if (lvalue()->typespec().is_array()) {
        indextype = "array ";
        if (! index()->typespec().is_int())
            error ("array index must be an integer");
        m_typespec = lvalue()->typespec();
        m_typespec.make_array (0);  // make it not be an array
    } else if (lvalue()->typespec().is_aggregate()) {
        indextype = "component ";
//        if (lvalue()->typespec().
    } else {
        error ("can only use [] indexing for arrays or multi-component types");
        return TypeSpec();
    }
    if (! index()->typespec().is_int())
        error ("%s index must be an integer, not a %s", 
               indextype, index()->typespec().string().c_str());
}


}; // namespace pvt
}; // namespace OSL
