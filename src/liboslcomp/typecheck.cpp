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


/// Base class default implementation of typecheck() -- just check each 
/// child node, and set this node's type to 
TypeSpec
ASTNode::typecheck (TypeSpec expected)
{
    typecheck_children (expected);
    return m_typespec = expected;
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
    typecheck_children (TypeSpec(TypeDesc::INT));  // All indices are integers
#if 0
    if (m_array_index) {
        if (! m_array_index->typespec().is_int())
            error ();
    }
#endif
}


}; // namespace pvt
}; // namespace OSL
