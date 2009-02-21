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



void
ASTNode::typecheck_children (TypeSpec expected)
{
    BOOST_FOREACH (ref &c, m_children) {
        typecheck_list (c, expected);
    }
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
ASTpreincdec::typecheck (TypeSpec expected)
{
    typecheck_children ();
    m_is_lvalue = var()->is_lvalue();
    m_typespec = var()->typespec();
    return m_typespec;
}


 
TypeSpec
ASTpostincdec::typecheck (TypeSpec expected)
{
    typecheck_children ();
    if (! var()->is_lvalue())
        error ("%s can only be applied to an lvalue", nodetypename());
    m_is_lvalue = false;
    m_typespec = var()->typespec();
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
ASTconditional_statement::typecheck (TypeSpec expected)
{
    typecheck_children ();
    TypeSpec c = cond()->typespec();
    if (c.is_closure())
        error ("Cannot use a closure as an 'if' condition");
    if (c.is_structure())
        error ("Cannot use a struct as an 'if' condition");
    if (c.is_array())
        error ("Cannot use an array as an 'if' condition");
    return m_typespec = TypeDesc (TypeDesc::VOID);
}



TypeSpec
ASTloop_statement::typecheck (TypeSpec expected)
{
    typecheck_children ();
    TypeSpec c = cond()->typespec();
    if (c.is_closure())
        error ("Cannot use a closure as an '%s' condition", opname());
    if (c.is_structure())
        error ("Cannot use a struct as an '%s' condition", opname());
    if (c.is_array())
        error ("Cannot use an array as an '%s' condition", opname());
    return m_typespec = TypeDesc (TypeDesc::VOID);
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
        if (! vt.is_int() || ! et.is_int()) {
            error ("Operator %s can only be used on int, not %s %s %s",
                   opname(), vt.string().c_str(), opname(), et.string().c_str());
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



TypeSpec
ASTunary_expression::typecheck (TypeSpec expected)
{
    // FIXME - closures
    typecheck_children (expected);
    TypeSpec t = expr()->typespec();
    if (t.is_structure()) {
        error ("Can't do '%s' to a %s.", opname(), t.string().c_str());
        return TypeSpec ();
    }
    switch (m_op) {
    case Sub :
    case Add :
        if (t.is_string()) {
            error ("Can't do '%s' to a %s.", opname(), t.string().c_str());
            return TypeSpec ();
        }
        m_typespec = t;
        break;
    case LogicalNot :
        m_typespec = TypeDesc::TypeInt;  // ! is always an int
        break;
    case BitwiseNot :
        if (! t.is_int()) {
            error ("Operator '~' can only be done to an int");
            return TypeSpec ();
        }
        m_typespec = t;
        break;
    default:
        error ("unknown unary operator");
    }
    return m_typespec;
}



/// Given two types (which are already compatible for numeric ops),
/// return which one has "more precision".  Let's say the op is '+'.  So
/// hp(int,float) == float, hp(vector,float) == vector, etc.
inline TypeDesc
higherprecision (const TypeDesc &a, const TypeDesc &b)
{
    // Aggregate always beats non-aggregate
    if (a.aggregate > b.aggregate)
        return a;
    else if (b.aggregate > a.aggregate)
        return b;
    // Float beats int
    if (b.basetype == TypeDesc::FLOAT)
        return b;
    else return a;
}



TypeSpec
ASTbinary_expression::typecheck (TypeSpec expected)
{
    // FIXME - closures
    typecheck_children (expected);
    TypeSpec l = left()->typespec();
    TypeSpec r = right()->typespec();

    // No binary ops work on structs or arrays
    if (l.is_structure() || r.is_structure() ||
        l.is_array() || r.is_array()) {
        error ("Not allowed: '%s %s %s'",
               l.string().c_str(), opname(), r.string().c_str());
        return TypeSpec ();
    }

    switch (m_op) {
    case Sub :
    case Add :
    case Mul :
    case Div :
        // Add/Sub/Mul/Div work for any equivalent types, and
        // combination of int/float and other numeric types, but do not
        // work with strings.  Add/Sub don't work with matrices, but
        // Mul/Div do.
        // FIXME -- currently, equivalent types combine to make the
        // left type.  But maybe we should be more careful, for example
        // point-point -> vector, etc.
        if (l.is_string() || r.is_string())
            break;   // Dispense with strings trivially
        if ((m_op == Sub || m_op == Add) && (l.is_matrix() || r.is_matrix()))
            break;   // Matrices don't combine for + and -
        if (equivalent (l, r) ||
                (l.is_numeric() && r.is_int_or_float()) ||
                (l.is_int_or_float() && r.is_numeric()))
            return m_typespec = higherprecision (l.simpletype(), r.simpletype());
        break;

    case Mod :
        // Mod only works with ints, and return ints.
        if (l.is_int() && r.is_int())
            return m_typespec = TypeDesc::TypeInt;
        break;

    case Equal :
    case NotEqual :
        // Any equivalent types can be compared with == and !=, also a 
        // float or int can be compared to any other numeric type.
        // Result is always an int.
        if (equivalent (l, r) || 
              (l.is_numeric() && r.is_int_or_float()) ||
              (l.is_int_or_float() && r.is_numeric()))
            return m_typespec = TypeDesc::TypeInt;
        break;

    case Greater :
    case Less :
    case GreaterEqual :
    case LessEqual :
        // G/L comparisons only work with floats or ints, and always
        // return int.
        if (l.is_int_or_float() && r.is_int_or_float())
            return m_typespec = TypeDesc::TypeInt;
        break;

    case BitwiseAnd :
    case BitwiseOr :
    case BitwiseXor :
    case ShiftLeft :
    case ShiftRight :
        // Bitwise ops only work with ints, and return ints.
        if (l.is_int() && r.is_int())
            return m_typespec = TypeDesc::TypeInt;
        break;

    case LogicalAnd :
    case LogicalOr :
        // Logical ops work on any simple type (since they test for
        // nonzeroness), but always return int.
        m_typespec = TypeDesc::TypeInt;
        break;

    default:
        error ("unknown binary operator");
    }

    // If we got this far, it's an op that's not allowed
    error ("Not allowed: '%s %s %s'",
           l.string().c_str(), opname(), r.string().c_str());
    return TypeSpec ();
}



TypeSpec
ASTternary_expression::typecheck (TypeSpec expected)
{
    // FIXME - closures
    TypeSpec c = typecheck_list (cond(), TypeDesc::TypeInt);
    TypeSpec t = typecheck_list (trueexpr(), expected);
    TypeSpec f = typecheck_list (falseexpr(), expected);

    if (c.is_closure())
        error ("Cannot use a closure as a condition");
    if (c.is_structure())
        error ("Cannot use a struct as a condition");
    if (c.is_array())
        error ("Cannot use a struct as a condition");

    // No arrays
    if (t.is_array() || t.is_array()) {
        error ("Not allowed: '%s ? %s : %s'",
               c.string().c_str(), t.string().c_str(), f.string().c_str());
        return TypeSpec ();
    }

    // The true and false clauses need to be equivalent types, or one
    // needs to be assignable to the other (so one can be upcast).
    if (assignable (t, f) || assignable (f, t))
        m_typespec = higherprecision (t.simpletype(), f.simpletype());
    else
        error ("Not allowed: '%s ? %s : %s'",
               c.string().c_str(), t.string().c_str(), f.string().c_str());

    return m_typespec;
}



TypeSpec
ASTtypecast_expression::typecheck (TypeSpec expected)
{
    // FIXME - closures
    typecheck_children (expected);
    TypeSpec t = expr()->typespec();
    if (! assignable (m_typespec, t))
        error ("Cannot cast '%s' to '%s'", t.string().c_str(),
               m_typespec.string().c_str());
    return m_typespec;
}


}; // namespace pvt
}; // namespace OSL
