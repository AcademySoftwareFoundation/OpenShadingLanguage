/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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

#include <vector>
#include <string>

#include "oslcomp_pvt.h"

#include <OpenImageIO/dassert.h>
#include <OpenImageIO/strutil.h>
namespace Strutil = OIIO::Strutil;


OSL_NAMESPACE_ENTER

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
    for (auto&& c : m_children) {
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
ASTfunction_declaration::typecheck (TypeSpec expected)
{
    // Typecheck the args, remember to push/pop the function so that the
    // typechecking for 'return' will know which function it belongs to.
    oslcompiler->push_function (func ());
    typecheck_children (expected);
    oslcompiler->pop_function ();
    if (m_typespec == TypeSpec())
        m_typespec = expected;
    return m_typespec;
}



TypeSpec
ASTvariable_declaration::typecheck (TypeSpec expected)
{
    typecheck_children (m_typespec);

    if (! init())
        return m_typespec;

    if (m_typespec.is_structure() && ! m_initlist &&
        init()->typespec().structure() != m_typespec.structure()) {
        // Can't do:  struct foo = 1
        error ("Cannot initialize structure '%s' with a scalar value",
               name().c_str());
    }

    // If it's a compound initializer, look at the individual pieces
    ref init = this->init();
    if (init->nodetype() == compound_initializer_node) {
        ASSERT (! init->nextptr() &&
                "compound_initializer should be the only initializer");
        init = ((ASTcompound_initializer *)init.get())->initlist();
    }

    if (m_typespec.is_structure()) {
        // struct initialization handled separately
        return typecheck_struct_initializers (init, m_typespec, m_name.c_str());
    }

    typecheck_initlist (init, m_typespec, m_name.c_str());

    // Warning to catch confusing comma operator in variable initializers.
    // One place this comes up is when somebody forgets the proper syntax
    // for constructors, for example
    //     color x = (a, b, c);   // Sets x to (c,c,c)!
    // when they really meant
    //     color x = color(a, b, c);
    if (init->nodetype() == comma_operator_node && !typespec().is_closure() &&
        (typespec().is_triple() || typespec().is_matrix())) {
        warning ("Comma operator is very confusing here. "
                 "Did you mean to use a constructor: %s = %s(...)?",
                 m_name.c_str(), typespec().c_str());
    }

    return m_typespec;
}



void
ASTNode::typecheck_initlist (ref init, TypeSpec type, string_view name)
{
    // Loop over a list of initializers (it's just 1 if not an array)...
    for (int i = 0;  init;  init = init->next(), ++i) {
        // Check for too many initializers for an array
        if (type.is_array() && !type.is_unsized_array() && i >= type.arraylength()) {
            error ("Too many initializers for a '%s'", type_c_str(type));
            break;
        }
        // Special case: ok to assign a literal 0 to a closure to
        // initialize it.
        if ((type.is_closure() || type.is_closure_array()) &&
              ! init->typespec().is_closure() &&
              init->typespec().is_int_or_float() &&
              init->nodetype() == literal_node &&
            ((ASTliteral *)init.get())->floatval() == 0.0f) {
            continue;  // it's ok
        }
        if (! type.is_array() && i > 0)
            error ("Can't assign array initializers to non-array %s %s",
                   type_c_str(type), name);
        if (! assignable(type.elementtype(), init->typespec()))
            error ("Can't assign '%s' to %s %s", type_c_str(init->typespec()),
                   type_c_str(type), name);
    }
}



TypeSpec
ASTNode::typecheck_struct_initializers (ref init, TypeSpec type,
                                        string_view name)
{
    ASSERT (type.is_structure());

    if (! init->next() && init->typespec() == type) {
        // Special case: just one initializer, it's a whole struct of
        // the right type.
        return type;
    }

    // General case -- per-field initializers

    const StructSpec *structspec (type.structspec());
    int numfields = (int)structspec->numfields();
    for (int i = 0;  init;  init = init->next(), ++i) {
        if (i >= numfields) {
            error ("Too many initializers for '%s %s'",
                   type_c_str(type), name);
            break;
        }
        const StructSpec::FieldSpec &field (structspec->field(i));

        if (init->nodetype() == compound_initializer_node) {
            // Initializer is itself a compound, it ought to be initializing
            // a field that is an array.
            ASTcompound_initializer *cinit = (ASTcompound_initializer *)init.get();
            ustring fieldname = ustring::format ("%s.%s", name, field.name);
            if (field.type.is_array ()) {
                typecheck_initlist (cinit->initlist(), field.type, fieldname);
            } else if (field.type.is_structure()) {
                typecheck_struct_initializers (cinit->initlist(), field.type,
                                               fieldname);
            } else {
                error ("Can't use '{...}' for a struct field that is not an array");
            }
            continue;
        }

        // Ok to assign a literal 0 to a closure to initialize it.
        if (field.type.is_closure() && ! init->typespec().is_closure() &&
            (init->typespec().is_float() || init->typespec().is_int()) &&
            init->nodetype() == literal_node &&
            ((ASTliteral *)init.get())->floatval() == 0.0f) {
            continue;  // it's ok
        }

        // Normal initializer, normal field.
        if (! assignable(field.type, init->typespec()))
            error ("Can't assign '%s' to '%s %s.%s'",
                   type_c_str(init->typespec()),
                   type_c_str(field.type), name, field.name);
    }
    return type;
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
    if (index3()) {
        if (! t.is_array() && ! t.elementtype().is_matrix())
            error ("[][][] only valid for a matrix array");
        m_typespec = TypeDesc::FLOAT;
    } else if (t.is_array()) {
        indextype = "array";
        m_typespec = t.elementtype();
        if (index2()) {
            if (t.aggregate() == TypeDesc::SCALAR)
                error ("can't use [][] on a simple array");
            m_typespec = TypeDesc::FLOAT;
        }
    } else if (t.aggregate() == TypeDesc::VEC3) {
        indextype = "component";
        TypeDesc tnew = t.simpletype();
        tnew.aggregate = TypeDesc::SCALAR;
        tnew.vecsemantics = TypeDesc::NOXFORM;
        m_typespec = tnew;
        if (index2())
            error ("can't use [][] on a %s", type_c_str(t));
    } else if (t.aggregate() == TypeDesc::MATRIX44) {
        indextype = "component";
        TypeDesc tnew = t.simpletype();
        tnew.aggregate = TypeDesc::SCALAR;
        tnew.vecsemantics = TypeDesc::NOXFORM;
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
                   indextype, type_c_str(index()->typespec()));

    // If the thing we're indexing is an lvalue, so is the indexed element
    m_is_lvalue = lvalue()->is_lvalue();

    return m_typespec;
}



TypeSpec
ASTstructselect::typecheck (TypeSpec expected)
{
    // The ctr already figured out if this was a valid structure selection
    if (m_fieldid < 0 || m_fieldsym == NULL) {
        return TypeSpec();
    }

    typecheck_children ();
    StructSpec *structspec (TypeSpec::structspec (m_structid));
    m_typespec = structspec->field(m_fieldid).type;
    m_is_lvalue = lvalue()->is_lvalue();
    return m_typespec;
}



TypeSpec
ASTconditional_statement::typecheck (TypeSpec expected)
{
    typecheck_list (cond ());
    oslcompiler->push_nesting (false);
    typecheck_list (truestmt ());
    typecheck_list (falsestmt ());
    oslcompiler->pop_nesting (false);

    TypeSpec c = cond()->typespec();
    if (c.is_structure())
        error ("Cannot use a struct as an 'if' condition");
    if (c.is_array())
        error ("Cannot use an array as an 'if' condition");
    return m_typespec = TypeDesc (TypeDesc::NONE);
}



TypeSpec
ASTloop_statement::typecheck (TypeSpec expected)
{
    typecheck_list (init ());
    oslcompiler->push_nesting (true);
    typecheck_list (cond ());
    typecheck_list (iter ());
    typecheck_list (stmt ());
    oslcompiler->pop_nesting (true);

    TypeSpec c = cond()->typespec();
    if (c.is_closure())
        error ("Cannot use a closure as an '%s' condition", opname());
    if (c.is_structure())
        error ("Cannot use a struct as an '%s' condition", opname());
    if (c.is_array())
        error ("Cannot use an array as an '%s' condition", opname());
    return m_typespec = TypeDesc (TypeDesc::NONE);
}



TypeSpec
ASTloopmod_statement::typecheck (TypeSpec expected)
{
    if (oslcompiler->nesting_level(true/*loops*/) < 1)
        error ("Cannot '%s' here -- not inside a loop.", opname());
    return m_typespec = TypeDesc (TypeDesc::NONE);
}



TypeSpec
ASTassign_expression::typecheck (TypeSpec expected)
{
    TypeSpec vt = var()->typecheck ();
    TypeSpec et = expr()->typecheck (vt);

    if (! var()->is_lvalue()) {
        error ("Can't assign via %s to something that isn't an lvalue", opname());
        return TypeSpec();
    }

    ASSERT (m_op == Assign);  // all else handled by binary_op

    // Handle array case
    if (vt.is_array() || et.is_array()) {
        if (vt.is_array() && et.is_array() &&
            vt.arraylength() >= et.arraylength()) {
            if (vt.structure() && (vt.structure() == et.structure())) {
                return m_typespec = vt;
            }
            if (equivalent(vt.elementtype(), et.elementtype()) &&
                !vt.structure() && !et.structure()) {
                return m_typespec = vt;
            }
        }
        error ("Cannot assign '%s' to '%s'", type_c_str(et), type_c_str(vt));
        return TypeSpec();
    }

    // Special case: ok to assign a literal 0 to a closure to
    // initialize it.
    if (vt.is_closure() && ! et.is_closure() &&
        (et.is_float() || et.is_int()) &&
        expr()->nodetype() == literal_node &&
        ((ASTliteral *)&(*expr()))->floatval() == 0.0f) {
        return TypeSpec(); // it's ok
    }

    // If either argument is a structure, they better both be the same
    // exact kind of structure.
    if (vt.is_structure() || et.is_structure()) {
        int vts = vt.structure(), ets = et.structure();
        if (vts == ets)
            return m_typespec = vt;
        // Otherwise, a structure mismatch
        error ("Cannot assign '%s' to '%s'", type_c_str(et), type_c_str(vt));
        return TypeSpec();
    }

    // Expression must be of a type assignable to the lvalue
    if (! assignable (vt, et)) {
        error ("Cannot assign '%s' to '%s'", type_c_str(et), type_c_str(vt));
        // FIXME - can we print the variable in question?
        return TypeSpec();
    }

    // Warning to catch confusing comma operator in assignment.
    // One place this comes up is when somebody forgets the proper syntax
    // for constructors, for example
    //     color x = (a, b, c);   // Sets x to (c,c,c)!
    // when they really meant
    //     color x = color(a, b, c);
    if (expr()->nodetype() == comma_operator_node && !vt.is_closure() &&
        (vt.is_triple() || vt.is_matrix())) {
        warning ("Comma operator is very confusing here. "
                 "Did you mean to use a constructor: = %s(...)?",
                 vt.c_str());
    }

    return m_typespec = vt;
}



TypeSpec
ASTreturn_statement::typecheck (TypeSpec expected)
{
    FunctionSymbol *myfunc = oslcompiler->current_function ();
    if (myfunc) {
        // If it's a user function (as opposed to a main shader body)...
        if (expr()) {
            // If we are returning a value, it must be assignable to the
            // kind of type the function actually returns.  This check
            // will also catch returning a value from a void function.
            TypeSpec et = expr()->typecheck (myfunc->typespec());
            if (! assignable (myfunc->typespec(), et)) {
                error ("Cannot return a '%s' from '%s %s()'",
                       type_c_str(et), type_c_str(myfunc->typespec()),
                       myfunc->name().c_str());
            }
        } else {
            // If we are not returning a value, it must be a void function.
            if (! myfunc->typespec().is_void ())
                error ("You must return a '%s' from function '%s'",
                       type_c_str(myfunc->typespec()),
                       myfunc->name().c_str());
        }
        myfunc->encountered_return ();
    } else {
        // We're not part of any user function, so this 'return' must
        // be from the main shader body.  That's fine (it's equivalent
        // to calling exit()), but it can't return a value.
        if (expr())
            error ("Cannot return a value from a shader body");
    }
    return TypeSpec(); // TODO: what should be returned here?
}



TypeSpec
ASTunary_expression::typecheck (TypeSpec expected)
{
    typecheck_children (expected);
    TypeSpec t = expr()->typespec();

    if (m_function_overload) {
        // There was a function with the special name. See if the types
        // match.
        for (FunctionSymbol *f = m_function_overload; f; f = f->nextpoly()) {
            const char *code = f->argcodes().c_str();
            int advance;
            TypeSpec returntype = m_compiler->type_from_code (code, &advance);
            code += advance;
            if (code[0] && check_simple_arg (t, code, true) && !code[0]) {
                return m_typespec = returntype;
            }
        }
        // No match, so forget about the potentially overloaded function
        m_function_overload = nullptr;
    }

    if (t.is_structure() || t.is_array()) {
        error ("Can't do '%s' to a %s.", opname(), type_c_str(t));
        return TypeSpec ();
    }
    switch (m_op) {
    case Sub :
    case Add :
        if (! (t.is_closure() || t.is_numeric())) {
            error ("Can't do '%s' to a %s.", opname(), type_c_str(t));
            return TypeSpec ();
        }
        m_typespec = t;
        break;
    case Not :
        m_typespec = TypeDesc::TypeInt;  // ! is always an int
        break;
    case Compl :
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
    typecheck_children (expected);
    TypeSpec l = left()->typespec();
    TypeSpec r = right()->typespec();

    if (m_function_overload) {
        // There was a function with the special name. See if the types
        // match.
        for (FunctionSymbol *f = m_function_overload; f; f = f->nextpoly()) {
            const char *code = f->argcodes().c_str();
            int advance;
            TypeSpec returntype = m_compiler->type_from_code (code, &advance);
            code += advance;
            if (code[0] && check_simple_arg (l, code, true) &&
                code[0] && check_simple_arg (r, code, true) &&
                ! code[0]) {
                return m_typespec = returntype;
            }
        }
        // No match, so forget about the potentially overloaded function
        m_function_overload = nullptr;
    }

    // No binary ops work on structs or arrays
    if (l.is_structure() || r.is_structure() || l.is_array() || r.is_array()) {
        error ("Not allowed: '%s %s %s'",
               type_c_str(l), opname(), type_c_str(r));
        return TypeSpec ();
    }

    // Special for closures -- just a few cases to worry about
    if (l.is_color_closure() || r.is_color_closure()) {
        if (m_op == Add) {
            if (l.is_color_closure() && r.is_color_closure())
                return m_typespec = l;
        }
        if (m_op == Mul) {
            if (l.is_color_closure() != r.is_color_closure()) {
                if (l.is_color_closure() && (r.is_color() || r.is_int_or_float()))
                    return m_typespec = l;
                if (r.is_color_closure() && (l.is_color() || l.is_int_or_float())) {
                    // N.B. Reorder so that it's always r = closure * k,
                    // not r = k * closure.  See codegen for why this helps.
                    std::swap (m_children[0], m_children[1]);
                    return m_typespec = r;
                }
            }
        }
        if (m_op == And || m_op == Or) {
            // Logical ops work can work on closures (since they test
            // for nonemptiness, but always return int.
            return m_typespec = TypeDesc::TypeInt;
        }
        // If we got this far, it's an op that's not allowed
        error ("Not allowed: '%s %s %s'",
               type_c_str(l), opname(), type_c_str(r));
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
        if (l.is_string() || r.is_string())
            break;   // Dispense with strings trivially
        if ((m_op == Sub || m_op == Add) && (l.is_matrix() || r.is_matrix()))
            break;   // Matrices don't combine for + and -
        if (equivalent (l, r)) {
            // handle a few couple special cases
            if (m_op == Sub && l.is_point() && r.is_point())  // p-p == v
                return m_typespec = TypeDesc::TypeVector;
            if ((m_op == Add || m_op == Sub) &&
                (l.is_point() || r.is_point()))  // p +/- v, v +/- p == p
                return m_typespec = TypeDesc::TypePoint;
            // everything else: the first operand is also the return type
            return m_typespec = l;
        }
        if ((l.is_numeric() && r.is_int_or_float()) ||
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

    case BitAnd :
    case BitOr :
    case Xor :
    case ShiftLeft :
    case ShiftRight :
        // Bitwise ops only work with ints, and return ints.
        if (l.is_int() && r.is_int())
            return m_typespec = TypeDesc::TypeInt;
        break;

    case And :
    case Or :
        // Logical ops work on any simple type (since they test for
        // nonzeroness), but always return int.
        return m_typespec = TypeDesc::TypeInt;

    default:
        error ("unknown binary operator");
    }

    // If we got this far, it's an op that's not allowed
    error ("Not allowed: '%s %s %s'",
           type_c_str(l), opname(), type_c_str(r));
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
        error ("Cannot use an array as a condition");

    // No arrays
    if (t.is_array() || t.is_array()) {
        error ("Not allowed: '%s ? %s : %s'",
               type_c_str(c), type_c_str(t), type_c_str(f));
        return TypeSpec ();
    }

    // The true and false clauses need to be equivalent types, or one
    // needs to be assignable to the other (so one can be upcast).
    if (assignable (t, f) || assignable (f, t))
        m_typespec = higherprecision (t.simpletype(), f.simpletype());
    else
        error ("Not allowed: '%s ? %s : %s'",
               type_c_str(c), type_c_str(t), type_c_str(f));

    return m_typespec;
}



TypeSpec
ASTcomma_operator::typecheck (TypeSpec expected)
{
    return m_typespec = typecheck_list (expr(), expected);
    // N.B. typecheck_list already returns the type of the LAST node in
    // the list, just like the comma operator is supposed to do.
}



TypeSpec
ASTtypecast_expression::typecheck (TypeSpec expected)
{
    // FIXME - closures
    typecheck_children (m_typespec);
    TypeSpec t = expr()->typespec();
    if (! assignable (m_typespec, t) &&
        ! (m_typespec.is_int() && t.is_float()) && // (int)float is ok
        ! (m_typespec.is_triple() && t.is_triple()))
        error ("Cannot cast '%s' to '%s'", type_c_str(t),
               type_c_str(m_typespec));
    return m_typespec;
}



TypeSpec
ASTtype_constructor::typecheck (TypeSpec expected)
{
    // FIXME - closures
    typecheck_children ();

    // Hijack the usual function arg-checking routines.
    // So we have a set of valid patterns for each type constructor:
    static const char *float_patterns[] = { "ff", "fi", NULL };
    static const char *triple_patterns[] = { "cf", "cfff", "csfff",
                                             "cc", "cp", "cv", "cn", NULL };
    static const char *matrix_patterns[] = { "mf", "msf", "mss",
                                             "mffffffffffffffff",
                                             "msffffffffffffffff", "mm", NULL };
    static const char *int_patterns[] = { "if", "ii", NULL };
    // Select the pattern for the type of constructor we are...
    const char **patterns = NULL;
    if (typespec().is_float())
        patterns = float_patterns;
    else if (typespec().is_triple())
        patterns = triple_patterns;
    else if (typespec().is_matrix())
        patterns = matrix_patterns;
    else if (typespec().is_int())
        patterns = int_patterns;
    if (! patterns) {
        error ("Cannot construct type '%s'", type_c_str(typespec()));
        return m_typespec;
    }

    // Try to get a match, first without type coercion of the arguments,
    // then with coercion.
    for (int co = 0;  co < 2;  ++co) {
        bool coerce = co;
        for (const char **pat = patterns;  *pat;  ++pat) {
            const char *code = *pat;
            if (check_arglist (type_c_str(typespec()), args(), code + 1, coerce))
                return m_typespec;
        }
    }

    // If we made it this far, no match could be found.
    std::string err = Strutil::format ("Cannot construct %s (",
                                       type_c_str(typespec()));
    for (ref a = args();  a;  a = a->next()) {
        err += a->typespec().string();
        if (a->next())
            err += ", ";
    }
    err += ")";
    error ("%s", err.c_str());
    // FIXME -- it might be nice here to enumerate for the user all the
    // valid combinations.
    return m_typespec;
}



bool
ASTNode::check_simple_arg (const TypeSpec &argtype,
                           const char * &formals, bool coerce)
{
    int advance;
    TypeSpec formaltype = m_compiler->type_from_code (formals, &advance);
    formals += advance;
    // std::cerr << "\targ is " << argtype.string()
    //           << ", formal is " << formaltype.string() << "\n";
    if (argtype == formaltype)
        return true;   // ok, move on to next arg
    if (coerce && assignable (formaltype, argtype))
        return true;
    // Allow a fixed-length array match to a formal array with
    // unspecified length, if the element types are the same.
    if (formaltype.is_unsized_array() && argtype.is_sized_array() &&
          formaltype.elementtype() == argtype.elementtype())
        return true;

    // anything that gets this far we don't consider a match
    return false;
}



bool
ASTNode::check_arglist (const char *funcname, ASTNode::ref arg,
                        const char *formals, bool coerce)
{
    // std::cerr << "ca " << funcname << " formals='" << formals << "\n";
    for ( ;  arg;  arg = arg->next()) {
        if (! *formals)   // More formal args, but no more actual args
            return false;
        if (*formals == '*')  // Will match anything left
            return true;
        if (*formals == '.') {  // Special case for token/value pairs
            // FIXME -- require that the tokens be string literals
            if (arg->typespec().is_string() && arg->next()) {
                arg = arg->next();
                continue;
            }
            return false;
        }
        if (*formals == '?') {
            if (formals[1] == '[' && formals[2] == ']') {
                // Any array
                formals += 3;
                if (arg->typespec().is_array())
                    continue;  // match
                else return false;  // wanted an array, didn't get one
            }
            if (arg->typespec().is_array())
                return false;   // wanted any scalar, got an array
            formals += 1;
            continue;  // match anything
        }

        if (! check_simple_arg (arg->typespec(), formals, coerce))
            return false;
        // If check_simple_arg succeeded, it advanced formals, and we
        // repeat for the next argument.
    }
    if (*formals && *formals != '*' && *formals != '.')
        return false;  // Non-*, non-... formals expected, no more actuals

    return true;  // Is this safe?
}



void
ASTfunction_call::mark_optional_output (int firstopt, const char **tags)
{
   bool mark_all = *tags && **tags == '*';
   std::vector<ASTNode::ref> argvec;
   list_to_vec (args(), argvec);

   // Find the beginning of the optional arguments
   int nargs = (int) listlength(args());
   while (firstopt < nargs &&
          ! argvec[firstopt]->typespec().is_string())
       ++firstopt;

   // Loop through the optional args, look for any tag
   for (int a = firstopt;  a < (int)argvec.size()-1;  a += 2) {
       ASTNode *s = argvec[a].get();
       bool isoutput = false;
       // compare against output tags
       if (s->typespec().is_string()) {
           if (s->nodetype() == ASTNode::literal_node) {
               // If the token is a string literal, see if it's one of the
               // ones designated as an output slot.
               for (const char **tag = tags; *tag && !isoutput; ++tag)
                   isoutput = isoutput || mark_all || (! strcmp (((ASTliteral *)s)->strval(), *tag));
           } else {
               // If the token is not a literal, we don't know what it'll
               // be at runtime, so mark it conservatively as possible output.
               isoutput = true;
           }
       }
       if (isoutput) {
           // writes to the next arg!
           if (a+2 < 32)
               argwriteonly (a+2);   // mark writeable
           else {
               // We can only designate the first 32 args
               // writeable.  So swap it with earlier optional args.
               std::swap (argvec[firstopt],   argvec[a]);
               std::swap (argvec[firstopt+1], argvec[a+1]);
               argwriteonly (firstopt+1);
               firstopt += 2;  // advance in case another is needed
           }
       }
   }

   m_children[0] = vec_to_list (argvec);
}



bool
ASTfunction_call::typecheck_printf_args (const char *format, ASTNode *arg)
{
    int argnum = (m_name == "fprintf") ? 3 : 2;
    while (*format != '\0') {
        if (*format == '%') {
            if (format[1] == '%') {
                // '%%' is a literal '%'
                format += 2;  // skip both percentages
                continue;
            }
            const char *oldfmt = format;  // mark beginning of format
            while (*format &&
                   *format != 'c' && *format != 'd' && *format != 'e' &&
                   *format != 'f' && *format != 'g' && *format != 'i' &&
                   *format != 'm' && *format != 'n' && *format != 'o' &&
                   *format != 'p' && *format != 's' && *format != 'u' &&
                   *format != 'v' && *format != 'x' && *format != 'X')
                ++format;
            char formatchar = *format++;  // Also eat the format char
            if (! arg) {
                error ("%s has mismatched format string and arguments (not enough args)",
                       m_name.c_str());
                return false;
            }

            std::string ourformat (oldfmt, format);  // straddle the format
            // Doctor it to fix mismatches between format and data
            TypeDesc simpletype (arg->typespec().simpletype());
            if ((arg->typespec().is_closure_based() ||
                 simpletype.basetype == TypeDesc::STRING)
                && formatchar != 's') {
                error ("%s has mismatched format string and arguments (arg %d needs %%s)",
                       m_name.c_str());
                return false;
            }
            if (simpletype.basetype == TypeDesc::INT && formatchar != 'd' &&
                formatchar != 'i' && formatchar != 'o' &&
                formatchar != 'x' && formatchar != 'X') {
                error ("%s has mismatched format string and arguments (arg %d needs %%d, %%i, %%o, %%x, or %%X)",
                       m_name.c_str(), argnum);
                return false;
            }
            if (simpletype.basetype == TypeDesc::FLOAT && formatchar != 'f' &&
                formatchar != 'g' && formatchar != 'c' && formatchar != 'e' &&
                formatchar != 'm' && formatchar != 'n' && formatchar != 'p' &&
                formatchar != 'v') {
                error ("%s has mismatched format string and arguments (arg %d needs %%f, %%g, or %%e)",
                       m_name.c_str(), argnum);
                return false;
            }

            arg = arg->nextptr();
            ++argnum;
        } else {
            // Everything else -- just copy the character and advance
            ++format;
        }
    }
    if (arg) {
        error ("%s has mismatched format string and arguments (too many args)",
               m_name.c_str());
        return false;
    }
    return true;  // all ok
}



void
ASTfunction_call::typecheck_builtin_specialcase ()
{
    const char *tex_out_args[] = {"alpha", "errormessage", NULL};
    const char *pointcloud_out_args[] = {"*", NULL};

    if (m_name == "transform") {
        // Special case for transform: under the covers, it selects
        // vector or normal special versions depending on its use.
        if (typespec().simpletype() == TypeDesc::TypeVector)
            m_name = ustring ("transformv");
        else if (typespec().simpletype() == TypeDesc::TypeNormal)
            m_name = ustring ("transformn");
    }

    // Void functions DO read their first arg, DON'T write it
    if (typespec().is_void()) {
        argread (0, true);
        argwrite (0, false);
    }

    if (func()->readwrite_special_case()) {
        if (m_name == "sincos") {
            argwriteonly (1);
            argwriteonly (2);
        } else if (m_name == "getattribute" || m_name == "getmessage" ||
                   m_name == "gettextureinfo" || m_name == "getmatrix" ||
                   m_name == "dict_value") {
            // these all write to their last argument
            argwriteonly ((int)listlength(args()));
        } else if (m_name == "pointcloud_get") {
            argwriteonly (5);
        } else if (m_name == "pointcloud_search") {
            mark_optional_output(5, pointcloud_out_args);
        } else if (m_name == "split") {
            argwriteonly (2);
        } else if (func()->texture_args()) {
            mark_optional_output(2, tex_out_args);
        }
    }

    if (func()->printf_args()) {
        ASTNode *arg = args().get();  // first arg
        if (arg && m_name == "fprintf")
            arg = arg->nextptr();  // skip filename param for fprintf
        const char *format = NULL;
        if (arg && arg->nodetype() == ASTNode::literal_node &&
            arg->typespec().is_string() &&
            (format = ((ASTliteral *)arg)->strval())) {
            arg = arg->nextptr ();
            typecheck_printf_args (format, arg);
        } else {
            warning ("%s() uses a format string that is not a constant.",
                     m_name.c_str());
        }
    }

    if (func()->takes_derivs()) {
        // Special handling for the few functions that take derivatives
        // of their arguments.  Mark those with argtakesderivs.
        // N.B. This counts arguments in the same way that opcodes do --
        // assuming "arg 0" is the return value.
        size_t nargs = listlength(args());
        if (m_name == "area" || m_name == "filterwidth") {
            argtakesderivs (1, true);
#if 0
        } else if (m_name == "aastep") {
            // all but the 5-arg version take derivs of edge param
            argtakesderivs (1, nargs<5);
            // aastep(f,f) and aastep(f,f,str) take derivs of s param
            if (nargs == 2 || list_nth(args(),2)->typespec().is_string())
                argtakesderivs (2, true);
#endif
        } else if (m_name == "bump" || m_name == "displace") {
            // FIXME -- come back to this
        } else if (m_name == "calculatenormal") {
            argtakesderivs (1, true);
        } else if (m_name == "Dx" || m_name == "Dy" || m_name == "Dz") {
            argtakesderivs (1, true);
        } else if (m_name == "texture") {
            if (nargs == 3 || list_nth(args(),3)->typespec().is_string()) {
                argtakesderivs (2, true);
                argtakesderivs (3, true);
            }
        } else if (m_name == "texture3d") {
            if (nargs == 2 || list_nth(args(),2)->typespec().is_string()) {
                argtakesderivs (2, true);
            }
        } else if (m_name == "environment") {
            if (nargs == 2 || list_nth(args(),2)->typespec().is_string()) {
                argtakesderivs (2, true);
            }
        } else if (m_name == "trace") {
            argtakesderivs (1, true);
            argtakesderivs (2, true);
        } else if (m_name == "noise" || m_name == "pnoise") {
            ASTNode *arg = args().get();  // first argument
            if (arg->typespec().is_string()) {
                // The kind of noise that names the type of noise
                ASTliteral *lit = (arg->nodetype() == ASTNode::literal_node)
                                   ? (ASTliteral *)arg : NULL;
                if (!lit || (lit->ustrval() == "gabor")) {
                    // unspecified (not a string literal), or known to be
                    // gabor -- take derivs of positional arguments
                    arg = arg->nextptr();  // advance to position
                    for (int n = 2; arg && ! arg->typespec().is_string(); ++n) {
                        argtakesderivs (n, true);
                        arg = arg->nextptr();
                    }
                }
            }
        } else {
            ASSERT (0 && "Missed a takes_derivs case!");
        }
    }
}



TypeSpec
ASTfunction_call::typecheck_struct_constructor ()
{
    StructSpec *structspec = m_sym->typespec().structspec();
    ASSERT (structspec);
    m_typespec = m_sym->typespec();
    if (structspec->numfields() != (int)listlength(args())) {
        error ("Constructor for '%s' has the wrong number of arguments (expected %d, got %d)",
               structspec->name(), structspec->numfields(), listlength(args()));
    }
    return typecheck_struct_initializers (args(), m_typespec, "this");
}


/// <doc CandidateFunctions>
///
/// Score a set of polymorphic functions based on arguments & return type.
///
/// When choosing which varaint to use precedence is given to arguments and the
/// return type is used as a possible tie breaker in case of any ambiguity.
/// The highest score for an argument is given when it is an exact match.
///
class CandidateFunctions {
    enum {
        kExactMatch     = 100,
        kIntegralToFP   = 77,
        kArrayMatch     = 44,
        kCoercable      = 23,
        kMatchAnything  = 1,
        kNoMatch        = 0,

        // Additional named rules
        kFPToIntegral   = 60,          // = kIntegralToFP to match c++
    };
    struct Candidate {
        FunctionSymbol* sym;
        TypeSpec rtype;
        int ascore;
        int rscore;

        Candidate(FunctionSymbol *s, TypeSpec rt, int as, int rs) :
            sym(s), rtype(rt), ascore(as), rscore(rs) {}

        string_view name() const { return sym->name(); }
    };
    typedef std::vector<Candidate> Candidates;

    OSLCompilerImpl* m_compiler;
    Candidates m_candidates;
    std::set<ustring> m_scored;
    TypeSpec m_rval;
    ASTNode::ref m_args;
    size_t m_nargs;

    const char* scoreWildcard(int& argscore, size_t& fargs, const char* args) const {
        while (fargs < m_nargs) {
            argscore += kMatchAnything;
            ++fargs;
        }
        return args + 1;
    }

    static int scoreType(TypeSpec expected, TypeSpec actual) {
        if (expected == actual)
            return kExactMatch;

        if (!actual.is_closure() && actual.is_scalarnum() &&
            !expected.is_closure() && expected.is_scalarnum())
            return expected.is_int() ? kFPToIntegral : kIntegralToFP;

        if (expected.is_unsized_array() && actual.is_sized_array() &&
            expected.elementtype() == actual.elementtype()) {
            // Allow a fixed-length array match to a formal array with
            // unspecified length, if the element types are the same.
            return kArrayMatch;
        }

        if (assignable (expected, actual))
            return kCoercable;

        return kNoMatch;
    }

    int addCandidate(FunctionSymbol* func) {
        // Early out if this declaration has already been scored
        if (m_scored.count(func->argcodes()))
            return kNoMatch;
        m_scored.insert(func->argcodes());

        int advance;
        const char *formals = func->argcodes().c_str();
        TypeSpec rtype = m_compiler->type_from_code (formals, &advance);
        formals += advance;

        int argscore = 0;
        size_t fargs = 0;
        for (ASTNode::ref arg = m_args; *formals && arg; ++fargs, arg = arg->next()) {
            switch (*formals) {
                case '*':  // Will match anything left
                    formals = scoreWildcard(argscore, fargs, formals);
                    ASSERT (*formals == 0);
                    continue;

                case '.':  // Token/value pairs
                    if (arg->typespec().is_string() && arg->next()) {
                        formals = scoreWildcard(argscore, fargs, formals);
                        ASSERT (*formals == 0);
                        continue;
                    }
                    return kNoMatch;

                case '?':
                    if (formals[1] == '[' && formals[2] == ']') {
                        // Any array
                        formals += 3;
                        if (!arg->typespec().is_array())
                            return kNoMatch; // wanted an array, didn't get one
                        argscore += kMatchAnything;
                    } else if (!arg->typespec().is_array()) {
                        formals += 1;       // match anything
                        argscore += kMatchAnything;
                    } else
                        return kNoMatch;    // wanted any scalar, got an array
                    continue;

                default:
                    break;
            }
            // To many arguments for the function, done without a match.
            if (fargs >= m_nargs)
                return kNoMatch;

            TypeSpec formaltype = m_compiler->type_from_code (formals, &advance);
            formals += advance;

            int score = scoreType(formaltype, arg->typespec());
            if (score == kNoMatch)
                return kNoMatch;

            argscore += score;
        }

        // Check any remaining arguments
        switch (*formals) {
            case '*':
            case '.':
                // Skip over the unused optional args
                ++formals;
                ++fargs;
            case '\0':
                if (fargs < m_nargs)
                    return kNoMatch;
                break;

            default:
                // TODO: Scoring default function arguments would go here
                // Curently an unused formal argument, so no match at all.
                return kNoMatch;
        }
        ASSERT (*formals == 0);

        int highscore = m_candidates.empty() ? 0 : m_candidates.front().ascore;
        if (argscore < highscore)
            return kNoMatch;

        // clear any prior ambiguous matches
        if (argscore != highscore)
            m_candidates.clear();

        // append the latest high scoring function
        m_candidates.emplace_back(func, rtype, argscore,
                                  scoreType(m_rval, rtype));

        return argscore;
    }

public:
    CandidateFunctions(OSLCompilerImpl* compiler, TypeSpec rval, ASTNode::ref args, FunctionSymbol* func) :
        m_compiler(compiler), m_rval(rval), m_args(args), m_nargs(0) {

        //std::cerr << "Matching " << func->name() << " formals='" << (rval.simpletype().basetype != TypeDesc::UNKNOWN ?  compiler->code_from_type (rval) : " ");
        for (ASTNode::ref arg = m_args; arg; arg = arg->next()) {
            //std::cerr << compiler->code_from_type (arg->typespec());
            ++m_nargs;
        }
        //std::cerr << "'\n";

        while (func) {
            //int score =
            addCandidate(func);
            //std::cerr << '\t' << func->name() << " formals='" << func->argcodes().c_str() << "'  " << score << ", " << (score ? m_candidates.back().rscore : 0) << "\n";
            func = func->nextpoly();
        }
    }

    void reportError(ASTNode* caller,
                     std::string name = "", bool showArgs = true,
                     bool candidateMsg = true,
                     const char* msg = "No matching function call to") const {
        if (showArgs) {
            name += " (";
            const char *comma = "";
            for (ASTNode::ref arg = m_args; arg; arg = arg->next()) {
                name += comma;
                name += arg->typespec().string();
                comma = ", ";
            }
            name += ")";
        }

        caller->error ("%s '%s'", msg, name);
        if (candidateMsg)
            m_compiler->errhandler().message("  Candidates are:\n");
    }

    void reportAmbiguity(FunctionSymbol* sym) const {
        int advance;
        const char *formals = sym->argcodes().c_str();
        TypeSpec returntype = m_compiler->type_from_code (formals, &advance);
        formals += advance;

        auto& errh =  m_compiler->errhandler();
        errh.message("    ");

        if (ASTNode* decl = sym->node())
            errh.message("%s:%d\t", decl->sourcefile(), decl->sourceline());

        errh.message("%s %s (%s)\n", m_compiler->type_c_str(returntype),
                     sym->name(),
                     m_compiler->typelist_from_code(formals).c_str());
    }

    std::pair<FunctionSymbol*, TypeSpec> best(ASTNode* caller, bool strict = 0) {
        switch (m_candidates.size()) {
            case 0:  return { nullptr, TypeSpec() };
            case 1:  return { m_candidates[0].sym, m_candidates[0].rtype };
            default: break;
        }

        int ambiguity = 0;
        std::pair<const Candidate*, int> c = { nullptr, -1 };
        for (auto& candidate : m_candidates) {
            // re-score based on matching return value
            if (candidate.rscore > c.second)
                c = std::make_pair(&candidate, candidate.rscore);
            else if (candidate.rscore == c.second)
                ambiguity = candidate.rscore;
        }

        if (ambiguity || strict) {
            ASSERT (caller);
            reportError(caller, m_candidates[0].name(), false,
                        !m_candidates.empty(), "Ambiguous call to");
            for (auto& candidate : m_candidates) {
                if (candidate.rscore >= ambiguity)
                    reportAmbiguity(candidate.sym);
            }
        }

        ASSERT (c.first);
        return {c.first->sym, c.first->rtype};
    }

    bool empty() const { return m_candidates.empty(); }
};



TypeSpec
ASTfunction_call::typecheck (TypeSpec expected)
{
    typecheck_children ();

    if (is_struct_ctr()) {
        // Looks like function call, but is actually struct constructor
        return typecheck_struct_constructor ();
    }

    // Save the currently choosen symbol for error reporting later
    FunctionSymbol* poly = func();

    CandidateFunctions candidates(m_compiler, expected, args(), poly);
    std::tie(m_sym, m_typespec) = candidates.best(this);

    if (m_sym != nullptr) {
        if (is_user_function()) {
            if (func()->number_of_returns() == 0 &&
                ! func()->typespec().is_void()) {
                error ("non-void function \"%s\" had no 'return' statement.",
                       func()->name().c_str());
            }
        } else {
            // built-in
            typecheck_builtin_specialcase ();
        }
        return m_typespec;
    }

    // Ambiguity has already been reported.
    if (!candidates.empty())
        return TypeSpec();

    // Couldn't find any way to match any polymorphic version of the
    // function that we know about.  OK, at least try for helpful error
    // message.
    candidates.reportError(this, m_name.string(), true, poly != nullptr);
    while (poly) {
        candidates.reportAmbiguity(poly);
        poly = poly->nextpoly();
    }

    return TypeSpec();
}



// Key:
//    x - void (only used for first char to indicate void return type)
//    i - int
//    f - float
//    c - color
//    p - point
//    v - vector
//    n - normal
//    m - matrix
//    s - string
//    ? - one arg of any type
//    X[] - an array of X's of any size
//    X[int] - an array of X's of definite length
//    * - 0 or more args of any type
//    . - 2*n args of alternating string/value
//    C - color closure
//
// There are also entries that don't describe polymorphisms, but just mark
// the functions as having special properties:
//   "!rw"      nonstandard behavior about which args are read vs written.
//   "!printf"  has a printf-like argument list
//   "!tex"     has a texture()-like token/value pair optinal argument list
//   "!deriv"   takes derivs of its arguments

#define ANY_ONE_FLOAT_BASED "ff", "cc", "pp", "vv", "nn"
#define NOISE_ARGS "ff", "fff", "fp", "fpf", \
                   "cf", "cff", "cp", "cpf", \
                   "vf", "vff", "vp", "vpf"
#define PNOISE_ARGS "fff", "fffff", "fpp", "fpfpf", \
                    "cff", "cffff", "cpp", "cpfpf", \
                    "vff", "vffff", "vpp", "vpfpf"
#define GNOISE_ARGS "fsf.", "fsff.", "fsp.", "fspf.", \
                    "csf.", "csff.", "csp.", "cspf.", \
                    "vsf.", "vsff.", "vsp.", "vspf."
#define PGNOISE_ARGS "fsff.", "fsffff.", "fspp.", "fspfpf.", \
                     "csff.", "csffff.", "cspp.", "cspfpf.", \
                     "vsff.", "vsffff.", "vspp.", "vspfpf."

static const char * builtin_func_args [] = {

    "area", "fp", "!deriv", NULL,
    "arraylength", "i?[]", NULL,
    "bump", "xf", "xsf", "xv", "!deriv", NULL,
    "calculatenormal", "vp", "!deriv", NULL,
    "cellnoise", NOISE_ARGS, NULL,
    "concat", "sss", /*"ss.",*/ NULL,   // FIXME -- further checking
    "dict_find", "iss", "iis", NULL,
    "dict_next", "ii", NULL,
    "dict_value", "iis?", "!rw", NULL,
    "Dx", "ff", "vp", "vv", "vn", "cc", "!deriv", NULL,
    "Dy", "ff", "vp", "vv", "vn", "cc", "!deriv", NULL,
    "Dz", "ff", "vp", "vv", "vn", "cc", "!deriv", NULL,
    "displace", "xf", "xsf", "xv", "!deriv", NULL,
    "environment", "fsv.", "fsvvv.","csv.", "csvvv.",
               "vsv.", "vsvvv.", "!tex", "!rw", "!deriv", NULL,
    "error", "xs*", "!printf", NULL,
    "exit", "x", NULL,
    "filterwidth", "ff", "vp", "vv", "!deriv", NULL,
    "format", "ss*", "!printf", NULL,
    "fprintf", "xss*", "!printf", NULL,
    "getattribute", "is?", "is?[]", "iss?", "iss?[]",  "isi?", "isi?[]", "issi?", "issi?[]", "!rw", NULL,  // FIXME -- further checking?
    "getmessage", "is?", "is?[]", "iss?", "iss?[]", "!rw", NULL,
    "gettextureinfo", "iss?", "iss?[]", "!rw", NULL,  // FIXME -- further checking?
    "hashnoise", NOISE_ARGS, NULL,
    "isconnected", "i?", NULL,
    "isconstant", "i?", NULL,
    "noise", GNOISE_ARGS, NOISE_ARGS, "!deriv", NULL,
    "pnoise", PGNOISE_ARGS, PNOISE_ARGS, "!deriv", NULL,
    "pointcloud_search", "ispfi.", "ispfii.", "!rw", NULL,
    "pointcloud_get", "isi[]is?[]", "!rw", NULL,
    "pointcloud_write", "isp.", NULL,
    "printf", "xs*", "!printf", NULL,
    "psnoise", PNOISE_ARGS, NULL,
    "random", "f", "c", "p", "v", "n", NULL,
    "regex_match", "iss", "isi[]s", NULL,
    "regex_search", "iss", "isi[]s", NULL,
    "setmessage", "xs?", "xs?[]", NULL,
    "sincos", "xfff", "xccc", "xppp", "xvvv", "xnnn", "!rw", NULL,
    "snoise", NOISE_ARGS, NULL,
    "spline", "fsff[]", "csfc[]", "psfp[]", "vsfv[]", "nsfn[]", "fsfif[]", "csfic[]", "psfip[]", "vsfiv[]", "nsfin[]", NULL,
    "splineinverse", "fsff[]", "fsfif[]", NULL,
    "split", "iss[]si", "iss[]s", "iss[]", "!rw", NULL,
    "surfacearea", "f", NULL,
    "texture", "fsff.", "fsffffff.","csff.", "csffffff.",
               "vsff.", "vsffffff.", "!tex", "!rw", "!deriv", NULL,
    "texture3d", "fsp.", "fspvvv.","csp.", "cspvvv.",
               "vsp.", "vspvvv.", "!tex", "!rw", "!deriv", NULL,
    "trace", "ipv.", "!deriv", NULL,
    "warning", "xs*", "!printf", NULL,   // FIXME -- further checking
    NULL
#undef ANY_ONE_FLOAT_BASED
#undef NOISE_ARGS
#undef PNOISE_ARGS
};


void
OSLCompilerImpl::initialize_builtin_funcs ()
{
    for (int i = 0;  builtin_func_args[i];  ++i) {
        ustring funcname (builtin_func_args[i++]);
        // Count the number of polymorphic versions and look for any
        // special hint markers.
        int npoly = 0;
        bool readwrite_special_case = false;
        bool texture_args = false;
        bool printf_args = false;
        bool takes_derivs = false;
        for (npoly = 0;  builtin_func_args[i+npoly];  ++npoly) {
            if (! strcmp (builtin_func_args[i+npoly], "!rw"))
                readwrite_special_case = true;
            else if (! strcmp (builtin_func_args[i+npoly], "!tex"))
                texture_args = true;
            else if (! strcmp (builtin_func_args[i+npoly], "!printf"))
                printf_args = true;
            else if (! strcmp (builtin_func_args[i+npoly], "!deriv"))
                takes_derivs = true;
        }
        // Now add them in reverse order, so the order in the table is
        // the priority order for approximate matches.
        for (int j = npoly-1;  j >= 0;  --j) {
            if (builtin_func_args[i+j][0] == '!')  // Skip special hints
                continue;
            ustring poly (builtin_func_args[i+j]);
            Symbol *last = symtab().clash (funcname);
            ASSERT (last == NULL || last->symtype() == SymTypeFunction);
            TypeSpec rettype = type_from_code (poly.c_str());
            FunctionSymbol *f = new FunctionSymbol (funcname, rettype);
            f->nextpoly ((FunctionSymbol *)last);
            f->argcodes (poly);
            f->readwrite_special_case (readwrite_special_case);
            f->texture_args (texture_args);
            f->printf_args (printf_args);
            f->takes_derivs (takes_derivs);
            symtab().insert (f);
        }
        i += npoly;
    }
}



TypeSpec
OSLCompilerImpl::type_from_code (const char *code, int *advance)
{
    TypeSpec t;
    int i = 0;
    switch (code[i]) {
    case 'i' : t = TypeDesc::TypeInt;          break;
    case 'f' : t = TypeDesc::TypeFloat;        break;
    case 'c' : t = TypeDesc::TypeColor;        break;
    case 'p' : t = TypeDesc::TypePoint;        break;
    case 'v' : t = TypeDesc::TypeVector;       break;
    case 'n' : t = TypeDesc::TypeNormal;       break;
    case 'm' : t = TypeDesc::TypeMatrix;       break;
    case 's' : t = TypeDesc::TypeString;       break;
    case 'x' : t = TypeDesc (TypeDesc::NONE);  break;
    case 'X' : t = TypeDesc (TypeDesc::PTR);   break;
    case 'L' : t = TypeDesc (TypeDesc::LONGLONG); break;
    case 'C' : // color closure
        t = TypeSpec (TypeDesc::TypeColor, true);
        break;
    case 'S' : // structure
        // Following the 'S' is the numeric structure ID
        t = TypeSpec ("struct", atoi (code+i+1));
        // Skip to the last digit
        while (isdigit(code[i+1]))
            ++i;
        break;
    case '?' : break; // anything will match, so keep 'UNKNOWN'
    case '*' : break; // anything will match, so keep 'UNKNOWN'
    case '.' : break; // anything will match, so keep 'UNKNOWN'
    default:
        ASSERTMSG (0, "Don't know how to decode type code '%d'", (int)code[0]);
        if (advance)
            *advance = 1;
        return TypeSpec();
    }
    ++i;

    if (code[i] == '[') {
        ++i;
        t.make_array (-1);   // signal arrayness, unknown length
        if (isdigit(code[i]) || code[i] == ']') {
            if (isdigit(code[i]))
                t.make_array (atoi (code+i));
            while (isdigit(code[i]))
                ++i;
            if (code[i] == ']')
                ++i;
        }
    }

    if (advance)
        *advance = i;
    return t;
}



std::string
OSLCompilerImpl::typelist_from_code (const char *code) const
{
    std::string ret;
    while (*code) {
        // Handle some special cases
        int advance = 1;
        if (ret.length())
            ret += ", ";
        if (*code == '.') {
            ret += "...";
        } else if (*code == 'T') {
            ret += "...";
        } else if (*code == '?') {
            ret += "<any>";
        } else {
            TypeSpec t = type_from_code (code, &advance);
            ret += type_c_str(t);
        }
        code += advance;
        if (*code == '[') {
            ret += "[]";
            ++code;
            while (isdigit(*code))
                ++code;
            if (*code == ']')
                ++code;
        }
    }

    return ret;
}



std::string
OSLCompilerImpl::code_from_type (TypeSpec type) const
{
    std::string out;
    TypeDesc elem = type.elementtype().simpletype();
    if (type.is_structure() || type.is_structure_array()) {
        out = Strutil::format ("S%d", type.structure());
    } else if (type.is_closure() || type.is_closure_array()) {
        out = 'C';
    } else {
        if (elem == TypeDesc::TypeInt)
            out = 'i';
        else if (elem == TypeDesc::TypeFloat)
            out = 'f';
        else if (elem == TypeDesc::TypeColor)
            out = 'c';
        else if (elem == TypeDesc::TypePoint)
            out = 'p';
        else if (elem == TypeDesc::TypeVector)
            out = 'v';
        else if (elem == TypeDesc::TypeNormal)
            out = 'n';
        else if (elem == TypeDesc::TypeMatrix)
            out = 'm';
        else if (elem == TypeDesc::TypeString)
            out = 's';
        else if (elem == TypeDesc::NONE)
            out = 'x';
        else
            ASSERT (0);
    }

    if (type.is_array()) {
        if (type.is_unsized_array())
            out += "[]";
        else
            out += Strutil::format ("[%d]", type.arraylength());
    }

    return out;
}



void
OSLCompilerImpl::typespecs_from_codes (const char *code,
                                       std::vector<TypeSpec> &types) const
{
    types.clear ();
    while (code && *code) {
        int advance;
        types.push_back (type_from_code (code, &advance));
        code += advance;
    }
}



}; // namespace pvt

OSL_NAMESPACE_EXIT
