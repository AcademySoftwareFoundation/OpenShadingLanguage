/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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

#include <boost/foreach.hpp>

#include "OpenImageIO/dassert.h"
#include "OpenImageIO/strutil.h"

#include "oslcomp_pvt.h"
#include "ast.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

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
    int i = 0;
    for (ASTNode::ref in = init();  in;  in = in->next(), ++i) {
        if (! m_sym->typespec().is_array() && i > 0)
            error ("Can't assign array initializers to non-array %s %s",
                   m_typespec.string().c_str(), m_name.c_str());
        if (! assignable(m_sym->typespec().elementtype(), in->typespec()))
            error ("can't assign '%s' to %s %s (%s)",
                   in->typespec().string().c_str(),
                   m_sym->typespec().string().c_str(), m_name.c_str(),
                   m_sym->typespec().elementtype().string().c_str());
    }
    return m_typespec;
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
            error ("can't use [][] on a %s", t.string().c_str());
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
    typecheck_list (cond ());
    oslcompiler->push_nesting (false);
    typecheck_list (truestmt ());
    typecheck_list (falsestmt ());
    oslcompiler->pop_nesting (false);

    TypeSpec c = cond()->typespec();
    if (c.is_closure())
        error ("Cannot use a closure as an 'if' condition");
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
ASTassign_expression::typecheck (TypeSpec expected)
{
    TypeSpec vt = var()->typecheck ();
    TypeSpec et = expr()->typecheck (vt);

    if (! var()->is_lvalue()) {
        error ("Can't assign via %s to something that isn't an lvalue", opname());
        return TypeSpec();
    }

    ASSERT (m_op == Assign);  // all else handled by binary_op

    // We don't currently support assignment of whole arrays
    if (vt.is_array() || et.is_array()) {
        error ("Can't assign entire arrays");
        return TypeSpec();
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
ASTreturn_statement::typecheck (TypeSpec expected)
{
    FunctionSymbol *myfunc = oslcompiler->current_function ();
    if (myfunc) {
        // If it's a user function (as opposed to a main shader body)...
        if (expr()) {
            // If we are returning a value, it must be assignable to the
            // kind of type the function actually returns.  This check
            // will also catch returning a value from a void function.
            TypeSpec et = expr()->typespec ();
            if (! assignable (myfunc->typespec(), et)) {
                error ("Cannot return a '%s' from '%s %s()'",
                       et.string().c_str(), myfunc->typespec().string().c_str(),
                       myfunc->name().c_str());
            }
        } else {
            // If we are not returning a value, it must be a void function.
            if (! myfunc->typespec().is_void ())
                error ("You must return a '%s' from function '%s'",
                       myfunc->typespec().string().c_str(),
                       myfunc->name().c_str());
        }
        // If the function has other statements AFTER 'return', or if
        // the return statement is in a conditional, we'll need to
        // handle it specially when generating code.
        myfunc->complex_return (this->nextptr() != NULL ||
                                myfunc->nesting_level() > 0);
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

    // Special for closures -- just a few cases to worry about
    if (l.is_color_closure() || r.is_color_closure()) {
        if (m_op == Add) {
            if (l.is_color_closure() && r.is_color_closure())
                return m_typespec = l;
        }
        if (m_op == Mul) {
            if (l.is_color_closure() && (r.is_color() || r.is_int_or_float()))
                return m_typespec = l;
            if (r.is_color_closure() && (l.is_color() || l.is_int_or_float()))
                return m_typespec = r;
        }
        // If we got this far, it's an op that's not allowed
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
    typecheck_children (m_typespec);
    TypeSpec t = expr()->typespec();
    if (! assignable (m_typespec, t) &&
        ! (m_typespec.is_int() && t.is_float()) && // (int)float is ok
        ! (m_typespec.is_triple() && t.is_triple()))
        error ("Cannot cast '%s' to '%s'", t.string().c_str(),
               m_typespec.string().c_str());
    return m_typespec;
}



TypeSpec
ASTtype_constructor::typecheck (TypeSpec expected)
{
    // FIXME - closures
    typecheck_children ();

    // Hijack the usual function arg-checking routines.
    // So we have a set of valid patterns for each type constructor:
    static const char *float_patterns[] = { "ff", NULL };
    static const char *triple_patterns[] = { "cf", "cfff", "csfff",
                                             "cc", "cp", "cv", "cn", NULL };
    static const char *matrix_patterns[] = { "mf", "msf",
                                             "mffffffffffffffff",
                                             "msffffffffffffffff", NULL };
    // Select the pattern for the type of constructor we are...
    const char **patterns = NULL;
    if (typespec().is_float())
        patterns = float_patterns;
    else if (typespec().is_triple())
        patterns = triple_patterns;
    else if (typespec().is_matrix())
        patterns = matrix_patterns;
    if (! patterns) {
        error ("Cannot construct type '%s'", typespec().string().c_str());
        return m_typespec;
    }

    // Try to get a match, first without type coercion of the arguments,
    // then with coercion.
    for (int co = 0;  co < 2;  ++co) {
        bool coerce = co;
        for (const char **pat = patterns;  *pat;  ++pat) {
            const char *code = *pat;
            if (check_arglist (typespec().string().c_str(), args(), code + 1, coerce))
                return m_typespec;
        }
    }

    // If we made it this far, no match could be found.
    std::string err = Strutil::format ("Cannot construct %s (", 
                                       typespec().string().c_str());
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
ASTNode::check_arglist (const char *funcname, ASTNode::ref arg,
                        const char *formals, bool coerce)
{
    for ( ;  arg;  arg = arg->next()) {
        if (! *formals)   // More formal args, but no more actual args
            return false;
        if (*formals == '*')  // Will match anything left
            return true;
        if (*formals == '.') {  // Special case for token/value pairs
            // FIXME -- require that the tokens be string literals
            if (arg->typespec().is_string() && arg->next() != NULL) {
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
            continue;  // match anything
        }

        TypeSpec argtype = arg->typespec();
        int advance;
        TypeSpec formaltype = m_compiler->type_from_code (formals, &advance);
        formals += advance;
        // std::cerr << "\targ is " << argtype.string() 
        //           << ", formal is " << formaltype.string() << "\n";
        if (argtype == formaltype)
            continue;   // ok, move on to next arg
        if (coerce && assignable (formaltype, argtype))
            continue;

        // anything that gets this far we don't consider a match
        return false;
    }
    if (*formals && *formals != '*' && *formals != '.') 
        return false;  // Non-*, non-... formals expected, no more actuals

    return true;  // Is this safe?
}



TypeSpec
ASTfunction_call::typecheck_all_poly (TypeSpec expected, bool coerce)
{
    for (FunctionSymbol *poly = func();  poly;  poly = poly->nextpoly()) {
        const char *code = poly->argcodes().c_str();
        int advance;
        TypeSpec returntype = m_compiler->type_from_code (code, &advance);
        code += advance;
        if (check_arglist (m_name.c_str(), args(), code, coerce)) {
            // Return types also must match if not coercible
            if (coerce || expected == TypeSpec() || expected == returntype)
                return returntype;
        }
    }
    return TypeSpec();
}



TypeSpec
ASTfunction_call::typecheck (TypeSpec expected)
{
    typecheck_children ();

    // Look for an exact match, including expected return type
    m_typespec = typecheck_all_poly (expected, false);
    if (m_typespec != TypeSpec())
        return m_typespec;

    // Now look for an exact match on args, but any assignable return type
    if (expected != TypeSpec()) {
        m_typespec = typecheck_all_poly (TypeSpec(), false);
        if (m_typespec != TypeSpec())
            return m_typespec;
    }

    // Now look for a coercible match of args, exact march on return type
    m_typespec = typecheck_all_poly (expected, true);
    if (m_typespec != TypeSpec())
        return m_typespec;

    // All that failed, try for a coercible match on everything
    if (expected != TypeSpec()) {
        m_typespec = typecheck_all_poly (TypeSpec(), true);
        if (m_typespec != TypeSpec())
            return m_typespec;
    }

    // Couldn't find any way to match any polymorphic version of the
    // function that we know about.  OK, at least try for helpful error
    // message.
    std::string choices ("");
    for (FunctionSymbol *poly = func();  poly;  poly = poly->nextpoly()) {
        const char *code = poly->argcodes().c_str();
        int advance;
        TypeSpec returntype = m_compiler->type_from_code (code, &advance);
        code += advance;
        if (choices.length())
            choices += "\n";
        choices += Strutil::format ("\t%s %s (%s)",
                              returntype.string().c_str(), m_name.c_str(),
                              m_compiler->typelist_from_code(code).c_str());
    }

    std::string actualargs;
    for (ASTNode::ref arg = args();  arg;  arg = arg->next()) {
        if (actualargs.length())
            actualargs += ", ";
        actualargs += arg->typespec().string();
    }

    error ("No matching function call to '%s (%s)'\n    Candidates are:\n%s", 
           m_name.c_str(), actualargs.c_str(), choices.c_str());
    return TypeSpec();
}



// FIXME -- should the type constructors be here?
// FIXME -- spline, inversespline -- hard case!
// FIXME -- regex, substr
// FIXME -- light and shadow

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

#define ANY_ONE_FLOAT_BASED "ff", "cc", "pp", "vv", "nn"
#define NOISE_ARGS "ff", "fff", "fp", "fpf", \
                   "cf", "cff", "cp", "cpf", \
                   "vf", "vff", "vp", "vpf"
#define PNOISE_ARGS "fff", "fffff", "fpp", "fpfpf", \
                    "cff", "cffff", "cpp", "cpfpf", \
                    "vff", "vffff", "vpp", "vpfpf"
#define DERIV_ARGS "ff", "vp", "vv", "vn", "cc"

static const char * builtin_func_args [] = {

    "aastep", "fff", "ffff", "fffff", "fffs", "ffffs", "fffffs", NULL,
    "abs", ANY_ONE_FLOAT_BASED,  "ii", NULL,  // alias for fabs()
    "acos", ANY_ONE_FLOAT_BASED, NULL,
    "area", "fp", NULL,
    "arraylength", "i?[]", NULL,
    "asin", ANY_ONE_FLOAT_BASED, NULL,
    "atan", ANY_ONE_FLOAT_BASED, NULL,
    "atan2", "fff", "ccc", "ppp", "vvv", "nnn", NULL,
    "bump", "xf", "xsf", "xv", NULL,
    "calculatenormal", "vp", NULL,
    "ceil", ANY_ONE_FLOAT_BASED, NULL,
    "cellnoise", NOISE_ARGS, NULL,
    "clamp", "ffff", "cccc", "pppp", "vvvv", "nnnn", NULL,
    "concat", "ss.", NULL,   // FIXME -- further checking
    "cos", ANY_ONE_FLOAT_BASED, NULL,
    "cosh", ANY_ONE_FLOAT_BASED, NULL,
    "cross", "vvv", NULL,
    "Dx", DERIV_ARGS, NULL,
    "Dy", DERIV_ARGS, NULL,
    "degrees", ANY_ONE_FLOAT_BASED, NULL,
    "deltau", DERIV_ARGS, NULL,
    "deltav", DERIV_ARGS, NULL,
    "determinant", "fm", NULL,
    "displace", "xf", "xsf", "xv", NULL,
    "distance", "fpp", "fppp", NULL,
    "dot", "fvv", NULL,
    "erf", "ff", NULL,
    "erfc", "ff", NULL,
    "error", "xs*", NULL,   // FIXME -- further checking
    "exit", "x", NULL,
    "exp", ANY_ONE_FLOAT_BASED, NULL,
    "exp2", ANY_ONE_FLOAT_BASED, NULL,
    "expm1", ANY_ONE_FLOAT_BASED, NULL,
    "fabs", ANY_ONE_FLOAT_BASED, "ii",  NULL,
    "faceforward", "vvvv", "vvv", NULL,
    "filterwidth", DERIV_ARGS, NULL,
    "floor", ANY_ONE_FLOAT_BASED, NULL,
    "fmod", "fff", "ccf", "ccc", "ppf", "ppp", "vvf", "vvv", NULL, // alias for mod()
    "format", "ss*", NULL,
    "fprintf", "xs*", NULL,   // FIXME -- further checking
    "fresnel", "xvvff", "xvvfffvv", NULL,
    "getattribute", "is?", NULL,  // FIXME -- further checking?
    "getmessage", "iss?", NULL,  // FIXME -- further checking?
    "gettextureinfo", "iss?", NULL,  // FIXME -- further checking?
    "inversesqrt", ANY_ONE_FLOAT_BASED, NULL,
    "isfinite", "if", NULL,
    "isindirectray", "i", NULL,
    "isinf", "if", NULL,
    "isnan", "if", NULL,
    "isshadowray", "i", NULL,
    "hash", NOISE_ARGS, NULL,
    "hypot", "fff", "ffff", NULL,
    "length", "fv", NULL,
    "log", ANY_ONE_FLOAT_BASED, "fff", "ccf", "ppf", "vvf", "nnf", NULL,
    "log2", ANY_ONE_FLOAT_BASED, NULL,
    "log10", ANY_ONE_FLOAT_BASED, NULL,
    "logb", ANY_ONE_FLOAT_BASED, NULL,
    "luminance", "fc", NULL,
    "max", "fff", "ccc", "ppp", "vvv", "nnn", NULL,
    "min", "fff", "ccc", "ppp", "vvv", "nnn", NULL,
    "mix", "ffff", "cccc", "pppp", "vvvv", "nnnn", 
                   "cccf", "pppf", "vvvf", "nnnf", NULL,
    "mod", "iii", "fff", "ccf", "ccc", "ppf", "ppp", "vvf", "vvv", NULL,  
    "noise", NOISE_ARGS, NULL,
    "normalize", "vv", "nn", NULL,
    "pnoise", PNOISE_ARGS, NULL,
    "pow", "fff", "ccf", "ppf", "vvf", "nnf", NULL,
    "printf", "xs*", NULL,   // FIXME -- further checking
    "psnoise", PNOISE_ARGS, NULL,
    "radians", ANY_ONE_FLOAT_BASED, NULL,
    "random", "f", "c", "p", "v", "n", NULL,
    "raylevel", "i", NULL,
    "reflect", "vvv", NULL,
    "refract", "vvvf", NULL,
    "rotate", "ppfpp", NULL,
    "round", ANY_ONE_FLOAT_BASED, NULL,
    "setmessage", "vs?", NULL,
    "sign", ANY_ONE_FLOAT_BASED, NULL,
    "sin", ANY_ONE_FLOAT_BASED, NULL,
    "sinh", ANY_ONE_FLOAT_BASED, NULL,
    "smoothstep", "ffff", NULL,
    "snoise", NOISE_ARGS, NULL,
    "sqrt", ANY_ONE_FLOAT_BASED, NULL,
    "step", "fff", NULL,
    "tan", ANY_ONE_FLOAT_BASED, NULL,
    "tanh", ANY_ONE_FLOAT_BASED, NULL,
    "texture", "fsff.", "fsffffff.","csff.", "csffffff.", 
               "vsff.", "vsffffff.", NULL,
    "transform", "psp", "vsv", "nsn", "pssp", "vssv", "nssn",
                 "pmp", "vmv", "nmn", NULL,
    "transformc", "csc", "cssc", NULL,
    "transformu", "fsf", "fssf", NULL,
    "transpose", "mm", NULL,
    "trunc", ANY_ONE_FLOAT_BASED, NULL,

    "ambient", "C", "Cn", NULL,
    "cooktorrance", "Cf", NULL,
    "diffuse", "Cn", NULL,
    "phong", "Cnf", NULL,
    "microfacet_beckmann", "Cnff", NULL,
    "microfacet_ggx", "Cnff", NULL,
    "ward", "Cnvff", NULL,
    "transparent", "C", NULL,
    "emission", "C", NULL,
    "orennayar", "Cnf", NULL,
    "reflection", "C", "Cf.", "Cs.", "Csf.", "Cv.", "Cvf.", "Csv.", "Csvf.", NULL,
    "refraction", "Cf", "Cff", "Csf", "Csff", 
                  "Cvf", "Cvff", "Csvf", "Csvff", NULL,
    "specular", "Cnf", NULL,
    "subsurface", "C.", NULL,
    "translucence", "C", NULL,
    NULL
#undef ANY_ONE_FLOAT_BASED
#undef NOISE_ARGS
#undef PNOISE_ARGS
#undef DERIV_ARGS
};


void
OSLCompilerImpl::initialize_builtin_funcs ()
{
    for (int i = 0;  builtin_func_args[i];  ++i) {
        ustring funcname (builtin_func_args[i++]);
        // Count the number of polymorphic versions
        int npoly = 0;
        for (npoly = 0;  builtin_func_args[i+npoly];  ++npoly) ;
        // Now add them in reverse order, so the order in the table is
        // the priority order for approximate matches.
        for (int j = npoly-1;  j >= 0;  --j) {
            ustring poly (builtin_func_args[i+j]);
            Symbol *last = symtab().clash (funcname);
            ASSERT (last == NULL || last->symtype() == SymTypeFunction);
            TypeSpec rettype = type_from_code (poly.c_str());
            FunctionSymbol *f = new FunctionSymbol (funcname, rettype);
            f->nextpoly ((FunctionSymbol *)last);
            f->argcodes (poly);
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
    case 'C' : // color closure
        t = TypeSpec (TypeDesc::TypeColor, true);
        break;
    case '?' : break; // anything will match, so keep 'UNKNOWN'
    case '*' : break; // anything will match, so keep 'UNKNOWN'
    default:
        std::cerr << "Don't know how to decode type code '" 
                  << code << "' " << (int)code[0] << "\n";
        ASSERT (0);   // FIXME
        if (advance)
            *advance = 1;
        return TypeSpec();
    }
    ++i;

    if (code[i] == '[') {
        ++i;
        t.make_array (-1);   // signal arrayness, unknown length
        if (isdigit(code[i])) {
            t.make_array (atoi (code+i));
            while (isdigit(code[i]))
                ++i;
            if (code[i] == ']')
                ++i;
        }
    }

    // FIXME -- closures, structs

    if (advance)
        *advance = i;
    return t;
}



std::string
OSLCompilerImpl::typelist_from_code (const char *code)
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
            ret += "any";
        } else {            
            TypeSpec t = type_from_code (code, &advance);
            ret += t.string();
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
OSLCompilerImpl::code_from_type (TypeSpec type)
{
    std::string out;
    TypeDesc elem = type.elementtype().simpletype();
    if (type.is_structure()) {
        ASSERT (0 && "code_from_type doesn't handle struct yet");
    } else if (type.is_closure()) {
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
        int len = type.arraylength ();
        if (len > 0)
            out += Strutil::format ("[%d]", len);
        else
            out += "[]";
    }

    return out;
}



}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
