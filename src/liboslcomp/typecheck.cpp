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

#include "OpenImageIO/dassert.h"
#include "OpenImageIO/strutil.h"

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
ASTvariable_declaration::typecheck (TypeSpec expected)
{
    typecheck_children (m_typespec);
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



bool
ASTNode::check_arglist (ASTNode::ref arg, const char *formals, bool coerce)
{
    for ( ;  arg;  arg = arg->next()) {
        if (! *formals)   // More formal args, but no more actual args
            return false;
        if (*formals == '*')  // Will match anything left
            return true;
        if (*formals == '.') {  // Special case for token/value pairs
            return false;
            // FIXME!
        }
        if (*formals == '?') {
            return false;
            // FIXME!
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
    if (*formals && *formals != '*') // Non-* formals expected, no more actuals
        return false;

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
        if (check_arglist (args(), code, coerce)) {
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

#define ANY_ONE_FLOAT_BASED "ff", "cc", "pp", "vv", "nn"
#define NOISE_ARGS "ff", "fff", "fp", "fpf", "cf", "cff", "cp", "cpf", \
                   "vf", "vff", "vp", "vpf"
#define PNOISE_ARGS "fff", "fffff", "fpp", "fpfpf", \
                    "cff", "cffff", "cpp", "cpfpf", \
                    "vff", "vffff", "vpp", "vpfpf"
#define DERIV_ARGS "ff", "vp", "vv", "vn", "cc"

static const char * builtin_func_args [] = {

    "aastep", "fff", "ffff", "fffff", "fffs", "ffffs", "fffffs", NULL,
    "abs", ANY_ONE_FLOAT_BASED, NULL,
    "acos", ANY_ONE_FLOAT_BASED, NULL,
    "area", "fp", NULL,
    "arraylength", "i?[", NULL,
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
    "Du", DERIV_ARGS, NULL,
    "Dv", DERIV_ARGS, NULL,
    "degrees", "ff", NULL,
    "deltau", DERIV_ARGS, NULL,
    "deltav", DERIV_ARGS, NULL,
    "determinant", "fm", NULL,
    "displace", "xf", "xsf", "xv", NULL,
    "distance", "fpp", "fppp", NULL,
    "dot", "fvv", NULL,
    "erf", "ff", NULL,
    "erfc", "ff", NULL,
    "error", "xs.", NULL,   // FIXME -- further checking
    "exit", "x", NULL,
    "exp", ANY_ONE_FLOAT_BASED, NULL,
    "exp2", ANY_ONE_FLOAT_BASED, NULL,
    "expm1", ANY_ONE_FLOAT_BASED, NULL,
    "fabs", ANY_ONE_FLOAT_BASED, NULL,
    "faceforward", "vvvv", "vvv", NULL,
    "filterwidth", DERIV_ARGS, NULL,
    "floor", ANY_ONE_FLOAT_BASED, NULL,
    "fmod", ANY_ONE_FLOAT_BASED, NULL,
    "format", "ss.", NULL,
    "fprintf", "xs.", NULL,   // FIXME -- further checking
    "fresnel", "xvvf", "xvvffvv", NULL,
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
    "log", ANY_ONE_FLOAT_BASED, "ccf", "ppf", "vvf", "nnf", NULL,
    "log2", ANY_ONE_FLOAT_BASED, NULL,
    "log10", ANY_ONE_FLOAT_BASED, NULL,
    "luminance", "fc", NULL,
    "max", "fff", "ccc", "ppp", "vvv", "nnn", NULL,
    "min", "fff", "ccc", "ppp", "vvv", "nnn", NULL,
    "mix", "ffff", "cccc", "pppp", "vvvv", "nnnn", 
                   "cccf", "pppf", "vvvf", "nnnf", NULL,
    "mod", ANY_ONE_FLOAT_BASED, NULL,
    "noise", NOISE_ARGS, NULL,
    "normalize", "vv", NULL,
    "pnoise", NOISE_ARGS, NULL,
    "pow", ANY_ONE_FLOAT_BASED, "ccf", "ppf", "vvf", "nnf", NULL,
    "printf", "xs.", NULL,   // FIXME -- further checking
    "psnoise", NOISE_ARGS, NULL,
    "radians", "ff", NULL,
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
    "texture", "fsffT", "fsffffffT","csffT", "csffffffT", 
               "vsffT", "vsffffffT", NULL,
    "transform", "psp", "vsv", "nsn", "pssp", "vssv", "nssn",
                 "pmp", "vmv", "nmn", NULL,
    "transformc", "csc", "cssc", NULL,
    "transformu", "fsf", "fssf", NULL,
    "transpose", "mm", NULL,
    "trunc", ANY_ONE_FLOAT_BASED, NULL,
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
            ASSERT (last == NULL || last->symtype() == Symbol::SymTypeFunction);
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
    case 'x' : t = TypeDesc (TypeDesc::VOID);  break;
    case 'C' : // color closure
        t = TypeSpec (TypeDesc::TypeColor, true);
        break;
    case '?' : break; // anything will match, so keep 'UNKNOWN'
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
            t.make_array (atoi (code));
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


}; // namespace pvt
}; // namespace OSL
