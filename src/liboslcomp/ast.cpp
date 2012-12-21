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
#include <sstream>

#include <boost/filesystem.hpp>

#include "osl_pvt.h"
#include "oslcomp_pvt.h"

#include "OpenImageIO/dassert.h"
#include "OpenImageIO/strutil.h"
namespace Strutil = OIIO::Strutil;

OSL_NAMESPACE_ENTER

namespace pvt {   // OSL::pvt


ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler) 
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(0), m_is_lvalue(false)
{
}



ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op,
                  ASTNode *a)
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(op), m_is_lvalue(false)
{
    addchild (a);
}



ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op)
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(op), m_is_lvalue(false)
{
}



ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op,
                  ASTNode *a, ASTNode *b)
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(op), m_is_lvalue(false)
{
    addchild (a);
    addchild (b);
}



ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op,
                  ASTNode *a, ASTNode *b, ASTNode *c)
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(op), m_is_lvalue(false)
{
    addchild (a);
    addchild (b);
    addchild (c);
}



ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op,
                  ASTNode *a, ASTNode *b, ASTNode *c, ASTNode *d)
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(op), m_is_lvalue(false)
{
    addchild (a);
    addchild (b);
    addchild (c);
    addchild (d);
}



void
ASTNode::error (const char *format, ...)
{
    va_list ap;
    va_start (ap, format);
    std::string errmsg = format ? Strutil::vformat (format, ap) : "syntax error";
    va_end (ap);
    m_compiler->error (sourcefile(), sourceline(), "%s", errmsg.c_str());
}



void
ASTNode::warning (const char *format, ...)
{
    va_list ap;
    va_start (ap, format);
    std::string errmsg = format ? Strutil::vformat (format, ap) : "unknown warning";
    va_end (ap);
    m_compiler->warning (sourcefile(), sourceline(), "%s", errmsg.c_str());
}



void
ASTNode::print (std::ostream &out, int indentlevel) const 
{
    indent (out, indentlevel);
    out << "(" << nodetypename() << " : " 
        << "    (type: " << typespec().string() << ") "
        << (opname() ? opname() : "") << "\n";
    printchildren (out, indentlevel);
    indent (out, indentlevel);
    out << ")\n";
}



void
ASTNode::printchildren (std::ostream &out, int indentlevel) const 
{
    for (size_t i = 0;  i < m_children.size();  ++i) {
        if (! child(i))
            continue;
        indent (out, indentlevel);
        if (childname(i))
            out << "  " << childname(i);
        else
            out << "  child" << i;
        out << ": ";
        if (typespec() != TypeSpec() && ! child(i)->next())
            out << " (type: " << typespec().string() << ")";
        out << "\n";
        printlist (out, child(i), indentlevel+1);
    }
}



const char *
ASTNode::type_c_str (const TypeSpec &type) const
{
    return m_compiler->type_c_str (type);
}



const char *
ASTshader_declaration::childname (size_t i) const
{
    static const char *name[] = { "metadata", "formals", "statements" };
    return name[i];
}



void
ASTshader_declaration::print (std::ostream &out, int indentlevel) const
{
    indent (out, indentlevel);
    out << "(" << nodetypename() << " " << shadertypename() 
              << " \"" << m_shadername << "\"\n";
    printchildren (out, indentlevel);
    indent (out, indentlevel);
    out << ")\n";
}



const char *
ASTshader_declaration::shadertypename () const
{
    return OSL::pvt::shadertypename ((ShaderType)m_op);
}



ASTfunction_declaration::ASTfunction_declaration (OSLCompilerImpl *comp,
                             TypeSpec type, ustring name,
                             ASTNode *form, ASTNode *stmts, ASTNode *meta)
    : ASTNode (function_declaration_node, comp, 0, meta, form, stmts),
      m_name(name), m_sym(NULL), m_is_builtin(false)
{
    m_typespec = type;
    Symbol *f = comp->symtab().clash (name);
    if (f && f->symtype() != SymTypeFunction) {
        error ("\"%s\" already declared in this scope as a ", name.c_str(),
               f->typespec().string().c_str());
        // FIXME -- print the file and line of the other definition
        f = NULL;
    }

    // FIXME -- allow multiple function declarations, but only if they
    // aren't the same polymorphic type.

    if (name[0] == '_' && name[1] == '_' && name[2] == '_') {
        error ("\"%s\" : sorry, can't start with three underscores",
               name.c_str());
    }

    m_sym = new FunctionSymbol (name, type, this);
    func()->nextpoly ((FunctionSymbol *)f);
    std::string argcodes = oslcompiler->code_from_type (m_typespec);
    for (ref arg = formals();  arg;  arg = arg->next()) {
        const TypeSpec &t (arg->typespec());
        if (t == TypeSpec() /* UNKNOWN */) {
            m_typespec = TypeDesc::UNKNOWN;
            return;
        }
        argcodes += oslcompiler->code_from_type (t);
    }
    func()->argcodes (ustring (argcodes));
    oslcompiler->symtab().insert (m_sym);

    // Typecheck it right now, upon declaration
    typecheck (typespec ());
}



void
ASTfunction_declaration::add_meta (ASTNode *meta)
{
    for (  ;  meta;  meta = meta->nextptr()) {
        ASSERT (meta->nodetype() == ASTNode::variable_declaration_node);
        const ASTvariable_declaration *metavar = static_cast<const ASTvariable_declaration *>(meta);
        Symbol *metasym = metavar->sym();
        if (metasym->name() == "builtin") {
            m_is_builtin = true;
            if (func()->typespec().is_closure()) // It is a builtin closure
                // Force keyword arguments at the end
                func()->argcodes(ustring(std::string(func()->argcodes().c_str()) + "."));

        }
        else if (metasym->name() == "derivs")
            func()->takes_derivs (true);
        else if (metasym->name() == "printf_args")
            func()->printf_args (true);
        else if (metasym->name() == "texture_args")
            func()->texture_args (true);
        else if (metasym->name() == "rw")
            func()->readwrite_special_case (true);
    }
}



const char *
ASTfunction_declaration::childname (size_t i) const
{
    static const char *name[] = { "metadata", "formals", "statements" };
    return name[i];
}



void
ASTfunction_declaration::print (std::ostream &out, int indentlevel) const
{
    indent (out, indentlevel);
    out << nodetypename() << " " << m_sym->mangled();
    if (m_sym->scope())
        out << " (" << m_sym->name() 
                  << " in scope " << m_sym->scope() << ")";
    out << "\n";
    printchildren (out, indentlevel);
}



ASTvariable_declaration::ASTvariable_declaration (OSLCompilerImpl *comp,
                                                  const TypeSpec &type,
                                                  ustring name, ASTNode *init,
                                                  bool isparam, bool ismeta,
                                                  bool isoutput, bool initlist)
    : ASTNode (variable_declaration_node, comp, 0, init, NULL /* meta */),
      m_name(name), m_sym(NULL), 
      m_isparam(isparam), m_isoutput(isoutput), m_ismetadata(ismeta),
      m_initlist(initlist)
{
    m_typespec = type;
    Symbol *f = comp->symtab().clash (name);
    if (f) {
        std::string e = Strutil::format ("\"%s\" already declared in this scope", name.c_str());
        if (f->node()) {
            boost::filesystem::path p(f->node()->sourcefile().string());
            e += Strutil::format ("\n\t\tprevious declaration was at %s:%d",
                                  p.filename().c_str(), f->node()->sourceline());
        }
        if (f->scope() == 0 && f->symtype() == SymTypeFunction && isparam) {
            // special case: only a warning for param to mask global function
            warning ("%s", e.c_str());
        } else {
            error ("%s", e.c_str());
        }
    }
    if (name[0] == '_' && name[1] == '_' && name[2] == '_') {
        error ("\"%s\" : sorry, can't start with three underscores",
               name.c_str());
    }
    SymType symtype = isparam ? (isoutput ? SymTypeOutputParam : SymTypeParam)
                              : SymTypeLocal;
    m_sym = new Symbol (name, type, symtype, this);
    if (! m_ismetadata)
        oslcompiler->symtab().insert (m_sym);

    // A struct really makes several subvariables
    if (type.is_structure() || type.is_structure_array()) {
        ASSERT (! m_ismetadata);
        // Add the fields as individual declarations
        m_compiler->add_struct_fields (type.structspec(), m_sym->name(),
                                       symtype, type.arraylength(), this);
    }
}



const char *
ASTvariable_declaration::nodetypename () const
{
    return m_isparam ? "parameter" : "variable_declaration";
}



const char *
ASTvariable_declaration::childname (size_t i) const
{
    static const char *name[] = { "initializer", "metadata" };
    return name[i];
}



void
ASTvariable_declaration::print (std::ostream &out, int indentlevel) const
{
    indent (out, indentlevel);
    out << "(" << nodetypename() << " " 
              << m_sym->typespec().string() << " " 
              << m_sym->mangled();
#if 0
    if (m_sym->scope())
        out << " (" << m_sym->name() 
                  << " in scope " << m_sym->scope() << ")";
#endif
    out << "\n";
    printchildren (out, indentlevel);
    indent (out, indentlevel);
    out << ")\n";
}



ASTvariable_ref::ASTvariable_ref (OSLCompilerImpl *comp, ustring name)
    : ASTNode (variable_ref_node, comp), m_name(name), m_sym(NULL)
{
    m_sym = comp->symtab().find (name);
    if (! m_sym) {
        error ("'%s' was not declared in this scope", name.c_str());
        // FIXME -- would be fun to troll through the symtab and try to
        // find the things that almost matched and offer suggestions.
        return;
    }
    m_typespec = m_sym->typespec();
}



void
ASTvariable_ref::print (std::ostream &out, int indentlevel) const
{
    indent (out, indentlevel);
    out << "(" << nodetypename() << " (type: "
        << (m_sym ? m_sym->typespec().string() : "unknown") << ") " 
        << (m_sym ? m_sym->mangled() : m_name.string()) << ")\n";
    DASSERT (nchildren() == 0);
}



const char *
ASTpreincdec::childname (size_t i) const
{
    static const char *name[] = { "expression" };
    return name[i];
}



const char *
ASTpostincdec::childname (size_t i) const
{
    static const char *name[] = { "expression" };
    return name[i];
}



ASTindex::ASTindex (OSLCompilerImpl *comp, ASTNode *expr, ASTNode *index)
    : ASTNode (index_node, comp, 0, expr, index)
{
    ASSERT (expr->nodetype() == variable_ref_node ||
            expr->nodetype() == structselect_node);
    if (expr->typespec().is_array())       // array dereference
        m_typespec = expr->typespec().elementtype();
    else if (expr->typespec().is_triple()) // component access
        m_typespec = TypeDesc::FLOAT;
    else {
        error ("indexing into non-array or non-component type");
    }
}



ASTindex::ASTindex (OSLCompilerImpl *comp, ASTNode *expr,
                    ASTNode *index, ASTNode *index2)
    : ASTNode (index_node, comp, 0, expr, index, index2)
{
    ASSERT (expr->nodetype() == variable_ref_node ||
            expr->nodetype() == structselect_node);
    if (expr->typespec().is_matrix())  // matrix component access
        m_typespec = TypeDesc::FLOAT;
    else if (expr->typespec().is_array() &&   // triplearray[][]
             expr->typespec().elementtype().is_triple())
        m_typespec = TypeDesc::FLOAT;
    else {
        error ("indexing into non-array or non-component type");
    }
}



ASTindex::ASTindex (OSLCompilerImpl *comp, ASTNode *expr, ASTNode *index,
          ASTNode *index2, ASTNode *index3)
    : ASTNode (index_node, comp, 0, expr, index, index2, index3)
{
    ASSERT (expr->nodetype() == variable_ref_node ||
            expr->nodetype() == structselect_node);
    if (expr->typespec().is_array() &&   // matrixarray[][]
             expr->typespec().elementtype().is_matrix())
        m_typespec = TypeDesc::FLOAT;
    else {
        error ("indexing into non-array or non-component type");
    }
}



const char *
ASTindex::childname (size_t i) const
{
    static const char *name[] = { "expression", "index", "index" };
    return name[i];
}



ASTstructselect::ASTstructselect (OSLCompilerImpl *comp, ASTNode *expr,
                                  ustring field)
    : ASTNode (structselect_node, comp, 0, expr), m_field(field),
      m_structid(-1), m_fieldid(-1), m_fieldsym(NULL)
{
    m_fieldsym = find_fieldsym (m_structid, m_fieldid);
    if (m_fieldsym) {
        m_fieldname = m_fieldsym->name();
        m_typespec = m_fieldsym->typespec();
    }
}



/// Return the symbol pointer to the individual field that this
/// structselect represents; also set structid to the ID of the
/// structure type, and fieldid to the field index within the struct.
Symbol *
ASTstructselect::find_fieldsym (int &structid, int &fieldid)
{
    if (! lvalue()->typespec().is_structure() &&
        ! lvalue()->typespec().is_structure_array()) {
        return NULL;
    }

    ustring structsymname;
    TypeSpec structtype;
    find_structsym (lvalue().get(), structsymname, structtype);

    structid = structtype.structure();
    StructSpec *structspec (structtype.structspec());
    fieldid = -1;
    for (int i = 0;  i < (int)structspec->numfields();  ++i) {
        if (structspec->field(i).name == m_field) {
            fieldid = i;
            break;
        }
    }

    if (fieldid < 0) {
        error ("struct type '%s' does not have a member '%s'",
               structspec->name().c_str(), m_field.c_str());
        return NULL;
    }

    const StructSpec::FieldSpec &fieldrec (structspec->field(fieldid));
    ustring fieldsymname = ustring::format ("%s.%s", structsymname.c_str(),
                                            fieldrec.name.c_str());
    Symbol *sym = m_compiler->symtab().find (fieldsymname);
    return sym;
}




/// structnode is an AST node representing a struct.  It could be a
/// struct variable, or a field of a struct (which is itself a struct),
/// or an array element of a struct.  Whatever, here we figure out some
/// vital information about it: the name of the symbol representing the
/// struct, and its type.
void
ASTstructselect::find_structsym (ASTNode *structnode, ustring &structname,
                                 TypeSpec &structtype)
{
    // This node selects a field from a struct. The purpose of this
    // method is to "flatten" the possibly-nested (struct in struct, and
    // or array of structs) down to a symbol that represents the
    // particular field.  In the process, we set structname and its
    // type structtype.
    ASSERT (structnode->typespec().is_structure() ||
            structnode->typespec().is_structure_array());
    if (structnode->nodetype() == variable_ref_node) {
        // The structnode is a top-level struct variable
        ASTvariable_ref *var = (ASTvariable_ref *) structnode;
        structname = var->name();
        structtype = var->typespec();
    }
    else if (structnode->nodetype() == structselect_node) {
        // The structnode is itself a field of another struct.
        ASTstructselect *thestruct = (ASTstructselect *) structnode;
        int structid, fieldid;
        Symbol *sym = thestruct->find_fieldsym (structid, fieldid);
        structname = sym->name();
        structtype = sym->typespec();
    }
    else if (structnode->nodetype() == index_node) {
        // The structnode is an element of an array of structs:
        ASTindex *arrayref = (ASTindex *) structnode;
        find_structsym (arrayref->lvalue().get(), structname, structtype);
        structtype.make_array (0);  // clear its arrayness
    }
    else {
        ASSERT (0 && "Malformed ASTstructselect");
    }
}



const char *
ASTstructselect::childname (size_t i) const
{
    static const char *name[] = { "structure" };
    return name[i];
}



void
ASTstructselect::print (std::ostream &out, int indentlevel) const
{
    ASTNode::print (out, indentlevel);
    indent (out, indentlevel+1);
    out << "select " << field() << "\n";
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
    case LoopDo    : return "dowhile";
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



ASTcompound_initializer::ASTcompound_initializer (OSLCompilerImpl *comp,
                                                  ASTNode *exprlist)
    : ASTNode (compound_initializer_node, comp, Nothing, exprlist)
{
}



const char *
ASTcompound_initializer::childname (size_t i) const
{
    return "expression_list";
}



ASTassign_expression::ASTassign_expression (OSLCompilerImpl *comp, ASTNode *var,
                                            Operator op, ASTNode *expr)
    : ASTNode (assign_expression_node, comp, op, var, expr)
{
    if (op != Assign) {
        // Rejigger to straight assignment and binary op
        m_op = Assign;
        m_children[1] = new ASTbinary_expression (comp, op, var, expr);
    }
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
    case Assign     : return "=";
    case Mul        : return "*=";
    case Div        : return "/=";
    case Add        : return "+=";
    case Sub        : return "-=";
    case BitAnd     : return "&=";
    case BitOr      : return "|=";
    case Xor        : return "^=";
    case ShiftLeft  : return "<<=";
    case ShiftRight : return ">>=";
    default: ASSERT (0 && "unknown assignment expression");
    }
}



const char *
ASTassign_expression::opword () const
{
    switch (m_op) {
    case Assign     : return "assign";
    case Mul        : return "mul";
    case Div        : return "div";
    case Add        : return "add";
    case Sub        : return "sub";
    case BitAnd     : return "bitand";
    case BitOr      : return "bitor";
    case Xor        : return "xor";
    case ShiftLeft  : return "shl";
    case ShiftRight : return "shr";
    default: ASSERT (0 && "unknown assignment expression");
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
    case Add   : return "+";
    case Sub   : return "-";
    case Not   : return "!";
    case Compl : return "~";
    default: ASSERT (0 && "unknown unary expression");
    }
}



const char *
ASTunary_expression::opword () const
{
    switch (m_op) {
    case Add   : return "add";
    case Sub   : return "neg";
    case Not   : return "not";
    case Compl : return "compl";
    default: ASSERT (0 && "unknown unary expression");
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
    case BitAnd       : return "&";
    case BitOr        : return "|";
    case Xor          : return "^";
    case And          : return "&&";
    case Or           : return "||";
    case ShiftLeft    : return "<<";
    case ShiftRight   : return ">>";
    default: ASSERT (0 && "unknown binary expression");
    }
}



const char *
ASTbinary_expression::opword () const
{
    switch (m_op) {
    case Mul          : return "mul";
    case Div          : return "div";
    case Add          : return "add";
    case Sub          : return "sub";
    case Mod          : return "mod";
    case Equal        : return "eq";
    case NotEqual     : return "neq";
    case Greater      : return "gt";
    case GreaterEqual : return "ge";
    case Less         : return "lt";
    case LessEqual    : return "le";
    case BitAnd       : return "bitand";
    case BitOr        : return "bitor";
    case Xor          : return "xor";
    case And          : return "and";
    case Or           : return "or";
    case ShiftLeft    : return "shl";
    case ShiftRight   : return "shr";
    default: ASSERT (0 && "unknown binary expression");
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



const char *
ASTtype_constructor::childname (size_t i) const
{
    static const char *name[] = { "args" };
    return name[i];
}



ASTfunction_call::ASTfunction_call (OSLCompilerImpl *comp, ustring name,
                                    ASTNode *args)
    : ASTNode (function_call_node, comp, 0, args), m_name(name),
      m_argread(~1),      // Default - all args are read except the first
      m_argwrite(1),      // Default - first arg only is written by the op
      m_argtakesderivs(0) // Default - doesn't take derivs
{
    m_sym = comp->symtab().find (name);
    if (! m_sym) {
        error ("function '%s' was not declared in this scope", name.c_str());
        // FIXME -- would be fun to troll through the symtab and try to
        // find the things that almost matched and offer suggestions.
        return;
    }
    if (m_sym->symtype() != SymTypeFunction) {
        error ("'%s' is not a function", name.c_str());
        return;
    }
}



const char *
ASTfunction_call::childname (size_t i) const
{
    return ustring::format ("param%d", (int)i).c_str();
}



const char *
ASTfunction_call::opname () const
{
    return m_name.c_str ();
}



void
ASTfunction_call::print (std::ostream &out, int indentlevel) const 
{
    ASTNode::print (out, indentlevel);
#if 0
    if (is_user_function()) { 
        out << "\n";
        user_function()->print (out, indentlevel+1);
        out << "\n";
    }
#endif
}



const char *
ASTliteral::childname (size_t i) const
{
    return NULL;
}



void
ASTliteral::print (std::ostream &out, int indentlevel) const
{
    indent (out, indentlevel);
    out << "(" << nodetypename() << " (type: " << m_typespec.string() << ") ";
    if (m_typespec.is_int())
        out << m_i;
    else if (m_typespec.is_float())
        out << m_f;
    else if (m_typespec.is_string())
        out << "\"" << m_s << "\"";
    out << ")\n";
}


}; // namespace pvt

OSL_NAMESPACE_EXIT
