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

#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <functional>
#ifndef NDEBUG
#include <atomic>
#endif

#include "osl_pvt.h"
#include "oslcomp_pvt.h"

#include <OpenImageIO/dassert.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/atomic.h>
namespace Strutil = OIIO::Strutil;

OSL_NAMESPACE_ENTER

namespace pvt {   // OSL::pvt


#ifndef NDEBUG
// When in DEBUG mode, track the number of AST nodes of each type that
// are allocated and remaining, and at program exit print a message about
// any leaked nodes.
namespace {
std::atomic<int> node_counts[ASTNode::_last_node];
std::atomic<int> node_counts_peak[ASTNode::_last_node];

class ScopeExit {
public:
    typedef std::function<void()> Task;
    explicit ScopeExit (Task&& task) : m_task(std::forward<Task>(task)) {}
    ~ScopeExit () { m_task(); }
private:
    Task m_task;
};

ScopeExit print_node_counts ([](){
    for (int i = 0; i < ASTNode::_last_node; ++i)
        if (node_counts[i] > 0)
            Strutil::printf ("ASTNode type %2d: %5d   (peak %5d)\n",
                             i, node_counts[i], node_counts_peak[i]);
});
}
#endif



ASTNode::ref
reverse (ASTNode::ref list)
{
    ASTNode::ref new_list;
    while (list) {
        ASTNode::ref next = list->next();
        list->m_next = new_list;
        new_list = list;
        list = next;
    }
    return new_list;
}



ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler) 
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(0), m_is_lvalue(false)
{
#ifndef NDEBUG
    node_counts[nodetype] += 1;
    node_counts_peak[nodetype] += 1;
#endif
}



ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op,
                  ASTNode *a)
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(op), m_is_lvalue(false)
{
    addchild (a);
#ifndef NDEBUG
    node_counts[nodetype] += 1;
    node_counts_peak[nodetype] += 1;
#endif
}



ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op)
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(op), m_is_lvalue(false)
{
#ifndef NDEBUG
    node_counts[nodetype] += 1;
    node_counts_peak[nodetype] += 1;
#endif
}



ASTNode::ASTNode (NodeType nodetype, OSLCompilerImpl *compiler, int op,
                  ASTNode *a, ASTNode *b)
    : m_nodetype(nodetype), m_compiler(compiler),
      m_sourcefile(compiler->filename()),
      m_sourceline(compiler->lineno()), m_op(op), m_is_lvalue(false)
{
    addchild (a);
    addchild (b);
#ifndef NDEBUG
    node_counts[nodetype] += 1;
    node_counts_peak[nodetype] += 1;
#endif
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
#ifndef NDEBUG
    node_counts[nodetype] += 1;
    node_counts_peak[nodetype] += 1;
#endif
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
#ifndef NDEBUG
    node_counts[nodetype] += 1;
    node_counts_peak[nodetype] += 1;
#endif
}



ASTNode::~ASTNode ()
{
#ifndef NDEBUG
    node_counts[nodetype()] -= 1;
#endif
}



void
ASTNode::error_impl (string_view msg) const
{
    m_compiler->error (sourcefile(), sourceline(), "%s", msg);
}



void
ASTNode::warning_impl (string_view msg) const
{
    m_compiler->warning (sourcefile(), sourceline(), "%s", msg);
}



void
ASTNode::info_impl (string_view msg) const
{
    m_compiler->info (sourcefile(), sourceline(), "%s", msg);
}



void
ASTNode::message_impl (string_view msg) const
{
    m_compiler->message (sourcefile(), sourceline(), "%s", msg);
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



void
ASTNode::list_to_vec (const ref &A, std::vector<ref> &vec)
{
    vec.clear ();
    for (ref node = A; node; node = node->next())
        vec.push_back (node);
}



ASTNode::ref
ASTNode::vec_to_list (std::vector<ref> &vec)
{
    if (vec.size()) {
        for (size_t i = 0;  i < vec.size()-1;  ++i)
            vec[i]->m_next = vec[i+1];
        vec[vec.size()-1]->m_next = NULL;
        return vec[0];
    } else {
        return ref();
    }
}



std::string
ASTNode::list_to_types_string (const ASTNode *node)
{
    std::ostringstream result;
    for (int i = 0; node; node = node->nextptr(), ++i) {
        if (i)
            result << ", ";
        result << node->typespec();
    }
    return result.str();
}



ASTshader_declaration::ASTshader_declaration (OSLCompilerImpl *comp,
                                int stype, ustring name, ASTNode *form,
                                ASTNode *stmts, ASTNode *meta)
    : ASTNode (shader_declaration_node, comp, stype, meta, form, stmts),
      m_shadername(name)
{
    // Double check some requirements of shader parameters
    for (ASTNode *arg = form;  arg;  arg = arg->nextptr()) {
        ASSERT (arg->nodetype() == variable_declaration_node);
        ASTvariable_declaration *v = (ASTvariable_declaration *)arg;
        if (! v->init())
            v->error ("shader parameter '%s' requires a default initializer",
                      v->name());
        if (v->is_output() && v->typespec().is_unsized_array())
            v->error ("shader output parameter '%s' can't be unsized array",
                      v->name());
    }
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



string_view
ASTshader_declaration::shadertypename () const
{
    return OSL::pvt::shadertypename ((ShaderType)m_op);
}



ASTfunction_declaration::ASTfunction_declaration (OSLCompilerImpl *comp,
                             TypeSpec type, ustring name,
                             ASTNode *form, ASTNode *stmts, ASTNode *meta,
                             int sourceline_start)
    : ASTNode (function_declaration_node, comp, 0, meta, form, stmts),
      m_name(name), m_sym(NULL), m_is_builtin(false)
{
    // Some trickery -- the compiler's idea of the "current" source line
    // is the END of the function body, so if a hint was passed about the
    // start of the declaration, substitute that.
    if (sourceline_start >= 0)
        m_sourceline = sourceline_start;

    if (Strutil::starts_with (name, "___"))
        error ("\"%s\" : sorry, can't start with three underscores", name);

    // Get a pointer to the first of the existing symbols of that name.
    Symbol *existing_syms = comp->symtab().clash (name);
    if (existing_syms && existing_syms->symtype() != SymTypeFunction) {
        error ("\"%s\" already declared in this scope as a %s",
               name, existing_syms->typespec());
        // FIXME -- print the file and line of the other definition
        existing_syms = NULL;
    }

    // Build up the argument signature for this declared function
    m_typespec = type;
    std::string argcodes = oslcompiler->code_from_type (m_typespec);
    for (ASTNode *arg = form;  arg;  arg = arg->nextptr()) {
        const TypeSpec &t (arg->typespec());
        if (t == TypeSpec() /* UNKNOWN */) {
            m_typespec = TypeDesc::UNKNOWN;
            return;
        }
        argcodes += oslcompiler->code_from_type (t);
        ASSERT (arg->nodetype() == variable_declaration_node);
        ASTvariable_declaration *v = (ASTvariable_declaration *)arg;
        if (v->init())
            v->error ("function parameter '%s' may not have a default initializer.",
                      v->name().c_str());
    }

    // Allow multiple function declarations, but only if they aren't the
    // same polymorphic type in the same scope.
    if (stmts) {
        std::string err;
        int current_scope = oslcompiler->symtab().scopeid();
        for (FunctionSymbol *f = static_cast<FunctionSymbol *>(existing_syms);
             f; f = f->nextpoly()) {
            if (f->scope() == current_scope && f->argcodes() == argcodes) {
                // If the argcodes match, only one should have statements.
                // If there is no ASTNode for the poly, must be a builtin, and
                // has 'implicit' statements.
                auto other = static_cast<ASTfunction_declaration*>(f->node());
                if (!other || (other->statements() || other->is_builtin())) {
                    if (err.empty()) {
                        err = Strutil::sprintf("Function '%s %s (%s)' redefined "
                                              "in the same scope\n"
                                              "  Previous definitions:", type,
                                              name, list_to_types_string(form));
                    }
                    err += "\n    ";
                    if (other) {
                        err += Strutil::sprintf("%s:%d",
                                    OIIO::Filesystem::filename(other->sourcefile().string()),
                                    other->sourceline());
                    } else
                        err += "built-in";
                }
            }
        }
        if (!err.empty())
            warning ("%s", err);
    }


    m_sym = new FunctionSymbol (name, type, this);
    func()->nextpoly ((FunctionSymbol *)existing_syms);

    func()->argcodes (ustring (argcodes));
    oslcompiler->symtab().insert (m_sym);

    // Typecheck it right now, upon declaration
    typecheck (typespec ());
}



void
ASTfunction_declaration::add_meta (ref metaref)
{
    for (ASTNode *meta = metaref.get();  meta;  meta = meta->nextptr()) {
        ASSERT (meta->nodetype() == ASTNode::variable_declaration_node);
        const ASTvariable_declaration *metavar = static_cast<const ASTvariable_declaration *>(meta);
        Symbol *metasym = metavar->sym();
        if (metasym->name() == "builtin") {
            m_is_builtin = true;
            if (func()->typespec().is_closure())  { // It is a builtin closure
                // Force keyword arguments at the end
                func()->argcodes(ustring(std::string(func()->argcodes().c_str()) + "."));
            }
            // For built-in functions, if any of the params are output,
            // also automatically mark it as readwrite_special_case.
            for (ASTNode *f = formals().get(); f; f = f->nextptr()) {
                ASSERT (f->nodetype() == variable_declaration_node);
                ASTvariable_declaration *v = (ASTvariable_declaration *)f;
                if (v->is_output())
                    func()->readwrite_special_case (true);
            }
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
                                                  bool isoutput, bool initlist,
                                                  int sourceline_start)
    : ASTNode (variable_declaration_node, comp, 0, init, NULL /* meta */),
      m_name(name), m_sym(NULL),
      m_isparam(isparam), m_isoutput(isoutput), m_ismetadata(ismeta),
      m_initlist(initlist)
{
    // Some trickery -- the compiler's idea of the "current" source line
    // is the END of the declaration, so if a hint was passed about the
    // start of the declaration, substitute that.
    if (sourceline_start >= 0)
        m_sourceline = sourceline_start;

    if (m_initlist && init) {
        // Typecheck the init list early.
        ASSERT (init->nodetype() == compound_initializer_node);
        static_cast<ASTcompound_initializer*>(init)->typecheck(type);
    }

    m_typespec = type;
    Symbol *f = comp->symtab().clash (name);
    if (f  &&  ! m_ismetadata) {
        std::string e = Strutil::sprintf ("\"%s\" already declared in this scope", name.c_str());
        if (f->node()) {
            std::string filename = OIIO::Filesystem::filename(f->node()->sourcefile().string());
            e += Strutil::sprintf ("\n\t\tprevious declaration was at %s:%d",
                                   filename, f->node()->sourceline());
        }
        if (f->scope() == 0 && f->symtype() == SymTypeFunction && isparam) {
            // special case: only a warning for param to mask global function
            warning ("%s", e);
        } else {
            error ("%s", e);
        }
    }
    if (OIIO::Strutil::starts_with (name, "___")) {
        error ("\"%s\" : sorry, can't start with three underscores", name);
    }
    SymType symtype = isparam ? (isoutput ? SymTypeOutputParam : SymTypeParam)
                              : SymTypeLocal;
    // Sneaky debugging aid: a local that starts with "__debug_tmp__"
    // gets declared as a temp. Don't do this on purpose!!!
    if (symtype == SymTypeLocal && Strutil::starts_with (name, "__debug_tmp__"))
        symtype = SymTypeTemp;
    m_sym = new Symbol (name, type, symtype, this);
    if (! m_ismetadata)
        oslcompiler->symtab().insert (m_sym);

    // A struct really makes several subvariables
    if (type.is_structure() || type.is_structure_array()) {
        ASSERT (! m_ismetadata);
        // Add the fields as individual declarations
        m_compiler->add_struct_fields (type.structspec(), m_sym->name(), symtype,
                                       type.is_unsized_array() ? -1 : type.arraylength(),
                                       this, init);
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
    if (m_sym->symtype() == SymTypeFunction) {
        error ("function '%s' can't be used as a variable", name.c_str());
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



ASTpreincdec::ASTpreincdec (OSLCompilerImpl *comp, int op, ASTNode *expr)
    : ASTNode (preincdec_node, comp, op, expr)
{
    check_symbol_writeability (expr);
}



const char *
ASTpreincdec::childname (size_t i) const
{
    static const char *name[] = { "expression" };
    return name[i];
}



ASTpostincdec::ASTpostincdec (OSLCompilerImpl *comp, int op, ASTNode *expr)
    : ASTNode (postincdec_node, comp, op, expr)
{
    check_symbol_writeability (expr);
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
    else if (!expr->typespec().is_closure() &&
             expr->typespec().is_triple()) // component access
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
        error ("type '%s' does not have a member '%s'",
               type_c_str(lvalue()->typespec()), m_field);
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
               structspec->name(), m_field);
        return NULL;
    }

    const StructSpec::FieldSpec &fieldrec (structspec->field(fieldid));
    ustring fieldsymname = ustring::format ("%s.%s", structsymname,
                                            fieldrec.name);
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



ASTloop_statement::ASTloop_statement (OSLCompilerImpl *comp, LoopType looptype,
                                      ASTNode *init, ASTNode *cond,
                                      ASTNode *iter, ASTNode *stmt)
    : ASTNode (loop_statement_node, comp, looptype, init, cond, iter, stmt)
{
    // Handle empty comparison, for(;;), is same as for(;1;)
    if (!cond)
        m_children[1] = new ASTliteral(comp, 1);
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
    : ASTtype_constructor (compound_initializer_node, comp, TypeSpec(), exprlist),
      m_ctor(false)
{
}



const char *
ASTcompound_initializer::childname (size_t i) const
{
    return canconstruct() ? "args" : "expression_list";
}



bool
ASTNode::check_symbol_writeability (ASTNode *var)
{
    if (var->nodetype() == index_node)
        return check_symbol_writeability (static_cast<ASTindex*>(var)->lvalue().get());
    if (var->nodetype() == structselect_node)
        return check_symbol_writeability (static_cast<ASTstructselect*>(var)->lvalue().get());

    Symbol *dest = nullptr;
    if (var->nodetype() == variable_ref_node)
        dest = static_cast<ASTvariable_ref*>(var)->sym();
    else if (var->nodetype() == variable_declaration_node)
        dest = static_cast<ASTvariable_declaration*>(var)->sym();

    if (dest) {
        if (dest->readonly()) {
            warning ("cannot write to non-output parameter \"%s\"", dest->name());
            // Note: Consider it only a warning to write to a non-output
            // parameter. Users who want it to be a hard error can use
            // -Werror. Writing to any other readonly symbols is a full
            // error.
            return false;
        }
    } else {
        // std::cout << "Don't know how to check_symbol_writeability "
        //           << var->nodetypename() << "\n";
    }
    return true;
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

    check_symbol_writeability (var);
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



ASTunary_expression::ASTunary_expression (OSLCompilerImpl *comp, int op,
                                          ASTNode *expr)
    : ASTNode (unary_expression_node, comp, op, expr)
{
    // Check for a user-overloaded function for this operator
    Symbol *sym = comp->symtab().find (ustring::format ("__operator__%s__", opword()));
    if (sym && sym->symtype() == SymTypeFunction)
        m_function_overload = (FunctionSymbol *)sym;
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



ASTbinary_expression::ASTbinary_expression (OSLCompilerImpl *comp, Operator op,
                                            ASTNode *left, ASTNode *right)
    : ASTNode (binary_expression_node, comp, op, left, right)
{
    // Check for a user-overloaded function for this operator.
    // Disallow a few ops from overloading.
    if (op != And && op != Or) {
        ustring funcname = ustring::format ("__operator__%s__", opword());
        Symbol *sym = comp->symtab().find (funcname);
        if (sym && sym->symtype() == SymTypeFunction)
            m_function_overload = (FunctionSymbol *)sym;
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
                                    ASTNode *args, FunctionSymbol *funcsym)
    : ASTNode (function_call_node, comp, 0, args), m_name(name),
      m_sym(funcsym ? funcsym : comp->symtab().find (name)), // Look it up.
      m_poly(funcsym),    // Default - resolved symbol or null
      m_argread(~1),      // Default - all args are read except the first
      m_argwrite(1),      // Default - first arg only is written by the op
      m_argtakesderivs(0) // Default - doesn't take derivs
{
    if (! m_sym) {
        error ("function '%s' was not declared in this scope", name);
        // FIXME -- would be fun to troll through the symtab and try to
        // find the things that almost matched and offer suggestions.
        return;
    }
    if (is_struct_ctr()) {
        return;  // It's a struct constructor
    }
    if (m_sym->symtype() != SymTypeFunction) {
        error ("'%s' is not a function", name.c_str());
        m_sym = NULL;
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
