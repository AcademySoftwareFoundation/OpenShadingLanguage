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
#include <fstream>
#include <cstdio>
#include <streambuf>
#ifdef __GNUC__
# include <ext/stdio_filebuf.h>
#endif
#include <cstdio>
#include <cerrno>

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/sysutil.h"
#include "OpenImageIO/dassert.h"

#include "oslcomp_pvt.h"


#define yyFlexLexer oslFlexLexer
#include "FlexLexer.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {


OSLCompiler *
OSLCompiler::create ()
{
    return new pvt::OSLCompilerImpl;
}



namespace pvt {   // OSL::pvt


OSLCompilerImpl *oslcompiler = NULL;


OSLCompilerImpl::OSLCompilerImpl ()
    : m_lexer(NULL), m_err(false), m_symtab(*this),
      m_current_typespec(TypeDesc::UNKNOWN), m_current_output(false),
      m_verbose(false), m_debug(false), m_next_temp(0), m_next_const(0),
      m_osofile(NULL), m_sourcefile(NULL), m_last_sourceline(0),
      m_total_nesting(0), m_loop_nesting(0)
{
    initialize_globals ();
    initialize_builtin_funcs ();
}



OSLCompilerImpl::~OSLCompilerImpl ()
{
    if (m_sourcefile) {
        fclose (m_sourcefile);
        m_sourcefile = NULL;
    }
}



void
OSLCompilerImpl::error (ustring filename, int line, const char *format, ...)
{
    va_list ap;
    va_start (ap, format);
    std::string errmsg = format ? Strutil::vformat (format, ap) : "syntax error";
    if (filename.c_str())
        fprintf (stderr, "%s:%d: error: %s\n", 
                 filename.c_str(), line, errmsg.c_str());
    else
        fprintf (stderr, "error: %s\n", errmsg.c_str());

    va_end (ap);
    m_err = true;
}



void
OSLCompilerImpl::warning (ustring filename, int line, const char *format, ...)
{
    va_list ap;
    va_start (ap, format);
    std::string errmsg = format ? Strutil::vformat (format, ap) : "";
    fprintf (stderr, "%s:%d: warning: %s\n", 
             filename.c_str(), line, errmsg.c_str());
    va_end (ap);
}



bool
OSLCompilerImpl::compile (const std::string &filename,
                          const std::vector<std::string> &options)
{
    if (! boost::filesystem::exists (filename)) {
        error (ustring(), 0, "Input file \"%s\" not found", filename.c_str());
        return false;
    }

    std::string cppcommand = "/usr/bin/cpp -xc -nostdinc ";

    // Determine where the installed shader include directory is, and
    // look for ../shaders/stdosl.h and force it to include.
    std::string program = Sysutil::this_program_path ();
    if (program.size()) {
        boost::filesystem::path path (program);  // our program
#if BOOST_VERSION >= 103600
        path = path.parent_path ();  // now the bin dir of our program
        path = path.parent_path ();  // now the parent dir
#else
        path = path.branch_path ();  // now the bin dir of our program
        path = path.branch_path ();  // now the parent dir
#endif
        path = path / "shaders";
        bool found = false;
        if (boost::filesystem::exists (path)) {
            path = path / "stdosl.h";
            if (boost::filesystem::exists (path)) {
                cppcommand += std::string("-include ") + path.string() + " ";
                found = true;
            }
        }
        if (! found)
            warning (ustring(filename), 0, "Unable to find \"%s\"",
                     path.string().c_str());
    }

    m_output_filename.clear ();
    for (size_t i = 0;  i < options.size();  ++i) {
        if (options[i] == "-v") {
            // verbose mode
            m_verbose = true;
        } else if (options[i] == "-d") {
            // debug mode
            m_debug = true;
        } else if (options[i] == "-o" && i < options.size()-1) {
            ++i;
            m_output_filename = options[i];
        } else {
            // something meant for the cpp command
            cppcommand += "\"";
            cppcommand += options[i];
            cppcommand += "\" ";
        }
    }
    cppcommand += "\"";
    cppcommand += filename;
    cppcommand += "\" ";

    // std::cout << "cpp command:\n>" << cppcommand << "<\n";

    FILE *cpppipe = popen (cppcommand.c_str(), "r");

#ifdef __GNUC__
    __gnu_cxx::stdio_filebuf<char> fb (cpppipe, std::ios::in);
#else
    std::filebuf fb (cpppipe);
#endif

    if (fb.is_open()) {
        std::istream in (&fb);
        oslcompiler = this;

        // Create a lexer, parse the file, delete the lexer
        m_lexer = new oslFlexLexer (&in);
        oslparse ();
        bool parseerr = error_encountered();
        delete m_lexer;

        // All done with the input, close the files
        fb.close ();
        pclose (cpppipe);
        cpppipe = NULL;

        if (! parseerr) {
            oslcompiler->shader()->typecheck ();
        }

        // Print the parse tree if there were no errors
        if (m_debug) {
            oslcompiler->symtab().print ();
            if (oslcompiler->shader())
                oslcompiler->shader()->print (std::cout);
        }

        if (! error_encountered()) {
            oslcompiler->shader()->codegen ();
            track_variable_usage ();
        }
 
        if (! error_encountered()) {
            if (m_output_filename.size() == 0)
                m_output_filename = default_output_filename ();
            write_oso_file (m_output_filename);
        }

        oslcompiler = NULL;
    }

    return ! error_encountered();
}



struct GlobalTable {
    const char *name;
    TypeSpec type;
};

static GlobalTable globals[] = {
    { "P", TypeDesc::TypePoint },
    { "I", TypeDesc::TypeVector },
    { "N", TypeDesc::TypeNormal },
    { "Ng", TypeDesc::TypeNormal },
    { "u", TypeDesc::TypeFloat },
    { "v", TypeDesc::TypeFloat },
    { "dPdu", TypeDesc::TypeVector },
    { "dPdv", TypeDesc::TypeVector },
#if 0
    // Light variables -- we don't seem to be on a route to support this
    // kind of light shader, so comment these out for now.
    { "L", TypeDesc::TypeVector },
    { "Cl", TypeDesc::TypeColor },
    { "Ns", TypeDesc::TypeNormal },
    { "Pl", TypeDesc::TypePoint },
    { "Nl", TypeDesc::TypeNormal },
#endif
    { "Ps", TypeDesc::TypePoint },
    { "Ci", TypeSpec (TypeDesc::TypeColor, true) },
    { "time", TypeDesc::TypeFloat },
    { "dtime", TypeDesc::TypeFloat },
    { "dPdtime", TypeDesc::TypeVector },
    { NULL }
};


void
OSLCompilerImpl::initialize_globals ()
{
    for (int i = 0;  globals[i].name;  ++i) {
        Symbol *s = new Symbol (ustring(globals[i].name), globals[i].type,
                                SymTypeGlobal);
        symtab().insert (s);
    }
}



std::string
OSLCompilerImpl::default_output_filename ()
{
    if (m_shader && shader_decl())
        return shader_decl()->shadername().string() + ".oso";
    return std::string();
}



void
OSLCompilerImpl::write_oso_metadata (const ASTNode *metanode) const
{
    const ASTvariable_declaration *metavar = dynamic_cast<const ASTvariable_declaration *>(metanode);
    ASSERT (metavar);
    Symbol *metasym = metavar->sym();
    ASSERT (metasym);
    TypeSpec ts = metasym->typespec();
    oso ("%%meta{%s,%s,", ts.string().c_str(), metasym->name().c_str());
    const ASTNode *init = metavar->init().get();
    ASSERT (init);
    if (ts.is_string() && init->nodetype() == ASTNode::literal_node)
        oso ("\"%s\"", ((const ASTliteral *)init)->strval());
    else if (ts.is_int() && init->nodetype() == ASTNode::literal_node)
        oso ("%d", ((const ASTliteral *)init)->intval());
    else if (ts.is_float() && init->nodetype() == ASTNode::literal_node)
        oso ("%.8g", ((const ASTliteral *)init)->floatval());
    // FIXME -- what about type constructors?
    else {
        std::cout << "Error, don't know how to print metadata " 
                  << ts.string() << " with node type " 
                  << init->nodetypename() << "\n";
        ASSERT (0);  // FIXME
    }
    oso ("} ");
}



void
OSLCompilerImpl::write_oso_const_value (const ConstantSymbol *sym) const
{
    ASSERT (sym);
    if (sym->typespec().is_string())
        oso ("\"%s\"", sym->strval().c_str());
    else if (sym->typespec().is_int())
        oso ("%d", sym->intval());
    else if (sym->typespec().is_float())
        oso ("%.8g", sym->floatval());
    else if (sym->typespec().is_triple())
        oso ("%.8g %.8g %.8g", sym->vecval()[0], sym->vecval()[1], sym->vecval()[2]);
    else {
        ASSERT (0 && "Only know how to output const vals that are single int, float, string");
    }
}



void
OSLCompilerImpl::write_oso_symbol (const Symbol *sym) const
{
    oso ("%s\t%s\t%s", sym->symtype_shortname(),
         type_c_str(sym->typespec()), sym->mangled().c_str());

    ASTvariable_declaration *v = NULL;
    if (sym->node())
        v = dynamic_cast<ASTvariable_declaration *>(sym->node());

    // Print default values
    if (sym->symtype() == SymTypeConst) {
        oso ("\t");
        write_oso_const_value (dynamic_cast<const ConstantSymbol *>(sym));
        oso ("\t");
    } else if (v && (sym->symtype() == SymTypeParam ||
                    sym->symtype() == SymTypeOutputParam)) {
        std::string out;
        v->param_default_literals (sym, out);
        oso ("\t%s\t", out.c_str());
    }

    int hints = 0;
    if (v) {
        ASSERT (v);
        for (ASTNode::ref m = v->meta();  m;  m = m->next()) {
            if (hints++ == 0)
                oso ("\t");
            write_oso_metadata (m.get());
        }
    }

    if (hints++ == 0)
        oso ("\t");
    oso (" %%read{%d,%d} %%write{%d,%d}", sym->firstread(), sym->lastread(),
         sym->firstwrite(), sym->lastwrite());

    if (sym->typespec().is_structure()) {
        if (hints++ == 0)
            oso ("\t");
        const StructSpec *structspec (sym->typespec().structspec());
        std::string fieldlist, signature;
        for (int i = 0;  i < (int)structspec->numfields();  ++i) {
            if (i > 0)
                fieldlist += ",";
            fieldlist += structspec->field(i).name.string();
            signature += code_from_type (structspec->field(i).type);
        }
        oso (" %%struct{\"%s\"} %%structfields{%s} %%structfieldtypes{\"%s\"} %%structnfields{%d}",
             structspec->mangled().c_str(), fieldlist.c_str(),
             signature.c_str(), structspec->numfields());
    }
    if (sym->fieldid() >= 0) {
        if (hints++ == 0)
            oso ("\t");
        ASTvariable_declaration *vd = (ASTvariable_declaration *) sym->node();
        oso (" %%mystruct{%s} %%mystructfield{%d}",
             vd->sym()->mangled().c_str(), sym->fieldid());
    }

    oso ("\n");
}



void
OSLCompilerImpl::write_oso_file (const std::string &outfilename)
{
    ASSERT (m_osofile == NULL);
    m_osofile = fopen (outfilename.c_str(), "w");
    if (! m_osofile) {
        error (ustring(), 0, "Could not open \"%s\"", outfilename.c_str());
        return;
    }

    // FIXME -- remove the hard-coded version!
    oso ("OpenShadingLanguage 0.0\n");
    oso ("# Compiled by oslc FIXME-VERSION\n");

    ASTshader_declaration *shaderdecl = shader_decl();
    oso ("%s %s", shaderdecl->shadertypename(), 
         shaderdecl->shadername().c_str());

    // FIXME -- output hints and metadata

    oso ("\n");

    // Output params, so they are first
    BOOST_FOREACH (const Symbol *s, symtab()) {
        if (s->symtype() == SymTypeParam || s->symtype() == SymTypeOutputParam)
            write_oso_symbol (s);
    }
    // Output globals, locals, temps, const
    BOOST_FOREACH (const Symbol *s, symtab()) {
        if (s->symtype() == SymTypeLocal || s->symtype() == SymTypeTemp ||
            s->symtype() == SymTypeGlobal || s->symtype() == SymTypeConst) {
            // Don't bother writing symbols that are never used
            if (s->lastuse() >= 0) {
                write_oso_symbol (s);
            }
        }
    }

    // FIXME -- output all opcodes
    int lastline = -1;
    ustring lastfile;
    ustring lastmethod ("___uninitialized___");
    for (OpcodeVec::iterator op = m_ircode.begin(); op != m_ircode.end();  ++op) {
        if (lastmethod != op->method()) {
            oso ("code %s\n", op->method().c_str());
            lastmethod = op->method();
            lastfile = ustring();
            lastline = -1;
        }

        if (m_debug && op->sourcefile()) {
            ustring file = op->sourcefile();
            int line = op->sourceline();
            if (file != lastfile || line != lastline)
                oso ("# %s:%d\n# %s\n", file.c_str(), line,
                     retrieve_source (file, line).c_str());
        }

        // Op name
        oso ("\t%s", op->opname().c_str());

        // Register arguments
        if (op->nargs())
            oso (op->opname().length() < 8 ? "\t\t" : "\t");
        for (int i = 0;  i < op->nargs();  ++i) {
            int arg = op->firstarg() + i;
            oso ("%s ", m_opargs[arg]->dealias()->mangled().c_str());
        }

        // Jump targets
        for (size_t i = 0;  i < Opcode::max_jumps;  ++i)
            if (op->jump(i) >= 0)
                oso ("%d ", op->jump(i));

        // Hints
        bool firsthint = true;
        // Hint the source file/line
        if (op->sourcefile()) {
            if (op->sourcefile() != lastfile) {
                lastfile = op->sourcefile();
                oso ("%c%%filename{\"%s\"}", firsthint ? '\t' : ' ', lastfile.c_str());
                firsthint = false;
            }
            if (op->sourceline() != lastline) {
                lastline = op->sourceline();
                oso ("%c%%line{%d}", firsthint ? '\t' : ' ', lastline);
                firsthint = false;
            }
        }
        // Hint which args are read/written
        if (op->nargs()) {
            oso ("%c%%argrw{\"", firsthint ? '\t' : ' ');
            for (int i = 0;  i < op->nargs();  ++i) {
                if (op->argwrite(i))
                    oso (op->argread(i) ? "W" : "w");
                else
                    oso (op->argread(i) ? "r" : "-");
            }
            oso ("\"}");
            firsthint = false;
        }
        oso ("\n");
    }

    if (m_ircode.size() == 0)   // If no code, still need a code marker
        oso ("code main\n");

    oso ("\tend\n");

    fclose (m_osofile);
    m_osofile = NULL;
}



void
OSLCompilerImpl::oso (const char *fmt, ...) const
{
    // FIXME -- might be nice to let this save to a memory buffer, not
    // just a file.
    va_list arg_ptr;
    va_start (arg_ptr, fmt);
    vfprintf (m_osofile, fmt, arg_ptr);
    va_end (arg_ptr);
}



std::string
OSLCompilerImpl::retrieve_source (ustring filename, int line)
{
    // If we don't already have the file open, open it
    if (filename != m_last_sourcefile) {
        // If we have another file open, close that one
        if (m_sourcefile)
            fclose (m_sourcefile);
        m_last_sourcefile = filename;
        m_sourcefile = fopen (filename.c_str(), "r");
        if (! m_sourcefile) {
            m_last_sourcefile = ustring();
            return "<not found>";
        }
    }

    // If we want something *before* the last line read in the open file,
    // rewind to the beginning.
    if (m_last_sourceline > line) {
        rewind (m_sourcefile);
        m_last_sourceline = 0;
    }

    // Now read lines up to and including the file we want.
    char buf[10240];
    while (m_last_sourceline < line) {
        fgets (buf, sizeof(buf), m_sourcefile);
        ++m_last_sourceline;
    }

    // strip trailing newline
    if (buf[strlen(buf)-1] == '\n')
        buf[strlen(buf)-1] = '\0';

    return std::string (buf);
}



void
OSLCompilerImpl::push_nesting (bool isloop)
{
    ++m_total_nesting;
    if (isloop)
        ++m_loop_nesting;
    if (current_function())
        current_function()->push_nesting (isloop);
}



void
OSLCompilerImpl::pop_nesting (bool isloop)
{
    --m_total_nesting;
    if (isloop)
        --m_loop_nesting;
    if (current_function())
        current_function()->pop_nesting (isloop);
}



const char *
OSLCompilerImpl::type_c_str (const TypeSpec &type) const
{
    if (type.is_structure())
        return ustring::format ("struct %s", type.structspec()->name().c_str()).c_str();
    else
        return type.c_str();
}



void
OSLCompilerImpl::struct_field_pair (Symbol *sym1, Symbol *sym2, int fieldnum,
                                    Symbol * &field1, Symbol * &field2)
{
    ASSERT (sym1 && sym2 && sym1->typespec().is_structure() &&
            sym1->typespec().structure() && sym2->typespec().structure());
    // Find the StructSpec for the type of struct that the symbols are
    StructSpec *structspec (sym1->typespec().structspec());
    ASSERT (structspec && fieldnum < (int)structspec->numfields());
    // Find the FieldSpec for the field we are interested in
    const StructSpec::FieldSpec &field (structspec->field(fieldnum));
    // Construct mangled names that describe the symbols for the
    // individual fields
    ustring name1 = ustring::format ("%s.%s", sym1->mangled().c_str(),
                                     field.name.c_str());
    ustring name2 = ustring::format ("%s.%s", sym2->mangled().c_str(),
                                     field.name.c_str());
    // Retrieve the symbols
    field1 = symtab().find_exact (name1);
    field2 = symtab().find_exact (name2);
    ASSERT (field1 && field2);
}



/// Called after code is generated, this function loops over all the ops
/// and figures out the lifetimes of all variables, based on whether the
/// args in each op are read or written.
void
OSLCompilerImpl::track_variable_usage ()
{
    // Clear the lifetimes for all symbols
    BOOST_FOREACH (Symbol *s, symtab())
        s->clear_rw ();

    // For each op, mark its arguments as being used at that op
    int opnum = 0;
    BOOST_FOREACH (Opcode &op, m_ircode) {
        for (int a = 0;  a < op.nargs();  ++a) {
            Symbol *s = m_opargs[op.firstarg()+a];
            s = s->dealias();
            s->mark_rw (opnum, op.argread(a), op.argwrite(a));
        }
        ++opnum;
    }
}



}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
