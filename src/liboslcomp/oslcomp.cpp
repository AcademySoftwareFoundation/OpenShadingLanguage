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
      m_verbose(false), m_debug(false), m_optimizelevel(1),
      m_next_temp(0), m_next_const(0),
      m_osofile(NULL), m_sourcefile(NULL), m_last_sourceline(0),
      m_total_nesting(0), m_loop_nesting(0), m_derivsym(NULL)
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
    delete m_derivsym;
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
        } else if (options[i] == "-O0") {
            m_optimizelevel = 0;
        } else if (options[i] == "-O" || options[i] == "-O1") {
            m_optimizelevel = 1;
        } else if (options[i] == "-O2") {
            m_optimizelevel = 2;
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
            shader()->typecheck ();
        }

        // Print the parse tree if there were no errors
        if (m_debug) {
            symtab().print ();
            if (shader())
                shader()->print (std::cout);
        }

        if (! error_encountered()) {
            shader()->codegen ();
            track_variable_dependencies ();
            track_variable_usage ();
            if (m_optimizelevel >= 1)
                coalesce_temporaries ();
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
OSLCompilerImpl::write_oso_symbol (const Symbol *sym)
{
    // symtype / datatype / name
    oso ("%s\t%s\t%s", sym->symtype_shortname(),
         type_c_str(sym->typespec()), sym->mangled().c_str());

    ASTvariable_declaration *v = NULL;
    if (sym->node())
        v = dynamic_cast<ASTvariable_declaration *>(sym->node());

    // Print default values
    bool isparam = (sym->symtype() == SymTypeParam ||
                    sym->symtype() == SymTypeOutputParam);
    if (sym->symtype() == SymTypeConst) {
        oso ("\t");
        write_oso_const_value (dynamic_cast<const ConstantSymbol *>(sym));
        oso ("\t");
    } else if (v && isparam) {
        std::string out;
        v->param_default_literals (sym, out);
        oso ("\t%s\t", out.c_str());
    }

    //
    // Now output all the hints, which is most of the work!
    //

    int hints = 0;

    // %meta{} encodes metadata (handled by write_oso_metadata)
    if (v) {
        ASSERT (v);
        for (ASTNode::ref m = v->meta();  m;  m = m->next()) {
            if (hints++ == 0)
                oso ("\t");
            write_oso_metadata (m.get());
        }
    }

    // %read and %write give the range of ops over which a symbol is used.
    if (hints++ == 0)
        oso ("\t");
    oso (" %%read{%d,%d} %%write{%d,%d}", sym->firstread(), sym->lastread(),
         sym->firstwrite(), sym->lastwrite());

    // %struct, %structfields, and %structfieldtypes document the
    // definition of a structure and which other symbols comprise the
    // individual fields.
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
    // %mystruct and %mystructfield document the symbols holding structure
    // fields, linking them back to the structures they are part of.
    if (sym->fieldid() >= 0) {
        if (hints++ == 0)
            oso ("\t");
        ASTvariable_declaration *vd = (ASTvariable_declaration *) sym->node();
        oso (" %%mystruct{%s} %%mystructfield{%d}",
             vd->sym()->mangled().c_str(), sym->fieldid());
    }

    // %derivs hint marks symbols that need to carry derivatives
    if (sym->has_derivs()) {
        if (hints++ == 0)
            oso ("\t");
        oso (" %%derivs");
    }

    // %depends marks, for potential OUTPUTs, which symbols they depend
    // upon.  This is so that derivativeness, etc., may be
    // back-propagated as shader networks are linked together.
    if (isparam || sym->symtype() == SymTypeGlobal) {
        // FIXME
        const SymPtrSet &deps (m_symdeps[sym]);
        std::vector<const Symbol *> inputdeps;
        BOOST_FOREACH (const Symbol *d, deps)
            if (d->symtype() == SymTypeParam ||
                  d->symtype() == SymTypeOutputParam ||
                  d->symtype() == SymTypeGlobal ||
                  d->symtype() == SymTypeLocal ||
                  d->symtype() == SymTypeTemp)
                inputdeps.push_back (d);
        if (inputdeps.size()) {
            if (hints++ == 0)
                oso ("\t");
            oso (" %%depends{");
            for (size_t i = 0;  i < inputdeps.size();  ++i) {
                if (i)
                    oso (",");
                oso ("%s",  inputdeps[i]->mangled().c_str());
            }
            oso ("}");
        }
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

    // FIXME -- output global hints and metadata

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

    // Output all opcodes
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

        //
        // Opcode Hints
        //

        bool firsthint = true;

        // %filename and %line document the source code file and line that
        // contained code that generated this op.  To avoid clutter, we
        // only output these hints when they DIFFER from the previous op.
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

        // %argrw documents which arguments are read, written, or both (rwW).
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

        // %argderivs documents which arguments have derivs taken of
        // them by the op.
        if (op->argtakesderivs_all()) {
            oso (" %%argderivs{");
            bool any = false;;
            for (int i = 0;  i < op->nargs();  ++i)
                if (op->argtakesderivs(i)) {
                    if (any++)
                        oso (",");
                    oso ("%d", i);
                }
            oso ("}");
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



/// Verify that the given symbol (written by the given op) is legal to
/// be written.
void
OSLCompilerImpl::check_write_legality (const Opcode &op, int opnum,
                                       const Symbol *sym)
{
    // We can never write to constant symbols
    if (sym->symtype() == SymTypeConst) {
        error (op.sourcefile(), op.sourceline(),
               "Attempted to write to a constant value");
    }

    // Params can only write if it's part of their initialization
    if (sym->symtype() == SymTypeParam && 
        (opnum < sym->initbegin() || opnum >= sym->initend())) {
        error (op.sourcefile(), op.sourceline(),
               "Cannot write to input parameter '%s'",
               sym->name().c_str());
    }

    // FIXME -- check for writing to globals.  But it's tricky, depends on
    // what kind of shader we are.
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
        // Some work to do for each argument to the op...
        for (int a = 0;  a < op.nargs();  ++a) {
            SymbolPtr &s = m_opargs[op.firstarg()+a];
            s = s->dealias();   // Make sure it's de-aliased

            // Mark that it's read and/or written for this op
            s->mark_rw (opnum, op.argread(a), op.argwrite(a));

            // Here's a good place to check that we aren't writing to
            // places we shouldn't.
            if (op.argwrite(a))
                check_write_legality (op, opnum, s);
        }

        // If this is a loop op, we need to mark its control variable
        // (the only arg) as used for the duration of the loop!
        if (op.opname() == "for" || op.opname() == "while" ||
                op.opname() == "dowhile") {
            ASSERT (op.nargs() == 1);  // loops should have just one arg
            Symbol * &s = m_opargs[op.firstarg()];
            s->mark_rw (opnum+1, true, true);
            s->mark_rw (op.farthest_jump()-1, true, true);
        }

        ++opnum;
    }

    // Special case: temporaries referenced both inside AND outside a
    // loop need their lifetimes extended to cover the entire loop so
    // they aren't accidentally coalesced incorrectly.  The specific
    // danger is for a function that contains a loop, and the function
    // is passed an argument that is a temporary calculation.
    opnum = 0;
    BOOST_FOREACH (Opcode &op, m_ircode) {
        if (op.opname() == "for" || op.opname() == "while" ||
                op.opname() == "dowhile") {
            int loopend = op.farthest_jump() - 1;
            BOOST_FOREACH (Symbol *s, symtab()) {
                if (s->symtype() == SymTypeTemp && 
                    ((s->firstuse() < opnum && s->lastuse() >= opnum) ||
                     (s->firstuse() < loopend && s->lastuse() >= loopend))) {
                    s->mark_rw (opnum, true, true);
                    s->mark_rw (loopend, true, true);
                }
            }
        }
        ++opnum;
    }
}



// Add to the dependency map that "A depends on B".
static void
add_dependency (SymDependencyMap &dmap, const Symbol *A, const Symbol *B)
{
    dmap[A].insert (B);
    // Perform unification -- all of B's dependencies are now
    // dependencies of A.
    BOOST_FOREACH (const Symbol *r, dmap[B])
        dmap[A].insert (r);
}



/// Run through all the ops, for each one marking its 'written'
/// arguments as dependent upon its 'read' arguments (and performing
/// unification as we go), yielding a dependency map that lets us look
/// up any symbol and see the set of other symbols on which it ever
/// depends on during execution of the shader.
void
OSLCompilerImpl::track_variable_dependencies ()
{
    // It's important to note that this is simplistically conservative
    // in that it overestimates dependencies.  To see why this is the
    // case, consider the following code:
    //       // inputs a,b; outputs x,y; local variable t
    //       t = a;
    //       x = t;
    //       t = b;
    //       y = t;
    // We can see that x depends on a and y depends on b.  But the
    // dependency analysis we do below thinks that y also depends on a
    // (because t depended on both a and b, but at different times).
    //
    // This naivite will never miss a dependency, but it may
    // overestimate dependencies.  (Hence we call this "conservative"
    // rather than "wrong.")  We deem this acceptable for now, since
    // it's so much easer to implement the conservative dependency
    // analysis, and it's not yet clear that getting it closer to
    // optimal will have any performance impact on final shaders. Also
    // because this is probably no worse than the "dependency slop" that
    // would happen with loops and conditionals.  But we certainly may
    // revisit with a more sophisticated algorithm if this crops up
    // a legitimate issue.
    //
    // Because of this conservative approach, it is critical that this
    // analysis is done BEFORE temporaries are coalesced (which would
    // cause them to be reassigned in exactly the way that confuses this
    // analysis).

    m_symdeps.clear ();
    std::vector<Symbol *> read, written;
    int opnum = 0;
    // We define a pseudo-symbol just for tracking derivatives.  This
    // symbol "depends on" whatever things have derivs taken of them.
    if (! m_derivsym)
        m_derivsym = new Symbol (ustring("$derivs"), TypeSpec(), SymTypeGlobal);
    // Loop over all ops...
    for (OpcodeVec::const_iterator op = m_ircode.begin();
           op != m_ircode.end(); ++op, ++opnum) {
        // Gather the list of syms read and written by the op.  Reuse the
        // vectors defined outside the loop to cut down on malloc/free.
        read.clear ();
        written.clear ();
        syms_used_in_op_range (op, op+1, &read, &written);

        // FIXME -- special cases here!  like if any ops implicitly read
        // or write to globals without them needing to be arguments.

        bool deriv = op->argtakesderivs_all();
        // For each sym written by the op...
        BOOST_FOREACH (const Symbol *wsym, written) {
            // For each sym read by the op...
            BOOST_FOREACH (const Symbol *rsym, read) {
                if (rsym->symtype() != SymTypeConst)
                    add_dependency (m_symdeps, wsym, rsym);
            }
            if (deriv) {
                // If the op takes derivs, make the pseudo-symbol m_derivsym
                // depend on those arguments.
                for (int a = 0;  a < op->nargs();  ++a)
                    if (op->argtakesderivs(a))
                        add_dependency (m_symdeps, m_derivsym, 
                                        m_opargs[a+op->firstarg()]);
            }
        }
    }

    // Mark all symbols needing derivatives as such
    BOOST_FOREACH (const Symbol *d, m_symdeps[m_derivsym])
        const_cast<Symbol *>(d)->has_derivs (true);

#if 0
    // Helpful for debugging

    std::cerr << "track_variable_dependencies\n";
    std::cerr << "\nDependencies:\n";
    BOOST_FOREACH (SymDependencyMap::value_type &m, m_symdeps) {
        std::cerr << m.first->mangled() << " depends on ";
        BOOST_FOREACH (const Symbol *d, m.second)
            std::cerr << d->mangled() << ' ';
        std::cerr << "\n";
    }
    std::cerr << "\n\n";

    // Invert the dependency
    SymDependencyMap influences;
    BOOST_FOREACH (SymDependencyMap::value_type &m, m_symdeps)
        BOOST_FOREACH (const Symbol *d, m.second)
            influences[d].insert (m.first);

    std::cerr << "\nReverse dependencies:\n";
    BOOST_FOREACH (SymDependencyMap::value_type &m, influences) {
        std::cerr << m.first->mangled() << " contributes to ";
        BOOST_FOREACH (const Symbol *d, m.second)
            std::cerr << d->mangled() << ' ';
        std::cerr << "\n";
    }
    std::cerr << "\n\n";
#endif
}



// Is the symbol coalescable?
inline bool
coalescable (const Symbol *s)
{
    return (s->symtype() == SymTypeTemp &&     // only coalesce temporaries
            s->everused() &&                   // only if they're used
            s->dealias() == s &&               // only if not already aliased
            ! s->typespec().is_structure() &&  // only if not a struct
            s->fieldid() < 0);                 //    or a struct field
}



/// Coalesce temporaries.  During code generation, we make a new
/// temporary EVERY time we need one.  Now we examine them all and merge
/// ones of identical type and non-overlapping lifetimes.
void
OSLCompilerImpl::coalesce_temporaries ()
{
    // We keep looping until we can't coalesce any more.
    int ncoalesced = 1;
    while (ncoalesced) {
        ncoalesced = 0;   // assume we're done, unless we coalesce something

        // We use a greedy algorithm that loops over each symbol, and
        // then examines all higher-numbered symbols (in order) and
        // tries to merge the first one it can find that doesn't overlap
        // lifetimes.  The temps were created as we generated code, so
        // they are already sorted by their "first use".  Thus, for any
        // pair t1 and t2 that are merged, it is guaranteed that t2 is
        // the symbol whose first use the earliest of all symbols whose
        // lifetimes do not overlap t1.

        SymbolPtrVec::iterator s;
        for (s = m_symtab.begin(); s != m_symtab.end();  ++s) {
            // Skip syms that can't be (or don't need to be) coalesced
            if (! coalescable(*s))
                continue;

            int sfirst = (*s)->firstuse ();
            int slast  = (*s)->lastuse ();

            // Loop through every other symbol
            for (SymbolPtrVec::iterator t = s+1;  t != m_symtab.end();  ++t) {
                // Coalesce s and t if both syms are coalescable,
                // equivalent types, and have nonoverlapping lifetimes.
                if (coalescable (*t) &&
                      equivalent ((*s)->typespec(), (*t)->typespec()) &&
                      (slast < (*t)->firstuse() || sfirst > (*t)->lastuse())) {
                    // Make all future t references alias to s
                    (*t)->alias (*s);
                    // s gets union of the lifetimes
                    (*s)->union_rw ((*t)->firstread(), (*t)->lastread(),
                                    (*t)->firstwrite(), (*t)->lastwrite());
                    sfirst = (*s)->firstuse ();
                    slast  = (*s)->lastuse ();
                    // t gets marked as unused
                    (*t)->clear_rw ();
                    ++ncoalesced;
                }
            }
        }
        // std::cerr << "Coalesced " << ncoalesced << "\n";
    }
}



bool
OSLCompilerImpl::op_uses_sym (const Opcode &op, const Symbol *sym,
                              bool read, bool write)
{
    // Loop through all the op's arguments, see if one matches sym
    for (int i = 0;  i < op.nargs();  ++i)
        if (m_opargs[i+op.firstarg()] == sym &&
            (read && op.argread(i)) || (write && op.argwrite(i)))
            return true;
    return false;
}



void
OSLCompilerImpl::syms_used_in_op_range (OpcodeVec::const_iterator opbegin,
                                        OpcodeVec::const_iterator opend,
                                        std::vector<Symbol *> *rsyms,
                                        std::vector<Symbol *> *wsyms)
{
    for (OpcodeVec::const_iterator op = opbegin; op != opend;  ++op) {
        for (int i = 0;  i < op->nargs();  ++i) {
            Symbol *s = m_opargs[i+op->firstarg()];
            if (rsyms && op->argread(i))
                if (std::find (rsyms->begin(), rsyms->end(), s) == rsyms->end())
                    rsyms->push_back (s);
            if (wsyms && op->argwrite(i))
                if (std::find (wsyms->begin(), wsyms->end(), s) == wsyms->end())
                    wsyms->push_back (s);
        }
    }
}



}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
