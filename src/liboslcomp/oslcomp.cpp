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
#ifdef __GNUC__
# include <ext/stdio_filebuf.h>
#endif
#include <cstdio>
#include <cerrno>

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"

#include "oslcomp_pvt.h"


#define yyFlexLexer oslFlexLexer
#include "FlexLexer.h"


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
      m_verbose(false), m_debug(false), m_next_temp(0), m_osofile(NULL),
      m_sourcefile(NULL), m_last_sourceline(0)
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
    std::string cppcommand = "/usr/bin/cpp -xc -nostdinc ";

    for (size_t i = 0;  i < options.size();  ++i) {
        if (options[i] == "-v") {
            // verbose mode
            m_verbose = true;
        } else if (options[i] == "-d") {
            // debug mode
            m_debug = true;
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

        if (! error_encountered()) {
            oslcompiler->shader()->typecheck ();
        }

        // Print the parse tree if there were no errors
        if (m_debug) {
            oslcompiler->symtab().print ();
//            if (! parseerr)
                oslcompiler->shader()->print ();
        }

        if (! error_encountered()) {
            oslcompiler->shader()->codegen ();
        }
 
        if (! error_encountered()) {
            std::string outname = output_filename (filename);
            write_oso_file (outname);
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
    { "L", TypeDesc::TypeVector },
    { "Cl", TypeDesc::TypeColor },
    { "Ps", TypeDesc::TypePoint },
    { "Ns", TypeDesc::TypeNormal },
    { "Pl", TypeDesc::TypePoint },
    { "Nl", TypeDesc::TypeNormal },
    { "Ci", TypeSpec (TypeDesc::TypeColor, true) },
    { "Oi", TypeDesc::TypeColor },
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
                                Symbol::SymTypeGlobal);
        symtab().insert (s);
    }
}



std::string
OSLCompilerImpl::output_filename (const std::string &inputfilename)
{
    if (m_shader && shader_decl())
        return shader_decl()->shadername().string() + ".oso";
    return std::string();
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

    // FIXME -- Output params

    // FIXME -- output all non-param symbols

    // FIXME -- output all opcodes
    int lastline = -1;
    ustring lastfile;
    oso ("code main\n");
    for (IROpcodeVec::iterator op = m_ircode.begin(); op != m_ircode.end();  ++op) {
        if (m_debug && op->node()) {
            ustring file = op->node()->sourcefile();
            int line = op->node()->sourceline();
            if (file != lastfile || line != lastline)
                oso ("# %s:%d\n# %s\n", file.c_str(), line,
                     retrieve_source (file, line).c_str());
        }

        oso ("\t%s", op->opname());
        for (size_t i = 0;  i < op->nargs();  ++i) {
            oso ("%c%s", (i ? ' ' : '\t'),
                 op->arg(i)->dealias()->mangled().c_str());
        }
        bool firsthint = true;
        if (op->node()) {
            if (op->node()->sourcefile() != lastfile) {
                lastfile = op->node()->sourcefile();
                oso ("%c%%filename{%s}", firsthint ? '\t' : ' ', lastfile.c_str());
                firsthint = false;
            }
            if (op->node()->sourceline() != lastline) {
                lastline = op->node()->sourceline();
                oso ("%c%%line{%d}", firsthint ? '\t' : ' ', lastline);
                firsthint = false;
            }
        }
        oso ("\n");
    }
    oso ("\tend\n");

    fclose (m_osofile);
    m_osofile = NULL;
}



void
OSLCompilerImpl::oso (const char *fmt, ...)
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


}; // namespace pvt
}; // namespace OSL
