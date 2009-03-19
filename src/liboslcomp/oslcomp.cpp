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

#include "OpenImageIO/strutil.h"

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


OSLCompilerImpl::OSLCompilerImpl (void)
    : m_lexer(NULL), m_err(false), m_symtab(*this),
      m_current_typespec(TypeDesc::UNKNOWN), m_current_output(false),
      m_verbose(false), m_debug(false)
{
    initialize_globals ();
    initialize_builtin_funcs ();
}



void
OSLCompilerImpl::error (ustring filename, int line, const char *format, ...)
{
    va_list ap;
    va_start (ap, format);
    std::string errmsg = format ? Strutil::vformat (format, ap) : "syntax error";
    fprintf (stderr, "%s:%d: error: %s\n", 
             filename.c_str(), line, errmsg.c_str());
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



}; // namespace pvt
}; // namespace OSL
