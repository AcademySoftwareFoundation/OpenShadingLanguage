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


bool
OSLCompilerImpl::compile (const std::string &filename,
                          const std::vector<std::string> &options)
{
    std::string cppcommand = "/usr/bin/cpp -xc -nostdinc ";

    for (size_t i = 0;  i < options.size();  ++i) {
        cppcommand += "\"";
        cppcommand += options[i];
        cppcommand += "\" ";
    }
    cppcommand += "\"";
    cppcommand += filename;
    cppcommand += "\" ";

    std::cout << "cpp command:\n>" << cppcommand << "<\n";

    FILE *cpppipe = popen (cppcommand.c_str(), "r");

#ifdef __GNUC__
    __gnu_cxx::stdio_filebuf<char> fb (cpppipe, std::ios::in);
#else
    std::filebuf fb (cpppipe);
#endif

    if (fb.is_open()) {
        std::istream in (&fb);
#if 0
        while (! in.eof()) {
            std::string s;
            in >> s;
            std::cout << "line: " << s << "\n";
        }
#endif
#if 1
        oslcompiler = this;
        m_lexer = new oslFlexLexer (&in);
        bool err = oslparse ();
        delete m_lexer;
        oslcompiler = NULL;
#endif
        fb.close ();
        pclose (cpppipe);
    }

    return false;
}


}; // namespace pvt
}; // namespace OSL
