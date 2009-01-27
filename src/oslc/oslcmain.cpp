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


#include <iostream>
#include <string>
#include <vector>

#include <boost/scoped_ptr.hpp>

#include "oslcomp.h"
using namespace OSL;



static void
usage ()
{
    std::cout <<
        "oslc -- Open Shading Language compiler\n"
        "(c) Copyright 2009 Sony Pictures Imageworks. All Rights Reserved.\n"
        "Usage:  oslc [options] file\n"
        "  Options:\n"
        "\t--help         Print this usage message\n"
        "\t-v             Verbose mode\n"
        "\t-Ipath         Add path to the #include search path\n"
        "\t-Dsym[=val]    Define preprocessor symbol\n"
        "\t-Usym          Undefine preprocessor symbol\n"
        ;
}



int
main (int argc, const char *argv[])
{
    std::vector <std::string> args;

    for (int a = 1;  a < argc;  ++a) {
        if (! strcmp (argv[a], "-v")) {
            // Valid command-line argument
            args.push_back (argv[a]);
        }
        else if (! strcmp (argv[a], "--help") | ! strcmp (argv[a], "-a")) {
            usage ();
            return EXIT_SUCCESS;
        }
        else {
            boost::scoped_ptr<OSLCompiler> compiler (OSLCompiler::create ());
            bool ok = compiler->compile (argv[a], args);
            if (ok)
                std::cout << "Compiled " << argv[a] << " -> " 
                          << " FIXME\n";
            else {
                std::cout << "FAILED " << argv[a] << "\n";
                return EXIT_FAILURE;
            }
        }
    }


    return EXIT_SUCCESS;
}
