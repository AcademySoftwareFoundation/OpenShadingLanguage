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
#include <fstream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "oslexec.h"
using namespace OSL;

#include "../liboslexec/oslexec_pvt.h"

#define yyFlexLexer osoFlexLexer
#include <FlexLexer.h>



static std::vector<std::string> inputfiles;



static void
getargs (int argc, char *argv[])
{
// Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Print help message")
        ("verbose,v", "Verbose output")
        ("sum,s", "Sum the image sizes")
        ("filename-prefix,f", "Prefix each line with the filename")
//        ("compression", po::value<int>(), "set compression level")
        ("input-file", po::value< std::vector<std::string> >(), "input file")

        ;
    
    po::positional_options_description p;
    p.add ("input-file", -1);

    try {
        po::variables_map vm;
        po::store(po::command_line_parser(argc,(char **)argv).
                  options(desc).positional(p).run(), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            std::cout <<
                "testshade -- Test Open Shading Language\n"
                "(c) Copyright 2009 Sony Pictures Imageworks. All Rights Reserved.\n";
            std::cout << desc << "\n";
            exit (EXIT_SUCCESS);
        }

        std::cout << "Verbose: " << vm.count("verbose") << "\n";
        std::cout << "filenameprefix: " << vm.count("filename-prefix") << "\n";
        std::cout << "Sum: " << vm.count("sum") << "\n";

        if (vm.count("compression")) {
            std::cout << "Compression level was set to " 
                      << vm["compression"].as<int>() << ".\n";
        } else {
            std::cout << "Compression level was not set.\n";
        }

        if (vm.count("input-file")) {
            inputfiles = vm["input-file"].as<std::vector<std::string> >();
            std::cout << "Input files are: " ;
            for (size_t i = 0;  i < inputfiles.size();  ++i) 
                std::cout << "\t" << inputfiles[i] << "\n";
        }
    }
    catch (std::exception& e) {
        std::cout <<
            "testshade -- Test Open Shading Language\n"
            "(c) Copyright 2009 Sony Pictures Imageworks. All Rights Reserved.\n";
        std::cout << "ERROR: " << e.what() << "\n";
        std::cout << desc << "\n";
        exit (EXIT_FAILURE);
    }
}



static void
test_shader (const std::string &filename)
{
    std::fstream input (filename.c_str(), std::ios::in);
    if (! input.is_open()) {
        std::cout << "File " << filename << " not found.\n";
        return;
    }

    ShadingSystem::osolexer = new osoFlexLexer (&input);
    assert (ShadingSystem::osolexer);
    bool ok = ! osoparse ();   // osoparse returns nonzero if error
    if (ok) {
        std::cout << "Correctly parse " << filename << "\n";
    } else {
        std::cout << "Failed parse of " << filename << "\n";
    }
    delete ShadingSystem::osolexer;
    ShadingSystem::osolexer = NULL;

    input.close ();
}



int
main (int argc, char *argv[])
{
    getargs (argc, argv);

    for (size_t i = 0;  i < inputfiles.size();  ++i) {
        test_shader (inputfiles[i]);
    }

    return EXIT_SUCCESS;
}
