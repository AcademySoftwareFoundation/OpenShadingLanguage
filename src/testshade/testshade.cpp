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

#include <OpenImageIO/argparse.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "oslexec.h"
#include "../liboslexec/oslexec_pvt.h"
using namespace OSL;
using namespace OSL::pvt;




static ShadingSystem *shadingsys = NULL;
static std::vector<std::string> inputfiles;
static std::string outputfile ("out.exr");




static int
parse_files (int argc, const char *argv[])
{
    for (int i = 0;  i < argc;  i++)
        inputfiles.push_back (argv[i]);
    return 0;
}



static int
getargs (int argc, const char *argv[])
{
#if 1
    static bool help = false;
    ArgParse ap;
    ap.options ("Usage:  testshade [options] shader...",
                "%*", parse_files, "",
                "--help", &help, "Print help message",
                "-o %s", &outputfile, "Output filename",
//                "-v", &verbose, "Verbose output",
//                "-m %s", &metamatch, "Metadata names to print (default: all)",
//                "-f", &filenameprefix, "Prefix each line with the filename",
//                "-s", &sum, "Sum the image sizes",
                NULL);
    if (ap.parse(argc, argv) < 0 || inputfiles.empty()) {
        std::cerr << ap.error_message() << std::endl;
        ap.usage ();
        exit (EXIT_FAILURE);
    }
    if (help) {
        std::cout <<
            "testshade -- Test Open Shading Language\n"
            "(c) Copyright 2009 Sony Pictures Imageworks. All Rights Reserved.\n";
        ap.usage ();
        exit (EXIT_SUCCESS);
    }
#else
// Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Print help message")
        ("verbose,v", "Verbose output")
        ("sum,s", "Sum the image sizes")
        ("filename-prefix,f", "Prefix each line with the filename")
        ("output-file,o", po::value<std::string>(), "Output file")
//        ("compression", po::value<int>(), "set compression level")
        ("input-file", po::value< std::vector<std::string> >(), "Input file")

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

#if 0
        std::cout << "Verbose: " << vm.count("verbose") << "\n";
        std::cout << "filenameprefix: " << vm.count("filename-prefix") << "\n";
        std::cout << "Sum: " << vm.count("sum") << "\n";
#endif

        if (vm.count("output-file")) {
            outputfile = vm["output-file"].as<std::string>();
            std::cout << "output file " << outputfile << "\n";
        }

        if (vm.count("compression")) {
            std::cout << "Compression level was set to " 
                      << vm["compression"].as<int>() << ".\n";
        }

        if (vm.count("input-file"))
            inputfiles = vm["input-file"].as<std::vector<std::string> >();
    }
    catch (std::exception& e) {
        std::cout <<
            "testshade -- Test Open Shading Language\n"
            "(c) Copyright 2009 Sony Pictures Imageworks. All Rights Reserved.\n";
        std::cout << "ERROR: " << e.what() << "\n";
        std::cout << desc << "\n";
        exit (EXIT_FAILURE);
    }
#endif
}



static void
test_shader (const std::string &filename)
{
    ShaderMaster::ref m = 
        ((ShadingSystemImpl *)shadingsys)->loadshader (filename.c_str());
    if (m)
        m->print ();
    else
        std::cerr << "ERR: " << shadingsys->geterror() << "\n";
    std::cout << "\n";

    float Kd = 0.75;
    shadingsys->Parameter ("Kd", TypeDesc::TypeFloat, &Kd);
    shadingsys->Shader ("surface", filename.c_str());
    ShadingAttribStateRef shaderstate = shadingsys->state ();

    ShaderGlobals shaderglobals;
    const int npoints = 1;
    Imath::V3f gP[npoints];
    shaderglobals.P.init (gP);

    ShadingSystemImpl *ssi = (ShadingSystemImpl *)shadingsys;
    shared_ptr<ShadingContext> ctx = ssi->get_context ();
    ctx->bind (npoints, *shaderstate, shaderglobals);
    ctx->execute (ShadUseSurface);
    std::cerr << "\n";
}



int
main (int argc, const char *argv[])
{
    getargs (argc, argv);

    shadingsys = ShadingSystem::create ();
    shadingsys->attribute ("statistics:level", 5);

    for (size_t i = 0;  i < inputfiles.size();  ++i) {
        test_shader (inputfiles[i]);
    }

    ShadingSystem::destroy (shadingsys);

    return EXIT_SUCCESS;
}
