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
static std::vector<std::string> outputfiles;
static std::vector<std::string> outputvars;
static bool debug = false;



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
    static bool help = false;
    ArgParse ap;
    ap.options ("Usage:  testshade [options] shader...",
                "%*", parse_files, "",
                "--help", &help, "Print help message",
                "--debug", &debug, "Lots of debugging info",
                "-o %L %L", &outputfiles, &outputvars, 
                        "Output (filename, variable)",
//                "-v", &verbose, "Verbose output",
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
}



int
main (int argc, const char *argv[])
{
    getargs (argc, argv);

    shadingsys = ShadingSystem::create ();
    shadingsys->attribute ("statistics:level", 5);
    shadingsys->attribute ("debug", (int)debug);

    for (size_t i = 0;  i < inputfiles.size();  ++i) {
        ShaderMaster::ref m = 
            ((ShadingSystemImpl *)shadingsys)->loadshader (inputfiles[i].c_str());
        if (! m)
            std::cerr << "ERR: " << shadingsys->geterror() << "\n";
        std::cout << "\n";

        float Kd = 0.75;
        shadingsys->Parameter ("Kd", TypeDesc::TypeFloat, &Kd);
        shadingsys->Shader ("surface", inputfiles[i].c_str());
    }

    ShadingAttribStateRef shaderstate = shadingsys->state ();

    // Set up shader globals
    ShaderGlobals shaderglobals;
    const int npoints = 1;
    Imath::V3f gP[npoints];
    shaderglobals.P.init (gP);

    ShadingSystemImpl *ssi = (ShadingSystemImpl *)shadingsys;
    shared_ptr<ShadingContext> ctx = ssi->get_context ();
    ctx->bind (npoints, *shaderstate, shaderglobals);
    ctx->execute (ShadUseSurface);
    std::cerr << "\n";


    ShadingSystem::destroy (shadingsys);

    return EXIT_SUCCESS;
}
