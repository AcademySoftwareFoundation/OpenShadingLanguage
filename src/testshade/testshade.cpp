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
#include <cmath>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/argparse.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/timer.h>

#include "oslexec.h"
#include "../liboslexec/oslexec_pvt.h"
using namespace OSL;
using namespace OSL::pvt;




static ShadingSystem *shadingsys = NULL;
static std::vector<std::string> shadernames;
static std::vector<std::string> outputfiles;
static std::vector<std::string> outputvars;
static bool debug = false;
static int xres = 1, yres = 1;



static int
add_shader (int argc, const char *argv[])
{
    shadingsys->attribute ("debug", (int)debug);
    for (int i = 0;  i < argc;  i++) {
        shadernames.push_back (argv[i]);
        shadingsys->Shader ("surface", argv[i]);
    }
    return 0;
}



static int
getargs (int argc, const char *argv[])
{
    static bool help = false;
    ArgParse ap;
    ap.options ("Usage:  testshade [options] shader...",
                "%*", add_shader, "",
                "--help", &help, "Print help message",
                "--debug", &debug, "Lots of debugging info",
                "-g %d %d", &xres, &yres, "Make an X x Y grid of shading points",
                "-o %L %L", &outputvars, &outputfiles,
                        "Output (variable, filename)",
//                "-v", &verbose, "Verbose output",
                NULL);
    if (ap.parse(argc, argv) < 0 || shadernames.empty()) {
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
    // Create a new shading system.
    Timer timer;
    shadingsys = ShadingSystem::create ();

    getargs (argc, argv);
    // getargs called 'add_shader' for each shader mentioned on the command
    // line.  So now we should have a valid shading state.
    ShadingAttribStateRef shaderstate = shadingsys->state ();

    // Set up shader globals and a little test grid of points to shade.
    ShaderGlobals shaderglobals;
    const int npoints = xres*yres;
    std::vector<Vec3> gP (npoints);
    std::vector<float> gu (npoints);
    std::vector<float> gv (npoints);
    shaderglobals.P.init (&gP[0], sizeof(gP[0]));
    shaderglobals.u.init (&gu[0], sizeof(gu[0]));
    shaderglobals.v.init (&gv[0], sizeof(gv[0]));
    for (int j = 0;  j < yres;  ++j) {
        for (int i = 0;  i < xres;  ++i) {
            int n = j*yres + i;
            gu[n] = (xres == 1) ? 0.5 : (float)i/(xres-1);
            gv[n] = (yres == 1) ? 0.5 : (float)j/(yres-1);
            gP[n] = Vec3 (gu[n], gv[n], 1.0f);
        }
    }
    double setuptime = timer ();
    timer.reset ();
    timer.start ();

    // Request a shading context, bind it, execute the shaders.
    // FIXME -- this will eventually be replaced with a public
    // ShadingSystem call that encapsulates it.
    ShadingSystemImpl *ssi = (ShadingSystemImpl *)shadingsys;
    shared_ptr<ShadingContext> ctx = ssi->get_context ();
    ctx->bind (npoints, *shaderstate, shaderglobals);
    double bindtime = timer ();
    timer.reset ();
    timer.start ();
    ctx->execute (ShadUseSurface);
    bool runtime = timer ();
    std::cout << "\n";

    std::vector<float> pixel;
    for (size_t i = 0;  i < outputfiles.size();  ++i) {
        Symbol *sym = ctx->symbol (ShadUseSurface, ustring(outputvars[i]));
        if (! sym) {
            std::cerr << "Output " << outputvars[i] << " not found, skipping.\n";
            continue;
        }
        std::cout << "Output " << outputvars[i] << " to " 
                  << outputfiles[i]<< "\n";
        TypeDesc t = sym->typespec().simpletype();
        TypeDesc tbase = TypeDesc ((TypeDesc::BASETYPE)t.basetype);
        int nchans = t.numelements() * t.aggregate;
        pixel.resize (nchans);
        OpenImageIO::ImageSpec spec (xres, yres, nchans, tbase);
        OpenImageIO::ImageBuf img (outputfiles[i], spec);
        img.zero ();
        for (int y = 0, n = 0;  y < yres;  ++y) {
            for (int x = 0;  x < xres;  ++x, ++n) {
                OpenImageIO::convert_types (tbase,
                                            (char *)sym->data() + n*sym->step(),
                                            TypeDesc::FLOAT, &pixel[0], nchans);
                img.setpixel (x, y, &pixel[0]);
            }
        }
        img.save ();
    }

    if (debug) {
        std::cout << "\n";
        std::cout << "Setup: " << Strutil::timeintervalformat (setuptime,2) << "\n";
        std::cout << "Bind : " << Strutil::timeintervalformat (bindtime,2) << "\n";
        std::cout << "Run  : " << Strutil::timeintervalformat (runtime,2) << "\n";
        std::cout << "\n";
        std::cout << shadingsys->getstats (5) << "\n";
    }

    ShadingSystem::destroy (shadingsys);
    return EXIT_SUCCESS;
}
