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
#include "simplerend.h"
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
    SimpleRenderer rend;
    shadingsys = ShadingSystem::create (&rend);

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
    float time = 0.0f;
    shaderglobals.time.init (&time, 0);
    
    // Make a shader space that is translated one unit in x and rotated
    // 45deg about the z axis.
    OSL::Matrix44 Mshad;
    Mshad.translate (OSL::Vec3 (1.0, 0.0, 0.0));
    Mshad.rotate (OSL::Vec3 (0.0, 0.0, M_PI_4));
    // std::cout << "shader-to-common matrix: " << Mshad << "\n";
    OSL::TransformationPtr Mshadptr (&Mshad);
    shaderglobals.shader2common.init ((OSL::TransformationPtr *)&Mshadptr, 0);

    // Make an object space that is translated one unit in y and rotated
    // 90deg about the z axis.
    OSL::Matrix44 Mobj;
    Mobj.translate (OSL::Vec3 (0.0, 1.0, 0.0));
    Mobj.rotate (OSL::Vec3 (0.0, 0.0, M_PI_2));
    // std::cout << "object-to-common matrix: " << Mobj << "\n";
    OSL::TransformationPtr Mobjptr (&Mobj);
    shaderglobals.object2common.init ((OSL::TransformationPtr *)&Mobjptr, 0);

    // Make a 'myspace that is non-uniformly scaled
    OSL::Matrix44 Mmyspace;
    Mmyspace.scale (OSL::Vec3 (1.0, 2.0, 1.0));
    // std::cout << "myspace-to-common matrix: " << Mmyspace << "\n";
    rend.name_transform ("myspace", Mmyspace);

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
