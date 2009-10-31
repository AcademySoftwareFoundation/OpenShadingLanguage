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
#include "oslclosure.h"
#include "simplerend.h"
using namespace OSL;
using namespace OSL::pvt;




static ShadingSystem *shadingsys = NULL;
static std::vector<std::string> shadernames;
static std::vector<std::string> outputfiles;
static std::vector<std::string> outputvars;
static std::string dataformatname = "";
static bool debug = false;
static int xres = 1, yres = 1;
static std::string layername;
static std::vector<std::string> connections;
static std::vector<std::string> iparams, fparams, vparams, sparams;
static float fparamdata[1000];   // bet that's big enough
static int fparamindex = 0;
static int iparamdata[1000];
static int iparamindex = 0;
static ustring sparamdata[1000];
static int sparamindex = 0;
static ErrorHandler errhandler;



static void
inject_params ()
{
    for (size_t p = 0;  p < fparams.size();  p += 2) {
        fparamdata[fparamindex] = atof (fparams[p+1].c_str());
        shadingsys->Parameter (fparams[p].c_str(), TypeDesc::TypeFloat,
                               &fparamdata[fparamindex]);
        fparamindex += 1;
    }
    for (size_t p = 0;  p < iparams.size();  p += 2) {
        iparamdata[iparamindex] = atoi (iparams[p+1].c_str());
        shadingsys->Parameter (iparams[p].c_str(), TypeDesc::TypeInt,
                               &iparamdata[iparamindex]);
        iparamindex += 1;
    }
    for (size_t p = 0;  p < vparams.size();  p += 4) {
        fparamdata[fparamindex+0] = atof (vparams[p+1].c_str());
        fparamdata[fparamindex+1] = atof (vparams[p+2].c_str());
        fparamdata[fparamindex+2] = atof (vparams[p+3].c_str());
        shadingsys->Parameter (vparams[p].c_str(), TypeDesc::TypeVector,
                               &fparamdata[fparamindex]);
        fparamindex += 3;
    }
    for (size_t p = 0;  p < sparams.size();  p += 2) {
        sparamdata[sparamindex] = ustring (sparams[p+1]);
        shadingsys->Parameter (sparams[p].c_str(), TypeDesc::TypeString,
                               &sparamdata[sparamindex]);
        sparamindex += 1;
    }
}



static int
add_shader (int argc, const char *argv[])
{
    shadingsys->attribute ("debug", (int)debug);
    if (debug)
        errhandler.verbosity (ErrorHandler::VERBOSE);

    for (int i = 0;  i < argc;  i++) {
        inject_params ();

        shadernames.push_back (argv[i]);
        shadingsys->Shader ("surface", argv[i],
                            layername.length() ? layername.c_str() : NULL);

        layername.clear ();
        iparams.clear ();
        fparams.clear ();
        vparams.clear ();
        sparams.clear ();
    }
    return 0;
}



static void
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
                "-od %s", &dataformatname, "Set the output data format to one of:\n"
                        "\t\t\tuint8, half, float",
                "--layer %s", &layername, "Set next layer name",
                "--fparam %L %L",
                        &fparams, &fparams,
                        "Add a float param (args: name value)",
                "--iparam %L %L",
                        &iparams, &iparams,
                        "Add an integer param (args: name value)",
                "--vparam %L %L %L %L",
                        &vparams, &vparams, &vparams, &vparams,
                        "Add a vector or color param (args: name xval yval zval)",
                "--sparam %L %L",
                        &sparams, &sparams,
                        "Add a string param (args: name value)",
                "--connect %L %L %L %L",
                    &connections, &connections, &connections, &connections,
                    "Connect fromlayer fromoutput tolayer toinput",
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
    shadingsys = ShadingSystem::create (&rend, NULL, &errhandler);

    shadingsys->ShaderGroupBegin ();
    getargs (argc, argv);

    for (size_t i = 0;  i < connections.size();  i += 4) {
        if (i+3 < connections.size()) {
            std::cout << "Connect " 
                      << connections[i] << "." << connections[i+1]
                      << " to " << connections[i+2] << "." << connections[i+3]
                      << "\n";
            shadingsys->ConnectShaders (connections[i].c_str(),
                                        connections[i+1].c_str(),
                                        connections[i+2].c_str(),
                                        connections[i+3].c_str());
        }
    }

    shadingsys->ShaderGroupEnd ();

    // getargs called 'add_shader' for each shader mentioned on the command
    // line.  So now we should have a valid shading state.
    ShadingAttribStateRef shaderstate = shadingsys->state ();

    // Set up shader globals and a little test grid of points to shade.
    ShaderGlobals shaderglobals;
    const int npoints = xres*yres;
    std::vector<Vec3> gP (npoints);
    std::vector<Vec3> gP_dx (npoints);
    std::vector<Vec3> gP_dy (npoints);
    std::vector<Vec3> gN (npoints);
    std::vector<float> gu (npoints);
    std::vector<float> gv (npoints);
    shaderglobals.P.init (&gP[0], sizeof(gP[0]));
    shaderglobals.dPdx.init (&gP_dx[0], sizeof(gP_dx[0]));
    shaderglobals.dPdy.init (&gP_dy[0], sizeof(gP_dy[0]));
    shaderglobals.N.init (&gN[0], sizeof(gN[0]));
    shaderglobals.Ng.init (&gN[0], sizeof(gN[0]));  // Ng = N for now
    shaderglobals.u.init (&gu[0], sizeof(gu[0]));
    shaderglobals.v.init (&gv[0], sizeof(gv[0]));
    shaderglobals.v.init (&gv[0], sizeof(gv[0]));
    shaderglobals.flipHandedness = false;
    float time = 0.0f;
    shaderglobals.time.init (&time, 0);

    std::vector<ClosureColor> Ci (npoints);
    std::vector<ClosureColor *> Ci_ptr (npoints);
    for (int i = 0;  i < npoints;  ++i)
        Ci_ptr[i] = &Ci[i];
    shaderglobals.Ci.init (&Ci_ptr[0], sizeof(Ci_ptr[0]));

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

    float dudx = 1.0f / xres, dudy = 0;
    float dvdx = 0, dvdy = 1.0f / yres;
    shaderglobals.dudx.init (&dudx, 0);
    shaderglobals.dudy.init (&dudy, 0);
    shaderglobals.dvdx.init (&dvdx, 0);
    shaderglobals.dvdy.init (&dvdy, 0);

    for (int j = 0;  j < yres;  ++j) {
        for (int i = 0;  i < xres;  ++i) {
            int n = j*yres + i;
            gu[n] = (xres == 1) ? 0.5 : (float)i/(xres-1);
            gv[n] = (yres == 1) ? 0.5 : (float)j/(yres-1);
            gP[n] = Vec3 (gu[n], gv[n], 1.0f);
            gP_dx[n] = Vec3 (dudx, dudy, 0.0f);
            gP_dy[n] = Vec3 (dvdx, dvdy, 0.0f);
            gN[n] = Vec3 (0, 0, 1);
        }
    }

    double setuptime = timer ();
    timer.reset ();
    timer.start ();

    // Request a shading context, bind it, execute the shaders.
    // FIXME -- this will eventually be replaced with a public
    // ShadingSystem call that encapsulates it.
    ShadingSystemImpl *ssi = (ShadingSystemImpl *)shadingsys;
    ShadingContext *ctx = ssi->get_context ();
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
        TypeDesc outtypebase = tbase;
        if (dataformatname == "uint8")
            outtypebase = TypeDesc::UINT8;
        else if (dataformatname == "half")
            outtypebase = TypeDesc::HALF;
        else if (dataformatname == "float")
            outtypebase = TypeDesc::FLOAT;
        int nchans = t.numelements() * t.aggregate;
        pixel.resize (nchans);
        OpenImageIO::ImageSpec spec (xres, yres, nchans, outtypebase);
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
    ssi->release_context (ctx);

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
