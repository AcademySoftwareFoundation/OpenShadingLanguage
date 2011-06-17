/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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
#if OPENIMAGEIO_VERSION >= 900 /* 0.9.0 */
# include <OpenImageIO/imagebufalgo.h>
#endif
#include <OpenImageIO/argparse.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/timer.h>

#include "oslexec.h"
#include "../liboslexec/oslexec_pvt.h"
#include "oslclosure.h"
#include "simplerend.h"
using namespace OSL;
using namespace OSL::pvt;

#ifdef OIIO_NAMESPACE
using OIIO::ArgParse;
using OIIO::Timer;
#endif



static ShadingSystem *shadingsys = NULL;
static std::vector<std::string> shadernames;
static std::vector<std::string> outputfiles;
static std::vector<std::string> outputvars;
static std::vector<OIIO::ImageBuf*>   outputimgs;
static std::string dataformatname = "";
static bool debug = false;
static bool verbose = false;
static bool stats = false;
static bool O0 = false, O1 = true, O2 = false;
static bool pixelcenters = false;
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
static int iters = 1;
static std::string raytype = "camera";



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
    shadingsys->attribute ("optimize", O2 ? 2 : (O0 ? 0 : 1));
    shadingsys->attribute ("lockgeom", 1);

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
                "-v", &verbose, "Verbose messages",
                "--debug", &debug, "Lots of debugging info",
                "--stats", &stats, "Print run statistics",
                "-g %d %d", &xres, &yres, "Make an X x Y grid of shading points",
                "-o %L %L", &outputvars, &outputfiles,
                        "Output (variable, filename)",
                "-od %s", &dataformatname, "Set the output data format to one of:\n"
                        "\t\t\t\tuint8, half, float",
                "--layer %s", &layername, "Set next layer name",
                "--fparam %L %L",
                        &fparams, &fparams,
                        "Add a float param (args: name value)",
                "--iparam %L %L",
                        &iparams, &iparams,
                        "Add an integer param (args: name value)",
                "--vparam %L %L %L %L",
                        &vparams, &vparams, &vparams, &vparams,
                        "Add a vector or color param (args: name x y z)",
                "--sparam %L %L",
                        &sparams, &sparams,
                        "Add a string param (args: name value)",
                "--connect %L %L %L %L",
                    &connections, &connections, &connections, &connections,
                    "Connect fromlayer fromoutput tolayer toinput",
                "--raytype %s", &raytype, "Set the raytype",
                "--iters %d", &iters, "Number of iterations",
                "-O0", &O0, "Do no runtime shader optimization",
                "-O1", &O1, "Do a little runtime shader optimization",
                "-O2", &O2, "Do lots of runtime shader optimization",
                "--center", &pixelcenters, "Shade at output pixel 'centers' rather than corners",
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
            "(c) Copyright 2009-2010 Sony Pictures Imageworks Inc. All Rights Reserved.\n";
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
    shadingsys->attribute("lockgeom", 1);

    shadingsys->ShaderGroupBegin ();
    getargs (argc, argv);

    if (debug || verbose)
        errhandler.verbosity (ErrorHandler::VERBOSE);

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
    memset(&shaderglobals, 0, sizeof(ShaderGlobals));

    // Make a shader space that is translated one unit in x and rotated
    // 45deg about the z axis.
    OSL::Matrix44 Mshad;
    Mshad.translate (OSL::Vec3 (1.0, 0.0, 0.0));
    Mshad.rotate (OSL::Vec3 (0.0, 0.0, M_PI_4));
    // std::cout << "shader-to-common matrix: " << Mshad << "\n";
    OSL::TransformationPtr Mshadptr (&Mshad);
    shaderglobals.shader2common = Mshadptr;

    // Make an object space that is translated one unit in y and rotated
    // 90deg about the z axis.
    OSL::Matrix44 Mobj;
    Mobj.translate (OSL::Vec3 (0.0, 1.0, 0.0));
    Mobj.rotate (OSL::Vec3 (0.0, 0.0, M_PI_2));
    // std::cout << "object-to-common matrix: " << Mobj << "\n";
    OSL::TransformationPtr Mobjptr (&Mobj);
    shaderglobals.object2common = Mobjptr;

    // Make a 'myspace that is non-uniformly scaled
    OSL::Matrix44 Mmyspace;
    Mmyspace.scale (OSL::Vec3 (1.0, 2.0, 1.0));
    // std::cout << "myspace-to-common matrix: " << Mmyspace << "\n";
    rend.name_transform ("myspace", Mmyspace);

    if (pixelcenters) {
        shaderglobals.dudx = 1.0f / xres;
        shaderglobals.dvdy = 1.0f / yres;
    } else {
        shaderglobals.dudx = 1.0f / std::max (1, xres-1);
        shaderglobals.dvdy = 1.0f / std::max (1, yres-1);
    }

    shaderglobals.raytype = ((ShadingSystemImpl *)shadingsys)->raytype_bit (ustring(raytype));

    double setuptime = timer ();
    double runtime = 0;

    std::vector<float> pixel;

    if (outputfiles.size() != 0)
        std::cout << "\n";

    // grab this once since we will be shading several points
    ShadingSystemImpl *ssi = (ShadingSystemImpl *)shadingsys;
    void* thread_info = ssi->create_thread_info();
    for (int iter = 0;  iter < iters;  ++iter) {
        for (int y = 0, n = 0;  y < yres;  ++y) {
            for (int x = 0;  x < xres;  ++x, ++n) {
                if (pixelcenters) {
                    shaderglobals.u = (float)(x+0.5f) / xres;
                    shaderglobals.v = (float)(y+0.5f) / yres;
                } else {
                    shaderglobals.u = (xres == 1) ? 0.5f : (float) x / (xres - 1);
                    shaderglobals.v = (yres == 1) ? 0.5f : (float) y / (yres - 1);
                }
                shaderglobals.P = Vec3 (shaderglobals.u, shaderglobals.v, 1.0f);
                shaderglobals.dPdx = Vec3 (shaderglobals.dudx, shaderglobals.dudy, 0.0f);
                shaderglobals.dPdy = Vec3 (shaderglobals.dvdx, shaderglobals.dvdy, 0.0f);
                shaderglobals.N    = Vec3 (0, 0, 1);
                shaderglobals.Ng   = Vec3 (0, 0, 1);
                shaderglobals.dPdu = Vec3 (1.0f, 0.0f, 0.0f);
                shaderglobals.dPdv = Vec3 (0.0f, 1.0f, 0.0f);
                shaderglobals.surfacearea = 1;

                // Request a shading context, bind it, execute the shaders.
                // FIXME -- this will eventually be replaced with a public
                // ShadingSystem call that encapsulates it.
                ShadingContext *ctx = ssi->get_context (thread_info);
                timer.reset ();
                timer.start ();
                // run shader for this point
                ctx->execute (ShadUseSurface, *shaderstate, shaderglobals);
                runtime += timer ();

                if (iter == (iters - 1)) {
                   // extract any output vars into images (on last iteration only)
                   for (size_t i = 0;  i < outputfiles.size();  ++i) {
                       Symbol *sym = ctx->symbol (ShadUseSurface, ustring(outputvars[i]));
                       if (! sym) {
                           if (n == 0) {
                              std::cout << "Output " << outputvars[i] << " not found, skipping.\n";
                              outputimgs.push_back(0); // invalid image
                           }
                           continue;
                       }
                       if (n == 0)
                           std::cout << "Output " << outputvars[i] << " to " << outputfiles[i]<< "\n";
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
                       if (n == 0) {
                           OIIO::ImageSpec spec (xres, yres, nchans, outtypebase);
                           OIIO::ImageBuf* img = new OIIO::ImageBuf(outputfiles[i], spec);
#if OPENIMAGEIO_VERSION >= 900 /* 0.9.0 */
                           OIIO::ImageBufAlgo::zero (*img);
#else
                           img->zero ();
#endif
                           outputimgs.push_back(img);
                       }
                       OIIO::convert_types (tbase, ctx->symbol_data (*sym, 0),
                                                   TypeDesc::FLOAT, &pixel[0], nchans);
                       outputimgs[i]->setpixel (x, y, &pixel[0]);
                   }
                }
                ssi->release_context (ctx, thread_info);
            }
        }
    }
    ssi->destroy_thread_info(thread_info);

    if (outputfiles.size() == 0)
        std::cout << "\n";

    // write any images to disk
    for (size_t i = 0;  i < outputimgs.size();  ++i) {
        if (outputimgs[i]) {
           outputimgs[i]->save();
            delete outputimgs[i];
        }
    }
    if (debug || stats) {
        std::cout << "\n";
        std::cout << "Setup: " << Strutil::timeintervalformat (setuptime,2) << "\n";
        std::cout << "Run  : " << Strutil::timeintervalformat (runtime,2) << "\n";
        std::cout << "\n";
        std::cout << shadingsys->getstats (5) << "\n";
    }

    ShadingSystem::destroy (shadingsys);
    return EXIT_SUCCESS;
}
