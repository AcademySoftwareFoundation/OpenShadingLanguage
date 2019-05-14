/*
Copyright (c) 2009-2018 Sony Pictures Imageworks Inc., et al.
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


#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/argparse.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/timer.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/thread.h>
#include <OpenImageIO/parallel.h>
#include <OpenImageIO/sysutil.h>

#include <OSL/oslexec.h>
#include "optixraytracer.h"
#include "shading.h"
#include "simpleraytracer.h"


using namespace OSL;

namespace { // anonymous namespace

static ShadingSystem *shadingsys = NULL;
static bool debug1 = false;
static bool debug2 = false;
static bool verbose = false;
static bool runstats = false;
static bool saveptx = false;
static bool warmup = false;
static bool profile = false;
static bool O0 = false, O1 = false, O2 = false;
static bool debugnan = false;
static bool debug_uninit = false;
static bool userdata_isconnected = false;
static std::string extraoptions;
static std::string texoptions;
static int xres = 640, yres = 480;
static int aa = 1, max_bounces = 1000000, rr_depth = 5;
static int num_threads = 0;
static int iters = 1;
static std::string scenefile, imagefile;
static std::string shaderpath;
static bool shadingsys_options_set = false;
static bool use_optix = OIIO::Strutil::stoi(OIIO::Sysutil::getenv("TESTSHADE_OPTIX"));



// Set shading system global attributes based on command line options.
static void
set_shadingsys_options ()
{
    shadingsys->attribute ("debug", debug2 ? 2 : (debug1 ? 1 : 0));
    shadingsys->attribute ("compile_report", debug1|debug2);
    int opt = 2;  // default
    if (O0) opt = 0;
    if (O1) opt = 1;
    if (O2) opt = 2;
    if (const char *opt_env = getenv ("TESTSHADE_OPT"))  // overrides opt
        opt = atoi(opt_env);
    shadingsys->attribute ("optimize", opt);
    shadingsys->attribute ("profile", int(profile));
    shadingsys->attribute ("lockgeom", 1);
    shadingsys->attribute ("debug_nan", debugnan);
    shadingsys->attribute ("debug_uninit", debug_uninit);
    shadingsys->attribute ("userdata_isconnected", userdata_isconnected);
    if (! shaderpath.empty())
        shadingsys->attribute ("searchpath:shader", shaderpath);
    if (extraoptions.size())
        shadingsys->attribute ("options", extraoptions);
    if (texoptions.size())
        shadingsys->texturesys()->attribute ("options", texoptions);
    shadingsys_options_set = true;
}



int get_filenames(int argc, const char *argv[])
{
    for (int i = 0; i < argc; i++) {
        if (scenefile.empty())
            scenefile = argv[i];
        else if (imagefile.empty())
            imagefile = argv[i];
    }
    return 0;
}

void getargs(int argc, const char *argv[])
{
    bool help = false;
    OIIO::ArgParse ap;
    ap.options ("Usage:  testrender [options] scene.xml outputfilename",
                "%*", get_filenames, "",
                "--help", &help, "Print help message",
                "-v", &verbose, "Verbose messages",
                "-t %d", &num_threads, "Render using N threads (default: auto-detect)",
                "--optix", &use_optix, "Use OptiX if available",
                "--debug", &debug1, "Lots of debugging info",
                "--debug2", &debug2, "Even more debugging info",
                "--runstats", &runstats, "Print run statistics",
                "--stats", &runstats, "", // DEPRECATED 1.7
                "--profile", &profile, "Print profile information",
                "--saveptx", &saveptx, "Save the generated PTX (OptiX mode only)",
                "--warmup", &warmup, "Perform a warmup launch",
                "--res %d %d", &xres, &yres, "Make an W x H image",
                "-r %d %d", &xres, &yres, "", // synonym for -res
                "-aa %d", &aa, "Trace NxN rays per pixel",
                "--iters %d", &iters, "Number of iterations",
                "-O0", &O0, "Do no runtime shader optimization",
                "-O1", &O1, "Do a little runtime shader optimization",
                "-O2", &O2, "Do lots of runtime shader optimization",
                "--debugnan", &debugnan, "Turn on 'debugnan' mode",
                "--path %s", &shaderpath, "Specify oso search path",
                "--options %s", &extraoptions, "Set extra OSL options",
                "--texoptions %s", &texoptions, "Set extra TextureSystem options",
                NULL);
    if (ap.parse(argc, argv) < 0) {
        std::cerr << ap.geterror() << std::endl;
        ap.usage ();
        exit (EXIT_FAILURE);
    }
    if (help) {
        std::cout <<
            "testrender -- Test Renderer for Open Shading Language\n"
             OSL_COPYRIGHT_STRING "\n";
        ap.usage ();
        exit (EXIT_SUCCESS);
    }
    if (scenefile.empty()) {
        std::cerr << "testrender: Must specify an xml scene file to open\n";
        ap.usage();
        exit (EXIT_FAILURE);
    }
    if (imagefile.empty()) {
        std::cerr << "testrender: Must specify a filename for output render\n";
        ap.usage();
        exit (EXIT_FAILURE);
    }
}

} // anonymous namespace



int
main (int argc, const char *argv[])
{
#ifdef OIIO_HAS_STACKTRACE
    // Helpful for debugging to make sure that any crashes dump a stack
    // trace.
    OIIO::Sysutil::setup_crash_stacktrace("stdout");
#endif

    try {
        using namespace OIIO;
        Timer timer;

        // Read command line arguments
        getargs (argc, argv);

        SimpleRaytracer *rend = nullptr;
        if (use_optix)
            rend = new OptixRaytracer;
        else
            rend = new SimpleRaytracer;

        // Other renderer and global options
        if (debug1 || verbose)
            rend->errhandler().verbosity (ErrorHandler::VERBOSE);
        rend->attribute("saveptx", (int)saveptx);
        rend->attribute("max_bounces", max_bounces);
        rend->attribute("rr_depth", rr_depth);
        rend->attribute("aa", aa);
        OIIO::attribute("threads", num_threads);

        // Create a new shading system.  We pass it the RendererServices
        // object that services callbacks from the shading system, the
        // TextureSystem (note: passing nullptr just makes the ShadingSystem
        // make its own TS), and an error handler.
        shadingsys = new ShadingSystem (rend, nullptr, &rend->errhandler());
        rend->shadingsys = shadingsys;

        // Register the layout of all closures known to this renderer
        // Any closure used by the shader which is not registered, or
        // registered with a different number of arguments will lead
        // to a runtime error.
        register_closures(shadingsys);

        // Setup common attributes
        set_shadingsys_options();

        // Loads a scene, creating camera, geometry and assigning shaders
        rend->camera.resolution (xres, yres);
        rend->parse_scene_xml (scenefile);

        rend->prepare_render ();

        rend->pixelbuf.reset (ImageSpec(xres, yres, 3, TypeDesc::FLOAT));

        double setuptime = timer.lap ();

        if (warmup)
            rend->warmup();
        double warmuptime = timer.lap ();

        // Launch the kernel to render the scene
        for (int i = 0; i < iters; ++i)
            rend->render (xres, yres);
        double runtime = timer.lap ();

        rend->finalize_pixel_buffer ();

        // Write image to disk
        if (Strutil::iends_with (imagefile, ".jpg") ||
            Strutil::iends_with (imagefile, ".jpeg") ||
            Strutil::iends_with (imagefile, ".gif") ||
            Strutil::iends_with (imagefile, ".png")) {
            // JPEG, GIF, and PNG images should be automatically saved as sRGB
            // because they are almost certainly supposed to be displayed on web
            // pages.
            ImageBufAlgo::colorconvert (rend->pixelbuf, rend->pixelbuf,
                                        "linear", "sRGB", false, "", "");
        }
        rend->pixelbuf.set_write_format (TypeDesc::HALF);
        if (! rend->pixelbuf.write (imagefile))
            rend->errhandler().error ("Unable to write output image: %s",
                                      rend->pixelbuf.geterror());
        double writetime = timer.lap();

        // Print some debugging info
        if (debug1 || runstats || profile) {
            std::cout << "\n";
            std::cout << "Setup : " << OIIO::Strutil::timeintervalformat (setuptime,4) << "\n";
            std::cout << "Warmup: " << OIIO::Strutil::timeintervalformat (warmuptime,4) << "\n";
            std::cout << "Run   : " << OIIO::Strutil::timeintervalformat (runtime,4) << "\n";
            std::cout << "Write : " << OIIO::Strutil::timeintervalformat (writetime,4) << "\n";
            std::cout << "\n";
            std::cout << shadingsys->getstats (5) << "\n";
            OIIO::TextureSystem *texturesys = shadingsys->texturesys();
            if (texturesys)
                std::cout << texturesys->getstats (5) << "\n";
            std::cout << ustring::getstats() << "\n";
        }

        // We're done with the shading system now, destroy it
        rend->clear();
        delete shadingsys;
        delete rend;
    } catch (const optix::Exception& e) {
        printf("Optix Error: %s\n", e.what());
    } catch (const std::exception& e) {
        printf("Unknown Error: %s\n", e.what());
    }

    return EXIT_SUCCESS;
}
