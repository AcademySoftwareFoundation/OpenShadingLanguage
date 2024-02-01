// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <OSL/oslconfig.h>

#include <OpenImageIO/argparse.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/parallel.h>
#include <OpenImageIO/strutil.h>
#include <OpenImageIO/sysutil.h>
#include <OpenImageIO/thread.h>
#include <OpenImageIO/timer.h>

#include <OSL/oslexec.h>
#include "shading.h"
#include "simpleraytracer.h"

#if OSL_USE_OPTIX
#    include "optixraytracer.h"
#endif

using namespace OSL;

namespace {  // anonymous namespace

static ShadingSystem* shadingsys = NULL;
static bool debug1               = false;
static bool debug2               = false;
static bool verbose              = false;
static bool runstats             = false;
static bool saveptx              = false;
static bool warmup               = false;
static bool profile              = false;
static bool O0 = false, O1 = false, O2 = false;
static int llvm_opt              = 1;  // LLVM optimization level
static bool debugnan             = false;
static bool debug_uninit         = false;
static bool userdata_isconnected = false;
static std::string extraoptions;
static std::string texoptions;
static int xres = 640, yres = 480;
static int aa = 1, max_bounces = 1000000, rr_depth = 5;
static float show_albedo_scale = 0.0f;
static int num_threads         = 0;
static int iters               = 1;
static std::string scenefile, imagefile;
static std::string shaderpath;
static bool shadingsys_options_set = false;
static bool use_optix              = OIIO::Strutil::stoi(
    OIIO::Sysutil::getenv("TESTSHADE_OPTIX"));
static bool optix_no_inline             = false;
static bool optix_no_inline_layer_funcs = false;
static bool optix_no_merge_layer_funcs  = false;
static bool optix_no_inline_rend_lib    = false;
static bool optix_no_rend_lib_bitcode   = false;
static int optix_no_inline_thresh       = 100000;
static int optix_force_inline_thresh    = 0;


// Set shading system global attributes based on command line options.
static void
set_shadingsys_options()
{
    shadingsys->attribute("debug", debug2 ? 2 : (debug1 ? 1 : 0));
    shadingsys->attribute("compile_report", debug1 | debug2);
    int opt = 2;  // default
    if (O0)
        opt = 0;
    if (O1)
        opt = 1;
    if (O2)
        opt = 2;
    if (const char* opt_env = getenv("TESTSHADE_OPT"))  // overrides opt
        opt = atoi(opt_env);
    shadingsys->attribute("optimize", opt);

    // The cost of more optimization passes usually pays for itself by
    // reducing the number of instructions JIT ultimately has to lower to
    // the target ISA.
    if (const char* llvm_opt_env = getenv(
            "TESTSHADE_LLVM_OPT"))  // overrides llvm_opt
        llvm_opt = atoi(llvm_opt_env);
    shadingsys->attribute("llvm_optimize", llvm_opt);

    // Experimental: Control the inlining behavior when compiling for OptiX.
    // These attributes have been added to aid tuning the GPU optimization
    // passes and may be removed or changed in the future.
    shadingsys->attribute("optix_no_inline", optix_no_inline);
    shadingsys->attribute("optix_no_inline_layer_funcs",
                          optix_no_inline_layer_funcs);
    shadingsys->attribute("optix_merge_layer_funcs",
                          !optix_no_merge_layer_funcs);
    shadingsys->attribute("optix_no_inline_rend_lib", optix_no_inline_rend_lib);
    shadingsys->attribute("optix_no_inline_thresh", optix_no_inline_thresh);
    shadingsys->attribute("optix_force_inline_thresh",
                          optix_force_inline_thresh);

    shadingsys->attribute("profile", int(profile));
    shadingsys->attribute("debug_nan", debugnan);
    shadingsys->attribute("debug_uninit", debug_uninit);
    shadingsys->attribute("userdata_isconnected", userdata_isconnected);
    if (!shaderpath.empty())
        shadingsys->attribute("searchpath:shader", shaderpath);
    if (extraoptions.size())
        shadingsys->attribute("options", extraoptions);
    if (texoptions.size())
        shadingsys->texturesys()->attribute("options", texoptions);
    // Always generate llvm debugging info and profiling events
    shadingsys->attribute("llvm_debugging_symbols", 1);
    shadingsys->attribute("llvm_profiling_events", 1);

    // We rely on the default set of "raytypes" tags. To use a custom set,
    // this is where we would do:
    //      shadingsys->attribute("raytypes", TypeDesc(TypeDesc::STRING, num_raytypes),
    //                           raytype_names);

    shadingsys_options_set = true;
}



void
getargs(int argc, const char* argv[])
{
    OIIO::ArgParse ap;
    // clang-format off
    ap.intro("testrender -- Test Renderer for Open Shading Language\n" OSL_COPYRIGHT_STRING);
    ap.usage("testrender [options] scene.xml outputfilename");
    ap.arg("filename")
      .hidden()
      .action([&](cspan<const char*> argv){
          if (scenefile.empty())
              scenefile = argv[0];
          else if (imagefile.empty())
              imagefile = argv[0];
          });
    ap.arg("-v", &verbose)
      .help("Verbose output");
    ap.arg("-t %d:NTHREADS", &num_threads)
      .help("Set thread count (default = 0: auto-detect #cores)");
    ap.arg("--res %d:XRES %d:YRES", &xres, &yres)
      .help("Set resolution");
    ap.arg("--optix", &use_optix)
      .help("Use OptiX if available");
    ap.arg("--debug", &debug1)
      .help("Lots of debugging info");
    ap.arg("--debug2", &debug2)
      .help("Even more debugging info");
    ap.arg("--runstats", &runstats)
      .help("Print run statistics");
    ap.arg("--stats", &runstats)
      .hidden(); // DEPRECATED 1.7
    ap.arg("--profile", &profile)
      .help("Print profile information");
    ap.arg("--saveptx", &saveptx)
      .help("Save the generated PTX (OptiX mode only)");
    ap.arg("--warmup", &warmup)
      .help("Perform a warmup launch");
    ap.arg("--res %d:W %d:H", &xres, &yres)
      .help("Set resolution of output image to W x H");
    ap.arg("-r %d:W %d:H", &xres, &yres)  // synonym for -res
      .hidden();
    ap.arg("-aa %d:N", &aa)
      .help("Trace NxN rays per pixel");
    ap.arg("-albedo %f:SCALE", &show_albedo_scale)
      .help("Visualize the albedo of each pixel instead of path tracing");
    ap.arg("--iters %d:N", &iters)
      .help("Number of iterations");
    ap.arg("-O0", &O0)
      .help("Do no runtime shader optimization");
    ap.arg("-O1", &O1)
      .help("Do a little runtime shader optimization");
    ap.arg("-O2", &O2)
      .help("Do lots of runtime shader optimization");
    ap.arg("--llvm_opt %d:LEVEL", &llvm_opt)
      .help("LLVM JIT optimization level");
    ap.arg("--optix_no_inline", &optix_no_inline)
      .help("Disable function inlining when compiling for OptiX");
    ap.arg("--optix_no_inline_layer_funcs", &optix_no_inline_layer_funcs)
      .help("Disable inlining the group layer functions when compiling for OptiX");
    ap.arg("--optix_no_merge_layer_funcs", &optix_no_merge_layer_funcs)
      .help("Disable merging group layer functions with only one caller when compiling for OptiX");
    ap.arg("--optix_no_inline_rend_lib", &optix_no_inline_rend_lib)
      .help("Disable inlining the rend_lib functions when compiling for OptiX");
    ap.arg("--optix_no_rend_lib_bitcode", &optix_no_rend_lib_bitcode)
      .help("Don't pass LLVM bitcode for the rend_lib functions to the ShadingSystem");
    ap.arg("--optix_no_inline_thresh %d:THRESH", &optix_no_inline_thresh)
      .help("Don't inline functions larger than the threshold when compiling for OptiX");
    ap.arg("--optix_force_inline_thresh %d:THRESH", &optix_force_inline_thresh)
      .help("Force inline functions smaller than the threshold when compiling for OptiX");
    ap.arg("--debugnan", &debugnan)
      .help("Turn on 'debugnan' mode");
    ap.arg("--path SEARCHPATH", &shaderpath)
      .help("Specify oso search path");
    ap.arg("--options %s:LIST", &extraoptions)
      .help("Set extra OSL options");
    ap.arg("--texoptions %s:LIST", &texoptions)
      .help("Set extra TextureSystem options");

    // clang-format on
    if (ap["help"].get<int>()) {
        ap.print_help();
        ap.abort();
        exit(EXIT_SUCCESS);
    }
    if (ap.parse(argc, argv) < 0) {
        std::cerr << ap.geterror() << "\n\n";
        ap.usage();
        exit(EXIT_FAILURE);
    }
    if (scenefile.empty()) {
        std::cerr << "testrender: Must specify an xml scene file to open\n\n";
        ap.usage();
        exit(EXIT_FAILURE);
    }
    if (imagefile.empty()) {
        std::cerr << "testrender: Must specify a filename for output render\n\n";
        ap.usage();
        exit(EXIT_FAILURE);
    }
}

}  // anonymous namespace



int
main(int argc, const char* argv[])
{
#ifdef OIIO_HAS_STACKTRACE
    // Helpful for debugging to make sure that any crashes dump a stack
    // trace.
    OIIO::Sysutil::setup_crash_stacktrace("stdout");
#endif

#if OIIO_SIMD_SSE && !OIIO_F16C_ENABLED
    // Some rogue libraries (and icc runtime libs?) will turn on the cpu mode
    // that causes floating point denormals get crushed to 0.0 in certain ops,
    // and leave it that way! This can give us the wrong results for the
    // particular sequence of SSE intrinsics we use to convert half->float for
    // exr files containing pixels with denorm values.
    OIIO::simd::set_denorms_zero_mode(false);
#endif

    using namespace OIIO;
    Timer timer;

    // Read command line arguments
    getargs(argc, argv);

    // Allow magic env variable TESTRENDER_AA to override the --aa option,
    // this is helpful for certain CI tests in special debug modes that would
    // be too slow to be practical.
    int aaoverride = OIIO::Strutil::stoi(
        OIIO::Sysutil::getenv("TESTRENDER_AA"));
    if (aaoverride)
        aa = aaoverride;

    SimpleRaytracer* rend = nullptr;
#if OSL_USE_OPTIX
    if (use_optix)
        rend = new OptixRaytracer;
    else
#endif
        rend = new SimpleRaytracer;

    // Other renderer and global options
    if (debug1 || verbose)
        rend->errhandler().verbosity(ErrorHandler::VERBOSE);
    rend->attribute("max_bounces", max_bounces);
    rend->attribute("rr_depth", rr_depth);
    rend->attribute("aa", aa);
    rend->attribute("show_albedo_scale", show_albedo_scale);
    OIIO::attribute("threads", num_threads);

#if OSL_USE_OPTIX
    rend->attribute("saveptx", (int)saveptx);
    rend->attribute("no_rend_lib_bitcode", (int)optix_no_rend_lib_bitcode);
#endif

    // Create a new shading system.  We pass it the RendererServices
    // object that services callbacks from the shading system, the
    // TextureSystem (note: passing nullptr just makes the ShadingSystem
    // make its own TS), and an error handler.
    shadingsys       = new ShadingSystem(rend, nullptr, &rend->errhandler());
    rend->shadingsys = shadingsys;

    // Register the layout of all closures known to this renderer
    // Any closure used by the shader which is not registered, or
    // registered with a different number of arguments will lead
    // to a runtime error.
    register_closures(shadingsys);

    // Setup common attributes
    set_shadingsys_options();

#if OSL_USE_OPTIX
    if (use_optix)
        reinterpret_cast<OptixRaytracer*>(rend)->synch_attributes();
#endif

    // Loads a scene, creating camera, geometry and assigning shaders
    rend->camera.resolution(xres, yres);
    rend->parse_scene_xml(scenefile);

    rend->prepare_render();

    rend->pixelbuf.reset(ImageSpec(xres, yres, 3, TypeDesc::FLOAT));

    double setuptime = timer.lap();

    if (warmup)
        rend->warmup();
    double warmuptime = timer.lap();

    // Launch the kernel to render the scene
    for (int i = 0; i < iters; ++i)
        rend->render(xres, yres);
    double runtime = timer.lap();

    rend->finalize_pixel_buffer();

    // Write image to disk
    if (Strutil::iends_with(imagefile, ".jpg")
        || Strutil::iends_with(imagefile, ".jpeg")
        || Strutil::iends_with(imagefile, ".gif")
        || Strutil::iends_with(imagefile, ".png")) {
        // JPEG, GIF, and PNG images should be automatically saved as sRGB
        // because they are almost certainly supposed to be displayed on web
        // pages.
        ImageBufAlgo::colorconvert(rend->pixelbuf, rend->pixelbuf, "linear",
                                   "sRGB", false, "", "");
    }
    rend->pixelbuf.set_write_format(TypeDesc::HALF);
    if (!rend->pixelbuf.write(imagefile))
        rend->errhandler().errorfmt("Unable to write output image: {}",
                                    rend->pixelbuf.geterror());
    double writetime = timer.lap();

    // Print some debugging info
    if (debug1 || runstats || profile) {
        std::cout << "\n";
        std::cout << "Setup : "
                  << OIIO::Strutil::timeintervalformat(setuptime, 4) << "\n";
        std::cout << "Warmup: "
                  << OIIO::Strutil::timeintervalformat(warmuptime, 4) << "\n";
        std::cout << "Run   : " << OIIO::Strutil::timeintervalformat(runtime, 4)
                  << "\n";
        std::cout << "Write : "
                  << OIIO::Strutil::timeintervalformat(writetime, 4) << "\n";
        std::cout << "\n";
        std::cout << shadingsys->getstats(5) << "\n";
        OIIO::TextureSystem* texturesys = shadingsys->texturesys();
        if (texturesys)
            std::cout << texturesys->getstats(5) << "\n";
        std::cout << ustring::getstats() << "\n";
    }

    // We're done with the shading system now, destroy it
    rend->clear();
    delete rend;
    delete shadingsys;
    return EXIT_SUCCESS;
}
