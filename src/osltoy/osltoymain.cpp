// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <iostream>

#include <QApplication>

// QT's extension foreach defines a foreach macro which interferes
// with an OSL internal foreach method.  So we will undefine it here
#undef foreach
// It is recommended any uses of QT's foreach be migrated
// to use C++11 range based loops.

#include <OpenImageIO/argparse.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>

#include <OSL/oslexec.h>
#include "osltoyapp.h"
#include "osltoyrenderer.h"
using namespace OSL;


#ifdef _MSC_VER
// if we are not in DEBUG mode this code switch the app to
// full windowed mode (no console and no need to define WinMain)
// FIXME: this should be done in CMakeLists.txt but first we have to
// fix Windows Debug build
#    ifdef NDEBUG
#        pragma comment(linker, "/subsystem:windows /entry:mainCRTStartup")
#    endif
#endif


static bool verbose         = false;
static bool foreground_mode = true;
static int threads          = 0;
static int xres = 512, yres = 512;
static std::vector<std::string> filenames;
static std::vector<std::string> include_paths;


static void
getargs(int argc, char* argv[])
{
    bool help = false;
    OIIO::ArgParse ap;
    // clang-format off
    ap.intro("osltoy -- interactive OSL plaything\n" OSL_INTRO_STRING);
    ap.usage("osltoy [options] [filename...]");
    ap.arg("filename")
      .hidden()
      .action([&](cspan<const char*> argv){ filenames.emplace_back(argv[0]); });
    ap.arg("-v", &verbose)
      .help("Verbose output");
    ap.arg("--threads %d:NTHREADS", &threads)
      .help("Set thread count (0=cores)");
    ap.arg("--res %d:XRES %d:YRES", &xres, &yres)
      .help("Set resolution");
    ap.arg("-I DIRPATH")
      .action([&](cspan<const char*> argv){ include_paths.emplace_back(argv[1]); })
      .help("Add DIRPATH to the list of header search paths.");
    // clang-format on
    if (ap.parse(argc, (const char**)argv) < 0) {
        std::cerr << ap.geterror() << std::endl;
        ap.usage();
        exit(EXIT_FAILURE);
    }
    if (help) {
        ap.usage();
        exit(EXIT_FAILURE);
    }
}



int
main(int argc, char* argv[])
{
#ifdef OIIO_HAS_STACKTRACE
    // Helpful for debugging to make sure that any crashes dump a stack
    // trace.
    OIIO::Sysutil::setup_crash_stacktrace("stdout");
#endif

    OIIO::Filesystem::convert_native_arguments(argc, (const char**)argv);

    getargs(argc, argv);
    if (!foreground_mode)
        OIIO::Sysutil::put_in_background(argc, argv);

    OIIO::attribute("threads", threads);
    OSLToyRenderer* rend = new OSLToyRenderer;
    rend->set_resolution(xres, yres);

    QApplication app(argc, argv);
    OSLToyMainWindow mainwin(rend, xres, yres);
    mainwin.show();

    mainwin.set_include_search_paths(include_paths);

    for (auto&& filename : filenames)
        mainwin.open_file(filename);

    int qtresult = app.exec();

    // Clean up here

    return qtresult;
}
