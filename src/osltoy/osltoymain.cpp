/*
Copyright (c) 2017 Sony Pictures Imageworks Inc., et al.
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

#include <QApplication>

#include <OpenImageIO/argparse.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>

#include <OSL/oslexec.h>
#include "osltoyapp.h"
#include "osltoyrenderer.h"
using namespace OSL;


#ifdef WIN32
    // if we are not in DEBUG mode this code switch the app to
    // full windowed mode (no console and no need to define WinMain)
    // FIXME: this should be done in CMakeLists.txt but first we have to
    // fix Windows Debug build
# ifdef NDEBUG
#  pragma comment(linker, "/subsystem:windows /entry:mainCRTStartup")
# endif
#endif


static bool verbose = false;
static bool foreground_mode = true;
static int threads = 0;
static int xres = 512, yres = 512;
static std::vector<std::string> filenames;


static int
parse_files (int argc, const char *argv[])
{
    for (int i = 0;  i < argc;  i++)
        filenames.emplace_back(argv[i]);
    return 0;
}


static void
getargs (int argc, char *argv[])
{
    bool help = false;
    OIIO::ArgParse ap;
    ap.options ("osltoy -- interactive OSL plaything\n"
                OSL_INTRO_STRING "\n"
                "Usage:  osltoy [options] [filename...]",
                  "%*", parse_files, "",
                  "--help", &help, "Print help message",
                  "-v", &verbose, "Verbose status messages",
                  "--threads %d", &threads, "Set thread count (0=cores)",
                  "--res %d %d", &xres, &yres, "Set resolution (x, y)",
                  NULL);
    if (ap.parse (argc, (const char**)argv) < 0) {
        std::cerr << ap.geterror() << std::endl;
        ap.usage ();
        exit (EXIT_FAILURE);
    }
    if (help) {
        ap.usage ();
        exit (EXIT_FAILURE);
    }
}




int
main (int argc, char* argv[])
{
    OIIO::Filesystem::convert_native_arguments (argc, (const char **)argv);

    getargs (argc, argv);
    if (! foreground_mode)
        OIIO::Sysutil::put_in_background (argc, argv);

    OIIO::attribute ("threads", threads);
    OSLToyRenderer *rend = new OSLToyRenderer;
    rend->set_resolution (xres, yres);

    QApplication app(argc, argv);
    OSLToyMainWindow mainwin (rend, xres, yres);
    mainwin.show();
    for (auto&& filename : filenames)
        mainwin.open_file (filename);

    int qtresult = app.exec();

    // Clean up here

    return qtresult;
}
