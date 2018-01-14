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


#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>
#include <OpenImageIO/thread.h>

#include <OSL/oslcomp.h>
#include <OSL/oslexec.h>
using namespace OSL;



static void
usage ()
{
    std::cout <<
        "oslc -- Open Shading Language compiler " OSL_LIBRARY_VERSION_STRING "\n"
        OSL_COPYRIGHT_STRING "\n"
        "Usage:  oslc [options] file\n"
        "  Options:\n"
        "\t--help         Print this usage message\n"
        "\t-o filename    Specify output filename\n"
        "\t-v             Verbose mode\n"
        "\t-q             Quiet mode\n"
        "\t-Ipath         Add path to the #include search path\n"
        "\t-Dsym[=val]    Define preprocessor symbol\n"
        "\t-Usym          Undefine preprocessor symbol\n"
        "\t-O0, -O1, -O2  Set optimization level (default=1)\n"
        "\t-d             Debug mode\n"
        "\t-E             Only preprocess the input and output to stdout\n"
        ;
}



namespace { // anonymous

// Subclass ErrorHandler because we want our messages to appear somewhat
// differant than the default ErrorHandler base class, in order to match
// typical compiler command line messages.
class OSLC_ErrorHandler : public ErrorHandler {
public:
    virtual void operator () (int errcode, const std::string &msg) {
        static OIIO::mutex err_mutex;
        OIIO::lock_guard guard (err_mutex);
        switch (errcode & 0xffff0000) {
        case EH_INFO :
            if (verbosity() >= VERBOSE)
                std::cout << msg << std::endl;
            break;
        case EH_WARNING :
            if (verbosity() >= NORMAL)
                std::cerr << msg << std::endl;
            break;
        case EH_ERROR :
            std::cerr << msg << std::endl;
            break;
        case EH_SEVERE :
            std::cerr << msg << std::endl;
            break;
        case EH_DEBUG :
#ifdef NDEBUG
            break;
#endif
        default :
            if (verbosity() > QUIET)
                std::cout << msg;
            break;
        }
    }
};

static OSLC_ErrorHandler default_oslc_error_handler;
} // anonymous namespace




int
main (int argc, const char *argv[])
{
    // Globally force classic "C" locale, and turn off all formatting
    // internationalization, for the entire oslc application.
    std::locale::global (std::locale::classic());

    OIIO::Filesystem::convert_native_arguments (argc, (const char **)argv);


    if (argc <= 1) {
        usage ();
        return EXIT_SUCCESS;
    }

    std::vector<std::string> args;
    bool quiet = false;
    std::string shader_path;

    // Parse arguments from command line
    for (int a = 1;  a < argc;  ++a) {
        if (! strcmp (argv[a], "--help") | ! strcmp (argv[a], "-h")) {
            usage ();
            return EXIT_SUCCESS;
        }
        else if (! strcmp (argv[a], "-v") ||
                 ! strcmp (argv[a], "-q") ||
                 ! strcmp (argv[a], "-d") ||
                 ! strcmp (argv[a], "-E") ||
                 ! strcmp (argv[a], "-O") || ! strcmp (argv[a], "-O0") ||
                 ! strcmp (argv[a], "-O1") || ! strcmp (argv[a], "-O2")) {
            // Valid command-line argument
            args.emplace_back(argv[a]);
            quiet |= (strcmp (argv[a], "-q") == 0);
        }
        else if (! strcmp (argv[a], "-o") && a < argc-1) {
            // Output filepath
            args.emplace_back(argv[a]);
            ++a;
            args.emplace_back(argv[a]);
        }
        else if (argv[a][0] == '-' &&
                 (argv[a][1] == 'D' || argv[a][1] == 'U' || argv[a][1] == 'I')) {
            args.emplace_back(argv[a]);
        }
        else {
            // Shader to compile
            shader_path = argv[a];
        }
    }

    if (shader_path.empty ()) {
        std::cout << "ERROR: Missing shader path" << "\n\n";
        usage ();
        return EXIT_FAILURE;
    }

    OSLCompiler compiler (&default_oslc_error_handler);
    bool ok = compiler.compile (shader_path, args);
    if (ok) {
        if (!quiet)
            std::cout << "Compiled " << shader_path << " -> " << compiler.output_filename() << "\n";
    }
    else {
        std::cout << "FAILED " << shader_path << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
