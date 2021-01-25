// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <string>
#include <iostream>
#include <cstdlib>

#ifdef __linux__
#include <dlfcn.h>
#endif

#include <OpenImageIO/plugin.h>
#include <OpenImageIO/sysutil.h>
using namespace OIIO;


typedef int (*EntryPoint)(int argc, const char *argv[]);



int
main (int argc, const char *argv[])
{
#ifdef OIIO_HAS_STACKTRACE
    // Helpful for debugging to make sure that any crashes dump a stack
    // trace.
    OIIO::Sysutil::setup_crash_stacktrace("stdout");
#endif

    std::string pluginname = std::string("libtestshade.") 
                             + Plugin::plugin_extension();
    Plugin::Handle handle = Plugin::open (pluginname, 
                                          false /* NOT RTLD_GLOBAL! */);
    if (! handle) {
        std::cerr << "Could not open " << pluginname << "\n";
        exit (1);
    }

    EntryPoint entry = (EntryPoint) Plugin::getsym (handle, "test_shade");
    if (! entry) {
        std::cerr << "Cound not find test_shade symbol\n";
        exit (1);
    }

    int r = entry (argc, argv);

    Plugin::close (handle);
    return r;
}

