// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "optix_compat.h" // Also tests this header can be included first
#include <stdio.h>
#include <stdlib.h>
#include <exception>

#include <OpenImageIO/sysutil.h>

using namespace OSL;         // For OSL::optix when OSL_USE_OPTIX=0

extern "C" int test_shade (int argc, const char *argv[]);


int
main (int argc, const char *argv[])
{
#ifdef OIIO_HAS_STACKTRACE
    // Helpful for debugging to make sure that any crashes dump a stack
    // trace.
    OIIO::Sysutil::setup_crash_stacktrace("stdout");
#endif

    int result = EXIT_FAILURE;
    try {
        result = test_shade (argc, argv);
    }
#if (OPTIX_VERSION < 70000)
    catch (const OSL::optix::Exception& e) {
        printf("Optix Error: %s\n", e.what());
    }
#endif
    catch (const std::exception& e) {
        printf("Unknown Error: %s\n", e.what());
    }
    return result;
}
