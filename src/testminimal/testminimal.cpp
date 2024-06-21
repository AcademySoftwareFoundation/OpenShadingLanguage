// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>
#include <iostream>
#include "oslmaterial.h"

using namespace OSL;

int
main(int argc, char** argv)
{
    int batch_width;
    char* shader_name;
    bool use_fma = false;
    if (argc < 2) {
        std::cout
            << "usage: shader_name(without .oso) [+optional] batch_width (0/8/16) [+optional] use_fma 0/1"
            << std::endl;
        return 0;
    } else if (argc >= 3) {
        shader_name = argv[1];
        batch_width = atoi(argv[2]);
#if OSL_USE_BATCHED
        if (batch_width != -1 && batch_width != 1 && batch_width != 8
            && batch_width != 16)
            batch_width = 1;
#else
        if (batch_width != 1)
            batch_width = 1;
#endif
        if (argc >= 4) {
            use_fma = atoi(argv[3]);
        }
    } else {
        shader_name = argv[1];
        batch_width = -1;
    }

    OSLMaterial* oslmat = NULL;
#if OSL_USE_BATCHED
    BatchedOSLMaterial<8>* boslmat8   = NULL;
    BatchedOSLMaterial<16>* boslmat16 = NULL;
#endif

    TextureSystem* texturesys = TextureSystem::create();
    ShadingSystem* ss         = NULL;

    if (batch_width == -1) {
#if OSL_USE_BATCHED
        oslmat = new OSLMaterial();
        ss     = new ShadingSystem(oslmat, NULL, &oslmat->errhandler());
        if (use_fma)
            ss->attribute("llvm_jit_fma", true);
        if (ss->configure_batch_execution_at(16))
            batch_width = 16;
        else if (ss->configure_batch_execution_at(8))
            batch_width = 8;
        else
            batch_width = 1;
        delete oslmat;
        oslmat = NULL;
        delete ss;
        ss = NULL;
#else
        if (use_fma)
            ss->attribute("llvm_jit_fma", true);
        batch_width = 1;
#endif
    }

    switch (batch_width) {
    case 1:
        oslmat = new OSLMaterial();
        ss     = new ShadingSystem(oslmat, texturesys, &oslmat->errhandler());
        break;
#if OSL_USE_BATCHED
    case 8:
        boslmat8 = new BatchedOSLMaterial<8>();
        ss = new ShadingSystem(boslmat8, texturesys, &boslmat8->errhandler());
        break;
    case 16:
        boslmat16 = new BatchedOSLMaterial<16>();
        ss = new ShadingSystem(boslmat16, texturesys, &boslmat16->errhandler());
        break;
#endif
    }

#if OSL_USE_BATCHED
    if (batch_width > 1) {
        if (use_fma)
            ss->attribute("llvm_jit_fma", true);
        ss->configure_batch_execution_at(batch_width);

        // build searchpath for ISA specific OSL shared libraries based on expected
        // location of library directories relative to the executables path.
        static const char* relative_lib_dirs[] =
#    if (defined(_WIN32) || defined(_WIN64))
            { "\\..\\lib64", "\\..\\lib" };
#    else
            { "/../lib64", "/../lib" };
#    endif
        auto executable_directory = OIIO::Filesystem::parent_path(
            OIIO::Sysutil::this_program_path());
        int dirNum = 0;
        std::string librarypath;
        for (const char* relative_lib_dir : relative_lib_dirs) {
            if (dirNum++ > 0)
                librarypath += ":";
            librarypath += executable_directory + relative_lib_dir;
        }
        ss->attribute("searchpath:library", librarypath);
    }
#endif

    PerThreadInfo* thread_info;
    ShadingContext* context;
    thread_info = ss->create_thread_info();
    context     = ss->get_context(thread_info);

    switch (batch_width) {
    case 1: oslmat->run_test(ss, thread_info, context, shader_name); break;
#if OSL_USE_BATCHED
    case 8: boslmat8->run_test(ss, thread_info, context, shader_name); break;
    case 16: boslmat16->run_test(ss, thread_info, context, shader_name); break;
#endif
    }

    ss->release_context(context);
    ss->destroy_thread_info(thread_info);

    delete oslmat;
#if OSL_USE_BATCHED
    delete boslmat8;
    delete boslmat16;
#endif
    delete ss;

    return 0;
}
