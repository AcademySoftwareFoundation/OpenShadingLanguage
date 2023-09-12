// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <vector>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>

#include <OSL/oslconfig.h>

#include "optixgridrender.h"

#include "render_params.h"

#include <cuda.h>
#include <nvrtc.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

// The pre-compiled renderer support library LLVM bitcode is embedded
// into the executable and made available through these variables.
extern int rend_lib_llvm_compiled_ops_size;
extern unsigned char rend_lib_llvm_compiled_ops_block[];


// The entry point for OptiX Module creation changed in OptiX 7.7
#if OPTIX_VERSION < 70700
const auto optixModuleCreateFn = optixModuleCreateFromPTX;
#else
const auto optixModuleCreateFn = optixModuleCreate;
#endif


OSL_NAMESPACE_ENTER


#define CUDA_CHECK(call)                                               \
    {                                                                  \
        cudaError_t res = call;                                        \
        if (res != cudaSuccess) {                                      \
            print(stderr,                                              \
                  "[CUDA ERROR] Cuda call '{}' failed with error:"     \
                  " {} ({}:{})\n",                                     \
                  #call, cudaGetErrorString(res), __FILE__, __LINE__); \
        }                                                              \
    }

#define OPTIX_CHECK(call)                                             \
    {                                                                 \
        OptixResult res = call;                                       \
        if (res != OPTIX_SUCCESS) {                                   \
            print(stderr,                                             \
                  "[OPTIX ERROR] OptiX call '{}' failed with error:"  \
                  " {} ({}:{})\n",                                    \
                  #call, optixGetErrorName(res), __FILE__, __LINE__); \
            exit(1);                                                  \
        }                                                             \
    }

#define OPTIX_CHECK_MSG(call, msg)                                         \
    {                                                                      \
        OptixResult res = call;                                            \
        if (res != OPTIX_SUCCESS) {                                        \
            print(stderr,                                                  \
                  "[OPTIX ERROR] OptiX call '{}' failed with error:"       \
                  " {} ({}:{})\nMessage: {}\n",                            \
                  #call, optixGetErrorName(res), __FILE__, __LINE__, msg); \
            exit(1);                                                       \
        }                                                                  \
    }

#define CUDA_SYNC_CHECK()                                                  \
    {                                                                      \
        cudaDeviceSynchronize();                                           \
        cudaError_t error = cudaGetLastError();                            \
        if (error != cudaSuccess) {                                        \
            print(stderr, "error ({}: line {}): {}\n", __FILE__, __LINE__, \
                  cudaGetErrorString(error));                              \
            exit(1);                                                       \
        }                                                                  \
    }


static void
context_log_cb(unsigned int level, const char* tag, const char* message,
               void* /*cbdata */)
{
    //    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}



OptixGridRenderer::OptixGridRenderer()
{
    // Initialize CUDA
    cudaFree(0);

    CUcontext cuCtx = nullptr;  // zero means take the current context

    OptixDeviceContextOptions ctx_options = {};
    ctx_options.logCallbackFunction       = context_log_cb;
    ctx_options.logCallbackLevel          = 4;

    OPTIX_CHECK(optixInit());
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &ctx_options, &m_optix_ctx));

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaStreamCreate(&m_cuda_stream));

    m_fused_callable = false;
    if (const char* fused_env = getenv("TESTSHADE_FUSED"))
        m_fused_callable = atoi(fused_env);
}



void*
OptixGridRenderer::device_alloc(size_t size)
{
    void* ptr       = nullptr;
    cudaError_t res = cudaMalloc(reinterpret_cast<void**>(&ptr), size);
    if (res != cudaSuccess) {
        errhandler().errorfmt("cudaMalloc({}) failed with error: {}\n", size,
                              cudaGetErrorString(res));
    }
    return ptr;
}


void
OptixGridRenderer::device_free(void* ptr)
{
    cudaError_t res = cudaFree(ptr);
    if (res != cudaSuccess) {
        errhandler().errorfmt("cudaFree() failed with error: {}\n",
                              cudaGetErrorString(res));
    }
}


void*
OptixGridRenderer::copy_to_device(void* dst_device, const void* src_host,
                                  size_t size)
{
    cudaError_t res = cudaMemcpy(dst_device, src_host, size,
                                 cudaMemcpyHostToDevice);
    if (res != cudaSuccess) {
        errhandler().errorfmt(
            "cudaMemcpy host->device of size {} failed with error: {}\n", size,
            cudaGetErrorString(res));
    }
    return dst_device;
}



std::string
OptixGridRenderer::load_ptx_file(string_view filename)
{
    std::vector<std::string> paths
        = { OIIO::Filesystem::parent_path(OIIO::Sysutil::this_program_path()),
            PTX_PATH };
    std::string filepath = OIIO::Filesystem::searchpath_find(filename, paths,
                                                             false);
    if (OIIO::Filesystem::exists(filepath)) {
        std::string ptx_string;
        if (OIIO::Filesystem::read_text_file(filepath, ptx_string))
            return ptx_string;
    }
    errhandler().severefmt("Unable to load {}", filename);
    return {};
}



OptixGridRenderer::~OptixGridRenderer()
{
    for (void* p : m_ptrs_to_free)
        cudaFree(p);
    if (m_optix_ctx)
        OPTIX_CHECK(optixDeviceContextDestroy(m_optix_ctx));
}



void
OptixGridRenderer::init_shadingsys(ShadingSystem* ss)
{
    shadingsys = ss;
}



bool
OptixGridRenderer::init_optix_context(int xres OSL_MAYBE_UNUSED,
                                      int yres OSL_MAYBE_UNUSED)
{
    if (!options.get_int("no_rend_lib_bitcode")) {
        shadingsys->attribute("lib_bitcode",
                              { OSL::TypeDesc::UINT8,
                                rend_lib_llvm_compiled_ops_size },
                              rend_lib_llvm_compiled_ops_block);
    }
    if (options.get_int("optix_register_inline_funcs")) {
        register_inline_functions();
    }
    return true;
}



bool
OptixGridRenderer::synch_attributes()
{
    // FIXME -- this is for testing only
    // Make some device strings to test userdata parameters
    ustring userdata_str1("ud_str_1");
    ustring userdata_str2("userdata string");

    // Store the user-data
    test_str_1 = userdata_str1.hash();
    test_str_2 = userdata_str2.hash();

    {
        char* colorSys            = nullptr;
        long long cpuDataSizes[2] = { 0, 0 };
        if (!shadingsys->getattribute("colorsystem", TypeDesc::PTR,
                                      (void*)&colorSys)
            || !shadingsys->getattribute("colorsystem:sizes",
                                         TypeDesc(TypeDesc::LONGLONG, 2),
                                         (void*)&cpuDataSizes)
            || !colorSys || !cpuDataSizes[0]) {
            errhandler().errorfmt("No colorsystem available.");
            return false;
        }

        auto cpuDataSize = cpuDataSizes[0];
        auto numStrings  = cpuDataSizes[1];

        // Get the size data-size, minus the ustring size
        const size_t podDataSize = cpuDataSize
                                   - sizeof(StringParam) * numStrings;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_color_system),
                              podDataSize + sizeof(uint64_t) * numStrings));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_color_system), colorSys,
                              podDataSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_osl_printf_buffer),
                              OSL_PRINTF_BUFFER_SIZE));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_osl_printf_buffer), 0,
                              OSL_PRINTF_BUFFER_SIZE));

        // Transforms
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_object2common),
                              sizeof(OSL::Matrix44)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_object2common),
                              &m_object2common, sizeof(OSL::Matrix44),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_shader2common),
                              sizeof(OSL::Matrix44)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_shader2common),
                              &m_shader2common, sizeof(OSL::Matrix44),
                              cudaMemcpyHostToDevice));

        m_ptrs_to_free.push_back(reinterpret_cast<void*>(d_color_system));
        m_ptrs_to_free.push_back(reinterpret_cast<void*>(d_osl_printf_buffer));

        // then copy the device string to the end, first strings starting at dataPtr - (numStrings)
        // FIXME -- Should probably handle alignment better.
        const ustring* cpuString
            = (const ustring*)(colorSys
                               + (cpuDataSize
                                  - sizeof(StringParam) * numStrings));
        CUdeviceptr gpuStrings = d_color_system + podDataSize;
        for (const ustring* end = cpuString + numStrings; cpuString < end;
             ++cpuString) {
            // convert the ustring to a device string
            uint64_t devStr = cpuString->hash();
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(gpuStrings), &devStr,
                                  sizeof(devStr), cudaMemcpyHostToDevice));
            gpuStrings += sizeof(DeviceString);
        }
    }
    return true;
}



bool
OptixGridRenderer::make_optix_materials()
{
    // Stand-in: names of shader outputs to preserve
    // FIXME
    std::vector<const char*> outputs { "Cout" };

    // Optimize each ShaderGroup in the scene, and use the resulting
    // PTX to create OptiX Programs which can be called by the closest
    // hit program in the wrapper to execute the compiled OSL shader.
    int mtl_id = 0;

    std::vector<OptixModule> modules;

    // Space for message logging
    char msg_log[8192];
    size_t sizeof_msg_log;

    // Make module that contains programs we'll use in this scene
    OptixModuleCompileOptions module_compile_options = {};

    module_compile_options.maxRegisterCount
        = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
#if OPTIX_VERSION >= 70400
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif

    OptixPipelineCompileOptions pipeline_compile_options = {};

    pipeline_compile_options.traversableGraphFlags
        = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipeline_compile_options.usesMotionBlur     = false;
    pipeline_compile_options.numPayloadValues   = 0;
    pipeline_compile_options.numAttributeValues = 0;
    pipeline_compile_options.exceptionFlags
        = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "render_params";

    // Create 'raygen' program

    // Load the renderer CUDA source and generate PTX for it
    std::string progName    = "optix_grid_renderer.ptx";
    std::string program_ptx = load_ptx_file(progName);
    if (program_ptx.empty()) {
        errhandler().severefmt("Could not find PTX for the raygen program");
        return false;
    }

    sizeof_msg_log = sizeof(msg_log);
    OptixModule program_module;
    OPTIX_CHECK_MSG(optixModuleCreateFn(m_optix_ctx, &module_compile_options,
                                        &pipeline_compile_options,
                                        program_ptx.c_str(), program_ptx.size(),
                                        msg_log, &sizeof_msg_log,
                                        &program_module),
                    fmtformat("Creating Module from PTX-file {}", msg_log));

    // Record it so we can destroy it later
    modules.push_back(program_module);

    OptixProgramGroupOptions program_options = {};
    std::vector<OptixProgramGroup> program_groups;
    std::vector<void*> material_interactive_params;

    // Raygen group
    OptixProgramGroupDesc raygen_desc    = {};
    raygen_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module            = program_module;
    raygen_desc.raygen.entryFunctionName = "__raygen__";

    OptixProgramGroup raygen_group;
    sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK_MSG(optixProgramGroupCreate(m_optix_ctx, &raygen_desc,
                                            1,  // number of program groups
                                            &program_options,  // program options
                                            msg_log, &sizeof_msg_log,
                                            &raygen_group),
                    fmtformat("Creating 'ray-gen' program group: {}", msg_log));

    // Set Globals Raygen group
    OptixProgramGroupDesc setglobals_raygen_desc = {};
    setglobals_raygen_desc.kind          = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    setglobals_raygen_desc.raygen.module = program_module;
    setglobals_raygen_desc.raygen.entryFunctionName = "__raygen__setglobals";

    OptixProgramGroup setglobals_raygen_group;
    sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK_MSG(optixProgramGroupCreate(
                        m_optix_ctx, &setglobals_raygen_desc,
                        1,                 // number of program groups
                        &program_options,  // program options
                        msg_log, &sizeof_msg_log, &setglobals_raygen_group),
                    fmtformat("Creating 'ray-gen' program group: {}", msg_log));

    // Miss group
    OptixProgramGroupDesc miss_desc  = {};
    miss_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module            = program_module;
    miss_desc.miss.entryFunctionName = "__miss__";

    OptixProgramGroup miss_group;
    sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK_MSG(optixProgramGroupCreate(m_optix_ctx, &miss_desc, 1,
                                            &program_options, msg_log,
                                            &sizeof_msg_log, &miss_group),
                    fmtformat("Creating 'miss' program group: {}", msg_log));

    // Set Globals Miss group
    OptixProgramGroupDesc setglobals_miss_desc  = {};
    setglobals_miss_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    setglobals_miss_desc.miss.module            = program_module;
    setglobals_miss_desc.miss.entryFunctionName = "__miss__setglobals";

    OptixProgramGroup setglobals_miss_group;
    sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK_MSG(optixProgramGroupCreate(m_optix_ctx, &setglobals_miss_desc,
                                            1, &program_options, msg_log,
                                            &sizeof_msg_log,
                                            &setglobals_miss_group),
                    fmtformat("Creating set-globals 'miss' program group: {}",
                              msg_log));

    // Hitgroup
    OptixProgramGroupDesc hitgroup_desc = {};
    hitgroup_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_desc.hitgroup.moduleCH     = program_module;
    hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__";
    hitgroup_desc.hitgroup.moduleAH            = program_module;
    hitgroup_desc.hitgroup.entryFunctionNameAH = "__anyhit__";

    OptixProgramGroup hitgroup_group;

    sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK_MSG(
        optixProgramGroupCreate(m_optix_ctx, &hitgroup_desc,
                                1,                 // number of program groups
                                &program_options,  // program options
                                msg_log, &sizeof_msg_log, &hitgroup_group),
        fmtformat("Creating 'hitgroup' program group: {}", msg_log));

    // Retrieve the compiled shadeops PTX
    const char* shadeops_ptx = nullptr;
    shadingsys->getattribute("shadeops_cuda_ptx", OSL::TypeDesc::PTR,
                             &shadeops_ptx);

    int shadeops_ptx_size = 0;
    shadingsys->getattribute("shadeops_cuda_ptx_size", OSL::TypeDesc::INT,
                             &shadeops_ptx_size);

    if (shadeops_ptx == nullptr || shadeops_ptx_size == 0) {
        errhandler().severefmt(
            "Could not retrieve PTX for the shadeops library");
        return false;
    }

    // Create the shadeops library program group
    OptixModule shadeops_module;
    sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK_MSG(optixModuleCreateFn(m_optix_ctx, &module_compile_options,
                                        &pipeline_compile_options, shadeops_ptx,
                                        shadeops_ptx_size, msg_log,
                                        &sizeof_msg_log, &shadeops_module),
                    fmtformat("Creating module for shadeops library{}",
                              msg_log));

    // Record it so we can destroy it later
    modules.push_back(shadeops_module);

    // Load the PTX for the rend_lib
    std::string rend_libName = "rend_lib_testshade.ptx";
    std::string rend_lib_ptx = load_ptx_file(rend_libName);
    if (rend_lib_ptx.empty()) {
        errhandler().severefmt("Could not find PTX for the renderer library");
        return false;
    }

    // Create rend_lib program group
    sizeof_msg_log = sizeof(msg_log);
    OptixModule rend_lib_module;
    OPTIX_CHECK_MSG(optixModuleCreateFn(m_optix_ctx, &module_compile_options,
                                        &pipeline_compile_options,
                                        rend_lib_ptx.c_str(),
                                        rend_lib_ptx.size(), msg_log,
                                        &sizeof_msg_log, &rend_lib_module),
                    fmtformat("Creating module from PTX-file: {}", msg_log));

    // Record it so we can destroy it later
    modules.push_back(rend_lib_module);

    // Direct-callable -- built-in support functions for OSL on the device
    OptixProgramGroupDesc shadeops_desc = {};
    shadeops_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    shadeops_desc.callables.moduleDC    = shadeops_module;
    shadeops_desc.callables.entryFunctionNameDC
        = "__direct_callable__dummy_shadeops";
    shadeops_desc.callables.moduleCC            = 0;
    shadeops_desc.callables.entryFunctionNameCC = nullptr;

    OptixProgramGroup shadeops_group;
    sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK_MSG(
        optixProgramGroupCreate(m_optix_ctx, &shadeops_desc,
                                1,                 // number of program groups
                                &program_options,  // program options
                                msg_log, &sizeof_msg_log, &shadeops_group),
        fmtformat("Creating 'shadeops' program group: {}", msg_log));

    // Direct-callable -- renderer-specific support functions for OSL on the device
    OptixProgramGroupDesc rend_lib_desc = {};
    rend_lib_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    rend_lib_desc.callables.moduleDC    = rend_lib_module;
    rend_lib_desc.callables.entryFunctionNameDC
        = "__direct_callable__dummy_rend_lib";
    rend_lib_desc.callables.moduleCC            = 0;
    rend_lib_desc.callables.entryFunctionNameCC = nullptr;

    OptixProgramGroup rend_lib_group;
    sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK_MSG(
        optixProgramGroupCreate(m_optix_ctx, &rend_lib_desc,
                                1,                 // number of program groups
                                &program_options,  // program options
                                msg_log, &sizeof_msg_log, &rend_lib_group),
        fmtformat("Creating 'rend_lib' program group: {}", msg_log));

    int callables = m_fused_callable ? 1 : 2;

    // Create materials
    for (const auto& groupref : shaders()) {
        shadingsys->attribute(groupref.get(), "renderer_outputs",
                              TypeDesc(TypeDesc::STRING, outputs.size()),
                              outputs.data());

        shadingsys->optimize_group(groupref.get(), nullptr);

        if (!shadingsys->find_symbol(*groupref.get(), ustring(outputs[0]))) {
            // FIXME: This is for cases where testshade is run with 1x1 resolution
            //        Those tests may not have a Cout parameter to write to.
            if (m_xres > 1 && m_yres > 1) {
                errhandler().warningfmt(
                    "Requested output '{}', which wasn't found", outputs[0]);
            }
        }

        std::string group_name, init_name, entry_name, fused_name;
        shadingsys->getattribute(groupref.get(), "groupname", group_name);
        shadingsys->getattribute(groupref.get(), "group_init_name", init_name);
        shadingsys->getattribute(groupref.get(), "group_entry_name",
                                 entry_name);
        shadingsys->getattribute(groupref.get(), "group_fused_name",
                                 fused_name);

        // Retrieve the compiled ShaderGroup PTX
        std::string osl_ptx;
        shadingsys->getattribute(groupref.get(), "ptx_compiled_version",
                                 OSL::TypeDesc::PTR, &osl_ptx);

        if (osl_ptx.empty()) {
            errhandler().errorfmt("Failed to generate PTX for ShaderGroup {}",
                                  group_name);
            return false;
        }

        if (options.get_int("saveptx")) {
            std::string filename
                = OIIO::Strutil::fmt::format("{}_{}.ptx", group_name, mtl_id++);
            OIIO::ofstream out;
            OIIO::Filesystem::open(out, filename);
            out << osl_ptx;
        }

        OptixModule optix_module;

        // Create Programs from the init and group_entry functions,
        // and set the OSL functions as Callable Programs so that they
        // can be executed by the closest hit program in the wrapper
        sizeof_msg_log = sizeof(msg_log);
        OPTIX_CHECK_MSG(optixModuleCreateFn(m_optix_ctx,
                                            &module_compile_options,
                                            &pipeline_compile_options,
                                            osl_ptx.c_str(), osl_ptx.size(),
                                            msg_log, &sizeof_msg_log,
                                            &optix_module),
                        fmtformat("Creating Module from PTX-file {}", msg_log));

        modules.push_back(optix_module);


        // Create shader program groups (for direct callables)
        OptixProgramGroupOptions program_options = {};
        OptixProgramGroupDesc pgDesc[2]          = {};

        if (m_fused_callable) {
            pgDesc[0].kind               = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            pgDesc[0].callables.moduleDC = optix_module;
            pgDesc[0].callables.entryFunctionNameDC = fused_name.c_str();
            pgDesc[0].callables.moduleCC            = 0;
            pgDesc[0].callables.entryFunctionNameCC = nullptr;
        } else {
            pgDesc[0].kind               = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            pgDesc[0].callables.moduleDC = optix_module;
            pgDesc[0].callables.entryFunctionNameDC = init_name.c_str();
            pgDesc[0].callables.moduleCC            = 0;
            pgDesc[0].callables.entryFunctionNameCC = nullptr;
            pgDesc[1].kind               = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
            pgDesc[1].callables.moduleDC = optix_module;
            pgDesc[1].callables.entryFunctionNameDC = entry_name.c_str();
            pgDesc[1].callables.moduleCC            = 0;
            pgDesc[1].callables.entryFunctionNameCC = nullptr;
        }
        program_groups.resize(program_groups.size() + callables);
        void* interactive_params = nullptr;
        shadingsys->getattribute(groupref.get(), "device_interactive_params",
                                 TypeDesc::PTR, &interactive_params);
        material_interactive_params.push_back(interactive_params);

        sizeof_msg_log = sizeof(msg_log);
        OPTIX_CHECK_MSG(optixProgramGroupCreate(
                            m_optix_ctx, &pgDesc[0],
                            callables,         // number of program groups
                            &program_options,  // program options
                            msg_log, &sizeof_msg_log,
                            &program_groups[program_groups.size() - callables]),
                        fmtformat("Creating 'shader' group for group {}: {}",
                                  group_name, msg_log));
    }

    OptixPipelineLinkOptions pipeline_link_options;
    pipeline_link_options.maxTraceDepth = 1;
#if (OPTIX_VERSION < 70700)
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
#if (OPTIX_VERSION < 70100)
    pipeline_link_options.overrideUsesMotionBlur = false;
#endif

    // Set up OptiX pipeline
    std::vector<OptixProgramGroup> final_groups = {
        shadeops_group,        rend_lib_group,
        raygen_group,          miss_group,
        hitgroup_group,        setglobals_raygen_group,
        setglobals_miss_group,
    };
    if (m_fused_callable) {
        final_groups.push_back(program_groups[0]);  // fused
    } else {
        final_groups.push_back(program_groups[0]);  // init
        final_groups.push_back(program_groups[1]);  // entry
    }

    sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK_MSG(optixPipelineCreate(m_optix_ctx, &pipeline_compile_options,
                                        &pipeline_link_options,
                                        final_groups.data(),
                                        int(final_groups.size()), msg_log,
                                        &sizeof_msg_log, &m_optix_pipeline),
                    fmtformat("Creating optix pipeline: {}", msg_log));

    // Set the pipeline stack size
    OptixStackSizes stack_sizes = {};
    for (OptixProgramGroup& program_group : final_groups) {
#if (OPTIX_VERSION < 70700)
        OPTIX_CHECK(optixUtilAccumulateStackSizes(program_group, &stack_sizes));
#else
        // OptiX 7.7+ is able to take the whole pipeline into account
        // when calculating the stack requirements.
        OPTIX_CHECK(optixUtilAccumulateStackSizes(program_group, &stack_sizes,
                                                  m_optix_pipeline));
#endif
    }

    uint32_t max_trace_depth = 1;
    uint32_t max_cc_depth    = 1;
    uint32_t max_dc_depth    = 1;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes, max_trace_depth, max_cc_depth, max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state, &continuation_stack_size));

#if (OPTIX_VERSION < 70700)
    // NB: Older versions of OptiX are unable to compute the stack requirements
    //     for extern functions (e.g., the shadeops functions), so we need to
    //     pad the direct callable stack size to accommodate these functions.
    direct_callable_stack_size_from_state += 512;
#endif

    const uint32_t max_traversal_depth = 1;
    OPTIX_CHECK(optixPipelineSetStackSize(
        m_optix_pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        max_traversal_depth));

    // Build OptiX Shader Binding Table (SBT)
    CUdeviceptr d_raygenRecord;
    CUdeviceptr d_missRecord;
    CUdeviceptr d_hitgroupRecord;
    CUdeviceptr d_callablesRecord;
    CUdeviceptr d_setglobals_raygenRecord;
    CUdeviceptr d_setglobals_missRecord;

    GenericRecord raygenRecord, missRecord, hitgroupRecord, callablesRecord[2];
    GenericRecord setglobals_raygenRecord, setglobals_missRecord;

    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_group, &raygenRecord));
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_group, &missRecord));
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_group, &hitgroupRecord));
    if (m_fused_callable) {
        OPTIX_CHECK(
            optixSbtRecordPackHeader(program_groups[0], &callablesRecord[0]));
    } else {
        OPTIX_CHECK(
            optixSbtRecordPackHeader(program_groups[0], &callablesRecord[0]));
        OPTIX_CHECK(
            optixSbtRecordPackHeader(program_groups[1], &callablesRecord[1]));
    }
    OPTIX_CHECK(optixSbtRecordPackHeader(setglobals_raygen_group,
                                         &setglobals_raygenRecord));
    OPTIX_CHECK(optixSbtRecordPackHeader(setglobals_miss_group,
                                         &setglobals_missRecord));

    raygenRecord.data            = material_interactive_params[0];
    missRecord.data              = nullptr;
    hitgroupRecord.data          = nullptr;
    callablesRecord[0].data      = nullptr;
    callablesRecord[1].data      = nullptr;
    setglobals_raygenRecord.data = nullptr;
    setglobals_missRecord.data   = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygenRecord),
                          sizeof(GenericRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missRecord),
                          sizeof(GenericRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroupRecord),
                          sizeof(GenericRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_callablesRecord),
                          callables * sizeof(GenericRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_setglobals_raygenRecord),
                          sizeof(GenericRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_setglobals_missRecord),
                          sizeof(GenericRecord)));

    m_ptrs_to_free.push_back(reinterpret_cast<void*>(d_raygenRecord));
    m_ptrs_to_free.push_back(reinterpret_cast<void*>(d_missRecord));
    m_ptrs_to_free.push_back(reinterpret_cast<void*>(d_hitgroupRecord));
    m_ptrs_to_free.push_back(reinterpret_cast<void*>(d_callablesRecord));
    m_ptrs_to_free.push_back(
        reinterpret_cast<void*>(d_setglobals_raygenRecord));
    m_ptrs_to_free.push_back(reinterpret_cast<void*>(d_setglobals_missRecord));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygenRecord),
                          &raygenRecord, sizeof(GenericRecord),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_missRecord), &missRecord,
                          sizeof(GenericRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroupRecord),
                          &hitgroupRecord, sizeof(GenericRecord),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_callablesRecord),
                          &callablesRecord[0],
                          callables * sizeof(GenericRecord),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_setglobals_raygenRecord),
                          &setglobals_raygenRecord, sizeof(GenericRecord),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_setglobals_missRecord),
                          &setglobals_missRecord, sizeof(GenericRecord),
                          cudaMemcpyHostToDevice));

    // Looks like OptixShadingTable needs to be filled out completely
    m_optix_sbt.raygenRecord                 = d_raygenRecord;
    m_optix_sbt.missRecordBase               = d_missRecord;
    m_optix_sbt.missRecordStrideInBytes      = sizeof(GenericRecord);
    m_optix_sbt.missRecordCount              = 1;
    m_optix_sbt.hitgroupRecordBase           = d_hitgroupRecord;
    m_optix_sbt.hitgroupRecordStrideInBytes  = sizeof(GenericRecord);
    m_optix_sbt.hitgroupRecordCount          = 1;
    m_optix_sbt.callablesRecordBase          = d_callablesRecord;
    m_optix_sbt.callablesRecordStrideInBytes = sizeof(GenericRecord);
    m_optix_sbt.callablesRecordCount         = callables;

    // Shader binding table for SetGlobals stage
    m_setglobals_optix_sbt                         = {};
    m_setglobals_optix_sbt.raygenRecord            = d_setglobals_raygenRecord;
    m_setglobals_optix_sbt.missRecordBase          = d_setglobals_missRecord;
    m_setglobals_optix_sbt.missRecordStrideInBytes = sizeof(GenericRecord);
    m_setglobals_optix_sbt.missRecordCount         = 1;
    return true;
}



bool
OptixGridRenderer::finalize_scene()
{
    make_optix_materials();
    return true;
}



/// Return true if the texture handle (previously returned by
/// get_texture_handle()) is a valid texture that can be subsequently
/// read or sampled.
bool
OptixGridRenderer::good(TextureHandle* handle OSL_MAYBE_UNUSED)
{
    return handle != nullptr;
}



/// Given the name of a texture, return an opaque handle that can be
/// used with texture calls to avoid the name lookups.
RendererServices::TextureHandle*
OptixGridRenderer::get_texture_handle(ustring filename,
                                      ShadingContext* /*shading_context*/,
                                      const TextureOpt* /*options*/)
{
    auto itr = m_samplers.find(filename);
    if (itr == m_samplers.end()) {
        // Open image
        OIIO::ImageBuf image;
        if (!image.init_spec(filename, 0, 0)) {
            errhandler().errorfmt("Could not load: {} (hash {})", filename,
                                  filename);
            return (TextureHandle*)nullptr;
        }

        OIIO::ROI roi = OIIO::get_roi_full(image.spec());
        int32_t width = roi.width(), height = roi.height();
        std::vector<float> pixels(width * height * 4);

        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                image.getpixel(i, j, 0, &pixels[((j * width) + i) * 4 + 0]);
            }
        }
        cudaResourceDesc res_desc = {};

        // hard-code textures to 4 channels
        int32_t pitch = width * 4 * sizeof(float);
        cudaChannelFormatDesc channel_desc
            = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

        cudaArray_t pixelArray;
        CUDA_CHECK(cudaMallocArray(&pixelArray, &channel_desc, width, height));

        m_ptrs_to_free.push_back(reinterpret_cast<void*>(pixelArray));

        CUDA_CHECK(cudaMemcpy2DToArray(pixelArray,
                                       /* offset */ 0, 0, pixels.data(), pitch,
                                       pitch, height, cudaMemcpyHostToDevice));

        res_desc.resType         = cudaResourceTypeArray;
        res_desc.res.array.array = pixelArray;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0]  = cudaAddressModeWrap;
        tex_desc.addressMode[1]  = cudaAddressModeWrap;
        tex_desc.filterMode      = cudaFilterModeLinear;
        tex_desc.readMode
            = cudaReadModeElementType;  //cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords    = 1;
        tex_desc.maxAnisotropy       = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode    = cudaFilterModePoint;
        tex_desc.borderColor[0]      = 1.0f;
        tex_desc.sRGB                = 0;

        // Create texture object
        cudaTextureObject_t cuda_tex = 0;
        CUDA_CHECK(
            cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
        itr = m_samplers
                  .emplace(std::move(filename.hash()), std::move(cuda_tex))
                  .first;
    }
    return reinterpret_cast<RendererServices::TextureHandle*>(itr->second);
}



void
OptixGridRenderer::prepare_render()
{
    // Set up the OptiX Context
    init_optix_context(m_xres, m_yres);

    // Set up the OptiX scene graph
    finalize_scene();
}



void
OptixGridRenderer::warmup()
{
    // Perform a tiny launch to warm up the OptiX context
    OPTIX_CHECK(optixLaunch(m_optix_pipeline, m_cuda_stream, d_launch_params,
                            sizeof(RenderParams), &m_optix_sbt, 0, 0, 1));
    CUDA_SYNC_CHECK();
}


//extern "C" void setTestshadeGlobals(float h_invw, float h_invh, CUdeviceptr d_output_buffer, bool h_flipv);

void
OptixGridRenderer::render(int xres OSL_MAYBE_UNUSED, int yres OSL_MAYBE_UNUSED)
{
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output_buffer),
                          xres * yres * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_launch_params),
                          sizeof(RenderParams)));

    m_ptrs_to_free.push_back(reinterpret_cast<void*>(d_output_buffer));
    m_ptrs_to_free.push_back(reinterpret_cast<void*>(d_launch_params));

    m_xres = xres;
    m_yres = yres;

    RenderParams params;
    params.invw  = 1.0f / m_xres;
    params.invh  = 1.0f / m_yres;
    params.flipv = false; /* I don't see flipv being initialized anywhere */
    params.output_buffer           = d_output_buffer;
    params.osl_printf_buffer_start = d_osl_printf_buffer;
    // maybe send buffer size to CUDA instead of the buffer 'end'
    params.osl_printf_buffer_end = d_osl_printf_buffer + OSL_PRINTF_BUFFER_SIZE;
    params.color_system          = d_color_system;
    params.test_str_1            = test_str_1;
    params.test_str_2            = test_str_2;
    params.object2common         = d_object2common;
    params.shader2common         = d_shader2common;
    params.num_named_xforms      = m_num_named_xforms;
    params.xform_name_buffer     = d_xform_name_buffer;
    params.xform_buffer          = d_xform_buffer;
    params.fused_callable        = m_fused_callable;

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_launch_params), &params,
                          sizeof(RenderParams), cudaMemcpyHostToDevice));

    // Set up global variables
    OPTIX_CHECK(optixLaunch(m_optix_pipeline, m_cuda_stream, d_launch_params,
                            sizeof(RenderParams), &m_setglobals_optix_sbt, 1, 1,
                            1));
    CUDA_SYNC_CHECK();

    // Launch real render
    OPTIX_CHECK(optixLaunch(m_optix_pipeline, m_cuda_stream, d_launch_params,
                            sizeof(RenderParams), &m_optix_sbt, xres, yres, 1));
    CUDA_SYNC_CHECK();

    //
    //  Let's print some basic stuff
    //
    std::vector<uint8_t> printf_buffer(OSL_PRINTF_BUFFER_SIZE);
    CUDA_CHECK(cudaMemcpy(printf_buffer.data(),
                          reinterpret_cast<void*>(d_osl_printf_buffer),
                          OSL_PRINTF_BUFFER_SIZE, cudaMemcpyDeviceToHost));

    processPrintfBuffer(printf_buffer.data(), OSL_PRINTF_BUFFER_SIZE);
}



void
OptixGridRenderer::processPrintfBuffer(void* buffer_data, size_t buffer_size)
{
    const uint8_t* ptr = reinterpret_cast<uint8_t*>(buffer_data);
    // process until
    std::string fmt_string;
    size_t total_read = 0;
    while (total_read < buffer_size) {
        size_t src = 0;
        // set max size of each output string
        const size_t BufferSize = 4096;
        char buffer[BufferSize];
        size_t dst = 0;
        // get hash of the format string
        uint64_t fmt_str_hash = *reinterpret_cast<const uint64_t*>(&ptr[src]);
        src += sizeof(uint64_t);
        // get sizeof the argument stack
        uint64_t args_size = *reinterpret_cast<const uint64_t*>(&ptr[src]);
        src += sizeof(size_t);
        uint64_t next_args = src + args_size;

        // have we reached the end?
        if (fmt_str_hash == 0)
            break;
        const char* format = ustring::from_hash(fmt_str_hash).c_str();
        OSL_ASSERT(format != nullptr
                   && "The string should have been a valid ustring");
        const size_t len = strlen(format);

        for (size_t j = 0; j < len; j++) {
            // If we encounter a '%', then we'll copy the format string to 'fmt_string'
            // and provide that to printf() directly along with a pointer to the argument
            // we're interested in printing.
            if (format[j] == '%') {
                fmt_string            = "%";
                bool format_end_found = false;
                for (size_t i = 0; !format_end_found; i++) {
                    j++;
                    fmt_string += format[j];
                    switch (format[j]) {
                    case '%':
                        // seems like a silly to print a '%', but it keeps the logic parallel with the other cases
                        dst += snprintf(&buffer[dst], BufferSize - dst, "%s",
                                        fmt_string.c_str());
                        format_end_found = true;
                        break;
                    case 'd':
                    case 'i':
                    case 'o':
                    case 'x':
                        dst += snprintf(&buffer[dst], BufferSize - dst,
                                        fmt_string.c_str(),
                                        *reinterpret_cast<const int*>(
                                            &ptr[src]));
                        src += sizeof(int);
                        format_end_found = true;
                        break;
                    case 'f':
                    case 'g':
                    case 'e':
                        // TODO:  For OptiX llvm_gen_printf() aligns doubles on sizeof(double) boundaries -- since we're not
                        // printing from the device anymore, maybe we don't need this alignment?
                        src = (src + sizeof(double) - 1)
                              & ~(sizeof(double) - 1);
                        dst += snprintf(&buffer[dst], BufferSize - dst,
                                        fmt_string.c_str(),
                                        *reinterpret_cast<const double*>(
                                            &ptr[src]));
                        src += sizeof(double);
                        format_end_found = true;
                        break;
                    case 's':
                        src = (src + sizeof(uint64_t) - 1)
                              & ~(sizeof(uint64_t) - 1);
                        uint64_t str_hash = *reinterpret_cast<const uint64_t*>(
                            &ptr[src]);
                        ustring str = ustring::from_hash(str_hash);
                        dst += snprintf(&buffer[dst], BufferSize - dst,
                                        fmt_string.c_str(), str.c_str());
                        src += sizeof(uint64_t);
                        format_end_found = true;
                        break;

                        break;
                    }
                }
            } else {
                buffer[dst++] = format[j];
            }
        }
        // realign
        ptr = ptr + next_args;
        total_read += next_args;

        buffer[dst++] = '\0';
        print("{}", buffer);
    }
}



void
OptixGridRenderer::finalize_pixel_buffer()
{
    std::string buffer_name = "output_buffer";
    std::vector<float> tmp_buff(m_xres * m_yres * 3);
    CUDA_CHECK(cudaMemcpy(tmp_buff.data(),
                          reinterpret_cast<void*>(d_output_buffer),
                          m_xres * m_yres * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    OIIO::ImageBuf* buf = outputbuf(0);
    if (buf)
        buf->set_pixels(OIIO::ROI::All(), OIIO::TypeFloat, tmp_buff.data());
}



void
OptixGridRenderer::clear()
{
    shaders().clear();
    if (m_optix_ctx) {
        OPTIX_CHECK(optixDeviceContextDestroy(m_optix_ctx));
        m_optix_ctx = 0;
    }
}



void
OptixGridRenderer::set_transforms(const OSL::Matrix44& object2common,
                                  const OSL::Matrix44& shader2common)
{
    m_object2common = object2common;
    m_shader2common = shader2common;
}



void
OptixGridRenderer::register_named_transforms()
{
    std::vector<uint64_t> xform_name_buffer;
    std::vector<OSL::Matrix44> xform_buffer;

    // Gather:
    //   1) All of the named transforms
    //   2) The "string" value associated with the transform name, which is
    //      actually the ustring hash of the transform name.
    for (const auto& item : m_named_xforms) {
        const uint64_t addr = item.first.hash();
        xform_name_buffer.push_back(addr);
        xform_buffer.push_back(*item.second);
    }

    // Push the names and transforms to the device
    size_t sz = sizeof(uint64_t) * xform_name_buffer.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_xform_name_buffer), sz));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_xform_name_buffer),
                          xform_name_buffer.data(), sz,
                          cudaMemcpyHostToDevice));
    m_ptrs_to_free.push_back(reinterpret_cast<void*>(d_xform_name_buffer));

    sz = sizeof(OSL::Matrix44) * xform_buffer.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_xform_buffer), sz));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_xform_buffer),
                          xform_buffer.data(), sz, cudaMemcpyHostToDevice));
    m_ptrs_to_free.push_back(reinterpret_cast<void*>(d_xform_buffer));

    m_num_named_xforms = xform_name_buffer.size();
}

void
OptixGridRenderer::register_inline_functions()
{
    // clang-format off

    // Depending on the inlining options and optimization level, some functions
    // might not be inlined even when it would be beneficial to do so. We can
    // register such functions with the ShadingSystem to ensure that they are
    // inlined regardless of the other inlining options or the optimization
    // level.
    //
    // Conversely, there are some functions which should rarely be inlined. If that
    // is known in advance, we can register those functions with the ShadingSystem
    // so they can be excluded before running the ShaderGroup optimization, which
    // can help speed up the optimization and JIT stages.
    //
    // The default behavior of the optimizer should be sufficient for most
    // cases, and the inline/noinline thresholds available through the
    // ShadingSystem attributes enable some degree of fine tuning. This
    // mechanism has been added to offer a finer degree of control
    //
    // Please refer to doc/app_integration/OptiX-Inlining-Options.md for more
    // details about the inlining options.

    // These functions are all 5 instructions or less in the PTX, with most of
    // those instructions related to reading the parameters and writing out the
    // return value. It would be beneficial to inline them in all cases. We can
    // register them to ensure that they are inlined regardless of the other
    // compile options.
    shadingsys->register_inline_function(ustring("osl_abs_ff"));
    shadingsys->register_inline_function(ustring("osl_abs_ii"));
    shadingsys->register_inline_function(ustring("osl_ceil_ff"));
    shadingsys->register_inline_function(ustring("osl_cos_ff"));
    shadingsys->register_inline_function(ustring("osl_exp2_ff"));
    shadingsys->register_inline_function(ustring("osl_exp_ff"));
    shadingsys->register_inline_function(ustring("osl_fabs_ff"));
    shadingsys->register_inline_function(ustring("osl_fabs_ii"));
    shadingsys->register_inline_function(ustring("osl_floor_ff"));
    shadingsys->register_inline_function(ustring("osl_get_texture_options"));
    shadingsys->register_inline_function(ustring("osl_getchar_isi"));
    shadingsys->register_inline_function(ustring("osl_hash_is"));
    shadingsys->register_inline_function(ustring("osl_log10_ff"));
    shadingsys->register_inline_function(ustring("osl_log2_ff"));
    shadingsys->register_inline_function(ustring("osl_log_ff"));
    shadingsys->register_inline_function(ustring("osl_noiseparams_set_anisotropic"));
    shadingsys->register_inline_function(ustring("osl_noiseparams_set_bandwidth"));
    shadingsys->register_inline_function(ustring("osl_noiseparams_set_do_filter"));
    shadingsys->register_inline_function(ustring("osl_noiseparams_set_impulses"));
    shadingsys->register_inline_function(ustring("osl_nullnoise_ff"));
    shadingsys->register_inline_function(ustring("osl_nullnoise_fff"));
    shadingsys->register_inline_function(ustring("osl_nullnoise_fv"));
    shadingsys->register_inline_function(ustring("osl_nullnoise_fvf"));
    shadingsys->register_inline_function(ustring("osl_sin_ff"));
    shadingsys->register_inline_function(ustring("osl_strlen_is"));
    shadingsys->register_inline_function(ustring("osl_texture_set_interp_code"));
    shadingsys->register_inline_function(ustring("osl_texture_set_stwrap_code"));
    shadingsys->register_inline_function(ustring("osl_trunc_ff"));
    shadingsys->register_inline_function(ustring("osl_unullnoise_ff"));
    shadingsys->register_inline_function(ustring("osl_unullnoise_fff"));
    shadingsys->register_inline_function(ustring("osl_unullnoise_fv"));
    shadingsys->register_inline_function(ustring("osl_unullnoise_fvf"));

    // These large functions are unlikely to ever been inlined. In such cases,
    // we may be able to speed up ShaderGroup compilation by registering these
    // functions as "noinline" so they can be excluded from the ShaderGroup
    // module prior to optimization/JIT.
    shadingsys->register_noinline_function(ustring("osl_gabornoise_dfdfdf"));
    shadingsys->register_noinline_function(ustring("osl_gabornoise_dfdv"));
    shadingsys->register_noinline_function(ustring("osl_gabornoise_dfdvdf"));
    shadingsys->register_noinline_function(ustring("osl_gaborpnoise_dfdfdfff"));
    shadingsys->register_noinline_function(ustring("osl_gaborpnoise_dfdvdfvf"));
    shadingsys->register_noinline_function(ustring("osl_gaborpnoise_dfdvv"));
    shadingsys->register_noinline_function(ustring("osl_genericnoise_dfdvdf"));
    shadingsys->register_noinline_function(ustring("osl_genericpnoise_dfdvv"));
    shadingsys->register_noinline_function(ustring("osl_get_inverse_matrix"));
    shadingsys->register_noinline_function(ustring("osl_noise_dfdfdf"));
    shadingsys->register_noinline_function(ustring("osl_noise_dfdff"));
    shadingsys->register_noinline_function(ustring("osl_noise_dffdf"));
    shadingsys->register_noinline_function(ustring("osl_noise_fv"));
    shadingsys->register_noinline_function(ustring("osl_noise_vff"));
    shadingsys->register_noinline_function(ustring("osl_pnoise_dfdfdfff"));
    shadingsys->register_noinline_function(ustring("osl_pnoise_dfdffff"));
    shadingsys->register_noinline_function(ustring("osl_pnoise_dffdfff"));
    shadingsys->register_noinline_function(ustring("osl_pnoise_fffff"));
    shadingsys->register_noinline_function(ustring("osl_pnoise_vffff"));
    shadingsys->register_noinline_function(ustring("osl_psnoise_dfdfdfff"));
    shadingsys->register_noinline_function(ustring("osl_psnoise_dfdffff"));
    shadingsys->register_noinline_function(ustring("osl_psnoise_dffdfff"));
    shadingsys->register_noinline_function(ustring("osl_psnoise_fffff"));
    shadingsys->register_noinline_function(ustring("osl_psnoise_vffff"));
    shadingsys->register_noinline_function(ustring("osl_simplexnoise_dvdf"));
    shadingsys->register_noinline_function(ustring("osl_simplexnoise_vf"));
    shadingsys->register_noinline_function(ustring("osl_simplexnoise_vff"));
    shadingsys->register_noinline_function(ustring("osl_snoise_dfdfdf"));
    shadingsys->register_noinline_function(ustring("osl_snoise_dfdff"));
    shadingsys->register_noinline_function(ustring("osl_snoise_dffdf"));
    shadingsys->register_noinline_function(ustring("osl_snoise_fv"));
    shadingsys->register_noinline_function(ustring("osl_snoise_vff"));
    shadingsys->register_noinline_function(ustring("osl_transform_triple"));
    shadingsys->register_noinline_function(ustring("osl_transformn_dvmdv"));
    shadingsys->register_noinline_function(ustring("osl_usimplexnoise_dvdf"));
    shadingsys->register_noinline_function(ustring("osl_usimplexnoise_vf"));
    shadingsys->register_noinline_function(ustring("osl_usimplexnoise_vff"));

    // It's also possible to unregister functions to restore the default
    // inlining behavior when needed.
    shadingsys->unregister_inline_function(ustring("osl_get_texture_options"));
    shadingsys->unregister_noinline_function(ustring("osl_get_inverse_matrix"));

    // clang-format on
}

OSL_NAMESPACE_EXIT
