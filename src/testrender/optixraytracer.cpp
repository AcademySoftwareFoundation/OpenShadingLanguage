// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <vector>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>

#include <OSL/oslconfig.h>

#include "optixraytracer.h"

#include "render_params.h"

#include <cuda.h>
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
    //    std::cerr << "[ ** LOGCALLBACK** " << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}



OptixRaytracer::OptixRaytracer()
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
}



OptixRaytracer::~OptixRaytracer()
{
    if (m_optix_ctx)
        OPTIX_CHECK(optixDeviceContextDestroy(m_optix_ctx));
}



void*
OptixRaytracer::device_alloc(size_t size)
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
OptixRaytracer::device_free(void* ptr)
{
    cudaError_t res = cudaFree(ptr);
    if (res != cudaSuccess) {
        errhandler().errorfmt("cudaFree() failed with error: {}\n",
                              cudaGetErrorString(res));
    }
}


void*
OptixRaytracer::copy_to_device(void* dst_device, const void* src_host,
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
OptixRaytracer::load_ptx_file(string_view filename)
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



bool
OptixRaytracer::init_optix_context(int xres OSL_MAYBE_UNUSED,
                                   int yres OSL_MAYBE_UNUSED)
{
    if (!options.get_int("no_rend_lib_bitcode")) {
        shadingsys->attribute("lib_bitcode",
                              { OSL::TypeDesc::UINT8,
                                rend_lib_llvm_compiled_ops_size },
                              rend_lib_llvm_compiled_ops_block);
    }
    return true;
}



bool
OptixRaytracer::synch_attributes()
{
    // FIXME -- this is for testing only
    // Make some device strings to test userdata parameters
    ustring userdata_str1("ud_str_1");
    ustring userdata_str2("userdata string");

    // Store the user-data
    test_str_1 = userdata_str1.hash();
    test_str_2 = userdata_str2.hash();

    // Set the maximum groupdata buffer allocation size
    shadingsys->attribute("max_optix_groupdata_alloc", 1024);

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
                              podDataSize + sizeof(DeviceString) * numStrings));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_color_system), colorSys,
                              podDataSize, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_osl_printf_buffer),
                              OSL_PRINTF_BUFFER_SIZE));
        CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(d_osl_printf_buffer), 0,
                              OSL_PRINTF_BUFFER_SIZE));

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
OptixRaytracer::load_optix_module(
    const char* filename,
    const OptixModuleCompileOptions* module_compile_options,
    const OptixPipelineCompileOptions* pipeline_compile_options,
    OptixModule* program_module)
{
    char msg_log[8192];

    // Load the renderer CUDA source and generate PTX for it
    std::string program_ptx = load_ptx_file(filename);
    if (program_ptx.empty()) {
        errhandler().severefmt("Could not find PTX file:  {}", filename);
        return false;
    }

    size_t sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK_MSG(optixModuleCreateFn(m_optix_ctx, module_compile_options,
                                        pipeline_compile_options,
                                        program_ptx.c_str(), program_ptx.size(),
                                        msg_log, &sizeof_msg_log,
                                        program_module),
                    fmtformat("Creating Module from PTX-file {}", msg_log));
    return true;
}



bool
OptixRaytracer::create_optix_pg(const OptixProgramGroupDesc* pg_desc,
                                const int num_pg,
                                OptixProgramGroupOptions* program_options,
                                OptixProgramGroup* pg)
{
    char msg_log[8192];
    size_t sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK_MSG(optixProgramGroupCreate(m_optix_ctx, pg_desc, num_pg,
                                            program_options, msg_log,
                                            &sizeof_msg_log, pg),
                    fmtformat("Creating program group: {}", msg_log));

    return true;
}



bool
OptixRaytracer::make_optix_materials()
{
    // Stand-in: names of shader outputs to preserve
    std::vector<const char*> outputs { "Cout" };

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
    pipeline_compile_options.numPayloadValues   = 3;
    pipeline_compile_options.numAttributeValues = 3;
    pipeline_compile_options.exceptionFlags
        = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "render_params";

    // Create 'raygen' program

    // Load the renderer CUDA source and generate PTX for it
    OptixModule program_module;
    load_optix_module("optix_raytracer.ptx", &module_compile_options,
                      &pipeline_compile_options, &program_module);

    // Record it so we can destroy it later
    modules.push_back(program_module);

    OptixModule quad_module;
    load_optix_module("quad.ptx", &module_compile_options,
                      &pipeline_compile_options, &quad_module);

    OptixModule sphere_module;
    load_optix_module("sphere.ptx", &module_compile_options,
                      &pipeline_compile_options, &sphere_module);

    OptixModule wrapper_module;
    load_optix_module("wrapper.ptx", &module_compile_options,
                      &pipeline_compile_options, &wrapper_module);

    OptixModule rend_lib_module;
    load_optix_module("rend_lib_testrender.ptx", &module_compile_options,
                      &pipeline_compile_options, &rend_lib_module);

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
                    fmtformat("Creating module for shadeops library: {}",
                              msg_log));
    modules.push_back(shadeops_module);

    OptixProgramGroupOptions program_options = {};
    std::vector<OptixProgramGroup> shader_groups;

    // Raygen group
    OptixProgramGroupDesc raygen_desc    = {};
    raygen_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module            = program_module;
    raygen_desc.raygen.entryFunctionName = "__raygen__";

    OptixProgramGroup raygen_group;
    create_optix_pg(&raygen_desc, 1, &program_options, &raygen_group);

    // Set Globals Raygen group
    OptixProgramGroupDesc setglobals_raygen_desc = {};
    setglobals_raygen_desc.kind          = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    setglobals_raygen_desc.raygen.module = program_module;
    setglobals_raygen_desc.raygen.entryFunctionName = "__raygen__setglobals";

    OptixProgramGroup setglobals_raygen_group;
    sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK_MSG(
        optixProgramGroupCreate(m_optix_ctx, &setglobals_raygen_desc,
                                1,                 // number of program groups
                                &program_options,  // program options
                                msg_log, &sizeof_msg_log,
                                &setglobals_raygen_group),
        fmtformat("Creating set-globals 'ray-gen' program group: {}", msg_log));

    // Miss group
    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module
        = program_module;  // raygen file/module contains miss program
    miss_desc.miss.entryFunctionName = "__miss__";

    OptixProgramGroup miss_group;
    create_optix_pg(&miss_desc, 1, &program_options, &miss_group);

    // Set Globals Miss group
    OptixProgramGroupDesc setglobals_miss_desc  = {};
    setglobals_miss_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    setglobals_miss_desc.miss.module            = program_module;
    setglobals_miss_desc.miss.entryFunctionName = "__miss__setglobals";
    OptixProgramGroup setglobals_miss_group;
    create_optix_pg(&setglobals_miss_desc, 1, &program_options,
                    &setglobals_miss_group);

    // Hitgroup -- quads
    OptixProgramGroupDesc quad_hitgroup_desc = {};
    quad_hitgroup_desc.kind              = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    quad_hitgroup_desc.hitgroup.moduleCH = wrapper_module;
    quad_hitgroup_desc.hitgroup.entryFunctionNameCH
        = "__closesthit__closest_hit_osl";
    quad_hitgroup_desc.hitgroup.moduleAH = wrapper_module;
    quad_hitgroup_desc.hitgroup.entryFunctionNameAH = "__anyhit__any_hit_shadow";
    quad_hitgroup_desc.hitgroup.moduleIS            = quad_module;
    quad_hitgroup_desc.hitgroup.entryFunctionNameIS = "__intersection__quad";
    OptixProgramGroup quad_hitgroup;
    create_optix_pg(&quad_hitgroup_desc, 1, &program_options, &quad_hitgroup);

    // Direct-callable -- renderer-specific support functions for OSL on the device
    OptixProgramGroupDesc rend_lib_desc = {};
    rend_lib_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    rend_lib_desc.callables.moduleDC    = rend_lib_module;
    rend_lib_desc.callables.entryFunctionNameDC
        = "__direct_callable__dummy_rend_lib";
    rend_lib_desc.callables.moduleCC            = 0;
    rend_lib_desc.callables.entryFunctionNameCC = nullptr;
    OptixProgramGroup rend_lib_group;
    create_optix_pg(&rend_lib_desc, 1, &program_options, &rend_lib_group);

    // Direct-callable -- built-in support functions for OSL on the device
    OptixProgramGroupDesc shadeops_desc = {};
    shadeops_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    shadeops_desc.callables.moduleDC    = shadeops_module;
    shadeops_desc.callables.entryFunctionNameDC
        = "__direct_callable__dummy_shadeops";
    shadeops_desc.callables.moduleCC            = 0;
    shadeops_desc.callables.entryFunctionNameCC = nullptr;
    OptixProgramGroup shadeops_group;
    create_optix_pg(&shadeops_desc, 1, &program_options, &shadeops_group);

    // Direct-callable -- fills in ShaderGlobals for Quads
    OptixProgramGroupDesc quad_fillSG_desc = {};
    quad_fillSG_desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    quad_fillSG_desc.callables.moduleDC    = quad_module;
    quad_fillSG_desc.callables.entryFunctionNameDC
        = "__direct_callable__quad_shaderglobals";
    quad_fillSG_desc.callables.moduleCC            = 0;
    quad_fillSG_desc.callables.entryFunctionNameCC = nullptr;
    OptixProgramGroup quad_fillSG_dc;
    create_optix_pg(&quad_fillSG_desc, 1, &program_options, &quad_fillSG_dc);

    // Hitgroup -- sphere
    OptixProgramGroupDesc sphere_hitgroup_desc = {};
    sphere_hitgroup_desc.kind              = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    sphere_hitgroup_desc.hitgroup.moduleCH = wrapper_module;
    sphere_hitgroup_desc.hitgroup.entryFunctionNameCH
        = "__closesthit__closest_hit_osl";
    sphere_hitgroup_desc.hitgroup.moduleAH = wrapper_module;
    sphere_hitgroup_desc.hitgroup.entryFunctionNameAH
        = "__anyhit__any_hit_shadow";
    sphere_hitgroup_desc.hitgroup.moduleIS = sphere_module;
    sphere_hitgroup_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    OptixProgramGroup sphere_hitgroup;
    create_optix_pg(&sphere_hitgroup_desc, 1, &program_options,
                    &sphere_hitgroup);

    // Direct-callable -- fills in ShaderGlobals for Sphere
    OptixProgramGroupDesc sphere_fillSG_desc = {};
    sphere_fillSG_desc.kind               = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    sphere_fillSG_desc.callables.moduleDC = sphere_module;
    sphere_fillSG_desc.callables.entryFunctionNameDC
        = "__direct_callable__sphere_shaderglobals";
    sphere_fillSG_desc.callables.moduleCC            = 0;
    sphere_fillSG_desc.callables.entryFunctionNameCC = nullptr;
    OptixProgramGroup sphere_fillSG_dc;
    create_optix_pg(&sphere_fillSG_desc, 1, &program_options,
                    &sphere_fillSG_dc);

    // Create materials
    int mtl_id = 0;
    std::vector<void*> material_interactive_params;
    for (const auto& groupref : shaders()) {
        std::string group_name, fused_name;
        shadingsys->getattribute(groupref.get(), "groupname", group_name);
        shadingsys->getattribute(groupref.get(), "group_fused_name",
                                 fused_name);

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
            std::string filename = fmtformat("{}_{}.ptx", group_name, mtl_id++);
            OIIO::Filesystem::write_text_file(filename, osl_ptx);
        }

        void* interactive_params = nullptr;
        shadingsys->getattribute(groupref.get(), "device_interactive_params",
                                 TypeDesc::PTR, &interactive_params);
        material_interactive_params.push_back(interactive_params);

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
                        fmtformat("Creating module for PTX group {}: {}",
                                  group_name, msg_log));
        modules.push_back(optix_module);

        // Create program groups (for direct callables)
        OptixProgramGroupDesc pgDesc[1] = {};
        pgDesc[0].kind                  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        pgDesc[0].callables.moduleDC    = optix_module;
        pgDesc[0].callables.entryFunctionNameDC = fused_name.c_str();
        pgDesc[0].callables.moduleCC            = 0;
        pgDesc[0].callables.entryFunctionNameCC = nullptr;

        shader_groups.resize(shader_groups.size() + 1);
        sizeof_msg_log = sizeof(msg_log);
        OPTIX_CHECK_MSG(
            optixProgramGroupCreate(m_optix_ctx, &pgDesc[0], 1,
                                    &program_options, msg_log, &sizeof_msg_log,
                                    &shader_groups[shader_groups.size() - 1]),
            fmtformat("Creating 'shader' group for group {}: {}", group_name,
                      msg_log));
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
    std::vector<OptixProgramGroup> final_groups = { rend_lib_group,
                                                    raygen_group, miss_group };

    if (scene.quads.size() > 0)
        final_groups.push_back(quad_hitgroup);
    if (scene.spheres.size() > 0)
        final_groups.push_back(sphere_hitgroup);

    final_groups.push_back(quad_fillSG_dc);
    final_groups.push_back(sphere_fillSG_dc);

    // append the shader groups to our "official" list of program groups
    // size_t shader_groups_start_index = final_groups.size();
    final_groups.insert(final_groups.end(), shader_groups.begin(),
                        shader_groups.end());

    // append the program group for the built-in shadeops module
    final_groups.push_back(shadeops_group);

    // append set-globals groups
    final_groups.push_back(setglobals_raygen_group);
    final_groups.push_back(setglobals_miss_group);

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

    std::vector<GenericRecord> sbt_records(final_groups.size());

    CUdeviceptr d_raygen_record;
    CUdeviceptr d_miss_record;
    CUdeviceptr d_hitgroup_records;
    CUdeviceptr d_callable_records;
    CUdeviceptr d_setglobals_raygen_record;
    CUdeviceptr d_setglobals_miss_record;

    std::vector<CUdeviceptr> d_sbt_records(final_groups.size());

    for (size_t i = 0; i < final_groups.size(); i++) {
        OPTIX_CHECK(optixSbtRecordPackHeader(final_groups[i], &sbt_records[i]));
    }

    int sbtIndex             = 3;
    const int hitRecordStart = sbtIndex;
    size_t setglobals_start  = final_groups.size() - 2;

    // Copy geometry data to appropriate SBT records
    if (scene.quads.size() > 0) {
        sbt_records[sbtIndex].data = reinterpret_cast<void*>(d_quads_list);
        sbt_records[sbtIndex].sbtGeoIndex
            = 0;  // DC index for filling in Quad ShaderGlobals
        ++sbtIndex;
    }

    if (scene.spheres.size() > 0) {
        sbt_records[sbtIndex].data = reinterpret_cast<void*>(d_spheres_list);
        sbt_records[sbtIndex].sbtGeoIndex
            = 1;  // DC index for filling in Sphere ShaderGlobals
        ++sbtIndex;
    }

    const int callableRecordStart = sbtIndex;

    // Copy geometry data to our DC (direct-callable) funcs that fill ShaderGlobals
    sbt_records[sbtIndex++].data = reinterpret_cast<void*>(d_quads_list);
    sbt_records[sbtIndex++].data = reinterpret_cast<void*>(d_spheres_list);

    const int nshaders   = int(shader_groups.size());
    const int nhitgroups = (scene.quads.size() > 0)
                           + (scene.spheres.size() > 0);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record),
                          sizeof(GenericRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record),
                          sizeof(GenericRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records),
                          nhitgroups * sizeof(GenericRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_callable_records),
                          (2 + nshaders) * sizeof(GenericRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_setglobals_raygen_record),
                          sizeof(GenericRecord)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_setglobals_miss_record),
                          sizeof(GenericRecord)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record),
                          &sbt_records[1], sizeof(GenericRecord),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_record),
                          &sbt_records[2], sizeof(GenericRecord),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records),
                          &sbt_records[hitRecordStart],
                          nhitgroups * sizeof(GenericRecord),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_callable_records),
                          &sbt_records[callableRecordStart],
                          (2 + nshaders) * sizeof(GenericRecord),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_setglobals_raygen_record),
                          &sbt_records[setglobals_start + 0],
                          sizeof(GenericRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_setglobals_miss_record),
                          &sbt_records[setglobals_start + 1],
                          sizeof(GenericRecord), cudaMemcpyHostToDevice));

    // Looks like OptixShadingTable needs to be filled out completely
    m_optix_sbt.raygenRecord                 = d_raygen_record;
    m_optix_sbt.missRecordBase               = d_miss_record;
    m_optix_sbt.missRecordStrideInBytes      = sizeof(GenericRecord);
    m_optix_sbt.missRecordCount              = 1;
    m_optix_sbt.hitgroupRecordBase           = d_hitgroup_records;
    m_optix_sbt.hitgroupRecordStrideInBytes  = sizeof(GenericRecord);
    m_optix_sbt.hitgroupRecordCount          = nhitgroups;
    m_optix_sbt.callablesRecordBase          = d_callable_records;
    m_optix_sbt.callablesRecordStrideInBytes = sizeof(GenericRecord);
    m_optix_sbt.callablesRecordCount         = 2 + nshaders;

    // Shader binding table for SetGlobals stage
    m_setglobals_optix_sbt                         = {};
    m_setglobals_optix_sbt.raygenRecord            = d_setglobals_raygen_record;
    m_setglobals_optix_sbt.missRecordBase          = d_setglobals_miss_record;
    m_setglobals_optix_sbt.missRecordStrideInBytes = sizeof(GenericRecord);
    m_setglobals_optix_sbt.missRecordCount         = 1;

    // Upload per-material interactive buffer table
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_interactive_params),
                          sizeof(void*) * material_interactive_params.size()));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_interactive_params),
                          material_interactive_params.data(),
                          sizeof(void*) * material_interactive_params.size(),
                          cudaMemcpyHostToDevice));

    // Pipeline has been created so we can clean some things up
    for (auto&& i : final_groups) {
        optixProgramGroupDestroy(i);
    }
    for (auto&& i : modules) {
        optixModuleDestroy(i);
    }
    modules.clear();

    return true;
}



bool
OptixRaytracer::finalize_scene()
{
    // Build acceleration structures
    OptixAccelBuildOptions accelOptions;
    OptixBuildInput buildInputs[2];

    memset(&accelOptions, 0, sizeof(OptixAccelBuildOptions));
    accelOptions.buildFlags            = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation             = OPTIX_BUILD_OPERATION_BUILD;
    accelOptions.motionOptions.numKeys = 0;
    memset(buildInputs, 0, sizeof(OptixBuildInput) * 2);

    // Set up quads input
    void* d_quadsAabb;
    std::vector<OptixAabb> quadsAabb;
    std::vector<QuadParams> quadsParams;
    quadsAabb.reserve(scene.quads.size());
    quadsParams.reserve(scene.quads.size());
    std::vector<int> quadShaders;
    quadShaders.reserve(scene.quads.size());
    for (const auto& quad : scene.quads) {
        OptixAabb aabb;
        quad.getBounds(aabb.minX, aabb.minY, aabb.minZ, aabb.maxX, aabb.maxY,
                       aabb.maxZ);
        quadsAabb.push_back(aabb);
        QuadParams quad_params;
        quad.setOptixVariables(&quad_params);
        quadsParams.push_back(quad_params);
    }
    // Copy Quads bounding boxes to cuda device
    CUDA_CHECK(
        cudaMalloc(&d_quadsAabb, sizeof(OptixAabb) * scene.quads.size()));
    CUDA_CHECK(cudaMemcpy(d_quadsAabb, quadsAabb.data(),
                          sizeof(OptixAabb) * scene.quads.size(),
                          cudaMemcpyHostToDevice));

    // Copy Quads to cuda device
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_quads_list),
                          sizeof(QuadParams) * scene.quads.size()));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_quads_list),
                          quadsParams.data(),
                          sizeof(QuadParams) * scene.quads.size(),
                          cudaMemcpyHostToDevice));

    // Fill in Quad shaders
    CUdeviceptr d_quadsIndexOffsetBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_quadsIndexOffsetBuffer),
                          scene.quads.size() * sizeof(int)));

    int numBuildInputs = 0;

    unsigned int quadSbtRecord;
    quadSbtRecord = OPTIX_GEOMETRY_FLAG_NONE;
    if (scene.quads.size() > 0) {
#if (OPTIX_VERSION < 70100)
        OptixBuildInputCustomPrimitiveArray& quadsInput
            = buildInputs[numBuildInputs].aabbArray;
#else
        OptixBuildInputCustomPrimitiveArray& quadsInput
            = buildInputs[numBuildInputs].customPrimitiveArray;
#endif
        buildInputs[numBuildInputs].type
            = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        quadsInput.flags         = &quadSbtRecord;
        quadsInput.aabbBuffers   = reinterpret_cast<CUdeviceptr*>(&d_quadsAabb);
        quadsInput.numPrimitives = scene.quads.size();
        quadsInput.numSbtRecords = 1;
        quadsInput.sbtIndexOffsetSizeInBytes   = sizeof(int);
        quadsInput.sbtIndexOffsetStrideInBytes = sizeof(int);
        quadsInput.sbtIndexOffsetBuffer = 0;  // d_quadsIndexOffsetBuffer;
        ++numBuildInputs;
    }

    //  Set up spheres input
    void* d_spheresAabb;
    std::vector<OptixAabb> spheresAabb;
    std::vector<SphereParams> spheresParams;
    spheresAabb.reserve(scene.spheres.size());
    spheresParams.reserve(scene.spheres.size());
    std::vector<int> sphereShaders;
    sphereShaders.reserve(scene.spheres.size());
    for (const auto& sphere : scene.spheres) {
        OptixAabb aabb;
        sphere.getBounds(aabb.minX, aabb.minY, aabb.minZ, aabb.maxX, aabb.maxY,
                         aabb.maxZ);
        spheresAabb.push_back(aabb);

        SphereParams sphere_params;
        sphere.setOptixVariables(&sphere_params);
        spheresParams.push_back(sphere_params);
    }
    // Copy Spheres bounding boxes to cuda device
    CUDA_CHECK(
        cudaMalloc(&d_spheresAabb, sizeof(OptixAabb) * scene.spheres.size()));
    CUDA_CHECK(cudaMemcpy(d_spheresAabb, spheresAabb.data(),
                          sizeof(OptixAabb) * scene.spheres.size(),
                          cudaMemcpyHostToDevice));

    // Copy Spheres to cuda device
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_spheres_list),
                          sizeof(SphereParams) * scene.spheres.size()));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_spheres_list),
                          spheresParams.data(),
                          sizeof(SphereParams) * scene.spheres.size(),
                          cudaMemcpyHostToDevice));

    // Fill in Sphere shaders
    CUdeviceptr d_spheresIndexOffsetBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_spheresIndexOffsetBuffer),
                          scene.spheres.size() * sizeof(int)));

    unsigned int sphereSbtRecord;
    sphereSbtRecord = OPTIX_GEOMETRY_FLAG_NONE;
    if (scene.spheres.size() > 0) {
#if (OPTIX_VERSION < 70100)
        OptixBuildInputCustomPrimitiveArray& spheresInput
            = buildInputs[numBuildInputs].aabbArray;
#else
        OptixBuildInputCustomPrimitiveArray& spheresInput
            = buildInputs[numBuildInputs].customPrimitiveArray;
#endif
        buildInputs[numBuildInputs].type
            = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        spheresInput.flags       = &sphereSbtRecord;
        spheresInput.aabbBuffers = reinterpret_cast<CUdeviceptr*>(
            &d_spheresAabb);
        spheresInput.numPrimitives               = scene.spheres.size();
        spheresInput.numSbtRecords               = 1;
        spheresInput.sbtIndexOffsetSizeInBytes   = sizeof(int);
        spheresInput.sbtIndexOffsetStrideInBytes = sizeof(int);
        spheresInput.sbtIndexOffsetBuffer = 0;  // d_spheresIndexOffsetBuffer;
        ++numBuildInputs;
    }

    // Compute memory usage by acceleration structures
    OptixAccelBufferSizes bufferSizes;
    optixAccelComputeMemoryUsage(m_optix_ctx, &accelOptions, buildInputs,
                                 numBuildInputs, &bufferSizes);

    void *d_output, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_output, bufferSizes.outputSizeInBytes));
    CUDA_CHECK(cudaMalloc(&d_temp, bufferSizes.tempSizeInBytes));

    // Get the bounding box for the AS
    void* d_aabb;
    CUDA_CHECK(cudaMalloc(&d_aabb, sizeof(OptixAabb)));

    OptixAccelEmitDesc property;
    property.type   = OPTIX_PROPERTY_TYPE_AABBS;
    property.result = (CUdeviceptr)d_aabb;

    OPTIX_CHECK(optixAccelBuild(
        m_optix_ctx, m_cuda_stream, &accelOptions, buildInputs, numBuildInputs,
        reinterpret_cast<CUdeviceptr>(d_temp), bufferSizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(d_output), bufferSizes.outputSizeInBytes,
        &m_travHandle, &property, 1));

    OptixAabb h_aabb;
    CUDA_CHECK(cudaMemcpy((void*)&h_aabb, reinterpret_cast<void*>(d_aabb),
                          sizeof(OptixAabb), cudaMemcpyDeviceToHost));
    cudaFree(d_aabb);

    // Sanity check the AS bounds
    // printf ("AABB min: [%0.6f, %0.6f, %0.6f], max: [%0.6f, %0.6f, %0.6f]\n",
    //         h_aabb.minX, h_aabb.minY, h_aabb.minZ,
    //         h_aabb.maxX, h_aabb.maxY, h_aabb.maxZ );

    make_optix_materials();
    return true;
}



/// Return true if the texture handle (previously returned by
/// get_texture_handle()) is a valid texture that can be subsequently
/// read or sampled.
bool
OptixRaytracer::good(TextureHandle* handle OSL_MAYBE_UNUSED)
{
    return handle != nullptr;
}



/// Given the name of a texture, return an opaque handle that can be
/// used with texture calls to avoid the name lookups.
RendererServices::TextureHandle*
OptixRaytracer::get_texture_handle(ustring filename,
                                   ShadingContext* /*shading_context*/,
                                   const TextureOpt* options)
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

        CUDA_CHECK(cudaMemcpy2DToArray(pixelArray, 0, 0, pixels.data(), pitch,
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
OptixRaytracer::prepare_render()
{
    // Set up the OptiX Context
    init_optix_context(camera.xres, camera.yres);

    // Set up the OptiX scene graph
    finalize_scene();
}



void
OptixRaytracer::warmup()
{
    // Perform a tiny launch to warm up the OptiX context
    OPTIX_CHECK(optixLaunch(m_optix_pipeline, m_cuda_stream, d_launch_params,
                            sizeof(RenderParams), &m_optix_sbt, 0, 0, 1));
    CUDA_SYNC_CHECK();
}



void
OptixRaytracer::render(int xres OSL_MAYBE_UNUSED, int yres OSL_MAYBE_UNUSED)
{
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output_buffer),
                          xres * yres * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_launch_params),
                          sizeof(RenderParams)));

    m_xres = xres;
    m_yres = yres;

    RenderParams params;
    params.eye.x                   = camera.eye.x;
    params.eye.y                   = camera.eye.y;
    params.eye.z                   = camera.eye.z;
    params.dir.x                   = camera.dir.x;
    params.dir.y                   = camera.dir.y;
    params.dir.z                   = camera.dir.z;
    params.cx.x                    = camera.cx.x;
    params.cx.y                    = camera.cx.y;
    params.cx.z                    = camera.cx.z;
    params.cy.x                    = camera.cy.x;
    params.cy.y                    = camera.cy.y;
    params.cy.z                    = camera.cy.z;
    params.invw                    = 1.0f / m_xres;
    params.invh                    = 1.0f / m_yres;
    params.interactive_params      = d_interactive_params;
    params.output_buffer           = d_output_buffer;
    params.traversal_handle        = m_travHandle;
    params.osl_printf_buffer_start = d_osl_printf_buffer;
    // maybe send buffer size to CUDA instead of the buffer 'end'
    params.osl_printf_buffer_end = d_osl_printf_buffer + OSL_PRINTF_BUFFER_SIZE;
    params.color_system          = d_color_system;
    params.test_str_1            = test_str_1;
    params.test_str_2            = test_str_2;

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
OptixRaytracer::processPrintfBuffer(void* buffer_data, size_t buffer_size)
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
            // If we encounter a '%', then we'l copy the format string to 'fmt_string'
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
                        src = (src + sizeof(double) - 1)
                              & ~(sizeof(double) - 1);
                        uint64_t str_hash = *reinterpret_cast<const uint64_t*>(
                            &ptr[src]);
                        const char* str = ustring::from_hash(str_hash).c_str();
                        OSL_ASSERT(
                            str != nullptr
                            && "The string should have been a valid ustring");
                        dst += snprintf(&buffer[dst], BufferSize - dst,
                                        fmt_string.c_str(), str);
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
        printf("%s", buffer);
    }
}



void
OptixRaytracer::finalize_pixel_buffer()
{
    std::string buffer_name = "output_buffer";
    std::vector<float> tmp_buff(m_xres * m_yres * 3);
    CUDA_CHECK(cudaMemcpy(tmp_buff.data(),
                          reinterpret_cast<void*>(d_output_buffer),
                          m_xres * m_yres * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    pixelbuf.set_pixels(OIIO::ROI::All(), OIIO::TypeFloat, tmp_buff.data());
}



void
OptixRaytracer::clear()
{
    shaders().clear();
    OPTIX_CHECK(optixDeviceContextDestroy(m_optix_ctx));
    m_optix_ctx = 0;
}

OSL_NAMESPACE_EXIT
