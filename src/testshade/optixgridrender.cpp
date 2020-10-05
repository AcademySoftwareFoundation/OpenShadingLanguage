// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include <vector>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>

#include <OSL/oslconfig.h>

#include "optixgridrender.h"
#include "render_params.h"

#ifdef OSL_USE_OPTIX
#  if (OPTIX_VERSION >= 70000)
#  include <optix_function_table_definition.h>
#  include <optix_stack_size.h>
#  include <optix_stubs.h>
#  include <cuda.h>
#  include <nvrtc.h>
#  endif
#endif

// The pre-compiled renderer support library LLVM bitcode is embedded
// into the executable and made available through these variables.
extern int rend_llvm_compiled_ops_size;
extern unsigned char rend_llvm_compiled_ops_block[];



OSL_NAMESPACE_ENTER


#if (OPTIX_VERSION >= 70000)

#define CUDA_CHECK(call)                                                  \
{                                                                         \
    cudaError_t error = call;                                             \
    if (error != cudaSuccess)                                             \
    {                                                                     \
        std::stringstream ss;                                             \
        ss << "CUDA call (" << #call << " ) failed with error: '"         \
           << cudaGetErrorString( error )                                 \
           << "' (" __FILE__ << ":" << __LINE__ << ")\n";                 \
           fprintf (stderr, "[CUDA ERROR]  %s", ss.str().c_str() );       \
        exit(1);                                                          \
    }                                                                     \
}

#define NVRTC_CHECK(call)                                                 \
{                                                                         \
    nvrtcResult error = call;                                             \
    if (error != NVRTC_SUCCESS)                                           \
    {                                                                     \
        std::stringstream ss;                                             \
        ss << "NVRTC call (" << #call << " ) failed with error: '"        \
           << nvrtcGetErrorString( error )                                \
           << "' (" __FILE__ << ":" << __LINE__ << ")\n";                 \
           fprintf (stderr, "[NVRTC ERROR]  %s", ss.str().c_str() );      \
        exit(1);                                                          \
    }                                                                     \
}


#define OPTIX_CHECK(call)                                                 \
{                                                                         \
    OptixResult res = call;                                               \
    if (res != OPTIX_SUCCESS)                                             \
    {                                                                     \
        std::stringstream ss;                                             \
        ss  << "Optix call '" << #call << "' failed with error: "         \
            << optixGetErrorName( res )                                   \
            << " (" __FILE__ ":"   << __LINE__ << ")\n";                  \
        fprintf (stderr,"[OPTIX ERROR]  %s", ss.str().c_str() );          \
        exit(1);                                                          \
    }                                                                     \
}
#endif

#define CUDA_SYNC_CHECK()                                               \
{                                                                       \
    cudaDeviceSynchronize();                                            \
    cudaError_t error = cudaGetLastError();                             \
    if (error != cudaSuccess) {                                         \
        fprintf ( stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString( error ) ); \
        exit(1);                                                        \
    }                                                                   \
}

#ifdef OSL_USE_OPTIX
#if (OPTIX_VERSION >= 70000)
static void context_log_cb (unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
//    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}
#endif
#endif

OptixGridRenderer::OptixGridRenderer ()
{
#ifdef OSL_USE_OPTIX

#if (OPTIX_VERSION < 70000)
    // Set up the OptiX context
    m_optix_ctx = optix::Context::create();
    if (m_optix_ctx->getEnabledDeviceCount() != 1)
        errhandler().warning ("Only one CUDA device is currently supported");

    // Set up the string table. This allocates a block of CUDA device memory to
    // hold all of the static strings used by the OSL shaders. The strings can
    // be accessed via OptiX variables that hold pointers to the table entries.
    m_str_table.init(m_optix_ctx);
#else
    // Initialize CUDA
    cudaFree(0);

    CUcontext cuCtx = nullptr;  // zero means take the current context

    OptixDeviceContextOptions ctx_options = {};
    ctx_options.logCallbackFunction = context_log_cb;
    ctx_options.logCallbackLevel    = 4;

    OPTIX_CHECK (optixInit());
    OPTIX_CHECK (optixDeviceContextCreate (cuCtx, &ctx_options, &m_optix_ctx));

    CUDA_CHECK (cudaSetDevice (0));
    CUDA_CHECK (cudaStreamCreate (&m_cuda_stream));

    // Set up the string table. This allocates a block of CUDA device memory to
    // hold all of the static strings used by the OSL shaders. The strings can
    // be accessed via OptiX variables that hold pointers to the table entries.
    m_str_table.init(m_optix_ctx);
    // Register all of our string table entries
    for (auto &&gvar : m_str_table.contents())
        register_global(gvar.first.c_str(), gvar.second);
#endif //#if (OPTIX_VERSION < 70000)

#endif //#ifdef OSL_USE_OPTIX
}

uint64_t
OptixGridRenderer::register_global (const std::string &str, uint64_t value)
{
    auto it = m_globals_map.find (ustring(str));

    if (it != m_globals_map.end()) {
       return it->second;
    }
    m_globals_map[ustring(str)] = value;
    return value;
}

bool
OptixGridRenderer::fetch_global (const std::string &str, uint64_t *value)
{
    auto it = m_globals_map.find (ustring(str));

    if (it != m_globals_map.end()) {
       *value = it->second;
       return true;
    }
    return false;
}



std::string
OptixGridRenderer::load_ptx_file (string_view filename)
{
#ifdef OSL_USE_OPTIX
    std::vector<std::string> paths = {
        OIIO::Filesystem::parent_path(OIIO::Sysutil::this_program_path()),
        PTX_PATH
    };
    std::string filepath = OIIO::Filesystem::searchpath_find (filename, paths,
                                                              false);
    if (OIIO::Filesystem::exists(filepath)) {
        std::string ptx_string;
        if (OIIO::Filesystem::read_text_file (filepath, ptx_string))
            return ptx_string;
    }
#endif
    errhandler().severe ("Unable to load %s", filename);
    return {};
}



OptixGridRenderer::~OptixGridRenderer ()
{
#ifdef OSL_USE_OPTIX
    m_str_table.freetable();
#if (OPTIX_VERSION < 70000)
    if (m_optix_ctx)
        m_optix_ctx->destroy();
#else
    if (m_optix_ctx)
        OPTIX_CHECK (optixDeviceContextDestroy (m_optix_ctx));
#endif
#endif
}



void
OptixGridRenderer::init_shadingsys (ShadingSystem *ss)
{
    shadingsys = ss;

#ifdef OSL_USE_OPTIX
    shadingsys->attribute ("lib_bitcode", {OSL::TypeDesc::UINT8, rend_llvm_compiled_ops_size},
                           rend_llvm_compiled_ops_block);
#endif
}



bool
OptixGridRenderer::init_optix_context (int xres OSL_MAYBE_UNUSED,
                                       int yres OSL_MAYBE_UNUSED)
{
#ifdef OSL_USE_OPTIX
#if (OPTIX_VERSION < 70000)
    m_optix_ctx->setRayTypeCount (2);
    m_optix_ctx->setEntryPointCount (1);
    m_optix_ctx->setStackSize (2048);
    m_optix_ctx->setPrintEnabled (true);

    // Load the renderer CUDA source and generate PTX for it
    std::string progName = "optix_grid_renderer.ptx";
    std::string renderer_ptx = load_ptx_file(progName);
    if (renderer_ptx.empty()) {
        errhandler().severe ("Could not find PTX for the raygen program");
        return false;
    }

    // Create the OptiX programs and set them on the optix::Context
    m_program = m_optix_ctx->createProgramFromPTXString(renderer_ptx, "raygen");
    m_optix_ctx->setRayGenerationProgram(0, m_program);
#endif //#if (OPTIX_VERSION < 70000)
#endif
    return true;
}



bool
OptixGridRenderer::synch_attributes ()
{
#ifdef OSL_USE_OPTIX

#if (OPTIX_VERSION < 70000)
    // FIXME -- this is for testing only
    // Make some device strings to test userdata parameters
    uint64_t addr1 = register_string ("ud_str_1", "");
    uint64_t addr2 = register_string ("userdata string", "");
    m_optix_ctx["test_str_1"]->setUserData (sizeof(char*), &addr1);
    m_optix_ctx["test_str_2"]->setUserData (sizeof(char*), &addr2);

    {
        const char* name = OSL_NAMESPACE_STRING "::pvt::s_color_system";

        char* colorSys = nullptr;
        long long cpuDataSizes[2] = {0,0};
        if (!shadingsys->getattribute("colorsystem", TypeDesc::PTR, (void*)&colorSys) ||
            !shadingsys->getattribute("colorsystem:sizes", TypeDesc(TypeDesc::LONGLONG,2), (void*)&cpuDataSizes) ||
            !colorSys || !cpuDataSizes[0]) {
            errhandler().error ("No colorsystem available.");
            return false;
        }

        auto cpuDataSize = cpuDataSizes[0];
        auto numStrings = cpuDataSizes[1];

        // Get the size data-size, minus the ustring size
        const size_t podDataSize = cpuDataSize - sizeof(StringParam)*numStrings;

        optix::Buffer buffer = m_optix_ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
        if (!buffer) {
            errhandler().error ("Could not create buffer for '%s'.", name);
            return false;
        }

        // set the element size to char
        buffer->setElementSize(sizeof(char));

        // and number of elements to the actual size needed.
        buffer->setSize(podDataSize + sizeof(DeviceString)*numStrings);

        // copy the base data
        char* gpuData = (char*) buffer->map();
        if (!gpuData) {
            errhandler().error ("Could not map buffer for '%s' (size: %lu).",
                                name, podDataSize + sizeof(DeviceString)*numStrings);
            return false;
        }
        ::memcpy(gpuData, colorSys, podDataSize);
        // then copy the device string to the end, first strings starting at dataPtr - (numStrings)
        // FIXME -- Should probably handle alignment better.
        const ustring* cpuString = (const ustring*)(colorSys + (cpuDataSize - sizeof(StringParam)*numStrings));
        char* gpuStrings = gpuData + podDataSize;
        for (const ustring* end = cpuString + numStrings; cpuString < end; ++cpuString) {
            // convert the ustring to a device string
            uint64_t devStr = register_string (cpuString->string(), "");
            ::memcpy(gpuStrings, &devStr, sizeof(devStr));
            gpuStrings += sizeof(DeviceString);
        }

        buffer->unmap();
        m_optix_ctx[name]->setBuffer(buffer);

#else // #if (OPTIX_VERSION < 70000)

    // FIXME -- this is for testing only
    // Make some device strings to test userdata parameters
    uint64_t addr1 = register_string ("ud_str_1", "");
    uint64_t addr2 = register_string ("userdata string", "");

    // Store the user-data
    register_global("test_str_1", addr1);
    register_global("test_str_2", addr2);

    {
        char* colorSys = nullptr;
        long long cpuDataSizes[2] = {0,0};
        if (!shadingsys->getattribute("colorsystem", TypeDesc::PTR, (void*)&colorSys) ||
            !shadingsys->getattribute("colorsystem:sizes", TypeDesc(TypeDesc::LONGLONG,2), (void*)&cpuDataSizes) ||
            !colorSys || !cpuDataSizes[0]) {
            errhandler().error ("No colorsystem available.");
            return false;
        }

        const char* name = OSL_NAMESPACE_STRING "::pvt::s_color_system";

        auto cpuDataSize = cpuDataSizes[0];
        auto numStrings = cpuDataSizes[1];

        // Get the size data-size, minus the ustring size
        const size_t podDataSize = cpuDataSize - sizeof(StringParam)*numStrings;

        CUdeviceptr d_buffer;

        CUDA_CHECK (cudaMalloc (reinterpret_cast<void **>(&d_buffer), podDataSize + sizeof(DeviceString)*numStrings));

        CUDA_CHECK (cudaMemcpy (reinterpret_cast<void *>(d_buffer), colorSys, podDataSize, cudaMemcpyHostToDevice));

        // then copy the device string to the end, first strings starting at dataPtr - (numStrings)
        // FIXME -- Should probably handle alignment better.
        const ustring* cpuString = (const ustring*)(colorSys + (cpuDataSize - sizeof(StringParam)*numStrings));
        CUdeviceptr gpuStrings = d_buffer + podDataSize;
        for (const ustring* end = cpuString + numStrings; cpuString < end; ++cpuString) {
            // convert the ustring to a device string
            uint64_t devStr = register_string (cpuString->string(), "");
            CUDA_CHECK (cudaMemcpy (reinterpret_cast<void *>(gpuStrings), &devStr, sizeof(devStr), cudaMemcpyHostToDevice));
            gpuStrings += sizeof(DeviceString);
        }
        register_global (name, d_buffer);

#endif
    }
#endif
    return true;
}



bool
OptixGridRenderer::make_optix_materials ()
{
#ifdef OSL_USE_OPTIX

    // Stand-in: names of shader outputs to preserve
    // FIXME
    std::vector<const char*> outputs { "Cout" };

    // Optimize each ShaderGroup in the scene, and use the resulting
    // PTX to create OptiX Programs which can be called by the closest
    // hit program in the wrapper to execute the compiled OSL shader.
    int mtl_id = 0;

#if (OPTIX_VERSION < 70000)
    for (const auto& groupref : shaders()) {
        shadingsys->attribute (groupref.get(), "renderer_outputs",
                               TypeDesc(TypeDesc::STRING, outputs.size()),
                               outputs.data());

        shadingsys->optimize_group (groupref.get(), nullptr);

        if (!shadingsys->find_symbol (*groupref.get(), ustring(outputs[0]))) {
            // FIXME: This is for cases where testshade is run with 1x1 resolution
            //        Those tests may not have a Cout parameter to write to.
            if (m_xres > 1 && m_yres > 1) {
                errhandler().warning ("Requested output '%s', which wasn't found",
                                      outputs[0]);
            }
        }

        std::string group_name, init_name, entry_name;
        shadingsys->getattribute (groupref.get(), "groupname",        group_name);
        shadingsys->getattribute (groupref.get(), "group_init_name",  init_name);
        shadingsys->getattribute (groupref.get(), "group_entry_name", entry_name);

        // Retrieve the compiled ShaderGroup PTX
        std::string osl_ptx;
        shadingsys->getattribute (groupref.get(), "ptx_compiled_version",
                                  OSL::TypeDesc::PTR, &osl_ptx);

        if (osl_ptx.empty()) {
            errhandler().error ("Failed to generate PTX for ShaderGroup %s",
                                group_name);
            return false;
        }

        if (options.get_int("saveptx")) {
            std::string filename = OIIO::Strutil::sprintf("%s_%d.ptx",
                                                          group_name, mtl_id++);
            OIIO::ofstream out;
            OIIO::Filesystem::open (out, filename);
            out << osl_ptx;
        }

        // Create Programs from the init and group_entry functions,
        // and set the OSL functions as Callable Programs so that they
        // can be executed by the closest hit program in the wrapper
        optix::Program osl_init = m_optix_ctx->createProgramFromPTXString (
            osl_ptx, init_name);
        optix::Program osl_group = m_optix_ctx->createProgramFromPTXString (
            osl_ptx, entry_name);

        // Grid shading
        m_program["osl_init_func" ]->setProgramId (osl_init );
        m_program["osl_group_func"]->setProgramId (osl_group);
    }

#else //#if (OPTIX_VERSION < 70000)

    std::vector<OptixModule> modules;

    // Space for mesage logging
    char msg_log[8192];
    size_t sizeof_msg_log;

    // Make module that contains programs we'll use in this scene
    OptixModuleCompileOptions module_compile_options = {};

    module_compile_options.maxRegisterCount  = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    OptixPipelineCompileOptions pipeline_compile_options = {};

    pipeline_compile_options.traversableGraphFlags      = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipeline_compile_options.usesMotionBlur             = false;
    pipeline_compile_options.numPayloadValues           = 0;
    pipeline_compile_options.numAttributeValues         = 0;
    pipeline_compile_options.exceptionFlags             = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "render_params";

    // Create 'raygen' program

    // Load the renderer CUDA source and generate PTX for it
    std::string progName = "optix_grid_renderer.ptx";
    std::string program_ptx = load_ptx_file(progName);
    if (program_ptx.empty()) {
        errhandler().severe ("Could not find PTX for the raygen program");
        return false;
    }

    sizeof_msg_log = sizeof(msg_log);
    OptixModule program_module;
    OPTIX_CHECK (optixModuleCreateFromPTX (m_optix_ctx,
                                           &module_compile_options,
                                           &pipeline_compile_options,
                                           program_ptx.c_str(),
                                           program_ptx.size(),
                                           msg_log, &sizeof_msg_log,
                                           &program_module));
    //if (sizeof_msg_log > 1)
    //    printf ("Creating module from PTX-file %s:\n%s\n", progName.c_str(), msg_log);

    // Record it so we can destroy it later
    modules.push_back(program_module);

    OptixProgramGroupOptions program_options = {};
    std::vector<OptixProgramGroup> program_groups;

    // Raygen group
    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module            = program_module;
    raygen_desc.raygen.entryFunctionName = "__raygen__";

    OptixProgramGroup  raygen_group;
    sizeof_msg_log = sizeof (msg_log);
    OPTIX_CHECK (optixProgramGroupCreate (m_optix_ctx,
                                          &raygen_desc,
                                          1, // number of program groups
                                          &program_options, // program options
                                          msg_log, &sizeof_msg_log,
                                          &raygen_group));
    //if (sizeof_msg_log > 1)
    //    printf ("Creating 'ray-gen' program group:\n%s\n", msg_log);

    // Miss group
    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module            = program_module;
    miss_desc.miss.entryFunctionName = "__miss__";

    OptixProgramGroup  miss_group;
    sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK (optixProgramGroupCreate (m_optix_ctx,
                                          &miss_desc,
                                          1,
                                          &program_options,
                                          msg_log, &sizeof_msg_log,
                                          &miss_group));
    //if (sizeof_msg_log > 1)
    //    printf ("Creating 'miss' program group:\n%s\n", msg_log);

    // Hitgroup
    OptixProgramGroupDesc hitgroup_desc = {};
    hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_desc.hitgroup.moduleCH = program_module;
    hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__";
    hitgroup_desc.hitgroup.moduleAH = program_module;
    hitgroup_desc.hitgroup.entryFunctionNameAH = "__anyhit__";

    OptixProgramGroup  hitgroup_group;

    sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK (optixProgramGroupCreate (m_optix_ctx,
                                          &hitgroup_desc,
                                          1, // number of program groups
                                          &program_options, // program options
                                          msg_log, &sizeof_msg_log,
                                          &hitgroup_group));
    //if (sizeof_msg_log > 1)
    //    printf ("Creating 'hitgroup' program group:\n%s\n", msg_log);

    // Create materials
    for (const auto& groupref : shaders()) {
        shadingsys->attribute (groupref.get(), "renderer_outputs",
                               TypeDesc(TypeDesc::STRING, outputs.size()),
                               outputs.data());

        shadingsys->optimize_group (groupref.get(), nullptr);

        if (!shadingsys->find_symbol (*groupref.get(), ustring(outputs[0]))) {
            // FIXME: This is for cases where testshade is run with 1x1 resolution
            //        Those tests may not have a Cout parameter to write to.
            if (m_xres > 1 && m_yres > 1) {
                errhandler().warning ("Requested output '%s', which wasn't found",
                                      outputs[0]);
            }
        }

        std::string group_name, init_name, entry_name;
        shadingsys->getattribute (groupref.get(), "groupname",        group_name);
        shadingsys->getattribute (groupref.get(), "group_init_name",  init_name);
        shadingsys->getattribute (groupref.get(), "group_entry_name", entry_name);

        // Retrieve the compiled ShaderGroup PTX
        std::string osl_ptx;
        shadingsys->getattribute (groupref.get(), "ptx_compiled_version",
                                  OSL::TypeDesc::PTR, &osl_ptx);

        if (osl_ptx.empty()) {
            errhandler().error ("Failed to generate PTX for ShaderGroup %s",
                                group_name);
            return false;
        }

        if (options.get_int("saveptx")) {
            std::string filename = OIIO::Strutil::sprintf("%s_%d.ptx", group_name,
                                                          mtl_id++);
            OIIO::ofstream out;
            OIIO::Filesystem::open (out, filename);
            out << osl_ptx;
        }

        OptixModule optix_module;

        // Create Programs from the init and group_entry functions,
        // and set the OSL functions as Callable Programs so that they
        // can be executed by the closest hit program in the wrapper
        sizeof_msg_log = sizeof(msg_log);
        OPTIX_CHECK (optixModuleCreateFromPTX (m_optix_ctx,
                                               &module_compile_options,
                                               &pipeline_compile_options,
                                               osl_ptx.c_str(),
                                               osl_ptx.size(),
                                               msg_log, &sizeof_msg_log,
                                               &optix_module));
        //if (sizeof_msg_log > 1)
        //    printf ("Creating module from PTX group '%s':\n%s\n", group_name.c_str(), msg_log);

        modules.push_back (optix_module);

        // Create 2x program groups (for direct callables)
        OptixProgramGroupOptions program_options = {};
        OptixProgramGroupDesc pgDesc[3] = {};
        pgDesc[0].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        pgDesc[0].callables.moduleDC            = optix_module;
        pgDesc[0].callables.entryFunctionNameDC = init_name.c_str();
        pgDesc[0].callables.moduleCC            = 0;
        pgDesc[0].callables.entryFunctionNameCC = nullptr;
        pgDesc[1].kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        pgDesc[1].callables.moduleDC            = optix_module;
        pgDesc[1].callables.entryFunctionNameDC = entry_name.c_str();
        pgDesc[1].callables.moduleCC            = 0;
        pgDesc[1].callables.entryFunctionNameCC = nullptr;

        program_groups.resize (program_groups.size() + 2);

        sizeof_msg_log = sizeof(msg_log);
        OPTIX_CHECK (optixProgramGroupCreate (m_optix_ctx,
                                              &pgDesc[0],
                                              2, // number of program groups
                                              &program_options, // program options
                                              msg_log, &sizeof_msg_log,
                                              &program_groups[program_groups.size() - 2]));
        //if (sizeof_msg_log > 1)
        //    printf ("Creating 'shader' group for group '%s':\n%s\n", group_name.c_str(), msg_log);
    }


    OptixPipelineLinkOptions pipeline_link_options;
    pipeline_link_options.maxTraceDepth          = 1;
    pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#if (OPTIX_VERSION < 70100)
    pipeline_link_options.overrideUsesMotionBlur = false;
#endif

    // Build string-table "library"
    nvrtcProgram str_lib;

    auto extractNamespaces = [] (const OIIO::ustring &s) {
        const char *str = s.c_str();
        std::vector<std::string> ns;
        do {
            const char *begin = str;
            // get to first ':'
            while (*str != ':' && *str)
                str++;
            ns.push_back(std::string(begin, str));
            // advance to second ':'
            if (*str && *str == ':')
                str++;
        } while (*str++ != 0);
        return ns;
    };

    std::stringstream strlib_ss;

    strlib_ss << "// so things name-mangle properly\n";
    strlib_ss << "struct DeviceString {\n";
    strlib_ss << "    const char* m_chars;\n";
    strlib_ss << "};\n";

    // write out all the global strings
    for (auto &&gvar : m_globals_map) {
        std::vector<std::string> var_ns = extractNamespaces(gvar.first);

        // build namespace
        for (size_t i = 0; i < var_ns.size() - 1; i++)
            strlib_ss << "namespace " << var_ns[i] << " {\n";

        strlib_ss << "__device__ DeviceString " << var_ns.back() << " = { (const char *)" << gvar.second << "};\n";
        // close namespace up
        for (size_t i = 0; i < var_ns.size() - 1; i++)
            strlib_ss << "}\n";
    }

    strlib_ss << "\n";
    strlib_ss << "extern \"C\" __global__ void __direct_callable__strlib_dummy(int *j)\n";
    strlib_ss << "{\n";
    strlib_ss << "   // must have a __direct_callable__ function for the module to compile\n";
    strlib_ss << "}\n";

    // XXX: Should this move to compute_60 (compute_35-compute_50 is now deprecated)
    const char *cuda_compile_options[] = { "--gpu-architecture=compute_35"  ,
                                           "--use_fast_math"                ,
                                           "-dc"                            ,
                                           "--std=c++11"
                                         };

    int num_compile_flags = int (sizeof (cuda_compile_options) / sizeof (cuda_compile_options[0]));
    size_t str_lib_size, cuda_log_size;

    std::string cuda_string = strlib_ss.str();

    NVRTC_CHECK (nvrtcCreateProgram (&str_lib,
                                     cuda_string.c_str(),
                                     "cuda_strng_library",
                                     0,         // number of headers
                                     nullptr,   // header paths
                                     nullptr)); // header files
    nvrtcResult compileResult = nvrtcCompileProgram (str_lib,  num_compile_flags, cuda_compile_options);
    if (compileResult != NVRTC_SUCCESS) {
        NVRTC_CHECK (nvrtcGetProgramLogSize (str_lib, &cuda_log_size));
        std::vector<char> cuda_log(cuda_log_size+1);
        NVRTC_CHECK (nvrtcGetProgramLog (str_lib, cuda_log.data()));
        cuda_log.back() = 0;
        errhandler().error ("nvrtcCompileProgram failure for:\n%s\n"
                            "====================================\n"
                            "%s\n", cuda_string.c_str(), cuda_log.data());
        return false;
    }
    NVRTC_CHECK (nvrtcGetPTXSize (str_lib, &str_lib_size));
    std::vector<char> str_lib_ptx (str_lib_size);
    NVRTC_CHECK (nvrtcGetPTX(str_lib, str_lib_ptx.data()));
    NVRTC_CHECK (nvrtcDestroyProgram (&str_lib));

    std::string strlib_string (str_lib_ptx.begin (), str_lib_ptx.end ());

    OptixModule strlib_module;
    sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK (optixModuleCreateFromPTX (m_optix_ctx,
                                           &module_compile_options,
                                           &pipeline_compile_options,
                                           str_lib_ptx.data(),
                                           str_lib_ptx.size(),
                                           msg_log, &sizeof_msg_log,
                                           &strlib_module));
    //if (sizeof_msg_log > 1)
    //    printf ("Creating module from string-library PTX:\n%s\n", msg_log);

    OptixProgramGroupDesc strlib_pg_desc = {};
    strlib_pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    strlib_pg_desc.callables.moduleDC              = strlib_module;
    strlib_pg_desc.callables.entryFunctionNameDC   = "__direct_callable__strlib_dummy";
    strlib_pg_desc.callables.moduleCC              = 0;
    strlib_pg_desc.callables.entryFunctionNameCC   = nullptr;

    OptixProgramGroup strlib_group;

    sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK (optixProgramGroupCreate (m_optix_ctx,
                                          &strlib_pg_desc,
                                          1,
                                          &program_options,
                                          msg_log, &sizeof_msg_log,
                                          &strlib_group));
    //if (sizeof_msg_log > 1)
    //    printf ("Creating program group for string-library:\n%s\n", msg_log);

    // Set up OptiX pipeline
    std::vector<OptixProgramGroup> final_groups = {
        strlib_group,      // string globals
        raygen_group,
        miss_group,
        hitgroup_group,
        program_groups[0], // init
        program_groups[1], // entry
    };

    sizeof_msg_log = sizeof(msg_log);
    OPTIX_CHECK (optixPipelineCreate (m_optix_ctx,
                                      &pipeline_compile_options,
                                      &pipeline_link_options,
                                      final_groups.data(),
                                      int(final_groups.size()),
                                      msg_log, &sizeof_msg_log,
                                      &m_optix_pipeline));
    //if (sizeof_msg_log > 1)
    //    printf ("Creating optix pipeline:\n%s\n", msg_log);

    // Set the pipeline stack size
    OptixStackSizes stack_sizes = {};
    for( OptixProgramGroup& program_group : final_groups )
        OPTIX_CHECK (optixUtilAccumulateStackSizes (program_group, &stack_sizes));

    uint32_t max_trace_depth = 1;
    uint32_t max_cc_depth    = 1;
    uint32_t max_dc_depth    = 1;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK (optixUtilComputeStackSizes (&stack_sizes,
                                             max_trace_depth,
                                             max_cc_depth,
                                             max_dc_depth,
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state,
                                             &continuation_stack_size ) );

    const uint32_t max_traversal_depth = 1;
    OPTIX_CHECK (optixPipelineSetStackSize (m_optix_pipeline,
                                            direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state,
                                            continuation_stack_size,
                                            max_traversal_depth ));

    // Build OptiX Shader Binding Table (SBT)
    CUdeviceptr d_raygenRecord;
    CUdeviceptr d_missRecord;
    CUdeviceptr d_hitgroupRecord;
    CUdeviceptr d_callablesRecord;

    EmptyRecord raygenRecord, missRecord, hitgroupRecord, callablesRecord[2];

    OPTIX_CHECK (optixSbtRecordPackHeader (raygen_group     , &raygenRecord  ));
    OPTIX_CHECK (optixSbtRecordPackHeader (miss_group       , &missRecord    ));
    OPTIX_CHECK (optixSbtRecordPackHeader (hitgroup_group   , &hitgroupRecord));
    OPTIX_CHECK (optixSbtRecordPackHeader (program_groups[0], &callablesRecord[0]));
    OPTIX_CHECK (optixSbtRecordPackHeader (program_groups[1], &callablesRecord[1]));

    raygenRecord.data       = reinterpret_cast<void *>(5);
    missRecord.data         = nullptr;
    hitgroupRecord.data     = nullptr;
    callablesRecord[0].data = reinterpret_cast<void *>(1);
    callablesRecord[1].data = reinterpret_cast<void *>(2);

    CUDA_CHECK (cudaMalloc (reinterpret_cast<void **> (&d_raygenRecord)   ,     sizeof(EmptyRecord)));
    CUDA_CHECK (cudaMalloc (reinterpret_cast<void **> (&d_missRecord)     ,     sizeof(EmptyRecord)));
    CUDA_CHECK (cudaMalloc (reinterpret_cast<void **> (&d_hitgroupRecord) ,     sizeof(EmptyRecord)));
    CUDA_CHECK (cudaMalloc (reinterpret_cast<void **> (&d_callablesRecord), 2 * sizeof(EmptyRecord)));

    CUDA_CHECK (cudaMemcpy (reinterpret_cast<void *>( d_raygenRecord)   , &raygenRecord      , sizeof(EmptyRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK (cudaMemcpy (reinterpret_cast<void *>( d_missRecord)     , &missRecord        , sizeof(EmptyRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK (cudaMemcpy (reinterpret_cast<void *>( d_hitgroupRecord) , &hitgroupRecord    , sizeof(EmptyRecord), cudaMemcpyHostToDevice));
    CUDA_CHECK (cudaMemcpy (reinterpret_cast<void *>( d_callablesRecord), &callablesRecord[0], 2 * sizeof(EmptyRecord), cudaMemcpyHostToDevice));

    // Looks like OptixShadingTable needs to be filled out completely
    m_optix_sbt.raygenRecord                 = d_raygenRecord;
    m_optix_sbt.missRecordBase               = d_missRecord;
    m_optix_sbt.missRecordStrideInBytes      = sizeof(EmptyRecord);
    m_optix_sbt.missRecordCount              = 1;
    m_optix_sbt.hitgroupRecordBase           = d_hitgroupRecord;
    m_optix_sbt.hitgroupRecordStrideInBytes  = sizeof(EmptyRecord);
    m_optix_sbt.hitgroupRecordCount          = 1;
    m_optix_sbt.callablesRecordBase          = d_callablesRecord;
    m_optix_sbt.callablesRecordStrideInBytes = sizeof(EmptyRecord);
    m_optix_sbt.callablesRecordCount         = 2;

#endif //#if (OPTIX_VERSION < 70000)

#endif //#ifdef OSL_USE_OPTIX
    return true;
}



bool
OptixGridRenderer::finalize_scene()
{
#ifdef OSL_USE_OPTIX
    make_optix_materials();

#if (OPTIX_VERSION < 70000)

    m_optix_ctx["invw"]->setFloat (1.0f/m_xres);
    m_optix_ctx["invh"]->setFloat (1.0f/m_yres);

    // Create the output buffer
    optix::Buffer buffer = m_optix_ctx->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, m_xres, m_yres);
    m_optix_ctx["output_buffer"]->set(buffer);
#else

#endif //#if (OPTIX_VERSION < 70000)

#if (OPTIX_VERSION < 70000)
    if (!synch_attributes ())
        return false;
#endif

#if (OPTIX_VERSION < 70000)
    m_optix_ctx->validate();
#else
#endif

#endif
    return true;
}



/// Return true if the texture handle (previously returned by
/// get_texture_handle()) is a valid texture that can be subsequently
/// read or sampled.
bool
OptixGridRenderer::good(TextureHandle *handle OSL_MAYBE_UNUSED)
{
#ifdef OSL_USE_OPTIX
#if (OPTIX_VERSION < 70000)
    return intptr_t(handle) != RT_TEXTURE_ID_NULL;
#else
    return handle != nullptr;
#endif
#else
    return false;
#endif
}



/// Given the name of a texture, return an opaque handle that can be
/// used with texture calls to avoid the name lookups.
RendererServices::TextureHandle*
OptixGridRenderer::get_texture_handle (ustring filename OSL_MAYBE_UNUSED,
                                       ShadingContext* shading_context OSL_MAYBE_UNUSED)
{
#ifdef OSL_USE_OPTIX

#if (OPTIX_VERSION < 70000)
    auto itr = m_samplers.find(filename);
    if (itr == m_samplers.end()) {
        optix::TextureSampler sampler = context()->createTextureSampler();
        sampler->setWrapMode(0, RT_WRAP_REPEAT);
        sampler->setWrapMode(1, RT_WRAP_REPEAT);
        sampler->setWrapMode(2, RT_WRAP_REPEAT);

        sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
        sampler->setIndexingMode(false ? RT_TEXTURE_INDEX_ARRAY_INDEX : RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
        sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
        sampler->setMaxAnisotropy(1.0f);


        OIIO::ImageBuf image;
        if (!image.init_spec(filename, 0, 0)) {
            errhandler().error ("Could not load: %s", filename);
            return (TextureHandle*)(intptr_t(RT_TEXTURE_ID_NULL));
        }
        int nchan = image.spec().nchannels;

        OIIO::ROI roi = OIIO::get_roi_full(image.spec());
        int width = roi.width(), height = roi.height();
        std::vector<float> pixels(width * height * nchan);
        image.get_pixels(roi, OIIO::TypeDesc::FLOAT, pixels.data());

        optix::Buffer buffer = context()->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, width, height);

        float* device_ptr = static_cast<float*>(buffer->map());
        unsigned int pixel_idx = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                memcpy(device_ptr, &pixels[pixel_idx], sizeof(float) * nchan);
                device_ptr += 4;
                pixel_idx += nchan;
            }
        }
        buffer->unmap();
        sampler->setBuffer(buffer);
        itr = m_samplers.emplace(std::move(filename), std::move(sampler)).first;

    }
    return (RendererServices::TextureHandle*) intptr_t(itr->second->getId());
#else //#if (OPTIX_VERSION < 70000)
    auto itr = m_samplers.find(filename);
    if (itr == m_samplers.end()) {

        // Open image
        OIIO::ImageBuf image;
        if (!image.init_spec(filename, 0, 0)) {
            errhandler().error ("Could not load: %s", filename);
            return (TextureHandle*)(intptr_t(nullptr));
        }

        OIIO::ROI roi = OIIO::get_roi_full(image.spec());
        int32_t width = roi.width(), height = roi.height();
        std::vector<float> pixels(width * height * 4);

        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                image.getpixel(i, j, 0, &pixels[((j*width) + i) * 4 + 0]);
            }
        }
        cudaResourceDesc res_desc = {};

        // hard-code textures to 4 channels
        int32_t pitch  = width * 4 * sizeof(float);
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

        cudaArray_t   pixelArray;
        CUDA_CHECK (cudaMallocArray (&pixelArray,
                                     &channel_desc,
                                     width,height));

        CUDA_CHECK (cudaMemcpy2DToArray (pixelArray,
                                         /* offset */0,0,
                                         pixels.data(),
                                         pitch,pitch,height,
                                         cudaMemcpyHostToDevice));

        res_desc.resType          = cudaResourceTypeArray;
        res_desc.res.array.array  = pixelArray;

        cudaTextureDesc tex_desc     = {};
        tex_desc.addressMode[0]      = cudaAddressModeWrap;
        tex_desc.addressMode[1]      = cudaAddressModeWrap;
        tex_desc.filterMode          = cudaFilterModeLinear;
        tex_desc.readMode            = cudaReadModeElementType; //cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords    = 1;
        tex_desc.maxAnisotropy       = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode    = cudaFilterModePoint;
        tex_desc.borderColor[0]      = 1.0f;
        tex_desc.sRGB                = 0;

        // Create texture object
        cudaTextureObject_t cuda_tex = 0;
        CUDA_CHECK (cudaCreateTextureObject (&cuda_tex, &res_desc, &tex_desc, nullptr));
        itr = m_samplers.emplace (std::move(filename), std::move(cuda_tex)).first;
    }
    return reinterpret_cast<RendererServices::TextureHandle *>(itr->second);

#endif //#if (OPTIX_VERSION < 70000)

#else
    return nullptr;
#endif
}



void
OptixGridRenderer::prepare_render()
{
#ifdef OSL_USE_OPTIX
    // Set up the OptiX Context
    init_optix_context(m_xres, m_yres);

    // Set up the OptiX scene graph
    finalize_scene ();
#endif
}



void
OptixGridRenderer::warmup()
{
#ifdef OSL_USE_OPTIX
    // Perform a tiny launch to warm up the OptiX context
#if (OPTIX_VERSION < 70000)
    m_optix_ctx->launch (0, 1, 1);
#else
    OPTIX_CHECK (optixLaunch (m_optix_pipeline,
                              m_cuda_stream,
                              d_launch_params,
                              sizeof(RenderParams),
                              &m_optix_sbt,
                              0, 0, 1));
    CUDA_SYNC_CHECK();
#endif

#endif
}


//extern "C" void setTestshadeGlobals(float h_invw, float h_invh, CUdeviceptr d_output_buffer, bool h_flipv);

void
OptixGridRenderer::render(int xres OSL_MAYBE_UNUSED, int yres OSL_MAYBE_UNUSED)
{
#ifdef OSL_USE_OPTIX
#if (OPTIX_VERSION < 70000)
    m_optix_ctx->launch (0, xres, yres);
#else
    CUDA_CHECK (cudaMalloc (reinterpret_cast<void **>(&d_output_buffer), xres * yres * 4 * sizeof(float)));
    CUDA_CHECK (cudaMalloc (reinterpret_cast<void **>(&d_launch_params), sizeof(RenderParams)));


    m_xres = xres;
    m_yres = yres;

    RenderParams params;
    params.invw = 1.0f / m_xres;
    params.invh = 1.0f / m_yres;
    params.flipv = false; /* I don't see flipv being initialized anywhere */
    params.output_buffer = d_output_buffer;

    CUDA_CHECK (cudaMemcpy (reinterpret_cast<void *>(d_launch_params), &params, sizeof(RenderParams), cudaMemcpyHostToDevice));

    OPTIX_CHECK (optixLaunch (m_optix_pipeline,
                              m_cuda_stream,
                              d_launch_params,
                              sizeof(RenderParams),
                              &m_optix_sbt,
                              xres, yres, 1));
    CUDA_SYNC_CHECK();
#endif
#endif
}



void
OptixGridRenderer::finalize_pixel_buffer ()
{
#ifdef OSL_USE_OPTIX

    std::string buffer_name = "output_buffer";
#if (OPTIX_VERSION < 70000)
    const void* buffer_ptr = m_optix_ctx[buffer_name]->getBuffer()->map();
    if (! buffer_ptr)
        errhandler().severe ("Unable to map buffer %s", buffer_name);
    outputbuf(0)->set_pixels (OIIO::ROI::All(), OIIO::TypeFloat, buffer_ptr);
#else
    std::vector<float> tmp_buff(m_xres * m_yres * 3);
    CUDA_CHECK (cudaMemcpy (tmp_buff.data(), reinterpret_cast<void *>(d_output_buffer), m_xres * m_yres * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    outputbuf(0)->set_pixels (OIIO::ROI::All(), OIIO::TypeFloat, tmp_buff.data());
#endif
#endif
}



void
OptixGridRenderer::clear()
{
    shaders().clear();
#ifdef OSL_USE_OPTIX
#if (OPTIX_VERSION < 70000)
    if (m_optix_ctx) {
        m_optix_ctx->destroy();
        m_optix_ctx = nullptr;
    }
#else
    if (m_optix_ctx) {
        OPTIX_CHECK (optixDeviceContextDestroy (m_optix_ctx));
        m_optix_ctx = 0;
    }
#endif

#endif
}

OSL_NAMESPACE_EXIT

