// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include <iostream>

#include <OSL/genclosure.h>
#include <OSL/oslclosure.h>
#include <OSL/oslexec.h>

#include <OSL/device_string.h>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>

#include <nvrtc.h>

#include "cuda_grid_renderer.h"
#include "cuda_macro.h"

#include "closures.h"

using namespace OSL;
using namespace OIIO;

// The pre-compiled renderer support library LLVM bitcode is embedded
// into the executable and made available through these variables.
extern int rend_llvm_compiled_ops_size;
extern unsigned char rend_llvm_compiled_ops_block[];

void
register_closures(ShadingSystem& ss);
std::string
build_string_table_ptx(const CudaGridRenderer& rs);
std::string
build_trampoline_ptx(OSL::ShaderGroup& group, std::string init_name,
                     std::string entry_name);

int
main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage:\n    " << argv[0] << " <shader>" << std::endl;
        return -2;
    }

    // Initialize CUDA
    cudaFree(0);

    CUcontext cuCtx = nullptr;  // zero means take the current context

    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create renderer services and shading system
    CudaGridRenderer rs;
    auto ts = TextureSystem::create();
    ShadingSystem ss(&rs, ts);
    register_closures(ss);

    ss.attribute("lockgeom", 1);

    auto shader_name = argv[1];
    auto layer_name  = shader_name;

    // Create our ShaderGroup consisting of just the shader we specified on the
    // command line
    auto shader_group = ss.ShaderGroupBegin("");
    ss.Shader(*shader_group, "surface", shader_name, layer_name);
    ss.ShaderGroupEnd(*shader_group);

    int w = 512;
    int h = 512;

    // We need to do a dummy run of the shader group in order to populate the
    // string table before we build the string table PTX, otherwise the string
    // table will be incomplete and we'll get cryptic link errors
    PerThreadInfo* thread_info = ss.create_thread_info();
    ShadingContext* ctx        = ss.get_context(thread_info);

    ShaderGlobals sg;
    memset((char*)&sg, 0, sizeof(ShaderGlobals));
    ss.optimize_group(shader_group.get(), nullptr);
    ss.execute(*ctx, *shader_group, sg, false);

    ss.release_context(ctx);  // don't need this anymore for now
    ss.destroy_thread_info(thread_info);

    // build string table ptx
    std::string ptx_strlib = build_string_table_ptx(rs);

    // compile shader group ptx
    std::string ptx_shader;
    if (!ss.getattribute(shader_group.get(), "ptx_compiled_version",
                         TypeDesc::PTR, &ptx_shader)) {
        std::cerr << "Error getting shader group ptx" << std::endl;
        return 1;
    }

    // Get the cuda entry points from the ShadingSystem. The names are
    // auto-generated and mangled, so will be different for every shader group.
    // An alternative here would be to modify the generated group PTX to change
    // the function names
    std::string init_name, entry_name;
    if (!ss.getattribute(shader_group.get(), "group_init_name", init_name)) {
        std::cerr << "Error getting shader group init name" << std::endl;
        return 1;
    }

    if (!ss.getattribute(shader_group.get(), "group_entry_name", entry_name)) {
        std::cerr << "Error getting shader group entry name" << std::endl;
        return 1;
    }

    // compile trampoline ptx - this is just here to give us consistently named
    // functions to call from the renderer's shade() function without needing
    // to recompile the renderer itself.
    std::string ptx_trampoline = build_trampoline_ptx(*shader_group, init_name,
                                                      entry_name);

    // compile renderer - this simple example just treats the output image
    // as a Reyes-style grid of points.
    std::string cuda_renderer_ptx;
    if (!OIIO::Filesystem::read_text_file("cuda_grid_renderer.ptx",
                                          cuda_renderer_ptx)) {
        std::cerr << "Could not read cuda_grid_renderer.ptx" << std::endl;
    }

    std::string rend_lib_ptx;
    if (!OIIO::Filesystem::read_text_file("rend_lib.ptx", rend_lib_ptx)) {
        std::cerr << "Could not read rend_lib.ptx" << std::endl;
    }

    // Uncomment these to inspect the generated PTX
    // OIIO::Filesystem::write_text_file("se_renderer.ptx", cuda_renderer_ptx);
    // OIIO::Filesystem::write_text_file("se_trampoline.ptx", ptx_trampoline);
    // OIIO::Filesystem::write_text_file("se_shader.ptx", ptx_shader);
    // OIIO::Filesystem::write_text_file("se_strlib.ptx", ptx_strlib);

    // link everything together
    CUlinkState link_state;
    CU_CHECK(cuLinkCreate(0, nullptr, nullptr, &link_state));
    CU_CHECK(cuLinkAddData(link_state, CU_JIT_INPUT_PTX,
                           (void*)cuda_renderer_ptx.c_str(),
                           cuda_renderer_ptx.size(), "cuda_grid_renderer.ptx",
                           0, nullptr, nullptr));
    CU_CHECK(cuLinkAddData(link_state, CU_JIT_INPUT_PTX,
                           (void*)rend_lib_ptx.c_str(), rend_lib_ptx.size(),
                           "rend_lib.ptx", 0, nullptr, nullptr));
    CU_CHECK(cuLinkAddData(link_state, CU_JIT_INPUT_PTX,
                           (void*)ptx_trampoline.c_str(), ptx_trampoline.size(),
                           "trampoline.ptx", 0, nullptr, nullptr));
    CU_CHECK(cuLinkAddData(link_state, CU_JIT_INPUT_PTX,
                           (void*)ptx_shader.c_str(), ptx_shader.size(),
                           "shader.ptx", 0, nullptr, nullptr));
    CU_CHECK(cuLinkAddData(link_state, CU_JIT_INPUT_PTX,
                           (void*)ptx_strlib.c_str(), ptx_strlib.size(),
                           "strlib.ptx", 0, nullptr, nullptr));
    void* cubin;
    size_t cubin_size;
    CU_CHECK(cuLinkComplete(link_state, &cubin, &cubin_size));

    CUmodule mod_renderer;
    CU_CHECK(cuModuleLoadData(&mod_renderer, cubin));

    CUfunction fun_renderer_entry;
    CU_CHECK(cuModuleGetFunction(&fun_renderer_entry, mod_renderer, "shade"));

    // -------------------------------------------------------------------------
    // Render
    //

    // need to increase the stack size because we recursively explore the
    // evaluated closures
    size_t stack_size = 0;
    CUDA_CHECK(cudaDeviceGetLimit(&stack_size, cudaLimitStackSize));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

    CUdeviceptr d_output_buffer;
    CUDA_CHECK(cudaMalloc((void**)&d_output_buffer, w * h * 3 * sizeof(float)));
    CUDA_SYNC_CHECK();
    int block_size   = 8;
    int num_blocks_x = w / block_size;
    int num_blocks_y = h / block_size;

    void* params[] = {
        &d_output_buffer,
        &w,
        &h,
    };

    // Launch kernel
    CU_CHECK(cuLaunchKernel(fun_renderer_entry, num_blocks_x, num_blocks_y, 1,
                            block_size, block_size, 1, 0, stream, params,
                            nullptr))

    CUDA_SYNC_CHECK();

    // Write images
    float* output_buffer = new float[w * h * 3];
    CU_CHECK(cuMemcpyDtoH(output_buffer, d_output_buffer,
                          sizeof(float) * 3 * w * h));
    CUDA_SYNC_CHECK();
    auto out = ImageOutput::create("out.exr");
    if (!out) {
        std::cerr << "Failed to open out.exr" << std::endl;
        return 2;
    }
    out->open("out.exr", ImageSpec(w, h, 3, TypeDesc::FLOAT));
    out->write_image(TypeDesc::FLOAT, output_buffer);
    out->close();
    delete[] output_buffer;
}

// anonymous namespace
namespace {

// these structures hold the parameters of each closure type
// they will be contained inside ClosureComponent
struct EmptyParams {
};
struct DiffuseParams {
    Vec3 N;
    ustring label;
};
struct OrenNayarParams {
    Vec3 N;
    float sigma;
};
struct PhongParams {
    Vec3 N;
    float exponent;
    ustring label;
};
struct WardParams {
    Vec3 N, T;
    float ax, ay;
};
struct ReflectionParams {
    Vec3 N;
    float eta;
};
struct RefractionParams {
    Vec3 N;
    float eta;
};
struct MicrofacetParams {
    ustring dist;
    Vec3 N, U;
    float xalpha, yalpha, eta;
    int refract;
};
struct DebugParams {
    ustring tag;
};

}  // anonymous namespace

void
register_closures(ShadingSystem& ss)
{
    // Describe the memory layout of each closure type to the OSL runtime
    enum { MaxParams = 32 };
    struct BuiltinClosures {
        const char* name;
        int id;
        ClosureParam params[MaxParams];  // upper bound
    };
    BuiltinClosures builtins[] = {
        { "emission", EMISSION_ID, { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "background", BACKGROUND_ID, { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "diffuse",
          DIFFUSE_ID,
          { CLOSURE_VECTOR_PARAM(DiffuseParams, N),
            CLOSURE_STRING_KEYPARAM(DiffuseParams, label,
                                    "label"),  // example of custom key param
            CLOSURE_FINISH_PARAM(DiffuseParams) } },
        { "oren_nayar",
          OREN_NAYAR_ID,
          { CLOSURE_VECTOR_PARAM(OrenNayarParams, N),
            CLOSURE_FLOAT_PARAM(OrenNayarParams, sigma),
            CLOSURE_FINISH_PARAM(OrenNayarParams) } },
        { "translucent",
          TRANSLUCENT_ID,
          { CLOSURE_VECTOR_PARAM(DiffuseParams, N),
            CLOSURE_FINISH_PARAM(DiffuseParams) } },
        { "phong",
          PHONG_ID,
          { CLOSURE_VECTOR_PARAM(PhongParams, N),
            CLOSURE_FLOAT_PARAM(PhongParams, exponent),
            CLOSURE_STRING_KEYPARAM(PhongParams, label,
                                    "label"),  // example of custom key param
            CLOSURE_FINISH_PARAM(PhongParams) } },
        { "ward",
          WARD_ID,
          { CLOSURE_VECTOR_PARAM(WardParams, N),
            CLOSURE_VECTOR_PARAM(WardParams, T),
            CLOSURE_FLOAT_PARAM(WardParams, ax),
            CLOSURE_FLOAT_PARAM(WardParams, ay),
            CLOSURE_FINISH_PARAM(WardParams) } },
        { "microfacet",
          MICROFACET_ID,
          { CLOSURE_STRING_PARAM(MicrofacetParams, dist),
            CLOSURE_VECTOR_PARAM(MicrofacetParams, N),
            CLOSURE_VECTOR_PARAM(MicrofacetParams, U),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, xalpha),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, yalpha),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, eta),
            CLOSURE_INT_PARAM(MicrofacetParams, refract),
            CLOSURE_FINISH_PARAM(MicrofacetParams) } },
        { "reflection",
          REFLECTION_ID,
          { CLOSURE_VECTOR_PARAM(ReflectionParams, N),
            CLOSURE_FINISH_PARAM(ReflectionParams) } },
        { "reflection",
          FRESNEL_REFLECTION_ID,
          { CLOSURE_VECTOR_PARAM(ReflectionParams, N),
            CLOSURE_FLOAT_PARAM(ReflectionParams, eta),
            CLOSURE_FINISH_PARAM(ReflectionParams) } },
        { "refraction",
          REFRACTION_ID,
          { CLOSURE_VECTOR_PARAM(RefractionParams, N),
            CLOSURE_FLOAT_PARAM(RefractionParams, eta),
            CLOSURE_FINISH_PARAM(RefractionParams) } },
        { "transparent", TRANSPARENT_ID, { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "debug",
          DEBUG_ID,
          { CLOSURE_STRING_PARAM(DebugParams, tag),
            CLOSURE_FINISH_PARAM(DebugParams) } },
        { "holdout", HOLDOUT_ID, { CLOSURE_FINISH_PARAM(EmptyParams) } }
    };

    for (const auto& b : builtins) {
        ss.register_closure(b.name, b.id, b.params, nullptr, nullptr);
    }
}

const char* cuda_compile_options[] = { "--gpu-architecture=compute_35",
                                       "--use_fast_math", "-dc",
                                       "--std=c++11" };

std::string
build_trampoline_ptx(OSL::ShaderGroup& group, std::string init_name,
                     std::string entry_name)
{
    std::stringstream ss;
    ss << "class ShaderGlobals;\n";
    ss << "extern \"C\" __device__ void " << init_name
       << "(ShaderGlobals*,void*);\n";
    ss << "extern \"C\" __device__ void " << entry_name
       << "(ShaderGlobals*,void*);\n";
    ss << "extern \"C\" __device__ void __osl__init(ShaderGlobals* sg, void* "
          "params) { "
       << init_name << "(sg, params); }\n";
    ss << "extern \"C\" __device__ void __osl__entry(ShaderGlobals* sg, void* "
          "params) { "
       << entry_name << "(sg, params); }\n";

    auto cu_trampoline = ss.str();
    nvrtcProgram prg_trampoline;
    int num_compile_flags = int(sizeof(cuda_compile_options)
                                / sizeof(cuda_compile_options[0]));
    size_t cuda_log_size;
    NVRTC_CHECK(nvrtcCreateProgram(&prg_trampoline, cu_trampoline.c_str(),
                                   "trampoline", 0, nullptr, nullptr));
    auto compileResult = nvrtcCompileProgram(prg_trampoline, num_compile_flags,
                                             cuda_compile_options);
    if (compileResult != NVRTC_SUCCESS) {
        NVRTC_CHECK(nvrtcGetProgramLogSize(prg_trampoline, &cuda_log_size));
        std::vector<char> cuda_log(cuda_log_size + 1);
        NVRTC_CHECK(nvrtcGetProgramLog(prg_trampoline, cuda_log.data()));
        cuda_log.back() = 0;
        std::stringstream ss;
        ss << "nvrtcCompileProgram failure for: " << cu_trampoline
           << "====================================\n"
           << cuda_log.data();
        throw std::runtime_error(ss.str());
    }

    size_t ptx_trampoline_size;
    NVRTC_CHECK(nvrtcGetPTXSize(prg_trampoline, &ptx_trampoline_size));
    std::vector<char> ptx_trampoline(ptx_trampoline_size);
    NVRTC_CHECK(nvrtcGetPTX(prg_trampoline, ptx_trampoline.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&prg_trampoline));
    std::string ptx_trampoline_string(ptx_trampoline.begin(),
                                      ptx_trampoline.end());
    return ptx_trampoline_string;
}

std::string
build_string_table_ptx(const CudaGridRenderer& rs)
{
    nvrtcProgram str_lib;

    auto extractNamespaces = [](const OIIO::ustring& s) {
        const char* str = s.c_str();
        std::vector<std::string> ns;
        do {
            const char* begin = str;
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
    for (auto&& gvar : rs.globals_map()) {
        // std::cout << "global: " << gvar.first << " -> " << gvar.second
        //           << std::endl;
        std::vector<std::string> var_ns = extractNamespaces(gvar.first);

        // build namespace
        for (size_t i = 0; i < var_ns.size() - 1; i++)
            strlib_ss << "namespace " << var_ns[i] << " {\n";

        strlib_ss << "__device__ DeviceString " << var_ns.back()
                  << " = { (const char *)" << gvar.second << "};\n";
        // close namespace up
        for (size_t i = 0; i < var_ns.size() - 1; i++)
            strlib_ss << "}\n";
    }

    strlib_ss << "\n";
    strlib_ss << "extern \"C\" __global__ void "
                 "__direct_callable__strlib_dummy(int *j)\n";
    strlib_ss << "{\n";
    strlib_ss << "   // must have a __direct_callable__ function for the "
                 "module to compile\n";
    strlib_ss << "}\n";

    int num_compile_flags = int(sizeof(cuda_compile_options)
                                / sizeof(cuda_compile_options[0]));
    size_t str_lib_size, cuda_log_size;

    std::string cuda_string = strlib_ss.str();
    // std::cout << "str_lib: \n\n" << cuda_string << std::endl;

    NVRTC_CHECK(nvrtcCreateProgram(&str_lib, cuda_string.c_str(),
                                   "cuda_strng_library",
                                   0,          // number of headers
                                   nullptr,    // header paths
                                   nullptr));  // header files
    nvrtcResult compileResult = nvrtcCompileProgram(str_lib, num_compile_flags,
                                                    cuda_compile_options);
    if (compileResult != NVRTC_SUCCESS) {
        NVRTC_CHECK(nvrtcGetProgramLogSize(str_lib, &cuda_log_size));
        std::vector<char> cuda_log(cuda_log_size + 1);
        NVRTC_CHECK(nvrtcGetProgramLog(str_lib, cuda_log.data()));
        cuda_log.back() = 0;
        std::stringstream ss;
        ss << "nvrtcCompileProgram failure for: " << cuda_string
           << "====================================\n"
           << cuda_log.data();
        throw std::runtime_error(ss.str());
    }
    NVRTC_CHECK(nvrtcGetPTXSize(str_lib, &str_lib_size));
    std::vector<char> str_lib_ptx(str_lib_size);
    NVRTC_CHECK(nvrtcGetPTX(str_lib, str_lib_ptx.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&str_lib));

    std::string strlib_string(str_lib_ptx.begin(), str_lib_ptx.end());

    return strlib_string;
}
