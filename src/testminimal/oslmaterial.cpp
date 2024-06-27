// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include "oslmaterial.h"
#include <iostream>

using std::cout;
using std::endl;

#if OSL_USE_BATCHED
template<int batch_width>
CustomBatchedRendererServices<batch_width>::CustomBatchedRendererServices(
    BatchedOSLMaterial<batch_width>& m)
    : OSL::BatchedRendererServices<batch_width>(m.texturesys()), m_sr(m)
{
}
#endif

OSLMaterial::OSLMaterial() {}

#if OSL_USE_BATCHED
template<int batch_width>
BatchedOSLMaterial<batch_width>::BatchedOSLMaterial() : m_batch(*this)
{
}

template BatchedOSLMaterial<8>::BatchedOSLMaterial();
template BatchedOSLMaterial<16>::BatchedOSLMaterial();
#endif

// Supported closures and parameters
struct EmptyParams {};

enum ClosureIDs {
    EMISSION_ID,
    BACKGROUND_ID,
    MICROFACET_ID,
};

struct MicrofacetParams {
    OSL::ustringhash dist;
    OSL::Vec3 N, U;
    float xalpha, yalpha, eta;
    int refract;
};

void
register_closures(OSL::ShadingSystem* ss)
{
    // "Describe the memory layout of each closure type to the OSL runtime"
    constexpr int MaxParams = 32;
    struct BuiltinClosures {
        const char* name;
        int id;
        OSL::ClosureParam params[MaxParams];  // "upper bound"
    };

    using namespace OSL;

    // Closures with support built into OSL, connected by the 1st string
    BuiltinClosures supported[] = {
        { "emission", EMISSION_ID, { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "background", BACKGROUND_ID, { CLOSURE_FINISH_PARAM(EmptyParams) } },
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
    };
    // Closure registration here enables that type of closure, when executing or compiling a shader
    for (const BuiltinClosures& c : supported)
        ss->register_closure(c.name, c.id, c.params, nullptr, nullptr);
}

void
process_bsdf_closure(const OSL::ClosureColor* closure)
{
    static const ::OSL::ustringhash uh_ggx(OIIO::Strutil::strhash("ggx"));
    //static const ::OSL::ustringhash uh_beckmann(OIIO::Strutil::strhash("beckmann"));
    if (!closure)
        return;
    switch (closure->id) {
    case OSL::ClosureColor::MUL: {
        process_bsdf_closure(closure->as_mul()->closure);
        break;
    }
    case OSL::ClosureColor::ADD: {
        process_bsdf_closure(closure->as_add()->closureA);
        process_bsdf_closure(closure->as_add()->closureB);
        break;
    }
    default: {
        const OSL::ClosureComponent* comp = closure->as_comp();
        switch (comp->id) {
        case EMISSION_ID: cout << "parsing emission closure" << endl; break;
        case MICROFACET_ID: {
            cout << "parsing microfacet closure" << endl;
            const MicrofacetParams* mp = comp->as<MicrofacetParams>();
            if (mp->dist.hash() == uh_ggx.hash()) {
                cout << "uh_ggx" << endl;
            } else {
                cout << "uh_beckmann or default" << endl;
            }
        } break;
        default:
            OSL_ASSERT(false && "Invalid closure invoked in surface shader");
            break;
        }
    } break;
    }
}

void
OSLMaterial::run_test(OSL::ShadingSystem* ss, OSL::PerThreadInfo* thread_info,
                      OSL::ShadingContext* context, char* shader_name)
{
    register_closures(ss);
    OSL::ShaderGlobals globals;
    globals_from_hit(globals);

    std::vector<std::string> options;

    // Create a new shader group
    m_shaders.emplace_back();
    m_shaders[0]              = ss->ShaderGroupBegin(std::to_string(0));
    OSL::ShaderGroupRef group = m_shaders[0];

    //{
    //    OSL::OSLCompiler compiler;
    //    std::string name = std::string(shader_name) + ".osl";
    //    compiler.compile(name.c_str(), options);
    //}

    ss->Shader(*group, "surface", shader_name, "Test");
    ss->ShaderGroupEnd(*group);

    ss->execute(context, *group, globals);
    const OSL::ClosureColor* closure = globals.Ci;
    process_bsdf_closure(closure);
}

#if OSL_USE_BATCHED
template<int batch_width>
void
BatchedOSLMaterial<batch_width>::run_test(OSL::ShadingSystem* ss,
                                          OSL::PerThreadInfo* thread_info,
                                          OSL::ShadingContext* context,
                                          char* shader_name)
{
    register_closures(ss);
    OSL::BatchedShaderGlobals<batch_width> batched_globals;

    m_batch.globals_from_hit(batched_globals);

    std::vector<std::string> options;

    // Create a new shader group
    m_shaders.emplace_back();
    m_shaders[0]              = ss->ShaderGroupBegin(std::to_string(0));
    OSL::ShaderGroupRef group = m_shaders[0];

    //{
    //    OSL::OSLCompiler compiler;
    //    std::string name = std::string(shader_name) + ".osl";
    //    compiler.compile(name.c_str(), options);
    //}

    ss->Shader(*group, "surface", shader_name, "Test");
    ss->ShaderGroupEnd(*group);

    // Run the shader that was just created
    OSL::Block<int, batch_width> wide_shadeindex_block;
    char* userdata_base_ptr = NULL;
    char* output_base_ptr   = NULL;
    ss->batched<batch_width>().execute(*context, *group, batch_width,
                                       wide_shadeindex_block, batched_globals,
                                       userdata_base_ptr, output_base_ptr);
    const OSL::ClosureColor* closure = batched_globals.varying.Ci[0];
    process_bsdf_closure(closure);
}

template void
BatchedOSLMaterial<8>::run_test(OSL::ShadingSystem* ss,
                                OSL::PerThreadInfo* thread_info,
                                OSL::ShadingContext* context,
                                char* shader_name);
template void
BatchedOSLMaterial<16>::run_test(OSL::ShadingSystem* ss,
                                 OSL::PerThreadInfo* thread_info,
                                 OSL::ShadingContext* context,
                                 char* shader_name);
#endif
