#include "shading.h"
#include <OSL/device_string.h>
#include <OSL/genclosure.h>

using namespace OSL;

namespace { // anonymous namespace

// unique identifier for each closure supported by testrender
enum ClosureIDs {
    EMISSION_ID = 1,
    BACKGROUND_ID,
    DIFFUSE_ID,
    OREN_NAYAR_ID,
    TRANSLUCENT_ID,
    PHONG_ID,
    WARD_ID,
    MICROFACET_ID,
    REFLECTION_ID,
    FRESNEL_REFLECTION_ID,
    REFRACTION_ID,
    TRANSPARENT_ID,
};

// these structures hold the parameters of each closure type
// they will be contained inside ClosureComponent
struct EmptyParams      { };
struct DiffuseParams    { Vec3 N; };
struct OrenNayarParams  { Vec3 N; float sigma; };
struct PhongParams      { Vec3 N; float exponent; };
struct WardParams       { Vec3 N, T; float ax, ay; };
struct ReflectionParams { Vec3 N; float eta; };
struct RefractionParams { Vec3 N; float eta; };
struct MicrofacetParams { device_string dist; Vec3 N, U; float xalpha, yalpha, eta; int refract; };

} // anonymous namespace

OSL_NAMESPACE_ENTER

void register_closures(OSL::ShadingSystem* shadingsys) {
    // Describe the memory layout of each closure type to the OSL runtime
    enum { MaxParams = 32 };
    struct BuiltinClosures {
        const char* name;
        int id;
        ClosureParam params[MaxParams]; // upper bound
    };
    BuiltinClosures builtins[] = {
        { "emission"   , EMISSION_ID,           { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "background" , BACKGROUND_ID,         { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "diffuse"    , DIFFUSE_ID,            { CLOSURE_VECTOR_PARAM(DiffuseParams, N),
                                                  CLOSURE_FINISH_PARAM(DiffuseParams) } },
        { "oren_nayar" , OREN_NAYAR_ID,         { CLOSURE_VECTOR_PARAM(OrenNayarParams, N),
                                                  CLOSURE_FLOAT_PARAM (OrenNayarParams, sigma),
                                                  CLOSURE_FINISH_PARAM(OrenNayarParams) } },
        { "translucent", TRANSLUCENT_ID,        { CLOSURE_VECTOR_PARAM(DiffuseParams, N),
                                                  CLOSURE_FINISH_PARAM(DiffuseParams) } },
        { "phong"      , PHONG_ID,              { CLOSURE_VECTOR_PARAM(PhongParams, N),
                                                  CLOSURE_FLOAT_PARAM (PhongParams, exponent),
                                                  CLOSURE_FINISH_PARAM(PhongParams) } },
        { "ward"       , WARD_ID,               { CLOSURE_VECTOR_PARAM(WardParams, N),
                                                  CLOSURE_VECTOR_PARAM(WardParams, T),
                                                  CLOSURE_FLOAT_PARAM (WardParams, ax),
                                                  CLOSURE_FLOAT_PARAM (WardParams, ay),
                                                  CLOSURE_FINISH_PARAM(WardParams) } },
        { "microfacet", MICROFACET_ID,          { CLOSURE_DEVICE_STRING_PARAM(MicrofacetParams, dist),
                                                  CLOSURE_VECTOR_PARAM(MicrofacetParams, N),
                                                  CLOSURE_VECTOR_PARAM(MicrofacetParams, U),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, xalpha),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, yalpha),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, eta),
                                                  CLOSURE_INT_PARAM   (MicrofacetParams, refract),
                                                  CLOSURE_FINISH_PARAM(MicrofacetParams) } },
        { "reflection" , REFLECTION_ID,         { CLOSURE_VECTOR_PARAM(ReflectionParams, N),
                                                  CLOSURE_FINISH_PARAM(ReflectionParams) } },
        { "reflection" , FRESNEL_REFLECTION_ID, { CLOSURE_VECTOR_PARAM(ReflectionParams, N),
                                                  CLOSURE_FLOAT_PARAM (ReflectionParams, eta),
                                                  CLOSURE_FINISH_PARAM(ReflectionParams) } },
        { "refraction" , REFRACTION_ID,         { CLOSURE_VECTOR_PARAM(RefractionParams, N),
                                                  CLOSURE_FLOAT_PARAM (RefractionParams, eta),
                                                  CLOSURE_FINISH_PARAM(RefractionParams) } },
        { "transparent", TRANSPARENT_ID,        { CLOSURE_FINISH_PARAM(EmptyParams) } },
        // mark end of the array
        { NULL, 0, {} }
    };

    for (int i = 0; builtins[i].name; i++) {
        shadingsys->register_closure(
            builtins[i].name,
            builtins[i].id,
            builtins[i].params,
            NULL, NULL);
    }
}

void register_string_tags(OSL::ShadingSystem* shadingsys) {
#if 0
    // The renderer can optionally register strings with specific tags.
    // The registration will fail if the given string has already been
    // registered with a different tag.
    shadingsys->register_string_tag("my_cool_string", 1234u);
    uint64_t tag = shadingsys->lookup_string_tag("my_cool_string");
    ASSERT (tag != StringTags::UNKNOWN_STRING && "Unable to register string");

    // Similarly, the renderer can check whether or not a string has been
    // registered.
    uint64_t fake_tag = shadingsys->lookup_string_tag("my_fake_string");
    ASSERT (fake_tag == StringTags::UNKNOWN_STRING && "Unregistered string check failed");
#endif
}

OSL_NAMESPACE_EXIT
