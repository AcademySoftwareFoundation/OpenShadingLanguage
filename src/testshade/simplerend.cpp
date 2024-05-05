// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <OpenImageIO/imagebufalgo.h>

#include <OSL/encodedtypes.h>
#include <OSL/genclosure.h>
#include <OSL/journal.h>
#include <OSL/oslexec.h>

#include "simplerend.h"


// Instance all global string variables used by SimpleRend.cpp or
// rs_simplerend.cpp (free functions).
// NOTE:  C linkage with a "RS_" prefix is used to allow for unmangled
// non-colliding global symbol names, so its easier to pass them to
// OSL::register_JIT_Global(name, addr) for host execution
extern "C" {
#define RS_STRDECL(str, var_name) OSL::ustring RS_##var_name { str };
#include "rs_strdecls.h"
#undef RS_STRDECL
}


using namespace OSL;



// anonymous namespace
namespace {

// unique identifier for each closure supported by testshade
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
    DEBUG_ID,
    HOLDOUT_ID,
    PARAMETER_TEST_ID,
};

// these structures hold the parameters of each closure type
// they will be contained inside ClosureComponent
struct EmptyParams {};
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
struct ParameterTestParams {
    int int_param;
    float float_param;
    Color3 color_param;
    Vec3 vector_param;
    ustring string_param;
    int int_array[5];
    Vec3 vector_array[5];
    Color3 color_array[5];
    float float_array[5];
    ustring string_array[5];
    int int_key;
    float float_key;
    Color3 color_key;
    Vec3 vector_key;
    ustring string_key;
};

}  // anonymous namespace


OSL_NAMESPACE_ENTER

static ustring u_camera("camera"), u_screen("screen");
static ustring u_NDC("NDC"), u_raster("raster");
static ustring u_perspective("perspective");
static ustring u_s("s"), u_t("t");
static ustring u_red("red"), u_green("green"), u_blue("blue");
static TypeDesc TypeFloatArray2(TypeDesc::FLOAT, 2);
static TypeDesc TypeFloatArray4(TypeDesc::FLOAT, 4);
static TypeDesc TypeIntArray2(TypeDesc::INT, 2);


void
register_closures(OSL::ShadingSystem* shadingsys)
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
        { "holdout", HOLDOUT_ID, { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "parameter_test",
          PARAMETER_TEST_ID,
          { CLOSURE_INT_PARAM(ParameterTestParams, int_param),
            CLOSURE_FLOAT_PARAM(ParameterTestParams, float_param),
            CLOSURE_COLOR_PARAM(ParameterTestParams, color_param),
            CLOSURE_VECTOR_PARAM(ParameterTestParams, vector_param),
            CLOSURE_STRING_PARAM(ParameterTestParams, string_param),
            CLOSURE_INT_ARRAY_PARAM(ParameterTestParams, int_array, 5),
            CLOSURE_VECTOR_ARRAY_PARAM(ParameterTestParams, vector_array, 5),
            CLOSURE_COLOR_ARRAY_PARAM(ParameterTestParams, color_array, 5),
            CLOSURE_FLOAT_ARRAY_PARAM(ParameterTestParams, float_array, 5),
            CLOSURE_STRING_ARRAY_PARAM(ParameterTestParams, string_array, 5),
            CLOSURE_INT_KEYPARAM(ParameterTestParams, int_key, "int_key"),
            CLOSURE_FLOAT_KEYPARAM(ParameterTestParams, float_key, "float_key"),
            CLOSURE_COLOR_KEYPARAM(ParameterTestParams, color_key, "color_key"),
            CLOSURE_VECTOR_KEYPARAM(ParameterTestParams, vector_key,
                                    "vector_key"),
            CLOSURE_STRING_KEYPARAM(ParameterTestParams, string_key,
                                    "string_key"),
            CLOSURE_FINISH_PARAM(ParameterTestParams) } }
    };

    for (const auto& b : builtins)
        shadingsys->register_closure(b.name, b.id, b.params, nullptr, nullptr);
}



SimpleRenderer::SimpleRenderer()
#if OSL_USE_BATCHED
    : m_batch_16_simple_renderer(*this), m_batch_8_simple_renderer(*this)
#endif
{
    Matrix44 M;
    M.makeIdentity();
    camera_params(M, u_perspective, 90.0f, 0.1f, 1000.0f, 256, 256);

    // Set up getters
    m_attr_getters[ustring("osl:version")] = &SimpleRenderer::get_osl_version;
    m_attr_getters[ustring("camera:resolution")]
        = &SimpleRenderer::get_camera_resolution;
    m_attr_getters[ustring("camera:projection")]
        = &SimpleRenderer::get_camera_projection;
    m_attr_getters[ustring("camera:pixelaspect")]
        = &SimpleRenderer::get_camera_pixelaspect;
    m_attr_getters[ustring("camera:screen_window")]
        = &SimpleRenderer::get_camera_screen_window;
    m_attr_getters[ustring("camera:fov")]  = &SimpleRenderer::get_camera_fov;
    m_attr_getters[ustring("camera:clip")] = &SimpleRenderer::get_camera_clip;
    m_attr_getters[ustring("camera:clip_near")]
        = &SimpleRenderer::get_camera_clip_near;
    m_attr_getters[ustring("camera:clip_far")]
        = &SimpleRenderer::get_camera_clip_far;
    m_attr_getters[ustring("camera:shutter")]
        = &SimpleRenderer::get_camera_shutter;
    m_attr_getters[ustring("camera:shutter_open")]
        = &SimpleRenderer::get_camera_shutter_open;
    m_attr_getters[ustring("camera:shutter_close")]
        = &SimpleRenderer::get_camera_shutter_close;
}


// Ensure destructor code gen happens in this .cpp
SimpleRenderer::~SimpleRenderer() {}

int
SimpleRenderer::supports(string_view feature) const
{
    if (m_use_rs_bitcode && feature == "build_attribute_getter")
        return true;
    return false;
}



OIIO::ParamValue*
SimpleRenderer::find_attribute(string_view name, TypeDesc searchtype,
                               bool casesensitive)
{
    auto iter = options.find(name, searchtype, casesensitive);
    if (iter != options.end())
        return &(*iter);
    return nullptr;
}



const OIIO::ParamValue*
SimpleRenderer::find_attribute(string_view name, TypeDesc searchtype,
                               bool casesensitive) const
{
    auto iter = options.find(name, searchtype, casesensitive);
    if (iter != options.end())
        return &(*iter);
    return nullptr;
}



void
SimpleRenderer::attribute(string_view name, TypeDesc type, const void* value)
{
    if (name.empty())  // Guard against bogus empty names
        return;
    // Don't allow duplicates
    auto f = find_attribute(name);
    if (!f) {
        options.resize(options.size() + 1);
        f = &options.back();
    }
    f->init(name, type, 1, value);
}

void
SimpleRenderer::register_JIT_Global_Variables()  //callable from testshade
{
#define RS_STRDECL(str, var_name) \
    OSL::register_JIT_Global(__OSL_STRINGIFY(RS_##var_name), &RS_##var_name);
#include "rs_strdecls.h"
#undef RS_STRDECL
}


void
SimpleRenderer::camera_params(const Matrix44& world_to_camera,
                              ustring projection, float hfov, float hither,
                              float yon, int xres, int yres)
{
    m_world_to_camera  = world_to_camera;
    m_projection       = projection;
    m_fov              = hfov;
    m_pixelaspect      = 1.0f;  // hard-coded
    m_hither           = hither;
    m_yon              = yon;
    m_shutter[0]       = 0.0f;
    m_shutter[1]       = 1.0f;  // hard-coded
    float frame_aspect = float(xres) / float(yres) * m_pixelaspect;
    m_screen_window[0] = -frame_aspect;
    m_screen_window[1] = -1.0f;
    m_screen_window[2] = frame_aspect;
    m_screen_window[3] = 1.0f;
    m_xres             = xres;
    m_yres             = yres;
}



bool
SimpleRenderer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                           TransformationPtr xform, float /*time*/)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    result = *reinterpret_cast<const Matrix44*>(xform);
    return true;
}



bool
SimpleRenderer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                           ustringhash from, float /*time*/)
{
    TransformMap::const_iterator found = m_named_xforms.find(from);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        return true;
    } else {
        return false;
    }
}



bool
SimpleRenderer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                           TransformationPtr xform)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    result = *(OSL::Matrix44*)xform;
    return true;
}



bool
SimpleRenderer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                           ustringhash from)
{
    // SimpleRenderer doesn't understand motion blur, so we never fail
    // on account of time-varying transformations.
    TransformMap::const_iterator found = m_named_xforms.find(from);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        return true;
    } else {
        return false;
    }
}



bool
SimpleRenderer::get_inverse_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                                   ustringhash to, float /*time*/)
{
    if (to == u_camera || to == u_screen || to == u_NDC || to == u_raster) {
        Matrix44 M = m_world_to_camera;
        if (to == u_screen || to == u_NDC || to == u_raster) {
            float depthrange = (double)m_yon - (double)m_hither;
            if (m_projection == u_perspective) {
                float tanhalffov = tanf(0.5f * m_fov * M_PI / 180.0);
                Matrix44 camera_to_screen(1 / tanhalffov, 0, 0, 0, 0,
                                          1 / tanhalffov, 0, 0, 0, 0,
                                          m_yon / depthrange, 1, 0, 0,
                                          -m_yon * m_hither / depthrange, 0);
                M = M * camera_to_screen;
            } else {
                Matrix44 camera_to_screen(1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                          1 / depthrange, 0, 0, 0,
                                          -m_hither / depthrange, 1);
                M = M * camera_to_screen;
            }
            if (to == u_NDC || to == u_raster) {
                float screenleft = -1.0, screenwidth = 2.0;
                float screenbottom = -1.0, screenheight = 2.0;
                Matrix44 screen_to_ndc(1 / screenwidth, 0, 0, 0, 0,
                                       1 / screenheight, 0, 0, 0, 0, 1, 0,
                                       -screenleft / screenwidth,
                                       -screenbottom / screenheight, 0, 1);
                M = M * screen_to_ndc;
                if (to == u_raster) {
                    Matrix44 ndc_to_raster(m_xres, 0, 0, 0, 0, m_yres, 0, 0, 0,
                                           0, 1, 0, 0, 0, 0, 1);
                    M = M * ndc_to_raster;
                }
            }
        }
        result = M;
        return true;
    }

    TransformMap::const_iterator found = m_named_xforms.find(to);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        result.invert();
        return true;
    } else {
        return false;
    }
}



void
SimpleRenderer::name_transform(const char* name, const OSL::Matrix44& xform)
{
    std::shared_ptr<Transformation> M(new OSL::Matrix44(xform));
    m_named_xforms[ustringhash(name)] = M;
}



bool
SimpleRenderer::get_array_attribute(ShaderGlobals* sg, bool derivatives,
                                    ustringhash object, TypeDesc type,
                                    ustringhash name, int index, void* val)
{
    AttrGetterMap::const_iterator g = m_attr_getters.find(name);
    if (g != m_attr_getters.end()) {
        AttrGetter getter = g->second;
        return (this->*(getter))(sg, derivatives, object, type, name, val);
    }

    // In order to test getattribute(), respond positively to
    // "options"/"blahblah"
    if (object == "options" && name == "blahblah" && type == TypeFloat) {
        *(float*)val = 3.14159;
        return true;
    }

    if (object.empty() && name == "shading:index" && type == TypeInt) {
        *(int*)val = OSL::get_shade_index(sg);
        return true;
    }

    // If no named attribute was found, allow userdata to bind to the
    // attribute request.
    if (object.empty() && index == -1)
        return get_userdata(derivatives, name, type, sg, val);

    return false;
}



bool
SimpleRenderer::get_attribute(ShaderGlobals* sg, bool derivatives,
                              ustringhash object, TypeDesc type,
                              ustringhash name, void* val)
{
    return get_array_attribute(sg, derivatives, object, type, name, -1, val);
}



bool
SimpleRenderer::get_userdata(bool derivatives, ustringhash name, TypeDesc type,
                             ShaderGlobals* sg, void* val)
{
    // Just to illustrate how this works, respect s and t userdata, filled
    // in with the uv coordinates.  In a real renderer, it would probably
    // look up something specific to the primitive, rather than have hard-
    // coded names.

    if (name == u_s && type == TypeFloat) {
        ((float*)val)[0] = sg->u;
        if (derivatives) {
            ((float*)val)[1] = sg->dudx;
            ((float*)val)[2] = sg->dudy;
        }
        return true;
    }
    if (name == u_t && type == TypeFloat) {
        ((float*)val)[0] = sg->v;
        if (derivatives) {
            ((float*)val)[1] = sg->dvdx;
            ((float*)val)[2] = sg->dvdy;
        }
        return true;
    }
    if (name == u_red && type == TypeFloat && sg->P.x > 0.5f) {
        ((float*)val)[0] = sg->u;
        if (derivatives) {
            ((float*)val)[1] = sg->dudx;
            ((float*)val)[2] = sg->dudy;
        }
        return true;
    }
    if (name == u_green && type == TypeFloat && sg->P.x < 0.5f) {
        ((float*)val)[0] = sg->v;
        if (derivatives) {
            ((float*)val)[1] = sg->dvdx;
            ((float*)val)[2] = sg->dvdy;
        }
        return true;
    }
    if (name == u_blue && type == TypeFloat
        && ((static_cast<int>(sg->P.y * 12) % 2) == 0)) {
        ((float*)val)[0] = 1.0f - sg->u;
        if (derivatives) {
            ((float*)val)[1] = -sg->dudx;
            ((float*)val)[2] = -sg->dudy;
        }
        return true;
    }

    if (const OIIO::ParamValue* p = userdata.find_pv(name, type)) {
        size_t size = p->type().size();
        memcpy(val, p->data(), size);
        if (derivatives)
            memset((char*)val + size, 0, 2 * size);
        return true;
    }

    return false;
}


void
SimpleRenderer::build_attribute_getter(
    const ShaderGroup& group, bool is_object_lookup, const ustring* object_name,
    const ustring* attribute_name, bool is_array_lookup, const int* array_index,
    TypeDesc type, bool derivatives, AttributeGetterSpec& spec)
{
    static const OIIO::ustring rs_get_attribute_constant_int(
        "rs_get_attribute_constant_int");
    static const OIIO::ustring rs_get_attribute_constant_int2(
        "rs_get_attribute_constant_int2");
    static const OIIO::ustring rs_get_attribute_constant_int3(
        "rs_get_attribute_constant_int3");
    static const OIIO::ustring rs_get_attribute_constant_int4(
        "rs_get_attribute_constant_int4");

    static const OIIO::ustring rs_get_attribute_constant_float(
        "rs_get_attribute_constant_float");
    static const OIIO::ustring rs_get_attribute_constant_float2(
        "rs_get_attribute_constant_float2");
    static const OIIO::ustring rs_get_attribute_constant_float3(
        "rs_get_attribute_constant_float3");
    static const OIIO::ustring rs_get_attribute_constant_float4(
        "rs_get_attribute_constant_float4");

    static const OIIO::ustring rs_get_shade_index("rs_get_shade_index");

    static const OIIO::ustring rs_get_attribute("rs_get_attribute");

    if (m_use_rs_bitcode) {
        // For demonstration purposes we show how to build functions taking
        // advantage of known compile time information. Here we simply select
        // which function to call based on what we know at this point.

        if (object_name && object_name->empty() && attribute_name) {
            if (const OIIO::ParamValue* p = userdata.find_pv(*attribute_name,
                                                             type)) {
                if (p->type().basetype == OIIO::TypeDesc::INT) {
                    if (p->type().aggregate == 1) {
                        spec.set(rs_get_attribute_constant_int,
                                 ((int*)p->data())[0]);
                        return;
                    } else if (p->type().aggregate == 2) {
                        spec.set(rs_get_attribute_constant_int2,
                                 ((int*)p->data())[0], ((int*)p->data())[1]);
                        return;
                    } else if (p->type().aggregate == 3) {
                        spec.set(rs_get_attribute_constant_int3,
                                 ((int*)p->data())[0], ((int*)p->data())[1],
                                 ((int*)p->data())[2]);
                        return;
                    } else if (p->type().aggregate == 4) {
                        spec.set(rs_get_attribute_constant_int4,
                                 ((int*)p->data())[0], ((int*)p->data())[1],
                                 ((int*)p->data())[2], ((int*)p->data())[3]);
                        return;
                    }
                } else if (p->type().basetype == OIIO::TypeDesc::FLOAT) {
                    if (p->type().aggregate == 1) {
                        spec.set(rs_get_attribute_constant_float,
                                 ((float*)p->data())[0],
                                 AttributeSpecBuiltinArg::Derivatives);
                        return;
                    } else if (p->type().aggregate == 2) {
                        spec.set(rs_get_attribute_constant_float2,
                                 ((float*)p->data())[0], ((float*)p->data())[1],
                                 AttributeSpecBuiltinArg::Derivatives);
                        return;
                    } else if (p->type().aggregate == 3) {
                        spec.set(rs_get_attribute_constant_float3,
                                 ((float*)p->data())[0], ((float*)p->data())[1],
                                 ((float*)p->data())[2],
                                 AttributeSpecBuiltinArg::Derivatives);
                        return;
                    } else if (p->type().aggregate == 4) {
                        spec.set(rs_get_attribute_constant_float4,
                                 ((float*)p->data())[0], ((float*)p->data())[1],
                                 ((float*)p->data())[2], ((float*)p->data())[3],
                                 AttributeSpecBuiltinArg::Derivatives);
                        return;
                    }
                }
            }
        }

        if (object_name && *object_name == ustring("options") && attribute_name
            && *attribute_name == ustring("blahblah")
            && type == OSL::TypeFloat) {
            spec.set(rs_get_attribute_constant_float, 3.14159f,
                     AttributeSpecBuiltinArg::Derivatives);
        } else if (!is_object_lookup && attribute_name
                   && *attribute_name == ustring("shading:index")
                   && type == OSL::TypeInt) {
            spec.set(rs_get_shade_index,
                     AttributeSpecBuiltinArg::ShaderGlobalsPointer);
        } else {
            spec.set(rs_get_attribute,
                     AttributeSpecBuiltinArg::ShaderGlobalsPointer,
                     AttributeSpecBuiltinArg::ObjectName,
                     AttributeSpecBuiltinArg::AttributeName,
                     AttributeSpecBuiltinArg::Type,
                     AttributeSpecBuiltinArg::Derivatives,
                     AttributeSpecBuiltinArg::ArrayIndex);
        }
    }
}

bool
SimpleRenderer::trace(TraceOpt& options, ShaderGlobals* sg, const OSL::Vec3& P,
                      const OSL::Vec3& dPdx, const OSL::Vec3& dPdy,
                      const OSL::Vec3& R, const OSL::Vec3& dRdx,
                      const OSL::Vec3& dRdy)
{
    // Don't do real ray tracing, just
    // use source and direction to alter hit results
    // so they are repeatable values for testsuite.
    float dot_val = P.dot(R);

    if ((sg->u) / dot_val > 0.5) {
        return true;  //1 in batched
    } else {
        return false;
    }
}


bool
SimpleRenderer::getmessage(ShaderGlobals* sg, ustringhash source_,
                           ustringhash name_, TypeDesc type, void* val,
                           bool derivatives)
{
    ustring source = ustring_from(source_);
    ustring name   = ustring_from(name_);
    OSL_ASSERT(source == ustring("trace"));
    // Don't have any real ray tracing results
    // so just fill in some repeatable values for testsuite
    if (sg->u > 0.5) {
        if (name == ustring("hitdist")) {
            if (type == TypeFloat) {
                *reinterpret_cast<float*>(val) = 0.5f;
            }
        }
        if (name == ustring("hit")) {
            if (type == TypeInt) {
                *reinterpret_cast<int*>(val) = 1;
            }
        }
        if (name == ustring("geom:name")) {
            if (type == TypeString) {
                *reinterpret_cast<ustring*>(val) = ustring("teapot");
            }
        }
        if (name == ustring("N")) {
            if (type == TypeNormal) {
                *reinterpret_cast<Vec3*>(val) = Vec3(1.0 - sg->v, 0.25,
                                                     1.0 - sg->u);
            } else {
                OSL_ASSERT(0 && "Oops");
            }
        }
        return true;  //1 in batched

    } else {
        if (name == ustring("hit")) {
            if (type == TypeInt) {
                *reinterpret_cast<int*>(val) = 0;
            }
        }
        return false;
    }
}



bool
SimpleRenderer::get_osl_version(ShaderGlobals* /*sg*/, bool /*derivs*/,
                                ustringhash /*object*/, TypeDesc type,
                                ustringhash /*name*/, void* val)
{
    if (type == TypeInt) {
        ((int*)val)[0] = OSL_VERSION;
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_resolution(ShaderGlobals* /*sg*/, bool /*derivs*/,
                                      ustringhash /*object*/, TypeDesc type,
                                      ustringhash /*name*/, void* val)
{
    if (type == TypeIntArray2) {
        ((int*)val)[0] = m_xres;
        ((int*)val)[1] = m_yres;
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_projection(ShaderGlobals* /*sg*/, bool /*derivs*/,
                                      ustringhash /*object*/, TypeDesc type,
                                      ustringhash /*name*/, void* val)
{
    if (type == TypeString) {
        ((ustring*)val)[0] = m_projection;
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_fov(ShaderGlobals* /*sg*/, bool derivs,
                               ustringhash /*object*/, TypeDesc type,
                               ustringhash /*name*/, void* val)
{
    // N.B. in a real renderer, this may be time-dependent
    if (type == TypeFloat) {
        ((float*)val)[0] = m_fov;
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_pixelaspect(ShaderGlobals* /*sg*/, bool derivs,
                                       ustringhash /*object*/, TypeDesc type,
                                       ustringhash /*name*/, void* val)
{
    if (type == TypeFloat) {
        ((float*)val)[0] = m_pixelaspect;
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_clip(ShaderGlobals* /*sg*/, bool derivs,
                                ustringhash /*object*/, TypeDesc type,
                                ustringhash /*name*/, void* val)
{
    if (type == TypeFloatArray2) {
        ((float*)val)[0] = m_hither;
        ((float*)val)[1] = m_yon;
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_clip_near(ShaderGlobals* /*sg*/, bool derivs,
                                     ustringhash /*object*/, TypeDesc type,
                                     ustringhash /*name*/, void* val)
{
    if (type == TypeFloat) {
        ((float*)val)[0] = m_hither;
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_clip_far(ShaderGlobals* /*sg*/, bool derivs,
                                    ustringhash /*object*/, TypeDesc type,
                                    ustringhash /*name*/, void* val)
{
    if (type == TypeFloat) {
        ((float*)val)[0] = m_yon;
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}



bool
SimpleRenderer::get_camera_shutter(ShaderGlobals* /*sg*/, bool derivs,
                                   ustringhash /*object*/, TypeDesc type,
                                   ustringhash /*name*/, void* val)
{
    if (type == TypeFloatArray2) {
        ((float*)val)[0] = m_shutter[0];
        ((float*)val)[1] = m_shutter[1];
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_shutter_open(ShaderGlobals* /*sg*/, bool derivs,
                                        ustringhash /*object*/, TypeDesc type,
                                        ustringhash /*name*/, void* val)
{
    if (type == TypeFloat) {
        ((float*)val)[0] = m_shutter[0];
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_shutter_close(ShaderGlobals* /*sg*/, bool derivs,
                                         ustringhash /*object*/, TypeDesc type,
                                         ustringhash /*name*/, void* val)
{
    if (type == TypeFloat) {
        ((float*)val)[0] = m_shutter[1];
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_screen_window(ShaderGlobals* /*sg*/, bool derivs,
                                         ustringhash /*object*/, TypeDesc type,
                                         ustringhash /*name*/, void* val)
{
    // N.B. in a real renderer, this may be time-dependent
    if (type == TypeFloatArray4) {
        ((float*)val)[0] = m_screen_window[0];
        ((float*)val)[1] = m_screen_window[1];
        ((float*)val)[2] = m_screen_window[2];
        ((float*)val)[3] = m_screen_window[3];
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}



bool
SimpleRenderer::add_output(string_view varname, string_view filename,
                           TypeDesc datatype, int nchannels)
{
    // FIXME: use name to figure out
    OIIO::ImageSpec spec(m_xres, m_yres, nchannels, datatype);
    m_outputvars.emplace_back(varname);
    m_outputbufs.emplace_back(
        new OIIO::ImageBuf(filename, spec, OIIO::InitializePixels::Yes));
    // OIIO::ImageBufAlgo::zero (*m_outputbufs.back());
    return true;
}


void
SimpleRenderer::export_state(RenderState& state) const
{
    state.xres   = m_xres;
    state.yres   = m_yres;
    state.fov    = m_fov;
    state.hither = m_hither;
    state.yon    = m_yon;

    state.world_to_camera = OSL::Matrix44(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                          0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                          0.0, 1.0);
    //perspective is not  a member of StringParams (i.e not in strdecls.h)
    state.projection  = u_perspective;
    state.pixelaspect = m_pixelaspect;
    std::copy_n(m_screen_window, 4, state.screen_window);
    std::copy_n(m_shutter, 2, state.shutter);
}

void
SimpleRenderer::errorfmt(OSL::ShaderGlobals* sg,
                         OSL::ustringhash fmt_specification, int32_t arg_count,
                         const EncodedType* arg_types, uint32_t arg_values_size,
                         uint8_t* argValues)
{
    RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);
    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_errorfmt(OSL::get_thread_index(sg), OSL::get_shade_index(sg),
                       fmt_specification, arg_count, arg_types, arg_values_size,
                       argValues);
}

void
SimpleRenderer::warningfmt(OSL::ShaderGlobals* sg,
                           OSL::ustringhash fmt_specification,
                           int32_t arg_count, const EncodedType* arg_types,
                           uint32_t arg_values_size, uint8_t* argValues)
{
    RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);
    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_warningfmt(OSL::get_max_warnings_per_thread(sg),
                         OSL::get_thread_index(sg), OSL::get_shade_index(sg),
                         fmt_specification, arg_count, arg_types,
                         arg_values_size, argValues);
}



void
SimpleRenderer::printfmt(OSL::ShaderGlobals* sg,
                         OSL::ustringhash fmt_specification, int32_t arg_count,
                         const EncodedType* arg_types, uint32_t arg_values_size,
                         uint8_t* argValues)
{
    RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);
    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_printfmt(OSL::get_thread_index(sg), OSL::get_shade_index(sg),
                       fmt_specification, arg_count, arg_types, arg_values_size,
                       argValues);
}

void
SimpleRenderer::filefmt(OSL::ShaderGlobals* sg, OSL::ustringhash filename_hash,
                        OSL::ustringhash fmt_specification, int32_t arg_count,
                        const EncodedType* arg_types, uint32_t arg_values_size,
                        uint8_t* argValues)
{
    RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);
    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_filefmt(OSL::get_thread_index(sg), OSL::get_shade_index(sg),
                      filename_hash, fmt_specification, arg_count, arg_types,
                      arg_values_size, argValues);
}


OSL_NAMESPACE_EXIT
