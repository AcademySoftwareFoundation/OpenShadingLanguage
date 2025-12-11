// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#ifndef __CUDACC__
#    include <OpenImageIO/filesystem.h>
#    include <OpenImageIO/parallel.h>
#    include <OpenImageIO/timer.h>
#    include <pugixml.hpp>

#    ifdef USING_OIIO_PUGI
namespace pugi = OIIO::pugi;
#    endif

#    include <OSL/hashes.h>
#    include "raytracer.h"
#    include "shading.h"
#    include "simpleraytracer.h"
#endif

#include <cmath>

// Create ustrings for all strings used by the free function renderer services.
// Required to allow the reverse mapping of hash->string to work when processing messages
namespace RS {
namespace Strings {
#define RS_STRDECL(str, var_name) const OSL::ustring var_name { str };
#include "rs_strdecls.h"
#undef RS_STRDECL
}  // namespace Strings
}  // namespace RS

namespace RS {
namespace {
namespace Hashes {
#define RS_STRDECL(str, var_name) \
    constexpr OSL::ustringhash var_name(OSL::strhash(str));
#include "rs_strdecls.h"
#undef RS_STRDECL
};  //namespace Hashes
}  // unnamed namespace
};  //namespace RS

using namespace OSL;

OSL_NAMESPACE_BEGIN


#ifndef __CUDACC__
using ShaderGlobalsType = OSL::ShaderGlobals;
#else
using ShaderGlobalsType = OSL_CUDA::ShaderGlobals;
#endif


#ifndef __CUDACC__
static TypeDesc TypeFloatArray2(TypeDesc::FLOAT, 2);
static TypeDesc TypeFloatArray4(TypeDesc::FLOAT, 4);
static TypeDesc TypeIntArray2(TypeDesc::INT, 2);



// Subclass ErrorHandler
class SimpleRaytracer::ErrorHandler final : public OIIO::ErrorHandler {
public:
    ErrorHandler(SimpleRaytracer& rend) : m_rend(rend) {}

    virtual void operator()(int errcode, const std::string& msg)
    {
        OIIO::ErrorHandler::operator()(errcode, msg);
        if (errcode & OIIO::ErrorHandler::EH_ERROR
            || errcode & OIIO::ErrorHandler::EH_SEVERE)
            m_rend.m_had_error = true;
        if (errcode & OIIO::ErrorHandler::EH_SEVERE)
            exit(EXIT_FAILURE);
    }

private:
    SimpleRaytracer& m_rend;
};



SimpleRaytracer::SimpleRaytracer()
{
    m_errhandler.reset(new SimpleRaytracer::ErrorHandler(*this));

    Matrix44 M;
    M.makeIdentity();
    camera_params(M, RS::Hashes::perspective, 90.0f, 0.1f, 1000.0f, 256, 256);

    // Set up getters
    m_attr_getters[RS::Hashes::osl_version] = &SimpleRaytracer::get_osl_version;
    m_attr_getters[RS::Hashes::camera_resolution]
        = &SimpleRaytracer::get_camera_resolution;
    m_attr_getters[RS::Hashes::camera_projection]
        = &SimpleRaytracer::get_camera_projection;
    m_attr_getters[RS::Hashes::camera_pixelaspect]
        = &SimpleRaytracer::get_camera_pixelaspect;
    m_attr_getters[RS::Hashes::camera_screen_window]
        = &SimpleRaytracer::get_camera_screen_window;
    m_attr_getters[RS::Hashes::camera_fov]  = &SimpleRaytracer::get_camera_fov;
    m_attr_getters[RS::Hashes::camera_clip] = &SimpleRaytracer::get_camera_clip;
    m_attr_getters[RS::Hashes::camera_clip_near]
        = &SimpleRaytracer::get_camera_clip_near;
    m_attr_getters[RS::Hashes::camera_clip_far]
        = &SimpleRaytracer::get_camera_clip_far;
    m_attr_getters[RS::Hashes::camera_shutter]
        = &SimpleRaytracer::get_camera_shutter;
    m_attr_getters[RS::Hashes::camera_shutter_open]
        = &SimpleRaytracer::get_camera_shutter_open;
    m_attr_getters[RS::Hashes::camera_shutter_close]
        = &SimpleRaytracer::get_camera_shutter_close;
}



OIIO::ParamValue*
SimpleRaytracer::find_attribute(string_view name, TypeDesc searchtype,
                                bool casesensitive)
{
    auto iter = options.find(name, searchtype, casesensitive);
    if (iter != options.end())
        return &(*iter);
    return nullptr;
}



const OIIO::ParamValue*
SimpleRaytracer::find_attribute(string_view name, TypeDesc searchtype,
                                bool casesensitive) const
{
    auto iter = options.find(name, searchtype, casesensitive);
    if (iter != options.end())
        return &(*iter);
    return nullptr;
}



void
SimpleRaytracer::attribute(string_view name, TypeDesc type, const void* value)
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
SimpleRaytracer::camera_params(const Matrix44& world_to_camera,
                               ustringhash projection, float hfov, float hither,
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
    camera.xres        = xres;
    camera.yres        = yres;
}



inline Vec3
strtovec(string_view str)
{
    Vec3 v(0, 0, 0);
    OIIO::Strutil::parse_float(str, v[0]);
    OIIO::Strutil::parse_char(str, ',');
    OIIO::Strutil::parse_float(str, v[1]);
    OIIO::Strutil::parse_char(str, ',');
    OIIO::Strutil::parse_float(str, v[2]);
    return v;
}

inline bool
strtobool(const char* str)
{
    return strcmp(str, "1") == 0 || strcmp(str, "on") == 0
           || strcmp(str, "yes") == 0;
}

template<int N> struct ParamStorage {
    ParamStorage() : fparamindex(0), iparamindex(0), sparamindex(0) {}

    void* Int(int i)
    {
        OSL_DASSERT(iparamindex < N);
        iparamdata[iparamindex] = i;
        iparamindex++;
        return &iparamdata[iparamindex - 1];
    }

    void* Float(float f)
    {
        OSL_DASSERT(fparamindex < N);
        fparamdata[fparamindex] = f;
        fparamindex++;
        return &fparamdata[fparamindex - 1];
    }

    void* Vec(float x, float y, float z)
    {
        Float(x);
        Float(y);
        Float(z);
        return &fparamdata[fparamindex - 3];
    }

    void* Vec(const float* xyz)
    {
        Float(xyz[0]);
        Float(xyz[1]);
        Float(xyz[2]);
        return &fparamdata[fparamindex - 3];
    }

    void* Str(const char* str)
    {
        OSL_DASSERT(sparamindex < N);
        sparamdata[sparamindex] = ustring(str);
        sparamindex++;
        return &sparamdata[sparamindex - 1];
    }

private:
    // storage for shader parameters
    float fparamdata[N];
    int iparamdata[N];
    ustring sparamdata[N];

    int fparamindex;
    int iparamindex;
    int sparamindex;
};



inline bool
parse_prefix_and_floats(string_view str, string_view prefix, int nvals,
                        float* vals)
{
    bool ok = OIIO::Strutil::parse_prefix(str, prefix);
    for (int i = 0; i < nvals && ok; ++i)
        ok &= OIIO::Strutil::parse_float(str, vals[i]);
    return ok;
}


inline bool
parse_prefix_and_ints(string_view str, string_view prefix, int nvals, int* vals)
{
    bool ok = OIIO::Strutil::parse_prefix(str, prefix);
    for (int i = 0; i < nvals && ok; ++i)
        ok &= OIIO::Strutil::parse_int(str, vals[i]);
    return ok;
}



void
SimpleRaytracer::parse_scene_xml(const std::string& scenefile)
{
    ShaderMap shadermap;
    pugi::xml_document doc;
    pugi::xml_parse_result parse_result;
    if (OIIO::Strutil::ends_with(scenefile, ".xml")
        && OIIO::Filesystem::exists(scenefile)) {
        parse_result = doc.load_file(scenefile.c_str());
    } else {
        parse_result = doc.load_buffer(scenefile.c_str(), scenefile.size());
    }
    if (!parse_result)
        errhandler().severefmt("XML parsed with errors: {} at offset {}",
                               parse_result.description(), parse_result.offset);
    pugi::xml_node root = doc.child("World");
    if (!root)
        errhandler().severefmt(
            "Error reading scene: Root element <World> is missing");

    // loop over all children of world
    for (auto node = root.first_child(); node; node = node.next_sibling()) {
        if (strcmp(node.name(), "Option") == 0) {
            for (auto attr = node.first_attribute(); attr;
                 attr      = attr.next_attribute()) {
                int i = 0;
                if (parse_prefix_and_ints(attr.value(), "int ", 1, &i))
                    attribute(attr.name(), i);
                // TODO: pass any extra options to shading system (or texture system?)
            }
        } else if (strcmp(node.name(), "Camera") == 0) {
            // defaults
            Vec3 eye(0, 0, 0);
            Vec3 dir(0, 0, -1);
            Vec3 up(0, 1, 0);
            float fov = 90.f;

            // load camera (only first attribute counts if duplicates)
            pugi::xml_attribute eye_attr = node.attribute("eye");
            pugi::xml_attribute dir_attr = node.attribute("dir");
            pugi::xml_attribute at_attr  = node.attribute("look_at");
            pugi::xml_attribute up_attr  = node.attribute("up");
            pugi::xml_attribute fov_attr = node.attribute("fov");

            if (eye_attr)
                eye = strtovec(eye_attr.value());
            if (dir_attr)
                dir = strtovec(dir_attr.value());
            else if (at_attr)
                dir = strtovec(at_attr.value()) - eye;
            if (up_attr)
                up = strtovec(up_attr.value());
            if (fov_attr)
                fov = OIIO::Strutil::from_string<float>(fov_attr.value());

            camera.lookat(eye, dir, up, fov);
        } else if (strcmp(node.name(), "Sphere") == 0) {
            // load sphere
            pugi::xml_attribute center_attr = node.attribute("center");
            pugi::xml_attribute radius_attr = node.attribute("radius");

            if (center_attr && radius_attr) {
                Vec3 center  = strtovec(center_attr.value());
                float radius = OIIO::Strutil::from_string<float>(
                    radius_attr.value());
                if (radius > 0) {
                    pugi::xml_attribute resolution_attr = node.attribute(
                        "resolution");
                    int resolution = 64;
                    if (resolution_attr) {
                        OIIO::string_view str = resolution_attr.value();
                        OIIO::Strutil::parse_int(str, resolution);
                    }
                    scene.add_sphere(center, radius, int(shaders().size()) - 1,
                                     resolution);
                }
            }
        } else if (strcmp(node.name(), "Quad") == 0) {
            // load quad
            pugi::xml_attribute corner_attr = node.attribute("corner");
            pugi::xml_attribute edge_x_attr = node.attribute("edge_x");
            pugi::xml_attribute edge_y_attr = node.attribute("edge_y");
            if (corner_attr && edge_x_attr && edge_y_attr) {
                Vec3 co = strtovec(corner_attr.value());
                Vec3 ex = strtovec(edge_x_attr.value());
                Vec3 ey = strtovec(edge_y_attr.value());

                int resolution                      = 1;
                pugi::xml_attribute resolution_attr = node.attribute(
                    "resolution");
                if (resolution_attr) {
                    OIIO::string_view str = resolution_attr.value();
                    OIIO::Strutil::parse_int(str, resolution);
                }

                scene.add_quad(co, ex, ey, int(shaders().size()) - 1,
                               resolution);
            }
        } else if (strcmp(node.name(), "Model") == 0) {
            // load .obj model
            pugi::xml_attribute filename_attr = node.attribute("filename");
            if (filename_attr) {
                std::string filename = filename_attr.value();
                std::vector<std::string> searchpath;
                searchpath.emplace_back(
                    OIIO::Filesystem::parent_path(scenefile));
                std::string actual_filename
                    = OIIO::Filesystem::searchpath_find(filename, searchpath,
                                                        false);
                if (actual_filename.empty()) {
                    errhandler().errorfmt("Unable to find model file {}",
                                          filename);
                } else {
                    // we got a valid filename, try to load the model
                    scene.add_model(actual_filename, shadermap,
                                    int(shaders().size() - 1), errhandler());
                }
            }
        } else if (strcmp(node.name(), "Background") == 0) {
            pugi::xml_attribute res_attr = node.attribute("resolution");
            if (res_attr)
                backgroundResolution = OIIO::Strutil::from_string<int>(
                    res_attr.value());
            backgroundShaderID = int(shaders().size()) - 1;
        } else if (strcmp(node.name(), "ShaderGroup") == 0) {
            ShaderGroupRef group;
            pugi::xml_attribute is_light_attr = node.attribute("is_light");
            pugi::xml_attribute name_attr     = node.attribute("name");
            std::string name = name_attr ? name_attr.value() : "group";
            pugi::xml_attribute type_attr = node.attribute("type");
            std::string shadertype = type_attr ? type_attr.value() : "surface";
            pugi::xml_attribute commands_attr = node.attribute("commands");
            std::string commands = commands_attr ? commands_attr.value()
                                                 : node.text().get();
            if (commands.size())
                group = shadingsys->ShaderGroupBegin(name, shadertype,
                                                     commands);
            else
                group = shadingsys->ShaderGroupBegin(name);
            ParamStorage<1024>
                store;  // scratch space to hold parameters until they are read by Shader()
            for (pugi::xml_node gnode = node.first_child(); gnode;
                 gnode                = gnode.next_sibling()) {
                if (strcmp(gnode.name(), "Parameter") == 0) {
                    // handle parameters
                    for (auto attr = gnode.first_attribute(); attr;
                         attr      = attr.next_attribute()) {
                        int i = 0;
                        float f[3];
                        string_view val(attr.value());
                        if (parse_prefix_and_ints(val, "int ", 1, &i))
                            shadingsys->Parameter(*group, attr.name(), TypeInt,
                                                  store.Int(i));
                        else if (parse_prefix_and_floats(val, "float ", 1, f))
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypeFloat, store.Float(f[0]));
                        else if (parse_prefix_and_floats(val, "vector ", 3, f))
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypeVector, store.Vec(f));
                        else if (parse_prefix_and_floats(val, "point ", 3, f))
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypePoint, store.Vec(f));
                        else if (parse_prefix_and_floats(val, "color ", 3, f))
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypeColor, store.Vec(f));
                        else
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypeString,
                                                  store.Str(attr.value()));
                    }
                } else if (strcmp(gnode.name(), "Shader") == 0) {
                    pugi::xml_attribute type_attr  = gnode.attribute("type");
                    pugi::xml_attribute name_attr  = gnode.attribute("name");
                    pugi::xml_attribute layer_attr = gnode.attribute("layer");
                    const char* type = type_attr ? type_attr.value()
                                                 : "surface";
                    if (name_attr && layer_attr)
                        shadingsys->Shader(*group, type, name_attr.value(),
                                           layer_attr.value());
                } else if (strcmp(gnode.name(), "ConnectShaders") == 0) {
                    // FIXME: find a more elegant way to encode this
                    pugi::xml_attribute sl = gnode.attribute("srclayer");
                    pugi::xml_attribute sp = gnode.attribute("srcparam");
                    pugi::xml_attribute dl = gnode.attribute("dstlayer");
                    pugi::xml_attribute dp = gnode.attribute("dstparam");
                    if (sl && sp && dl && dp)
                        shadingsys->ConnectShaders(*group, sl.value(),
                                                   sp.value(), dl.value(),
                                                   dp.value());
                } else {
                    // unknown element?
                }
            }
            shadingsys->ShaderGroupEnd(*group);
            if (name_attr) {
                if (auto it = shadermap.find(name); it != shadermap.end()) {
                    int shaderID = it->second;
                    if (shaderID >= 0 && shaderID < int(shaders().size())) {
                        fprintf(stderr, "Updating shader %d - %s\n", shaderID,
                                shadertype.c_str());
                        // we already have a material under this name,
                        Material& m = shaders()[shaderID];
                        // TODO: could we query the shadertype directly from the ShaderGroup?
                        if (shadertype == "displacement")
                            m.disp = group;
                        else if (shadertype == "surface")
                            m.surf = group;
                        // skip the rest which would add a new material
                        continue;
                    }
                } else {
                    shadermap.emplace(name, int(shaders().size()));
                }
            }
            shaders().emplace_back(Material { group, nullptr });
            m_shader_is_light.emplace_back(
                is_light_attr ? strtobool(is_light_attr.value()) : false);
        } else {
            // unknown element?
        }
    }
    if (root.next_sibling())
        errhandler().severefmt(
            "Error reading {}: Found multiple top-level elements", scenefile);
    if (shaders().empty())
        errhandler().severefmt("No shaders in scene");
    if (scene.num_prims() == 0)
        errhandler().severefmt("No primitives in scene");
    camera.finalize();
}



bool
SimpleRaytracer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                            TransformationPtr xform, float /*time*/)
{
    // SimpleRaytracer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    if (xform)
        result = *reinterpret_cast<const Matrix44*>(xform);
    else
        result = Matrix44(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

    return true;
}



bool
SimpleRaytracer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
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
SimpleRaytracer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                            TransformationPtr xform)
{
    // SimpleRaytracer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    if (xform)
        result = *reinterpret_cast<const Matrix44*>(xform);
    else
        result = Matrix44(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    return true;
}



bool
SimpleRaytracer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                            ustringhash from)
{
    // SimpleRaytracer doesn't understand motion blur, so we never fail
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
SimpleRaytracer::get_inverse_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                                    ustringhash to, float /*time*/)
{
    if (to == OSL::Hashes::camera || to == OSL::Hashes::screen
        || to == OSL::Hashes::NDC || to == RS::Hashes::raster) {
        // clang-format off
        Matrix44 M = m_world_to_camera;
        if (to == OSL::Hashes::screen || to == OSL::Hashes::NDC || to == RS::Hashes::raster) {
            float depthrange = (double)m_yon-(double)m_hither;
            if (m_projection == RS::Hashes::perspective) {
                float tanhalffov = tanf (0.5f * m_fov * M_PI/180.0);
                Matrix44 camera_to_screen (1/tanhalffov, 0, 0, 0,
                                           0, 1/tanhalffov, 0, 0,
                                           0, 0, m_yon/depthrange, 1,
                                           0, 0, -m_yon*m_hither/depthrange, 0);
                M = M * camera_to_screen;
            } else {
                Matrix44 camera_to_screen (1, 0, 0, 0,
                                           0, 1, 0, 0,
                                           0, 0, 1/depthrange, 0,
                                           0, 0, -m_hither/depthrange, 1);
                M = M * camera_to_screen;
            }
            if (to == OSL::Hashes::NDC || to == RS::Hashes::raster) {
                float screenleft = -1.0, screenwidth = 2.0;
                float screenbottom = -1.0, screenheight = 2.0;
                Matrix44 screen_to_ndc (1/screenwidth, 0, 0, 0,
                                        0, 1/screenheight, 0, 0,
                                        0, 0, 1, 0,
                                        -screenleft/screenwidth, -screenbottom/screenheight, 0, 1);
                M = M * screen_to_ndc;
                if (to == RS::Hashes::raster) {
                    Matrix44 ndc_to_raster (camera.xres, 0, 0, 0,
                                            0, camera.yres, 0, 0,
                                            0, 0, 1, 0,
                                            0, 0, 0, 1);
                    M = M * ndc_to_raster;
                }
            }
        }
        // clang-format on
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
SimpleRaytracer::name_transform(const char* name, const OSL::Matrix44& xform)
{
    std::shared_ptr<Transformation> M(new OSL::Matrix44(xform));
    m_named_xforms[ustringhash(name)] = M;
}



bool
SimpleRaytracer::get_array_attribute(ShaderGlobals* sg, bool derivatives,
                                     ustringhash object, TypeDesc type,
                                     ustringhash name, int index, void* val)
{
    AttrGetterMap::const_iterator g = m_attr_getters.find(name);
    if (g != m_attr_getters.end()) {
        AttrGetter getter = g->second;
        return (this->*(getter))(sg, derivatives, object, type, name, val);
    }

    // If no named attribute was found, allow userdata to bind to the
    // attribute request.
    if (object.empty() && index == -1)
        return get_userdata(derivatives, name, type, sg, val);

    return false;
}



bool
SimpleRaytracer::get_attribute(ShaderGlobals* sg, bool derivatives,
                               ustringhash object, TypeDesc type,
                               ustringhash name, void* val)
{
    return get_array_attribute(sg, derivatives, object, type, name, -1, val);
}



bool
SimpleRaytracer::get_userdata(bool derivatives, ustringhash name, TypeDesc type,
                              ShaderGlobals* sg, void* val)
{
    // Just to illustrate how this works, respect s and t userdata, filled
    // in with the uv coordinates.  In a real renderer, it would probably
    // look up something specific to the primitive, rather than have hard-
    // coded names.

    if (name == RS::Hashes::s && type == TypeFloat) {
        ((float*)val)[0] = sg->u;
        if (derivatives) {
            ((float*)val)[1] = sg->dudx;
            ((float*)val)[2] = sg->dudy;
        }
        return true;
    }
    if (name == RS::Hashes::t && type == TypeFloat) {
        ((float*)val)[0] = sg->v;
        if (derivatives) {
            ((float*)val)[1] = sg->dvdx;
            ((float*)val)[2] = sg->dvdy;
        }
        return true;
    }

    return false;
}


bool
SimpleRaytracer::get_osl_version(ShaderGlobals* /*sg*/, bool /*derivs*/,
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
SimpleRaytracer::get_camera_resolution(ShaderGlobals* /*sg*/, bool /*derivs*/,
                                       ustringhash /*object*/, TypeDesc type,
                                       ustringhash /*name*/, void* val)
{
    if (type == TypeIntArray2) {
        ((int*)val)[0] = camera.xres;
        ((int*)val)[1] = camera.yres;
        return true;
    }
    return false;
}


bool
SimpleRaytracer::get_camera_projection(ShaderGlobals* /*sg*/, bool /*derivs*/,
                                       ustringhash /*object*/, TypeDesc type,
                                       ustringhash /*name*/, void* val)
{
    if (type == TypeString) {
        ((ustringhash*)val)[0] = m_projection;
        return true;
    }
    return false;
}


bool
SimpleRaytracer::get_camera_fov(ShaderGlobals* /*sg*/, bool derivs,
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
SimpleRaytracer::get_camera_pixelaspect(ShaderGlobals* /*sg*/, bool derivs,
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
SimpleRaytracer::get_camera_clip(ShaderGlobals* /*sg*/, bool derivs,
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
SimpleRaytracer::get_camera_clip_near(ShaderGlobals* /*sg*/, bool derivs,
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
SimpleRaytracer::get_camera_clip_far(ShaderGlobals* /*sg*/, bool derivs,
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
SimpleRaytracer::get_camera_shutter(ShaderGlobals* /*sg*/, bool derivs,
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
SimpleRaytracer::get_camera_shutter_open(ShaderGlobals* /*sg*/, bool derivs,
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
SimpleRaytracer::get_camera_shutter_close(ShaderGlobals* /*sg*/, bool derivs,
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
SimpleRaytracer::get_camera_screen_window(ShaderGlobals* /*sg*/, bool derivs,
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
#endif  // #ifndef __CUDACC__



void OSL_HOSTDEVICE
SimpleRaytracer::globals_from_hit(ShaderGlobalsType& sg, const Ray& r,
                                  const Dual2<float>& t, int id, float u,
                                  float v)
{
#ifndef __CUDACC__
    memset((char*)&sg, 0, sizeof(ShaderGlobals));
#endif

#ifndef __CUDACC__
    const int meshid = std::upper_bound(scene.last_index.begin(),
                                        scene.last_index.end(), id)
                       - scene.last_index.begin();
#else
    const int meshid = m_meshids[id];
#endif

    Dual2<Vec3> P = r.point(t);
    // We are missing the projection onto the surface here
    sg.P                  = P.val();
    sg.dPdx               = P.dx();
    sg.dPdy               = P.dy();
    sg.N                  = scene.normal(P, sg.Ng, id, u, v);
    Dual2<Vec2> uv        = scene.uv(P, sg.N, sg.dPdu, sg.dPdv, id, u, v);
    sg.u                  = uv.val().x;
    sg.dudx               = uv.dx().x;
    sg.dudy               = uv.dy().x;
    sg.v                  = uv.val().y;
    sg.dvdx               = uv.dx().y;
    sg.dvdy               = uv.dy().y;
    sg.surfacearea        = m_mesh_surfacearea[meshid];
    Dual2<Vec3> direction = r.dual_direction();
    sg.I                  = direction.val();
    sg.dIdx               = direction.dx();
    sg.dIdy               = direction.dy();
    sg.backfacing         = sg.Ng.dot(sg.I) > 0;
    if (sg.backfacing) {
        sg.N  = -sg.N;
        sg.Ng = -sg.Ng;
    }
    sg.raytype        = r.raytype;
    sg.flipHandedness = sg.dPdx.cross(sg.dPdy).dot(sg.N) < 0;
}



OSL_HOSTDEVICE Vec3
SimpleRaytracer::eval_background(const Dual2<Vec3>& dir, ShadingContext* ctx,
                                 int bounce)
{
    ShaderGlobalsType sg;
    memset((char*)&sg, 0, sizeof(ShaderGlobals));
    sg.I    = dir.val();
    sg.dIdx = dir.dx();
    sg.dIdy = dir.dy();
    if (bounce >= 0)
        sg.raytype = bounce > 0 ? Ray::DIFFUSE : Ray::CAMERA;
#ifndef __CUDACC__
    shadingsys->execute(*ctx, *m_shaders[backgroundShaderID].surf, sg);
#else
    StackClosurePool closure_pool;
    execute_shader(sg, render_params.bg_id, closure_pool);
#endif
    return process_background_closure((const ClosureColor*)sg.Ci);
}

Color3
SimpleRaytracer::subpixel_radiance(float x, float y, Sampler& sampler,
                                   ShadingContext* ctx)
{
#ifdef __CUDACC__
    // Scratch space for the output closures
    StackClosurePool closure_pool;
    StackClosurePool light_closure_pool;
#endif

    constexpr float inf = std::numeric_limits<float>::infinity();
    Ray r               = camera.get(x, y);
    Color3 path_weight(1, 1, 1);
    Color3 path_radiance(0, 0, 0);
    int prev_id    = -1;
    float bsdf_pdf = inf;  // camera ray has only one possible direction
    MediumStack medium_stack;


    for (int b = 0; b <= max_bounces; b++) {
        ShaderGlobalsType sg;

        // trace the ray against the scene
        Intersection hit = scene.intersect(r, inf, prev_id);
        if (hit.t == inf) {
            // we hit nothing? check background shader
            if (backgroundShaderID >= 0) {
                if (b > 0 && backgroundResolution > 0) {
                    float bg_pdf = 0;
                    Vec3 bg      = background.eval(r.direction, bg_pdf);
                    path_radiance
                        += path_weight * bg
                           * MIS::power_heuristic<MIS::WEIGHT_WEIGHT>(bsdf_pdf,
                                                                      bg_pdf);
                } else {
                    // we aren't importance sampling the background - so just run it directly
                    path_radiance += path_weight
                                     * eval_background(r.direction, ctx, b);
                }
            }
            break;
        }
        
        if (medium_stack.integrate(r, sampler, hit, path_weight, path_radiance, bsdf_pdf)) {
            continue;
        }

        // construct a shader globals for the hit point
        globals_from_hit(sg, r, hit.t, hit.id, hit.u, hit.v);

        if (show_globals) {
            // visualize the main fields of the shader globals
            Vec3 v = sg.Ng;
            if (show_globals == 2)
                v = sg.N;
            if (show_globals == 3)
                v = sg.dPdu.normalize();
            if (show_globals == 4)
                v = sg.dPdv.normalize();
            if (show_globals == 5)
                v = Vec3(sg.u, sg.v, 0);
            Color3 c(v.x, v.y, v.z);
            if (show_globals != 5)
                c = c * 0.5f + Color3(0.5f);
            path_radiance += path_weight * c;
            break;
        }

        const float radius = r.radius + r.spread * hit.t;

        int shaderID = scene.shaderid(hit.id);

#ifndef __CUDACC__
        if (shaderID < 0 || !m_shaders[shaderID].surf)
            break;  // no shader attached? done

        // execute shader and process the resulting list of closures
        shadingsys->execute(*ctx, *m_shaders[shaderID].surf, sg);
#else
        if (shaderID < 0)
            break;  // no shader attached? done
        execute_shader(sg, shaderID, closure_pool);
#endif
        ShadingResult result;
        bool last_bounce = b == max_bounces;
        process_closure(sg, r.roughness, result, medium_stack,
                        (const ClosureColor*)sg.Ci, last_bounce);

#ifndef __CUDACC__
        const size_t lightprims_size = m_lightprims.size();
#endif

        // add self-emission
        float k = 1;
        if (m_shader_is_light[shaderID] && lightprims_size > 0) {
            const float light_pick_pdf = 1.0f / lightprims_size;
            // figure out the probability of reaching this point
            float light_pdf = light_pick_pdf
                              * scene.shapepdf(hit.id, r.origin, sg.P);
            k = MIS::power_heuristic<MIS::WEIGHT_EVAL>(bsdf_pdf, light_pdf);
        }
        path_radiance += path_weight * k * result.Le;

        // last bounce? nothing left to do
        if (last_bounce)
            break;

        // build internal pdf for sampling between bsdf closures
        result.bsdf.prepare(-sg.I, path_weight, b >= rr_depth);

        if (show_albedo_scale > 0) {
            // Instead of path tracing, just visualize the albedo
            // of the bsdf. This can be used to validate the accuracy of
            // the get_albedo method for a particular bsdf.
            path_radiance += path_weight * result.bsdf.get_albedo(-sg.I)
                             * show_albedo_scale;
            break;
        }

        // get three random numbers
        Vec3 s   = sampler.get();
        float xi = s.x;
        float yi = s.y;
        float zi = s.z;

        // trace one ray to the background
        if (backgroundResolution > 0) {
            Dual2<Vec3> bg_dir;
            float bg_pdf   = 0;
            Vec3 bg        = background.sample(xi, yi, bg_dir, bg_pdf);
            BSDF::Sample b = result.bsdf.eval(-sg.I, bg_dir.val());
            Color3 contrib = path_weight * b.weight * bg
                             * MIS::power_heuristic<MIS::WEIGHT_WEIGHT>(bg_pdf,
                                                                        b.pdf);
            if ((contrib.x + contrib.y + contrib.z) > 0) {
                ShaderGlobalsType shadow_sg;
                Ray shadow_ray          = Ray(sg.P, bg_dir.val(), radius, 0, 0,
                                              Ray::SHADOW);
                Intersection shadow_hit = scene.intersect(shadow_ray, inf,
                                                          hit.id);
                if (shadow_hit.t == inf)  // ray reached the background?
                    path_radiance += contrib;
            }
        }

        // trace a shadow ray to one of the light emitting primitives
        if (lightprims_size > 0) {
            const float light_pick_pdf = 1.0f / lightprims_size;

            // uniform probability for each light
            float xl = xi * lightprims_size;
            int ls   = floorf(xl);
            xl -= ls;

            uint32_t lid = m_lightprims[ls];
            if (lid != hit.id) {
                int shaderID = scene.shaderid(lid);

                // sample a random direction towards the object
                LightSample sample = scene.sample(lid, sg.P, xl, yi);
                BSDF::Sample b     = result.bsdf.eval(-sg.I, sample.dir);
                Color3 contrib     = path_weight * b.weight
                                 * MIS::power_heuristic<MIS::EVAL_WEIGHT>(
                                     light_pick_pdf * sample.pdf, b.pdf);
                if ((contrib.x + contrib.y + contrib.z) > 0) {
                    ShaderGlobalsType light_sg;
                    Ray shadow_ray = Ray(sg.P, sample.dir, radius, 0, 0,
                                         Ray::SHADOW);
                    // trace a shadow ray and see if we actually hit the target
                    // in this tiny renderer, tracing a ray is probably cheaper than evaluating the light shader
                    Intersection shadow_hit
                        = scene.intersect(shadow_ray, sample.dist, hit.id, lid);

#ifndef __CUDACC__
                    const bool did_hit = shadow_hit.t == sample.dist;
#else
                    // The hit distance on the device is not as precise as on
                    // the CPU, so we need to allow a little wiggle room. An
                    // epsilon of 1e-3f empirically gives results that closely
                    // match the CPU for the test scenes, so that's what we're
                    // using.
                    const bool did_hit = fabsf(shadow_hit.t - sample.dist)
                                         < 1e-3f;
#endif
                    if (did_hit) {
                        // setup a shader global for the point on the light
                        globals_from_hit(light_sg, shadow_ray, sample.dist, lid,
                                         sample.u, sample.v);
#ifndef __CUDACC__
                        // execute the light shader (for emissive closures only)
                        shadingsys->execute(*ctx, *m_shaders[shaderID].surf,
                                            light_sg);
#else
                        execute_shader(light_sg, shaderID, light_closure_pool);
#endif
                        ShadingResult light_result;
                        process_closure(light_sg, r.roughness, light_result,
                                        medium_stack, (const ClosureColor*)light_sg.Ci, 
                                        true);
                        // accumulate contribution
                        path_radiance += contrib * light_result.Le;
                    }
                }
            }
        }

        // trace indirect ray and continue
        BSDF::Sample p = result.bsdf.sample(-sg.I, xi, yi, zi);
        path_weight *= p.weight;
        bsdf_pdf  = p.pdf;
        r.raytype = Ray::DIFFUSE;  // FIXME? Use DIFFUSE for all indiirect rays
        r.direction = p.wi;
        r.radius    = radius;
        // Just simply use roughness as spread slope
        r.spread    = std::max(r.spread, p.roughness);
        r.roughness = p.roughness;
        
        if (sg.backfacing) {  // if exiting
            medium_stack.pop_medium();
        }
        
        if (!(path_weight.x > 0) && !(path_weight.y > 0)
            && !(path_weight.z > 0) && b > 10)
            break;  // filter out all 0's or NaNs
        prev_id  = hit.id;
        r.origin = sg.P;
    }
    return path_radiance;
}



OSL_HOSTDEVICE Color3
SimpleRaytracer::antialias_pixel(int x, int y, ShadingContext* ctx)
{
    Color3 result(0, 0, 0);
    for (int si = 0, n = aa * aa; si < n; si++) {
        Sampler sampler(x, y, si);
        // jitter pixel coordinate [0,1)^2
        Vec3 j = no_jitter ? Vec3(0.5f, 0.5f, 0) : sampler.get();
        // warp distribution to approximate a tent filter [-1,+1)^2
        j.x *= 2;
        j.x = j.x < 1 ? sqrtf(j.x) - 1 : 1 - sqrtf(2 - j.x);
        j.y *= 2;
        j.y = j.y < 1 ? sqrtf(j.y) - 1 : 1 - sqrtf(2 - j.y);
        // trace eye ray (apply jitter from center of the pixel)
        Color3 r = subpixel_radiance(x + 0.5f + j.x, y + 0.5f + j.y, sampler,
                                     ctx);
        // mix in result via lerp for numerical stability
        result = OIIO::lerp(result, r, 1.0f / (si + 1));
    }
    return result;
}


#ifndef __CUDACC__
void
SimpleRaytracer::prepare_render()
{
    // Retrieve and validate options
    aa                = std::max(1, options.get_int("aa"));
    no_jitter         = options.get_int("no_jitter") != 0;
    max_bounces       = options.get_int("max_bounces");
    rr_depth          = options.get_int("rr_depth");
    show_albedo_scale = options.get_float("show_albedo_scale");
    show_globals      = options.get_int("show_globals");

    // prepare background importance table (if requested)
    if (backgroundResolution > 0 && backgroundShaderID >= 0) {
        // get a context so we can make several background shader calls
        OSL::PerThreadInfo* thread_info = shadingsys->create_thread_info();
        ShadingContext* ctx             = shadingsys->get_context(thread_info);

        // build importance table to optimize background sampling
        auto evaler = [this](const Dual2<Vec3>& dir, ShadingContext* ctx) {
            return this->eval_background(dir, ctx);
        };
        background.prepare(backgroundResolution, evaler, ctx);

        // release context
        shadingsys->release_context(ctx);
        shadingsys->destroy_thread_info(thread_info);
    } else {
        // we aren't directly evaluating the background
        backgroundResolution = 0;
    }

    prepare_geometry();

    // build bvh and prepare triangles
    scene.prepare(errhandler());
    prepare_lights();

#    if 0
    // dump scene to disk as obj for debugging purposes
    // TODO: make this a feature?
    FILE* fp = fopen("/tmp/test.obj", "w");
    for (Vec3 v : scene.verts)
        fprintf(fp, "v %.9g %.9g %.9g\n", v.x, v.y, v.z);
    for (TriangleIndices t : scene.triangles)
        fprintf(fp, "f %d %d %d\n", 1 + t.a, 1 + t.b, 1 + t.c);
    fclose(fp);
#    endif
}



void
SimpleRaytracer::prepare_lights()
{
    m_mesh_surfacearea.reserve(scene.last_index.size());

    // measure the total surface area of each mesh
    int first_index = 0;
    for (int last_index : scene.last_index) {
        float area = 0;
        for (int index = first_index; index < last_index; index++) {
            area += scene.primitivearea(index);
        }
        m_mesh_surfacearea.emplace_back(area);
        first_index = last_index;
    }
    // collect all light emitting triangles
    for (unsigned t = 0, n = scene.num_prims(); t < n; t++) {
        int shaderID = scene.shaderid(t);
        if (shaderID < 0 || !m_shaders[shaderID].surf)
            continue;  // no shader attached
        if (m_shader_is_light[shaderID])
            m_lightprims.emplace_back(t);
    }
    if (!m_lightprims.empty())
        errhandler().infofmt("Found {} triangles to be treated as lights",
                             m_lightprims.size());
}


void
SimpleRaytracer::prepare_geometry()
{
    bool have_displacement = false;
    for (const Material& m : shaders()) {
        if (m.disp) {
            have_displacement = true;
            break;
        }
    }
    if (have_displacement) {
        errhandler().infofmt("Evaluating displacement shaders");
        // Loop through all triangles and run displacement shader if there is one
        // or copy the input point if there is none
        std::vector<Vec3> disp_verts(scene.verts.size(), Vec3(0, 0, 0));
        std::vector<int> valance(
            scene.verts.size(),
            0);  // number of times each vertex has been displaced

        OSL::PerThreadInfo* thread_info = shadingsys->create_thread_info();
        ShadingContext* ctx             = shadingsys->get_context(thread_info);

        bool has_smooth_normals = false;
        for (int primID = 0, nprims = scene.triangles.size(); primID < nprims;
             primID++) {
            Vec3 p[3], n[3];
            Vec2 uv[3];

            p[0] = scene.verts[scene.triangles[primID].a];
            p[1] = scene.verts[scene.triangles[primID].b];
            p[2] = scene.verts[scene.triangles[primID].c];

            valance[scene.triangles[primID].a]++;
            valance[scene.triangles[primID].b]++;
            valance[scene.triangles[primID].c]++;

            int shaderID = scene.shaderid(primID);
            if (shaderID < 0 || !m_shaders[shaderID].disp) {
                disp_verts[scene.triangles[primID].a] += p[0];
                disp_verts[scene.triangles[primID].b] += p[1];
                disp_verts[scene.triangles[primID].c] += p[2];
                continue;
            }


            Vec3 Ng    = (p[0] - p[1]).cross(p[0] - p[2]);
            float area = 0.5f * Ng.length();
            Ng         = Ng.normalize();
            if (scene.n_triangles[primID].a >= 0) {
                n[0]               = scene.normals[scene.n_triangles[primID].a];
                n[1]               = scene.normals[scene.n_triangles[primID].b];
                n[2]               = scene.normals[scene.n_triangles[primID].c];
                has_smooth_normals = true;
            } else {
                n[0] = n[1] = n[2] = Ng;
            }

            if (scene.uv_triangles[primID].a >= 0) {
                uv[0] = scene.uvs[scene.uv_triangles[primID].a];
                uv[1] = scene.uvs[scene.uv_triangles[primID].b];
                uv[2] = scene.uvs[scene.uv_triangles[primID].c];
            } else {
                uv[0] = uv[1] = uv[2] = Vec2(0, 0);
            }

            // displace each vertex
            for (int i = 0; i < 3; i++) {
                ShaderGlobals sg = {};
                sg.P             = p[i];
                sg.Ng            = Ng;
                sg.N             = n[i];
                sg.u             = uv[i].x;
                sg.v             = uv[i].y;
                sg.I             = (p[i] - camera.eye).normalize();
                sg.surfacearea   = area;

                shadingsys->execute(*ctx, *m_shaders[shaderID].disp, sg);

                p[i] = sg.P;
            }
            disp_verts[scene.triangles[primID].a] += p[0];
            disp_verts[scene.triangles[primID].b] += p[1];
            disp_verts[scene.triangles[primID].c] += p[2];
        }

        // release context
        shadingsys->release_context(ctx);
        shadingsys->destroy_thread_info(thread_info);

        // average each vertex by the number of times it was displaced
        for (int i = 0, n = scene.verts.size(); i < n; i++) {
            if (valance[i] > 0)
                disp_verts[i] /= float(valance[i]);
            else
                disp_verts[i] = scene.verts[i];
        }
        // replace old data with the new
        scene.verts = std::move(disp_verts);

        if (has_smooth_normals) {
            // Recompute the vertex normals (if we had some)
            std::vector<Vec3> disp_normals(scene.normals.size(), Vec3(0, 0, 0));
            for (int primID = 0, nprims = scene.triangles.size();
                 primID < nprims; primID++) {
                if (scene.n_triangles[primID].a >= 0) {
                    Vec3 p[3];
                    p[0] = scene.verts[scene.triangles[primID].a];
                    p[1] = scene.verts[scene.triangles[primID].b];
                    p[2] = scene.verts[scene.triangles[primID].c];
                    // don't normalize to weight by area
                    Vec3 Ng = (p[0] - p[1]).cross(p[0] - p[2]);
                    disp_normals[scene.n_triangles[primID].a] += Ng;
                    disp_normals[scene.n_triangles[primID].b] += Ng;
                    disp_normals[scene.n_triangles[primID].c] += Ng;
                }
            }
            for (Vec3& n : disp_normals)
                n = n.normalize();
            scene.normals = std::move(disp_normals);
        }
    }
}

void
SimpleRaytracer::render(int xres, int yres)
{
    OIIO::Timer timer;
    ShadingSystem* shadingsys = this->shadingsys;
    OIIO::parallel_for_chunked(
        0, yres, 0, [&, this](int64_t ybegin, int64_t yend) {
            // Request an OSL::PerThreadInfo for this thread.
            OSL::PerThreadInfo* thread_info = shadingsys->create_thread_info();

            // Request a shading context so that we can execute the shader.
            // We could get_context/release_context for each shading point,
            // but to save overhead, it's more efficient to reuse a context
            // within a thread.
            ShadingContext* ctx = shadingsys->get_context(thread_info);

            OIIO::ImageBuf::Iterator<float> p(pixelbuf,
                                              OIIO::ROI(0, xres, ybegin, yend));
            for (; !p.done(); ++p) {
                Color3 c = antialias_pixel(p.x(), p.y(), ctx);
                p[0]     = c.x;
                p[1]     = c.y;
                p[2]     = c.z;
            }

            // We're done shading with this context.
            shadingsys->release_context(ctx);
            shadingsys->destroy_thread_info(thread_info);
        });
    double rendertime = timer();
    errhandler().infofmt("Rendered {}x{} image with {} samples in {}", xres,
                         yres, aa * aa,
                         OIIO::Strutil::timeintervalformat(rendertime, 2));
}



void
SimpleRaytracer::clear()
{
    shaders().clear();
}

#endif  // #ifndef __CUDACC__

OSL_NAMESPACE_END
