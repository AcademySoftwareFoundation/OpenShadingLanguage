// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <map>
#include <memory>
#include <unordered_map>

#include <OpenImageIO/imagebuf.h>

#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>
#include "background.h"
#include "raytracer.h"
#include "sampling.h"


OSL_NAMESPACE_BEGIN

struct Material {
    ShaderGroupRef surf;
    ShaderGroupRef disp;
};

using MaterialVec = std::vector<Material>;

class SimpleRaytracer : public RendererServices {
public:
    // Just use 4x4 matrix for transformations
    typedef Matrix44 Transformation;

    SimpleRaytracer();
    virtual ~SimpleRaytracer() {}

    // RendererServices support:
    bool get_matrix(ShaderGlobals* sg, Matrix44& result,
                    TransformationPtr xform, float time) override;
    bool get_matrix(ShaderGlobals* sg, Matrix44& result, ustringhash from,
                    float time) override;
    bool get_matrix(ShaderGlobals* sg, Matrix44& result,
                    TransformationPtr xform) override;
    bool get_matrix(ShaderGlobals* sg, Matrix44& result,
                    ustringhash from) override;
    bool get_inverse_matrix(ShaderGlobals* sg, Matrix44& result, ustringhash to,
                            float time) override;
    bool get_array_attribute(ShaderGlobals* sg, bool derivatives,
                             ustringhash object, TypeDesc type,
                             ustringhash name, int index, void* val) override;
    bool get_attribute(ShaderGlobals* sg, bool derivatives, ustringhash object,
                       TypeDesc type, ustringhash name, void* val) override;
    bool get_userdata(bool derivatives, ustringhash name, TypeDesc type,
                      ShaderGlobals* sg, void* val) override;

    void name_transform(const char* name, const Transformation& xform);

    // Set and get renderer attributes/options
    void attribute(string_view name, TypeDesc type, const void* value);
    void attribute(string_view name, int value)
    {
        attribute(name, TypeDesc::INT, &value);
    }
    void attribute(string_view name, float value)
    {
        attribute(name, TypeDesc::FLOAT, &value);
    }
    void attribute(string_view name, string_view value)
    {
        std::string valstr(value);
        const char* s = valstr.c_str();
        attribute(name, TypeDesc::STRING, &s);
    }
    OIIO::ParamValue* find_attribute(string_view name,
                                     TypeDesc searchtype = OIIO::TypeUnknown,
                                     bool casesensitive  = false);
    const OIIO::ParamValue*
    find_attribute(string_view name, TypeDesc searchtype = OIIO::TypeUnknown,
                   bool casesensitive = false) const;

    // Super simple camera and display parameters.  Many options not
    // available, no motion blur, etc.
    virtual void camera_params(const Matrix44& world_to_camera,
                               ustringhash projection, float hfov, float hither,
                               float yon, int xres, int yres);

    virtual void parse_scene_xml(const std::string& scenefile);
    virtual void prepare_render();
    void prepare_lights();
    void prepare_geometry();
    virtual void warmup() {}
    virtual void render(int xres, int yres);
    virtual void clear();

    // After render, get the pixels into pixelbuf, if they aren't already.
    virtual void finalize_pixel_buffer() {}

    // ShaderGroupRef storage
    MaterialVec& shaders() { return m_shaders; }

    OIIO::ErrorHandler& errhandler() const { return *m_errhandler; }

    const std::vector<bool>& shader_is_light() { return m_shader_is_light; }
    const std::vector<unsigned>& lightprims() { return m_lightprims; }

    Camera camera;
    Scene scene;
    Background background;
    ShadingSystem* shadingsys = nullptr;
    OIIO::ParamValueList options;
    OIIO::ImageBuf pixelbuf;

    int getBackgroundShaderID() const { return backgroundShaderID; }
    int getBackgroundResolution() const { return backgroundResolution; }

private:
    // Camera parameters
    Matrix44 m_world_to_camera;
    ustringhash m_projection;
    float m_fov, m_pixelaspect, m_hither, m_yon;
    float m_shutter[2];
    float m_screen_window[4];

    int backgroundShaderID   = -1;
    int backgroundResolution = 1024;
    int aa                   = 1;
    bool no_jitter           = false;
    int max_bounces          = 1000000;
    int rr_depth             = 5;
    float show_albedo_scale  = 0.0f;
    int show_globals         = 0;
    MaterialVec m_shaders;
    std::vector<bool> m_shader_is_light;
    std::vector<float>
        m_mesh_surfacearea;  // surface area of all triangles in each mesh (one entry per mesh)
    std::vector<unsigned>
        m_lightprims;  // array of all triangles that have a "light" shader on them

    class ErrorHandler;  // subclass ErrorHandler for SimpleRaytracer
    std::unique_ptr<OIIO::ErrorHandler> m_errhandler;
    bool m_had_error = false;

    // Named transforms
    typedef std::map<ustringhash, std::shared_ptr<Transformation>> TransformMap;
    TransformMap m_named_xforms;

    // Attribute and userdata retrieval -- for fast dispatch, use a hash
    // table to map attribute names to functions that retrieve them. We
    // imagine this to be fairly quick, but for a performance-critical
    // renderer, we would encourage benchmarking various methods and
    // alternate data structures.
    typedef bool (SimpleRaytracer::*AttrGetter)(ShaderGlobals* sg, bool derivs,
                                                ustringhash object,
                                                TypeDesc type, ustringhash name,
                                                void* val);
    typedef std::unordered_map<ustringhash, AttrGetter> AttrGetterMap;
    AttrGetterMap m_attr_getters;

    // Attribute getters
    bool get_osl_version(ShaderGlobals* sg, bool derivs, ustringhash object,
                         TypeDesc type, ustringhash name, void* val);
    bool get_camera_resolution(ShaderGlobals* sg, bool derivs,
                               ustringhash object, TypeDesc type,
                               ustringhash name, void* val);
    bool get_camera_projection(ShaderGlobals* sg, bool derivs,
                               ustringhash object, TypeDesc type,
                               ustringhash name, void* val);
    bool get_camera_fov(ShaderGlobals* sg, bool derivs, ustringhash object,
                        TypeDesc type, ustringhash name, void* val);
    bool get_camera_pixelaspect(ShaderGlobals* sg, bool derivs,
                                ustringhash object, TypeDesc type,
                                ustringhash name, void* val);
    bool get_camera_clip(ShaderGlobals* sg, bool derivs, ustringhash object,
                         TypeDesc type, ustringhash name, void* val);
    bool get_camera_clip_near(ShaderGlobals* sg, bool derivs,
                              ustringhash object, TypeDesc type,
                              ustringhash name, void* val);
    bool get_camera_clip_far(ShaderGlobals* sg, bool derivs, ustringhash object,
                             TypeDesc type, ustringhash name, void* val);
    bool get_camera_shutter(ShaderGlobals* sg, bool derivs, ustringhash object,
                            TypeDesc type, ustringhash name, void* val);
    bool get_camera_shutter_open(ShaderGlobals* sg, bool derivs,
                                 ustringhash object, TypeDesc type,
                                 ustringhash name, void* val);
    bool get_camera_shutter_close(ShaderGlobals* sg, bool derivs,
                                  ustringhash object, TypeDesc type,
                                  ustringhash name, void* val);
    bool get_camera_screen_window(ShaderGlobals* sg, bool derivs,
                                  ustringhash object, TypeDesc type,
                                  ustringhash name, void* val);

    // CPU renderer helpers
    void globals_from_hit(ShaderGlobals& sg, const Ray& r,
                          const Dual2<float>& t, int id, float u, float v);
    Vec3 eval_background(const Dual2<Vec3>& dir, ShadingContext* ctx,
                         int bounce = -1);
    Color3 subpixel_radiance(float x, float y, Sampler& sampler,
                             ShadingContext* ctx);
    Color3 antialias_pixel(int x, int y, ShadingContext* ctx);

    friend class ErrorHandler;
};

OSL_NAMESPACE_END
