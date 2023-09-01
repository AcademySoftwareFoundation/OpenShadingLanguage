// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <map>
#include <memory>
#include <unordered_map>

#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/ustring.h>

#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>

#if OSL_USE_BATCHED
#    include "batched_simplerend.h"
#endif

#include "render_state.h"

OSL_NAMESPACE_ENTER


void
register_closures(OSL::ShadingSystem* shadingsys);



class SimpleRenderer : public RendererServices {
    // Keep implementation in sync with rs_simplerend.cpp
    template<int> friend class BatchedSimpleRenderer;

public:
    // Just use 4x4 matrix for transformations
    typedef Matrix44 Transformation;

    SimpleRenderer();
    // Ensure destructor is in .cpp
    ~SimpleRenderer();

    int supports(string_view feature) const override;
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

    void name_transform(const char* name, const Transformation& xform);

    bool get_array_attribute(ShaderGlobals* sg, bool derivatives,
                             ustringhash object, TypeDesc type,
                             ustringhash name, int index, void* val) override;
    bool get_attribute(ShaderGlobals* sg, bool derivatives, ustringhash object,
                       TypeDesc type, ustringhash name, void* val) override;
    bool get_userdata(bool derivatives, ustringhash name, TypeDesc type,
                      ShaderGlobals* sg, void* val) override;

    void build_attribute_getter(const ShaderGroup& group, bool is_object_lookup,
                                const ustring* object_name,
                                const ustring* attribute_name,
                                bool is_array_lookup, const int* array_index,
                                TypeDesc type, bool derivatives,
                                AttributeGetterSpec& spec) override;

    bool trace(TraceOpt& options, ShaderGlobals* sg, const OSL::Vec3& P,
               const OSL::Vec3& dPdx, const OSL::Vec3& dPdy, const OSL::Vec3& R,
               const OSL::Vec3& dRdx, const OSL::Vec3& dRdy) override;

    bool getmessage(ShaderGlobals* sg, ustringhash source, ustringhash name,
                    TypeDesc type, void* val, bool derivatives) override;

    void errorfmt(OSL::ShaderGlobals* sg, OSL::ustringhash fmt_specification,
                  int32_t count, const EncodedType* argTypes,
                  uint32_t argValuesSize, uint8_t* argValues) override;
    void warningfmt(OSL::ShaderGlobals* sg, OSL::ustringhash fmt_specification,
                    int32_t count, const EncodedType* argTypes,
                    uint32_t argValuesSize, uint8_t* argValues) override;
    void printfmt(OSL::ShaderGlobals* sg, OSL::ustringhash fmt_specification,
                  int32_t count, const EncodedType* argTypes,
                  uint32_t argValuesSize, uint8_t* argValues) override;
    void filefmt(OSL::ShaderGlobals* sg, OSL::ustringhash filename_hash,
                 OSL::ustringhash fmt_specification, int32_t arg_count,
                 const EncodedType* argTypes, uint32_t argValuesSize,
                 uint8_t* argValues) override;

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
    void camera_params(const Matrix44& world_to_camera, ustring projection,
                       float hfov, float hither, float yon, int xres, int yres);

    virtual bool add_output(string_view varname, string_view filename,
                            TypeDesc datatype = OIIO::TypeFloat,
                            int nchannels     = 3);

    // Get the output ImageBuf by index
    OIIO::ImageBuf* outputbuf(int index)
    {
        return index < (int)m_outputbufs.size() ? m_outputbufs[index].get()
                                                : nullptr;
    }
    // Get the output ImageBuf by name
    OIIO::ImageBuf* outputbuf(string_view name)
    {
        for (size_t i = 0; i < m_outputbufs.size(); ++i)
            if (m_outputvars[i] == name)
                return m_outputbufs[i].get();
        return nullptr;
    }
    ustring outputname(int index) const { return m_outputvars[index]; }
    size_t noutputs() const { return m_outputbufs.size(); }

    virtual void init_shadingsys(ShadingSystem* ss) { shadingsys = ss; }
    virtual void export_state(RenderState&) const;
    virtual void prepare_render() {}
    virtual void warmup() {}
    virtual void render(int /*xres*/, int /*yres*/) {}
    virtual void clear() { m_shaders.clear(); }

    // After render, get the pixel data into the output buffers, if
    // they aren't already.
    virtual void finalize_pixel_buffer() {}

    void use_rs_bitcode(bool enabled) { m_use_rs_bitcode = enabled; }

    static void register_JIT_Global_Variables();

    // ShaderGroupRef storage
    std::vector<ShaderGroupRef>& shaders() { return m_shaders; }

    OIIO::ErrorHandler& errhandler() const { return *m_errhandler; }

    ShadingSystem* shadingsys = nullptr;
    OIIO::ParamValueList options;
    OIIO::ParamValueList userdata;

#if OSL_USE_BATCHED
    BatchedRendererServices<16>* batched(WidthOf<16>) override
    {
        return &m_batch_16_simple_renderer;
    }
    BatchedRendererServices<8>* batched(WidthOf<8>) override
    {
        return &m_batch_8_simple_renderer;
    }
#endif

protected:
#if OSL_USE_BATCHED
    BatchedSimpleRenderer<16> m_batch_16_simple_renderer;
    BatchedSimpleRenderer<8> m_batch_8_simple_renderer;
#endif

    // Camera parameters
    Matrix44 m_world_to_camera;
    ustring m_projection;
    float m_fov, m_pixelaspect, m_hither, m_yon;
    float m_shutter[2];
    float m_screen_window[4];
    int m_xres, m_yres;
    std::vector<ShaderGroupRef> m_shaders;
    std::vector<ustring> m_outputvars;
    std::vector<std::shared_ptr<OIIO::ImageBuf>> m_outputbufs;
    std::unique_ptr<OIIO::ErrorHandler> m_errhandler { new OIIO::ErrorHandler };
    bool m_use_rs_bitcode = false;

    // Named transforms
    typedef std::map<ustringhash, std::shared_ptr<Transformation>> TransformMap;
    TransformMap m_named_xforms;

    // Attribute and userdata retrieval -- for fast dispatch, use a hash
    // table to map attribute names to functions that retrieve them. We
    // imagine this to be fairly quick, but for a performance-critical
    // renderer, we would encourage benchmarking various methods and
    // alternate data structures.
    typedef bool (SimpleRenderer::*AttrGetter)(ShaderGlobals* sg, bool derivs,
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
};

OSL_NAMESPACE_EXIT
