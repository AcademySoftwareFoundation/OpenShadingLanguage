// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <map>
#include <memory>
#include <unordered_map>

#include <OpenImageIO/ustring.h>
#include <OpenImageIO/imagebuf.h>

#include <OSL/oslexec.h>

#include "batched_simplerend.h"


OSL_NAMESPACE_ENTER


void register_closures(OSL::ShadingSystem* shadingsys);



class SimpleRenderer : public RendererServices
{
    template<int>
    friend class BatchedSimpleRenderer;
public:
    // Just use 4x4 matrix for transformations
    typedef Matrix44 Transformation;

    SimpleRenderer ();
    // Ensure destructor is in .cpp
    ~SimpleRenderer ();

    virtual int supports (string_view feature) const;
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             TransformationPtr xform,
                             float time);
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             ustring from, float time);
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             TransformationPtr xform);
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             ustring from);
    virtual bool get_inverse_matrix (ShaderGlobals *sg, Matrix44 &result,
                                     ustring to, float time);

    void name_transform (const char *name, const Transformation &xform);

    virtual bool get_array_attribute (ShaderGlobals *sg, bool derivatives, 
                                      ustring object, TypeDesc type, ustring name,
                                      int index, void *val );
    virtual bool get_attribute (ShaderGlobals *sg, bool derivatives, ustring object,
                                TypeDesc type, ustring name, void *val);
    virtual bool get_userdata (bool derivatives, ustring name, TypeDesc type, 
                               ShaderGlobals *sg, void *val);


    // Set and get renderer attributes/options
    void attribute (string_view name, TypeDesc type, const void *value);
    void attribute (string_view name, int value) {
        attribute (name, TypeDesc::INT, &value);
    }
    void attribute (string_view name, float value) {
        attribute (name, TypeDesc::FLOAT, &value);
    }
    void attribute (string_view name, string_view value) {
        const char *s = value.c_str();
        attribute (name, TypeDesc::STRING, &s);
    }
    OIIO::ParamValue * find_attribute (string_view name,
                                       TypeDesc searchtype=OIIO::TypeUnknown,
                                       bool casesensitive=false);
    const OIIO::ParamValue *find_attribute (string_view name,
                                            TypeDesc searchtype=OIIO::TypeUnknown,
                                            bool casesensitive=false) const;

    // Super simple camera and display parameters.  Many options not
    // available, no motion blur, etc.
    void camera_params (const Matrix44 &world_to_camera, ustring projection,
                        float hfov, float hither, float yon,
                        int xres, int yres);

    virtual bool add_output (string_view varname, string_view filename,
                             TypeDesc datatype = OIIO::TypeFloat,
                             int nchannels = 3);

    OIIO::ImageBuf* outputbuf (int index) {
        return index < (int)m_outputbufs.size() ? m_outputbufs[index].get() : nullptr;
    }
    ustring outputname (int index) const { return m_outputvars[index]; }
    size_t noutputs () const { return m_outputbufs.size(); }

    virtual void init_shadingsys (ShadingSystem *ss) {
        shadingsys = ss;
    }
    virtual void prepare_render () { }
    virtual void warmup () { }
    virtual void render (int /*xres*/, int /*yres*/) { }
    virtual void clear () { }

    // After render, get the pixel data into the output buffers, if
    // they aren't already.
    virtual void finalize_pixel_buffer () { }

    // ShaderGroupRef storage
    std::vector<ShaderGroupRef>& shaders() { return m_shaders; }

    OIIO::ErrorHandler& errhandler() const { return *m_errhandler; }

    ShadingSystem *shadingsys = nullptr;
    OIIO::ParamValueList options;
    OIIO::ParamValueList userdata;

    virtual BatchedRendererServices<16> * batched(WidthOf<16>) { return &m_batch_16_simple_renderer; }
    virtual BatchedRendererServices<8> * batched(WidthOf<8>) { return &m_batch_8_simple_renderer; }

protected:
    BatchedSimpleRenderer<16> m_batch_16_simple_renderer;
    BatchedSimpleRenderer<8> m_batch_8_simple_renderer;

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

    // Named transforms
    typedef std::map <ustring, std::shared_ptr<Transformation> > TransformMap;
    TransformMap m_named_xforms;

    // Attribute and userdata retrieval -- for fast dispatch, use a hash
    // table to map attribute names to functions that retrieve them. We
    // imagine this to be fairly quick, but for a performance-critical
    // renderer, we would encourage benchmarking various methods and
    // alternate data structures.
    typedef bool (SimpleRenderer::*AttrGetter)(ShaderGlobals *sg, bool derivs,
                                               ustring object, TypeDesc type,
                                               ustring name, void *val);
    typedef std::unordered_map<ustring, AttrGetter, ustringHash> AttrGetterMap;
    AttrGetterMap m_attr_getters;

    // Attribute getters
    bool get_osl_version (ShaderGlobals *sg, bool derivs, ustring object,
                         TypeDesc type, ustring name, void *val);
    bool get_camera_resolution (ShaderGlobals *sg, bool derivs, ustring object,
                         TypeDesc type, ustring name, void *val);
    bool get_camera_projection (ShaderGlobals *sg, bool derivs, ustring object,
                         TypeDesc type, ustring name, void *val);
    bool get_camera_fov (ShaderGlobals *sg, bool derivs, ustring object,
                         TypeDesc type, ustring name, void *val);
    bool get_camera_pixelaspect (ShaderGlobals *sg, bool derivs, ustring object,
                         TypeDesc type, ustring name, void *val);
    bool get_camera_clip (ShaderGlobals *sg, bool derivs, ustring object,
                         TypeDesc type, ustring name, void *val);
    bool get_camera_clip_near (ShaderGlobals *sg, bool derivs, ustring object,
                         TypeDesc type, ustring name, void *val);
    bool get_camera_clip_far (ShaderGlobals *sg, bool derivs, ustring object,
                         TypeDesc type, ustring name, void *val);
    bool get_camera_shutter (ShaderGlobals *sg, bool derivs, ustring object,
                         TypeDesc type, ustring name, void *val);
    bool get_camera_shutter_open (ShaderGlobals *sg, bool derivs, ustring object,
                         TypeDesc type, ustring name, void *val);
    bool get_camera_shutter_close (ShaderGlobals *sg, bool derivs, ustring object,
                         TypeDesc type, ustring name, void *val);
    bool get_camera_screen_window (ShaderGlobals *sg, bool derivs, ustring object,
                         TypeDesc type, ustring name, void *val);

};

OSL_NAMESPACE_EXIT
