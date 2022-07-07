// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OpenImageIO/ustring.h>

#include <OSL/oslexec.h>

#include <OSL/device_string.h>

#include "optix_compat.h"
#include "simplerend.h"


OSL_NAMESPACE_ENTER


class OptixGridRenderer final : public SimpleRenderer {
public:
    // Just use 4x4 matrix for transformations
    typedef Matrix44 Transformation;

    OptixGridRenderer();
    virtual ~OptixGridRenderer();

    uint64_t register_global(const std::string& str, uint64_t value);
    bool fetch_global(const std::string& str, uint64_t* value);

    virtual int supports(string_view feature) const
    {
        if (feature == "OptiX")
            return true;
        return SimpleRenderer::supports(feature);
    }

    std::string load_ptx_file(string_view filename);
    bool synch_attributes();

    virtual void init_shadingsys(ShadingSystem* ss);
    virtual bool init_optix_context(int xres, int yres);
    virtual bool make_optix_materials();
    virtual bool finalize_scene();
    virtual void prepare_render();
    virtual void warmup();
    virtual void render(int xres, int yres);
    virtual void finalize_pixel_buffer();
    virtual void clear();

    virtual void set_transforms(const OSL::Matrix44& object2common,
                                const OSL::Matrix44& shader2common);

    virtual void register_named_transforms();

    /// Return true if the texture handle (previously returned by
    /// get_texture_handle()) is a valid texture that can be subsequently
    /// read or sampled.
    virtual bool good(TextureHandle* handle);

    /// Given the name of a texture, return an opaque handle that can be
    /// used with texture calls to avoid the name lookups.
    virtual TextureHandle* get_texture_handle(ustring filename,
                                              ShadingContext* shading_context);

    OptixDeviceContext optix_ctx() { return m_optix_ctx; }
    OptixDeviceContext context() { return m_optix_ctx; }
    OptixDeviceContext operator->() { return context(); }

    void processPrintfBuffer(void* buffer_data, size_t buffer_size);

private:
    optix::Context m_optix_ctx = nullptr;

    CUstream m_cuda_stream;
    OptixShaderBindingTable m_optix_sbt = {};
    OptixShaderBindingTable m_setglobals_optix_sbt = {};
    OptixPipeline m_optix_pipeline = {};
    CUdeviceptr d_output_buffer;
    CUdeviceptr d_launch_params = 0;
    CUdeviceptr d_osl_printf_buffer;
    CUdeviceptr d_color_system;
    CUdeviceptr d_object2common;
    CUdeviceptr d_shader2common;
    uint64_t m_num_named_xforms;
    CUdeviceptr d_xform_name_buffer;
    CUdeviceptr d_xform_buffer;
    uint64_t test_str_1;
    uint64_t test_str_2;
    const unsigned long OSL_PRINTF_BUFFER_SIZE = 8 * 1024 * 1024;

    std::string m_materials_ptx;
    std::unordered_map<OIIO::ustring, optix::TextureSampler, OIIO::ustringHash>
        m_samplers;
    std::unordered_map<OIIO::ustring, uint64_t, OIIO::ustringHash> m_globals_map;

    OSL::Matrix44 m_shader2common;  // "shader" space to "common" space matrix
    OSL::Matrix44 m_object2common;  // "object" space to "common" space matrix

    // CUdeviceptrs that need to be freed after we are done
    std::vector<void*> m_ptrs_to_free;
};



struct EmptyRecord {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void* data;
};

OSL_NAMESPACE_EXIT
