// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OpenImageIO/ustring.h>

#include <OSL/oslexec.h>

#include <OSL/device_string.h>

#include "optix_compat.h"
#include "render_params.h"
#include "simpleraytracer.h"

OSL_NAMESPACE_ENTER


class OptixRaytracer final : public SimpleRaytracer {
public:
    // Just use 4x4 matrix for transformations
    typedef Matrix44 Transformation;

    OptixRaytracer();
    virtual ~OptixRaytracer();

    int supports(string_view feature) const override
    {
        if (feature == "OptiX")
            return true;
        return SimpleRaytracer::supports(feature);
    }

    std::string load_ptx_file(string_view filename);
    bool synch_attributes();

    bool init_optix_context(int xres, int yres);
    bool make_optix_materials();
    bool finalize_scene();
    void prepare_render() override;
    void warmup() override;
    void render(int xres, int yres) override;
    void finalize_pixel_buffer() override;
    void clear() override;

    /// Return true if the texture handle (previously returned by
    /// get_texture_handle()) is a valid texture that can be subsequently
    /// read or sampled.
    bool good(TextureHandle* handle) override;

    /// Given the name of a texture, return an opaque handle that can be
    /// used with texture calls to avoid the name lookups.
    TextureHandle* get_texture_handle(ustring filename,
                                      ShadingContext* shading_context,
                                      const TextureOpt* options) override;

    // Easy way to do Optix calls
    optix::Context& optix_ctx() { return m_optix_ctx; }
    optix::Context& context() { return m_optix_ctx; }
    optix::Context& operator->() { return context(); }

    void processPrintfBuffer(void* buffer_data, size_t buffer_size);

    virtual void* device_alloc(size_t size) override;
    virtual void device_free(void* ptr) override;
    virtual void* copy_to_device(void* dst_device, const void* src_host,
                                 size_t size) override;

private:
    optix::Context m_optix_ctx = nullptr;

    CUstream m_cuda_stream;
    OptixTraversableHandle m_travHandle;
    OptixShaderBindingTable m_optix_sbt            = {};
    OptixShaderBindingTable m_setglobals_optix_sbt = {};
    OptixPipeline m_optix_pipeline                 = {};
    CUdeviceptr d_output_buffer;
    CUdeviceptr d_launch_params      = 0;
    CUdeviceptr d_quads_list         = 0;
    CUdeviceptr d_spheres_list       = 0;
    CUdeviceptr d_interactive_params = 0;
    int m_xres, m_yres;
    CUdeviceptr d_osl_printf_buffer;
    CUdeviceptr d_color_system;
    uint64_t test_str_1;
    uint64_t test_str_2;
    const unsigned long OSL_PRINTF_BUFFER_SIZE = 8 * 1024 * 1024;

    bool load_optix_module(
        const char* filename,
        const OptixModuleCompileOptions* module_compile_options,
        const OptixPipelineCompileOptions* pipeline_compile_options,
        OptixModule* program_module);
    bool create_optix_pg(const OptixProgramGroupDesc* pg_desc, const int num_pg,
                         OptixProgramGroupOptions* program_options,
                         OptixProgramGroup* pg);

    std::string m_materials_ptx;
    std::unordered_map<ustringhash, optix::TextureSampler> m_samplers;
};


OSL_NAMESPACE_EXIT
