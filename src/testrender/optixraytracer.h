// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OpenImageIO/ustring.h>

#include <OSL/oslexec.h>

#include <OSL/hashes.h>

#include "optix_compat.h"
#include "render_params.h"
#include "simpleraytracer.h"

OSL_NAMESPACE_BEGIN;


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

    bool make_optix_materials();
    bool init_optix_context(int xres, int yres);
    void create_modules();
    void create_programs();
    void create_shaders();
    void create_pipeline();
    void create_sbt();
    void cleanup_programs();
    void build_accel();
    void upload_mesh_data();
    void prepare_background();
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
    // OptiX state
    optix::Context m_optix_ctx                             = nullptr;
    CUstream m_cuda_stream                                 = 0;
    OptixTraversableHandle m_travHandle                    = {};
    OptixShaderBindingTable m_optix_sbt                    = {};
    OptixShaderBindingTable m_setglobals_optix_sbt         = {};
    OptixPipeline m_optix_pipeline                         = {};
    OptixModuleCompileOptions m_module_compile_options     = {};
    OptixPipelineCompileOptions m_pipeline_compile_options = {};
    OptixPipelineLinkOptions m_pipeline_link_options       = {};
    OptixProgramGroupOptions m_program_options             = {};
    OptixModule m_program_module                           = {};
    OptixModule m_rend_lib_module                          = {};
    OptixModule m_shadeops_module                          = {};
    OptixProgramGroup m_raygen_group                       = {};
    OptixProgramGroup m_miss_group                         = {};
    OptixProgramGroup m_rend_lib_group                     = {};
    OptixProgramGroup m_shadeops_group                     = {};
    OptixProgramGroup m_setglobals_raygen_group            = {};
    OptixProgramGroup m_setglobals_miss_group              = {};
    OptixProgramGroup m_closesthit_group                   = {};
    std::vector<OptixModule> m_shader_modules;
    std::vector<OptixProgramGroup> m_shader_groups;
    std::vector<OptixProgramGroup> m_final_groups;

    // Device pointers
    CUdeviceptr d_output_buffer       = 0;
    CUdeviceptr d_launch_params       = 0;
    CUdeviceptr d_accel_output_buffer = 0;
    CUdeviceptr d_vertices            = 0;
    CUdeviceptr d_normals             = 0;
    CUdeviceptr d_uvs                 = 0;
    CUdeviceptr d_vert_indices        = 0;
    CUdeviceptr d_normal_indices      = 0;
    CUdeviceptr d_uv_indices          = 0;
    CUdeviceptr d_shader_ids          = 0;
    CUdeviceptr d_shader_is_light     = 0;
    CUdeviceptr d_mesh_ids            = 0;
    CUdeviceptr d_surfacearea         = 0;
    CUdeviceptr d_lightprims          = 0;
    CUdeviceptr d_interactive_params  = 0;
    CUdeviceptr d_bg_values           = 0;
    CUdeviceptr d_bg_rows             = 0;
    CUdeviceptr d_bg_cols             = 0;
    CUdeviceptr d_osl_printf_buffer   = 0;
    CUdeviceptr d_color_system        = 0;

    uint64_t test_str_1                        = 0;
    uint64_t test_str_2                        = 0;
    int m_xres                                 = 0;
    int m_yres                                 = 0;
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

    // CUdeviceptrs that need to be freed after we are done
    std::vector<CUdeviceptr> m_ptrs_to_free;
    std::vector<cudaArray_t> m_arrays_to_free;
};


OSL_NAMESPACE_END
