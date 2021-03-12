// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OpenImageIO/ustring.h>
#include <OSL/oslexec.h>
#include <OSL/device_string.h>
#include "optix_compat.h"
#include "simplerend.h"
#if OPTIX_VERSION < 70000
#include "../testrender/optix_stringtable.h"
#endif

OSL_NAMESPACE_ENTER


class OptixGridRenderer final : public SimpleRenderer
{
public:
    // Just use 4x4 matrix for transformations
    typedef Matrix44 Transformation;

    OptixGridRenderer ();
    virtual ~OptixGridRenderer ();

    uint64_t register_string (const std::string& str, const std::string& var_name)
    {
        ustring ustr = ustring(str);
#if OPTIX_VERSION < 70000
        uint64_t addr = m_str_table.addString (ustr, ustring(var_name));
        if (!var_name.empty()) {
            register_global (var_name, addr);
        }
        return addr;
#else
        m_hash_map[ustr.hash()] = ustr.c_str();
        return 0;
#endif
    }

    uint64_t register_global (const std::string& str, uint64_t value);
    bool     fetch_global (const std::string& str, uint64_t *value);

    virtual int supports (string_view feature) const
    {
        if (feature == "OptiX")
            return true;
        return SimpleRenderer::supports(feature);
    }

    std::string load_ptx_file (string_view filename);
    bool synch_attributes ();

    virtual void init_shadingsys (ShadingSystem *ss);
    virtual bool init_optix_context (int xres, int yres);
    virtual bool make_optix_materials ();
    virtual bool finalize_scene ();
    virtual void prepare_render ();
    virtual void warmup ();
    virtual void render (int xres, int yres);
    virtual void finalize_pixel_buffer ();
    virtual void clear ();

    /// Return true if the texture handle (previously returned by
    /// get_texture_handle()) is a valid texture that can be subsequently
    /// read or sampled.
    virtual bool good (TextureHandle *handle);

    /// Given the name of a texture, return an opaque handle that can be
    /// used with texture calls to avoid the name lookups.
    virtual TextureHandle * get_texture_handle(ustring filename, ShadingContext* shading_context);

#if (OPTIX_VERSION < 70000)
    // Easy way to do Optix calls
    optix::Context& optix_ctx()            { return m_optix_ctx; }
    optix::Context& context()              { return m_optix_ctx; }
    optix::Context& operator -> ()         { return context(); }
#else
    OptixDeviceContext optix_ctx()         { return m_optix_ctx; }
    OptixDeviceContext context()           { return m_optix_ctx; }
    OptixDeviceContext operator -> ()      { return context(); }

    void processPrintfBuffer(void *buffer_data, size_t buffer_size);
#endif

private:

    optix::Context m_optix_ctx = nullptr;

#if (OPTIX_VERSION < 70000)
    OptiXStringTable m_str_table;
    optix::Program m_program = nullptr;
#else
    CUstream                m_cuda_stream;
    OptixShaderBindingTable m_optix_sbt = {};
    OptixShaderBindingTable m_setglobals_optix_sbt = {};
    OptixPipeline           m_optix_pipeline = {};
    CUdeviceptr             d_output_buffer;
    CUdeviceptr             d_launch_params = 0;
    CUdeviceptr             d_osl_printf_buffer;
    CUdeviceptr             d_color_system;
    uint64_t                test_str_1;
    uint64_t                test_str_2;
    const unsigned long     OSL_PRINTF_BUFFER_SIZE = 8 * 1024 * 1024;

    std::unordered_map<uint64_t, const char *> m_hash_map;
#endif
    std::string m_materials_ptx;
    std::unordered_map<OIIO::ustring, optix::TextureSampler, OIIO::ustringHash> m_samplers;
    std::unordered_map<OIIO::ustring, uint64_t, OIIO::ustringHash> m_globals_map;
};

#if (OPTIX_VERSION >= 70000)
struct EmptyRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
};
#endif

OSL_NAMESPACE_EXIT
