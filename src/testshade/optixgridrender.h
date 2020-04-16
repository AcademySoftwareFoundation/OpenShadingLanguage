// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#pragma once

#include <OpenImageIO/ustring.h>
#include <OSL/oslexec.h>
#include <OSL/device_string.h>
#include "optix_compat.h"
#include "simplerend.h"
#include "../testrender/optix_stringtable.h"


OSL_NAMESPACE_ENTER


class OptixGridRenderer : public SimpleRenderer
{
public:
    // Just use 4x4 matrix for transformations
    typedef Matrix44 Transformation;

    OptixGridRenderer ();
    virtual ~OptixGridRenderer ();

    uint64_t register_string (const std::string& str, const std::string& var_name)
    {
        return m_str_table.addString (ustring(str), ustring(var_name));
    }

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

    // Easy way to do Optix calls
    optix::Context& optix_ctx()            { return m_optix_ctx; }
    optix::Context& context()              { return m_optix_ctx; }
    optix::Context& operator -> ()         { return context(); }

private:
    OptiXStringTable m_str_table;
    optix::Context m_optix_ctx = nullptr;
    optix::Program m_program = nullptr;
    std::string m_materials_ptx;
    std::unordered_map<OIIO::ustring, optix::TextureSampler, OIIO::ustringHash> m_samplers;
};


OSL_NAMESPACE_EXIT
