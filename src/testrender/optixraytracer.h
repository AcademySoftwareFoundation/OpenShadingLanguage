/*
Copyright (c) 2019 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOTSS
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <OpenImageIO/ustring.h>
#include <OSL/oslexec.h>
#include <OSL/device_string.h>
#include "optix_compat.h"
#include "simpleraytracer.h"
#include "optix_stringtable.h"


OSL_NAMESPACE_ENTER


class OptixRaytracer : public SimpleRaytracer
{
public:
    // Just use 4x4 matrix for transformations
    typedef Matrix44 Transformation;

    OptixRaytracer () { }
    virtual ~OptixRaytracer ();

    uint64_t register_string (const std::string& str, const std::string& var_name)
    {
        return m_str_table.addString (ustring(str), ustring(var_name));
    }

    virtual int supports (string_view feature) const
    {
        if (feature == "OptiX")
            return true;
        return SimpleRaytracer::supports(feature);
    }

    std::string load_ptx_file (string_view filename);
    bool synch_attributes ();

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
    virtual TextureHandle * get_texture_handle(ustring filename);

    // Easy way to do Optix calls
    optix::Context& optix_ctx()            { return m_optix_ctx; }
    optix::Context& context()              { return m_optix_ctx; }
    optix::Context& operator -> ()         { return context(); }

private:
    OptiXStringTable m_str_table;
    optix::Context m_optix_ctx = nullptr;
    optix::Program m_program = nullptr;
    optix::Program sphere_intersect = nullptr;
    optix::Program sphere_bounds = nullptr;
    optix::Program quad_intersect = nullptr;
    optix::Program quad_bounds = nullptr;
    std::string m_materials_ptx;
    std::unordered_map<OIIO::ustring, optix::TextureSampler, OIIO::ustringHash> m_samplers;
};


OSL_NAMESPACE_EXIT
