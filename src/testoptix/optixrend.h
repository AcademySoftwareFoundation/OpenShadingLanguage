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
#include <OSL/optix_compat.h>

#include "../testrender/raytracer.h"
#include "../testrender/optix_stringtable.h"

OSL_NAMESPACE_ENTER



class OptixRenderer : public RendererServices
{
public:
    // Just use 4x4 matrix for transformations
    typedef Matrix44 Transformation;

    OptixRenderer () { }
    virtual ~OptixRenderer () { }

    uint64_t register_string (const std::string& str, const std::string& var_name)
    {
        return m_str_table.addString (ustring(str), ustring(var_name));
    }

    virtual int supports (string_view feature) const
    {
        if (feature == "OptiX") {
            return true;
        }

        return false;
    }

    // Function stubs
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             TransformationPtr xform, float time)
    {
        return 0;
    }

    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             ustring from, float time)
    {
        return 0;
    }

    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             TransformationPtr xform)
    {
        return 0;
    }

    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             ustring from)
    {
        return 0;
    }

    virtual bool get_inverse_matrix (ShaderGlobals *sg, Matrix44 &result,
                                     ustring to, float time)
    {
        return 0;
    }

    virtual bool get_array_attribute (ShaderGlobals *sg, bool derivatives,
                                      ustring object, TypeDesc type, ustring name,
                                      int index, void *val)
    {
        return 0;
    }

    virtual bool get_attribute (ShaderGlobals *sg, bool derivatives, ustring object,
                                TypeDesc type, ustring name, void *val)
    {
        return 0;
    }

    virtual bool get_userdata (bool derivatives, ustring name, TypeDesc type,
                               ShaderGlobals *sg, void *val)
    {
        return 0;
    }


    /// Return true if the texture handle (previously returned by
    /// get_texture_handle()) is a valid texture that can be subsequently
    /// read or sampled.
    virtual bool good (TextureHandle *handle);

    /// Given the name of a texture, return an opaque handle that can be
    /// used with texture calls to avoid the name lookups.
    virtual TextureHandle * get_texture_handle(ustring filename);

    static bool load_ptx_from_file (const std::string& progName,
                                    std::string& ptx_string);

    // Create the optix-program, for the given resolution
    //
    virtual bool init(const std::string& progName, int xres, int yres, Scene* = nullptr);

    // Convert the OSL ShaderGroups accumulated during scene parsing into
    // OptiX Materials and set up the OptiX scene graph
    virtual bool finalize(ShadingSystem* shadingsys, bool saveptx, Scene* scene = nullptr);

    virtual void warmup ();
    virtual void render (int xres, int yres);

    // Copies the specified device buffer into an output vector, assuming
    // that the buffer is in FLOAT3 format (and that Vec3 and float3 have
    // the same underlying representation).
    virtual std::vector<OSL::Color3>
    getPixelBuffer(const std::string& buffer_name, int width, int height);

    bool
    saveImage(const std::string& buffer_name, int width, int height,
              const std::string& imagefile, OIIO::ErrorHandler* errHandler);

    virtual void clear();

    // ShaderGroupRef storage
    std::vector<ShaderGroupRef>& shaders() { return m_shaders; }

    // Easy way to do Optix calls on the OptixRenderer
    optix::Context& context()      { return optix_ctx; }
    optix::Context& operator -> () { return optix_ctx; }

    Camera camera;
    Scene scene;
private:
    optix::Context optix_ctx = nullptr;
    optix::Program m_program = nullptr;
    OptiXStringTable m_str_table;
    std::string renderer_ptx;  // ray generation, etc.
    std::string wrapper_ptx;   // hit programs
    std::string m_materials_ptx;
    std::unordered_map<OIIO::ustring, optix::TextureSampler, OIIO::ustringHash> m_samplers;
    unsigned              m_width, m_height;
    optix::Program sphere_intersect;
    optix::Program sphere_bounds;
    optix::Program quad_intersect;
    optix::Program quad_bounds;
    std::vector<optix::Material> optix_mtls;
    std::vector<ShaderGroupRef> m_shaders;
};


OSL_NAMESPACE_EXIT
