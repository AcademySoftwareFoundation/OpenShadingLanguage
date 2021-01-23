// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <cuda.h>

#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>

#include "cuda_string_table.h"

using GlobalsMap
    = std::unordered_map<OIIO::ustring, uint64_t, OIIO::ustringHash>;
using TextureSamplerMap
    = std::unordered_map<OIIO::ustring, cudaTextureObject_t, OIIO::ustringHash>;

// Just use 4x4 matrix for transformations
typedef OSL::Matrix44 Transformation;
typedef std::map<OIIO::ustring, std::shared_ptr<Transformation>> TransformMap;

class CudaGridRenderer final : public OSL::RendererServices {
    CudaStringTable _string_table;
    TextureSamplerMap _samplers;
    GlobalsMap _globals_map;

    // Named transforms
    TransformMap _named_xforms;

    OSL::Matrix44 _world_to_camera;
    OIIO::ustring _projection;
    float _fov, _pixelaspect, _hither, _yon;
    float _shutter[2];
    float _screen_window[4];
    int _xres, _yres;

public:
    CudaGridRenderer();
    virtual ~CudaGridRenderer();

    uint64_t register_string(const std::string& str,
                             const std::string& var_name)
    {
        uint64_t addr = _string_table.addString(OIIO::ustring(str),
                                                OIIO::ustring(var_name));
        if (!var_name.empty()) {
            register_global(var_name, addr);
        }
        return addr;
    }

    uint64_t register_global(const std::string& str, uint64_t value);
    bool fetch_global(const std::string& str, uint64_t* value);

    const GlobalsMap& globals_map() const { return _globals_map; }

    virtual int supports(OIIO::string_view feature) const
    {
        if (feature == "OptiX") {
            return true;
        }

        return false;
    }

    /// Return true if the texture handle (previously returned by
    /// get_texture_handle()) is a valid texture that can be subsequently
    /// read or sampled.
    virtual bool good(TextureHandle* handle);

    /// Given the name of a texture, return an opaque handle that can be
    /// used with texture calls to avoid the name lookups.
    virtual TextureHandle*
    get_texture_handle(OIIO::ustring filename,
                       OSL::ShadingContext* shading_context);

    virtual bool get_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OSL::TransformationPtr xform, float time);
    virtual bool get_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OIIO::ustring from, float time);
    virtual bool get_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OSL::TransformationPtr xform);
    virtual bool get_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OIIO::ustring from);
    virtual bool get_inverse_matrix(OSL::ShaderGlobals* sg,
                                    OSL::Matrix44& result, OIIO::ustring to,
                                    float time);

    void name_transform(const char* name, const Transformation& xform);
};
