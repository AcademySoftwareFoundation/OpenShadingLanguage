// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <cuda.h>

#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>


using TextureSamplerMap = std::unordered_map<OSL::ustringhash, cudaTextureObject_t>;

// Just use 4x4 matrix for transformations
typedef OSL::Matrix44 Transformation;
typedef std::map<OSL::ustringhash, std::shared_ptr<Transformation>> TransformMap;

class CudaGridRenderer final : public OSL::RendererServices {
    TextureSamplerMap _samplers;

    // Named transforms
    TransformMap _named_xforms;

    OSL::Matrix44 _world_to_camera;
    OSL::ustring _projection;
    float _fov, _pixelaspect, _hither, _yon;
    float _shutter[2];
    float _screen_window[4];
    int _xres, _yres;

public:
    CudaGridRenderer() {}
    virtual ~CudaGridRenderer() {}

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
    virtual TextureHandle* get_texture_handle(ustring filename,
                                              ShadingContext* shading_context,
                                              const TextureOpt* options);

    virtual bool get_matrix(ShaderGlobals* sg, Matrix44& result,
                            TransformationPtr xform, float time);
    virtual bool get_matrix(ShaderGlobals* sg, Matrix44& result,
                            ustringhash from, float time);
    virtual bool get_matrix(ShaderGlobals* sg, Matrix44& result,
                            TransformationPtr xform);
    virtual bool get_matrix(ShaderGlobals* sg, Matrix44& result,
                            ustringhash from);
    virtual bool get_inverse_matrix(ShaderGlobals* sg, Matrix44& result,
                                    ustringhash to, float time);

    void name_transform(const char* name, const Transformation& xform);

    virtual void* device_alloc(size_t size) override;
    virtual void device_free(void* ptr) override;
    virtual void* copy_to_device(void* dst_device, const void* src_host,
                                 size_t size) override;
};
