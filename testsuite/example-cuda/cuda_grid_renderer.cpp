// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/sysutil.h>

#include <OSL/genclosure.h>
#include <OSL/oslclosure.h>
#include <OSL/oslexec.h>

#include <OSL/device_string.h>

#include <cuda.h>
#include <nvrtc.h>

#include "cuda_grid_renderer.h"
#include "cuda_macro.h"

using namespace OSL;
using OIIO::TextureSystem;
using OIIO::ustring;

static ustring u_camera("camera"), u_screen("screen");
static ustring u_NDC("NDC"), u_raster("raster");
static ustring u_perspective("perspective");
static ustring u_s("s"), u_t("t");

CudaGridRenderer::CudaGridRenderer()
{
    // Set up the string table. This allocates a block of CUDA device memory to
    // hold all of the static strings used by the OSL shaders. The strings can
    // be accessed via OptiX variables that hold pointers to the table entries.
    _string_table.init();

    // Register all of our string table entries
    for (auto&& gvar : _string_table.contents()) {
        register_global(gvar.first.c_str(), gvar.second);
    }
}

CudaGridRenderer::~CudaGridRenderer()
{
    _string_table.freetable();
}

uint64_t
CudaGridRenderer::register_global(const std::string& str, uint64_t value)
{
    auto it = _globals_map.find(ustring(str));

    if (it != _globals_map.end()) {
        return it->second;
    }
    _globals_map[ustring(str)] = value;
    return value;
}

bool
CudaGridRenderer::fetch_global(const std::string& str, uint64_t* value)
{
    auto it = _globals_map.find(ustring(str));

    if (it != _globals_map.end()) {
        *value = it->second;
        return true;
    }
    return false;
}

/// Return true if the texture handle (previously returned by
/// get_texture_handle()) is a valid texture that can be subsequently
/// read or sampled.
bool
CudaGridRenderer::good(TextureHandle* handle)
{
    return handle != nullptr;
}

/// Given the name of a texture, return an opaque handle that can be
/// used with texture calls to avoid the name lookups.
RendererServices::TextureHandle*
CudaGridRenderer::get_texture_handle(OIIO::ustring filename,
                                     ShadingContext* shading_context)
{
    auto itr = _samplers.find(filename);
    if (itr == _samplers.end()) {
        // Open image
        OIIO::ImageBuf image;
        if (!image.init_spec(filename, 0, 0)) {
            std::cerr << "Could not load " << filename << std::endl;
            return (TextureHandle*)(intptr_t(nullptr));
        }

        OIIO::ROI roi = OIIO::get_roi_full(image.spec());
        int32_t width = roi.width(), height = roi.height();
        std::vector<float> pixels(width * height * 4);

        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                image.getpixel(i, j, 0, &pixels[((j * width) + i) * 4 + 0]);
            }
        }
        cudaResourceDesc res_desc = {};

        // hard-code textures to 4 channels
        int32_t pitch = width * 4 * sizeof(float);
        cudaChannelFormatDesc channel_desc
            = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

        cudaArray_t pixelArray;
        CUDA_CHECK(cudaMallocArray(&pixelArray, &channel_desc, width, height));

        CUDA_CHECK(cudaMemcpy2DToArray(pixelArray,
                                       /* offset */ 0, 0, pixels.data(), pitch,
                                       pitch, height, cudaMemcpyHostToDevice));

        res_desc.resType         = cudaResourceTypeArray;
        res_desc.res.array.array = pixelArray;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0]  = cudaAddressModeWrap;
        tex_desc.addressMode[1]  = cudaAddressModeWrap;
        tex_desc.filterMode      = cudaFilterModeLinear;
        tex_desc.readMode
            = cudaReadModeElementType;  // cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords    = 1;
        tex_desc.maxAnisotropy       = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode    = cudaFilterModePoint;
        tex_desc.borderColor[0]      = 1.0f;
        tex_desc.sRGB                = 0;

        // Create texture object
        cudaTextureObject_t cuda_tex = 0;
        CUDA_CHECK(
            cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
        itr = _samplers.emplace(std::move(filename), std::move(cuda_tex)).first;
    }
    return reinterpret_cast<RendererServices::TextureHandle*>(itr->second);
}

bool
CudaGridRenderer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                             TransformationPtr xform, float /*time*/)
{
    // CudaGridRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    result = *reinterpret_cast<const Matrix44*>(xform);
    return true;
}

bool
CudaGridRenderer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                             ustring from, float /*time*/)
{
    TransformMap::const_iterator found = _named_xforms.find(from);
    if (found != _named_xforms.end()) {
        result = *(found->second);
        return true;
    } else {
        return false;
    }
}

bool
CudaGridRenderer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                             TransformationPtr xform)
{
    // CudaGridRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    result = *(OSL::Matrix44*)xform;
    return true;
}

bool
CudaGridRenderer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                             ustring from)
{
    // CudaGridRenderer doesn't understand motion blur, so we never fail
    // on account of time-varying transformations.
    TransformMap::const_iterator found = _named_xforms.find(from);
    if (found != _named_xforms.end()) {
        result = *(found->second);
        return true;
    } else {
        return false;
    }
}

bool
CudaGridRenderer::get_inverse_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                                     ustring to, float /*time*/)
{
    if (to == u_camera || to == u_screen || to == u_NDC || to == u_raster) {
        Matrix44 M = _world_to_camera;
        if (to == u_screen || to == u_NDC || to == u_raster) {
            float depthrange = (double)_yon - (double)_hither;
            if (_projection == u_perspective) {
                float tanhalffov = tanf(0.5f * _fov * M_PI / 180.0);
                Matrix44 camera_to_screen(1 / tanhalffov, 0, 0, 0, 0,
                                          1 / tanhalffov, 0, 0, 0, 0,
                                          _yon / depthrange, 1, 0, 0,
                                          -_yon * _hither / depthrange, 0);
                M = M * camera_to_screen;
            } else {
                Matrix44 camera_to_screen(1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                          1 / depthrange, 0, 0, 0,
                                          -_hither / depthrange, 1);
                M = M * camera_to_screen;
            }
            if (to == u_NDC || to == u_raster) {
                float screenleft = -1.0, screenwidth = 2.0;
                float screenbottom = -1.0, screenheight = 2.0;
                Matrix44 screen_to_ndc(1 / screenwidth, 0, 0, 0, 0,
                                       1 / screenheight, 0, 0, 0, 0, 1, 0,
                                       -screenleft / screenwidth,
                                       -screenbottom / screenheight, 0, 1);
                M = M * screen_to_ndc;
                if (to == u_raster) {
                    Matrix44 ndc_to_raster(_xres, 0, 0, 0, 0, _yres, 0, 0, 0, 0,
                                           1, 0, 0, 0, 0, 1);
                    M = M * ndc_to_raster;
                }
            }
        }
        result = M;
        return true;
    }

    TransformMap::const_iterator found = _named_xforms.find(to);
    if (found != _named_xforms.end()) {
        result = *(found->second);
        result.invert();
        return true;
    } else {
        return false;
    }
}

void
CudaGridRenderer::name_transform(const char* name, const OSL::Matrix44& xform)
{
    std::shared_ptr<Transformation> M(new OSL::Matrix44(xform));
    _named_xforms[ustring(name)] = M;
}
