// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#if !defined(__CUDACC__) && !defined(OSL_HOST_RS_BITCODE)
#    error OSL_HOST_RS_BITCODE must be defined by your build system.
#endif

#include <OpenImageIO/fmath.h>

#ifndef __CUDACC__
#    include <OSL/fmt_util.h>
#    include <OSL/journal.h>
#endif
#include <OSL/rendererservices.h>
#include <OSL/rs_free_function.h>

#include "render_state.h"

// Keep free functions in sync with virtual function based SimpleRenderer.

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_matrix_xform_time(OSL::OpaqueExecContextPtr /*ec*/,
                         OSL::Matrix44& result, OSL::TransformationPtr xform,
                         float /*time*/)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    auto ptr = reinterpret_cast<OSL::Matrix44 const*>(xform);
    result   = *ptr;
    return true;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_inverse_matrix_xform_time(OSL::OpaqueExecContextPtr ec,
                                 OSL::Matrix44& result,
                                 OSL::TransformationPtr xform, float time)
{
    bool ok = rs_get_matrix_xform_time(ec, result, xform, time);
    if (ok) {
        result.invert();
    }
    return ok;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_matrix_space_time(OSL::OpaqueExecContextPtr /*ec*/,
                         OSL::Matrix44& result, OSL::ustringhash from,
                         float /*time*/)
{
    if (from == RS::Hashes::myspace) {
        OSL::Matrix44 Mmyspace;
        Mmyspace.scale(OSL::Vec3(1.0, 2.0, 1.0));
        result = Mmyspace;
        return true;
    } else {
        return false;
    }
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_inverse_matrix_space_time(OSL::OpaqueExecContextPtr ec,
                                 OSL::Matrix44& result, OSL::ustringhash to,
                                 float time)
{
    using OSL::Matrix44;


    auto rs = OSL::get_rs<RenderState>(ec);
    if (to == OSL::Hashes::camera || to == OSL::Hashes::screen
        || to == OSL::Hashes::NDC || to == RS::Hashes::raster) {
        Matrix44 M { rs->world_to_camera };

        if (to == OSL::Hashes::screen || to == OSL::Hashes::NDC
            || to == RS::Hashes::raster) {
            float depthrange      = (double)rs->yon - (double)rs->hither;
            OSL::ustringhash proj = rs->projection;

            if (proj == RS::Hashes::perspective) {
                float tanhalffov = OIIO::fast_tan(0.5f * rs->fov * M_PI
                                                  / 180.0);
                // clang-format off
                Matrix44 camera_to_screen (1/tanhalffov, 0, 0, 0,
                                           0, 1/tanhalffov, 0, 0,
                                           0, 0, rs->yon/depthrange, 1,
                                           0, 0, -(rs->yon*rs->hither)/depthrange, 0);
                // clang-format on
                M = M * camera_to_screen;
            } else {
                // clang-format off
                Matrix44 camera_to_screen (1, 0, 0, 0,
                                           0, 1, 0, 0,
                                           0, 0, 1/depthrange, 0,
                                           0, 0, -(rs->hither)/depthrange, 1);
                // clang-format on
                M = M * camera_to_screen;
            }
            if (to == OSL::Hashes::NDC || to == RS::Hashes::raster) {
                float screenleft = -1.0, screenwidth = 2.0;
                float screenbottom = -1.0, screenheight = 2.0;
                // clang-format off
                Matrix44 screen_to_ndc (1/screenwidth, 0, 0, 0,
                                        0, 1/screenheight, 0, 0,
                                        0, 0, 1, 0,
                                        -screenleft/screenwidth, -screenbottom/screenheight, 0, 1);
                // clang-format on
                M = M * screen_to_ndc;
                if (to == RS::Hashes::raster) {
                    // clang-format off
                    Matrix44 ndc_to_raster (rs->xres, 0, 0, 0,
                                            0, rs->yres, 0, 0,
                                            0, 0, 1, 0,
                                            0, 0, 0, 1);
                    M = M * ndc_to_raster;
                    // clang-format on
                }
            }
        }

        result = M;
        return true;
    } else {
        bool ok = rs_get_matrix_space_time(ec, result, to, time);
        if (ok) {
            result.invert();
        }

        return ok;
    }
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_matrix_xform(OSL::OpaqueExecContextPtr /*ec*/, OSL::Matrix44& result,
                    OSL::TransformationPtr xform)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    auto ptr = reinterpret_cast<OSL::Matrix44 const*>(xform);
    result   = *ptr;
    return true;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_inverse_matrix_xform(OSL::OpaqueExecContextPtr ec, OSL::Matrix44& result,
                            OSL::TransformationPtr xform)
{
    bool ok = rs_get_matrix_xform(ec, result, xform);
    if (ok) {
        result.invert();
    }

    return ok;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_matrix_space(OSL::OpaqueExecContextPtr /*ec*/, OSL::Matrix44& /*result*/,
                    OSL::ustringhash from)
{
    if (from == RS::Hashes::myspace) {
        return true;
    } else {
        return false;
    }
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_inverse_matrix_space(OSL::OpaqueExecContextPtr ec, OSL::Matrix44& result,
                            OSL::ustringhash to)
{
    bool ok = rs_get_matrix_space(ec, result, to);
    if (ok) {
        result.invert();
    }
    return ok;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_transform_points(OSL::OpaqueExecContextPtr /*ec*/, OSL::ustringhash /*from*/,
                    OSL::ustringhash /*to*/, float /*time*/,
                    const OSL::Vec3* /*Pin*/, OSL::Vec3* /*Pout*/,
                    int /*npoints*/, OSL::TypeDesc::VECSEMANTICS /*vectype*/)
{
    return false;
}

#ifdef __CUDACC__
// This texture lookup function needs to be compiled by NVCC because clang
// doesn't know how to handle CUDA texture intrinsics. This function must be
// defined in the CUDA source for testshade and testrender.
extern "C" __device__ float4
osl_tex2DLookup(void* handle, float s, float t, float dsdx, float dtdx,
                float dsdy, float dtdy);
#endif

OSL_RSOP OSL_HOSTDEVICE bool
rs_texture(OSL::OpaqueExecContextPtr ec, OSL::ustringhash filename,
           OSL::TextureSystem::TextureHandle* texture_handle,
           OSL::TextureSystem::Perthread* texture_thread_info,
           OSL::TextureOpt& options, float s, float t, float dsdx, float dtdx,
           float dsdy, float dtdy, int nchannels, float* result,
           float* dresultds, float* dresultdt, OSL::ustringhash* errormessage)
{
#ifndef __CUDA_ARCH__
    auto sg = (OSL::ShaderGlobals*)ec;
    return sg->renderer->texture(filename, texture_handle, texture_thread_info,
                                 options, sg, s, t, dsdx, dtdx, dsdy, dtdy,
                                 nchannels, result, dresultds, dresultdt,
                                 errormessage);
#else
    if (!texture_handle)
        return false;
    const float4 fromTexture = osl_tex2DLookup(texture_handle, s, t, dsdx, dtdx,
                                               dsdy, dtdy);
    for (int c = 0; c < nchannels; ++c)
        result[c] = ((const float*)&fromTexture)[c];
    return true;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_texture3d(OSL::OpaqueExecContextPtr ec, OSL::ustringhash filename,
             OSL::TextureSystem::TextureHandle* texture_handle,
             OSL::TextureSystem::Perthread* texture_thread_info,
             OSL::TextureOpt& options, const OSL::Vec3& P,
             const OSL::Vec3& dPdx, const OSL::Vec3& dPdy,
             const OSL::Vec3& dPdz, int nchannels, float* result,
             float* dresultds, float* dresultdt, float* dresultdr,
             OSL::ustringhash* errormessage)
{
#ifndef __CUDA_ARCH__
    auto sg = (OSL::ShaderGlobals*)ec;
    return sg->renderer->texture3d(filename, texture_handle,
                                   texture_thread_info, options, sg, P, dPdx,
                                   dPdy, dPdz, nchannels, result, dresultds,
                                   dresultdt, dresultdr, errormessage);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_environment(OSL::OpaqueExecContextPtr ec, OSL::ustringhash filename,
               OSL::TextureSystem::TextureHandle* texture_handle,
               OSL::TextureSystem::Perthread* texture_thread_info,
               OSL::TextureOpt& options, const OSL::Vec3& R,
               const OSL::Vec3& dRdx, const OSL::Vec3& dRdy, int nchannels,
               float* result, float* dresultds, float* dresultdt,
               OSL::ustringhash* errormessage)
{
#ifndef __CUDA_ARCH__
    auto sg = (OSL::ShaderGlobals*)ec;
    return sg->renderer->environment(filename, texture_handle,
                                     texture_thread_info, options, sg, R, dRdx,
                                     dRdy, nchannels, result, dresultds,
                                     dresultdt, errormessage);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_texture_info(OSL::OpaqueExecContextPtr ec, OSL::ustringhash filename,
                    OSL::TextureSystem::TextureHandle* texture_handle,
                    OSL::TextureSystem::Perthread* texture_thread_info,
                    int subimage, OSL::ustringhash dataname,
                    OSL::TypeDesc datatype, void* data,
                    OSL::ustringhash* errormessage)
{
#ifndef __CUDA_ARCH__
    auto sg = (OSL::ShaderGlobals*)ec;
    return sg->renderer->get_texture_info(filename, texture_handle,
                                          texture_thread_info, sg, subimage,
                                          dataname, datatype, data,
                                          errormessage);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_texture_info_st(OSL::OpaqueExecContextPtr ec, OSL::ustringhash filename,
                       OSL::TextureSystem::TextureHandle* texture_handle,
                       float s, float t,
                       OSL::TextureSystem::Perthread* texture_thread_info,
                       int subimage, OSL::ustringhash dataname,
                       OSL::TypeDesc datatype, void* data,
                       OSL::ustringhash* errormessage)
{
#ifndef __CUDA_ARCH__
    auto sg = (OSL::ShaderGlobals*)ec;
    return sg->renderer->get_texture_info(filename, texture_handle, s, t,
                                          texture_thread_info, sg, subimage,
                                          dataname, datatype, data,
                                          errormessage);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE int
rs_pointcloud_search(OSL::OpaqueExecContextPtr ec, OSL::ustringhash filename,
                     const OSL::Vec3& center, float radius, int max_points,
                     bool sort, int* out_indices, float* out_distances,
                     int derivs_offset)
{
#ifndef __CUDA_ARCH__
    auto sg = (OSL::ShaderGlobals*)ec;
    return sg->renderer->pointcloud_search(sg, filename, center, radius,
                                           max_points, sort, out_indices,
                                           out_distances, derivs_offset);
#else
    return 0;
#endif
}

OSL_RSOP OSL_HOSTDEVICE int
rs_pointcloud_get(OSL::OpaqueExecContextPtr ec, OSL::ustringhash filename,
                  const int* indices, int count, OSL::ustringhash attr_name,
                  OSL::TypeDesc attr_type, void* out_data)
{
#ifndef __CUDA_ARCH__
    auto sg = (OSL::ShaderGlobals*)ec;
    return sg->renderer->pointcloud_get(sg, filename, indices, count, attr_name,
                                        attr_type, out_data);
#else
    return 0;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_pointcloud_write(OSL::OpaqueExecContextPtr ec, OSL::ustringhash filename,
                    const OSL::Vec3& pos, int nattribs,
                    const OSL::ustringhash* names, const OSL::TypeDesc* types,
                    const void** data)
{
#ifndef __CUDA_ARCH__
    auto sg = (OSL::ShaderGlobals*)ec;
    return sg->renderer->pointcloud_write(sg, filename, pos, nattribs, names,
                                          types, data);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_trace(OSL::OpaqueExecContextPtr ec, OSL::TraceOpt& options,
         const OSL::Vec3& P, const OSL::Vec3& dPdx, const OSL::Vec3& dPdy,
         const OSL::Vec3& R, const OSL::Vec3& dRdx, const OSL::Vec3& dRdy)
{
#ifndef __CUDA_ARCH__
    auto sg = (OSL::ShaderGlobals*)ec;
    return sg->renderer->trace(options, sg, P, dPdx, dPdy, R, dRdx, dRdy);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_trace_get(OSL::OpaqueExecContextPtr ec, OSL::ustringhash name,
             OSL::TypeDesc type, void* val, bool derivatives)
{
#ifndef __CUDA_ARCH__
    auto sg = (OSL::ShaderGlobals*)ec;
    return sg->renderer->getmessage(sg, OSL::Strings::trace, name, type, val,
                                    derivatives);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_attribute_constant_string(OSL::ustringhash value, void* result)
{
    reinterpret_cast<OSL::ustringhash*>(result)[0] = value;
    return true;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_attribute_constant_int(int value, void* result)
{
    reinterpret_cast<int*>(result)[0] = value;
    return true;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_attribute_constant_int2(int value1, int value2, void* result)
{
    reinterpret_cast<int*>(result)[0] = value1;
    reinterpret_cast<int*>(result)[1] = value2;
    return true;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_attribute_constant_int3(int value1, int value2, int value3, void* result)
{
    reinterpret_cast<int*>(result)[0] = value1;
    reinterpret_cast<int*>(result)[1] = value2;
    reinterpret_cast<int*>(result)[2] = value3;
    return true;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_attribute_constant_int4(int value1, int value2, int value3, int value4,
                               void* result)
{
    reinterpret_cast<int*>(result)[0] = value1;
    reinterpret_cast<int*>(result)[1] = value2;
    reinterpret_cast<int*>(result)[2] = value3;
    reinterpret_cast<int*>(result)[3] = value4;
    return true;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_attribute_constant_float(float value, bool derivatives, void* result)
{
    reinterpret_cast<float*>(result)[0] = value;
    if (derivatives) {
        reinterpret_cast<float*>(result)[1] = 0.f;
        reinterpret_cast<float*>(result)[2] = 0.f;
    }
    return true;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_attribute_constant_float2(float value1, float value2, bool derivatives,
                                 void* result)
{
    reinterpret_cast<float*>(result)[0] = value1;
    reinterpret_cast<float*>(result)[1] = value2;
    if (derivatives) {
        reinterpret_cast<float*>(result)[2] = 0.f;
        reinterpret_cast<float*>(result)[3] = 0.f;
        reinterpret_cast<float*>(result)[4] = 0.f;
        reinterpret_cast<float*>(result)[5] = 0.f;
    }
    return true;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_attribute_constant_float3(float value1, float value2, float value3,
                                 bool derivatives, void* result)
{
    reinterpret_cast<float*>(result)[0] = value1;
    reinterpret_cast<float*>(result)[1] = value2;
    reinterpret_cast<float*>(result)[2] = value3;
    if (derivatives) {
        reinterpret_cast<float*>(result)[3] = 0.f;
        reinterpret_cast<float*>(result)[4] = 0.f;
        reinterpret_cast<float*>(result)[5] = 0.f;
        reinterpret_cast<float*>(result)[6] = 0.f;
        reinterpret_cast<float*>(result)[7] = 0.f;
        reinterpret_cast<float*>(result)[8] = 0.f;
    }
    return true;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_attribute_constant_float4(float value1, float value2, float value3,
                                 float value4, bool derivatives, void* result)
{
    reinterpret_cast<float*>(result)[0] = value1;
    reinterpret_cast<float*>(result)[1] = value2;
    reinterpret_cast<float*>(result)[2] = value3;
    reinterpret_cast<float*>(result)[3] = value4;
    if (derivatives) {
        reinterpret_cast<float*>(result)[4]  = 0.f;
        reinterpret_cast<float*>(result)[5]  = 0.f;
        reinterpret_cast<float*>(result)[6]  = 0.f;
        reinterpret_cast<float*>(result)[7]  = 0.f;
        reinterpret_cast<float*>(result)[8]  = 0.f;
        reinterpret_cast<float*>(result)[9]  = 0.f;
        reinterpret_cast<float*>(result)[10] = 0.f;
        reinterpret_cast<float*>(result)[11] = 0.f;
    }
    return true;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_shade_index(OSL::OpaqueExecContextPtr oec, void* result)

{
    reinterpret_cast<int*>(result)[0] = OSL::get_shade_index(oec);
    return true;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_attribute(OSL::OpaqueExecContextPtr oec, OSL::ustringhash_pod object_,
                 OSL::ustringhash_pod name_, OSL::TypeDesc_pod _type,
                 bool derivatives, int index, void* result)
{
    auto object              = OSL::ustringhash_from(object_);
    auto name                = OSL::ustringhash_from(name_);
    const OSL::TypeDesc type = OSL::TypeDesc_from(_type);
    auto rs                  = OSL::get_rs<RenderState>(oec);

    // The many branches in the code below handle the case where we don't know
    // the attribute name at compile time. In the case it is known, dead-code
    // elimination should optimize this to only the relevant branch.
    if (name == RS::Hashes::osl_version && type == OSL::TypeInt)
        return rs_get_attribute_constant_int(OSL_VERSION, result);
    if (name == RS::Hashes::camera_resolution
        && type == OSL::TypeDesc(OSL::TypeDesc::INT, 2))
        return rs_get_attribute_constant_int2(rs->xres, rs->yres, result);
    if (name == RS::Hashes::camera_projection && type == OSL::TypeString)
        return rs_get_attribute_constant_string(rs->projection, result);
    if (name == RS::Hashes::camera_pixelaspect && type == OSL::TypeFloat)
        return rs_get_attribute_constant_float(rs->pixelaspect, derivatives,
                                               result);
    if (name == RS::Hashes::camera_screen_window
        && type == OSL::TypeDesc(OSL::TypeDesc::FLOAT, 4))
        return rs_get_attribute_constant_float4(rs->screen_window[0],
                                                rs->screen_window[1],
                                                rs->screen_window[2],
                                                rs->screen_window[3],
                                                derivatives, result);
    if (name == RS::Hashes::camera_fov && type == OSL::TypeFloat)
        return rs_get_attribute_constant_float(rs->fov, derivatives, result);
    if (name == RS::Hashes::camera_clip
        && type == OSL::TypeDesc(OSL::TypeDesc::FLOAT, 2))
        return rs_get_attribute_constant_float2(rs->hither, rs->yon,
                                                derivatives, result);
    if (name == RS::Hashes::camera_clip_near && type == OSL::TypeFloat)
        return rs_get_attribute_constant_float(rs->hither, derivatives, result);
    if (name == RS::Hashes::camera_clip_far && type == OSL::TypeFloat)
        return rs_get_attribute_constant_float(rs->yon, derivatives, result);
    if (name == RS::Hashes::camera_shutter
        && type == OSL::TypeDesc(OSL::TypeDesc::FLOAT, 2))
        return rs_get_attribute_constant_float2(rs->shutter[0], rs->shutter[1],
                                                derivatives, result);
    if (name == RS::Hashes::camera_shutter_open && type == OSL::TypeFloat)
        return rs_get_attribute_constant_float(rs->shutter[0], derivatives,
                                               result);
    if (name == RS::Hashes::camera_shutter_close && type == OSL::TypeFloat)
        return rs_get_attribute_constant_float(rs->shutter[1], derivatives,
                                               result);

    if (name == RS::Hashes::shading_index && type == OSL::TypeInt)
        return rs_get_shade_index(oec, result);

    if (object == RS::Hashes::options && name == RS::Hashes::blahblah
        && type == OSL::TypeFloat)
        return rs_get_attribute_constant_float(3.14159f, derivatives, result);

#ifndef __CUDA_ARCH__
    if (object.empty()) {
        // TODO: implement a true free function version of rs_get_userdata
        // that only uses data from RenderState.  Because that isn't done
        // yet, for host only compilation we break encapsulation and cast
        // to gain access to the virtual RendererServices.
        auto sg = reinterpret_cast<OSL::ShaderGlobals*>(oec);
        return sg->renderer->get_userdata(derivatives, name, type, sg, result);
    }
#endif

    return false;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_interpolated_face_idx(OSL::OpaqueExecContextPtr ec, void* val)
{
    ((int*)val)[0] = int(4 * OSL::get_u(ec));
    return true;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_interpolated_s(OSL::OpaqueExecContextPtr ec, bool derivatives, void* val)
{
    ((float*)val)[0] = OSL::get_u(ec);
    if (derivatives) {
        ((float*)val)[1] = OSL::get_dudx(ec);
        ((float*)val)[2] = OSL::get_dudy(ec);
    }
    return true;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_interpolated_t(OSL::OpaqueExecContextPtr ec, bool derivatives, void* val)
{
    ((float*)val)[0] = OSL::get_v(ec);
    if (derivatives) {
        ((float*)val)[1] = OSL::get_dvdx(ec);
        ((float*)val)[2] = OSL::get_dvdy(ec);
    }
    return true;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_interpolated_red(OSL::OpaqueExecContextPtr ec, bool derivatives,
                        void* val)
{
    if (OSL::get_P(ec).x > 0.5f) {
        ((float*)val)[0] = OSL::get_u(ec);
        if (derivatives) {
            ((float*)val)[1] = OSL::get_dudx(ec);
            ((float*)val)[2] = OSL::get_dudy(ec);
        }
        return true;
    }
    return false;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_interpolated_green(OSL::OpaqueExecContextPtr ec, bool derivatives,
                          void* val)
{
    if (OSL::get_P(ec).x < 0.5f) {
        ((float*)val)[0] = OSL::get_v(ec);
        if (derivatives) {
            ((float*)val)[1] = OSL::get_dvdx(ec);
            ((float*)val)[2] = OSL::get_dvdy(ec);
        }
        return true;
    }
    return false;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_interpolated_blue(OSL::OpaqueExecContextPtr ec, bool derivatives,
                         void* val)
{
    if ((static_cast<int>(OSL::get_P(ec).y * 12) % 2) == 0) {
        ((float*)val)[0] = 1.0f - OSL::get_u(ec);
        if (derivatives) {
            ((float*)val)[1] = -OSL::get_dudx(ec);
            ((float*)val)[2] = -OSL::get_dudy(ec);
        }
        return true;
    }
    return false;
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_interpolated_test(void* val)
{
    printf("rs_get_interpolated_test\n");
    ((float*)val)[0] = 0.42f;
    return true;
}

#ifndef __CUDACC__
OSL_RSOP OSL_HOSTDEVICE void
rs_errorfmt(OSL::OpaqueExecContextPtr ec, OSL::ustringhash fmt_specification,
            int32_t arg_count, const OSL::EncodedType* argTypes,
            uint32_t argValuesSize, uint8_t* argValues)
{
    auto rs = OSL::get_rs<RenderState>(ec);

    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_errorfmt(OSL::get_thread_index(ec), OSL::get_shade_index(ec),
                       fmt_specification, arg_count, argTypes, argValuesSize,
                       argValues);
}

OSL_RSOP OSL_HOSTDEVICE void
rs_warningfmt(OSL::OpaqueExecContextPtr ec, OSL::ustringhash fmt_specification,
              int32_t arg_count, const OSL::EncodedType* argTypes,
              uint32_t argValuesSize, uint8_t* argValues)
{
    auto rs = OSL::get_rs<RenderState>(ec);

    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_warningfmt(OSL::get_max_warnings_per_thread(ec),
                         OSL::get_thread_index(ec), OSL::get_shade_index(ec),
                         fmt_specification, arg_count, argTypes, argValuesSize,
                         argValues);
}


OSL_RSOP OSL_HOSTDEVICE void
rs_printfmt(OSL::OpaqueExecContextPtr ec, OSL::ustringhash fmt_specification,
            int32_t arg_count, const OSL::EncodedType* argTypes,
            uint32_t argValuesSize, uint8_t* argValues)
{
    auto rs = OSL::get_rs<RenderState>(ec);

    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_printfmt(OSL::get_thread_index(ec), OSL::get_shade_index(ec),
                       fmt_specification, arg_count, argTypes, argValuesSize,
                       argValues);
}


OSL_RSOP OSL_HOSTDEVICE void
rs_filefmt(OSL::OpaqueExecContextPtr ec, OSL::ustringhash filename_hash,
           OSL::ustringhash fmt_specification, int32_t arg_count,
           const OSL::EncodedType* argTypes, uint32_t argValuesSize,
           uint8_t* argValues)
{
    auto rs = OSL::get_rs<RenderState>(ec);

    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_filefmt(OSL::get_thread_index(ec), OSL::get_shade_index(ec),
                      filename_hash, fmt_specification, arg_count, argTypes,
                      argValuesSize, argValues);
}
#endif  // #ifndef __CUDACC__
