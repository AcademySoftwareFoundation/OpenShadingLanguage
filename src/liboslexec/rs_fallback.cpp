// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/fmt_util.h>

#include <OSL/rendererservices.h>

#include <OSL/journal.h>

#include "oslexec_pvt.h"

// Fallback is to reroute calls back through the virtual function
// based RendererServices from ShaderGlobals.
// We are intentially hiding ShaderGlobals and RendererServices from
// user supplied render service free functions as they should not
// allowed access to them.
// However to implement the fallback, we allow such access only right here

// Intentially private to this file
namespace {
inline OSL::ShaderGlobals*
get_sg(OSL::OpaqueExecContextPtr cptr)
{
    return reinterpret_cast<OSL::ShaderGlobals*>(cptr);
}
}  // namespace


// Host only fallback implementation of free function renderer services that
// simply forward the calls to the existing virtual RendererServices methods

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_matrix_xform_time(OSL::OpaqueExecContextPtr exec_ctx,
                         OSL::Matrix44& result, OSL::TransformationPtr from,
                         float time)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_matrix(sg, result, from, time);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_inverse_matrix_xform_time(OSL::OpaqueExecContextPtr exec_ctx,
                                 OSL::Matrix44& result,
                                 OSL::TransformationPtr xform, float time)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_inverse_matrix(sg, result, xform, time);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_matrix_space_time(OSL::OpaqueExecContextPtr exec_ctx,
                         OSL::Matrix44& result, OSL::ustringhash from,
                         float time)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_matrix(sg, result, from, time);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_inverse_matrix_space_time(OSL::OpaqueExecContextPtr exec_ctx,
                                 OSL::Matrix44& result, OSL::ustringhash to,
                                 float time)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_inverse_matrix(sg, result, to, time);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_matrix_xform(OSL::OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                    OSL::TransformationPtr xform)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_matrix(sg, result, xform);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_inverse_matrix_xform(OSL::OpaqueExecContextPtr exec_ctx,
                            OSL::Matrix44& result, OSL::TransformationPtr xform)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_inverse_matrix(sg, result, xform);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_matrix_space(OSL::OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                    OSL::ustringhash from)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_matrix(sg, result, from);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_inverse_matrix_space(OSL::OpaqueExecContextPtr exec_ctx,
                            OSL::Matrix44& result, OSL::ustringhash to)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_inverse_matrix(sg, result, to);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_transform_points(OSL::OpaqueExecContextPtr exec_ctx, OSL::ustringhash from,
                    OSL::ustringhash to, float time, const OSL::Vec3* Pin,
                    OSL::Vec3* Pout, int npoints,
                    OSL::TypeDesc::VECSEMANTICS vectype)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->transform_points(sg, from, to, time, Pin, Pout,
                                          npoints, vectype);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_texture(OSL::OpaqueExecContextPtr exec_ctx, OSL::ustringhash filename,
           OSL::TextureSystem::TextureHandle* texture_handle,
           OSL::TextureSystem::Perthread* texture_thread_info,
           OSL::TextureOpt& options, float s, float t, float dsdx, float dtdx,
           float dsdy, float dtdy, int nchannels, float* result,
           float* dresultds, float* dresultdt, OSL::ustringhash* errormessage)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->texture(filename, texture_handle, texture_thread_info,
                                 options, sg, s, t, dsdx, dtdx, dsdy, dtdy,
                                 nchannels, result, dresultds, dresultdt,
                                 errormessage);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_texture3d(OSL::OpaqueExecContextPtr exec_ctx, OSL::ustringhash filename,
             OSL::TextureSystem::TextureHandle* texture_handle,
             OSL::TextureSystem::Perthread* texture_thread_info,
             OSL::TextureOpt& options, const OSL::Vec3& P,
             const OSL::Vec3& dPdx, const OSL::Vec3& dPdy,
             const OSL::Vec3& dPdz, int nchannels, float* result,
             float* dresultds, float* dresultdt, float* dresultdr,
             OSL::ustringhash* errormessage)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->texture3d(filename, texture_handle,
                                   texture_thread_info, options, sg, P, dPdx,
                                   dPdy, dPdz, nchannels, result, dresultds,
                                   dresultdt, dresultdr, errormessage);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_environment(OSL::OpaqueExecContextPtr exec_ctx, OSL::ustringhash filename,
               OSL::TextureSystem::TextureHandle* texture_handle,
               OSL::TextureSystem::Perthread* texture_thread_info,
               OSL::TextureOpt& options, const OSL::Vec3& R,
               const OSL::Vec3& dRdx, const OSL::Vec3& dRdy, int nchannels,
               float* result, float* dresultds, float* dresultdt,
               OSL::ustringhash* errormessage)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->environment(filename, texture_handle,
                                     texture_thread_info, options, sg, R, dRdx,
                                     dRdy, nchannels, result, dresultds,
                                     dresultdt, errormessage);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_texture_info(OSL::OpaqueExecContextPtr exec_ctx,
                    OSL::ustringhash filename,
                    OSL::TextureSystem::TextureHandle* texture_handle,
                    OSL::TextureSystem::Perthread* texture_thread_info,
                    int subimage, OSL::ustringhash dataname,
                    OSL::TypeDesc datatype, void* data,
                    OSL::ustringhash* errormessage)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_texture_info(filename, texture_handle,
                                          texture_thread_info, sg, subimage,
                                          dataname, datatype, data,
                                          errormessage);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_get_texture_info_st(OSL::OpaqueExecContextPtr exec_ctx,
                       OSL::ustringhash filename,
                       OSL::TextureSystem::TextureHandle* texture_handle,
                       float s, float t,
                       OSL::TextureSystem::Perthread* texture_thread_info,
                       int subimage, OSL::ustringhash dataname,
                       OSL::TypeDesc datatype, void* data,
                       OSL::ustringhash* errormessage)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_texture_info(filename, texture_handle, s, t,
                                          texture_thread_info, sg, subimage,
                                          dataname, datatype, data,
                                          errormessage);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE int
rs_pointcloud_search(OSL::OpaqueExecContextPtr exec_ctx,
                     OSL::ustringhash filename, const OSL::Vec3& center,
                     float radius, int max_points, bool sort, int* out_indices,
                     float* out_distances, int derivs_offset)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->pointcloud_search(sg, filename, center, radius,
                                           max_points, sort, out_indices,
                                           out_distances, derivs_offset);
#else
    return 0;
#endif
}

OSL_RSOP OSL_HOSTDEVICE int
rs_pointcloud_get(OSL::OpaqueExecContextPtr exec_ctx, OSL::ustringhash filename,
                  const int* indices, int count, OSL::ustringhash attr_name,
                  OSL::TypeDesc attr_type, void* out_data)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->pointcloud_get(sg, filename, indices, count, attr_name,
                                        attr_type, out_data);
#else
    return 0;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_pointcloud_write(OSL::OpaqueExecContextPtr exec_ctx,
                    OSL::ustringhash filename, const OSL::Vec3& pos,
                    int nattribs, const OSL::ustringhash* names,
                    const OSL::TypeDesc* types, const void** data)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->pointcloud_write(sg, filename, pos, nattribs, names,
                                          types, data);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_trace(OSL::OpaqueExecContextPtr exec_ctx, OSL::TraceOpt& options,
         const OSL::Vec3& P, const OSL::Vec3& dPdx, const OSL::Vec3& dPdy,
         const OSL::Vec3& R, const OSL::Vec3& dRdx, const OSL::Vec3& dRdy)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->trace(options, sg, P, dPdx, dPdy, R, dRdx, dRdy);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE bool
rs_trace_get(OSL::OpaqueExecContextPtr exec_ctx, OSL::ustringhash name,
             OSL::TypeDesc type, void* val, bool derivatives)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->renderer->getmessage(sg, OSL::Strings::trace, name, type, val,
                                    derivatives);
#else
    return false;
#endif
}

OSL_RSOP OSL_HOSTDEVICE void*
rs_allocate_closure(OSL::OpaqueExecContextPtr exec_ctx, size_t size,
                    size_t alignment)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    return sg->context->allocate_closure(size, alignment);
#else
    return nullptr;
#endif
}

OSL_RSOP OSL_HOSTDEVICE void
rs_errorfmt(OSL::OpaqueExecContextPtr exec_ctx,
            OSL::ustringhash fmt_specification, int32_t count,
            const OSL::EncodedType* argTypes, uint32_t argValuesSize,
            uint8_t* argValues)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    sg->renderer->errorfmt(sg, fmt_specification, count, argTypes,
                           argValuesSize, argValues);
#endif
}

OSL_RSOP OSL_HOSTDEVICE void
rs_warningfmt(OSL::OpaqueExecContextPtr exec_ctx,
              OSL::ustringhash fmt_specification, int32_t count,
              const OSL::EncodedType* argTypes, uint32_t argValuesSize,
              uint8_t* argValues)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    sg->renderer->warningfmt(sg, fmt_specification, count, argTypes,
                             argValuesSize, argValues);
#endif
}

OSL_RSOP OSL_HOSTDEVICE void
rs_printfmt(OSL::OpaqueExecContextPtr exec_ctx,
            OSL::ustringhash fmt_specification, int32_t count,
            const OSL::EncodedType* argTypes, uint32_t argValuesSize,
            uint8_t* argValues)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    sg->renderer->printfmt(sg, fmt_specification, count, argTypes,
                           argValuesSize, argValues);
#endif
}

OSL_RSOP OSL_HOSTDEVICE void
rs_filefmt(OSL::OpaqueExecContextPtr exec_ctx, OSL::ustringhash filename,
           OSL::ustringhash fmt_specification, int32_t count,
           const OSL::EncodedType* argTypes, uint32_t argValuesSize,
           uint8_t* argValues)
{
#ifndef __CUDA_ARCH__
    auto sg = get_sg(exec_ctx);
    sg->renderer->filefmt(sg, filename, fmt_specification, count, argTypes,
                          argValuesSize, argValues);
#endif
}
