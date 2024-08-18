// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <cstdio>
#include <string>

#include "oslexec_pvt.h"
#include <OSL/batched_rendererservices.h>

using namespace OSL;
using namespace OSL::pvt;

OSL_NAMESPACE_ENTER


#ifdef OIIO_TEXTURESYSTEM_CREATE_SHARED
namespace {
std::mutex shared_texturesys_mutex;
std::shared_ptr<TextureSystem> shared_texturesys;
}  // namespace
#endif



template<int WidthT>
BatchedRendererServices<WidthT>::BatchedRendererServices(TextureSystem* texsys)
    : m_texturesys(texsys)
{
    if (!m_texturesys) {
#if OSL_NO_DEFAULT_TEXTURESYSTEM
        // This build option instructs OSL to never create a TextureSystem
        // itself. (Most likely reason: this build of OSL is for a renderer
        // that replaces OIIO's TextureSystem with its own, and therefore
        // wouldn't want to accidentally make an OIIO one here.
        OSL_ASSERT(
            0 && "RendererServices was not passed a working TextureSystem*");
#else
#    ifdef OIIO_TEXTURESYSTEM_CREATE_SHARED
        {
            std::lock_guard<std::mutex> lock(shared_texturesys_mutex);
            if (!shared_texturesys) {
                shared_texturesys = TextureSystem::create(true /* shared */);
            }
            m_texturesys = shared_texturesys.get();
        }
#    else
        m_texturesys = TextureSystem::create(true /* shared */);
#    endif
        // Make some good guesses about default options
        m_texturesys->attribute("automip", 1);
        m_texturesys->attribute("autotile", 64);
#endif
    }
}



template<int WidthT>
Mask<WidthT>
BatchedRendererServices<WidthT>::get_matrix(BatchedShaderGlobals* bsg,
                                            Masked<Matrix44> wresult,
                                            Wide<const ustringhash> wfrom,
                                            Wide<const float> wtime)
{
    OSL_ASSERT(
        0
        && "UNREACHABLE:  BatchedRendererServices<WidthT>::get_matrix calls should be overridden or the target specific version in wide_opmatrix.cpp should be called");
    return Mask(false);
}



template<int WidthT>
Mask<WidthT>
BatchedRendererServices<WidthT>::get_inverse_matrix(
    BatchedShaderGlobals* bsg, Masked<Matrix44> wresult,
    Wide<const TransformationPtr> wxform, Wide<const float> wtime)
{
    OSL_ASSERT(
        0
        && "UNREACHABLE:  BatchedRendererServices<WidthT>::get_inverse_matrix calls should be overridden or the target specific version in wide_opmatrix.cpp should be called");
    return Mask(false);
}



template<int WidthT>
Mask<WidthT>
BatchedRendererServices<WidthT>::get_inverse_matrix(BatchedShaderGlobals* bsg,
                                                    Masked<Matrix44> wresult,
                                                    ustringhash to,
                                                    Wide<const float> wtime)
{
    OSL_ASSERT(
        0
        && "UNREACHABLE:  BatchedRendererServices<WidthT>::get_inverse_matrix calls should be overridden or the target specific version in wide_opmatrix.cpp should be called");
    return Mask(false);
}



template<int WidthT>
Mask<WidthT>
BatchedRendererServices<WidthT>::get_inverse_matrix(BatchedShaderGlobals* bsg,
                                                    Masked<Matrix44> wresult,
                                                    Wide<const ustringhash> wto,
                                                    Wide<const float> wtime)
{
    OSL_ASSERT(
        0
        && "UNREACHABLE:  BatchedRendererServices<WidthT>::get_inverse_matrix calls should be overridden or the target specific version in wide_opmatrix.cpp should be called");
    return Mask(false);
}



template<int WidthT>
TextureSystem*
BatchedRendererServices<WidthT>::texturesys() const
{
    return m_texturesys;
}



template<int WidthT>
TextureSystem::TextureHandle*
BatchedRendererServices<WidthT>::resolve_udim_uniform(
    BatchedShaderGlobals* bsg, TexturePerthread* texture_thread_info,
    ustringhash filename, TextureSystem::TextureHandle* texture_handle, float S,
    float T)
{
    if (!texture_thread_info)
        texture_thread_info = bsg->uniform.context->texture_thread_info();
    if (!texture_handle)
        texture_handle = texturesys()->get_texture_handle(ustring_from(filename),
                                                          texture_thread_info);
    if (texturesys()->is_udim(texture_handle)) {
        // Newer versions of the TextureSystem interface are able to determine the
        // specific UDIM tile we're using.
        TextureSystem::TextureHandle* udim_handle
            = texturesys()->resolve_udim(texture_handle, texture_thread_info, S,
                                         T);
        // NOTE:  udim_handle may be nullptr if no corresponding texture exists
        if (udim_handle == nullptr) {
            // Optimization to just reuse the <udim> texture handle vs.
            // forcing get_texture_info_uniform to redo the lookup we have already done.
            udim_handle = texture_handle;
        }
        return udim_handle;
    } else {
        return texture_handle;
    }
}



template<int WidthT>
void
BatchedRendererServices<WidthT>::resolve_udim(
    BatchedShaderGlobals* bsg, TexturePerthread* texture_thread_info,
    ustringhash filename, TextureSystem::TextureHandle* texture_handle,
    Wide<const float> wS, Wide<const float> wT,
    Masked<TextureSystem::TextureHandle*> wresult)
{
    if (!texture_thread_info)
        texture_thread_info = bsg->uniform.context->texture_thread_info();
    if (!texture_handle)
        texture_handle = texturesys()->get_texture_handle(ustring_from(filename),
                                                          texture_thread_info);
    if (texturesys()->is_udim(texture_handle)) {
        // Newer versions of the TextureSystem interface are able to determine the
        // specific UDIM tile we're using.
        wresult.mask().foreach ([&](ActiveLane l) -> void {
            TextureSystem::TextureHandle* udim_handle
                = texturesys()->resolve_udim(texture_handle,
                                             texture_thread_info, wS[l], wT[l]);
            // NOTE:  udim_handle may be nullptr if no corresponding texture exists
            if (udim_handle == nullptr) {
                // Optimization to just reuse the <udim> texture handle vs.
                // forcing get_texture_info_uniform to redo the lookup we have already done.
                udim_handle = texture_handle;
            }
            wresult[l] = udim_handle;
        });
    } else {
        assign_all(wresult, texture_handle);
    }
}



template<int WidthT>
bool
BatchedRendererServices<WidthT>::get_texture_info_uniform(
    BatchedShaderGlobals* bsg, TexturePerthread* texture_thread_info,
    ustringhash filename, TextureSystem::TextureHandle* texture_handle,
    int subimage, ustringhash dataname, RefData val)
{
    if (!texture_thread_info)
        texture_thread_info = bsg->uniform.context->texture_thread_info();
    if (!texture_handle)
        texture_handle = texturesys()->get_texture_handle(ustring_from(filename),
                                                          texture_thread_info);
    bool status = texturesys()->get_texture_info(texture_handle, NULL, subimage,
                                                 ustring_from(dataname),
                                                 val.type(), val.ptr());

    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && bsg) {
            bsg->uniform.context->errorfmt(
                "[RendererServices::get_texture_info] {}", err);
        }
    }
    return status;
}



template<int WidthT>
Mask<WidthT>
BatchedRendererServices<WidthT>::texture(
    ustringhash filename, TextureSystem::TextureHandle* texture_handle,
    TextureSystem::Perthread* texture_thread_info,
    const BatchedTextureOptions& options, BatchedShaderGlobals* bsg,
    Wide<const float> ws, Wide<const float> wt, Wide<const float> wdsdx,
    Wide<const float> wdtdx, Wide<const float> wdsdy, Wide<const float> wdtdy,
    BatchedTextureOutputs& outputs)
{
    OSL_ASSERT(
        0
        && "UNREACHABLE:  BatchedRendererServices<WidthT>::texture calls should be overridden or the target specific version in wide_optexture.cpp should be called");
    return Mask(false);
}



template<int WidthT>
Mask<WidthT>
BatchedRendererServices<WidthT>::texture3d(
    ustringhash filename, TextureSystem::TextureHandle* texture_handle,
    TextureSystem::Perthread* texture_thread_info,
    const BatchedTextureOptions& options, BatchedShaderGlobals* bsg,
    Wide<const Vec3> wP, Wide<const Vec3> wdPdx, Wide<const Vec3> wdPdy,
    Wide<const Vec3> wdPdz, BatchedTextureOutputs& outputs)
{
    OSL_ASSERT(
        0
        && "UNREACHABLE:  BatchedRendererServices<WidthT>::texture calls should be overridden or the target specific version in wide_optexture.cpp should be called");
    return Mask(false);
}



template<int WidthT>
Mask<WidthT>
BatchedRendererServices<WidthT>::environment(
    ustringhash filename, TextureSystem::TextureHandle* texture_handle,
    TextureSystem::Perthread* texture_thread_info,
    const BatchedTextureOptions& options, BatchedShaderGlobals* bsg,
    Wide<const Vec3> wR, Wide<const Vec3> wdRdx, Wide<const Vec3> wdRdy,
    BatchedTextureOutputs& outputs)
{
    OSL_ASSERT(
        0
        && "UNREACHABLE:  BatchedRendererServices<WidthT>::environment calls should be overridden or the target specific version in wide_optexture.cpp should be called");
    return Mask(false);
}



template<int WidthT>
void
BatchedRendererServices<WidthT>::pointcloud_search(
    BatchedShaderGlobals* bsg, ustringhash filename, const void* wcenter,
    Wide<const float> wradius, int max_points, bool sort,
    PointCloudSearchResults& results)
{
    OSL_ASSERT(
        0
        && "UNREACHABLE:  BatchedRendererServices<WidthT>::pointcloud_search calls should be overridden or the target specific version in wide_oppointcloud.cpp should be called");
}



template<int WidthT>
Mask<WidthT>
BatchedRendererServices<WidthT>::pointcloud_get(
    BatchedShaderGlobals* bsg, ustringhash filename, Wide<const int[]> windices,
    Wide<const int> wnum_points, ustringhash attr_name, MaskedData wout_data)
{
    OSL_ASSERT(
        0
        && "UNREACHABLE:  BatchedRendererServices<WidthT>::pointcloud_get calls should be overridden or the target specific version in wide_oppointcloud.cpp should be called");
    return Mask(false);
}



template<int WidthT>
Mask<WidthT>
BatchedRendererServices<WidthT>::pointcloud_write(
    BatchedShaderGlobals* bsg, ustringhash filename, Wide<const OSL::Vec3> wpos,
    int nattribs, const ustring* attr_names, const TypeDesc* attr_types,
    const void** pointers_to_wide_attr_value, Mask mask)
{
    OSL_ASSERT(
        0
        && "UNREACHABLE:  BatchedRendererServices<WidthT>::pointcloud_write calls should be overridden or the target specific version in wide_oppointcloud.cpp should be called");
    return Mask(false);
}



template<int WidthT>
void
BatchedRendererServices<WidthT>::trace(
    TraceOpt& options, BatchedShaderGlobals* bsg, Masked<int> wresult,
    Wide<const Vec3> wP, Wide<const Vec3> wdPdx, Wide<const Vec3> wdPdy,
    Wide<const Vec3> wR, Wide<const Vec3> wdRdx, Wide<const Vec3> wdRdy)
{
    for (int lane = 0; lane < WidthT; ++lane) {
        wresult[lane] = 0;
    }
}



template<int WidthT>
void
BatchedRendererServices<WidthT>::getmessage(BatchedShaderGlobals* bsg,
                                            Masked<int> wresult,
                                            ustringhash source,
                                            ustringhash name, MaskedData wval)
{
    // Currently this code path should only be followed when source == "trace"
    OSL_DASSERT(wresult.mask() == wval.mask());
    for (int lane = 0; lane < WidthT; ++lane) {
        wresult[lane] = 0;
    }
}



// Explicitly instantiate BatchedRendererServices template
template class OSLEXECPUBLIC BatchedRendererServices<16>;
template class OSLEXECPUBLIC BatchedRendererServices<8>;

OSL_NAMESPACE_EXIT
