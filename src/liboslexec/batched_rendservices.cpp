// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include <cstdio>
#include <string>

#include "oslexec_pvt.h"
#include <OSL/batched_rendererservices.h>

using namespace OSL;
using namespace OSL::pvt;

OSL_NAMESPACE_ENTER

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
        m_texturesys = TextureSystem::create(true /* shared */);
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
                                            Wide<const ustring> wfrom,
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
                                                    ustring to,
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
                                                    Wide<const ustring> wto,
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
Mask<WidthT>
BatchedRendererServices<WidthT>::get_texture_info(
    BatchedShaderGlobals* bsg, TexturePerthread* /*texture_thread_info*/,
    Wide<const ustring> wfilename, int subimage, ustring dataname,
    MaskedData wval)
{
    Mask success(false);

#define TEXTURE_INFO_FOR_TYPE(data_type)                                                                                                                \
    if (Masked<data_type>::is(wval)) {                                                                                                                  \
        Masked<data_type> out(wval);                                                                                                                    \
        wval.mask().foreach ([=, &success](ActiveLane l) -> void {                                                                                      \
            data_type data;                                                                                                                             \
            bool status = texturesys()->get_texture_info(wfilename[l],                                                                                  \
                                                         subimage, dataname,                                                                            \
                                                         wval.type(), &data);                                                                           \
            if (status) {                                                                                                                               \
                /* masked assignment */                                                                                                                 \
                out[l] = data;                                                                                                                          \
                success.set_on(l);                                                                                                                      \
            } else {                                                                                                                                    \
                std::string err = texturesys()->geterror();                                                                                             \
                if (err.size() && bsg) {                                                                                                                \
                    /* TODO:  enable in future pull request */                                                                                          \
                    /* bsg->uniform.context->template batched<WidthT>().errorf (Mask(Lane(l)), "[BatchRendererServices::get_texture_info] %s", err); */ \
                }                                                                                                                                       \
            }                                                                                                                                           \
        });                                                                                                                                             \
        return success;                                                                                                                                 \
    }


#define TEXTURE_INFO_FOR_ARRAY(data_type)                                                                                                               \
    if (Masked<data_type[]>::is(wval)) {                                                                                                                \
        Masked<data_type[]> out(wval);                                                                                                                  \
        wval.mask().foreach ([=, &success](ActiveLane l) -> void {                                                                                      \
            auto arrayData = out[l];                                                                                                                    \
            OSL_STACK_ARRAY(data_type, data, arrayData.length());                                                                                       \
            bool status = texturesys()->get_texture_info(wfilename[l],                                                                                  \
                                                         subimage, dataname,                                                                            \
                                                         wval.type(), data);                                                                            \
            if (status) {                                                                                                                               \
                success.set_on(l);                                                                                                                      \
                /* masked assignment */                                                                                                                 \
                for (int i = 0; i < arrayData.length(); ++i) {                                                                                          \
                    arrayData[i] = data[i];                                                                                                             \
                }                                                                                                                                       \
            } else {                                                                                                                                    \
                std::string err = texturesys()->geterror();                                                                                             \
                if (err.size() && bsg) {                                                                                                                \
                    /* TODO:  enable in future pull request */                                                                                          \
                    /* bsg->uniform.context->template batched<WidthT>().errorf (Mask(Lane(l)), "[BatchRendererServices::get_texture_info] %s", err); */ \
                }                                                                                                                                       \
            }                                                                                                                                           \
        });                                                                                                                                             \
        return success;                                                                                                                                 \
    }

    TEXTURE_INFO_FOR_TYPE(int);
    TEXTURE_INFO_FOR_ARRAY(int);
    TEXTURE_INFO_FOR_TYPE(float);
    TEXTURE_INFO_FOR_ARRAY(float);
    TEXTURE_INFO_FOR_TYPE(Vec2);
    TEXTURE_INFO_FOR_TYPE(Vec3);
    TEXTURE_INFO_FOR_TYPE(Color3);
    TEXTURE_INFO_FOR_TYPE(Matrix44);
    TEXTURE_INFO_FOR_TYPE(ustring);

    return success;
}

template<int WidthT>
bool
BatchedRendererServices<WidthT>::get_texture_info_uniform(
    BatchedShaderGlobals* bsg, TexturePerthread* /*texture_thread_info*/,
    ustring filename, TextureSystem::TextureHandle* texture_handle,
    int subimage, ustring dataname, RefData val)
{
    bool status;
    if (texture_handle)
        status = texturesys()->get_texture_info(texture_handle, NULL, subimage,
                                                dataname, val.type(),
                                                val.ptr());
    else
        status = texturesys()->get_texture_info(filename, subimage, dataname,
                                                val.type(), val.ptr());
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && bsg) {
            // TODO:  enable in future pull request
            // bsg->uniform().context->errorf ("[BatchRendererServices::get_texture_info_uniform] %s", err);
        }
    }
    return status;
}

template<int WidthT>
Mask<WidthT>
BatchedRendererServices<WidthT>::texture(
    ustring filename, TextureSystem::TextureHandle* texture_handle,
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
    ustring filename, TextureSystem::TextureHandle* texture_handle,
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
                                            Masked<int> wresult, ustring source,
                                            ustring name, MaskedData wval)
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
