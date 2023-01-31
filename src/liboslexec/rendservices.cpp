// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <cstdio>
#include <string>
#include <vector>

#include "oslexec_pvt.h"
using namespace OSL;
using namespace OSL::pvt;

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/strutil.h>


OSL_NAMESPACE_ENTER



RendererServices::RendererServices(TextureSystem* texsys) : m_texturesys(texsys)
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



TextureSystem*
RendererServices::texturesys() const
{
    return m_texturesys;
}



bool
RendererServices::get_matrix(ShaderGlobals* sg, Matrix44& result,
                             ustringhash from, float time)
{
    return false;
}



bool
RendererServices::get_matrix(ShaderGlobals* sg, Matrix44& result,
                             ustringhash from)
{
    return false;
}



bool
RendererServices::get_inverse_matrix(ShaderGlobals* sg, Matrix44& result,
                                     TransformationPtr xform, float time)
{
    bool ok = get_matrix(sg, result, xform, time);
    if (ok)
        result.invert();
    return ok;
}



bool
RendererServices::get_inverse_matrix(ShaderGlobals* sg, Matrix44& result,
                                     TransformationPtr xform)
{
    bool ok = get_matrix(sg, result, xform);
    if (ok)
        result.invert();
    return ok;
}



bool
RendererServices::get_inverse_matrix(ShaderGlobals* sg, Matrix44& result,
                                     ustringhash to, float time)
{
    // return get_inverse_matrix(sg, result, ustring_from(to), time);
    bool ok = get_matrix(sg, result, to, time);
    if (ok)
        result.invert();
    return ok;
}



bool
RendererServices::get_inverse_matrix(ShaderGlobals* sg, Matrix44& result,
                                     ustringhash to)
{
    // return get_inverse_matrix(sg, result, ustring_from(to));
    bool ok = get_matrix(sg, result, to);
    if (ok)
        result.invert();
    return ok;
}



bool
RendererServices::transform_points(ShaderGlobals* sg, ustringhash from,
                                   ustringhash to, float time, const Vec3* Pin,
                                   Vec3* Pout, int npoints,
                                   TypeDesc::VECSEMANTICS vectype)
{
    return false;
}



bool
RendererServices::get_attribute(ShaderGlobals* sg, bool derivatives,
                                ustringhash object, TypeDesc type,
                                ustringhash name, void* val)
{
    return false;
}



bool
RendererServices::get_array_attribute(ShaderGlobals* sg, bool derivatives,
                                      ustringhash object, TypeDesc type,
                                      ustringhash name, int index, void* val)
{
    return false;
}



bool
RendererServices::get_userdata(bool derivatives, ustringhash name,
                               TypeDesc type, ShaderGlobals* sg, void* val)
{
    return false;
    // return get_userdata(derivatives, ustring_from(name), type, sg, val);
}



RendererServices::TextureHandle*
RendererServices::get_texture_handle(ustringhash filename,
                                     ShadingContext* context,
                                     const TextureOpt* options)
{
#ifdef OIIO_TEXTURESYSTEM_SUPPORTS_COLORSPACE
    return texturesys()->get_texture_handle(ustring_from(filename),
                                            context->texture_thread_info(),
                                            options);
#else
    return texturesys()->get_texture_handle(ustring_from(filename),
                                            context->texture_thread_info());
#endif
}



bool
RendererServices::good(TextureHandle* texture_handle)
{
    return texturesys()->good(texture_handle);
}


bool
RendererServices::is_udim(TextureHandle* texture_handle)
{
    return texturesys()->is_udim(texture_handle);
}



bool
RendererServices::texture(ustringhash filename, TextureHandle* texture_handle,
                          TexturePerthread* texture_thread_info,
                          TextureOpt& options, ShaderGlobals* sg, float s,
                          float t, float dsdx, float dtdx, float dsdy,
                          float dtdy, int nchannels, float* result,
                          float* dresultds, float* dresultdt,
                          ustringhash* errormessage)
{
    ShadingContext* context = sg->context;
    if (!texture_thread_info)
        texture_thread_info = context->texture_thread_info();
    if (!texture_handle)
#ifdef OIIO_TEXTURESYSTEM_SUPPORTS_COLORSPACE
        texture_handle
            = texturesys()->get_texture_handle(ustring_from(filename),
                                               texture_thread_info, &options);
#else
        texture_handle = texturesys()->get_texture_handle(ustring_from(filename),
                                                          texture_thread_info);
#endif
    bool status = texturesys()->texture(texture_handle, texture_thread_info,
                                        options, s, t, dsdx, dtdx, dsdy, dtdy,
                                        nchannels, result, dresultds,
                                        dresultdt);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sg) {
            if (errormessage) {
                *errormessage = ustringhash(err);
            } else {
                context->errorfmt("[RendererServices::texture] {}", err);
            }
        } else if (errormessage) {
            *errormessage = ustringhash(Strings::unknown);
        }
    }
    return status;
}



bool
RendererServices::texture3d(ustringhash filename, TextureHandle* texture_handle,
                            TexturePerthread* texture_thread_info,
                            TextureOpt& options, ShaderGlobals* sg,
                            const Vec3& P, const Vec3& dPdx, const Vec3& dPdy,
                            const Vec3& dPdz, int nchannels, float* result,
                            float* dresultds, float* dresultdt,
                            float* dresultdr, ustringhash* errormessage)
{
    ShadingContext* context = sg->context;
    if (!texture_thread_info)
        texture_thread_info = context->texture_thread_info();
    if (!texture_handle)
#ifdef OIIO_TEXTURESYSTEM_SUPPORTS_COLORSPACE
        texture_handle
            = texturesys()->get_texture_handle(ustring_from(filename),
                                               texture_thread_info, &options);
#else
        texture_handle = texturesys()->get_texture_handle(ustring_from(filename),
                                                          texture_thread_info);
#endif

    bool status = texturesys()->texture3d(texture_handle, texture_thread_info,
                                          options, P, dPdx, dPdy, dPdz,
                                          nchannels, result, dresultds,
                                          dresultdt, dresultdr);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sg) {
            if (errormessage) {
                *errormessage = ustringhash(err);
            } else {
                sg->context->errorfmt("[RendererServices::texture3d] {}", err);
            }
        } else if (errormessage) {
            *errormessage = Strings::unknown.uhash();
        }
    }
    return status;
}



bool
RendererServices::environment(ustringhash filename,
                              TextureHandle* texture_handle,
                              TexturePerthread* texture_thread_info,
                              TextureOpt& options, ShaderGlobals* sg,
                              const Vec3& R, const Vec3& dRdx, const Vec3& dRdy,
                              int nchannels, float* result, float* dresultds,
                              float* dresultdt, ustringhash* errormessage)
{
    ShadingContext* context = sg->context;
    if (!texture_thread_info)
        texture_thread_info = context->texture_thread_info();
    if (!texture_handle)
#ifdef OIIO_TEXTURESYSTEM_SUPPORTS_COLORSPACE
        texture_handle
            = texturesys()->get_texture_handle(ustring_from(filename),
                                               texture_thread_info, &options);
#else
        texture_handle = texturesys()->get_texture_handle(ustring_from(filename),
                                                          texture_thread_info);
#endif
    bool status = texturesys()->environment(texture_handle, texture_thread_info,
                                            options, R, dRdx, dRdy, nchannels,
                                            result, dresultds, dresultdt);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sg) {
            if (errormessage) {
                *errormessage = ustringhash(err);
            } else {
                sg->context->errorfmt("[RendererServices::environment] {}",
                                      err);
            }
        } else if (errormessage) {
            *errormessage = Strings::unknown.uhash();
        }
    }
    return status;
}



bool
RendererServices::get_texture_info(ustringhash filename,
                                   TextureHandle* texture_handle,
                                   TexturePerthread* texture_thread_info,
                                   ShadingContext* shading_context,
                                   int subimage, ustringhash dataname,
                                   TypeDesc datatype, void* data,
                                   ustringhash* errormessage)
{
    if (!texture_thread_info)
        texture_thread_info = shading_context->texture_thread_info();
    if (!texture_handle)
        texture_handle = texturesys()->get_texture_handle(ustring_from(filename),
                                                          texture_thread_info);
    bool status = texturesys()->get_texture_info(texture_handle,
                                                 texture_thread_info, subimage,
                                                 ustring_from(dataname),
                                                 datatype, data);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size()) {
            if (errormessage) {
                *errormessage = ustringhash(err);
            } else {
                shading_context->errorfmt(
                    "[RendererServices::get_texture_info] {}", err);
            }
        } else if (errormessage) {
            // gettextureinfo failed but did not provide an error, so none should be emitted
            *errormessage = ustringhash();
        }
    }
    return status;
}



bool
RendererServices::get_texture_info(
    ustringhash filename, TextureHandle* texture_handle, float s, float t,
    TexturePerthread* texture_thread_info, ShadingContext* shading_context,
    int subimage, ustringhash dataname, TypeDesc datatype, void* data,
    ustringhash* errormessage)
{
#if OIIO_VERSION >= 20307
    // Newer versions of the TextureSystem interface are able to determine the
    // specific UDIM tile we're using.
    if (!texture_thread_info)
        texture_thread_info = shading_context->texture_thread_info();
    if (!texture_handle)
        texture_handle = texturesys()->get_texture_handle(ustring_from(filename),
                                                          texture_thread_info);
    if (texturesys()->is_udim(texture_handle)) {
        TextureSystem::TextureHandle* udim_handle
            = texturesys()->resolve_udim(texture_handle, texture_thread_info, s,
                                         t);
        // NOTE:  udim_handle may be nullptr if no corresponding texture exists
        // Optimization to just reuse the <udim> texture handle vs.
        // forcing get_texture_info_uniform to redo the lookup we have already done.
        if (udim_handle != nullptr) {
            texture_handle = udim_handle;
        }
    }
#endif
    return get_texture_info(filename, texture_handle, texture_thread_info,
                            shading_context, subimage, dataname, datatype, data,
                            errormessage);
}



bool
RendererServices::trace(TraceOpt& options, ShaderGlobals* sg,
                        const OSL::Vec3& P, const OSL::Vec3& dPdx,
                        const OSL::Vec3& dPdy, const OSL::Vec3& R,
                        const OSL::Vec3& dRdx, const OSL::Vec3& dRdy)
{
    return false;
}



bool
RendererServices::getmessage(ShaderGlobals* sg, ustringhash source,
                             ustringhash name, TypeDesc type, void* val,
                             bool derivatives)
{
    return false;
}



BatchedRendererServices<16>*
RendererServices::batched(WidthOf<16>)
{
    // No default implementation for batched services
    return nullptr;
}

BatchedRendererServices<8>*
RendererServices::batched(WidthOf<8>)
{
    // No default implementation for batched services
    return nullptr;
}

OSL_NAMESPACE_EXIT
