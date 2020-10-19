// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include <vector>
#include <string>
#include <cstdio>

#include "oslexec_pvt.h"
using namespace OSL;
using namespace OSL::pvt;

#include <OpenImageIO/strutil.h>
#include <OpenImageIO/filesystem.h>


OSL_NAMESPACE_ENTER



RendererServices::RendererServices (TextureSystem *texsys)
    : m_texturesys(texsys)
{
    if (! m_texturesys) {
#if OSL_NO_DEFAULT_TEXTURESYSTEM
        // This build option instructs OSL to never create a TextureSystem
        // itself. (Most likely reason: this build of OSL is for a renderer
        // that replaces OIIO's TextureSystem with its own, and therefore
        // wouldn't want to accidentally make an OIIO one here.
        OSL_ASSERT (0 && "RendererServices was not passed a working TextureSystem*");
#else
        m_texturesys = TextureSystem::create (true /* shared */);
        // Make some good guesses about default options
        m_texturesys->attribute ("automip",  1);
        m_texturesys->attribute ("autotile", 64);
#endif
    }
}



TextureSystem *
RendererServices::texturesys () const
{
    return m_texturesys;
}



bool
RendererServices::get_inverse_matrix (ShaderGlobals *sg, Matrix44 &result,
                                      TransformationPtr xform, float time)
{
    bool ok = get_matrix (sg, result, xform, time);
    if (ok)
        result.invert ();
    return ok;
}



bool
RendererServices::get_inverse_matrix (ShaderGlobals *sg, Matrix44 &result,
                                      TransformationPtr xform)
{
    bool ok = get_matrix (sg, result, xform);
    if (ok)
        result.invert ();
    return ok;
}



bool
RendererServices::get_inverse_matrix (ShaderGlobals *sg, Matrix44 &result,
                                      ustring to, float time)
{
    bool ok = get_matrix (sg, result, to, time);
    if (ok)
        result.invert ();
    return ok;
}



bool
RendererServices::get_inverse_matrix (ShaderGlobals *sg, Matrix44 &result,
                                      ustring to)
{
    bool ok = get_matrix (sg, result, to);
    if (ok)
        result.invert ();
    return ok;
}




RendererServices::TextureHandle *
RendererServices::get_texture_handle (ustring filename, ShadingContext *context)
{
    return texturesys()->get_texture_handle (filename, context->texture_thread_info());
}



bool
RendererServices::good (TextureHandle *texture_handle)
{
    return texturesys()->good (texture_handle);
}


bool
RendererServices::texture (ustring filename, TextureHandle *texture_handle,
                           TexturePerthread *texture_thread_info,
                           TextureOpt &options, ShaderGlobals *sg,
                           float s, float t, float dsdx, float dtdx,
                           float dsdy, float dtdy, int nchannels,
                           float *result, float *dresultds, float *dresultdt,
                           ustring *errormessage)
{
    ShadingContext *context = sg->context;
    if (! texture_thread_info)
        texture_thread_info = context->texture_thread_info();
    if (! texture_handle)
        texture_handle = texturesys()->get_texture_handle (filename, texture_thread_info);
    bool status = texturesys()->texture (texture_handle, texture_thread_info,
                                         options, s, t, dsdx, dtdx, dsdy, dtdy,
                                         nchannels, result, dresultds, dresultdt);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sg) {
            if (errormessage) {
                *errormessage = ustring(err);
            } else {
                context->errorf("[RendererServices::texture] %s", err);
            }
        } else if (errormessage) {
            *errormessage = Strings::unknown;
        }
    }
    return status;
}



bool
RendererServices::texture3d (ustring filename, TextureHandle *texture_handle,
                             TexturePerthread *texture_thread_info,
                             TextureOpt &options, ShaderGlobals *sg,
                             const Vec3 &P, const Vec3 &dPdx, const Vec3 &dPdy,
                             const Vec3 &dPdz, int nchannels, float *result,
                             float *dresultds, float *dresultdt, float *dresultdr,
                             ustring *errormessage)
{
    ShadingContext *context = sg->context;
    if (! texture_thread_info)
        texture_thread_info = context->texture_thread_info();
    if (! texture_handle)
        texture_handle = texturesys()->get_texture_handle (filename, texture_thread_info);

    bool status = texturesys()->texture3d (texture_handle, texture_thread_info,
                                           options, P, dPdx, dPdy, dPdz,
                                           nchannels, result,
                                           dresultds, dresultdt, dresultdr);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sg) {
            if (errormessage) {
                *errormessage = ustring(err);
            } else {
                sg->context->errorf("[RendererServices::texture3d] %s", err);
            }
        } else if (errormessage) {
            *errormessage = Strings::unknown;
        }
    }
    return status;
}



bool
RendererServices::environment (ustring filename, TextureHandle *texture_handle,
                               TexturePerthread *texture_thread_info,
                               TextureOpt &options, ShaderGlobals *sg,
                               const Vec3 &R, const Vec3 &dRdx, const Vec3 &dRdy,
                               int nchannels, float *result,
                               float *dresultds, float *dresultdt,
                               ustring *errormessage)
{
    ShadingContext *context = sg->context;
    if (! texture_thread_info)
        texture_thread_info = context->texture_thread_info();
    if (! texture_handle)
        texture_handle = texturesys()->get_texture_handle (filename, texture_thread_info);
    bool status = texturesys()->environment (texture_handle, texture_thread_info,
                                             options, R, dRdx, dRdy,
                                             nchannels, result, dresultds, dresultdt);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sg) {
            if (errormessage) {
                *errormessage = ustring(err);
            } else {
                sg->context->errorf("[RendererServices::environment] %s", err);
            }
        } else if (errormessage) {
            *errormessage = Strings::unknown;
        }
    }
    return status;
}



bool
RendererServices::get_texture_info (ustring filename,
                                    TextureHandle *texture_handle,
                                    TexturePerthread *texture_thread_info,
                                    ShadingContext *shading_context,
                                    int subimage, ustring dataname,
                                    TypeDesc datatype,
                                    void *data, ustring *errormessage)
{
    if (! texture_thread_info)
        texture_thread_info = shading_context->texture_thread_info();
    if (! texture_handle)
        texture_handle = texturesys()->get_texture_handle (filename, texture_thread_info);
    bool status = texturesys()->get_texture_info (texture_handle, texture_thread_info, subimage,
                                                  dataname, datatype, data);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size()) {
            if (errormessage) {
                *errormessage = ustring(err);
            } else {
                shading_context->errorf("[RendererServices::get_texture_info] %s", err);
            }
        } else if (errormessage) {
            // gettextureinfo failed but did not provide an error, so none should be emitted
            *errormessage = ustring();
        }
    }
    return status;
}

BatchedRendererServices<16> *
RendererServices::batched(WidthOf<16>)
{
    // No default implementation for batched services
    return nullptr;
}

BatchedRendererServices<8> *
RendererServices::batched(WidthOf<8>)
{
    // No default implementation for batched services
    return nullptr;
}

OSL_NAMESPACE_EXIT
