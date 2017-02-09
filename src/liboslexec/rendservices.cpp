/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
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

#include <vector>
#include <string>
#include <cstdio>

#include "oslexec_pvt.h"
using namespace OSL;
using namespace OSL::pvt;

#include <OpenImageIO/strutil.h>
#include <OpenImageIO/dassert.h>
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
        ASSERT (0 && "RendererServices was not passed a working TextureSystem*");
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
RendererServices::get_texture_handle (ustring filename)
{
    return texturesys()->get_texture_handle (filename);
}



bool
RendererServices::good (TextureHandle *texture_handle)
{
    return texturesys()->good (texture_handle);
}



RendererServices::TexturePerthread *
RendererServices::get_texture_perthread (ShadingContext *context)
{
    return context ? context->texture_thread_info()
                   : texturesys()->get_perthread_info();
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
    bool status;
    if (texture_handle)
        status = texturesys()->texture (texture_handle, texture_thread_info,
                                        options, s, t, dsdx, dtdx, dsdy, dtdy,
                                        nchannels, result, dresultds, dresultdt);
    else
        status = texturesys()->texture (filename,
                                        options, s, t, dsdx, dtdx, dsdy, dtdy,
                                        nchannels, result, dresultds, dresultdt);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sg) {
            if (errormessage) {
                *errormessage = ustring(err);
            } else {
                context->error ("[RendererServices::texture] %s", err);
            }
        } else if (errormessage) {
            *errormessage = Strings::unknown;
        }
    }
    return status;
}



// Deprecated version
bool
RendererServices::texture (ustring filename, TextureHandle *texture_handle,
                           TexturePerthread *texture_thread_info,
                           TextureOpt &options, ShaderGlobals *sg,
                           float s, float t, float dsdx, float dtdx,
                           float dsdy, float dtdy, int nchannels,
                           float *result, float *dresultds, float *dresultdt)
{
    ShadingContext *context = sg->context;
    if (! texture_thread_info)
        texture_thread_info = context->texture_thread_info();
    bool status;
    if (texture_handle)
        status = texturesys()->texture (texture_handle, texture_thread_info,
                                        options, s, t, dsdx, dtdx, dsdy, dtdy,
                                        nchannels, result, dresultds, dresultdt);
    else
        status = texturesys()->texture (filename,
                                        options, s, t, dsdx, dtdx, dsdy, dtdy,
                                        nchannels, result, dresultds, dresultdt);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sg) {
            context->error ("[RendererServices::texture] %s", err);
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
    bool status;
    if (texture_handle)
        status = texturesys()->texture3d (texture_handle, texture_thread_info,
                                          options, P, dPdx, dPdy, dPdz,
                                          nchannels, result,
                                          dresultds, dresultdt, dresultdr);
    else
        status = texturesys()->texture3d (filename,
                                          options, P, dPdx, dPdy, dPdz,
                                          nchannels, result,
                                          dresultds, dresultdt, dresultdr);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sg) {
            if (errormessage) {
                *errormessage = ustring(err);
            } else {
                sg->context->error ("[RendererServices::texture3d] %s", err);
            }
        } else if (errormessage) {
            *errormessage = Strings::unknown;
        }
    }
    return status;
}



// Deprecated version
bool
RendererServices::texture3d (ustring filename, TextureHandle *texture_handle,
                             TexturePerthread *texture_thread_info,
                             TextureOpt &options, ShaderGlobals *sg,
                             const Vec3 &P, const Vec3 &dPdx, const Vec3 &dPdy,
                             const Vec3 &dPdz, int nchannels, float *result,
                             float *dresultds, float *dresultdt, float *dresultdr)
{
    ShadingContext *context = sg->context;
    if (! texture_thread_info)
        texture_thread_info = context->texture_thread_info();
    bool status;
    if (texture_handle)
        status = texturesys()->texture3d (texture_handle, texture_thread_info,
                                          options, P, dPdx, dPdy, dPdz,
                                          nchannels, result,
                                          dresultds, dresultdt, dresultdr);
    else
        status = texturesys()->texture3d (filename,
                                          options, P, dPdx, dPdy, dPdz,
                                          nchannels, result,
                                          dresultds, dresultdt, dresultdr);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sg) {
            sg->context->error ("[RendererServices::texture3d] %s", err);
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
    bool status;
    if (texture_handle)
        status = texturesys()->environment (texture_handle, texture_thread_info,
                                            options, R, dRdx, dRdy,
                                            nchannels, result, dresultds, dresultdt);
    else
        status = texturesys()->environment (filename, options, R, dRdx, dRdy,
                                            nchannels, result, dresultds, dresultdt);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sg) {
            if (errormessage) {
                *errormessage = ustring(err);
            } else {
                sg->context->error ("[RendererServices::environment] %s", err);
            }
        } else if (errormessage) {
            *errormessage = Strings::unknown;
        }
    }
    return status;
}



// Deprecated version
bool
RendererServices::environment (ustring filename, TextureHandle *texture_handle,
                               TexturePerthread *texture_thread_info,
                               TextureOpt &options, ShaderGlobals *sg,
                               const Vec3 &R, const Vec3 &dRdx, const Vec3 &dRdy,
                               int nchannels, float *result,
                               float *dresultds, float *dresultdt)
{
    ShadingContext *context = sg->context;
    if (! texture_thread_info)
        texture_thread_info = context->texture_thread_info();
    bool status;
    if (texture_handle)
        status = texturesys()->environment (texture_handle, texture_thread_info,
                                            options, R, dRdx, dRdy,
                                            nchannels, result, dresultds, dresultdt);
    else
        status = texturesys()->environment (filename, options, R, dRdx, dRdy,
                                            nchannels, result, dresultds, dresultdt);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sg) {
            sg->context->error ("[RendererServices::environment] %s", err);
        }
    }
    return status;
}



bool
RendererServices::get_texture_info (ShaderGlobals *sg, ustring filename,
                                    TextureHandle *texture_handle,
                                    int subimage, ustring dataname,
                                    TypeDesc datatype, void *data)
{
    bool status;
    if (texture_handle)
        status = texturesys()->get_texture_info (texture_handle, NULL, subimage,
                                                 dataname, datatype, data);
    else
        status = texturesys()->get_texture_info (filename, subimage,
                                                 dataname, datatype, data);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sg) {
            sg->context->error ("[RendererServices::get_texture_info] %s", err);
        }
    }
    return status;
}



OSL_NAMESPACE_EXIT
