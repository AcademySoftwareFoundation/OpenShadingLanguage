/*
Copyright (c) 2009-2015 Sony Pictures Imageworks Inc., et al.
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


/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of texture operations.
///
/////////////////////////////////////////////////////////////////////////

#include <OpenImageIO/fmath.h>
#include <OpenImageIO/simd.h>

#include <iostream>
#include <cmath>

#include "oslexec_pvt.h"
#include "OSL/dual.h"


OSL_NAMESPACE_ENTER
namespace pvt {


// Utility: retrieve a pointer to the ShadingContext's texture options
// struct, also re-initialize its contents.
OSL_SHADEOP void *
osl_get_texture_options (void *sg_)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    TextureOpt *opt = sg->context->texture_options_ptr ();
    new (opt) TextureOpt;
    return opt;
}


OSL_SHADEOP void
osl_texture_set_firstchannel (void *opt, int x)
{
    ((TextureOpt *)opt)->firstchannel = x;
}


OSL_SHADEOP void
osl_texture_set_swrap (void *opt, const char *x)
{
    ((TextureOpt *)opt)->swrap = TextureOpt::decode_wrapmode(USTR(x));
}

OSL_SHADEOP void
osl_texture_set_twrap (void *opt, const char *x)
{
    ((TextureOpt *)opt)->twrap = TextureOpt::decode_wrapmode(USTR(x));
}

OSL_SHADEOP void
osl_texture_set_rwrap (void *opt, const char *x)
{
    ((TextureOpt *)opt)->rwrap = TextureOpt::decode_wrapmode(USTR(x));
}

OSL_SHADEOP void
osl_texture_set_stwrap (void *opt, const char *x)
{
    TextureOpt::Wrap code = TextureOpt::decode_wrapmode(USTR(x));
    ((TextureOpt *)opt)->swrap = code;
    ((TextureOpt *)opt)->twrap = code;
}

OSL_SHADEOP void
osl_texture_set_swrap_code (void *opt, int mode)
{
    ((TextureOpt *)opt)->swrap = (TextureOpt::Wrap)mode;
}

OSL_SHADEOP void
osl_texture_set_twrap_code (void *opt, int mode)
{
    ((TextureOpt *)opt)->twrap = (TextureOpt::Wrap)mode;
}

OSL_SHADEOP void
osl_texture_set_rwrap_code (void *opt, int mode)
{
    ((TextureOpt *)opt)->rwrap = (TextureOpt::Wrap)mode;
}

OSL_SHADEOP void
osl_texture_set_stwrap_code (void *opt, int mode)
{
    ((TextureOpt *)opt)->swrap = (TextureOpt::Wrap)mode;
    ((TextureOpt *)opt)->twrap = (TextureOpt::Wrap)mode;
}

OSL_SHADEOP void
osl_texture_set_sblur (void *opt, float x)
{
    ((TextureOpt *)opt)->sblur = x;
}

OSL_SHADEOP void
osl_texture_set_tblur (void *opt, float x)
{
    ((TextureOpt *)opt)->tblur = x;
}

OSL_SHADEOP void
osl_texture_set_rblur (void *opt, float x)
{
    ((TextureOpt *)opt)->rblur = x;
}

OSL_SHADEOP void
osl_texture_set_stblur (void *opt, float x)
{
    ((TextureOpt *)opt)->sblur = x;
    ((TextureOpt *)opt)->tblur = x;
}

OSL_SHADEOP void
osl_texture_set_swidth (void *opt, float x)
{
    ((TextureOpt *)opt)->swidth = x;
}

OSL_SHADEOP void
osl_texture_set_twidth (void *opt, float x)
{
    ((TextureOpt *)opt)->twidth = x;
}

OSL_SHADEOP void
osl_texture_set_rwidth (void *opt, float x)
{
    ((TextureOpt *)opt)->rwidth = x;
}

OSL_SHADEOP void
osl_texture_set_stwidth (void *opt, float x)
{
    ((TextureOpt *)opt)->swidth = x;
    ((TextureOpt *)opt)->twidth = x;
}

OSL_SHADEOP void
osl_texture_set_fill (void *opt, float x)
{
    ((TextureOpt *)opt)->fill = x;
}

OSL_SHADEOP void
osl_texture_set_time (void *opt, float x)
{
    ((TextureOpt *)opt)->time = x;
}

inline int
tex_interp_to_code (ustring modename)
{
    static ustring u_linear ("linear");
    static ustring u_smartcubic ("smartcubic");
    static ustring u_cubic ("cubic");
    static ustring u_closest ("closest");

    int mode = -1;
    if (modename == u_smartcubic)
        mode = TextureOpt::InterpSmartBicubic;
    else if (modename == u_linear)
        mode = TextureOpt::InterpBilinear;
    else if (modename == u_cubic)
        mode = TextureOpt::InterpBicubic;
    else if (modename == u_closest)
        mode = TextureOpt::InterpClosest;
    return mode;
}

OSL_SHADEOP void
osl_texture_set_interp (void *opt, const char *modename)
{
    int mode = tex_interp_to_code (USTR(modename));
    if (mode >= 0)
        ((TextureOpt *)opt)->interpmode = (TextureOpt::InterpMode)mode;
}

OSL_SHADEOP void
osl_texture_set_interp_code (void *opt, int mode)
{
    ((TextureOpt *)opt)->interpmode = (TextureOpt::InterpMode)mode;
}

OSL_SHADEOP void
osl_texture_set_subimage (void *opt, int subimage)
{
    ((TextureOpt *)opt)->subimage = subimage;
}


OSL_SHADEOP void
osl_texture_set_subimagename (void *opt, const char *subimagename)
{
    ((TextureOpt *)opt)->subimagename = USTR(subimagename);
}

OSL_SHADEOP void
osl_texture_set_missingcolor_arena (void *opt, const void *missing)
{
    ((TextureOpt *)opt)->missingcolor = (const float *)missing;
}

OSL_SHADEOP void
osl_texture_set_missingcolor_alpha (void *opt, int alphaindex,
                                    float missingalpha)
{
    float *m = (float *)((TextureOpt *)opt)->missingcolor;
    if (m)
        m[alphaindex] = missingalpha;
}



OSL_SHADEOP int
osl_texture (void *sg_, const char *name, void *handle,
             void *opt_, float s, float t,
             float dsdx, float dtdx, float dsdy, float dtdy,
             int chans, void *result, void *dresultdx, void *dresultdy,
             void *alpha, void *dalphadx, void *dalphady,
             ustring *errormessage)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    TextureOpt *opt = (TextureOpt *)opt_;
    bool derivs = (dresultdx != NULL);
    // It's actually faster to ask for 4 channels (even if we need fewer)
    // and ensure that they're being put in aligned memory.
    OIIO::simd::float4 result_simd, dresultds_simd, dresultdt_simd;
    bool ok = sg->renderer->texture (USTR(name),
                                     (TextureSystem::TextureHandle *)handle, NULL,
                                     *opt, sg, s, t, dsdx, dtdx, dsdy, dtdy, 4,
                                     (float *)&result_simd,
                                     derivs ? (float *)&dresultds_simd : NULL,
                                     derivs ? (float *)&dresultdt_simd : NULL,
                                     errormessage);

    for (int i = 0;  i < chans;  ++i)
        ((float *)result)[i] = result_simd[i];
    if (alpha)
        ((float *)alpha)[0] = result_simd[chans];

    // Correct our st texture space gradients into xy-space gradients
    if (derivs) {
        OIIO::simd::float4 dresultdx_simd = dresultds_simd * dsdx + dresultdt_simd * dtdx;
        OIIO::simd::float4 dresultdy_simd = dresultds_simd * dsdy + dresultdt_simd * dtdy;
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdx)[i] = dresultdx_simd[i];
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdy)[i] = dresultdy_simd[i];
        if (dalphadx) {
            ((float *)dalphadx)[0] = dresultdx_simd[chans];
            ((float *)dalphady)[0] = dresultdy_simd[chans];
        }
    }

    if (ok && errormessage)
        *errormessage = Strings::_emptystring_;
    return ok;
}

OSL_SHADEOP int
osl_texture_batched (void *sgb_, const char *name, void *handle,
                     void *opt_, float s, float t,
                     float dsdx, float dtdx, float dsdy, float dtdy,
                     int chans, void *result, void *dresultdx, void *dresultdy,
                     void *alpha, void *dalphadx, void *dalphady,
                     ustring *errormessage,
                     int mask_)
{
    std::cout << "osl_texture_batched" << std::endl;
    Mask mask(mask_);
    // TODO: LLVM could check this before calling this function
    if (mask.all_off()) {
        return 0;
    }
    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;
    TextureOpt *opt = (TextureOpt *)opt_;
    bool derivs = (dresultdx != NULL);
    // It's actually faster to ask for 4 channels (even if we need fewer)
    // and ensure that they're being put in aligned memory.
    OIIO::simd::float4 result_simd, dresultds_simd, dresultdt_simd;
    Mask retVal = sgb->uniform().renderer->batched()->texture (USTR(name),
                                                               (TextureSystem::TextureHandle *)handle, NULL,
                                                               *opt, sgb, s, t, dsdx, dtdx, dsdy, dtdy, 4,
                                                               (float *)&result_simd,
                                                               derivs ? (float *)&dresultds_simd : NULL,
                                                               derivs ? (float *)&dresultdt_simd : NULL,
                                                               errormessage, mask_);

    for (int i = 0;  i < chans;  ++i)
        ((float *)result)[i] = result_simd[i];
    if (alpha)
        ((float *)alpha)[0] = result_simd[chans];

    // Correct our st texture space gradients into xy-space gradients
    if (derivs) {
        OIIO::simd::float4 dresultdx_simd = dresultds_simd * dsdx + dresultdt_simd * dtdx;
        OIIO::simd::float4 dresultdy_simd = dresultds_simd * dsdy + dresultdt_simd * dtdy;
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdx)[i] = dresultdx_simd[i];
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdy)[i] = dresultdy_simd[i];
        if (dalphadx) {
            ((float *)dalphadx)[0] = dresultdx_simd[chans];
            ((float *)dalphady)[0] = dresultdy_simd[chans];
        }
    }

    /// XXX lfeng: maybe we need to pass in wide errormessages?
//    if (ok && errormessage)
//        *errormessage = Strings::_emptystring_;
    return retVal.value();
}

OSL_SHADEOP int
osl_texture3d (void *sg_, const char *name, void *handle,
               void *opt_, void *P_,
               void *dPdx_, void *dPdy_, void *dPdz_, int chans,
               void *result, void *dresultdx,
               void *dresultdy, void *dresultdz,
               void *alpha, void *dalphadx,
               void *dalphady, void *dalphadz,
               ustring *errormessage)
{
    const Vec3 &P (*(Vec3 *)P_);
    const Vec3 &dPdx (*(Vec3 *)dPdx_);
    const Vec3 &dPdy (*(Vec3 *)dPdy_);
    const Vec3 &dPdz (*(Vec3 *)dPdz_);
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    TextureOpt *opt = (TextureOpt *)opt_;
    bool derivs = (dresultdx != NULL || dalphadx != NULL);
    // It's actually faster to ask for 4 channels (even if we need fewer)
    // and ensure that they're being put in aligned memory.
    OIIO::simd::float4 result_simd, dresultds_simd, dresultdt_simd, dresultdr_simd;
    bool ok = sg->renderer->texture3d (USTR(name),
                                       (TextureSystem::TextureHandle *)handle, NULL,
                                       *opt, sg, P, dPdx, dPdy, dPdz,
                                       4, (float *)&result_simd,
                                       derivs ? (float *)&dresultds_simd : NULL,
                                       derivs ? (float *)&dresultdt_simd : NULL,
                                       derivs ? (float *)&dresultdr_simd : NULL,
                                       errormessage);

    for (int i = 0;  i < chans;  ++i)
        ((float *)result)[i] = result_simd[i];
    if (alpha)
        ((float *)alpha)[0] = result_simd[chans];

    // Correct our str texture space gradients into xyz-space gradients
    if (derivs) {
        OIIO::simd::float4 dresultdx_simd = dresultds_simd * dPdx[0] + dresultdt_simd * dPdx[1] + dresultdr_simd * dPdx[2];
        OIIO::simd::float4 dresultdy_simd = dresultds_simd * dPdy[0] + dresultdt_simd * dPdy[1] + dresultdr_simd * dPdy[2];
        OIIO::simd::float4 dresultdz_simd = dresultds_simd * dPdz[0] + dresultdt_simd * dPdz[1] + dresultdr_simd * dPdz[2];
        if (dresultdx) {
            for (int i = 0;  i < chans;  ++i)
                ((float *)dresultdx)[i] = dresultdx_simd[i];
            for (int i = 0;  i < chans;  ++i)
                ((float *)dresultdy)[i] = dresultdy_simd[i];
            for (int i = 0;  i < chans;  ++i)
                ((float *)dresultdz)[i] = dresultdz_simd[i];
        }
        if (dalphadx) {
            ((float *)dalphadx)[0] = dresultdx_simd[chans];
            ((float *)dalphady)[0] = dresultdy_simd[chans];
            ((float *)dalphadz)[0] = dresultdz_simd[chans];
        }
    }

    if (ok && errormessage)
        *errormessage = Strings::_emptystring_;
    return ok;
}



OSL_SHADEOP int
osl_environment (void *sg_, const char *name, void *handle,
                 void *opt_, void *R_,
                 void *dRdx_, void *dRdy_, int chans,
                 void *result, void *dresultdx, void *dresultdy,
                 void *alpha, void *dalphadx, void *dalphady,
                 ustring *errormessage)
{
    const Vec3 &R (*(Vec3 *)R_);
    const Vec3 &dRdx (*(Vec3 *)dRdx_);
    const Vec3 &dRdy (*(Vec3 *)dRdy_);
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    TextureOpt *opt = (TextureOpt *)opt_;
    // It's actually faster to ask for 4 channels (even if we need fewer)
    // and ensure that they're being put in aligned memory.
    OIIO::simd::float4 local_result;
    bool ok = sg->renderer->environment (USTR(name),
                                         (TextureSystem::TextureHandle *)handle,
                                         NULL, *opt, sg, R, dRdx, dRdy, 4,
                                         (float *)&local_result, NULL, NULL,
                                         errormessage);

    for (int i = 0;  i < chans;  ++i)
        ((float *)result)[i] = local_result[i];

    // For now, just zero out the result derivatives.  If somebody needs
    // derivatives of environment lookups, we'll fix it.  The reason
    // that this is a pain is that OIIO's environment call (unwisely?)
    // returns the st gradients, but we want the xy gradients, which is
    // tricky because we (this function you're reading) don't know which
    // projection is used to generate st from R.  Ugh.  Sweep under the
    // rug for a day when somebody is really asking for it.
    if (dresultdx) {
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdx)[i] = 0.0f;
        for (int i = 0;  i < chans;  ++i)
            ((float *)dresultdy)[i] = 0.0f;
    }
    if (alpha) {
        ((float *)alpha)[0] = local_result[chans];
        // Zero out the alpha derivatives, for the same reason as above.
        if (dalphadx)
            ((float *)dalphadx)[0] = 0.0f;
        if (dalphady)
            ((float *)dalphady)[0] = 0.0f;
    }

    if (ok && errormessage)
        *errormessage = Strings::_emptystring_;
    return ok;
}



OSL_SHADEOP int
osl_get_textureinfo (void *sg_, const char *name, void *handle,
                     void *dataname,  int type,
                     int arraylen, int aggregate, void *data)
{
    // recreate TypeDesc
    TypeDesc typedesc;
    typedesc.basetype  = type;
    typedesc.arraylen  = arraylen;
    typedesc.aggregate = aggregate;

    ShaderGlobals *sg   = (ShaderGlobals *)sg_;

    return sg->renderer->get_texture_info (sg, USTR(name),
                                           (RendererServices::TextureHandle *)handle,
                                           0 /*FIXME-ptex*/,
                                           USTR(dataname), typedesc, data);
}


OSL_SHADEOP int
osl_get_textureinfo_batched_uniform (void *sgb_, const char *name, void *handle,
                     void *dataname,  int type,
                     int arraylen, int aggregate, void *data)
{
    // recreate TypeDesc
    TypeDesc typedesc;
    typedesc.basetype  = type;
    typedesc.arraylen  = arraylen;
    typedesc.aggregate = aggregate;

    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;

    return sgb->uniform().renderer->batched()->get_texture_info (sgb, USTR(name),
                                                                 (RendererServices::TextureHandle *)handle,
                                                                 0 /*FIXME-ptex*/,
                                                                 USTR(dataname), typedesc, data);
}

OSL_SHADEOP int
osl_get_textureinfo_batched (void *sgb_, void* name,
                             void *dataname,  int type,
                             int arraylen, int aggregate, void *data,
                             int mask_)
{
    std::cout << "osl_get_textureinfo_batched" << std::endl;
    Mask mask(mask_);
    // TODO: LLVM could check this before calling this function
    if (mask.all_off()) {
        return 0;
    }

    // recreate TypeDesc
    TypeDesc typedesc;
    typedesc.basetype  = type;
    typedesc.arraylen  = arraylen;
    typedesc.aggregate = aggregate;

    ShaderGlobalsBatch *sgb = (ShaderGlobalsBatch *)sgb_;

    Mask retVal = sgb->uniform().renderer->batched()->get_texture_info (sgb, WUSTR(name),
                                                             0 /*FIXME-ptex*/,
                                                             USTR(dataname), typedesc, data, mask);
    return retVal.value();
}

// Trace

// Utility: retrieve a pointer to the ShadingContext's trace options
// struct, also re-initialize its contents.
OSL_SHADEOP void *
osl_get_trace_options (void *sg_)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    RendererServices::TraceOpt *opt = sg->context->trace_options_ptr ();
    new (opt) RendererServices::TraceOpt;
    return opt;
}

OSL_SHADEOP void
osl_trace_set_mindist (void *opt, float x)
{
    ((RendererServices::TraceOpt *)opt)->mindist = x;
}

OSL_SHADEOP void
osl_trace_set_maxdist (void *opt, float x)
{
    ((RendererServices::TraceOpt *)opt)->maxdist = x;
}

OSL_SHADEOP void
osl_trace_set_shade (void *opt, int x)
{
    ((RendererServices::TraceOpt *)opt)->shade = x;
}


OSL_SHADEOP void
osl_trace_set_traceset (void *opt, const char *x)
{
    ((RendererServices::TraceOpt *)opt)->traceset = USTR(x);
}


OSL_SHADEOP int
osl_trace (void *sg_, void *opt_, void *Pos_, void *dPosdx_, void *dPosdy_,
           void *Dir_, void *dDirdx_, void *dDirdy_)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    RendererServices::TraceOpt *opt = (RendererServices::TraceOpt *)opt_;
    static const Vec3 Zero (0.0f, 0.0f, 0.0f);
    const Vec3 *Pos = (Vec3 *)Pos_;
    const Vec3 *dPosdx = dPosdx_ ? (Vec3 *)dPosdx_ : &Zero;
    const Vec3 *dPosdy = dPosdy_ ? (Vec3 *)dPosdy_ : &Zero;
    const Vec3 *Dir = (Vec3 *)Dir_;
    const Vec3 *dDirdx = dDirdx_ ? (Vec3 *)dDirdx_ : &Zero;
    const Vec3 *dDirdy = dDirdy_ ? (Vec3 *)dDirdy_ : &Zero;
    return sg->renderer->trace (*opt, sg, *Pos, *dPosdx, *dPosdy,
                                *Dir, *dDirdx, *dDirdy);
}


} // namespace pvt
OSL_NAMESPACE_EXIT
