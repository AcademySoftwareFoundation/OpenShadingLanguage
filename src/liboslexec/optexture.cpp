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



OSL_SHADEOP int
osl_texture (void *sg_, const char *name, void *handle,
             void *opt_, float s, float t,
             float dsdx, float dtdx, float dsdy, float dtdy,
             int chans, void *result, void *dresultdx, void *dresultdy,
             void *alpha, void *dalphadx, void *dalphady)
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
                                     derivs ? (float *)&dresultdt_simd : NULL);

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

    return ok;
}



OSL_SHADEOP int
osl_texture3d (void *sg_, const char *name, void *handle,
               void *opt_, void *P_,
               void *dPdx_, void *dPdy_, void *dPdz_, int chans,
               void *result, void *dresultdx,
               void *dresultdy, void *dresultdz,
               void *alpha, void *dalphadx,
               void *dalphady, void *dalphadz)
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
                                       derivs ? (float *)&dresultdr_simd : NULL);

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
    return ok;
}



OSL_SHADEOP int
osl_environment (void *sg_, const char *name, void *handle,
                 void *opt_, void *R_,
                 void *dRdx_, void *dRdy_, int chans,
                 void *result, void *dresultdx, void *dresultdy,
                 void *alpha, void *dalphadx, void *dalphady)
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
                                         (float *)&local_result, NULL, NULL);

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


} // namespace pvt
OSL_NAMESPACE_EXIT
