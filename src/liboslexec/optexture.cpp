// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of texture operations.
///
/////////////////////////////////////////////////////////////////////////

#include <OpenImageIO/fmath.h>
#include <OpenImageIO/simd.h>

#include <cmath>
#include <iostream>

#include "oslexec_pvt.h"
#include <OSL/dual.h>
#include <OSL/rs_free_function.h>


OSL_NAMESPACE_ENTER
namespace pvt {


OSL_SHADEOP OSL_HOSTDEVICE void
osl_init_texture_options(OpaqueExecContextPtr oec, void* opt)
{
    // TODO: Simplify when TextureOpt() has __device__ marker.
    // new (opt) TextureOpt;
    TextureOpt* o          = reinterpret_cast<TextureOpt*>(opt);
    o->firstchannel        = 0;
    o->subimage            = 0;
    o->swrap               = TextureOpt::WrapDefault;
    o->twrap               = TextureOpt::WrapDefault;
    o->mipmode             = TextureOpt::MipModeDefault;
    o->interpmode          = TextureOpt::InterpSmartBicubic;
    o->anisotropic         = 32;
    o->conservative_filter = true;
    o->sblur               = 0.0f;
    o->tblur               = 0.0f;
    o->swidth              = 1.0f;
    o->twidth              = 1.0f;
    o->fill                = 0.0f;
    o->missingcolor        = nullptr;
    o->time                = 0.0f;
    o->rnd                 = -1.0f;
    o->samples             = 1;
    o->rwrap               = TextureOpt::WrapDefault;
    o->rblur               = 0.0f;
    o->rwidth              = 1.0f;
#ifdef OIIO_TEXTURESYSTEM_SUPPORTS_COLORSPACE
    o->colortransformid = 0;
#endif
    //o->envlayout = 0;
}


OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_firstchannel(void* opt, int x)
{
    ((TextureOpt*)opt)->firstchannel = x;
}

OSL_HOSTDEVICE inline TextureOpt::Wrap
decode_wrapmode(ustringhash_pod name_)
{
    // TODO: Enable when decode_wrapmode has __device__ marker.
#ifndef __CUDA_ARCH__
    ustringhash name_hash = ustringhash_from(name_);
#    ifdef OIIO_TEXTURESYSTEM_SUPPORTS_DECODE_BY_USTRINGHASH
    return OIIO::TextureOpt::decode_wrapmode(name_hash);
#    else
    ustring name = ustring_from(name_hash);
    return OIIO::TextureOpt::decode_wrapmode(name);
#    endif
#else
    return TextureOpt::WrapDefault;
#endif
}

OSL_SHADEOP OSL_HOSTDEVICE int
osl_texture_decode_wrapmode(ustringhash_pod name_)
{
    return decode_wrapmode(name_);
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_swrap(void* opt, ustringhash_pod x_)
{
    ((TextureOpt*)opt)->swrap = decode_wrapmode(x_);
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_twrap(void* opt, ustringhash_pod x_)
{
    ((TextureOpt*)opt)->twrap = decode_wrapmode(x_);
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_rwrap(void* opt, ustringhash_pod x_)
{
    ((TextureOpt*)opt)->rwrap = decode_wrapmode(x_);
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_stwrap(void* opt, ustringhash_pod x_)
{
    TextureOpt::Wrap code     = decode_wrapmode(x_);
    ((TextureOpt*)opt)->swrap = code;
    ((TextureOpt*)opt)->twrap = code;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_swrap_code(void* opt, int mode)
{
    ((TextureOpt*)opt)->swrap = (TextureOpt::Wrap)mode;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_twrap_code(void* opt, int mode)
{
    ((TextureOpt*)opt)->twrap = (TextureOpt::Wrap)mode;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_rwrap_code(void* opt, int mode)
{
    ((TextureOpt*)opt)->rwrap = (TextureOpt::Wrap)mode;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_stwrap_code(void* opt, int mode)
{
    ((TextureOpt*)opt)->swrap = (TextureOpt::Wrap)mode;
    ((TextureOpt*)opt)->twrap = (TextureOpt::Wrap)mode;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_sblur(void* opt, float x)
{
    ((TextureOpt*)opt)->sblur = x;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_tblur(void* opt, float x)
{
    ((TextureOpt*)opt)->tblur = x;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_rblur(void* opt, float x)
{
    ((TextureOpt*)opt)->rblur = x;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_stblur(void* opt, float x)
{
    ((TextureOpt*)opt)->sblur = x;
    ((TextureOpt*)opt)->tblur = x;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_swidth(void* opt, float x)
{
    ((TextureOpt*)opt)->swidth = x;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_twidth(void* opt, float x)
{
    ((TextureOpt*)opt)->twidth = x;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_rwidth(void* opt, float x)
{
    ((TextureOpt*)opt)->rwidth = x;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_stwidth(void* opt, float x)
{
    ((TextureOpt*)opt)->swidth = x;
    ((TextureOpt*)opt)->twidth = x;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_fill(void* opt, float x)
{
    ((TextureOpt*)opt)->fill = x;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_time(void* opt, float x)
{
    ((TextureOpt*)opt)->time = x;
}

OSL_SHADEOP OSL_HOSTDEVICE int
osl_texture_decode_interpmode(ustringhash_pod name_)
{
    ustringhash name_hash = ustringhash_from(name_);
    return tex_interp_to_code(name_hash);
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_interp(void* opt, ustringhash_pod modename_)
{
    ustringhash modename_hash = ustringhash_from(modename_);
    int mode                  = tex_interp_to_code(modename_hash);
    if (mode >= 0)
        ((TextureOpt*)opt)->interpmode = (TextureOpt::InterpMode)mode;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_interp_code(void* opt, int mode)
{
    ((TextureOpt*)opt)->interpmode = (TextureOpt::InterpMode)mode;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_subimage(void* opt, int subimage)
{
    ((TextureOpt*)opt)->subimage = subimage;
}


OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_subimagename(void* opt, ustringhash_pod subimagename_)
{
    // TODO: Enable when subimagename is ustringhash.
#ifndef __CUDA_ARCH__
    ustringhash subimagename_hash    = ustringhash_from(subimagename_);
    ustring subimagename             = ustring_from(subimagename_hash);
    ((TextureOpt*)opt)->subimagename = subimagename;
#endif
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_missingcolor_arena(void* opt, const void* missing)
{
    ((TextureOpt*)opt)->missingcolor = (const float*)missing;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_texture_set_missingcolor_alpha(void* opt, int alphaindex,
                                   float missingalpha)
{
    float* m = (float*)((TextureOpt*)opt)->missingcolor;
    if (m)
        m[alphaindex] = missingalpha;
}



OSL_SHADEOP OSL_HOSTDEVICE int
osl_texture(OpaqueExecContextPtr oec, ustringhash_pod name_, void* handle,
            void* opt_, float s, float t, float dsdx, float dtdx, float dsdy,
            float dtdy, int chans, void* result_, void* dresultdx_,
            void* dresultdy_, void* alpha_, void* dalphadx_, void* dalphady_,
            void* errormessage_)
{
#ifndef __CUDA_ARCH__
    using float4 = OIIO::simd::vfloat4;
#else
    using float4 = Imath::Vec4<float>;
#endif
    TextureOpt* opt               = (TextureOpt*)opt_;
    float* result                 = (float*)result_;
    float* dresultdx              = (float*)dresultdx_;
    float* dresultdy              = (float*)dresultdy_;
    float* alpha                  = (float*)alpha_;
    float* dalphadx               = (float*)dalphadx_;
    float* dalphady               = (float*)dalphady_;
    ustringhash_pod* errormessage = (ustringhash_pod*)errormessage_;
    bool derivs                   = (dresultdx || dalphadx);
#ifndef __CUDA_ARCH__
    ShaderGlobals* sg = (ShaderGlobals*)oec;
#endif
    // It's actually faster to ask for 4 channels (even if we need fewer)
    // and ensure that they're being put in aligned memory.
    float4 result_simd, dresultds_simd, dresultdt_simd;
    ustringhash em;
    ustringhash name = ustringhash_from(name_);
    bool ok = rs_texture(oec, name, (TextureSystem::TextureHandle*)handle,
#ifndef __CUDA_ARCH__
                         sg->context->texture_thread_info(),
#else
                         nullptr,
#endif
                         *opt, s, t, dsdx, dtdx, dsdy, dtdy, 4,
                         (float*)&result_simd,
                         derivs ? (float*)&dresultds_simd : NULL,
                         derivs ? (float*)&dresultdt_simd : NULL,
                         errormessage ? &em : nullptr);

    for (int i = 0; i < chans; ++i)
        result[i] = result_simd[i];
    if (alpha)
        alpha[0] = result_simd[chans];

    // Correct our st texture space gradients into xy-space gradients
    if (derivs) {
        OSL_DASSERT((dresultdx == nullptr) == (dresultdy == nullptr));
        OSL_DASSERT((dalphadx == nullptr) == (dalphady == nullptr));
        float4 dresultdx_simd = dresultds_simd * dsdx + dresultdt_simd * dtdx;
        float4 dresultdy_simd = dresultds_simd * dsdy + dresultdt_simd * dtdy;
        if (dresultdx) {
            for (int i = 0; i < chans; ++i)
                dresultdx[i] = dresultdx_simd[i];
            for (int i = 0; i < chans; ++i)
                dresultdy[i] = dresultdy_simd[i];
        }
        if (dalphadx) {
            dalphadx[0] = dresultdx_simd[chans];
            dalphady[0] = dresultdy_simd[chans];
        }
    }

    if (errormessage)
        *errormessage = ok ? ustringhash {}.hash() : em.hash();
    return ok;
}



OSL_SHADEOP OSL_HOSTDEVICE int
osl_texture3d(OpaqueExecContextPtr oec, ustringhash_pod name_, void* handle,
              void* opt_, void* P_, void* dPdx_, void* dPdy_, void* dPdz_,
              int chans, void* result_, void* dresultdx_, void* dresultdy_,
              void* alpha_, void* dalphadx_, void* dalphady_,
              void* errormessage_)
{
#ifndef __CUDA_ARCH__
    using float4 = OIIO::simd::vfloat4;
#else
    using float4 = Imath::Vec4<float>;
#endif
    const Vec3& P(*(Vec3*)P_);
    const Vec3& dPdx(*(Vec3*)dPdx_);
    const Vec3& dPdy(*(Vec3*)dPdy_);
    Vec3 dPdz(0.0f);
    if (dPdz_ != nullptr) {
        dPdz = (*(Vec3*)dPdz_);
    }
    TextureOpt* opt               = (TextureOpt*)opt_;
    float* result                 = (float*)result_;
    float* dresultdx              = (float*)dresultdx_;
    float* dresultdy              = (float*)dresultdy_;
    float* alpha                  = (float*)alpha_;
    float* dalphadx               = (float*)dalphadx_;
    float* dalphady               = (float*)dalphady_;
    ustringhash_pod* errormessage = (ustringhash_pod*)errormessage_;
    bool derivs                   = (dresultdx || dalphadx);
#ifndef __CUDA_ARCH__
    ShaderGlobals* sg = (ShaderGlobals*)oec;
#endif
    // It's actually faster to ask for 4 channels (even if we need fewer)
    // and ensure that they're being put in aligned memory.
    float4 result_simd, dresultds_simd, dresultdt_simd, dresultdr_simd;
    ustringhash em;
    ustringhash name = ustringhash_from(name_);
    bool ok = rs_texture3d(oec, name, (TextureSystem::TextureHandle*)handle,
#ifndef __CUDA_ARCH__
                           sg->context->texture_thread_info(),
#else
                           nullptr,
#endif
                           *opt, P, dPdx, dPdy, dPdz, 4, (float*)&result_simd,
                           derivs ? (float*)&dresultds_simd : nullptr,
                           derivs ? (float*)&dresultdt_simd : nullptr,
                           derivs ? (float*)&dresultdr_simd : nullptr,
                           errormessage ? &em : nullptr);

    for (int i = 0; i < chans; ++i)
        result[i] = result_simd[i];
    if (alpha)
        alpha[0] = result_simd[chans];

    // Correct our str texture space gradients into xyz-space gradients
    if (derivs) {
        float4 dresultdx_simd = dresultds_simd * dPdx.x
                                + dresultdt_simd * dPdx.y
                                + dresultdr_simd * dPdx.z;
        float4 dresultdy_simd = dresultds_simd * dPdy.x
                                + dresultdt_simd * dPdy.y
                                + dresultdr_simd * dPdy.z;
        if (dresultdx) {
            for (int i = 0; i < chans; ++i)
                dresultdx[i] = dresultdx_simd[i];
            for (int i = 0; i < chans; ++i)
                dresultdy[i] = dresultdy_simd[i];
        }
        if (dalphadx) {
            dalphadx[0] = dresultdx_simd[chans];
            dalphady[0] = dresultdy_simd[chans];
        }
    }

    if (errormessage)
        *errormessage = ok ? ustringhash {}.hash() : em.hash();
    return ok;
}



OSL_SHADEOP OSL_HOSTDEVICE int
osl_environment(OpaqueExecContextPtr oec, ustringhash_pod name_, void* handle,
                void* opt_, void* R_, void* dRdx_, void* dRdy_, int chans,
                void* result_, void* dresultdx_, void* dresultdy_, void* alpha_,
                void* dalphadx_, void* dalphady_, void* errormessage_)
{
#ifndef __CUDA_ARCH__
    using float4 = OIIO::simd::vfloat4;
#else
    using float4 = Imath::Vec4<float>;
#endif
    const Vec3& R(*(Vec3*)R_);
    const Vec3& dRdx(*(Vec3*)dRdx_);
    const Vec3& dRdy(*(Vec3*)dRdy_);
    TextureOpt* opt               = (TextureOpt*)opt_;
    float* result                 = (float*)result_;
    float* dresultdx              = (float*)dresultdx_;
    float* dresultdy              = (float*)dresultdy_;
    float* alpha                  = (float*)alpha_;
    float* dalphadx               = (float*)dalphadx_;
    float* dalphady               = (float*)dalphady_;
    ustringhash_pod* errormessage = (ustringhash_pod*)errormessage_;
#ifndef __CUDA_ARCH__
    ShaderGlobals* sg = (ShaderGlobals*)oec;
#endif
    // It's actually faster to ask for 4 channels (even if we need fewer)
    // and ensure that they're being put in aligned memory.
    float4 local_result;
    ustringhash em;
    ustringhash name = ustringhash_from(name_);
    bool ok = rs_environment(oec, name, (TextureSystem::TextureHandle*)handle,
#ifndef __CUDA_ARCH__
                             sg->context->texture_thread_info(),
#else
                             nullptr,
#endif
                             *opt, R, dRdx, dRdy, 4, (float*)&local_result,
                             NULL, NULL, errormessage ? &em : nullptr);

    for (int i = 0; i < chans; ++i)
        result[i] = local_result[i];

    // For now, just zero out the result derivatives.  If somebody needs
    // derivatives of environment lookups, we'll fix it.  The reason
    // that this is a pain is that OIIO's environment call (unwisely?)
    // returns the st gradients, but we want the xy gradients, which is
    // tricky because we (this function you're reading) don't know which
    // projection is used to generate st from R.  Ugh.  Sweep under the
    // rug for a day when somebody is really asking for it.
    if (dresultdx) {
        for (int i = 0; i < chans; ++i)
            dresultdx[i] = 0.0f;
        for (int i = 0; i < chans; ++i)
            dresultdy[i] = 0.0f;
    }
    if (alpha) {
        alpha[0] = local_result[chans];
        // Zero out the alpha derivatives, for the same reason as above.
        if (dalphadx)
            dalphadx[0] = 0.0f;
        if (dalphady)
            dalphady[0] = 0.0f;
    }

    if (errormessage)
        *errormessage = ok ? ustringhash {}.hash() : em.hash();
    return ok;
}



OSL_SHADEOP OSL_HOSTDEVICE int
osl_get_textureinfo(OpaqueExecContextPtr oec, ustringhash_pod name_,
                    void* handle_, ustringhash_pod dataname_, int type,
                    int arraylen, int aggregate, void* data,
                    void* errormessage_)
{
    // recreate TypeDesc
    TypeDesc typedesc;
    typedesc.basetype  = type;
    typedesc.arraylen  = arraylen;
    typedesc.aggregate = aggregate;

    ustringhash name     = ustringhash_from(name_);
    ustringhash dataname = ustringhash_from(dataname_);

    TextureSystem::TextureHandle* handle
        = (TextureSystem::TextureHandle*)handle_;

    ustringhash_pod* errormessage = (ustringhash_pod*)errormessage_;

#ifndef __CUDA_ARCH__
    ShaderGlobals* sg = (ShaderGlobals*)oec;
#endif

    ustringhash em;
    bool ok = rs_get_texture_info(oec, name, handle,
#ifndef __CUDA_ARCH__
                                  sg->context->texture_thread_info(),
#else
                                  nullptr,
#endif
                                  0 /*FIXME-ptex*/, dataname, typedesc, data,
                                  errormessage ? &em : nullptr);
    if (errormessage)
        *errormessage = ok ? ustringhash {}.hash() : em.hash();
    return ok;
}



OSL_SHADEOP OSL_HOSTDEVICE int
osl_get_textureinfo_st(OpaqueExecContextPtr oec, ustringhash_pod name_,
                       void* handle_, float s, float t,
                       ustringhash_pod dataname_, int type, int arraylen,
                       int aggregate, void* data, void* errormessage_)
{
    // recreate TypeDesc
    TypeDesc typedesc;
    typedesc.basetype  = type;
    typedesc.arraylen  = arraylen;
    typedesc.aggregate = aggregate;

    ustringhash name     = ustringhash_from(name_);
    ustringhash dataname = ustringhash_from(dataname_);

    TextureSystem::TextureHandle* handle
        = (TextureSystem::TextureHandle*)handle_;

    ustringhash_pod* errormessage = (ustringhash_pod*)errormessage_;

#ifndef __CUDA_ARCH__
    ShaderGlobals* sg = (ShaderGlobals*)oec;
#endif

    ustringhash em;
    bool ok = rs_get_texture_info_st(oec, name, handle, s, t,
#ifndef __CUDA_ARCH__
                                     sg->context->texture_thread_info(),
#else
                                     nullptr,
#endif
                                     0 /*FIXME-ptex*/, dataname, typedesc, data,
                                     errormessage ? &em : nullptr);
    if (errormessage)
        *errormessage = ok ? ustringhash {}.hash() : em.hash();
    return ok;
}



// Trace

OSL_SHADEOP OSL_HOSTDEVICE void
osl_init_trace_options(OpaqueExecContextPtr oec, void* opt)
{
    new (opt) TraceOpt;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_trace_set_mindist(void* opt, float x)
{
    ((TraceOpt*)opt)->mindist = x;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_trace_set_maxdist(void* opt, float x)
{
    ((TraceOpt*)opt)->maxdist = x;
}

OSL_SHADEOP OSL_HOSTDEVICE void
osl_trace_set_shade(void* opt, int x)
{
    ((TraceOpt*)opt)->shade = x;
}


OSL_SHADEOP OSL_HOSTDEVICE void
osl_trace_set_traceset(void* opt, const ustringhash_pod x)
{
    ((TraceOpt*)opt)->traceset = ustringhash_from(x);
}


OSL_SHADEOP OSL_HOSTDEVICE int
osl_trace(OpaqueExecContextPtr oec, void* opt_, void* Pos_, void* dPosdx_,
          void* dPosdy_, void* Dir_, void* dDirdx_, void* dDirdy_)
{
    TraceOpt* opt = (TraceOpt*)opt_;
    static const Vec3 Zero(0.0f, 0.0f, 0.0f);
    const Vec3* Pos    = (Vec3*)Pos_;
    const Vec3* dPosdx = dPosdx_ ? (Vec3*)dPosdx_ : &Zero;
    const Vec3* dPosdy = dPosdy_ ? (Vec3*)dPosdy_ : &Zero;
    const Vec3* Dir    = (Vec3*)Dir_;
    const Vec3* dDirdx = dDirdx_ ? (Vec3*)dDirdx_ : &Zero;
    const Vec3* dDirdy = dDirdy_ ? (Vec3*)dDirdy_ : &Zero;
    return rs_trace(oec, *opt, *Pos, *dPosdx, *dPosdy, *Dir, *dDirdx, *dDirdy);
}


}  // namespace pvt
OSL_NAMESPACE_EXIT
