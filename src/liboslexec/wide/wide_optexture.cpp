// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of texture operations.
///
/////////////////////////////////////////////////////////////////////////
#include <OSL/oslconfig.h>

#include <OSL/batched_rendererservices.h>
#include <OSL/batched_shaderglobals.h>
#include <OSL/wide.h>

#include <OpenImageIO/fmath.h>

#include "oslexec_pvt.h"

OSL_NAMESPACE_ENTER

namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

using BatchedRendererServices = OSL::BatchedRendererServices<__OSL_WIDTH>;
using WidthTag                = OSL::WidthOf<__OSL_WIDTH>;

#include "define_opname_macros.h"

namespace {


Mask
default_texture(BatchedRendererServices* bsr, ustring filename,
                TextureSystem::TextureHandle* texture_handle,
                TextureSystem::Perthread* texture_thread_info,
                const BatchedTextureOptions& options, BatchedShaderGlobals* bsg,
                Wide<const float> ws, Wide<const float> wt,
                Wide<const float> wdsdx, Wide<const float> wdtdx,
                Wide<const float> wdsdy, Wide<const float> wdtdy,
                BatchedTextureOutputs& outputs)
{
    Mask status(false);
    OSL_ASSERT(nullptr != bsg);
    ShadingContext* context = bsg->uniform.context;
    if (!texture_thread_info)
        texture_thread_info = context->texture_thread_info();
    if (!texture_handle)
        texture_handle
            = bsr->texturesys()->get_texture_handle(filename,
                                                    texture_thread_info);

    Mask mask = outputs.mask();

    MaskedData resultRef     = outputs.result();
    MaskedData alphaRef      = outputs.alpha();
    bool alphaIsValid        = outputs.alpha().valid();
    bool errormessageIsValid = outputs.errormessage().valid();
    bool has_derivs          = resultRef.has_derivs() || alphaRef.has_derivs();

    OSL_ASSERT(resultRef.valid());

    // Convert our BatchedTextureOptions to a single TextureOpt
    // and submit them 1 at a time through existing non-batched interface
    // Renderers could implement their own batched texturing,
    // although expected a future version of OIIO will support
    // this exact batched interface
    const auto& uniform_opt = options.uniform;
    TextureOpt opt;
    // opt.time = ignoring (deprecated?)
    // opt.bias = ignoring (deprecated?)
    // opt.samples = ignoring (deprecated?)
    opt.firstchannel = uniform_opt.firstchannel;
    opt.subimage     = uniform_opt.subimage;
    opt.subimagename = uniform_opt.subimagename;
    opt.swrap        = (OIIO::TextureOpt::Wrap)uniform_opt.swrap;
    opt.twrap        = (OIIO::TextureOpt::Wrap)uniform_opt.twrap;
    opt.rwrap        = (OIIO::TextureOpt::Wrap)uniform_opt.rwrap;
    opt.mipmode      = (OIIO::TextureOpt::MipMode)uniform_opt.mipmode;
    opt.interpmode   = (OIIO::TextureOpt::InterpMode)uniform_opt.interpmode;
    opt.anisotropic  = uniform_opt.anisotropic;
    opt.conservative_filter = uniform_opt.conservative_filter;
    opt.fill                = uniform_opt.fill;
    opt.missingcolor        = uniform_opt.missingcolor;

    const auto& vary_opt = options.varying;

    mask.foreach ([=, &opt, &vary_opt, &outputs, &status](ActiveLane lane) {
        opt.sblur  = vary_opt.sblur[lane];
        opt.tblur  = vary_opt.tblur[lane];
        opt.swidth = vary_opt.swidth[lane];
        opt.twidth = vary_opt.twidth[lane];

#if OIIO_VERSION_GREATER_EQUAL(2, 4, 0)
        opt.rnd = vary_opt.rnd[lane];
#endif

        // For 3D volume texture lookups only:
        //opt.rblur = vary_opt.rblur[lane];
        //opt.rwidth = vary_opt.rwidth[lane];

        // For debugging
        //std::cout << "BatchedRendererServices::texture[lane=" << lane << "opt = " << opt << std::endl;

        // It's actually faster to ask for 4 channels (even if we need fewer)
        // and ensure that they're being put in aligned memory.
        // TODO:  investigate if the above statement is true when nchannels==1
        // NOTE: using simd::float4 to speedup texture transformation below
        OIIO::simd::float4 result_simd, dresultds_simd, dresultdt_simd;

        bool retVal = false;

        float dsdx = wdsdx[lane];
        float dtdx = wdtdx[lane];
        float dsdy = wdsdy[lane];
        float dtdy = wdtdy[lane];
        retVal     = bsr->texturesys()->texture(
            texture_handle, texture_thread_info, opt, ws[lane], wt[lane], dsdx,
            dtdx, dsdy, dtdy, 4, (float*)&result_simd,
            has_derivs ? (float*)&dresultds_simd : NULL,
            has_derivs ? (float*)&dresultdt_simd : NULL);

        OIIO::simd::float4 dresultdx_simd;
        OIIO::simd::float4 dresultdy_simd;
        if (has_derivs) {
            // Correct our st texture space gradients into xy-space gradients
            dresultdx_simd = dresultds_simd * dsdx + dresultdt_simd * dtdx;
            dresultdy_simd = dresultds_simd * dsdy + dresultdt_simd * dtdy;
        }

        // NOTE: regardless of the value of "retVal" we will always copy over the texture system's results.
        // We are relying on the texture system properly filling in missing or fill colors

        // Per the OSL language specification
        // "The alpha channel (presumed to be the next channel following the channels returned by the texture() call)"
        // so despite the fact the alpha channel really is
        // we will always use +1 the final channel requested
        int alphaChannelIndex = 0;
        if (Masked<Color3>::is(resultRef)) {
            alphaChannelIndex = 3;
            Masked<Color3> result(resultRef);
            result[lane] = Color3(result_simd[0], result_simd[1],
                                  result_simd[2]);
            if (resultRef.has_derivs()) {
                MaskedDx<Color3> resultDx(resultRef);
                MaskedDy<Color3> resultDy(resultRef);

                resultDx[lane] = Color3(dresultdx_simd[0], dresultdx_simd[1],
                                        dresultdx_simd[2]);
                resultDy[lane] = Color3(dresultdy_simd[0], dresultdy_simd[1],
                                        dresultdy_simd[2]);
            }
        } else if (Masked<float>::is(resultRef)) {
            alphaChannelIndex = 1;
            Masked<float> result(resultRef);
            MaskedDx<float> resultDx(resultRef);
            MaskedDy<float> resultDy(resultRef);
            result[lane] = result_simd[0];
            if (resultRef.has_derivs()) {
                resultDx[lane] = dresultdx_simd[0];
                resultDy[lane] = dresultdy_simd[0];
            }
        }


        if (alphaIsValid) {
            Masked<float> alpha(alphaRef);
            alpha[lane] = result_simd[alphaChannelIndex];
            if (alphaRef.has_derivs()) {
                MaskedDx<float> alphaDx(alphaRef);
                MaskedDy<float> alphaDy(alphaRef);
                alphaDx[lane] = dresultdx_simd[alphaChannelIndex];
                alphaDy[lane] = dresultdy_simd[alphaChannelIndex];
            }
        }
        //std::cout << "s: " << s.get(i) << " t: " << t.get(i) << " color: " << resultColor << " " << wideResult.get(i) << std::endl;
        if (retVal) {
            status.set_on(lane);
        } else {
            std::string err = bsr->texturesys()->geterror();
            bool errMsgSize = err.size() > 0;
            if (errormessageIsValid) {
                Masked<ustring> errormessage(outputs.errormessage());
                if (errMsgSize) {
                    errormessage[lane] = ustring(err);
                } else {
                    errormessage[lane] = Strings::unknown;
                }
            } else if (errMsgSize) {
                context->batched<__OSL_WIDTH>().errorf(
                    Mask(Lane(lane)), "[RendererServices::texture] %s", err);
            }
        }
    });
    return status;
}

OSL_FORCEINLINE Mask
dispatch_texture(BatchedRendererServices* bsr, ustring filename,
                 TextureSystem::TextureHandle* texture_handle,
                 TextureSystem::Perthread* texture_thread_info,
                 const BatchedTextureOptions& options,
                 BatchedShaderGlobals* bsg, Wide<const float> s,
                 Wide<const float> t, Wide<const float> dsdx,
                 Wide<const float> dtdx, Wide<const float> dsdy,
                 Wide<const float> dtdy, BatchedTextureOutputs& outputs)
{
    if (bsr->is_overridden_texture()) {
        return bsr->texture(filename, texture_handle, texture_thread_info,
                            options, bsg, s, t, dsdx, dtdx, dsdy, dtdy,
                            outputs);
    } else {
        return default_texture(bsr, filename, texture_handle,
                               texture_thread_info, options, bsg, s, t, dsdx,
                               dtdx, dsdy, dtdy, outputs);
    }
}

Mask
default_texture3d(BatchedRendererServices* bsr, ustring filename,
                  TextureSystem::TextureHandle* texture_handle,
                  TextureSystem::Perthread* texture_thread_info,
                  const BatchedTextureOptions& options,
                  BatchedShaderGlobals* bsg, Wide<const Vec3> wP,
                  Wide<const Vec3> wdPdx, Wide<const Vec3> wdPdy,
                  BatchedTextureOutputs& outputs)
{
    Mask status(false);
    ASSERT(nullptr != bsg);
    ShadingContext* context = bsg->uniform.context;
    if (!texture_thread_info)
        texture_thread_info = context->texture_thread_info();
    if (!texture_handle)
        texture_handle
            = bsr->texturesys()->get_texture_handle(filename,
                                                    texture_thread_info);

    Mask mask = outputs.mask();

    MaskedData resultRef     = outputs.result();
    MaskedData alphaRef      = outputs.alpha();
    bool alphaIsValid        = outputs.alpha().valid();
    bool has_derivs          = resultRef.has_derivs() | alphaRef.has_derivs();
    bool errormessageIsValid = outputs.errormessage().valid();

    ASSERT(resultRef.valid());

    // Convert our BatchedTextureOptions to a single TextureOpt
    // and submit them 1 at a time through existing non-batched interface
    // Renderers could implement their own batched texturing,
    // although expected a future version of OIIO will support
    // this exact batched iterface
    const auto& uniform_opt = options.uniform;
    TextureOpt opt;
    // opt.time = ignoring (deprecated?)
    // opt.bias = ignoring (deprecated?)
    // opt.samples = ignoring (deprecated?)
    opt.firstchannel = uniform_opt.firstchannel;
    opt.subimage     = uniform_opt.subimage;
    opt.subimagename = uniform_opt.subimagename;
    opt.swrap        = (OIIO::TextureOpt::Wrap)uniform_opt.swrap;
    opt.twrap        = (OIIO::TextureOpt::Wrap)uniform_opt.twrap;
    opt.rwrap        = (OIIO::TextureOpt::Wrap)uniform_opt.rwrap;
    opt.mipmode      = (OIIO::TextureOpt::MipMode)uniform_opt.mipmode;
    opt.interpmode   = (OIIO::TextureOpt::InterpMode)uniform_opt.interpmode;
    opt.anisotropic  = uniform_opt.anisotropic;
    opt.conservative_filter = uniform_opt.conservative_filter;
    opt.fill                = uniform_opt.fill;
    opt.missingcolor        = uniform_opt.missingcolor;

    const auto& vary_opt = options.varying;

    mask.foreach ([=, &opt, &vary_opt, &outputs, &status](ActiveLane lane) {
        opt.sblur = vary_opt.sblur[lane];
        opt.tblur = vary_opt.tblur[lane];
        // For 3D volume texture lookups only:
        opt.rblur = vary_opt.rblur[lane];

        opt.swidth = vary_opt.swidth[lane];
        opt.twidth = vary_opt.twidth[lane];
        // For 3D volume texture lookups only:
        opt.rwidth = vary_opt.rwidth[lane];

#if 0
        // For debugging

        //std::cout << "BatchedRendererServices::texture[lane=" << lane << std::endl;
        std::cout << "BatchedRendererServices::texture:" << std::endl;
        std::cout << "P = " << (const Vec3)wP[lane] << std::endl;
        std::cout << "dPdx = " << (const Vec3)wdPdx[lane] << std::endl;
        std::cout << "dPdy = " << (const Vec3)wdPdy[lane] << std::endl;
        std::cout << "opt.wrap(s,r,t)=" << opt.swrap << "," << opt.twrap << "," << opt.rwrap << ")" << std::endl;
#endif

        // It's actually faster to ask for 4 channels (even if we need fewer)
        // and ensure that they're being put in aligned memory.
        // TODO:  investigate if the above statement is true when nchannels==1

        // NOTE: using simd::float4 to speedup texture transformation below
        OIIO::simd::float4 result_simd, dresultds_simd, dresultdt_simd,
            dresultdr_simd;

        bool retVal = false;

        const Vec3 dPdx = wdPdx[lane];
        const Vec3 dPdy = wdPdy[lane];

        retVal = bsr->texturesys()->texture3d(
            texture_handle, texture_thread_info, opt, wP[lane], dPdx, dPdy,
            Vec3(0), 4, (float*)&result_simd,
            has_derivs ? (float*)&dresultds_simd : nullptr,
            has_derivs ? (float*)&dresultdt_simd : nullptr,
            has_derivs ? (float*)&dresultdr_simd : nullptr);

        OIIO::simd::float4 dresultdx_simd;
        OIIO::simd::float4 dresultdy_simd;
        if (has_derivs) {
            // Correct our str texture space gradients into xyz-space gradients
            dresultdx_simd = dresultds_simd * dPdx.x + dresultdt_simd * dPdx.y
                             + dresultdr_simd * dPdx.z;
            dresultdy_simd = dresultds_simd * dPdy.x + dresultdt_simd * dPdy.y
                             + dresultdr_simd * dPdy.z;
        }

        // NOTE: regardless of the value of "retVal" we will always copy over the texture system's results.
        // We are relying on the texture system properly filling in missing or fill colors

        // Per the OSL language specification
        // "The alpha channel (presumed to be the next channel following the channels returned by the texture() call)"
        // so despite the fact the alpha channel really is
        // we will always use +1 the final channel requested
        int alphaChannelIndex = 0;
        if (Masked<Color3>::is(resultRef)) {
            alphaChannelIndex = 3;
            Masked<Color3> result(resultRef);
            result[lane] = Color3(result_simd[0], result_simd[1],
                                  result_simd[2]);
            if (resultRef.has_derivs()) {
                MaskedDx<Color3> resultDx(resultRef);
                MaskedDy<Color3> resultDy(resultRef);
                resultDx[lane] = Color3(dresultdx_simd[0], dresultdx_simd[1],
                                        dresultdx_simd[2]);
                resultDy[lane] = Color3(dresultdy_simd[0], dresultdy_simd[1],
                                        dresultdy_simd[2]);
            }
        } else if (Masked<float>::is(resultRef)) {
            alphaChannelIndex = 1;
            Masked<float> result(resultRef);
            result[lane] = result_simd[0];
            if (resultRef.has_derivs()) {
                MaskedDx<float> resultDx(resultRef);
                MaskedDy<float> resultDy(resultRef);
                resultDx[lane] = dresultdx_simd[0];
                resultDy[lane] = dresultdy_simd[0];
            }
        }


        if (alphaIsValid) {
            Masked<float> alpha(alphaRef);
            alpha[lane] = result_simd[alphaChannelIndex];
            if (alphaRef.has_derivs()) {
                MaskedDx<float> alphaDx(alphaRef);
                MaskedDy<float> alphaDy(alphaRef);
                alphaDx[lane] = dresultdx_simd[alphaChannelIndex];
                alphaDy[lane] = dresultdy_simd[alphaChannelIndex];
            }
        }

        //std::cout << "s: " << s.get(i) << " t: " << t.get(i) << " color: " << resultColor << " " << wideResult.get(i) << std::endl;
        if (retVal) {
            status.set_on(lane);
        } else {
            std::string err = bsr->texturesys()->geterror();
            bool errMsgSize = err.size() > 0;
            if (errormessageIsValid) {
                Masked<ustring> errormessage(outputs.errormessage());
                if (errMsgSize) {
                    errormessage[lane] = ustring(err);
                } else {
                    errormessage[lane] = Strings::unknown;
                }
            } else if (errMsgSize) {
                context->batched<__OSL_WIDTH>().errorf(
                    Mask(Lane(lane)), "[RendererServices::texture3d] %s", err);
            }
        }
    });
    return status;
}

OSL_FORCEINLINE Mask
dispatch_texture3d(BatchedRendererServices* bsr, ustring filename,
                   TextureSystem::TextureHandle* texture_handle,
                   TextureSystem::Perthread* texture_thread_info,
                   const BatchedTextureOptions& options,
                   BatchedShaderGlobals* bsg, Wide<const Vec3> P,
                   Wide<const Vec3> dPdx, Wide<const Vec3> dPdy,
                   BatchedTextureOutputs& outputs)
{
    if (bsr->is_overridden_texture3d()) {
        return bsr->texture3d(filename, texture_handle, texture_thread_info,
                              options, bsg, P, dPdx, dPdy, outputs);
    } else {
        return default_texture3d(bsr, filename, texture_handle,
                                 texture_thread_info, options, bsg, P, dPdx,
                                 dPdy, outputs);
    }
}


}  // namespace


OSL_BATCHOP int __OSL_MASKED_OP(texture)(
    void* bsg_, void* name, void* handle, const void* opt_, const void* s,
    const void* t, const void* dsdx, const void* dtdx, const void* dsdy,
    const void* dtdy, int chans, void* result, int resultHasDerivs, void* alpha,
    int alphaHasDerivs, void* errormessage, int mask_)
{
    Mask mask(mask_);
    OSL_ASSERT(!mask.all_off());

    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    auto& opt = *reinterpret_cast<const BatchedTextureOptions*>(opt_);

    // NOTE:  If overriden, BatchedRendererServiced::texture is responsible
    // for correcting our st texture space gradients into xy-space gradients
    BatchedTextureOutputs outputs(result, (bool)resultHasDerivs, chans, alpha,
                                  (bool)alphaHasDerivs, errormessage, mask);

    Mask retVal
        = dispatch_texture(bsg->uniform.renderer->batched(WidthTag()),
                           USTR(name), (TextureSystem::TextureHandle*)handle,
                           bsg->uniform.context->texture_thread_info(), opt,
                           bsg, Wide<const float>(s), Wide<const float>(t),
                           Wide<const float>(dsdx), Wide<const float>(dtdx),
                           Wide<const float>(dsdy), Wide<const float>(dtdy),
                           outputs);

    OSL_FORCEINLINE_BLOCK
    if (outputs.errormessage().valid()) {
        Masked<ustring> err(outputs.errormessage());
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int i = 0; i < __OSL_WIDTH; ++i) {
            if (retVal[i]) {
                err[i] = Strings::_emptystring_;
            }
        }
    }

    return retVal.value();
}



OSL_BATCHOP int __OSL_MASKED_OP(texture3d)(void* bsg_, void* name, void* handle,
                                           const void* opt_, const void* P,
                                           const void* Pdx, const void* Pdy,
                                           int chans, void* result,
                                           int resultHasDerivs, void* alpha,
                                           int alphaHasDerivs,
                                           void* errormessage, int mask_)
{
    Mask mask(mask_);
    ASSERT(!mask.all_off());

    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    auto& opt = *reinterpret_cast<const BatchedTextureOptions*>(opt_);

    BatchedTextureOutputs outputs(result, (bool)resultHasDerivs, chans, alpha,
                                  (bool)alphaHasDerivs, errormessage, mask);

    // NOTE:  If overriden, BatchedRendererServiced::texture is responsible
    // for correcting our str texture space gradients into xyz-space gradients
    Mask retVal
        = dispatch_texture3d(bsg->uniform.renderer->batched(WidthTag()),
                             USTR(name), (TextureSystem::TextureHandle*)handle,
                             bsg->uniform.context->texture_thread_info(), opt,
                             bsg, Wide<const Vec3>(P), Wide<const Vec3>(Pdx),
                             Wide<const Vec3>(Pdy), outputs);

    OSL_FORCEINLINE_BLOCK
    if (outputs.errormessage().valid()) {
        Masked<ustring> err(outputs.errormessage());
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int i = 0; i < __OSL_WIDTH; ++i) {
            if (retVal[i]) {
                err[i] = Strings::_emptystring_;
            }
        }
    }

    return retVal.value();
}



OSL_BATCHOP TextureSystem::TextureHandle*
    __OSL_OP(resolve_udim_uniform)(void* bsg_, const char* name, void* handle,
                                   float S, float T)
{
    // recreate TypeDesc
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);

    return bsg->uniform.renderer->batched(WidthTag())
        ->resolve_udim_uniform(bsg, bsg->uniform.context->texture_thread_info(),
                               USTR(name),
                               (RendererServices::TextureHandle*)handle, S, T);
}



OSL_BATCHOP void __OSL_MASKED_OP(resolve_udim)(void* bsg_, const char* name,
                                               void* handle, void* wS_,
                                               void* wT_, void* wResult_,
                                               int mask_value)
{
    // recreate TypeDesc
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);

    bsg->uniform.renderer->batched(WidthTag())
        ->resolve_udim(bsg, bsg->uniform.context->texture_thread_info(),
                       USTR(name), (RendererServices::TextureHandle*)handle,
                       Wide<const float>(wS_), Wide<const float>(wT_),
                       Masked<RendererServices::TextureHandle*>(
                           wResult_, Mask(mask_value)));
}



OSL_BATCHOP int __OSL_OP(get_textureinfo_uniform)(void* bsg_, const char* name,
                                                  void* handle, void* dataname,
                                                  const void* attr_type,
                                                  void* attr_dest)
{
    // recreate TypeDesc
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);

    RefData dest(*(const TypeDesc*)attr_type, false, attr_dest);

    bool retVal = bsg->uniform.renderer->batched(WidthTag())
                      ->get_texture_info_uniform(
                          bsg, bsg->uniform.context->texture_thread_info(),
                          USTR(name), (RendererServices::TextureHandle*)handle,
                          0 /*FIXME-ptex*/, USTR(dataname), dest);
    return retVal;
}


}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#include "undef_opname_macros.h"
