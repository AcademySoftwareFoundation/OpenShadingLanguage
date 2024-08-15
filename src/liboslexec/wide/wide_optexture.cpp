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

    MaskedData resultRef = outputs.result();
    MaskedData alphaRef  = outputs.alpha();
    bool has_derivs      = resultRef.has_derivs() || alphaRef.has_derivs();

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
        opt.rnd    = vary_opt.rnd[lane];

        // For 3D volume texture lookups only:
        //opt.rblur = vary_opt.rblur[lane];
        //opt.rwidth = vary_opt.rwidth[lane];

        // For debugging
        //std::cout << "BatchedRendererServices::texture[lane=" << lane << "opt = " << opt << std::endl;

        // It's actually faster to ask for 4 channels (even if we need fewer)
        // and ensure that they're being put in aligned memory.
        // TODO:  investigate if the above statement is true when nchannels==1
        // NOTE: using simd::vfloat4 to speedup texture transformation below
        OIIO::simd::vfloat4 result_simd, dresultds_simd, dresultdt_simd;

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

                resultDx[lane] = Color3(dresultds_simd[0], dresultds_simd[1],
                                        dresultds_simd[2]);
                resultDy[lane] = Color3(dresultdt_simd[0], dresultdt_simd[1],
                                        dresultdt_simd[2]);
            }
        } else if (Masked<float>::is(resultRef)) {
            alphaChannelIndex = 1;
            Masked<float> result(resultRef);
            MaskedDx<float> resultDx(resultRef);
            MaskedDy<float> resultDy(resultRef);
            result[lane] = result_simd[0];
            if (resultRef.has_derivs()) {
                resultDx[lane] = dresultds_simd[0];
                resultDy[lane] = dresultdt_simd[0];
            }
        }

        if (alphaRef.valid()) {
            Masked<float> alpha(alphaRef);
            alpha[lane] = result_simd[alphaChannelIndex];
            if (alphaRef.has_derivs()) {
                MaskedDx<float> alphaDx(alphaRef);
                MaskedDy<float> alphaDy(alphaRef);
                alphaDx[lane] = dresultds_simd[alphaChannelIndex];
                alphaDy[lane] = dresultdt_simd[alphaChannelIndex];
            }
        }
        //std::cout << "s: " << s.get(i) << " t: " << t.get(i) << " color: " << resultColor << " " << wideResult.get(i) << std::endl;
        if (retVal) {
            status.set_on(lane);
        } else {
            std::string err = bsr->texturesys()->geterror();
            bool errMsgSize = err.size() > 0;
            if (outputs.errormessage().valid()) {
                Masked<ustring> errormessage(outputs.errormessage());
                if (errMsgSize) {
                    errormessage[lane] = ustring(err);
                } else {
                    errormessage[lane] = Strings::unknown;
                }
            } else if (errMsgSize) {
                context->batched<__OSL_WIDTH>().errorfmt(
                    Mask(Lane(lane)), "[RendererServices::texture] {}", err);
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
                  Wide<const Vec3> wdPdz, BatchedTextureOutputs& outputs)
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

    MaskedData resultRef = outputs.result();
    MaskedData alphaRef  = outputs.alpha();
    bool has_derivs      = resultRef.has_derivs() | alphaRef.has_derivs();

    ASSERT(resultRef.valid());

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

        // NOTE: using simd::vfloat4 to speedup texture transformation below
        OIIO::simd::vfloat4 result_simd, dresultds_simd, dresultdt_simd,
            dresultdr_simd;

        bool retVal = false;

        const Vec3 dPdx = wdPdx[lane];
        const Vec3 dPdy = wdPdy[lane];
        const Vec3 dPdz = wdPdz[lane];

        retVal = bsr->texturesys()->texture3d(
            texture_handle, texture_thread_info, opt, Vec3(wP[lane]),
            Vec3(dPdx), Vec3(dPdy), dPdz, 4, (float*)&result_simd,
            has_derivs ? (float*)&dresultds_simd : nullptr,
            has_derivs ? (float*)&dresultdt_simd : nullptr,
            has_derivs ? (float*)&dresultdr_simd : nullptr);

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
                resultDx[lane] = Color3(dresultds_simd[0], dresultds_simd[1],
                                        dresultds_simd[2]);
                resultDy[lane] = Color3(dresultdt_simd[0], dresultdt_simd[1],
                                        dresultdt_simd[2]);
            }
        } else if (Masked<float>::is(resultRef)) {
            alphaChannelIndex = 1;
            Masked<float> result(resultRef);
            result[lane] = result_simd[0];
            if (resultRef.has_derivs()) {
                MaskedDx<float> resultDx(resultRef);
                MaskedDy<float> resultDy(resultRef);
                resultDx[lane] = dresultds_simd[0];
                resultDy[lane] = dresultdt_simd[0];
            }
        }


        if (alphaRef.valid()) {
            Masked<float> alpha(alphaRef);
            alpha[lane] = result_simd[alphaChannelIndex];
            if (alphaRef.has_derivs()) {
                MaskedDx<float> alphaDx(alphaRef);
                MaskedDy<float> alphaDy(alphaRef);
                alphaDx[lane] = dresultds_simd[alphaChannelIndex];
                alphaDy[lane] = dresultdt_simd[alphaChannelIndex];
            }
        }

        //std::cout << "s: " << s.get(i) << " t: " << t.get(i) << " color: " << resultColor << " " << wideResult.get(i) << std::endl;
        if (retVal) {
            status.set_on(lane);
        } else {
            std::string err = bsr->texturesys()->geterror();
            bool errMsgSize = err.size() > 0;
            if (outputs.errormessage().valid()) {
                Masked<ustring> errormessage(outputs.errormessage());
                if (errMsgSize) {
                    errormessage[lane] = ustring(err);
                } else {
                    errormessage[lane] = Strings::unknown;
                }
            } else if (errMsgSize) {
                context->batched<__OSL_WIDTH>().errorfmt(
                    Mask(Lane(lane)), "[RendererServices::texture3d] {}", err);
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
                   Wide<const Vec3> dPdz, BatchedTextureOutputs& outputs)
{
    if (bsr->is_overridden_texture3d()) {
        return bsr->texture3d(filename, texture_handle, texture_thread_info,
                              options, bsg, P, dPdx, dPdy, dPdz, outputs);
    } else {
        return default_texture3d(bsr, filename, texture_handle,
                                 texture_thread_info, options, bsg, P, dPdx,
                                 dPdy, dPdz, outputs);
    }
}



Mask
default_environment(BatchedRendererServices* bsr, ustring filename,
                    TextureSystem::TextureHandle* texture_handle,
                    TextureSystem::Perthread* texture_thread_info,
                    const BatchedTextureOptions& options,
                    BatchedShaderGlobals* bsg, Wide<const Vec3> wR,
                    Wide<const Vec3> wdRdx, Wide<const Vec3> wdRdy,
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

    MaskedData resultRef = outputs.result();
    MaskedData alphaRef  = outputs.alpha();

    ASSERT(resultRef.valid());

    // Convert our BatchedTextureOptions to a single TextureOpt
    // and submit them 1 at a time through existing non-batched interface
    // Renderers could implement their own batched environment,
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
        opt.sblur = vary_opt.sblur[lane];
        opt.tblur = vary_opt.tblur[lane];
        // For 3D volume texture lookups only:
        // opt.rblur = vary_opt.rblur[lane];

        opt.swidth = vary_opt.swidth[lane];
        opt.twidth = vary_opt.twidth[lane];
        // For 3D volume texture lookups only:
        // opt.rwidth = vary_opt.rwidth[lane];

        // It's actually faster to ask for 4 channels (even if we need fewer)
        // and ensure that they're being put in aligned memory.
        // TODO:  investigate if the above statement is true when nchannels==1

        // NOTE: using simd::vfloat4 to speedup texture transformation below
        OIIO::simd::vfloat4 result_simd;

        bool retVal = false;

        const Vec3 dPdx = wdRdx[lane];
        const Vec3 dPdy = wdRdy[lane];

        retVal = bsr->texturesys()->environment(
            texture_handle, texture_thread_info, opt, Vec3(wR[lane]),
            Vec3(dPdx), Vec3(dPdy), 4, (float*)&result_simd, nullptr, nullptr);

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
        } else if (Masked<float>::is(resultRef)) {
            alphaChannelIndex = 1;
            Masked<float> result(resultRef);
            result[lane] = result_simd[0];
        }

        if (alphaRef.valid()) {
            Masked<float> alpha(alphaRef);
            alpha[lane] = result_simd[alphaChannelIndex];
        }

        //std::cout << "s: " << s.get(i) << " t: " << t.get(i) << " color: " << resultColor << " " << wideResult.get(i) << std::endl;
        if (retVal) {
            status.set_on(lane);
        } else {
            std::string err = bsr->texturesys()->geterror();
            bool errMsgSize = err.size() > 0;
            if (outputs.errormessage().valid()) {
                Masked<ustring> errormessage(outputs.errormessage());
                if (errMsgSize) {
                    errormessage[lane] = ustring(err);
                } else {
                    errormessage[lane] = Strings::unknown;
                }
            } else if (errMsgSize) {
                context->batched<__OSL_WIDTH>().errorfmt(
                    Mask(Lane(lane)), "[RendererServices::environment] {}",
                    err);
            }
        }
    });
    return status;
}



OSL_FORCEINLINE Mask
dispatch_environment(BatchedRendererServices* bsr, ustring filename,
                     TextureSystem::TextureHandle* texture_handle,
                     TextureSystem::Perthread* texture_thread_info,
                     const BatchedTextureOptions& options,
                     BatchedShaderGlobals* bsg, Wide<const Vec3> R,
                     Wide<const Vec3> dRdx, Wide<const Vec3> dRdy,
                     BatchedTextureOutputs& outputs)
{
    if (bsr->is_overridden_texture3d()) {
        return bsr->environment(filename, texture_handle, texture_thread_info,
                                options, bsg, R, dRdx, dRdy, outputs);
    } else {
        return default_environment(bsr, filename, texture_handle,
                                   texture_thread_info, options, bsg, R, dRdx,
                                   dRdy, outputs);
    }
}


}  // namespace


static OSL_NOINLINE void
transformWideTextureGradients(BatchedTextureOutputs& outputs,
                              Wide<const float> dsdx, Wide<const float> dtdx,
                              Wide<const float> dsdy, Wide<const float> dtdy)
{
    MaskedData resultRef = outputs.result();
    if (resultRef.valid() && resultRef.has_derivs()) {
        if (Masked<float>::is(resultRef)) {
            OSL_FORCEINLINE_BLOCK
            {
                MaskedDx<float> drds(resultRef);
                MaskedDy<float> drdt(resultRef);

                OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
                for (int i = 0; i < __OSL_WIDTH; ++i) {
                    float drdsVal = drds[i];
                    float drdtVal = drdt[i];
                    float drdx    = drdsVal * dsdx[i] + drdtVal * dtdx[i];
                    float drdy    = drdsVal * dsdy[i] + drdtVal * dtdy[i];
                    drds[i]       = drdx;
                    drdt[i]       = drdy;
                }
            }
        } else {
            // keep assert out of inlined code
            OSL_DASSERT(Masked<Color3>::is(resultRef));
            OSL_FORCEINLINE_BLOCK
            {
                //printf("doint color\n");
                MaskedDx<Color3> widedrds(resultRef);
                MaskedDy<Color3> widedrdt(resultRef);
                OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
                for (int i = 0; i < __OSL_WIDTH; ++i) {
                    Color3 drdsColor = widedrds[i];
                    Color3 drdtColor = widedrdt[i];

                    widedrds[i] = drdsColor * dsdx[i] + drdtColor * dtdx[i];
                    widedrdt[i] = drdsColor * dsdy[i] + drdtColor * dtdy[i];
                }
            }
        }
    }

    MaskedData alphaRef = outputs.alpha();
    OSL_FORCEINLINE_BLOCK
    if (alphaRef.valid() && alphaRef.has_derivs()) {
        MaskedDx<float> dads(alphaRef);
        MaskedDy<float> dadt(alphaRef);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int i = 0; i < __OSL_WIDTH; ++i) {
            float dadsVal = dads[i];
            float dadtVal = dadt[i];
            float dadx    = dadsVal * dsdx[i] + dadtVal * dtdx[i];
            float dady    = dadsVal * dsdy[i] + dadtVal * dtdy[i];
            dads[i]       = dadx;
            dadt[i]       = dady;
        }
    }
}

static OSL_NOINLINE void
transformWideTextureGradientsTexture3d(BatchedTextureOutputs& outputs,
                                       Wide<const Vec3> Pdx,
                                       Wide<const Vec3> Pdy,
                                       Wide<const Vec3> Pdz)
{
    MaskedData resultRef = outputs.result();
    if (resultRef.valid() && resultRef.has_derivs()) {
        if (Masked<float>::is(resultRef)) {
            OSL_FORCEINLINE_BLOCK
            {
                MaskedDx<float> drds(resultRef);
                MaskedDy<float> drdt(resultRef);
                //MaskedDz<float> drdr(resultRef); // our duals don't actually have space for this

                OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
                for (int i = 0; i < __OSL_WIDTH; ++i) {
                    float dres_xVal = drds[i];
                    float dres_yVal = drdt[i];
                    //float dres_zVal = drdr[i];

                    Vec3 v3pdx = Pdx[i];
                    Vec3 v3pdy = Pdy[i];
                    //Vec3 v3pdz = Pdz[i];

                    float dres_x = dres_xVal * v3pdx.x
                                   + dres_yVal
                                         * v3pdx.y;  // + dres_zVal * v3pdx.z;
                    float dres_y = dres_xVal * v3pdy.x
                                   + dres_yVal
                                         * v3pdy.y;  // + dres_zVal * v3pdy.z;
                    //float dres_z = dres_xVal * v3pdz.x + dres_yVal * v3pdz.y + dres_zVal * v3pdz.z;

                    drds[i] = dres_x;
                    drdt[i] = dres_y;
                    //drdr[i] = dres_z;
                }
            }
        } else {
            // keep assert out of inlined code
            OSL_DASSERT(Masked<Color3>::is(resultRef));
            OSL_FORCEINLINE_BLOCK
            {
                MaskedDx<Color3> widedrp1(resultRef);
                MaskedDy<Color3> widedrp2(resultRef);
                //MaskedDz<Color3> widedrp3(resultRef);

                OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
                for (int i = 0; i < __OSL_WIDTH; ++i) {
                    Color3 drdp1Color = widedrp1[i];
                    Color3 drdp2Color = widedrp2[i];
                    //Color3 drdp3Color = widedrp3[i];

                    Vec3 v3pdx = Pdx[i];
                    Vec3 v3pdy = Pdy[i];
                    //Vec3 v3pdz = Pdz[i];

                    widedrp1[i] = drdp1Color * v3pdx.x
                                  + drdp2Color
                                        * v3pdx.y;  // + drdp3Color * v3pdx.z;
                    widedrp2[i] = drdp1Color * v3pdy.x
                                  + drdp2Color
                                        * v3pdy.y;  // + drdp3Color * v3pdy.z;
                    //widedrp3[i] = drdp1Color * v3pdz.x +  drdp2Color * v3pdz.y + drdp3Color * v3pdz.z;
                }
            }
        }
    }

    MaskedData alphaRef = outputs.alpha();
    OSL_FORCEINLINE_BLOCK
    if (alphaRef.valid() && alphaRef.has_derivs()) {
        MaskedDx<float> dap1(alphaRef);
        MaskedDy<float> dap2(alphaRef);
        // MaskedDz<float> dap3(alphaRef);

        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int i = 0; i < __OSL_WIDTH; ++i) {
            float dadp1Val = dap1[i];
            float dadp2Val = dap2[i];
            //float dadp3Val = dap3[i];

            Vec3 v3pdx = Pdx[i];
            Vec3 v3pdy = Pdy[i];
            //Vec3 v3pdz = Pdz[i];

            float dadpx = dadp1Val * v3pdx.x
                          + dadp2Val * v3pdx.y;  // + dadp3Val * v3pdx.z;
            float dadpy = dadp1Val * v3pdy.x
                          + dadp2Val * v3pdy.y;  // + dadp3Val * v3pdy.z;
            //float dadpz = dadp1Val * v3pdz.x + dadp2Val * v3pdz.y + dadp3Val * v3pdz.z;

            dap1[i] = dadpx;
            dap2[i] = dadpy;
            //dap3[i] = dadpz;
        }
    }
}

OSL_BATCHOP int
__OSL_MASKED_OP(texture)(void* bsg_, void* name, void* handle, const void* opt_,
                         const void* s, const void* t, const void* dsdx,
                         const void* dtdx, const void* dsdy, const void* dtdy,
                         int chans, void* result, int resultHasDerivs,
                         void* alpha, int alphaHasDerivs, void* errormessage,
                         int mask_)
{
    Mask mask(mask_);
    OSL_ASSERT(!mask.all_off());

    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    auto& opt = *reinterpret_cast<const BatchedTextureOptions*>(opt_);

    // NOTE:  If overridden, BatchedRendererServiced::texture is responsible
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

    // Correct our st texture space gradients into xy-space gradients
    if (resultHasDerivs || alphaHasDerivs) {
        transformWideTextureGradients(outputs, Wide<const float>(dsdx),
                                      Wide<const float>(dtdx),
                                      Wide<const float>(dsdy),
                                      Wide<const float>(dtdy));
    }

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



OSL_BATCHOP int
__OSL_MASKED_OP(texture3d)(void* bsg_, void* name, void* handle,
                           const void* opt_, const void* wP, const void* wPdx,
                           const void* wPdy, const void* wPdz, int chans,
                           void* result, int resultHasDerivs, void* alpha,
                           int alphaHasDerivs, void* errormessage, int mask_)
{
    Mask mask(mask_);
    ASSERT(!mask.all_off());

    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    auto& opt = *reinterpret_cast<const BatchedTextureOptions*>(opt_);

    BatchedTextureOutputs outputs(result, (bool)resultHasDerivs, chans, alpha,
                                  (bool)alphaHasDerivs, errormessage, mask);

    Block<Vec3> blockPdz;
    if (wPdz == nullptr) {
        assign_all(blockPdz, Vec3(0.0f));
        wPdz = &blockPdz;
    }
    // NOTE:  If overridden, BatchedRendererServiced::texture is responsible
    // for correcting our str texture space gradients into xyz-space gradients
    Mask retVal
        = dispatch_texture3d(bsg->uniform.renderer->batched(WidthTag()),
                             USTR(name), (TextureSystem::TextureHandle*)handle,
                             bsg->uniform.context->texture_thread_info(), opt,
                             bsg, Wide<const Vec3>(wP), Wide<const Vec3>(wPdx),
                             Wide<const Vec3>(wPdy), Wide<const Vec3>(wPdz),
                             outputs);

    // Correct our P (Vec3) space gradients into xyz-space gradients
    if (resultHasDerivs || alphaHasDerivs) {
        transformWideTextureGradientsTexture3d(outputs, Wide<const Vec3>(wPdx),
                                               Wide<const Vec3>(wPdy),
                                               Wide<const Vec3>(wPdz));
    }

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


OSL_BATCHOP int
__OSL_MASKED_OP(environment)(void* bsg_, void* name, void* handle,
                             const void* opt_, const void* wR, const void* wRdx,
                             const void* wRdy, int chans, void* result,
                             int resultHasDerivs, void* alpha,
                             int alphaHasDerivs, void* errormessage, int mask_)
{
    Mask mask(mask_);
    ASSERT(!mask.all_off());

    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    auto& opt = *reinterpret_cast<const BatchedTextureOptions*>(opt_);

    BatchedTextureOutputs outputs(result, (bool)resultHasDerivs, chans, alpha,
                                  (bool)alphaHasDerivs, errormessage, mask);

    // NOTE:  If overridden, BatchedRendererServiced::texture is responsible
    // for correcting our str texture space gradients into xyz-space gradients
    Mask retVal = dispatch_environment(
        bsg->uniform.renderer->batched(WidthTag()), USTR(name),
        (TextureSystem::TextureHandle*)handle,
        bsg->uniform.context->texture_thread_info(), opt, bsg,
        Wide<const Vec3>(wR), Wide<const Vec3>(wRdx), Wide<const Vec3>(wRdy),
        outputs);

    // For now, just zero out the result derivatives.  If somebody needs
    // derivatives of environment lookups, we'll fix it.  The reason
    // that this is a pain is that OIIO's environment call (unwisely?)
    // returns the st gradients, but we want the xy gradients, which is
    // tricky because we (this function you're reading) don't know which
    // projection is used to generate st from R.  Ugh.  Sweep under the
    // rug for a day when somebody is really asking for it.
    auto resultRef = outputs.result();
    if (resultRef.has_derivs()) {
        if (Masked<Color3>::is(resultRef)) {
            MaskedDx<Color3> resultDx(resultRef);
            MaskedDy<Color3> resultDy(resultRef);
            assign_all(resultDx, Color3(0.0f));
            assign_all(resultDy, Color3(0.0f));
        } else if (Masked<float>::is(resultRef)) {
            MaskedDx<float> resultDx(resultRef);
            MaskedDy<float> resultDy(resultRef);
            assign_all(resultDx, 0.0f);
            assign_all(resultDy, 0.0f);
        }
    }
    auto alphaRef = outputs.alpha();
    if (alphaRef.valid() && alphaRef.has_derivs()) {
        MaskedDx<float> alphaDx(alphaRef);
        MaskedDy<float> alphaDy(alphaRef);
        assign_all(alphaDx, 0.0f);
        assign_all(alphaDy, 0.0f);
    }

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



OSL_BATCHOP void
__OSL_MASKED_OP(resolve_udim)(void* bsg_, const char* name, void* handle,
                              void* wS_, void* wT_, void* wResult_,
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



OSL_BATCHOP int
__OSL_OP(get_textureinfo_uniform)(void* bsg_, const char* name, void* handle,
                                  void* dataname, const void* attr_type,
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
