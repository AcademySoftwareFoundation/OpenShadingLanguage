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
using WidthTag = OSL::WidthOf<__OSL_WIDTH>;

#include "define_opname_macros.h"

namespace {


Mask
default_texture(BatchedRendererServices *bsr, ustring filename, TextureSystem::TextureHandle * texture_handle,
                                          TextureSystem::Perthread * texture_thread_info,
                                          const BatchedTextureOptions & options, BatchedShaderGlobals * bsg,
                                          Wide<const float> s, Wide<const float> t,
                                          Wide<const float> dsdx, Wide<const float> dtdx,
                                          Wide<const float> dsdy, Wide<const float> dtdy,
                                          BatchedTextureOutputs & outputs)
{
    Mask status(false);
    OSL_ASSERT(nullptr != bsg);
    ShadingContext *context = bsg->uniform.context;
    if (! texture_thread_info)
        texture_thread_info = context->texture_thread_info();
    if (! texture_handle)
        texture_handle = bsr->texturesys()->get_texture_handle (filename, texture_thread_info);

    Mask mask = outputs.mask();

    MaskedData resultRef = outputs.result();
    MaskedData alphaRef = outputs.alpha();
    bool alphaIsValid = outputs.alpha().valid();
    bool errormessageIsValid = outputs.errormessage().valid();
    bool has_derivs = resultRef.has_derivs() || alphaRef.has_derivs();

    OSL_ASSERT(resultRef.valid());

    // Convert our BatchedTextureOptions to a single TextureOpt
    // and submit them 1 at a time through existing non-batched interface
    // Renderers could implement their own batched texturing,
    // although expected a future version of OIIO will support
    // this exact batched interface
    const auto & uniform_opt = options.uniform;
    TextureOpt opt;
    // opt.time = ignoring (deprecated?)
    // opt.bias = ignoring (deprecated?)
    // opt.samples = ignoring (deprecated?)
    opt.firstchannel = uniform_opt.firstchannel;
    opt.subimage = uniform_opt.subimage;
    opt.subimagename = uniform_opt.subimagename;
    opt.swrap = (OIIO::TextureOpt::Wrap)uniform_opt.swrap;
    opt.twrap = (OIIO::TextureOpt::Wrap)uniform_opt.twrap;
    opt.rwrap = (OIIO::TextureOpt::Wrap)uniform_opt.rwrap;
    opt.mipmode = (OIIO::TextureOpt::MipMode)uniform_opt.mipmode;
    opt.interpmode = (OIIO::TextureOpt::InterpMode)uniform_opt.interpmode;
    opt.anisotropic = uniform_opt.anisotropic;
    opt.conservative_filter = uniform_opt.conservative_filter;
    opt.fill = uniform_opt.fill;
    opt.missingcolor = uniform_opt.missingcolor;

    const auto & vary_opt = options.varying;

    mask.foreach([=,&opt,&vary_opt,&outputs,&status](ActiveLane lane) {

        opt.sblur = vary_opt.sblur[lane];
        opt.tblur = vary_opt.tblur[lane];
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
        OIIO::simd::float4 result_simd, dresultds_simd, dresultdt_simd;
        // TODO:  investigate if there any magic to this simd::float4 or
        // can we just use a float[4] to the same effect and avoid confusion

        bool retVal = false;

        retVal = bsr->texturesys()->texture (texture_handle, texture_thread_info, opt,
                                        s[lane], t[lane],
                                        dsdx[lane], dtdx[lane],
                                        dsdy[lane], dtdy[lane],
                                        4,
                                        (float *)&result_simd,
                                        has_derivs ? (float *)&dresultds_simd : NULL,
                                        has_derivs ? (float *)&dresultdt_simd : NULL);

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
            result[lane] = Color3(result_simd[0], result_simd[1], result_simd[2]);
            if (resultRef.has_derivs()) {
                MaskedDx<Color3> resultDs(resultRef);
                MaskedDy<Color3> resultDt(resultRef);
                resultDs[lane] = Color3(dresultds_simd[0], dresultds_simd[1], dresultds_simd[2]);
                resultDt[lane] = Color3(dresultdt_simd[0], dresultdt_simd[1], dresultdt_simd[2]);
            }
        } else if (Masked<float>::is(resultRef)) {
            alphaChannelIndex = 1;
            Masked<float> result(resultRef);
            result[lane] = result_simd[0];
            if (resultRef.has_derivs()) {
                MaskedDx<float> resultDs(resultRef);
                MaskedDy<float> resultDt(resultRef);
                resultDs[lane] = dresultds_simd[0];
                resultDt[lane] = dresultdt_simd[0];
            }
        }
        if (alphaIsValid) {
            Masked<float> alpha(alphaRef);
            alpha[lane] = result_simd[alphaChannelIndex];
            if (alphaRef.has_derivs()) {
                MaskedDx<float> alphaDs(alphaRef);
                MaskedDy<float> alphaDt(alphaRef);
                alphaDs[lane] = dresultds_simd[alphaChannelIndex];
                alphaDt[lane] = dresultdt_simd[alphaChannelIndex];
            }
        }
        //std::cout << "s: " << s.get(i) << " t: " << t.get(i) << " color: " << resultColor << " " << wideResult.get(i) << std::endl;
        if(retVal) {
            status.set_on(lane);
        } else {
            std::string err = bsr->texturesys()->geterror();
            bool errMsgSize = err.size() > 0;
            if (errormessageIsValid) {
                Masked<ustring> errormessage(outputs.errormessage());
                if (errMsgSize) {
                    errormessage[lane] = ustring(err);
                }
                else {
                    errormessage[lane] = Strings::unknown;
                }
            }
            else if (errMsgSize) {
                context->batched<__OSL_WIDTH>().errorf (Mask(Lane(lane)), "[RendererServices::texture] %s", err);
            }
        }
    });
    return status;
}

OSL_FORCEINLINE Mask
dispatch_texture (BatchedRendererServices *bsr, ustring filename, TextureSystem::TextureHandle *texture_handle,
        TextureSystem::Perthread *texture_thread_info,
        const BatchedTextureOptions &options, BatchedShaderGlobals *bsg,
        Wide<const float> s, Wide<const float> t,
        Wide<const float> dsdx, Wide<const float> dtdx,
        Wide<const float> dsdy, Wide<const float> dtdy,
        BatchedTextureOutputs& outputs) {
    if (bsr->is_overridden_texture()) {
        return bsr->texture(filename, texture_handle,
                texture_thread_info,
                options, bsg,
                s, t,
                dsdx, dtdx,
                dsdy, dtdy,
                outputs);
    } else  {
        return default_texture(bsr, filename, texture_handle,
                texture_thread_info,
                options, bsg,
                s, t,
                dsdx, dtdx,
                dsdy, dtdy,
                outputs);
    }
}



static OSL_NOINLINE  void transformWideTextureGradients(BatchedTextureOutputs & outputs,
                                   Wide<const float> dsdx, Wide<const float> dsdy,
                                   Wide<const float> dtdx, Wide<const float> dtdy)
{
    MaskedData resultRef = outputs.result();
    // keep assert out of inlined code
    OSL_ASSERT(resultRef.valid());


    if (resultRef.has_derivs()) {
        if (Masked<float>::is(resultRef)) {
            OSL_FORCEINLINE_BLOCK
            {
                MaskedDx<float> drds(resultRef);
                MaskedDy<float> drdt(resultRef);

                OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
                for (int i = 0; i < __OSL_WIDTH; ++i) {
                    float drdsVal = drds[i];
                    float drdtVal = drdt[i];
                    float drdx = drdsVal * dsdx[i] +  drdtVal * dtdx[i];
                    float drdy = drdsVal * dsdy[i] +  drdtVal * dtdy[i];
                    drds[i] = drdx;
                    drdt[i] = drdy;
                }
            }
        }
        else {
            // keep assert out of inlined code
            OSL_ASSERT(Masked<Color3>::is(resultRef));
            OSL_FORCEINLINE_BLOCK
            {
                MaskedDx<Color3> widedrds(resultRef);
                MaskedDy<Color3> widedrdt(resultRef);
                OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
                for (int i = 0; i < __OSL_WIDTH; ++i) {
                    Color3 drdsColor = widedrds[i];
                    Color3 drdtColor = widedrdt[i];

                    widedrds[i] = drdsColor * dsdx[i] +  drdtColor * dtdx[i];
                    widedrdt[i] = drdsColor * dsdy[i] +  drdtColor * dtdy[i];
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
            float dadx = dadsVal * dsdx[i] +  dadtVal * dtdx[i];
            float dady = dadsVal * dsdy[i] +  dadtVal * dtdy[i];
            dads[i] = dadx;
            dadt[i] = dady;
        }
    }
}

} // namespace


OSL_BATCHOP int
__OSL_MASKED_OP(texture)
(   void *bsg_, void *name, void *handle,
    const void *opt_, const void *s, const void *t,
    const void *dsdx, const void *dtdx, const void *dsdy, const void *dtdy,
    int chans, void *result, int resultHasDerivs,
    void *alpha, int alphaHasDerivs,
    void *errormessage, int mask_)
{
    Mask mask(mask_);
    OSL_ASSERT(!mask.all_off());

    auto *bsg = reinterpret_cast<BatchedShaderGlobals *>(bsg_);
    auto &opt = *reinterpret_cast<const BatchedTextureOptions *>(opt_);

    BatchedTextureOutputs outputs(result, (bool)resultHasDerivs, chans,
                                  alpha, (bool)alphaHasDerivs,
                                  errormessage, mask);
    // TODO:  Original code use simd float4, then copy back to result.
    // for batched. Shouldn't render services make that decision?
    Mask retVal = dispatch_texture(bsg->uniform.renderer->batched(WidthTag()), USTR(name),
                                                              (TextureSystem::TextureHandle *)handle,
                                                              bsg->uniform.context->texture_thread_info(),
                                                              opt,
                                                              bsg,
                                                              Wide<const float>(s),
                                                              Wide<const float>(t),
                                                              Wide<const float>(dsdx),
                                                              Wide<const float>(dtdx),
                                                              Wide<const float>(dsdy),
                                                              Wide<const float>(dtdy),
                                                              outputs);

    // Correct our st texture space gradients into xy-space gradients
    if (resultHasDerivs | alphaHasDerivs) {
        transformWideTextureGradients(outputs,
                                      Wide<const float>(dsdx), Wide<const float>(dsdy),
                                      Wide<const float>(dtdx), Wide<const float>(dtdy));
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
__OSL_OP(resolve_udim_uniform)
(   void *bsg_, const char *name, void *handle,
    float S, float T)
{
    // recreate TypeDesc
    auto *bsg = reinterpret_cast<BatchedShaderGlobals *>(bsg_);

    return bsg->uniform.renderer->batched(WidthTag())->resolve_udim_uniform(
        bsg,
        bsg->uniform.context->texture_thread_info(),
        USTR(name),
        (RendererServices::TextureHandle *)handle,
        S,
        T);
}



OSL_BATCHOP void
__OSL_MASKED_OP(resolve_udim)
(   void *bsg_, const char *name, void *handle,
    void *wS_, void *wT_,
    void *wResult_, int mask_value)
{
    // recreate TypeDesc
    auto *bsg = reinterpret_cast<BatchedShaderGlobals *>(bsg_);

    bsg->uniform.renderer->batched(WidthTag())->resolve_udim(
        bsg,
        bsg->uniform.context->texture_thread_info(),
        USTR(name),
        (RendererServices::TextureHandle *)handle,
        Wide<const float>(wS_),
        Wide<const float>(wT_),
        Masked<RendererServices::TextureHandle *>(wResult_, Mask(mask_value)));
}



OSL_BATCHOP int
__OSL_OP(get_textureinfo_uniform)
(   void *bsg_, const char *name, void *handle,
    void *dataname,
    const void *attr_type,
    void *attr_dest)
{
    // recreate TypeDesc
    auto *bsg = reinterpret_cast<BatchedShaderGlobals *>(bsg_);

    RefData dest(*(const TypeDesc *)attr_type, false, attr_dest);

    bool retVal = bsg->uniform.renderer->batched(WidthTag())->get_texture_info_uniform(
        bsg,
        bsg->uniform.context->texture_thread_info(),
        USTR(name),
        (RendererServices::TextureHandle *)handle,
        0 /*FIXME-ptex*/,
        USTR(dataname), dest);
    return retVal;
}


} // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#include "undef_opname_macros.h"
