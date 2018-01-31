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
#include <OSL/Imathx.h>

using namespace OSL;
using namespace OSL::pvt;

#include <OpenImageIO/strutil.h>
#include <OpenImageIO/dassert.h>
#include <OpenImageIO/filesystem.h>


OSL_NAMESPACE_ENTER



namespace { // anonymous
std::ostream& operator<<(std::ostream& os, const TextureOpt& opt)  
{     
	os << "Updated Texture Options:" << std::endl;
	os << "firstchannel =" << opt.firstchannel << std::endl;
	os << "subimage =" << opt.subimage << std::endl;
	os << "subimagename =" << opt.subimagename << std::endl;
	os << "swrap =" << opt.swrap << std::endl;
	os << "twrap =" << opt.twrap << std::endl;
	os << "mipmode =" << opt.mipmode << std::endl;
	os << "interpmode =" << opt.interpmode << std::endl;
	os << "anisotropic =" << opt.anisotropic << std::endl;
	os << "conservative_filter =" << opt.conservative_filter << std::endl;
	os << "sblur =" << opt.sblur << std::endl;
	os << "tblur =" << opt.tblur << std::endl;
	os << "swidth =" << opt.swidth << std::endl;
	os << "twidth =" << opt.twidth << std::endl;
	os << "fill =" << opt.fill << std::endl;
	os << "missingcolor =" << opt.missingcolor << std::endl;
	os << "time =" << opt.time << std::endl;
	os << "bias =" << opt.bias << std::endl;
	os << "samples =" << opt.samples << std::endl;
	os << "rwrap =" << opt.rwrap << std::endl;
	os << "rblur =" << opt.rblur << std::endl;
	os << "rwidth =" << opt.rwidth << std::endl;
    
    return os;  
}  
} // anonymous namespace

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

BatchedRendererServices *
RendererServices::batched()
{
	// No default implementation for batched services
    return nullptr;
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


BatchedRendererServices::BatchedRendererServices (TextureSystem *texsys)
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


OSL_INLINE static void invert_wide_matrix(MaskedAccessor<Matrix44> result, ConstWideAccessor<Matrix44> wmatrix)
{
	if (result.mask().any_on()) {
		int allAreAffine = 1;
		OSL_OMP_PRAGMA(omp simd simdlen(wmatrix.width))
		for(int lane=0; lane < wmatrix.width; ++lane) {
			Matrix44 m = wmatrix[lane];
			if (result.mask().is_on(lane) &&
			    (m.x[0][3] != 0.0f || m.x[1][3] != 0.0f || m.x[2][3] != 0.0f || m.x[3][3] != 1.0f)) {
				allAreAffine = 0;
			}
		}
		
		if (allAreAffine) {
			OSL_INTEL_PRAGMA(omp simd simdlen(wmatrix.width))
			for(int lane=0; lane < wmatrix.width; ++lane) {    
				Matrix44 m = wmatrix[lane];
				//bool ok = get_matrix (sgb, r, xform.get(lane), time.get(lane));
				//r.invert();
				Matrix44 r = OSL::affineInvert(m);
				result[lane] = r;
			}
		} else
		{
			// Scalar code for non affine matrix (well at least 1 lane isn't)
			for(int lane=0; lane < SimdLaneCount; ++lane) {
				if (result.mask().is_on(lane)) {
					Matrix44 r = wmatrix[lane];
					r.invert();
					result[lane] = r;
				}
			}			
		}
	}
}

Mask
BatchedRendererServices::get_inverse_matrix (ShaderGlobalsBatch *sgb, MaskedAccessor<Matrix44> result,
		ConstWideAccessor<TransformationPtr> xform, ConstWideAccessor<float> time)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		Wide<Matrix44> wmatrix;
		Mask succeeded = get_matrix (sgb, MaskedAccessor<Matrix44>(wmatrix, result.mask()), xform, time);
		invert_wide_matrix(result&succeeded, wmatrix);
	    return succeeded;	
	}
}


bool
BatchedRendererServices::get_inverse_matrix (ShaderGlobalsBatch *sgb, Matrix44 &result,
                                      TransformationPtr xform)
{
    bool ok = get_matrix (sgb, result, xform);
    if (ok)
        result.invert ();
    return ok;
}



Mask
BatchedRendererServices::get_inverse_matrix (ShaderGlobalsBatch *sgb,  MaskedAccessor<Matrix44> result,
                                      ustring to, ConstWideAccessor<float> time)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		Wide<Matrix44> wmatrix;
		Mask succeeded = get_matrix (sgb, MaskedAccessor<Matrix44>(wmatrix, result.mask()), to, time);
		invert_wide_matrix(result&succeeded, wmatrix);
	    return succeeded;
	}    
}

Mask
BatchedRendererServices::get_inverse_matrix (ShaderGlobalsBatch *sgb,  MaskedAccessor<Matrix44> result,
		ConstWideAccessor<ustring> to, ConstWideAccessor<float> time)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		Wide<Matrix44> wmatrix;
		Mask succeeded = get_matrix (sgb, MaskedAccessor<Matrix44>(wmatrix,result.mask()), to, time);
		invert_wide_matrix(result& succeeded, wmatrix);
	    return succeeded;
	}
}



bool
BatchedRendererServices::get_inverse_matrix (ShaderGlobalsBatch *sgb, Matrix44 &result,
                                      ustring to)
{
    bool ok = get_matrix (sgb, result, to);
    if (ok)
        result.invert ();
    return ok;
}

TextureSystem *
BatchedRendererServices::texturesys () const
{
    return m_texturesys;
}

Mask
BatchedRendererServices::get_texture_info (ShaderGlobalsBatch *sgb,
										   ConstWideAccessor<ustring> filename,
                                           int subimage,
                                           ustring dataname,
                                           MaskedDataRef val)
{
    Mask success(false);

#define TEXTURE_INFO_FOR_TYPE(data_type)                                                        \
    if (val.is<data_type>()) {                                                                  \
        auto out = val.masked<data_type>();                                                     \
        for (int l = 0; l < out.width; ++l) {                                                   \
        	if(val.mask()[l]) {                                                                 \
				data_type data;                                                                 \
				bool status = texturesys()->get_texture_info (filename[l], subimage,        \
															  dataname, val.type(), &data);     \
															  success.set(l, status);           \
				if (status) {                                                                   \
					/* masked assignment */                                                     \
					out[l] = data;                                                              \
				} else {                                                                        \
					std::string err = texturesys()->geterror();                                 \
					if (err.size() && sgb) {                                                    \
						sgb->uniform().context->error (Mask(Lane(l)), "[BatchRendererServices::get_texture_info] %s", err);\
					}                                                                           \
				}                                                                               \
			}                                                                                   \
        }                                                                                       \
        return success;                                                                         \
    }

#define TEXTURE_INFO_FOR_ARRAY(data_type)                                                       \
    if (val.is<data_type[]>()) {                                                                \
        auto out = val.masked<data_type[]>();                                                   \
        for (int l = 0; l < out.width; ++l) {                                                   \
        	if(val.mask()[l]) {                                                                 \
				auto arrayData = out[l];                                                        \
				data_type data[arrayData.length()];                                             \
				bool status = texturesys()->get_texture_info (filename[l], subimage,        \
															  dataname, val.type(), data);      \
				success.set(l, status);                                                         \
				if (status) {                                                                   \
					/* masked assignment */                                                     \
					for (int i = 0; i < arrayData.length(); ++i) {                              \
						arrayData[i] = data[i];                                                 \
					}                                                                           \
				} else {                                                                        \
					std::string err = texturesys()->geterror();                                 \
					if (err.size() && sgb) {                                                    \
						sgb->uniform().context->error (Mask(Lane(l)), "[BatchRendererServices::get_texture_info] %s", err);\
					}                                                                           \
				}                                                                               \
			}                                                                                   \
        }                                                                                       \
        return success;                                                                         \
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

bool
BatchedRendererServices::get_texture_info_uniform (ShaderGlobalsBatch *sgb, ustring filename,
                                                   TextureHandle *texture_handle,
                                                   int subimage,
                                                   ustring dataname,
                                                   DataRef val)
{
    bool status;
    if (texture_handle)
        status = texturesys()->get_texture_info (texture_handle, NULL, subimage,
                                                 dataname, val.type(), val.ptr());
    else
        status = texturesys()->get_texture_info (filename, subimage,
                                                 dataname, val.type(), val.ptr());
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sgb) {
            sgb->uniform().context->error ("[BatchRendererServices::get_texture_info_uniform] %s", err);
        }
    }
    return status;
}

#ifdef OSL_EXPERIMENTAL_BATCHED_TEXTURE

Mask
BatchedRendererServices::texture(ustring filename, TextureHandle * texture_handle,
                                          TexturePerthread * texture_thread_info,
                                          const BatchedTextureOptions & options, ShaderGlobalsBatch * sgb,
                                          ConstWideAccessor<float> s, ConstWideAccessor<float> t,
                                          ConstWideAccessor<float> dsdx, ConstWideAccessor<float> dtdx,
                                          ConstWideAccessor<float> dsdy, ConstWideAccessor<float> dtdy,
                                          BatchedTextureOutputs & outputs)
{
    Mask status(false);
    ASSERT(nullptr != sgb);
    ShadingContext *context = sgb->uniform().context;
    if (! texture_thread_info)
        texture_thread_info = context->texture_thread_info();

    Mask mask = outputs.mask();

    MaskedDataRef resultRef = outputs.result();
    bool has_derivs = resultRef.has_derivs();
    MaskedDataRef alphaRef = outputs.alpha();
    bool alphaIsValid = outputs.alpha().valid();
    bool errormessageIsValid = outputs.errormessage().valid();

    ASSERT(resultRef.valid());

    // Convert our BatchedTextureOptions to a single TextureOpt
    // and submit them 1 at a time through existing non-batched interface
    // Renderers could implement their own batched texturing,
    // although expected a future version of OIIO will support
    // this exact batched iterface
    const auto & uniform_opt = options.uniform;
    TextureOpt opt;
    // opt.time = ignoring (deprecated?)
    // opt.bias = ignoring (deprecated?)
    // opt.samples = ignoring (deprecated?)
    opt.firstchannel = uniform_opt.firstchannel;
    opt.subimage = uniform_opt.subimage;
    opt.subimagename = uniform_opt.subimagename;
    opt.swrap = (OpenImageIO::v1_7::TextureOpt::Wrap)uniform_opt.swrap;
    opt.twrap = (OpenImageIO::v1_7::TextureOpt::Wrap)uniform_opt.twrap;
    opt.rwrap = (OpenImageIO::v1_7::TextureOpt::Wrap)uniform_opt.rwrap;
    opt.mipmode = (OpenImageIO::v1_7::TextureOpt::MipMode)uniform_opt.mipmode;
    opt.interpmode = (OpenImageIO::v1_7::TextureOpt::InterpMode)uniform_opt.interpmode;
    opt.anisotropic = uniform_opt.anisotropic;
    opt.conservative_filter = uniform_opt.conservative_filter;
    opt.fill = uniform_opt.fill;
    opt.missingcolor = uniform_opt.missingcolor;

    const auto & vary_opt = options.varying;

    for (int i = 0; i < SimdLaneCount; ++i) {
        if (mask[i]) {

            opt.sblur = vary_opt.sblur.get(i);
            opt.tblur = vary_opt.tblur.get(i);
            opt.swidth = vary_opt.swidth.get(i);
            opt.twidth = vary_opt.twidth.get(i);

            // For 3D volume texture lookups only:
            //opt.rblur = vary_opt.rblur.get(i);
            //opt.rwidth = vary_opt.rwidth.get(i);

            // For debugging
            //std::cout << "BatchedRendererServices::texture[lane=" << i << "opt = " << opt << std::endl;

            // It's actually faster to ask for 4 channels (even if we need fewer)
            // and ensure that they're being put in aligned memory.
            // TODO:  investigate if the above statement is true when nchannels==1
            OIIO::simd::float4 result_simd, dresultds_simd, dresultdt_simd;
            // TODO:  investigate if there any magic to this simd::float4 or
            // can we just use a float[4] to the same effect and avoid confusion

            bool retVal = false;

            if (texture_handle) {
                retVal = texturesys()->texture (texture_handle, texture_thread_info, opt,
                                                s[i], t[i],
                                                dsdx[i], dtdx[i],
                                                dsdy[i], dtdy[i],
                                                4,
                                                (float *)&result_simd,
                                                has_derivs ? (float *)&dresultds_simd : NULL,
                                                has_derivs ? (float *)&dresultdt_simd : NULL);
            }
            else {
                retVal = texturesys()->texture (filename, opt,
                                                s[i], t[i],
                                                dsdx[i], dtdx[i],
                                                dsdy[i], dtdy[i],
                                                4,
                                                (float *)&result_simd,
                                                has_derivs ? (float *)&dresultds_simd : NULL,
                                                has_derivs ? (float *)&dresultdt_simd : NULL);
            }

            if (retVal) {
                // Per the OSL language specification
                // "The alpha channel (presumed to be the next channel following the channels returned by the texture() call)"
                // so despite the fact the alpha channel really is
                // we will always use +1 the final channel requested
                int alphaChannelIndex;
                if (resultRef.is<Color3>()) {
                    alphaChannelIndex = 3;
                    auto result= resultRef.masked<Color3>();
                    auto resultDs = resultRef.maskedDx<Color3>();
                    auto resultDt = resultRef.maskedDy<Color3>();
                    result[i] = Color3(result_simd[0], result_simd[1], result_simd[2]);
                    if (has_derivs) {
                        resultDs[i] = Color3(dresultds_simd[0], dresultds_simd[1], dresultds_simd[2]);
                        resultDt[i] = Color3(dresultdt_simd[0], dresultdt_simd[1], dresultdt_simd[2]);
                    }
                } else if (resultRef.is<float>()) {
                    alphaChannelIndex = 1;
                    auto result= resultRef.masked<float>();
                    auto resultDs = resultRef.maskedDx<float>();
                    auto resultDt = resultRef.maskedDy<float>();
                    result[i] = result_simd[0];
                    if (has_derivs) {
                        resultDs[i] = dresultds_simd[0];
                        resultDt[i] = dresultdt_simd[0];
                    }
                }
                if (alphaIsValid) {
                    auto alpha = alphaRef.masked<float>();
                    alpha[i] = result_simd[alphaChannelIndex];
                    if (alphaRef.has_derivs()) {
                        auto alphaDs = alphaRef.maskedDx<float>();
                        auto alphaDt = alphaRef.maskedDy<float>();
                        alphaDs[i] = dresultds_simd[alphaChannelIndex];
                        alphaDt[i] = dresultdt_simd[alphaChannelIndex];
                    }
                }
                //std::cout << "s: " << s.get(i) << " t: " << t.get(i) << " color: " << resultColor << " " << wideResult.get(i) << std::endl;
            } else {
                std::string err = texturesys()->geterror();
                bool errMsgSize = err.size() > 0;
                if (errormessageIsValid) {
                    auto errormessage = outputs.errormessage().masked<ustring>();
                    if (errMsgSize) {
                        errormessage[i] = ustring(err);
                    }
                    else {
                        errormessage[i] = Strings::unknown;
                    }
                }
                else if (errMsgSize) {
                    // compilation error when using commented out form, investigate further...
                    //Mask errMask(Lane(i));
                    context->error (Mask(Lane(i)), "[BatchedRendererServices::texture] %s", err);
                }
            }
            status.set(i, retVal);
        }
    }
    return status;
}
#else

Mask
BatchedRendererServices::texture_uniform (ustring filename, TextureHandle * texture_handle,
                                          TexturePerthread * texture_thread_info,
                                          BatchedTextureOptionProvider & options, ShaderGlobalsBatch * sgb,
                                          ConstWideAccessor<float> s, ConstWideAccessor<float> t,
                                          ConstWideAccessor<float> dsdx, ConstWideAccessor<float> dtdx,
                                          ConstWideAccessor<float> dsdy, ConstWideAccessor<float> dtdy,
                                          BatchedTextureOutputs & outputs)
{
    Mask status(false);
    ASSERT(nullptr != sgb);
    ShadingContext *context = sgb->uniform().context;
    if (! texture_thread_info)
        texture_thread_info = context->texture_thread_info();

    Mask mask = outputs.mask();
    MaskedDataRef resultRef = outputs.result();
    bool has_derivs = resultRef.has_derivs();
    MaskedDataRef alphaRef = outputs.alpha();
    bool alphaIsValid = outputs.alpha().valid();
    bool errormessageIsValid = outputs.errormessage().valid();

    ASSERT(resultRef.valid());

    for (int i = 0; i < SimdLaneCount; ++i) {
        if (mask[i]) {
        	// Apparently the members of TextureOpt get modified from calls into
        	// the texture system, so lets start with a fresh set of defaults for now
        	TextureOpt opt;
            options.updateOption(opt, i);
            // It's actually faster to ask for 4 channels (even if we need fewer)
            // and ensure that they're being put in aligned memory.
            // TODO:  investigate if the above statement is true when nchannels==1
            OIIO::simd::float4 result_simd, dresultds_simd, dresultdt_simd;
            // TODO:  investigate if there any magic to this simd::float4 or 
            // can we just use a float[4] to the same effect and avoid confustion
            
            bool retVal = false;

            if (texture_handle) {
                retVal = texturesys()->texture (texture_handle, texture_thread_info, opt,
                                                s[i], t[i],
                                                dsdx[i], dtdx[i],
                                                dsdy[i], dtdy[i],
                                                4,
                                                (float *)&result_simd,
                                                has_derivs ? (float *)&dresultds_simd : NULL,
                                                has_derivs ? (float *)&dresultdt_simd : NULL);
            }
            else {
                retVal = texturesys()->texture (filename, opt,
                                                s[i], t[i],
                                                dsdx[i], dtdx[i],
                                                dsdy[i], dtdy[i],
                                                4,
                                                (float *)&result_simd,
                                                has_derivs ? (float *)&dresultds_simd : NULL,
                                                has_derivs ? (float *)&dresultdt_simd : NULL);
            }
            
            if (retVal) {
            	// Per the OSL language specification
            	// "The alpha channel (presumed to be the next channel following the channels returned by the texture() call)"
            	// so despite the fact the alpha channel really is
            	// we will always use +1 the final channel requested
            	int alphaChannelIndex;
                if (resultRef.is<Color3>()) {
                	alphaChannelIndex = 3;
                    auto result= resultRef.masked<Color3>();
                    auto resultDs = resultRef.maskedDx<Color3>();
                    auto resultDt = resultRef.maskedDy<Color3>();
                    result[i] = Color3(result_simd[0], result_simd[1], result_simd[2]);
					if (has_derivs) {
                        resultDs[i] = Color3(dresultds_simd[0], dresultds_simd[1], dresultds_simd[2]);
                        resultDt[i] = Color3(dresultdt_simd[0], dresultdt_simd[1], dresultdt_simd[2]);
					}
                } else if (resultRef.is<float>()) {
                	alphaChannelIndex = 1;
                    auto result= resultRef.masked<float>();
                    auto resultDs = resultRef.maskedDx<float>();
                    auto resultDt = resultRef.maskedDy<float>();
                    result[i] = result_simd[0];
					if (has_derivs) {
                        resultDs[i] = dresultds_simd[0];
                        resultDt[i] = dresultdt_simd[0];
					}
            	}
                if (alphaIsValid) {
				    auto alpha = alphaRef.masked<float>();
                    alpha[i] = result_simd[alphaChannelIndex];
                    if (alphaRef.has_derivs()) {
						auto alphaDs = alphaRef.maskedDx<float>();
    					auto alphaDt = alphaRef.maskedDy<float>();
                        alphaDs[i] = dresultds_simd[alphaChannelIndex];
                        alphaDt[i] = dresultdt_simd[alphaChannelIndex];
                    }
                }
                //std::cout << "s: " << s.get(i) << " t: " << t.get(i) << " color: " << resultColor << " " << wideResult.get(i) << std::endl;
            } else {
                std::string err = texturesys()->geterror();
				bool errMsgSize = err.size() > 0;
				if (errormessageIsValid) {
					auto errormessage = outputs.errormessage().masked<ustring>();
					if (errMsgSize) {
	                    errormessage[i] = ustring(err);
					}
					else {
						errormessage[i] = Strings::unknown;
					}
				}
				else if (errMsgSize) {
					// compilation error when using commented out form, investigate further...
					//Mask errMask(Lane(i));
					context->error (Mask(Lane(i)), "[BatchedRendererServices::texture] %s", err);
				}
            }
            status.set(i, retVal);
        }
    }
    return status;
}

Mask
BatchedRendererServices::texture (ConstWideAccessor<ustring> filename,
                           TexturePerthread *texture_thread_info,
                           BatchedTextureOptionProvider & options, ShaderGlobalsBatch *sgb,
                           ConstWideAccessor<float> s, ConstWideAccessor<float> t,
                           ConstWideAccessor<float> dsdx, ConstWideAccessor<float> dtdx,
                           ConstWideAccessor<float> dsdy, ConstWideAccessor<float> dtdy,
                           BatchedTextureOutputs& outputs)
{
	ASSERT(sgb);
    Mask status(false);
    ShadingContext *context = sgb->uniform().context;
    if (! texture_thread_info)
        texture_thread_info = context->texture_thread_info();

    Mask mask = outputs.mask();
    MaskedDataRef resultRef = outputs.result();
    bool has_derivs = resultRef.has_derivs();
    MaskedDataRef alphaRef = outputs.alpha();
    bool alphaIsValid = outputs.alpha().valid();
    bool errormessageIsValid = outputs.errormessage().valid();

    ASSERT(resultRef.valid());

    for (int i = 0; i < SimdLaneCount; ++i) {
        if (mask[i]) {
            // Apparently the members of TextureOpt get modified from calls into
            // the texture system, so lets start with a fresh set of defaults for now
            TextureOpt opt;
            options.updateOption(opt, i);
            // It's actually faster to ask for 4 channels (even if we need fewer)
            // and ensure that they're being put in aligned memory.
            // TODO:  investigate if the above statement is true when nchannels==1
            OIIO::simd::float4 result_simd, dresultds_simd, dresultdt_simd;
            // TODO:  investigate if there any magic to this simd::float4 or
            // can we just use a float[4] to the same effect and avoid confustion

            bool retVal = false;

            retVal = texturesys()->texture (filename[i], opt,
                                            s[i], t[i],
                                            dsdx[i], dtdx[i],
                                            dsdy[i], dtdy[i],
                                            4,
                                            (float *)&result_simd,
                                            has_derivs ? (float *)&dresultds_simd : NULL,
                                            has_derivs ? (float *)&dresultdt_simd : NULL);

            if (retVal) {
            	// Per the OSL language specification
            	// "The alpha channel (presumed to be the next channel following the channels returned by the texture() call)"
            	// so despite the fact the alpha channel really is
            	// we will always use +1 the final channel requested
            	int alphaChannelIndex;
                if (resultRef.is<Color3>()) {
                	alphaChannelIndex = 3;
                    auto result= resultRef.masked<Color3>();
                    auto resultDs = resultRef.maskedDx<Color3>();
                    auto resultDt = resultRef.maskedDy<Color3>();
                    result[i] = Color3(result_simd[0], result_simd[1], result_simd[2]);
                    if (has_derivs) {
                        resultDs[i] = Color3(dresultds_simd[0], dresultds_simd[1], dresultds_simd[2]);
                        resultDt[i] = Color3(dresultdt_simd[0], dresultdt_simd[1], dresultdt_simd[2]);
                    }
                } else if (resultRef.is<float>()) {
                	alphaChannelIndex = 1;
                    auto result= resultRef.masked<float>();
                    auto resultDs = resultRef.maskedDx<float>();
                    auto resultDt = resultRef.maskedDy<float>();
                    result[i] = result_simd[0];
                    if (has_derivs) {
                        resultDs[i] = dresultds_simd[0];
                        resultDt[i] = dresultdt_simd[0];
                    }
                }
                if (alphaIsValid) {
                    auto alpha = alphaRef.masked<float>();
                    alpha[i] = result_simd[alphaChannelIndex];
                    if (alphaRef.has_derivs()) {
                        auto alphaDs = alphaRef.maskedDx<float>();
                        auto alphaDt = alphaRef.maskedDy<float>();
                        alphaDs[i] = dresultds_simd[alphaChannelIndex];
                        alphaDt[i] = dresultdt_simd[alphaChannelIndex];
                    }
                }
                //std::cout << "s: " << s.get(i) << " t: " << t.get(i) << " color: " << resultColor << " " << wideResult.get(i) << std::endl;
            } else {
                std::string err = texturesys()->geterror();
                bool errMsgSize = err.size() > 0;
                if (errormessageIsValid) {
                    auto errormessage = outputs.errormessage().masked<ustring>();
                    if (errMsgSize) {
                        errormessage[i] = ustring(err);
                    }
                    else {
                        errormessage[i] = Strings::unknown;
                    }
                }
                else if (errMsgSize) {
                    context->error (Mask(Lane(i)), "[BatchedRendererServices::texture] %s", err);
                }
            }
            status.set(i, retVal);
        }
    }
    return status;
}
#endif

OSL_NAMESPACE_EXIT
