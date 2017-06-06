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

OSL_INLINE static Matrix44 affineInvert(const Matrix44 &m)
{
    //assert(__builtin_expect(m.x[0][3] == 0.0f && m.x[1][3] == 0.0f && m.x[2][3] == 0.0f && m.x[3][3] == 1.0f, 1))
	Matrix44 s (m.x[1][1] * m.x[2][2] - m.x[2][1] * m.x[1][2],
				m.x[2][1] * m.x[0][2] - m.x[0][1] * m.x[2][2],
				m.x[0][1] * m.x[1][2] - m.x[1][1] * m.x[0][2],
				0.0f,

				m.x[2][0] * m.x[1][2] - m.x[1][0] * m.x[2][2],
				m.x[0][0] * m.x[2][2] - m.x[2][0] * m.x[0][2],
				m.x[1][0] * m.x[0][2] - m.x[0][0] * m.x[1][2],
				0.0f,

				m.x[1][0] * m.x[2][1] - m.x[2][0] * m.x[1][1],
				m.x[2][0] * m.x[0][1] - m.x[0][0] * m.x[2][1],
				m.x[0][0] * m.x[1][1] - m.x[1][0] * m.x[0][1],
				0.0f,

				0.0f,
				0.0f,
				0.0f,
				1.0f);

	float r = m.x[0][0] * s[0][0] + m.x[0][1] * s[1][0] + m.x[0][2] * s[2][0];
	float abs_r = IMATH_INTERNAL_NAMESPACE::abs (r);


	int may_have_divided_by_zero = 0;
	if (__builtin_expect(abs_r < 1.0f, 0))
	{
		float mr = abs_r / Imath::limits<float>::smallest();
		OSL_INTEL_PRAGMA("unroll")
		for (int i = 0; i < 3; ++i)
		{
			OSL_INTEL_PRAGMA("unroll")
			for (int j = 0; j < 3; ++j)
			{
				if (mr <= IMATH_INTERNAL_NAMESPACE::abs (s[i][j]))
				{
					may_have_divided_by_zero = 1;
				}
			}
		}
	}
	
	OSL_INTEL_PRAGMA("unroll")
	for (int i = 0; i < 3; ++i)
	{
		OSL_INTEL_PRAGMA("unroll")
		for (int j = 0; j < 3; ++j)
		{
			s[i][j] /= r;
		}
	}

	s[3][0] = -m.x[3][0] * s[0][0] - m.x[3][1] * s[1][0] - m.x[3][2] * s[2][0];
	s[3][1] = -m.x[3][0] * s[0][1] - m.x[3][1] * s[1][1] - m.x[3][2] * s[2][1];
	s[3][2] = -m.x[3][0] * s[0][2] - m.x[3][1] * s[1][2] - m.x[3][2] * s[2][2];
	
	if (__builtin_expect(may_have_divided_by_zero == 1, 0))
	{
		s = Matrix44();
	}
	return s;
}
 

OSL_INLINE static void invert_wide_matrix(Wide<Matrix44> &result, const Wide<Matrix44> &wmatrix, WeakMask mask)
{
	// TODO: As we don't expect failure, not sure it is worth overhead to skip this work
	if (mask.any_on()) {
		int allAreAffine = 1;
		OSL_INTEL_PRAGMA("omp simd simdlen(wmatrix.width)")
		for(int lane=0; lane < wmatrix.width; ++lane) {
			Matrix44 m = wmatrix.get(lane);        
			if (mask.is_on(lane) && 
			    (m.x[0][3] != 0.0f || m.x[1][3] != 0.0f || m.x[2][3] != 0.0f || m.x[3][3] != 1.0f)) {
				allAreAffine = 0;
			}
		}
		
		if (allAreAffine) {
			OSL_INTEL_PRAGMA("omp simd simdlen(wmatrix.width)")
			for(int lane=0; lane < wmatrix.width; ++lane) {    
				Matrix44 m = wmatrix.get(lane);        
				//bool ok = get_matrix (sgb, r, xform.get(lane), time.get(lane));
				//r.invert();
				Matrix44 r = affineInvert(m);
				result.set(lane, r);        
			}
		} else
		{
			// Scalar code for non affine matrix (well at least 1 lane isn't)
			for(int lane=0; lane < SimdLaneCount; ++lane) {
				if (mask.is_on(lane)) {
					Matrix44 r = wmatrix.get(lane);
					r.invert();
					result.set(lane, r);
				}
			}			
		}
	}
}
Mask
BatchedRendererServices::get_inverse_matrix (ShaderGlobalsBatch *sgb, Wide<Matrix44> &result,
                                      const Wide<TransformationPtr> & xform, const Wide<float> &time,
                                      WeakMask weak_mask)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		Wide<Matrix44> wmatrix;
		Mask succeeded = get_matrix (sgb, wmatrix, xform, time, weak_mask);
		invert_wide_matrix(result, wmatrix, succeeded&weak_mask);
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
BatchedRendererServices::get_inverse_matrix (ShaderGlobalsBatch *sgb,  Wide<Matrix44> &result,
                                      ustring to, const Wide<float> &time, WeakMask weak_mask)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		Wide<Matrix44> wmatrix;
		Mask succeeded = get_matrix (sgb, wmatrix, to, time, weak_mask);
		invert_wide_matrix(result, wmatrix, succeeded&weak_mask);
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
                                           const Wide<ustring>& filename,
                                           int subimage,
                                           ustring dataname,
                                           MaskedDataRef val)
{
    Mask success(false);

#define TEXTURE_INFO_FOR_TYPE(data_type)                                                        \
    if (val.is<data_type>()) {                                                                  \
        auto out = val.masked<data_type>();                                                     \
        for (int i = 0; i < out.width; ++i) {                                                   \
            data_type data;                                                                     \
            bool status = texturesys()->get_texture_info (filename.get(i), subimage,            \
                                                          dataname, val.type(), &data);         \
            /* masked assignment */                                                             \
            out[i] = data;                                                                      \
            success.set(i, status);                                                             \
            if (!status) {                                                                      \
                std::string err = texturesys()->geterror();                                     \
                if (err.size() && sgb) {                                                        \
                    sgb->uniform().context->error ("[BatchRendererServices::get_texture_info] %s", err);\
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
        return success;                                                                         \
    }

#define TEXTURE_INFO_FOR_ARRAY(data_type)                                                       \
    if (val.is<data_type[]>()) {                                                                \
        auto out = val.masked<data_type[]>();                                                   \
        for (int l = 0; l < out.width; ++l) {                                                   \
            auto arrayData = out[l];                                                            \
            data_type data[arrayData.length()];                                                       \
            bool status = texturesys()->get_texture_info (filename.get(l), subimage,            \
                                                          dataname, val.type(), data);          \
            for (int i = 0; i < arrayData.length(); ++i) {                                      \
                arrayData[i] = data[i];                                                         \
            }                                                                                   \
            success.set(l, status);                                                             \
            if (!status) {                                                                      \
                std::string err = texturesys()->geterror();                                     \
                if (err.size() && sgb) {                                                        \
                    sgb->uniform().context->error ("[BatchRendererServices::get_texture_info] %s", err);\
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

    /*
    for (int i = 0; i < filename.width; ++i) {
        std::cout << "osl_get_textureinfo_batched: " << filename.get(i) << " mask: " << mask[i] << std::endl;
        if (mask[i]) {
            if (val.is<float>()) {

            }
            bool status = texturesys()->get_texture_info (filename.get(i), subimage,
                                             dataname, datatype, static_cast<char*>(data)+stride*i);
            if (!status) {
                std::string err = texturesys()->geterror();
                if (err.size() && sgb) {
                    sgb->uniform().context->error ("[BatchRendererServices::get_texture_info] %s", err);
                }
            }
            success.set(i, status);
        }
    }
    */

    return success;
}

bool
BatchedRendererServices::get_texture_info_uniform (ShaderGlobalsBatch *sgb, ustring filename,
                                                   TextureHandle *texture_handle,
                                                   int subimage,
                                                   ustring dataname,
                                                   DataRef val)
{
    std::cout << "dataname: " << dataname << std::endl;;
    bool status;
    if (texture_handle)
        status = texturesys()->get_texture_info (texture_handle, NULL, subimage,
                                                 dataname, val.type(), val.ptr());
    else
        status = texturesys()->get_texture_info (filename, subimage,
                                                 dataname, val.type(), val.ptr());
    std::cout << "status: " << status << " val: " << *((int*)val.ptr()) << std::endl;
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sgb) {
            sgb->uniform().context->error ("[BatchRendererServices::get_texture_info] %s", err);
        }
    }
    return status;
}

Mask
BatchedRendererServices::texture_uniform (ustring filename, TextureHandle *texture_handle,
                                          TexturePerthread *texture_thread_info,
                                          const TextureOptions *options, ShaderGlobalsBatch *sgb,
                                          const Wide<float>& s, const Wide<float>& t,
                                          const Wide<float>& dsdx, const Wide<float>& dtdx,
                                          const Wide<float>& dsdy, const Wide<float>& dtdy,
                                          int nchannels,
                                          void* result, void* dresultds, void* dresultdt,
                                          Wide<float>* alpha, Wide<float>* dalphadx, Wide<float>* dalphady,
                                          Wide<ustring>* errormessage, Mask mask)
{
    ShadingContext *context = sgb->uniform().context;
    if (! texture_thread_info)
        texture_thread_info = context->texture_thread_info();

    // TODO: change to DASSERT once confidenty
    ASSERT((dresultds != NULL) == (dresultdt != NULL));
    bool has_derivs = dresultds != NULL;
    
    Mask status(false);
    //std::cout << "nchannels: " << nchannels << std::endl;
    for (int i = 0; i < SimdLaneCount; ++i) {
        if (mask[i]) {
        	// Apparently the members of TextureOpt get modified from calls into
        	// the texture system, so lets start with a fresh set of defaults for now
        	TextureOpt opt;
            if(options) {
                options->updateOption(opt, i);
            }
            // It's actually faster to ask for 4 channels (even if we need fewer)
            // and ensure that they're being put in aligned memory.
            // TODO:  investigate if the above statement is true when nchannels==1
            OIIO::simd::float4 result_simd, dresultds_simd, dresultdt_simd;
            // TODO:  investigate if there any magic to this simd::float4 or 
            // can we just use a float[4] to the same effect and avoid confustion
            
            bool retVal = false;
            if (texture_handle) {
                retVal = texturesys()->texture (texture_handle, texture_thread_info, opt,
                                                s.get(i), t.get(i),
                                                dsdx.get(i), dtdx.get(i),
                                                dsdy.get(i), dtdy.get(i),
                                                4, 
                                                (float *)&result_simd,
                                                has_derivs ? (float *)&dresultds_simd : NULL,
												has_derivs ? (float *)&dresultdt_simd : NULL);
            }
            else {
                retVal = texturesys()->texture (filename, opt,
                                                s.get(i), t.get(i),
                                                dsdx.get(i), dtdx.get(i),
                                                dsdy.get(i), dtdy.get(i),
                                                4, 
                                                (float *)&result_simd,
                                                has_derivs ? (float *)&dresultds_simd : NULL,
												has_derivs ? (float *)&dresultdt_simd : NULL);
            }
            
            if (retVal) {
            	if (nchannels == 3) {
					Wide<Color3>& wideResult = *reinterpret_cast<Wide<Color3>*>(result);
					wideResult.set(i, Color3(result_simd[0], result_simd[1], result_simd[2]));
					if (has_derivs) {
						Wide<Color3>& wideResultds = *reinterpret_cast<Wide<Color3>*>(dresultds);
						wideResultds.set(i, Color3(dresultds_simd[0], dresultds_simd[1], dresultds_simd[2]));
						Wide<Color3>& wideResultdt = *reinterpret_cast<Wide<Color3>*>(dresultdt);
						wideResultdt.set(i, Color3(dresultdt_simd[0], dresultdt_simd[1], dresultdt_simd[2]));
					}
            	} else if (nchannels == 1) {
					Wide<float>& wideResult = *reinterpret_cast<Wide<float>*>(result);
					wideResult.set(i, result_simd[0]);
					if (has_derivs) {
						Wide<float>& wideResultds = *reinterpret_cast<Wide<float>*>(dresultds);
						wideResultds.set(i, dresultds_simd[0]);
						Wide<float>& wideResultdt = *reinterpret_cast<Wide<float>*>(dresultdt);
						wideResultdt.set(i, dresultdt_simd[0]);
					}
            	}
            	if (alpha)
            	{
            		alpha->set(i, result_simd[nchannels]);
            	}
                //std::cout << "s: " << s.get(i) << " t: " << t.get(i) << " color: " << resultColor << " " << wideResult.get(i) << std::endl;
            } else {
                std::string err = texturesys()->geterror();
                if (err.size() && sgb) {
                    if (errormessage) {
                        errormessage->set(i, ustring(err));
                    } else {
                        context->error ("[RendererServices::texture] %s", err);
                    }
                } else if (errormessage) {
                    errormessage->set(i, ustring(err));
                }
            }
            status.set(i, retVal);
        }
    }

    return status;
}

Mask
BatchedRendererServices::texture (const Wide<ustring>& filename,
                           TexturePerthread *texture_thread_info,
                           const TextureOptions* options, ShaderGlobalsBatch *sgb,
                           const Wide<float>& s, const Wide<float>& t,
                           const Wide<float>& dsdx, const Wide<float>& dtdx,
                           const Wide<float>& dsdy, const Wide<float>& dtdy,
                           int nchannels,
                           void* result, void* dresultds, void* dresultdt,
                           Wide<float>* alpha, Wide<float>* dalphadx, Wide<float>* dalphady,
                           Wide<ustring>* errormessage, Mask mask)
{
    ShadingContext *context = sgb->uniform().context;
    if (! texture_thread_info)
        texture_thread_info = context->texture_thread_info();
    // TODO: change to DASSERT once confidenty
    ASSERT((dresultds != NULL) == (dresultdt != NULL));
    bool has_derivs = dresultds != NULL;

    Mask status(false);
    //std::cout << "nchannels: " << nchannels << std::endl;
    for (int i = 0; i < SimdLaneCount; ++i) {
        if (mask[i]) {
            // Apparently the members of TextureOpt get modified from calls into
            // the texture system, so lets start with a fresh set of defaults for now
            TextureOpt opt;
            if(options) {
                options->updateOption(opt, i);
            }
            // It's actually faster to ask for 4 channels (even if we need fewer)
            // and ensure that they're being put in aligned memory.
            // TODO:  investigate if the above statement is true when nchannels==1
            OIIO::simd::float4 result_simd, dresultds_simd, dresultdt_simd;
            // TODO:  investigate if there any magic to this simd::float4 or
            // can we just use a float[4] to the same effect and avoid confustion

            bool retVal = false;
            retVal = texturesys()->texture (filename.get(i), opt,
                                            s.get(i), t.get(i),
                                            dsdx.get(i), dtdx.get(i),
                                            dsdy.get(i), dtdy.get(i),
                                            4,
                                            (float *)&result_simd,
                                            has_derivs ? (float *)&dresultds_simd : NULL,
                                            has_derivs ? (float *)&dresultdt_simd : NULL);

            if (retVal) {
                if (nchannels == 3) {
                    Wide<Color3>& wideResult = *reinterpret_cast<Wide<Color3>*>(result);
                    wideResult.set(i, Color3(result_simd[0], result_simd[1], result_simd[2]));
                    if (has_derivs) {
                        Wide<Color3>& wideResultds = *reinterpret_cast<Wide<Color3>*>(dresultds);
                        wideResultds.set(i, Color3(dresultds_simd[0], dresultds_simd[1], dresultds_simd[2]));
                        Wide<Color3>& wideResultdt = *reinterpret_cast<Wide<Color3>*>(dresultdt);
                        wideResultdt.set(i, Color3(dresultdt_simd[0], dresultdt_simd[1], dresultdt_simd[2]));
                    }
                } else if (nchannels == 1) {
                    Wide<float>& wideResult = *reinterpret_cast<Wide<float>*>(result);
                    wideResult.set(i, result_simd[0]);
                    if (has_derivs) {
                        Wide<float>& wideResultds = *reinterpret_cast<Wide<float>*>(dresultds);
                        wideResultds.set(i, dresultds_simd[0]);
                        Wide<float>& wideResultdt = *reinterpret_cast<Wide<float>*>(dresultdt);
                        wideResultdt.set(i, dresultdt_simd[0]);
                    }
                }
                if (alpha)
                {
                    alpha->set(i, result_simd[nchannels]);
                }
                //std::cout << "s: " << s.get(i) << " t: " << t.get(i) << " color: " << resultColor << " " << wideResult.get(i) << std::endl;
            } else {
                std::string err = texturesys()->geterror();
                if (err.size() && sgb) {
                    if (errormessage) {
                        errormessage->set(i, ustring(err));
                    } else {
                        context->error ("[RendererServices::texture] %s", err);
                    }
                } else if (errormessage) {
                    errormessage->set(i, ustring(err));
                }
            }
            status.set(i, retVal);
        }
    }

    return status;
}

OSL_NAMESPACE_EXIT
