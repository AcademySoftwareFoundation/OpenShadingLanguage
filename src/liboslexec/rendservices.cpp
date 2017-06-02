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

inline Matrix44 affineInvert(const Matrix44 &m)
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
 
bool
BatchedRendererServices::get_inverse_matrix (ShaderGlobalsBatch *sgb, Wide<Matrix44> &result,
                                      const Wide<TransformationPtr> & xform, const Wide<float> &time)
{
	int wok = true;
	
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		Wide<Matrix44> wmatrix;
		/*bool ok =*/ get_matrix (sgb, wmatrix, xform, time);
		
	    int allAreAffine = 1;
		OSL_INTEL_PRAGMA("simd assert")
		for(int lane=0; lane < SimdLaneCount; ++lane) {
			Matrix44 m = wmatrix.get(lane);        
		    if (m.x[0][3] != 0.0f || m.x[1][3] != 0.0f || m.x[2][3] != 0.0f || m.x[3][3] != 1.0f) {
		    	allAreAffine = 0;
		    }
		}
		
#if 1
		if (allAreAffine) {
			OSL_INTEL_PRAGMA("simd assert vectorlength(SimdLaneCount)")
			for(int lane=0; lane < SimdLaneCount; ++lane) {    
				Matrix44 m = wmatrix.get(lane);        
				//bool ok = get_matrix (sgb, r, xform.get(lane), time.get(lane));
				//r.invert();
				Matrix44 r = affineInvert(m);
				result.set(lane, r);        
			}
		} else
#endif
		{
			// Scalar code for non affine matrix (well at least 1 lane isn't)
			for(int lane=0; lane < SimdLaneCount; ++lane) {    
				Matrix44 r = wmatrix.get(lane);
				r.invert();
				result.set(lane, r);
			}			
		}
		
		//{
		//	Matrix44 r = result.get(0);
	    //	std::cout << "get_inverse_matrix " << std::endl << r << std::endl;
		//}

	}
    return wok;	
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



bool
BatchedRendererServices::get_inverse_matrix (ShaderGlobalsBatch *sgb, Matrix44 &result,
                                      ustring to, float time)
{
    bool ok = get_matrix (sgb, result, to, time);
    if (ok)
        result.invert ();
    return ok;
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
    Mask success;
//    for (int i = 0; i < filename.width; ++i) {
//        std::cout << "osl_get_textureinfo_batched: " << filename.get(i) << " mask: " << mask[i] << std::endl;
//        if (mask[i]) {
//            bool status = texturesys()->get_texture_info (filename.get(i), subimage,
//                                             dataname, datatype, static_cast<char*>(data)+stride*i);
//            if (!status) {
//                std::string err = texturesys()->geterror();
//                if (err.size() && sgb) {
//                    sgb->uniform().context->error ("[BatchRendererServices::get_texture_info] %s", err);
//                }
//            }
//            success.set(i, status);
//        }
//    }

    return success;
}

bool
BatchedRendererServices::get_texture_info_uniform (ShaderGlobalsBatch *sgb, ustring filename,
                                                   TextureHandle *texture_handle,
                                                   int subimage, ustring dataname,
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
                           void* alpha, void* dalphadx, void* dalphady,
                           ustring *errormessage, Mask mask)
{
    ShadingContext *context = sgb->uniform().context;
    if (! texture_thread_info)
        texture_thread_info = context->texture_thread_info();

    Mask status(false);
    std::cout << "nchannels: " << nchannels << std::endl;
    if (texture_handle) {
        for (int i = 0; i < SimdLaneCount; ++i) {
            if (mask[i]) {
                TextureOpt opt = options ? options->getOption(i) : TextureOpt();
                float* texResult = nullptr;
                Color3 resultColor;
                if (nchannels == 1) {
                    texResult = reinterpret_cast<float*>(result);
                }
                else if (nchannels == 3) {
                    texResult = (float*)&(resultColor.x);
                }
                bool retVal = texturesys()->texture (texture_handle, texture_thread_info, opt,
                                                     s.get(i), t.get(i),
                                                     dsdx.get(i), dtdx.get(i),
                                                     dsdy.get(i), dtdy.get(i),
                                                     nchannels, texResult/*, dresultds, dresultdt*/);
                if (nchannels == 3) {
                    Wide<Color3>& wideResult= *reinterpret_cast<Wide<Color3>*>(result);
                    wideResult.set(i, resultColor);
                    std::cout << "s: " << s.get(i) << " t: " << t.get(i) << " color: " << resultColor << " " << wideResult.get(i) << std::endl;
                }
                status.set(i, retVal);
            }
        }
    }
    else {
        for (int i = 0; i < SimdLaneCount; ++i) {
            if (mask[i]) {
                TextureOpt opt = options ? options->getOption(i) : TextureOpt();
                float* texResult = nullptr;
                Color3 resultColor;
                if (nchannels == 1) {
                    texResult = reinterpret_cast<float*>(result);
                }
                else if (nchannels == 3) {
                    texResult = (float*)&(resultColor.x);
                }
                bool retVal = texturesys()->texture (filename, opt,
                                                     s.get(i), t.get(i),
                                                     dsdx.get(i), dtdx.get(i),
                                                     dsdy.get(i), dtdy.get(i),
                                                     nchannels, texResult/*, dresultds, dresultdt*/);
                std::cout << "s: " << s.get(i) << " t: " << t.get(i) << " color: " << resultColor << std::endl;
                if (nchannels == 3) {
                    Wide<Color3>& wideResult= *reinterpret_cast<Wide<Color3>*>(result);
                    wideResult.set(i, resultColor);
                }
                status.set(i, retVal);
            }
        }
    }
    /*
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sgb) {
            if (errormessage) {
                *errormessage = ustring(err);
            } else {
                context->error ("[RendererServices::texture] %s", err);
            }
        } else if (errormessage) {
            *errormessage = Strings::unknown;
        }
    }
    */
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
                           void* alpha, void* dalphadx, void* dalphady,
                           ustring *errormessage, Mask mask)
{
    ShadingContext *context = sgb->uniform().context;
    if (! texture_thread_info)
        texture_thread_info = context->texture_thread_info();
    bool status;
    /*
    status = texturesys()->texture (filename,
                                    options, s, t, dsdx, dtdx, dsdy, dtdy,
                                    nchannels, result, dresultds, dresultdt);
    if (!status) {
        std::string err = texturesys()->geterror();
        if (err.size() && sgb) {
            if (errormessage) {
                *errormessage = ustring(err);
            } else {
                context->error ("[RendererServices::texture] %s", err);
            }
        } else if (errormessage) {
            *errormessage = Strings::unknown;
        }
    }
    */
    return Mask(status);
}

OSL_NAMESPACE_EXIT
