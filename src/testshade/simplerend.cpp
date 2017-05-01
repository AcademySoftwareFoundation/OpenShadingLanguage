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


#include "OSL/oslexec.h"
#include "OSL/genclosure.h"
#include "simplerend.h"
using namespace OSL;


// anonymous namespace
namespace {

// unique identifier for each closure supported by testshade
enum ClosureIDs {
    EMISSION_ID = 1,
    BACKGROUND_ID,
    DIFFUSE_ID,
    OREN_NAYAR_ID,
    TRANSLUCENT_ID,
    PHONG_ID,
    WARD_ID,
    MICROFACET_ID,
    REFLECTION_ID,
    FRESNEL_REFLECTION_ID,
    REFRACTION_ID,
    TRANSPARENT_ID,
    DEBUG_ID,
    HOLDOUT_ID,
};

// these structures hold the parameters of each closure type
// they will be contained inside ClosureComponent
struct EmptyParams      { };
struct DiffuseParams    { Vec3 N; ustring label; };
struct OrenNayarParams  { Vec3 N; float sigma; };
struct PhongParams      { Vec3 N; float exponent; ustring label; };
struct WardParams       { Vec3 N, T; float ax, ay; };
struct ReflectionParams { Vec3 N; float eta; };
struct RefractionParams { Vec3 N; float eta; };
struct MicrofacetParams { ustring dist; Vec3 N, U; float xalpha, yalpha, eta; int refract; };
struct DebugParams      { ustring tag; };

} // anonymous namespace


OSL_NAMESPACE_ENTER

static ustring u_camera("camera"), u_screen("screen");
static ustring u_NDC("NDC"), u_raster("raster");
static ustring u_perspective("perspective");
static ustring u_s("s"), u_t("t");
static TypeDesc TypeFloatArray2 (TypeDesc::FLOAT, 2);
static TypeDesc TypeFloatArray4 (TypeDesc::FLOAT, 4);
static TypeDesc TypeIntArray2 (TypeDesc::INT, 2);


void register_closures(OSL::ShadingSystem* shadingsys) {
    // Describe the memory layout of each closure type to the OSL runtime
    enum { MaxParams = 32 };
    struct BuiltinClosures {
        const char* name;
        int id;
        ClosureParam params[MaxParams]; // upper bound
    };
    BuiltinClosures builtins[] = {
        { "emission"   , EMISSION_ID,           { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "background" , BACKGROUND_ID,         { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "diffuse"    , DIFFUSE_ID,            { CLOSURE_VECTOR_PARAM(DiffuseParams, N),
                                                  CLOSURE_STRING_KEYPARAM(DiffuseParams, label, "label"), // example of custom key param
                                                  CLOSURE_FINISH_PARAM(DiffuseParams) } },
        { "oren_nayar" , OREN_NAYAR_ID,         { CLOSURE_VECTOR_PARAM(OrenNayarParams, N),
                                                  CLOSURE_FLOAT_PARAM (OrenNayarParams, sigma),
                                                  CLOSURE_FINISH_PARAM(OrenNayarParams) } },
        { "translucent", TRANSLUCENT_ID,        { CLOSURE_VECTOR_PARAM(DiffuseParams, N),
                                                  CLOSURE_FINISH_PARAM(DiffuseParams) } },
        { "phong"      , PHONG_ID,              { CLOSURE_VECTOR_PARAM(PhongParams, N),
                                                  CLOSURE_FLOAT_PARAM (PhongParams, exponent),
                                                  CLOSURE_STRING_KEYPARAM(PhongParams, label, "label"), // example of custom key param
                                                  CLOSURE_FINISH_PARAM(PhongParams) } },
        { "ward"       , WARD_ID,               { CLOSURE_VECTOR_PARAM(WardParams, N),
                                                  CLOSURE_VECTOR_PARAM(WardParams, T),
                                                  CLOSURE_FLOAT_PARAM (WardParams, ax),
                                                  CLOSURE_FLOAT_PARAM (WardParams, ay),
                                                  CLOSURE_FINISH_PARAM(WardParams) } },
        { "microfacet", MICROFACET_ID,          { CLOSURE_STRING_PARAM(MicrofacetParams, dist),
                                                  CLOSURE_VECTOR_PARAM(MicrofacetParams, N),
                                                  CLOSURE_VECTOR_PARAM(MicrofacetParams, U),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, xalpha),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, yalpha),
                                                  CLOSURE_FLOAT_PARAM (MicrofacetParams, eta),
                                                  CLOSURE_INT_PARAM   (MicrofacetParams, refract),
                                                  CLOSURE_FINISH_PARAM(MicrofacetParams) } },
        { "reflection" , REFLECTION_ID,         { CLOSURE_VECTOR_PARAM(ReflectionParams, N),
                                                  CLOSURE_FINISH_PARAM(ReflectionParams) } },
        { "reflection" , FRESNEL_REFLECTION_ID, { CLOSURE_VECTOR_PARAM(ReflectionParams, N),
                                                  CLOSURE_FLOAT_PARAM (ReflectionParams, eta),
                                                  CLOSURE_FINISH_PARAM(ReflectionParams) } },
        { "refraction" , REFRACTION_ID,         { CLOSURE_VECTOR_PARAM(RefractionParams, N),
                                                  CLOSURE_FLOAT_PARAM (RefractionParams, eta),
                                                  CLOSURE_FINISH_PARAM(RefractionParams) } },
        { "transparent", TRANSPARENT_ID,        { CLOSURE_FINISH_PARAM(EmptyParams) } },
        { "debug"      , DEBUG_ID,              { CLOSURE_STRING_PARAM(DebugParams, tag),
                                                  CLOSURE_FINISH_PARAM(DebugParams) } },
        { "holdout"    , HOLDOUT_ID,            { CLOSURE_FINISH_PARAM(EmptyParams) } },
        // mark end of the array
        { NULL, 0, {} }
    };

    for (int i = 0; builtins[i].name; i++) {
        shadingsys->register_closure(
            builtins[i].name,
            builtins[i].id,
            builtins[i].params,
            NULL, NULL);
    }
}



#if OSL_USE_WIDE_LLVM_BACKEND

BatchedSimpleRenderer::BatchedSimpleRenderer(SimpleRenderer &sr)
: m_sr(sr)
{}

BatchedSimpleRenderer::~BatchedSimpleRenderer()
{}

#if 0
bool
BatchedSimpleRenderer::get_matrix (ShaderGlobalsBatch *sgb, Wide<Matrix44> &result,
        const Wide<TransformationPtr> & xform, const Wide<float> &time)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    //result = *reinterpret_cast<const Matrix44*>(xform);
	
#if 0
	int is_uniform_xform = 1;
	
	OSL_INTEL_PRAGMA("simd")
	for(int lane=0; lane < SimdLaneCount; ++lane) {
		if (uniform_xform != xform.get(lane))
			is_uniform_xform = 0;
	}
#endif 
	TransformationPtr uniform_xform = xform.get(0);
	
#if 0
	bool is_uniform_xform = 
		(xform.get(0) == xform.get(1)) &&
		(xform.get(1) == xform.get(2)) &&
		(xform.get(2) == xform.get(3)) &&
		(xform.get(3) == xform.get(4)) &&
		(xform.get(4) == xform.get(5)) &&
		(xform.get(5) == xform.get(6)) &&
		(xform.get(6) == xform.get(7));
#endif

#if 0
	int numLanesMatch1st = 0;
	OSL_INTEL_PRAGMA("simd reduction(+:numLanesMatch1st)  assert")
	for(int lane=0; lane < SimdLaneCount; ++lane) {
		int match = (uniform_xform == xform.get(lane)) ? 1 : 0;
		numLanesMatch1st += match;
	}

	if (numLanesMatch1st == 8) {
#elif 0
		int numLanesMatch1st = 0;
		OSL_INTEL_PRAGMA("simd reduction(+:numLanesMatch1st) vectorlength(8) assert")
		for(int lane=0; lane < SimdLaneCount; ++lane) {
			numLanesMatch1st += (uniform_xform == xform.get(lane));
		}

		if (numLanesMatch1st == 8) {
		
#else
		register __m256i xformPointers  = _mm2565_load_epi32(xform.data);
		
		if (1)

#endif
		const Matrix44 & transformFromShaderGlobals = *reinterpret_cast<const Matrix44*>(uniform_xform);
		OSL_INTEL_PRAGMA("simd")
		for(int lane=0; lane < SimdLaneCount; ++lane) {    
			result.set(lane,transformFromShaderGlobals);
		}		
	} else {			
		for(int lane=0; lane < SimdLaneCount; ++lane) {    
			const Matrix44 & transformFromShaderGlobals = *reinterpret_cast<const Matrix44*>(xform.get(lane));
			result.set(lane,transformFromShaderGlobals);
		}
	}
    return true;
}

#else
	bool
	BatchedSimpleRenderer::get_matrix (ShaderGlobalsBatch *sgb, Wide<Matrix44> &result,
	        const Wide<TransformationPtr> & xform, const Wide<float> &time)
	{
	    // SimpleRenderer doesn't understand motion blur and transformations
	    // are just simple 4x4 matrices.
	    //result = *reinterpret_cast<const Matrix44*>(xform);
		
		TransformationPtr uniform_xform = xform.get(0);
		
#if 0
		// In general, one can't assume that the transformation is uniform
		const Matrix44 & uniformTransform = *reinterpret_cast<const Matrix44*>(uniform_xform);
		OSL_INTEL_PRAGMA("simd")
		for(int lane=0; lane < SimdLaneCount; ++lane) {
			if (__builtin_expect((uniform_xform == xform.get(lane)),1)) {
				result.set(lane,uniformTransform);	
			} else {				
				const Matrix44 & transformFromShaderGlobals = *reinterpret_cast<const Matrix44*>(xform.get(lane));
				result.set(lane,transformFromShaderGlobals);
			}
		}	
#else
		// But this is "testshade" and we know we only have one object, so lets just 
		// use that fact
		const Matrix44 & uniformTransform = *reinterpret_cast<const Matrix44*>(uniform_xform);
		
		OSL_INTEL_PRAGMA("simd")
		for(int lane=0; lane < SimdLaneCount; ++lane) {
			result.set(lane,uniformTransform);	
		}	
		
#endif
		
	    return true;
	}

	
#endif
	bool
	BatchedSimpleRenderer::get_matrix (ShaderGlobalsBatch *sgb, Matrix44 &result,
	                            ustring from, float time)
	{
		auto found = m_sr.m_named_xforms.find (from);
	    if (found != m_sr.m_named_xforms.end()) {
	        result = *(found->second);
	        return true;
	    } else {
	        return false;
	    }
	}



	bool
	BatchedSimpleRenderer::get_matrix (ShaderGlobalsBatch *sgb, Matrix44 &result,
	                            TransformationPtr xform)
	{
	    // SimpleRenderer doesn't understand motion blur and transformations
	    // are just simple 4x4 matrices.
	    result = *(OSL::Matrix44 *)xform;
	    return true;
	}



	bool
	BatchedSimpleRenderer::get_matrix (ShaderGlobalsBatch *sgb, Matrix44 &result,
	                            ustring from)
	{
	    // SimpleRenderer doesn't understand motion blur, so we never fail
	    // on account of time-varying transformations.
	    auto found = m_sr.m_named_xforms.find (from);
	    if (found != m_sr.m_named_xforms.end()) {
	        result = *(found->second);
	        return true;
	    } else {
	        return false;
	    }
	}



	bool
	BatchedSimpleRenderer::get_inverse_matrix (ShaderGlobalsBatch *sgb, Matrix44 &result,
	                                    ustring to, float time)
	{
	    if (to == u_camera || to == u_screen || to == u_NDC || to == u_raster) {
	        Matrix44 M = m_sr.m_world_to_camera;
	        if (to == u_screen || to == u_NDC || to == u_raster) {
	            float depthrange = (double)m_sr.m_yon-(double)m_sr.m_hither;
	            if (m_sr.m_projection == u_perspective) {
	                float tanhalffov = tanf (0.5f * m_sr.m_fov * M_PI/180.0);
	                Matrix44 camera_to_screen (1/tanhalffov, 0, 0, 0,
	                                           0, 1/tanhalffov, 0, 0,
	                                           0, 0, m_sr.m_yon/depthrange, 1,
	                                           0, 0, -m_sr.m_yon*m_sr.m_hither/depthrange, 0);
	                M = M * camera_to_screen;
	            } else {
	                Matrix44 camera_to_screen (1, 0, 0, 0,
	                                           0, 1, 0, 0,
	                                           0, 0, 1/depthrange, 0,
	                                           0, 0, -m_sr.m_hither/depthrange, 1);
	                M = M * camera_to_screen;
	            }
	            if (to == u_NDC || to == u_raster) {
	                float screenleft = -1.0, screenwidth = 2.0;
	                float screenbottom = -1.0, screenheight = 2.0;
	                Matrix44 screen_to_ndc (1/screenwidth, 0, 0, 0,
	                                        0, 1/screenheight, 0, 0,
	                                        0, 0, 1, 0,
	                                        -screenleft/screenwidth, -screenbottom/screenheight, 0, 1);
	                M = M * screen_to_ndc;
	                if (to == u_raster) {
	                    Matrix44 ndc_to_raster (m_sr.m_xres, 0, 0, 0,
	                                            0, m_sr.m_yres, 0, 0,
	                                            0, 0, 1, 0,
	                                            0, 0, 0, 1);
	                    M = M * ndc_to_raster;
	                }
	            }
	        }
	        result = M;
	        return true;
	    }

	    auto found = m_sr.m_named_xforms.find (to);
	    if (found != m_sr.m_named_xforms.end()) {
	        result = *(found->second);
	        result.invert();
	        return true;
	    } else {
	        return false;
	    }
	}
#endif



SimpleRenderer::SimpleRenderer ()
: m_batched_simple_renderer(*this)
{
    Matrix44 M;  M.makeIdentity();
    camera_params (M, u_perspective, 90.0f,
                   0.1f, 1000.0f, 256, 256);

    // Set up getters
    m_attr_getters[ustring("camera:resolution")] = &SimpleRenderer::get_camera_resolution;
    m_attr_getters[ustring("camera:projection")] = &SimpleRenderer::get_camera_projection;
    m_attr_getters[ustring("camera:pixelaspect")] = &SimpleRenderer::get_camera_pixelaspect;
    m_attr_getters[ustring("camera:screen_window")] = &SimpleRenderer::get_camera_screen_window;
    m_attr_getters[ustring("camera:fov")] = &SimpleRenderer::get_camera_fov;
    m_attr_getters[ustring("camera:clip")] = &SimpleRenderer::get_camera_clip;
    m_attr_getters[ustring("camera:clip_near")] = &SimpleRenderer::get_camera_clip_near;
    m_attr_getters[ustring("camera:clip_far")] = &SimpleRenderer::get_camera_clip_far;
    m_attr_getters[ustring("camera:shutter")] = &SimpleRenderer::get_camera_shutter;
    m_attr_getters[ustring("camera:shutter_open")] = &SimpleRenderer::get_camera_shutter_open;
    m_attr_getters[ustring("camera:shutter_close")] = &SimpleRenderer::get_camera_shutter_close;
}



int
SimpleRenderer::supports (string_view feature) const
{
    return false;
}



void
SimpleRenderer::camera_params (const Matrix44 &world_to_camera,
                               ustring projection, float hfov,
                               float hither, float yon,
                               int xres, int yres)
{
    m_world_to_camera = world_to_camera;
    m_projection = projection;
    m_fov = hfov;
    m_pixelaspect = 1.0f; // hard-coded
    m_hither = hither;
    m_yon = yon;
    m_shutter[0] = 0.0f; m_shutter[1] = 1.0f;  // hard-coded
    float frame_aspect = float(xres)/float(yres) * m_pixelaspect;
    m_screen_window[0] = -frame_aspect;
    m_screen_window[1] = -1.0f;
    m_screen_window[2] =  frame_aspect;
    m_screen_window[3] =  1.0f;
    m_xres = xres;
    m_yres = yres;
}

BatchedRendererServices * 
SimpleRenderer::batched()
{
	return & m_batched_simple_renderer;
}


bool
SimpleRenderer::get_matrix (ShaderGlobals *sg, Matrix44 &result,
                            TransformationPtr xform,
                            float time)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    result = *reinterpret_cast<const Matrix44*>(xform);
    return true;
}



bool
SimpleRenderer::get_matrix (ShaderGlobals *sg, Matrix44 &result,
                            ustring from, float time)
{
    TransformMap::const_iterator found = m_named_xforms.find (from);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        return true;
    } else {
        return false;
    }
}



bool
SimpleRenderer::get_matrix (ShaderGlobals *sg, Matrix44 &result,
                            TransformationPtr xform)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    result = *(OSL::Matrix44 *)xform;
    return true;
}



bool
SimpleRenderer::get_matrix (ShaderGlobals *sg, Matrix44 &result,
                            ustring from)
{
    // SimpleRenderer doesn't understand motion blur, so we never fail
    // on account of time-varying transformations.
    TransformMap::const_iterator found = m_named_xforms.find (from);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        return true;
    } else {
        return false;
    }
}



bool
SimpleRenderer::get_inverse_matrix (ShaderGlobals *sg, Matrix44 &result,
                                    ustring to, float time)
{
    if (to == u_camera || to == u_screen || to == u_NDC || to == u_raster) {
        Matrix44 M = m_world_to_camera;
        if (to == u_screen || to == u_NDC || to == u_raster) {
            float depthrange = (double)m_yon-(double)m_hither;
            if (m_projection == u_perspective) {
                float tanhalffov = tanf (0.5f * m_fov * M_PI/180.0);
                Matrix44 camera_to_screen (1/tanhalffov, 0, 0, 0,
                                           0, 1/tanhalffov, 0, 0,
                                           0, 0, m_yon/depthrange, 1,
                                           0, 0, -m_yon*m_hither/depthrange, 0);
                M = M * camera_to_screen;
            } else {
                Matrix44 camera_to_screen (1, 0, 0, 0,
                                           0, 1, 0, 0,
                                           0, 0, 1/depthrange, 0,
                                           0, 0, -m_hither/depthrange, 1);
                M = M * camera_to_screen;
            }
            if (to == u_NDC || to == u_raster) {
                float screenleft = -1.0, screenwidth = 2.0;
                float screenbottom = -1.0, screenheight = 2.0;
                Matrix44 screen_to_ndc (1/screenwidth, 0, 0, 0,
                                        0, 1/screenheight, 0, 0,
                                        0, 0, 1, 0,
                                        -screenleft/screenwidth, -screenbottom/screenheight, 0, 1);
                M = M * screen_to_ndc;
                if (to == u_raster) {
                    Matrix44 ndc_to_raster (m_xres, 0, 0, 0,
                                            0, m_yres, 0, 0,
                                            0, 0, 1, 0,
                                            0, 0, 0, 1);
                    M = M * ndc_to_raster;
                }
            }
        }
        result = M;
        return true;
    }

    TransformMap::const_iterator found = m_named_xforms.find (to);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        result.invert();
        return true;
    } else {
        return false;
    }
}





    


void
SimpleRenderer::name_transform (const char *name, const OSL::Matrix44 &xform)
{
    std::shared_ptr<Transformation> M (new OSL::Matrix44 (xform));
    m_named_xforms[ustring(name)] = M;
}



bool
SimpleRenderer::get_array_attribute (ShaderGlobals *sg, bool derivatives, ustring object,
                                     TypeDesc type, ustring name,
                                     int index, void *val)
{
    AttrGetterMap::const_iterator g = m_attr_getters.find (name);
    if (g != m_attr_getters.end()) {
        AttrGetter getter = g->second;
        return (this->*(getter)) (sg, derivatives, object, type, name, val);
    }

    // In order to test getattribute(), respond positively to
    // "options"/"blahblah"
    if (object == "options" && name == "blahblah" &&
        type == TypeDesc::TypeFloat) {
        *(float *)val = 3.14159;
        return true;
    }

    // If no named attribute was found, allow userdata to bind to the
    // attribute request.
    if (object.empty() && index == -1)
        return get_userdata (derivatives, name, type, sg, val);

    return false;
}



bool
SimpleRenderer::get_attribute (ShaderGlobals *sg, bool derivatives, ustring object,
                               TypeDesc type, ustring name, void *val)
{
    return get_array_attribute (sg, derivatives, object,
                                type, name, -1, val);
}



bool
SimpleRenderer::get_userdata (bool derivatives, ustring name, TypeDesc type,
                              ShaderGlobals *sg, void *val)
{
    // Just to illustrate how this works, respect s and t userdata, filled
    // in with the uv coordinates.  In a real renderer, it would probably
    // look up something specific to the primitive, rather than have hard-
    // coded names.

	ASSERT(false && "unsupported for batch mode, not refactored this function");
    if (name == u_s && type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = sg->u;
        if (derivatives) {
            ((float *)val)[1] = sg->dudx;
            ((float *)val)[2] = sg->dudy;
        }
        return true;
    }
    if (name == u_t && type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = sg->v;
        if (derivatives) {
            ((float *)val)[1] = sg->dvdx;
            ((float *)val)[2] = sg->dvdy;
        }
        return true;
    }

    return false;
}


bool
SimpleRenderer::get_camera_resolution (ShaderGlobals *sg, bool derivs, ustring object,
                                    TypeDesc type, ustring name, void *val)
{
    if (type == TypeIntArray2) {
        ((int *)val)[0] = m_xres;
        ((int *)val)[1] = m_yres;
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_projection (ShaderGlobals *sg, bool derivs, ustring object,
                                    TypeDesc type, ustring name, void *val)
{
    if (type == TypeDesc::TypeString) {
        ((ustring *)val)[0] = m_projection;
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_fov (ShaderGlobals *sg, bool derivs, ustring object,
                                    TypeDesc type, ustring name, void *val)
{
    // N.B. in a real rederer, this may be time-dependent
    if (type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = m_fov;
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_pixelaspect (ShaderGlobals *sg, bool derivs, ustring object,
                                    TypeDesc type, ustring name, void *val)
{
    if (type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = m_pixelaspect;
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_clip (ShaderGlobals *sg, bool derivs, ustring object,
                                    TypeDesc type, ustring name, void *val)
{
    if (type == TypeFloatArray2) {
        ((float *)val)[0] = m_hither;
        ((float *)val)[1] = m_yon;
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_clip_near (ShaderGlobals *sg, bool derivs, ustring object,
                                    TypeDesc type, ustring name, void *val)
{
    if (type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = m_hither;
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_clip_far (ShaderGlobals *sg, bool derivs, ustring object,
                                    TypeDesc type, ustring name, void *val)
{
    if (type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = m_yon;
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}



bool
SimpleRenderer::get_camera_shutter (ShaderGlobals *sg, bool derivs, ustring object,
                                    TypeDesc type, ustring name, void *val)
{
    if (type == TypeFloatArray2) {
        ((float *)val)[0] = m_shutter[0];
        ((float *)val)[1] = m_shutter[1];
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_shutter_open (ShaderGlobals *sg, bool derivs, ustring object,
                                    TypeDesc type, ustring name, void *val)
{
    if (type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = m_shutter[0];
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_shutter_close (ShaderGlobals *sg, bool derivs, ustring object,
                                    TypeDesc type, ustring name, void *val)
{
    if (type == TypeDesc::TypeFloat) {
        ((float *)val)[0] = m_shutter[1];
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_screen_window (ShaderGlobals *sg, bool derivs, ustring object,
                                    TypeDesc type, ustring name, void *val)
{
    // N.B. in a real rederer, this may be time-dependent
    if (type == TypeFloatArray4) {
        ((float *)val)[0] = m_screen_window[0];
        ((float *)val)[1] = m_screen_window[1];
        ((float *)val)[2] = m_screen_window[2];
        ((float *)val)[3] = m_screen_window[3];
        if (derivs)
            memset ((char *)val+type.size(), 0, 2*type.size());
        return true;
    }
    return false;
}




OSL_NAMESPACE_EXIT
