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

#ifdef __OSL_DEBUG_MISSING_USER_DATA
	#include <unordered_set>
#endif

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
static ustring u_lookupTable("lookupTable");
static ustring u_blahblah("blahblah");
static ustring u_options("options");
static ustring u_global("global");



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



BatchedSimpleRenderer::BatchedSimpleRenderer(SimpleRenderer &sr)
    : m_sr(sr)
{
	m_uniform_objects.insert(u_global);	
	m_uniform_objects.insert(u_options);		
}

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

    OSL_OMP_PRAGMA(omp simd simdlen(SimdLaneCount))
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
    OSL_INTEL_PRAGMA(simd reduction(+:numLanesMatch1st)  assert)
	for(int lane=0; lane < SimdLaneCount; ++lane) {
        int match = (uniform_xform == xform.get(lane)) ? 1 : 0;
        numLanesMatch1st += match;
    }

    if (numLanesMatch1st == 8) {
#elif 0
    int numLanesMatch1st = 0;
    OSL_INTEL_PRAGMA(simd reduction(+:numLanesMatch1st) vectorlength(8) assert)
            for(int lane=0; lane < SimdLaneCount; ++lane) {
        numLanesMatch1st += (uniform_xform == xform.get(lane));
    }

    if (numLanesMatch1st == 8) {

#else
    register __m256i xformPointers  = _mm2565_load_epi32(xform.data);

    if (1)

#endif
        const Matrix44 & transformFromShaderGlobals = *reinterpret_cast<const Matrix44*>(uniform_xform);
    OSL_OMP_PRAGMA(omp simd simdlen(SimdLaneCount))
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
Mask
BatchedSimpleRenderer::get_matrix (
	ShaderGlobalsBatch *sgb, 
	MaskedAccessor<Matrix44> result,
    ConstWideAccessor<TransformationPtr> xform,
    ConstWideAccessor<float> time)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    //result = *reinterpret_cast<const Matrix44*>(xform);

    TransformationPtr uniform_xform = xform[0];

#if 0
    // In general, one can't assume that the transformation is uniform
    const Matrix44 & uniformTransform = *reinterpret_cast<const Matrix44*>(uniform_xform);
    OSL_OMP_PRAGMA(omp simd simdlen(result.width))
    for(int lane=0; lane < result.width; ++lane) {
        if (__builtin_expect((uniform_xform == xform[lane]),1)) {
            result[lane] = uniformTransform;
        } else {
            const Matrix44 & transformFromShaderGlobals = *reinterpret_cast<const Matrix44*>(xform.get(lane));
            result[lane] = transformFromShaderGlobals;
        }
    }
#else
    // But this is "testshade" and we know we only have one object, so lets just
    // use that fact
    const Matrix44 & uniformTransform = *reinterpret_cast<const Matrix44*>(uniform_xform);

	// Workaround clang omp when it cant perform a runtime pointer check
	// to ensure no overlap in output variables
	// But we can tell clang to assume its safe
	OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(result.width))
	OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(result.width))
    for(int lane=0; lane < result.width; ++lane) {
        result[lane] = uniformTransform;
    }

#endif

    return Mask(true);
}


#endif
Mask
BatchedSimpleRenderer::get_matrix (
	ShaderGlobalsBatch * /*sgb*/,
	MaskedAccessor<Matrix44> wresult,
	ustring from,
	ConstWideAccessor<float> /*wtime*/)
{
    auto found = m_sr.m_named_xforms.find (from);
    if (found != m_sr.m_named_xforms.end()) {
        const Matrix44 & uniformTransform =  *(found->second);
        
		// Workaround clang omp when it cant perform a runtime pointer check
        // to ensure no overlap in output variables
        // But we can tell clang to assume its safe
    	OSL_OMP_AND_CLANG_PRAGMA(clang loop vectorize(assume_safety) vectorize_width(wresult.width))
    	OSL_OMP_NOT_CLANG_PRAGMA(omp simd simdlen(wresult.width))

        for(int lane=0; lane < wresult.width; ++lane) {
            wresult[lane] = uniformTransform;
        }
        
        return Mask(true);
    } else {
        return Mask(false);
    }
}

Mask BatchedSimpleRenderer::get_matrix (
	ShaderGlobalsBatch * /*sgb*/,
	MaskedAccessor<Matrix44> wresult,
	ConstWideAccessor<ustring> wfrom,
	ConstWideAccessor<float> /*wtime*/)
{
	Mask succeeded(false);
    OSL_OMP_PRAGMA(omp simd simdlen(wresult.width))
	for(int lane=0; lane < wresult.width; ++lane) {

		if (wresult.mask().is_on(lane)) {

			ustring from = wfrom[lane];
			auto found = m_sr.m_named_xforms.find (from);
			if (found != m_sr.m_named_xforms.end()) {
				const Matrix44 & transform =  *(found->second);
				wresult[lane] = transform;
				succeeded.set_on(lane);
			}
		}
	}
	return succeeded;
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

bool 
BatchedSimpleRenderer::is_attribute_uniform(ustring object, ustring name)
{
	
	
	if (m_uniform_objects.find(object) != m_uniform_objects.end())
		return true;

	if ((!object) && m_uniform_attributes.find(name) != m_uniform_attributes.end())
		return true;
		
	return false;
}


Mask
BatchedSimpleRenderer::get_array_attribute (ShaderGlobalsBatch *sgb, 
                                            ustring object, ustring name,
                                            int index, MaskedDataRef val)
{
	ASSERT(is_attribute_uniform(object, name) == false);
	
    if (object == nullptr && name == u_blahblah) {
    	if(val.is<float>()) {
			auto out = val.masked<float>();
			for(int i=0; i < out.width; ++i) {
#if 0 
				// Example of how to test the lane of the proxy 
				// if you wantto skip expensive code on the right
				// hand side of the assignment, such as a function call
				if(out[i].is_on()) {
					out[i] = 1.0f - sgb->varying(i).P().get().x;
				}
#else
				// Masking is silently handled by the assignment operator
				// of the proxy out[i]
				out[i] = 1.0f - Vec3(sgb->varying(i).P()).x;
#endif
			}
			return val.mask();
    	} else if(val.is<Vec3>()) {    		
			auto out = val.masked<Vec3>();
			for(int i=0; i < out.width; ++i) {							
				out[i] = Vec3(1.0f) - sgb->varying(i).P();
			}
			return val.mask();
    	}      
    }
    
    if (object == nullptr && name == u_lookupTable) {
    	
    	if(val.is<float[]>())
    	{
    		auto out = val.masked<float[]>();
			for(int lane_index=0; lane_index < out.width; ++lane_index) {
				
				auto lut = out[lane_index];
	    		for(int i=0; i < lut.length(); ++i)
	    		{
	    			lut[i] = 1.0 - float(i)/float(lut.length());
	    		}
			
			}    		
			return Mask(true);
    	}
    }
    
    
    if (object == nullptr && name == "not_a_color") {
    	if(val.is<float[3]>()) {
			float valArray[3] = { 0.0f, 0.5f, 1.0f };
    		
			auto out = val.masked<float[3]>();
			
			for(int i=0; i < out.width; ++i) {
				out[i] = valArray;
			}

			
#if 0 // Some test code to show that arrays can be extracted
	  // from the proxy out[i] in multiple ways
			for(int i=0; i < out.width; ++i) {
				if(out[i].is_on()) {
					float valArray[3];
					// Illegal to return an array by value
					// so will have to pass the local array to
					// be populated
					out[i].get(valArray);
					// Alternatively can use the [arrayindex] operator on 
					// the proxy out[i] to get access to underlying data
					valArray[0] = out[i][0];
					valArray[1] = out[i][1];
					valArray[2] = out[i][2];
					

					valArray[0] = 0.0;
					valArray[1] = 0.5;
					valArray[2] = 1.0;
					
					// NOTE: proxy handles pulling values out of the
					// the local array and distributing to the
					// correct wide arrays, including testing
					// the mask during assignment
					out[i] = valArray;
				}
			}
#endif
			return val.mask();
    	} 
    }
    
#if 0 // extra hackery to create a reproducer 
    if (name == ustring("user:testvartype") && val.is<float>()) {
    	auto out = val.masked<float>();
		
		for(int i=0; i < 4; ++i) {
			out[i] = 0.0;
		}
		for(int i=4; i < 8; ++i) {
			out[i] = 1.0;
		}
		for(int i=8; i < 12; ++i) {
			out[i] = 2.0;
		}
		for(int i=12; i < 16; ++i) {
			out[i] = -1;
		}
		
    	
        return val.mask();
		//return Mask(0xFF);
    	//return Mask(false);
    }
    if (object == ustring("primvar") && name == ustring("vryfloat") && val.is<float>()) {
    	
    	std::cout << "MADE IT, mask=" << val.mask().value() << std::endl;
    	auto out = val.masked<float>();
		for(int i=0; i < out.width; ++i) {
			out[i] = 2.0;
		}
    	
        //return val.mask();
		return Mask(true);
    }
    
    if (object == ustring("primvar") && name == ustring("unffloat") && val.is<float>()) {
    	
    	std::cout << "MADE IT, mask=" << val.mask().value() << std::endl;
    	auto out = val.masked<float>();
		for(int i=0; i < out.width; ++i) {
			out[i] = 2.0;
		}
    	
        //return val.mask();
		return Mask(true);
    }
    
#endif
    
    // If no named attribute was found, allow userdata to bind to the
    // attribute request.
    if (object.empty() && index == -1)
#ifdef OSL_EXPERIMENTAL_BIND_USER_DATA_WITH_LAYERNAME
        return get_userdata (name, ustring(), sgb, val);
#else
        return get_userdata (name, sgb, val);
#endif

    return Mask(false);
}

Mask
BatchedSimpleRenderer::get_attribute (ShaderGlobalsBatch *sgb, ustring object,
                            ustring name, MaskedDataRef val)
{
	
    return get_array_attribute (sgb, object,
                                name, -1, val);
}



bool
BatchedSimpleRenderer::get_array_attribute_uniform (ShaderGlobalsBatch *sgb, 
                                            ustring object, ustring name,
                                            int index, DataRef val)
{
	
	ASSERT(!name.empty());

	if (m_sr.common_get_attribute (object, name, val) )
		return true;
	
	// NOTE: we do not bother calling through to get_userdata
	// The only way to get inside this call was to of had the
	// is_attribute_uniform return true.  It is a logic bug on the renderer
	// if it claims user data is uniform.
	// TODO: validate the above comment
	return false;
}

bool
BatchedSimpleRenderer::get_attribute_uniform (ShaderGlobalsBatch *sgb, ustring object,
                            ustring name, DataRef val)
{
    return get_array_attribute_uniform (sgb, object,
                                		name, -1, val);
}

#if OSL_EXPERIMENTAL_BIND_USER_DATA_WITH_LAYERNAME
Mask
BatchedSimpleRenderer::get_userdata (ustring name, ustring layername,
									 ShaderGlobalsBatch *sgb, MaskedDataRef val)
#else
Mask
BatchedSimpleRenderer::get_userdata (ustring name,
                                     ShaderGlobalsBatch *sgb, MaskedDataRef val)
#endif
{
    // Just to illustrate how this works, respect s and t userdata, filled
    // in with the uv coordinates.  In a real renderer, it would probably
    // look up something specific to the primitive, rather than have hard-
    // coded names.
	
	// For testing of interactions with default values
	// may not provide data for all lanes
    if (name == u_s && val.is<float>()) {
    
    	auto out = val.masked<float>();
		for(int i=0; i < out.width; ++i) {
			// NOTE: assigning to out[i] will mask by itself
			// this check is just to show how you could do it if you
			// wanted to skip executing right hand side of assignment
			if(out[i].is_on()) {
                out[i] = sgb->varying(i).u();
			}
		}
        if (val.has_derivs()) {
        	auto out_dx = val.maskedDx<float>();
    		for(int i=0; i < out_dx.width; ++i) {
    			out_dx[i] = sgb->varying(i).dudx();
    		}
        	auto out_dy = val.maskedDy<float>();
    		for(int i=0; i < out.width; ++i) {
    			out_dy[i] = sgb->varying(i).dudy();
    		}
        }
    	
        return val.mask();
    }
    if (name == u_t && val.is<float>()) {
    	auto out = val.masked<float>();
		for(int i=0; i < out.width; ++i) {
			out[i] = sgb->varying(i).v();
		}
        if (val.has_derivs()) {
        	auto out_dx = val.maskedDx<float>();
    		for(int i=0; i < out_dx.width; ++i) {
    			out_dx[i] = sgb->varying(i).dvdx();
    		}
        	auto out_dy = val.maskedDy<float>();
    		for(int i=0; i < out.width; ++i) {
    			out_dy[i] = sgb->varying(i).dvdy();
    		}
        }
    	
        return val.mask();
    }

#ifdef __OSL_DEBUG_MISSING_USER_DATA
    static std::unordered_set<ustring> missingUserData;
    if (missingUserData.find(name) == missingUserData.end()) {
        std::cout << "Missing user data for " << name << std::endl;
        missingUserData.insert(name);
    }
#endif
    
    return Mask(false);
}


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

    for(const auto & entry: m_attr_getters)
    {
    	m_batched_simple_renderer.m_uniform_attributes.insert(entry.first);
    }
    m_batched_simple_renderer.m_uniform_attributes.insert(u_lookupTable);
    
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
SimpleRenderer::common_get_attribute (ustring object, ustring name,
		  DataRef val)
{
	//std::cout << " common_get_attribute object=" << (object.empty() ? "" : object.c_str() ) << " name = " << name.c_str() << std::endl;
	

    AttrGetterMap::const_iterator g = m_attr_getters.find (name);
    if (g != m_attr_getters.end()) {
        AttrGetter getter = g->second;
        return (this->*(getter)) (object, name, val);
    }

    // In order to test getattribute(), respond positively to
    // "options"/"blahblah"
    if (object == u_options && name == u_blahblah && val.is<float>()) {
    	val.ref<float>() = 3.14159;
        return true;
    }
    
    
    if (/*object == nullptr &&*/ name == u_lookupTable) {
#if 0 // old way of checking typedesc and arraylength
    	if (val.type().is_array() && val.type().basetype == TypeDesc::FLOAT && val.type().aggregate == TypeDesc::SCALAR)
    	{
    		float * lt = reinterpret_cast<float *>(val.ptr());
    		for(int i=0; i < val.type().arraylen; ++i)
    		{
    			lt[i] = 1.0 - float(i)/float(val.type().arraylen);
    		}
    		return true;
    	}
#endif
    	// New way of checking for array of a given type
    	if (val.is<float[]>()) {    
			auto lut = val.ref<float[]>();
			for(int i=0; i < lut.length(); ++i)
			{
				lut[i] = 1.0 - float(i)/float(lut.length());
			}
			return true;
    	}
    }
    
    return false;
}



bool
SimpleRenderer::get_array_attribute (ShaderGlobals *sg, bool derivatives, ustring object,
                                     TypeDesc type, ustring name,
                                     int index, void *val_ptr)
{
	DataRef val(type, derivatives, val_ptr);
	if (common_get_attribute (object, name, val) )
		return true;
	
    if (object == nullptr && name == u_blahblah) {
		if (val.is<float>()) {
			val.ref<float>() = 1.0f - sg->P.x;
			return true;
		} else if (val.is<Vec3>()) {
			val.ref<Vec3>() = Vec3(1.0f) - sg->P;
			return true;
		}

    }

    // If no named attribute was found, allow userdata to bind to the
    // attribute request.
    if (object.empty() && index == -1)
#if OSL_EXPERIMENTAL_BIND_USER_DATA_WITH_LAYERNAME
        return get_userdata (derivatives, name, ustring(), type, sg, val_ptr);
#else
        return get_userdata (derivatives, name, type, sg, val_ptr);
#endif

    return false;
}

bool
SimpleRenderer::get_attribute (ShaderGlobals *sg, bool derivatives, ustring object,
                               TypeDesc type, ustring name, void *val)
{
    return get_array_attribute (sg, derivatives, object,
                                type, name, -1, val);
}



#if OSL_EXPERIMENTAL_BIND_USER_DATA_WITH_LAYERNAME
bool
SimpleRenderer::get_userdata (bool derivatives, ustring name, ustring layername, TypeDesc type,
                              ShaderGlobals *sg, void *val_ptr)
#else
bool
SimpleRenderer::get_userdata (bool derivatives, ustring name, TypeDesc type,
                              ShaderGlobals *sg, void *val_ptr)
#endif
{
    // Just to illustrate how this works, respect s and t userdata, filled
    // in with the uv coordinates.  In a real renderer, it would probably
    // look up something specific to the primitive, rather than have hard-
    // coded names.
	DataRef val(type, derivatives, val_ptr);

    if (name == u_s && val.is<float>()) {
        val.ref<float>() = sg->u;

        if (val.has_derivs()) {
            val.refDx<float>() = sg->dudx;
            val.refDy<float>() = sg->dudy;
        }
        return true;
    }
    if (name == u_t && type == val.is<float>()) {
    	val.ref<float>() = sg->v;
        if (derivatives) {
        	val.refDx<float>() = sg->dvdx;
        	val.refDy<float>() = sg->dvdy;
        }
        return true;
    }

    return false;
}


bool
SimpleRenderer::get_camera_resolution (ustring object, ustring name, DataRef val)
{
    if (val.is<int[2]>()) {    	
    	auto v = val.ref<int[2]>();
        v[0] = m_xres;
        v[1] = m_yres;
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_projection (ustring object, ustring name, DataRef val)
{
    if (val.is<ustring>()) {
        val.ref<ustring>() = m_projection;
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_fov (ustring object, ustring name, DataRef val)
{
    // N.B. in a real rederer, this may be time-dependent
    if (val.is<float>()) {
        val.ref<float>() = m_fov;
        if(val.has_derivs())
        {
			val.refDx<float>() = 0.0f;
			val.refDy<float>() = 0.0f;
        }
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_pixelaspect (ustring object, ustring name, DataRef val)
{
    if (val.is<float>()) {
        val.ref<float>() = m_pixelaspect;
        if(val.has_derivs())
        {
			val.refDx<float>() = 0.0f;
			val.refDy<float>() = 0.0f;
        }
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_clip (ustring object, ustring name, DataRef val)
{
    if (val.is<float[2]>()) {
    	
    	float clip[2];
    	clip[0] = m_hither;
    	clip[1] = m_yon;
    	
    	// various ways of assigning through a DataRef proxy
        val.ref<float[2]>() = clip;
    	
        //val.ref<float[2]>()[0] = clip[0];
        //val.ref<float[2]>()[1] = clip[1];
    	
    	//auto r = val.ref<float[2]>();
    	//r = clip;
    	
    	//typedef float (&RefFloat2)[2];
    	//RefFloat2 ar = val.ref<float[2]>();
    	//ar[0] = clip[0];
    	//ar[1] = clip[1];
        
        if(val.has_derivs())
        {        	
        	float zero2[2] = {0.0f, 0.0f};
			val.refDx<float[2]>() = zero2;
			val.refDy<float[2]>() = zero2;
        }
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_clip_near (ustring object, ustring name, DataRef val)
{
    if (val.is<float>()) {
    	
		// various ways of assigning through a DataRef proxy
        val.ref<float>() = m_hither;
    	
    	//float & r =val.ref<float>();
    	//r=m_hither;

    	// NOTE: r is a proxy object, so no need to think about &, 
    	// as the reference is inside the object
    	//auto r =val.ref<float>();
    	//r=m_hither;
    	
    	
        if(val.has_derivs())
        {
			val.refDx<float>() = 0.0f;
			val.refDy<float>() = 0.0f;
        }
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_clip_far (ustring object, ustring name, DataRef val)
{
    if (val.is<float>()) {
        val.ref<float>() = m_yon;
        if(val.has_derivs())
        {
			val.refDx<float>() = 0.0f;
			val.refDy<float>() = 0.0f;
        }
        return true;
    }
    return false;
}



bool
SimpleRenderer::get_camera_shutter (ustring object, ustring name, DataRef val)
{
    if (val.is<float[2]>()) {    	
        val.ref<float[2]>() = m_shutter;        
        if(val.has_derivs())
        {        	
        	float zero2[2] = {0.0f, 0.0f};
			val.refDx<float[2]>() = zero2;
			val.refDy<float[2]>() = zero2;
        }
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_shutter_open (ustring object, ustring name, DataRef val)
{
    if (val.is<float>()) {
        val.ref<float>() = m_shutter[0];
        if(val.has_derivs())
        {
			val.refDx<float>() = 0.0f;
			val.refDy<float>() = 0.0f;
        }
        return true;
    }
    return false;    
}


bool
SimpleRenderer::get_camera_shutter_close (ustring object, ustring name, DataRef val)
{
    if (val.is<float>()) {
        val.ref<float>() = m_shutter[1];
        if(val.has_derivs())
        {
			val.refDx<float>() = 0.0f;
			val.refDy<float>() = 0.0f;
        }
        return true;
    }
    return false;
}


bool
SimpleRenderer::get_camera_screen_window (ustring object, ustring name, DataRef val)
{
    // N.B. in a real rederer, this may be time-dependent
    if (val.is<float[4]>()) {    	
        val.ref<float[4]>() = m_screen_window;        
        if(val.has_derivs())
        {        	
        	float zero4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
			val.refDx<float[4]>() = zero4;
			val.refDy<float[4]>() = zero4;
        }
        return true;
    }
    return false;
    
}




OSL_NAMESPACE_EXIT
