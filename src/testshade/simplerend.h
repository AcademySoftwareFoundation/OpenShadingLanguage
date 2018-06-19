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

#pragma once

#include <map>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <OpenImageIO/ustring.h>
#include "OSL/oslexec.h"

OSL_NAMESPACE_ENTER


void register_closures(OSL::ShadingSystem* shadingsys);


class SimpleRenderer;

class BatchedSimpleRenderer : public BatchedRendererServices
{
	friend class SimpleRenderer;
public:
	BatchedSimpleRenderer(SimpleRenderer &sr);
	virtual ~BatchedSimpleRenderer();
		
	Mask get_matrix (ShaderGlobalsBatch *sgb,
			                 MaskedAccessor<Matrix44> result,
		                     ConstWideAccessor<TransformationPtr> xform,
							 ConstWideAccessor<float> time) override;
	Mask get_matrix (ShaderGlobalsBatch *sgb,
			MaskedAccessor<Matrix44> result,
							 ustring from,
							 ConstWideAccessor<float> time) override;
    Mask get_matrix (ShaderGlobalsBatch *sgb,
    						 MaskedAccessor<Matrix44> result,
    						 ConstWideAccessor<ustring> from,
							 ConstWideAccessor<float> time) override;

	bool get_matrix (ShaderGlobalsBatch *sgb, Matrix44 &result,
							 TransformationPtr xform) override;
	bool get_matrix (ShaderGlobalsBatch *sgb, Matrix44 &result,
							 ustring from) override;
	void trace (TraceOpt &options,  ShaderGlobalsBatch *sgb, MaskedAccessor<int> result,
            ConstWideAccessor<Vec3> P, ConstWideAccessor<Vec3> dPdx,
            ConstWideAccessor<Vec3> dPdy, ConstWideAccessor<Vec3> R,
            ConstWideAccessor<Vec3> dRdx, ConstWideAccessor<Vec3> dRdy) override;

private:
	template<typename RAccessorT>
	OSL_INLINE bool impl_get_inverse_matrix (
	    RAccessorT & result,
	    ustring to) const;

public:
	Mask get_inverse_matrix (ShaderGlobalsBatch *sgb, MaskedAccessor<Matrix44> result,
                             ustring to, ConstWideAccessor<float> time) override;
    Mask get_inverse_matrix (ShaderGlobalsBatch *sgb, MaskedAccessor<Matrix44> result,
                             ConstWideAccessor<ustring> to, ConstWideAccessor<float> time) override;

	
	bool is_attribute_uniform(ustring object, ustring name) override;
	
    Mask get_array_attribute (ShaderGlobalsBatch *sgb,
                                      ustring object, ustring name,
                                      int index, MaskedDataRef amd) override;
    
    Mask get_attribute (ShaderGlobalsBatch *sgb, ustring object,
                                ustring name, MaskedDataRef amd) override;

    
    bool get_array_attribute_uniform (ShaderGlobalsBatch *sgb,
                                      ustring object, ustring name,
                                      int index, DataRef val) override;
    
    bool get_attribute_uniform (ShaderGlobalsBatch *sgb, ustring object,
                                ustring name, DataRef val) override;
    
#ifdef OSL_EXPERIMENTAL_BIND_USER_DATA_WITH_LAYERNAME
    Mask get_userdata (ustring name, ustring layername,
    						   ShaderGlobalsBatch *sgb, MaskedDataRef val) override;
#else
    Mask get_userdata (ustring name,
                               ShaderGlobalsBatch *sgb, MaskedDataRef val) override;
#endif

private:
	SimpleRenderer &m_sr;
	std::unordered_set<ustring, ustringHash> m_uniform_objects;
	std::unordered_set<ustring, ustringHash> m_uniform_attributes;
};

class SimpleRenderer : public RendererServices
{
	friend class BatchedSimpleRenderer;
public:
    // Just use 4x4 matrix for transformations
    typedef Matrix44 Transformation;

    SimpleRenderer ();
    ~SimpleRenderer () { }

    virtual int supports (string_view feature) const;
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             TransformationPtr xform,
                             float time);
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             ustring from, float time);
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             TransformationPtr xform);
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             ustring from);
    virtual bool get_inverse_matrix (ShaderGlobals *sg, Matrix44 &result,
                                     ustring to, float time);

    void name_transform (const char *name, const Transformation &xform);
    
    virtual bool get_array_attribute (ShaderGlobals *sg, bool derivatives, 
                                      ustring object, TypeDesc type, ustring name,
                                      int index, void *val );
    
    virtual bool trace (TraceOpt &options, ShaderGlobals *sg,
                            const OSL::Vec3 &P, const OSL::Vec3 &dPdx,
                            const OSL::Vec3 &dPdy, const OSL::Vec3 &R,
                            const OSL::Vec3 &dRdx, const OSL::Vec3 &dRdy);


    // Common impl shared with BatchedSimpleRenderer  
    bool common_get_attribute(ustring object, ustring name, DataRef val);
    
    virtual bool get_attribute (ShaderGlobals *sg, bool derivatives, ustring object,
                                TypeDesc type, ustring name, void *val);

#ifdef OSL_EXPERIMENTAL_BIND_USER_DATA_WITH_LAYERNAME
    virtual bool get_userdata (bool derivatives, ustring name, ustring layername, TypeDesc type,
                               ShaderGlobals *sg, void *val);
#else
    virtual bool get_userdata (bool derivatives, ustring name, TypeDesc type, 
                               ShaderGlobals *sg, void *val);
#endif
    // Super simple camera and display parameters.  Many options not
    // available, no motion blur, etc.
    void camera_params (const Matrix44 &world_to_camera, ustring projection,
                        float hfov, float hither, float yon,
                        int xres, int yres);

    virtual BatchedRendererServices * batched();    
private:
    BatchedSimpleRenderer m_batched_simple_renderer;

private:
    // Camera parameters
    Matrix44 m_world_to_camera;
    ustring m_projection;
    float m_fov, m_pixelaspect, m_hither, m_yon;
    float m_shutter[2];
    float m_screen_window[4];
    int m_xres, m_yres;

    // Named transforms
    typedef std::map <ustring, std::shared_ptr<Transformation> > TransformMap;
    TransformMap m_named_xforms;
    
    // Attribute and userdata retrieval -- for fast dispatch, use a hash
    // table to map attribute names to functions that retrieve them. We
    // imagine this to be fairly quick, but for a performance-critical
    // renderer, we would encourage benchmarking various methods and
    // alternate data structures.
    typedef bool (SimpleRenderer::*AttrGetter)(ustring object, ustring name, DataRef val);
    typedef std::unordered_map<ustring, AttrGetter, ustringHash> AttrGetterMap;
    AttrGetterMap m_attr_getters;

    // Attribute getters
    bool get_camera_resolution (ustring object, ustring name, DataRef val);
    bool get_camera_projection (ustring object, ustring name, DataRef val);
    bool get_camera_fov (ustring object, ustring name, DataRef val);
    bool get_camera_pixelaspect (ustring object, ustring name, DataRef val);
    bool get_camera_clip (ustring object, ustring name, DataRef val);
    bool get_camera_clip_near (ustring object, ustring name, DataRef val);
    bool get_camera_clip_far (ustring object, ustring name, DataRef val);
    bool get_camera_shutter (ustring object, ustring name, DataRef val);
    bool get_camera_shutter_open (ustring object, ustring name, DataRef val);
    bool get_camera_shutter_close (ustring object, ustring name, DataRef val);
    bool get_camera_screen_window (ustring object, ustring name, DataRef val);

};

OSL_NAMESPACE_EXIT
