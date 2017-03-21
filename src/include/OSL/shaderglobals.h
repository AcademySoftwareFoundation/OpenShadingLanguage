/*
Copyright (c) 2009-2013 Sony Pictures Imageworks Inc., et al.
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

#include <OSL/wide.h>

OSL_NAMESPACE_ENTER

struct ClosureColor;
class ShadingContext;
class RendererServices;
class ShaderSymbol;



/// Type for an opaque pointer to whatever the renderer uses to represent a
/// coordinate transformation.
typedef const void * TransformationPtr;




/// The ShaderGlobals structure represents the state describing a particular
/// point to be shaded. It serves two primary purposes: (1) it holds the
/// values of the "global" variables accessible from a shader (such as P, N,
/// Ci, etc.); (2) it serves as a means of passing (via opaque pointers)
/// additional state between the renderer when it invokes the shader, and
/// the RendererServices that fields requests from OSL back to the renderer.
///
/// Except where noted, it is expected that all values are filled in by the
/// renderer before passing it to ShadingSystem::execute() to actually run
/// the shader. Not all fields will be valid in all contexts. In particular,
/// a few are only needed for lights and volumes.
///
/// All points, vectors and normals are given in "common" space.
///
struct ShaderGlobals {

    /// Surface position (and its x & y differentials).
    Vec3 P, dPdx, dPdy;
    /// P's z differential, used for volume shading only.
    Vec3 dPdz;

    /// Incident ray, and its x and y derivatives.
    Vec3 I, dIdx, dIdy;

    /// Shading normal, already front-facing.
    Vec3 N;

    /// True geometric normal.
    Vec3 Ng;

    /// 2D surface parameter u, and its differentials.
    float u, dudx, dudy;
    /// 2D surface parameter v, and its differentials.
    float v, dvdx, dvdy;

    /// Surface tangents: derivative of P with respect to surface u and v.
    Vec3 dPdu, dPdv;

    /// Time for this shading sample.
    float time;
    /// Time interval for the frame (or shading sample).
    float dtime;
    ///  Velocity vector: derivative of position P with respect to time.
    Vec3 dPdtime;

    /// For lights or light attenuation shaders: the point being illuminated
    /// (Ps), and its differentials.
    Vec3 Ps, dPsdx, dPsdy;

    /// There are three opaque pointers that may be set by the renderer here
    /// in the ShaderGlobals before shading execution begins, and then
    /// retrieved again from the within the implementation of various
    /// RendererServices methods. Exactly what they mean and how they are
    /// used is renderer-dependent, but roughly speaking it's probably a
    /// pointer to some internal renderer state (needed for, say, figuring
    /// out how to retrieve userdata), state about the ray tree (needed to
    /// resume for a trace() call), and information about the object being
    /// shaded.
    void* renderstate;
    void* tracedata;
    void* objdata;

    /// Back-pointer to the ShadingContext (set and used by OSL itself --
    /// renderers shouldn't mess with this at all).
    ShadingContext* context;

    /// Pointer to the RendererServices object. This is how OSL finds its
    /// way back to the renderer for callbacks.
    RendererServices* renderer;

    /// Opaque pointers set by the renderer before shader execution, to
    /// allow later retrieval of the object->common and shader->common
    /// transformation matrices, by the RendererServices
    /// get_matrix/get_inverse_matrix methods. This doesn't need to point
    /// to the 4x4 matrix itself; rather, it's just a pointer to whatever
    /// structure the RenderServices::get_matrix() needs to (if and when
    /// requested) generate the 4x4 matrix for the right time value.
    TransformationPtr object2common;
    TransformationPtr shader2common;

    /// The output closure will be placed here. The rendererer should
    /// initialize this to NULL before shading execution, and this is where
    /// it can retrieve the output closure from after shader execution has
    /// completed.
    ClosureColor *Ci;

    /// Surface area of the emissive object (used by light shaders for
    /// energy normalization).
    float surfacearea;

    /// Bit field of ray type flags.
    int raytype;

    /// If nonzero, will flip the result of calculatenormal().
    int flipHandedness;

    /// If nonzero, we are shading the back side of a surface.
    int backfacing;
};

struct UniformShaderGlobals {

    /// There are three opaque pointers that may be set by the renderer here
    /// in the ShaderGlobals before shading execution begins, and then
    /// retrieved again from the within the implementation of various
    /// RendererServices methods. Exactly what they mean and how they are
    /// used is renderer-dependent, but roughly speaking it's probably a
    /// pointer to some internal renderer state (needed for, say, figuring
    /// out how to retrieve userdata), state about the ray tree (needed to
    /// resume for a trace() call), and information about the object being
    /// shaded.
    void* renderstate;
    void* tracedata;
    void* objdata;

    /// Back-pointer to the ShadingContext (set and used by OSL itself --
    /// renderers shouldn't mess with this at all).
    ShadingContext* context;

    /// Pointer to the RendererServices object. This is how OSL finds its
    /// way back to the renderer for callbacks.
    RendererServices* renderer;

    /// The output closure will be placed here. The rendererer should
    /// initialize this to NULL before shading execution, and this is where
    /// it can retrieve the output closure from after shader execution has
    /// completed.
    /// DESIGN DECISION:  NOT CURRENTLY SUPPORTING CLOSURES IN BATCH MODE
    ClosureColor *Ci;

    /// Bit field of ray type flags.
    int raytype;

    // We want to manually pad this structure out to 64 byte boundary
    // and make it simple to duplicate in LLVM without relying on
    // compiler structure alignment rules
    int pad0;
    int pad1;
    int pad2;
    
	void dump()
	{
		#define __OSL_DUMP(VARIABLE_NAME) std::cout << #VARIABLE_NAME << " = " << VARIABLE_NAME << std::endl;
		std::cout << "UniformShaderGlobals = {"  << std::endl;
		__OSL_DUMP(renderstate);
		__OSL_DUMP(tracedata);
		__OSL_DUMP(objdata);
		__OSL_DUMP(context);
		__OSL_DUMP(renderer);
		__OSL_DUMP(Ci); 
		__OSL_DUMP(raytype);

	    std::cout << "};" << std::endl;
		#undef __OSL_DUMP
	}    
};

template<int WidthT> 
struct alignas(64) VaryingShaderGlobals {

	template<typename T>
	using  Wide = OSL::Wide<T, WidthT>;
	
    /// Surface position (and its x & y differentials).
	Wide<Vec3> P, dPdx, dPdy;
    /// P's z differential, used for volume shading only.
	Wide<Vec3> dPdz;

    /// Incident ray, and its x and y derivatives.
	Wide<Vec3> I, dIdx, dIdy;

    /// Shading normal, already front-facing.
	Wide<Vec3> N;

    /// True geometric normal.
	Wide<Vec3> Ng;

    /// 2D surface parameter u, and its differentials.
	Wide<float> u, dudx, dudy;
    /// 2D surface parameter v, and its differentials.
	Wide<float> v, dvdx, dvdy;

    /// Surface tangents: derivative of P with respect to surface u and v.
    Wide<Vec3> dPdu, dPdv;

    /// Time for this shading sample.
    Wide<float> time;
    /// Time interval for the frame (or shading sample).
    Wide<float> dtime;
    ///  Velocity vector: derivative of position P with respect to time.
    Wide<Vec3> dPdtime;

    /// For lights or light attenuation shaders: the point being illuminated
    /// (Ps), and its differentials.
    Wide<Vec3> Ps, dPsdx, dPsdy;

    /// Opaque pointers set by the renderer before shader execution, to
    /// allow later retrieval of the object->common and shader->common
    /// transformation matrices, by the RendererServices
    /// get_matrix/get_inverse_matrix methods. This doesn't need to point
    /// to the 4x4 matrix itself; rather, it's just a pointer to whatever
    /// structure the RenderServices::get_matrix() needs to (if and when
    /// requested) generate the 4x4 matrix for the right time value.
    Wide<TransformationPtr> object2common;
    Wide<TransformationPtr> shader2common;

    /// Surface area of the emissive object (used by light shaders for
    /// energy normalization).
    Wide<float> surfacearea;

    /// If nonzero, will flip the result of calculatenormal().
    Wide<int> flipHandedness;

    /// If nonzero, we are shading the back side of a surface.
    Wide<int> backfacing;
    
	void dump()
	{
		#define __OSL_DUMP(VARIABLE_NAME) VARIABLE_NAME.dump(#VARIABLE_NAME)
		std::cout << "VaryingShaderGlobals = {"  << std::endl;
		__OSL_DUMP(P);
		__OSL_DUMP(dPdx);
		__OSL_DUMP(dPdy);
		__OSL_DUMP(dPdz);
		__OSL_DUMP(I);
		__OSL_DUMP(dIdx); 
		__OSL_DUMP(dIdy);
		__OSL_DUMP(N);
		__OSL_DUMP(Ng);
		__OSL_DUMP(u); 
		__OSL_DUMP(dudx); 
		__OSL_DUMP(dudy);
		__OSL_DUMP(v); 
		__OSL_DUMP(dvdx); 
		__OSL_DUMP(dvdy);
		__OSL_DUMP(dPdu); 
		__OSL_DUMP(dPdv);
		__OSL_DUMP(time);
		__OSL_DUMP(dtime);
		__OSL_DUMP(dPdtime);
		__OSL_DUMP(Ps); 
		__OSL_DUMP(dPsdx); 
		__OSL_DUMP(dPsdy);
		__OSL_DUMP(object2common);
		__OSL_DUMP(shader2common);
		__OSL_DUMP(surfacearea);
		__OSL_DUMP(flipHandedness);
		__OSL_DUMP(backfacing);
	    std::cout << "};" << std::endl;
		#undef __OSL_DUMP
	}
    
};

template<int WidthT> 
struct VaryingShaderProxy {
private:
	VaryingShaderGlobals<WidthT> & m_vsg;
	int m_index;
	
public:
	
	VaryingShaderProxy(VaryingShaderGlobals<WidthT> & vsg, int index)
	: m_vsg(vsg)
	, m_index(index)
	{}

	template<typename T>
	using  Proxy = WideProxy<T, WidthT>;
		
    /// Surface position (and its x & y differentials).
	Proxy<Vec3> P() const { return Proxy<Vec3>(m_vsg.P, m_index); }
	Proxy<Vec3> dPdx() const { return Proxy<Vec3>(m_vsg.dPdx, m_index); }
	Proxy<Vec3> dPdy() const { return Proxy<Vec3>(m_vsg.dPdy, m_index); }
	
    /// P's z differential, used for volume shading only.
	Proxy<Vec3> dPdz() const { return Proxy<Vec3>(m_vsg.dPdz, m_index); }

    /// Incident ray, and its x and y derivatives.
	Proxy<Vec3> I() const { return Proxy<Vec3>(m_vsg.I, m_index); }
	Proxy<Vec3> dIdx() const { return Proxy<Vec3>(m_vsg.dIdx, m_index); }
	Proxy<Vec3> dIdy() const { return Proxy<Vec3>(m_vsg.dIdy, m_index); }

    /// Shading normal, already front-facing.
	Proxy<Vec3> N() const { return Proxy<Vec3>(m_vsg.N, m_index); }

    /// True geometric normal.
	Proxy<Vec3> Ng() const { return Proxy<Vec3>(m_vsg.Ng, m_index); }

    /// 2D surface parameter u, and its differentials.
	Proxy<float> u() const { return Proxy<float>(m_vsg.u, m_index); }
	Proxy<float> dudx() const { return Proxy<float>(m_vsg.dudx, m_index); }
	Proxy<float> dudy() const { return Proxy<float>(m_vsg.dudy, m_index); }
	
    /// 2D surface parameter v, and its differentials.
	Proxy<float> v() const { return Proxy<float>(m_vsg.v, m_index); }
	Proxy<float> dvdx() const { return Proxy<float>(m_vsg.dvdx, m_index); }
	Proxy<float> dvdy() const { return Proxy<float>(m_vsg.dvdy, m_index); }

    /// Surface tangents: derivative of P with respect to surface u and v.
	Proxy<Vec3> dPdu() const { return Proxy<Vec3>(m_vsg.dPdu, m_index); }
	Proxy<Vec3> dPdv() const { return Proxy<Vec3>(m_vsg.dPdv, m_index); }

    /// Time for this shading sample.
	Proxy<float> time() const { return Proxy<float>(m_vsg.time, m_index); }
    /// Time interval for the frame (or shading sample).
	Proxy<float> dtime() const { return Proxy<float>(m_vsg.dtime, m_index); }
    ///  Velocity vector: derivative of position P with respect to time.
	Proxy<Vec3> dPdtime() const { return Proxy<Vec3>(m_vsg.dPdtime, m_index); }

    /// For lights or light attenuation shaders: the point being illuminated
    /// (Ps), and its differentials.
	Proxy<Vec3> Ps() const { return Proxy<Vec3>(m_vsg.Ps, m_index); }
	Proxy<Vec3> dPsdx() const { return Proxy<Vec3>(m_vsg.dPsdx, m_index); }
	Proxy<Vec3> dPsdy() const { return Proxy<Vec3>(m_vsg.dPsdy, m_index); }

    /// Opaque pointers set by the renderer before shader execution, to
    /// allow later retrieval of the object->common and shader->common
    /// transformation matrices, by the RendererServices
    /// get_matrix/get_inverse_matrix methods. This doesn't need to point
    /// to the 4x4 matrix itself; rather, it's just a pointer to whatever
    /// structure the RenderServices::get_matrix() needs to (if and when
    /// requested) generate the 4x4 matrix for the right time value.
	Proxy<TransformationPtr> object2common() const { return Proxy<TransformationPtr>(m_vsg.object2common, m_index); }
	Proxy<TransformationPtr> shader2common() const { return Proxy<TransformationPtr>(m_vsg.shader2common, m_index); }

    /// Surface area of the emissive object (used by light shaders for
    /// energy normalization).
	Proxy<float> surfacearea() const { return Proxy<float>(m_vsg.surfacearea, m_index); }

    /// If nonzero, will flip the result of calculatenormal().
	Proxy<int> flipHandedness() const { return Proxy<int>(m_vsg.flipHandedness, m_index); }

    /// If nonzero, we are shading the back side of a surface.
	Proxy<int> backfacing() const { return Proxy<int>(m_vsg.backfacing, m_index); }
};

struct alignas(64) ShaderGlobalsBatch
{
	ShaderGlobalsBatch()
	{
		clear();
	}
	ShaderGlobalsBatch(const ShaderGlobalsBatch &) = delete;
	
	static constexpr int maxSize = SimdLaneCount;
	
	UniformShaderGlobals & uniform() { return m_uniform; }
	
	
	typedef VaryingShaderProxy<maxSize> VaryingProxyType;
	// proxy to the "next" varying dataset  
	VaryingProxyType varying() { return VaryingProxyType(m_varying, m_size); }
	// TODO: consider removing/demoting the ASSERT with a debug only option
	VaryingProxyType varying(int batchIndex) { ASSERT(batchIndex < m_size); return VaryingProxyType(m_varying, batchIndex); }
	
	typedef VaryingShaderGlobals<maxSize> VaryingData;
	VaryingData & varyingData() { return m_varying; } 
	const VaryingData & varyingData() const { return m_varying; } 
	
	int size() const
	{
		return m_size;
	}
	
	bool isFull() const
	{
		return (m_size == maxSize);
	}

	bool isEmpty() const
	{
		return (m_size == 0);
	}
	
	void clear() {
		m_size=0;
	}
	
	void commitVarying() {
		assert(m_size < maxSize);
		++m_size;
	}	

	template<typename T>
	using  OutputData = Wide<T, maxSize>;
	
	template<typename DataT>
	using OutputAccessor = WideAccessor<DataT, maxSize>;
	
	void dump()
	{
		std::cout << "ShaderGlobalsBatch(m_size=" << m_size << ")" << " = {"  << std::endl;
		m_uniform.dump();
		m_varying.dump();
		std::cout << "};" << std::endl;
	}
	
//private:
	
	UniformShaderGlobals m_uniform;
	VaryingData m_varying;// __attribute__((aligned(16)));
	int m_size;
};
	


OSL_NAMESPACE_EXIT
