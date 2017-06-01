/*
Copyright (c) 2009-2014 Sony Pictures Imageworks Inc., et al.
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


#include "oslconfig.h"

#include <OpenImageIO/ustring.h>


OSL_NAMESPACE_ENTER

class RendererServices;
class BatchedRendererServices; 
class ShadingContext;
struct ShaderGlobals;



/// Opaque pointer to whatever the renderer uses to represent a
/// (potentially motion-blurred) coordinate transformation.
typedef const void * TransformationPtr;


// Callbacks for closure creation
typedef void (*PrepareClosureFunc)(RendererServices *, int id, void *data);
typedef void (*SetupClosureFunc)(RendererServices *, int id, void *data);




/// RendererServices defines an abstract interface through which a
/// renderer may provide callback to the ShadingSystem.
class OSLEXECPUBLIC RendererServices {
public:
    typedef TextureSystem::TextureHandle TextureHandle;
    typedef TextureSystem::Perthread TexturePerthread;


    RendererServices (TextureSystem *texsys=NULL);
    virtual ~RendererServices () { }

    /// Given the name of a 'feature', return whether this RendererServices
    /// supports it. Feature names include:
    ///    <none>
    ///
    /// This allows some customization of JIT generated code based on the
    /// facilities and features of a particular renderer. It also allows
    /// future expansion of RendererServices methods (with trivial default
    /// implementations) without requiring every renderer implementation to
    /// support it, as long as the OSL runtime only uses that feature if the
    /// supports("feature") says it's present, thus preserving source
    /// compatibility.
    virtual int supports (string_view feature) const { return false; }

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation at the given time.  Return true if ok, false
    /// on error.
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             TransformationPtr xform, float time) = 0;

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation at the given time.  Return true if ok, false on
    /// error.  The default implementation is to use get_matrix and
    /// invert it, but a particular renderer may have a better technique
    /// and overload the implementation.
    virtual bool get_inverse_matrix (ShaderGlobals *sg, Matrix44 &result,
                                     TransformationPtr xform, float time);

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation.  Return true if ok, false on error.  Since no
    /// time value is given, also return false if the transformation may
    /// be time-varying.
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             TransformationPtr xform) = 0;

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation.  Return true if ok, false on error.  Since no
    /// time value is given, also return false if the transformation may
    /// be time-varying.  The default implementation is to use
    /// get_matrix and invert it, but a particular renderer may have a
    /// better technique and overload the implementation.
    virtual bool get_inverse_matrix (ShaderGlobals *sg, Matrix44 &result,
                                     TransformationPtr xform);

    /// Get the 4x4 matrix that transforms points from the named
    /// 'from' coordinate system to "common" space at the given time.
    /// Returns true if ok, false if the named matrix is not known.
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             ustring from, float time) = 0;

    /// Get the 4x4 matrix that transforms points from "common" space to
    /// the named 'to' coordinate system to at the given time.  The
    /// default implementation is to use get_matrix and invert it, but a
    /// particular renderer may have a better technique and overload the
    /// implementation.
    virtual bool get_inverse_matrix (ShaderGlobals *sg, Matrix44 &result,
                                     ustring to, float time);

    /// Get the 4x4 matrix that transforms 'from' to "common" space.
    /// Since there is no time value passed, return false if the
    /// transformation may be time-varying (as well as if it's not found
    /// at all).
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             ustring from) = 0;

    /// Get the 4x4 matrix that transforms points from "common" space to
    /// the named 'to' coordinate system.  Since there is no time value
    /// passed, return false if the transformation may be time-varying
    /// (as well as if it's not found at all).  The default
    /// implementation is to use get_matrix and invert it, but a
    /// particular renderer may have a better technique and overload the
    /// implementation.
    virtual bool get_inverse_matrix (ShaderGlobals *sg, Matrix44 &result,
                                     ustring to);

    /// Transform points Pin[0..npoints-1] in named coordinate system
    /// 'from' into 'to' coordinates, storing the result in Pout[] using
    /// the specified vector semantic (POINT, VECTOR, NORMAL).  The
    /// function returns true if the renderer correctly transformed the
    /// points, false if it failed (for example, because it did not know
    /// the name of one of the coordinate systems).  A renderer is free
    /// to not implement this, in which case the default implementation
    /// is simply to make appropriate calls to get_matrix and
    /// get_inverse_matrix.  The existance of this method is to allow
    /// some renderers to provide transformations that cannot be
    /// expressed by a 4x4 matrix.
    ///
    /// If npoints == 0, the function should just return true if a
    /// known nonlinear transformation is available to transform points
    /// between the two spaces, otherwise false.  (For this calling
    /// pattern, sg, Pin, Pout, and time will not be used and may be 0.
    /// As a special case, if from and to are both empty strings, it
    /// returns true if *any* nonlinear transformations are supported,
    /// otherwise false.
    ///
    /// Note to RendererServices implementations: just return 'false'
    /// if there isn't a special nonlinear transformation between the
    /// two spaces.
    virtual bool transform_points (ShaderGlobals *sg,
                                   ustring from, ustring to, float time,
                                   const Vec3 *Pin, Vec3 *Pout, int npoints,
                                   TypeDesc::VECSEMANTICS vectype)
        { return false; }


    /// Get the named attribute from the renderer and if found then
    /// write it into 'val'.  Otherwise, return false.  If no object is
    /// specified (object == ustring()), then the renderer should search *first*
    /// for the attribute on the currently shaded object, and next, if
    /// unsuccessful, on the currently shaded "scene".
    ///
    /// Note to renderers: if sg is NULL, that means
    /// get_attribute is being called speculatively by the runtime
    /// optimizer, and it doesn't know which object the shader will be
    /// run on. Be robust to this situation, return 'true' (retrieve the
    /// attribute) if you can (known object and attribute name), but
    /// otherwise just fail by returning 'false'.
    virtual bool get_attribute (ShaderGlobals *sg, bool derivatives,
                                ustring object, TypeDesc type, ustring name,
                                void *val ) = 0;

    /// Similar to get_attribute();  this method will return the 'index'
    /// element of an attribute array.
    virtual bool get_array_attribute (ShaderGlobals *sg, bool derivatives,
                                      ustring object, TypeDesc type,
                                      ustring name, int index, void *val ) = 0;

    /// Get the named user-data from the current object and write it into
    /// 'val'. If derivatives is true, the derivatives should be written into val
    /// as well. Return false if no user-data with the given name and type was
    /// found.
    virtual bool get_userdata (bool derivatives, ustring name, TypeDesc type,
                               ShaderGlobals *sg, void *val) = 0;

    /// Given the name of a texture, return an opaque handle that can be
    /// used with texture calls to avoid the name lookups.
    virtual TextureHandle * get_texture_handle (ustring filename);

    /// Return true if the texture handle (previously returned by
    /// get_texture_handle()) is a valid texture that can be subsequently
    /// read or sampled.
    virtual bool good (TextureHandle *texture_handle);

    /// Return a thread-specific opaque pointer to the texture system.
    /// Knowing the context may help this go faster.
    virtual TexturePerthread * get_texture_perthread (ShadingContext *context=NULL);

    /// Filtered 2D texture lookup for a single point.
    ///
    /// s,t are the texture coordinates; dsdx, dtdx, dsdy, and dtdy are
    /// the differentials of s and t change in some canonical directions
    /// x and y.  The choice of x and y are not important to the
    /// implementation; it can be any imposed 2D coordinates, such as
    /// pixels in screen space, adjacent samples in parameter space on a
    /// surface, etc.
    ///
    /// The filename will always be passed, and it's ok for the renderer
    /// implementation to use only that (and in fact should be prepared to
    /// deal with texture_handle and texture_thread_info being NULL). But
    /// sometimes OSL can figure out the texture handle or thread info also
    /// and may pass them as non-NULL, in which case the renderer may (if it
    /// can) use that extra information to perform a less expensive texture
    /// lookup.
    ///
    /// Return true if the file is found and could be opened, otherwise
    /// return false.
    ///
    /// If the errormessage parameter is NULL, this method is expected to
    /// handle the errors fully, including forwarding them to the renderer
    /// or shading system. If errormessage is non-NULL, any resulting error
    /// messages (in case of failure, when the function returns false) will
    /// be stored there, leaving it up to the caller/shader to handle the
    /// error.
    virtual bool texture (ustring filename, TextureHandle *texture_handle,
                          TexturePerthread *texture_thread_info,
                          TextureOpt &options, ShaderGlobals *sg,
                          float s, float t, float dsdx, float dtdx,
                          float dsdy, float dtdy, int nchannels,
                          float *result, float *dresultds, float *dresultdt,
                          ustring *errormessage);
    // Deprecated version, with no errormessage parameter. This will
    // eventually disappear.
    virtual bool texture (ustring filename, TextureHandle *texture_handle,
                          TexturePerthread *texture_thread_info,
                          TextureOpt &options, ShaderGlobals *sg,
                          float s, float t, float dsdx, float dtdx,
                          float dsdy, float dtdy, int nchannels,
                          float *result, float *dresultds, float *dresultdt);

    /// Filtered 3D texture lookup for a single point.
    ///
    /// P is the volumetric texture coordinate; dPd{x,y,z} are the
    /// differentials of P in some canonical directions x, y, and z.
    /// The choice of x,y,z are not important to the implementation; it
    /// can be any imposed 3D coordinates, such as pixels in screen
    /// space and depth along the ray, etc.
    ///
    /// The filename will always be passed, and it's ok for the renderer
    /// implementation to use only that (and in fact should be prepared to
    /// deal with texture_handle and texture_thread_info being NULL). But
    /// sometimes OSL can figure out the texture handle or thread info also
    /// and may pass them as non-NULL, in which case the renderer may (if it
    /// can) use that extra information to perform a less expensive texture
    /// lookup.
    ///
    /// Return true if the file is found and could be opened, otherwise
    /// return false.
    ///
    /// If the errormessage parameter is NULL, this method is expected to
    /// handle the errors fully, including forwarding them to the renderer
    /// or shading system. If errormessage is non-NULL, any resulting error
    /// messages (in case of failure, when the function returns false) will
    /// be stored there, leaving it up to the caller/shader to handle the
    /// error.
    virtual bool texture3d (ustring filename, TextureHandle *texture_handle,
                            TexturePerthread *texture_thread_info,
                            TextureOpt &options, ShaderGlobals *sg,
                            const Vec3 &P, const Vec3 &dPdx, const Vec3 &dPdy,
                            const Vec3 &dPdz, int nchannels,
                            float *result, float *dresultds,
                            float *dresultdt, float *dresultdr,
                            ustring *errormessage);
    // Deprecated version, with no errormessage parameter. This will
    // eventually disappear.
    virtual bool texture3d (ustring filename, TextureHandle *texture_handle,
                            TexturePerthread *texture_thread_info,
                            TextureOpt &options, ShaderGlobals *sg,
                            const Vec3 &P, const Vec3 &dPdx, const Vec3 &dPdy,
                            const Vec3 &dPdz, int nchannels,
                            float *result, float *dresultds,
                            float *dresultdt, float *dresultdr);

    /// Filtered environment lookup for a single point.
    ///
    /// R is the directional texture coordinate; dRd[xy] are the
    /// differentials of R in canonical directions x, y.
    ///
    /// The filename will always be passed, and it's ok for the renderer
    /// implementation to use only that (and in fact should be prepared to
    /// deal with texture_handle and texture_thread_info being NULL). But
    /// sometimes OSL can figure out the texture handle or thread info also
    /// and may pass them as non-NULL, in which case the renderer may (if it
    /// can) use that extra information to perform a less expensive texture
    /// lookup.
    ///
    /// Return true if the file is found and could be opened, otherwise
    /// return false.
    ///
    /// If the errormessage parameter is NULL, this method is expected to
    /// handle the errors fully, including forwarding them to the renderer
    /// or shading system. If errormessage is non-NULL, any resulting error
    /// messages (in case of failure, when the function returns false) will
    /// be stored there, leaving it up to the caller/shader to handle the
    /// error.
    virtual bool environment (ustring filename, TextureHandle *texture_handle,
                              TexturePerthread *texture_thread_info,
                              TextureOpt &options, ShaderGlobals *sg,
                              const Vec3 &R, const Vec3 &dRdx, const Vec3 &dRdy,
                              int nchannels, float *result,
                              float *dresultds, float *dresultdt,
                              ustring *errormessage);
    // Deprecated version, with no errormessage parameter. This will
    // eventually disappear.
    virtual bool environment (ustring filename, TextureHandle *texture_handle,
                              TexturePerthread *texture_thread_info,
                              TextureOpt &options, ShaderGlobals *sg,
                              const Vec3 &R, const Vec3 &dRdx, const Vec3 &dRdy,
                              int nchannels, float *result,
                              float *dresultds, float *dresultdt);

    /// Get information about the given texture.  Return true if found
    /// and the data has been put in *data.  Return false if the texture
    /// doesn't exist, doesn't have the requested data, if the data
    /// doesn't match the type requested. or some other failure.
    ///
    /// The filename will always be passed, and it's ok for the renderer
    /// implementation to use only that (and in fact should be prepared to
    /// deal with texture_handle and texture_thread_info being NULL). But
    /// sometimes OSL can figure out the texture handle or thread info also
    /// and may pass them as non-NULL, in which case the renderer may (if it
    /// can) use that extra information to perform a less expensive texture
    /// lookup.
    ///
    /// Note to renderers: if sg is NULL, that means get_texture_info is
    /// being called speculatively by the runtime optimizer, and it doesn't
    /// know which object the shader will be run on.
    virtual bool get_texture_info (ShaderGlobals *sg, ustring filename,
                                   TextureHandle *texture_handle,
                                   int subimage,
                                   ustring dataname, TypeDesc datatype,
                                   void *data);


    /// Lookup nearest points in a point cloud. It will search for
    /// points around the given center within the specified radius. A
    /// list of indices is returned so the programmer can later retrieve
    /// attributes with pointcloud_get. The indices array is mandatory,
    /// but distances can be NULL.  If a derivs_offset > 0 is given,
    /// derivatives will be computed for distances (when provided).
    ///
    /// Return the number of points found, always < max_points
    virtual int pointcloud_search (ShaderGlobals *sg,
                                   ustring filename, const OSL::Vec3 &center,
                                   float radius, int max_points, bool sort,
                                   size_t *out_indices,
                                   float *out_distances, int derivs_offset);

    /// Retrieve an attribute for an index list. The result is another array
    /// of the requested type stored in out_data.
    ///
    /// Return 1 if the attribute is found, 0 otherwise.
    virtual int pointcloud_get (ShaderGlobals *sg,
                                ustring filename, size_t *indices, int count,
                                ustring attr_name, TypeDesc attr_type,
                                void *out_data);

    /// Write a point to the named pointcloud, which will be saved
    /// at the end of the frame.  Return true if everything is ok,
    /// false if there was an error.
    virtual bool pointcloud_write (ShaderGlobals *sg,
                                   ustring filename, const OSL::Vec3 &pos,
                                   int nattribs, const ustring *names,
                                   const TypeDesc *types,
                                   const void **data);

    /// Options for the trace call.
    struct TraceOpt {
        float mindist;    ///< ignore hits closer than this
        float maxdist;    ///< ignore hits farther than this
        bool shade;       ///< whether to shade what is hit
        ustring traceset; ///< named trace set
        TraceOpt () : mindist(0.0f), maxdist(1.0e30), shade(false) { }
    };

    /// Immediately trace a ray from P in the direction R.  Return true
    /// if anything hit, otherwise false.
    virtual bool trace (TraceOpt &options, ShaderGlobals *sg,
                        const OSL::Vec3 &P, const OSL::Vec3 &dPdx,
                        const OSL::Vec3 &dPdy, const OSL::Vec3 &R,
                        const OSL::Vec3 &dRdx, const OSL::Vec3 &dRdy) {
        return false;
    }

    /// Get the named message from the renderer and if found then
    /// write it into 'val'.  Otherwise, return false.  This is only
    /// called for "sourced" messages, not ordinary intra-group messages.
    virtual bool getmessage (ShaderGlobals *sg, ustring source, ustring name,
                             TypeDesc type, void *val, bool derivatives) {
        return false;
    }

    /// Return a pointer to the texture system (if available).
    virtual TextureSystem *texturesys () const;

    /// Options we use for noise calls.
    struct NoiseOpt {
        int anisotropic;
        int do_filter;
        Vec3 direction;
        float bandwidth;
        float impulses;
        NoiseOpt () : anisotropic(0), do_filter(true),
            direction(1.0f,0.0f,0.0f), bandwidth(1.0f), impulses(16.0f) { }
    };

    virtual BatchedRendererServices * batched();
    
    
protected:
    TextureSystem *m_texturesys;   // A place to hold a TextureSystem
};


class OSLEXECPUBLIC BatchedRendererServices {
public:
    typedef TextureSystem::TextureHandle TextureHandle;
    typedef TextureSystem::Perthread TexturePerthread;


    BatchedRendererServices (TextureSystem *texsys=NULL);
    virtual ~BatchedRendererServices () { }

    
#if 0
    /// Given the name of a 'feature', return whether this RendererServices
    /// supports it. Feature names include:
    ///    <none>
    ///
    /// This allows some customization of JIT generated code based on the
    /// facilities and features of a particular renderer. It also allows
    /// future expansion of RendererServices methods (with trivial default
    /// implementations) without requiring every renderer implementation to
    /// support it, as long as the OSL runtime only uses that feature if the
    /// supports("feature") says it's present, thus preserving source
    /// compatibility.
    virtual int supports (string_view feature) const { return false; }
#endif

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation at the given time.  Return true if ok, false
    /// on error.
#if 0
    virtual bool get_matrix (ShaderGlobalsBatch *sgb, Matrix44 &result,
                             TransformationPtr xform, float time) = 0;
#endif
    virtual Mask get_matrix (ShaderGlobalsBatch *sgb, Wide<Matrix44> &result,
            const Wide<TransformationPtr> & xform, const Wide<float> &time, WeakMask weak_mask) = 0;    

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation at the given time.  Return true if ok, false on
    /// error.  The default implementation is to use get_matrix and
    /// invert it, but a particular renderer may have a better technique
    /// and overload the implementation.
    virtual Mask get_inverse_matrix (ShaderGlobalsBatch *sgb, Wide<Matrix44> &result,
            const Wide<TransformationPtr> & xform, const Wide<float> &time, WeakMask weak_mask);    
										  

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation.  Return true if ok, false on error.  Since no
    /// time value is given, also return false if the transformation may
    /// be time-varying.
    virtual bool get_matrix (ShaderGlobalsBatch *sgb, Matrix44 &result,
                             TransformationPtr xform) = 0;

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation.  Return true if ok, false on error.  Since no
    /// time value is given, also return false if the transformation may
    /// be time-varying.  The default implementation is to use
    /// get_matrix and invert it, but a particular renderer may have a
    /// better technique and overload the implementation.
    virtual bool get_inverse_matrix (ShaderGlobalsBatch *sgb, Matrix44 &result,
                                     TransformationPtr xform);

    /// Get the 4x4 matrix that transforms points from the named
    /// 'from' coordinate system to "common" space at the given time.
    /// Returns true if ok, false if the named matrix is not known.
    virtual Mask get_matrix (ShaderGlobalsBatch *sgb, Wide<Matrix44> &result,
                             ustring from, const Wide<float> &time, WeakMask weak_mask) = 0;

    /// Get the 4x4 matrix that transforms points from "common" space to
    /// the named 'to' coordinate system to at the given time.  The
    /// default implementation is to use get_matrix and invert it, but a
    /// particular renderer may have a better technique and overload the
    /// implementation.
    virtual Mask get_inverse_matrix (ShaderGlobalsBatch *sgb, Wide<Matrix44> &result,
                                     ustring to, const Wide<float> &time, WeakMask weak_mask);

    /// Get the 4x4 matrix that transforms 'from' to "common" space.
    /// Since there is no time value passed, return false if the
    /// transformation may be time-varying (as well as if it's not found
    /// at all).
    virtual bool get_matrix (ShaderGlobalsBatch *sgb, Matrix44 &result,
                             ustring from) = 0;

    /// Get the 4x4 matrix that transforms points from "common" space to
    /// the named 'to' coordinate system.  Since there is no time value
    /// passed, return false if the transformation may be time-varying
    /// (as well as if it's not found at all).  The default
    /// implementation is to use get_matrix and invert it, but a
    /// particular renderer may have a better technique and overload the
    /// implementation.
    virtual bool get_inverse_matrix (ShaderGlobalsBatch *sgb, Matrix44 &result,
                                     ustring to);
#if 0

    /// Transform points Pin[0..npoints-1] in named coordinate system
    /// 'from' into 'to' coordinates, storing the result in Pout[] using
    /// the specified vector semantic (POINT, VECTOR, NORMAL).  The
    /// function returns true if the renderer correctly transformed the
    /// points, false if it failed (for example, because it did not know
    /// the name of one of the coordinate systems).  A renderer is free
    /// to not implement this, in which case the default implementation
    /// is simply to make appropriate calls to get_matrix and
    /// get_inverse_matrix.  The existance of this method is to allow
    /// some renderers to provide transformations that cannot be
    /// expressed by a 4x4 matrix.
    ///
    /// If npoints == 0, the function should just return true if a
    /// known nonlinear transformation is available to transform points
    /// between the two spaces, otherwise false.  (For this calling
    /// pattern, sg, Pin, Pout, and time will not be used and may be 0.
    /// As a special case, if from and to are both empty strings, it
    /// returns true if *any* nonlinear transformations are supported,
    /// otherwise false.
    ///
    /// Note to RendererServices implementations: just return 'false'
    /// if there isn't a special nonlinear transformation between the
    /// two spaces.
    virtual bool transform_points (ShaderGlobals *sg,
                                   ustring from, ustring to, float time,
                                   const Vec3 *Pin, Vec3 *Pout, int npoints,
                                   TypeDesc::VECSEMANTICS vectype)
        { return false; }

#endif

    /// Identify if the the named attribute from the renderer can be treated as uniform
    /// accross all batches to the shader.  We assume all attributes are varying unless
    /// identified as uniform by the renderer.  NOTE:  To enable constant folding of
    // an attribute value, it must be uniform and retrievable with a NULL ShaderGlobalsBatch
    virtual bool is_attribute_uniform(ustring object, ustring name) = 0;
    
    /// Get the named attribute from the renderer and if found then
    /// write it into 'val'.  Otherwise, return false.  If no object is
    /// specified (object == ustring()), then the renderer should search *first*
    /// for the attribute on the currently shaded object, and next, if
    /// unsuccessful, on the currently shaded "scene".
    ///
    /// Note to renderers: if sg is NULL, that means
    /// get_attribute is being called speculatively by the runtime
    /// optimizer, and it doesn't know which object the shader will be
    /// run on. Be robust to this situation, return 'true' (retrieve the
    /// attribute) if you can (known object and attribute name), but
    /// otherwise just fail by returning 'false'.

    virtual Mask get_attribute (ShaderGlobalsBatch *sgb, 
                                ustring object, ustring name,
                                MaskedDataRef val) = 0;
    
    /// Similar to get_attribute();  this method will fetch the 'index'
    /// element of an attribute array.
    virtual Mask get_array_attribute (ShaderGlobalsBatch *sgb, 
                                      ustring object,
                                      ustring name, int index, MaskedDataRef val) = 0;

    virtual bool get_attribute_uniform (ShaderGlobalsBatch *sgb, 
                                ustring object, ustring name, DataRef val) = 0;

    /// Similar to get_attribute();  this method will fetch the 'index'
    /// element of an attribute array.
    virtual bool get_array_attribute_uniform (ShaderGlobalsBatch *sgb,
                                      ustring object,
                                      ustring name, int index, DataRef val) = 0;
    
    /// Get multiple named user-data from the current object and write them into
    /// 'val'. If derivatives is true, the derivatives should be written into val
    /// as well. It is assumed the results are varying and returns Mask 
    // with its bit set to off if no user-data with the given name and type was
    /// found.
    virtual Mask get_userdata (ustring name, 
    						   ShaderGlobalsBatch *sgb, MaskedDataRef val) = 0;

#if 0
    
    /// Given the name of a texture, return an opaque handle that can be
    /// used with texture calls to avoid the name lookups.
    virtual TextureHandle * get_texture_handle (ustring filename);

    /// Return true if the texture handle (previously returned by
    /// get_texture_handle()) is a valid texture that can be subsequently
    /// read or sampled.
    virtual bool good (TextureHandle *texture_handle);

    /// Return a thread-specific opaque pointer to the texture system.
    /// Knowing the context may help this go faster.
    virtual TexturePerthread * get_texture_perthread (ShadingContext *context=NULL);

#endif
    /// Filtered 2D texture lookup for a single point.
    ///
    /// s,t are the texture coordinates; dsdx, dtdx, dsdy, and dtdy are
    /// the differentials of s and t change in some canonical directions
    /// x and y.  The choice of x and y are not important to the
    /// implementation; it can be any imposed 2D coordinates, such as
    /// pixels in screen space, adjacent samples in parameter space on a
    /// surface, etc.
    ///
    /// The filename will always be passed, and it's ok for the renderer
    /// implementation to use only that (and in fact should be prepared to
    /// deal with texture_handle and texture_thread_info being NULL). But
    /// sometimes OSL can figure out the texture handle or thread info also
    /// and may pass them as non-NULL, in which case the renderer may (if it
    /// can) use that extra information to perform a less expensive texture
    /// lookup.
    ///
    /// Return true if the file is found and could be opened, otherwise
    /// return false.
    ///
    /// If the errormessage parameter is NULL, this method is expected to
    /// handle the errors fully, including forwarding them to the renderer
    /// or shading system. If errormessage is non-NULL, any resulting error
    /// messages (in case of failure, when the function returns false) will
    /// be stored there, leaving it up to the caller/shader to handle the
    /// error.
    virtual Mask texture (ustring filename, TextureHandle *texture_handle,
                          TexturePerthread *texture_thread_info,
                          TextureOpt &options, ShaderGlobalsBatch *sgb,
                          float s, float t, float dsdx, float dtdx,
                          float dsdy, float dtdy, int nchannels,
                          float *result, float *dresultds, float *dresultdt,
                          ustring *errormessage, Mask mask);
#if 0
    // Deprecated version, with no errormessage parameter. This will
    // eventually disappear.
    virtual bool texture (ustring filename, TextureHandle *texture_handle,
                          TexturePerthread *texture_thread_info,
                          TextureOpt &options, ShaderGlobals *sg,
                          float s, float t, float dsdx, float dtdx,
                          float dsdy, float dtdy, int nchannels,
                          float *result, float *dresultds, float *dresultdt);

    /// Filtered 3D texture lookup for a single point.
    ///
    /// P is the volumetric texture coordinate; dPd{x,y,z} are the
    /// differentials of P in some canonical directions x, y, and z.
    /// The choice of x,y,z are not important to the implementation; it
    /// can be any imposed 3D coordinates, such as pixels in screen
    /// space and depth along the ray, etc.
    ///
    /// The filename will always be passed, and it's ok for the renderer
    /// implementation to use only that (and in fact should be prepared to
    /// deal with texture_handle and texture_thread_info being NULL). But
    /// sometimes OSL can figure out the texture handle or thread info also
    /// and may pass them as non-NULL, in which case the renderer may (if it
    /// can) use that extra information to perform a less expensive texture
    /// lookup.
    ///
    /// Return true if the file is found and could be opened, otherwise
    /// return false.
    ///
    /// If the errormessage parameter is NULL, this method is expected to
    /// handle the errors fully, including forwarding them to the renderer
    /// or shading system. If errormessage is non-NULL, any resulting error
    /// messages (in case of failure, when the function returns false) will
    /// be stored there, leaving it up to the caller/shader to handle the
    /// error.
    virtual bool texture3d (ustring filename, TextureHandle *texture_handle,
                            TexturePerthread *texture_thread_info,
                            TextureOpt &options, ShaderGlobals *sg,
                            const Vec3 &P, const Vec3 &dPdx, const Vec3 &dPdy,
                            const Vec3 &dPdz, int nchannels,
                            float *result, float *dresultds,
                            float *dresultdt, float *dresultdr,
                            ustring *errormessage);
    // Deprecated version, with no errormessage parameter. This will
    // eventually disappear.
    virtual bool texture3d (ustring filename, TextureHandle *texture_handle,
                            TexturePerthread *texture_thread_info,
                            TextureOpt &options, ShaderGlobals *sg,
                            const Vec3 &P, const Vec3 &dPdx, const Vec3 &dPdy,
                            const Vec3 &dPdz, int nchannels,
                            float *result, float *dresultds,
                            float *dresultdt, float *dresultdr);

    /// Filtered environment lookup for a single point.
    ///
    /// R is the directional texture coordinate; dRd[xy] are the
    /// differentials of R in canonical directions x, y.
    ///
    /// The filename will always be passed, and it's ok for the renderer
    /// implementation to use only that (and in fact should be prepared to
    /// deal with texture_handle and texture_thread_info being NULL). But
    /// sometimes OSL can figure out the texture handle or thread info also
    /// and may pass them as non-NULL, in which case the renderer may (if it
    /// can) use that extra information to perform a less expensive texture
    /// lookup.
    ///
    /// Return true if the file is found and could be opened, otherwise
    /// return false.
    ///
    /// If the errormessage parameter is NULL, this method is expected to
    /// handle the errors fully, including forwarding them to the renderer
    /// or shading system. If errormessage is non-NULL, any resulting error
    /// messages (in case of failure, when the function returns false) will
    /// be stored there, leaving it up to the caller/shader to handle the
    /// error.
    virtual bool environment (ustring filename, TextureHandle *texture_handle,
                              TexturePerthread *texture_thread_info,
                              TextureOpt &options, ShaderGlobals *sg,
                              const Vec3 &R, const Vec3 &dRdx, const Vec3 &dRdy,
                              int nchannels, float *result,
                              float *dresultds, float *dresultdt,
                              ustring *errormessage);
    // Deprecated version, with no errormessage parameter. This will
    // eventually disappear.
    virtual bool environment (ustring filename, TextureHandle *texture_handle,
                              TexturePerthread *texture_thread_info,
                              TextureOpt &options, ShaderGlobals *sg,
                              const Vec3 &R, const Vec3 &dRdx, const Vec3 &dRdy,
                              int nchannels, float *result,
                              float *dresultds, float *dresultdt);

#endif

    /// Get information about the given texture.  Return true if found
    /// and the data has been put in *data.  Return false if the texture
    /// doesn't exist, doesn't have the requested data, if the data
    /// doesn't match the type requested. or some other failure.
    ///
    /// The filename will always be passed, and it's ok for the renderer
    /// implementation to use only that (and in fact should be prepared to
    /// deal with texture_handle and texture_thread_info being NULL). But
    /// sometimes OSL can figure out the texture handle or thread info also
    /// and may pass them as non-NULL, in which case the renderer may (if it
    /// can) use that extra information to perform a less expensive texture
    /// lookup.
    ///
    /// Note to renderers: if sg is NULL, that means get_texture_info is
    /// being called speculatively by the runtime optimizer, and it doesn't
    /// know which object the shader will be run on.
    virtual Mask get_texture_info (ShaderGlobalsBatch *sgb,
                                   const Wide<ustring>& filename,
                                   // We do not need to support texture handle for varying data.
                                   int subimage,
                                   ustring dataname, TypeDesc datatype,
                                   void *data,
                                   Mask mask);

    virtual bool get_texture_info_uniform (ShaderGlobalsBatch *sgb, ustring filename,
                                           TextureHandle *texture_handle,
                                           int subimage,
                                           ustring dataname, TypeDesc datatype,
                                           void *data);


#if 0
    /// Lookup nearest points in a point cloud. It will search for
    /// points around the given center within the specified radius. A
    /// list of indices is returned so the programmer can later retrieve
    /// attributes with pointcloud_get. The indices array is mandatory,
    /// but distances can be NULL.  If a derivs_offset > 0 is given,
    /// derivatives will be computed for distances (when provided).
    ///
    /// Return the number of points found, always < max_points
    virtual int pointcloud_search (ShaderGlobals *sg,
                                   ustring filename, const OSL::Vec3 &center,
                                   float radius, int max_points, bool sort,
                                   size_t *out_indices,
                                   float *out_distances, int derivs_offset);

    /// Retrieve an attribute for an index list. The result is another array
    /// of the requested type stored in out_data.
    ///
    /// Return 1 if the attribute is found, 0 otherwise.
    virtual int pointcloud_get (ShaderGlobals *sg,
                                ustring filename, size_t *indices, int count,
                                ustring attr_name, TypeDesc attr_type,
                                void *out_data);

    /// Write a point to the named pointcloud, which will be saved
    /// at the end of the frame.  Return true if everything is ok,
    /// false if there was an error.
    virtual bool pointcloud_write (ShaderGlobals *sg,
                                   ustring filename, const OSL::Vec3 &pos,
                                   int nattribs, const ustring *names,
                                   const TypeDesc *types,
                                   const void **data);

    /// Options for the trace call.
    struct TraceOpt {
        float mindist;    ///< ignore hits closer than this
        float maxdist;    ///< ignore hits farther than this
        bool shade;       ///< whether to shade what is hit
        ustring traceset; ///< named trace set
        TraceOpt () : mindist(0.0f), maxdist(1.0e30), shade(false) { }
    };

    /// Immediately trace a ray from P in the direction R.  Return true
    /// if anything hit, otherwise false.
    virtual bool trace (TraceOpt &options, ShaderGlobals *sg,
                        const OSL::Vec3 &P, const OSL::Vec3 &dPdx,
                        const OSL::Vec3 &dPdy, const OSL::Vec3 &R,
                        const OSL::Vec3 &dRdx, const OSL::Vec3 &dRdy) {
        return false;
    }

    /// Get the named message from the renderer and if found then
    /// write it into 'val'.  Otherwise, return false.  This is only
    /// called for "sourced" messages, not ordinary intra-group messages.
    virtual bool getmessage (ShaderGlobals *sg, ustring source, ustring name,
                             TypeDesc type, void *val, bool derivatives) {
        return false;
    }
#endif

    /// Return a pointer to the texture system (if available).
    virtual TextureSystem *texturesys () const;

#if 0
    /// Options we use for noise calls.
    struct NoiseOpt {
        int anisotropic;
        int do_filter;
        Vec3 direction;
        float bandwidth;
        float impulses;
        NoiseOpt () : anisotropic(0), do_filter(true),
            direction(1.0f,0.0f,0.0f), bandwidth(1.0f), impulses(16.0f) { }
    };
#endif
    

    
protected:
    TextureSystem *m_texturesys;   // A place to hold a TextureSystem
};


OSL_NAMESPACE_EXIT
