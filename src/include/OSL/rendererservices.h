// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// clang-format off

#pragma once


#include <OSL/oslconfig.h>


OSL_NAMESPACE_ENTER

class RendererServices;
template<int WidthT>
class BatchedRendererServices;
class ShadingContext;
struct ShaderGlobals;

// Tags for polymorphic dispatch
template <int SimdWidthT>
class WidthOf {};


/// Opaque pointer to whatever the renderer uses to represent a
/// (potentially motion-blurred) coordinate transformation.
typedef const void* TransformationPtr;


// Callbacks for closure creation
typedef void (*PrepareClosureFunc)(RendererServices*, int id, void* data);
typedef void (*SetupClosureFunc)(RendererServices*, int id, void* data);

// Turn off warnings about unused params for this file, since we have lots
// of declarations with stub function bodies.
OSL_PRAGMA_WARNING_PUSH
OSL_GCC_PRAGMA(GCC diagnostic ignored "-Wunused-parameter")



/// RendererServices defines an abstract interface through which a
/// renderer may provide callback to the ShadingSystem.
class OSLEXECPUBLIC RendererServices {
public:
    typedef TextureSystem::TextureHandle TextureHandle;
    typedef TextureSystem::Perthread TexturePerthread;


    RendererServices(TextureSystem* texsys = NULL);
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
                             TransformationPtr xform, float time) { return false; }

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
                             TransformationPtr xform) { return false; }

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
                             ustring from, float time) { return false; }

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
                             ustring from) { return false; }

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
    /// get_inverse_matrix.  The existence of this method is to allow
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
                                void *val) { return false; }

    /// Similar to get_attribute();  this method will return the 'index'
    /// element of an attribute array.
    virtual bool get_array_attribute (ShaderGlobals *sg, bool derivatives,
                                      ustring object, TypeDesc type,
                                      ustring name, int index, void *val) { return false; }

    /// Get the named user-data from the current object and write it into
    /// 'val'. If derivatives is true, the derivatives should be written into val
    /// as well. Return false if no user-data with the given name and type was
    /// found.
    virtual bool get_userdata (bool derivatives, ustring name, TypeDesc type,
                               ShaderGlobals *sg, void *val) { return false; }

    /// Given the name of a texture, return an opaque handle that can be
    /// used with texture calls to avoid the name lookups.
    virtual TextureHandle * get_texture_handle (ustring filename,
                                                ShadingContext *context);

    /// Return true if the texture handle (previously returned by
    /// get_texture_handle()) is a valid texture that can be subsequently
    /// read or sampled.
    virtual bool good (TextureHandle *texture_handle);

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
    /// If the errormessage parameter is NULL, this method is expected to
    /// handle the errors fully, including forwarding them to the renderer
    /// or shading system. If errormessage is non-NULL, any resulting error
    /// messages (in case of failure, when the function returns false) will
    /// be stored there, leaving it up to the caller/shader to handle the
    /// error.
    virtual bool get_texture_info (ustring filename,
                                   TextureHandle *texture_handle,
                                   TexturePerthread *texture_thread_info,
                                   ShadingContext *shading_context,
                                   int subimage,
                                   ustring dataname, TypeDesc datatype,
                                   void *data,
                                   ustring *errormessage);

    virtual bool get_texture_info (ustring filename,
                                   TextureHandle *texture_handle,
                                   float s, float t,
                                   TexturePerthread *texture_thread_info,
                                   ShadingContext *shading_context,
                                   int subimage,
                                   ustring dataname, TypeDesc datatype,
                                   void *data,
                                   ustring *errormessage);


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

    /// Register a string with the renderer, so that the renderer can arrange
    /// to make the string available in GPU memory as needed. Optionally specify
    /// a variable name to associate with the string value.
    virtual uint64_t register_string (const std::string& str,
                                      const std::string& var_name)
    {
        return 0;
    }

    virtual uint64_t register_global (const std::string& var_name,
                                      uint64_t           value)
    {
        return 0;
    }

    virtual bool fetch_global (const std::string& var_name,
                               uint64_t*          value)
    {
        return false;
    }



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

    /// A renderer may choose to support batched execution by providing pointers
    /// to objects satisfying the BatchedRendererServices<WidthOf<#>> interface
    /// for specific batch sizes.
    /// Unless overridden, a nullptr is returned.
    virtual BatchedRendererServices<16> * batched(WidthOf<16>);
    virtual BatchedRendererServices<8> * batched(WidthOf<8>);

protected:
    TextureSystem *m_texturesys;   // A place to hold a TextureSystem
};


OSL_PRAGMA_WARNING_POP
OSL_NAMESPACE_EXIT
