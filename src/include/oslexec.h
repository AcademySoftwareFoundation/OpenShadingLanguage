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

#ifndef OSLEXEC_H
#define OSLEXEC_H


#include "oslconfig.h"

#include <OpenImageIO/refcnt.h>     // just to get shared_ptr from boost ?!
#include <OpenImageIO/ustring.h>


OSL_NAMESPACE_ENTER

class RendererServices;
class ShadingAttribState;
typedef shared_ptr<ShadingAttribState> ShadingAttribStateRef;
struct ShaderGlobals;
struct ClosureColor;
struct ClosureParam;
struct PerThreadInfo;
class ShadingContext;



/// Opaque pointer to whatever the renderer uses to represent a
/// (potentially motion-blurred) coordinate transformation.
typedef const void * TransformationPtr;


// Callbacks for closure creation
typedef void (*PrepareClosureFunc)(RendererServices *, int id, void *data);
typedef void (*SetupClosureFunc)(RendererServices *, int id, void *data);
typedef bool (*CompareClosureFunc)(int id, const void *dataA, const void *dataB);


class OSLEXECPUBLIC ShadingSystem
{
protected:
    /// ShadingSystem is an abstract class, its constructor is protected
    /// so that ordinary users can't make an instance, but instead are
    /// forced to request one via ShadingSystem::create().
    ShadingSystem ();
    virtual ~ShadingSystem ();

public:
    static ShadingSystem *create (RendererServices *renderer=NULL,
                                  TextureSystem *texturesystem=NULL,
                                  ErrorHandler *err=NULL);
    static void destroy (ShadingSystem *x);

    /// Set an attribute controlling the texture system.  Return true
    /// if the name and type were recognized and the attrib was set.
    /// Documented attributes:
    ///
    virtual bool attribute (const std::string &name, TypeDesc type,
                            const void *val) = 0;
    // Shortcuts for common types
    bool attribute (const std::string &name, int val) {
        return attribute (name, TypeDesc::INT, &val);
    }
    bool attribute (const std::string &name, float val) {
        return attribute (name, TypeDesc::FLOAT, &val);
    }
    bool attribute (const std::string &name, double val) {
        float f = (float) val;
        return attribute (name, TypeDesc::FLOAT, &f);
    }
    bool attribute (const std::string &name, const char *val) {
        return attribute (name, TypeDesc::STRING, &val);
    }
    bool attribute (const std::string &name, const std::string &val) {
        const char *s = val.c_str();
        return attribute (name, TypeDesc::STRING, &s);
    }

    /// Get the named attribute, store it in value.
    ///
    virtual bool getattribute (const std::string &name, TypeDesc type,
                               void *val) = 0;
    // Shortcuts for common types
    bool getattribute (const std::string &name, int &val) {
        return getattribute (name, TypeDesc::INT, &val);
    }
    bool getattribute (const std::string &name, float &val) {
        return getattribute (name, TypeDesc::FLOAT, &val);
    }
    bool getattribute (const std::string &name, double &val) {
        float f;
        bool ok = getattribute (name, TypeDesc::FLOAT, &f);
        if (ok)
            val = f;
        return ok;
    }
    bool getattribute (const std::string &name, char **val) {
        return getattribute (name, TypeDesc::STRING, val);
    }
    bool getattribute (const std::string &name, std::string &val) {
        const char *s = NULL;
        bool ok = getattribute (name, TypeDesc::STRING, &s);
        if (ok)
            val = s;
        return ok;
    }

    /// Set a parameter of the next shader.
    ///
    virtual bool Parameter (const char *name, TypeDesc t, const void *val)
        { return true; }
#if 0
    virtual bool Parameter (const char *name, int val) {
        Parameter (name, TypeDesc::IntType, &val);
    }
    virtual bool Parameter (const char *name, float val) {
        Parameter (name, TypeDesc::FloatType, &val);
    }
    virtual bool Parameter (const char *name, double val) {}
    virtual bool Parameter (const char *name, const char *val) {}
    virtual bool Parameter (const char *name, const std::string &val) {}
    virtual bool Parameter (const char *name, TypeDesc t, const int *val) {}
    virtual bool Parameter (const char *name, TypeDesc t, const float *val) {}
    virtual bool Parameter (const char *name, TypeDesc t, const char **val) {}
#endif

    /// Create a new shader instance, either replacing the one for the
    /// specified usage (if not within a group) or appending to the 
    /// current group (if a group has been started).
    virtual bool Shader (const char *shaderusage,
                         const char *shadername=NULL,
                         const char *layername=NULL) = 0;

    /// Signal the start of a new shader group.
    ///
    virtual bool ShaderGroupBegin (const char *groupname=NULL) = 0;

    /// Signal the end of a new shader group.
    ///
    virtual bool ShaderGroupEnd (void) = 0;

    /// Connect two shaders within the current group
    ///
    virtual bool ConnectShaders (const char *srclayer, const char *srcparam,
                                 const char *dstlayer, const char *dstparam)=0;

    /// Return a reference-counted (but opaque) reference to the current
    /// shading attribute state maintained by the ShadingSystem.
    virtual ShadingAttribStateRef state () = 0;

    /// Clear the current shading attribute state, i.e., no shaders
    /// specified.
    virtual void clear_state () = 0;

    /// Optional: create the per-thread data needed for shader
    /// execution.  Doing this and passing it to get_context speeds is a
    /// bit faster than get_context having to do a thread-specific
    /// lookup on its own, but if you do it, it's important for the app
    /// to use one and only one PerThreadInfo per renderer thread, and
    /// destroy it with destroy_thread_info when the thread terminates.
    virtual PerThreadInfo * create_thread_info() = 0;

    /// Destroy a PerThreadInfo that was allocated by
    /// create_thread_info().
    virtual void destroy_thread_info (PerThreadInfo *threadinfo) = 0;

    /// Get a ShadingContext that we can use.  The context is specific
    /// to the renderer thread.  The 'threadinfo' parameter should be a
    /// thread-specific pointer created by create_thread_info, or NULL,
    /// in which case the ShadingSystem will do the thread-specific
    /// lookup automatically (and at some additional cost).  The context
    /// can be used to shade many points; a typical usage is to allocate
    /// just one context per thread and use it for the whole run.
    virtual ShadingContext *get_context (PerThreadInfo *threadinfo=NULL) = 0;

    /// Return a ShadingContext to the pool.
    ///
    virtual void release_context (ShadingContext *ctx) = 0;

    /// Execute the shader bound to context ctx, with the given
    /// ShadingAttribState (that specifies the shader group to run) and
    /// ShaderGlobals (specific information for this shade point).  If
    /// 'run' is false, do all the usual preparation, but don't actually
    /// run the shader.  Return true if the shader executed (or could
    /// have executed, if 'run' had been true), false the shader turned
    /// out to be empty.
    virtual bool execute (ShadingContext &ctx, ShadingAttribState &sas,
                          ShaderGlobals &ssg, bool run=true) = 0;

    /// Get a raw pointer to a named symbol (such as you'd need to pull
    /// out the value of an output parameter).  ctx is the shading
    /// context (presumably already run), name is the name of the
    /// symbol.  If found, get_symbol will return the pointer to the
    /// symbol's data, and type will get the symbol's type.  If the
    /// symbol is not found, get_symbol will return NULL.
    virtual const void* get_symbol (ShadingContext &ctx, ustring name,
                                    TypeDesc &type) = 0;

    /// Return the statistics output as a huge string.
    ///
    virtual std::string getstats (int level=1) const = 0;

    virtual void register_closure(const char *name, int id, const ClosureParam *params,
                                  PrepareClosureFunc prepare, SetupClosureFunc setup, CompareClosureFunc compare) = 0;

    void register_builtin_closures();

    /// For the proposed raytype name, return the bit pattern that
    /// describes it, or 0 for an unrecognized name.  (This retrieves
    /// data passed in via attribute("raytypes")).
    virtual int raytype_bit (ustring name) = 0;

    /// If option "greedyjit" was set, this call will trigger all
    /// shader groups that have not yet been compiled to do so with the
    /// specified number of threads (0 means use all available HW cores).
    virtual void optimize_all_groups (int nthreads=0) = 0;

    /// Helper function -- copy or convert a source value (described by
    /// srctype) to destination (described by dsttype).  The function
    /// returns true upon success, or false if the types differ in a way
    /// that cannot be converted.  As a special case, if dst==NULL or
    /// src==NULL, no copying is performed, and convert_value merely
    /// returns a bool indicating if the proposed type conversion is
    /// allowed.
    ///
    /// The following type conversions are supported:
    /// 1. Identical types copy without modification.
    /// 2. Conversions following the same rules as type casting and
    /// assignment in OSL itself:
    ///   int -> float             convert to float
    ///   int -> triple            convert to float and replicate x3
    ///   float -> triple          replicate x3
    ///   float -> int             truncate like a (int) type cast
    ///   triple -> triple         copy, regarless of differing vector types
    /// 3. Additional rules not allowed in OSL source code:
    ///   float -> float[2]        replicate x2
    ///   int -> float[2]          convert to float and replicate x2
    ///   float[2] -> triple       (f[0], f[1], 0)
    ///
    /// Observation: none of the supported conversions require more
    /// storage for src than for dst.
    static bool convert_value (void *dst, TypeDesc dsttype,
                               const void *src, TypeDesc srctype);

private:
    // Make delete private and unimplemented in order to prevent apps
    // from calling it.  Instead, they should call ShadingSystem::destroy().
    void operator delete (void *todel) { }
};



/// This struct represents the global variables accessible from a shader, note 
/// that not all fields will be valid in all contexts.
///
/// All points, vectors and normals are given in "common" space.
struct ShaderGlobals {
    Vec3 P, dPdx, dPdy;              /**< Position */
    Vec3 dPdz;                       /**< z zeriv for volume shading */
    Vec3 I, dIdx, dIdy;              /**< Incident ray */
    Vec3 N;                          /**< Shading normal */
    Vec3 Ng;                         /**< True geometric normal */
    float u, dudx, dudy;             /**< Surface parameter u */
    float v, dvdx, dvdy;             /**< Surface parameter v */
    Vec3 dPdu, dPdv;                 /**< Tangents on the surface */
    float time;                      /**< Time for each sample */
    float dtime;                     /**< Time interval for each sample */
    Vec3 dPdtime;                    /**< Velocity */
    Vec3 Ps, dPsdx, dPsdy;           /**< Point being lit (valid only in light
                                          attenuation shaders */
    void* renderstate;               /**< Opaque pointer to renderer state (can
                                          be used to retrieve renderer specific
                                          details like userdata) */
    void* tracedata;                 /**< Opaque pointer to renderer state
                                          resuling from a trace() call. */
    void* objdata;                   /**< Opaque pointer to object data */
    ShadingContext* context;         /**< ShadingContext (this will be set by
                                          OSL itself) */
    TransformationPtr object2common; /**< Object->common xform */
    TransformationPtr shader2common; /**< Shader->common xform */
    ClosureColor *Ci;                /**< Output closure (should be initialized
                                          to NULL) */
    float surfacearea;               /**< Total area of the object (defined by
                                          light shaders for energy normalization) */
    int raytype;                     /**< Bit field of ray type flags */
    int flipHandedness;              /**< flips the result of calculatenormal() */
    int backfacing;                  /**< True if we want are shading the
                                          backside of the surface */
};



/// RendererServices defines an abstract interface through which a 
/// renderer may provide callback to the ShadingSystem.
class OSLEXECPUBLIC RendererServices {
public:
    RendererServices () { }
    virtual ~RendererServices () { }

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation at the given time.  Return true if ok, false
    /// on error.
    virtual bool get_matrix (Matrix44 &result, TransformationPtr xform,
                             float time) = 0;

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation at the given time.  Return true if ok, false on
    /// error.  The default implementation is to use get_matrix and
    /// invert it, but a particular renderer may have a better technique
    /// and overload the implementation.
    virtual bool get_inverse_matrix (Matrix44 &result, TransformationPtr xform,
                                     float time);

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation.  Return true if ok, false on error.  Since no
    /// time value is given, also return false if the transformation may
    /// be time-varying.
    virtual bool get_matrix (Matrix44 &result, TransformationPtr xform) = 0;

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation.  Return true if ok, false on error.  Since no
    /// time value is given, also return false if the transformation may
    /// be time-varying.  The default implementation is to use
    /// get_matrix and invert it, but a particular renderer may have a
    /// better technique and overload the implementation.
    virtual bool get_inverse_matrix (Matrix44 &result, TransformationPtr xform);

    /// Get the 4x4 matrix that transforms points from the named
    /// 'from' coordinate system to "common" space at the given time.
    /// Returns true if ok, false if the named matrix is not known.
    virtual bool get_matrix (Matrix44 &result, ustring from, float time) = 0;

    /// Get the 4x4 matrix that transforms points from "common" space to
    /// the named 'to' coordinate system to at the given time.  The
    /// default implementation is to use get_matrix and invert it, but a
    /// particular renderer may have a better technique and overload the
    /// implementation.
    virtual bool get_inverse_matrix (Matrix44 &result, ustring to, float time);

    /// Get the 4x4 matrix that transforms 'from' to "common" space.
    /// Since there is no time value passed, return false if the
    /// transformation may be time-varying (as well as if it's not found
    /// at all).
    virtual bool get_matrix (Matrix44 &result, ustring from) = 0;

    /// Get the 4x4 matrix that transforms points from "common" space to
    /// the named 'to' coordinate system.  Since there is no time value
    /// passed, return false if the transformation may be time-varying
    /// (as well as if it's not found at all).  The default
    /// implementation is to use get_matrix and invert it, but a
    /// particular renderer may have a better technique and overload the
    /// implementation.
    virtual bool get_inverse_matrix (Matrix44 &result, ustring to);

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
    virtual bool get_attribute (void *renderstate, bool derivatives, 
                                ustring object, TypeDesc type, ustring name, 
                                void *val ) = 0;

    /// Similar to get_attribute();  this method will return the 'index'
    /// element of an attribute array.
    virtual bool get_array_attribute (void *renderstate, bool derivatives, 
                                      ustring object, TypeDesc type, 
                                      ustring name, int index, void *val ) = 0;

    /// Get the named user-data from the current object and write it into
    /// 'val'. If derivatives is true, the derivatives should be written into val
    /// as well. Return false if no user-data with the given name and type was
    /// found.
    virtual bool get_userdata (bool derivatives, ustring name, TypeDesc type,
                               void *renderstate, void *val) = 0;

    /// Does the current object have the named user-data associated with it?
    virtual bool has_userdata (ustring name, TypeDesc type, void *renderstate) = 0;

    /// Filtered 2D texture lookup for a single point.
    ///
    /// s,t are the texture coordinates; dsdx, dtdx, dsdy, and dtdy are
    /// the differentials of s and t change in some canonical directions
    /// x and y.  The choice of x and y are not important to the
    /// implementation; it can be any imposed 2D coordinates, such as
    /// pixels in screen space, adjacent samples in parameter space on a
    /// surface, etc.
    ///
    /// Return true if the file is found and could be opened, otherwise
    /// return false.
    virtual bool texture (ustring filename, TextureOpt &options,
                          ShaderGlobals *sg,
                          float s, float t, float dsdx, float dtdx,
                          float dsdy, float dtdy, float *result);

    /// Filtered 3D texture lookup for a single point.
    ///
    /// P is the volumetric texture coordinate; dPd{x,y,z} are the
    /// differentials of P in some canonical directions x, y, and z.
    /// The choice of x,y,z are not important to the implementation; it
    /// can be any imposed 3D coordinates, such as pixels in screen
    /// space and depth along the ray, etc.
    ///
    /// Return true if the file is found and could be opened, otherwise
    /// return false.
    virtual bool texture3d (ustring filename, TextureOpt &options,
                            ShaderGlobals *sg, const Vec3 &P,
                            const Vec3 &dPdx, const Vec3 &dPdy,
                            const Vec3 &dPdz, float *result);

    /// Filtered environment lookup for a single point.
    ///
    /// R is the directional texture coordinate; dRd[xy] are the
    /// differentials of R in canonical directions x, y.
    ///
    /// Return true if the file is found and could be opened, otherwise
    /// return false.
    virtual bool environment (ustring filename, TextureOpt &options,
                              ShaderGlobals *sg, const Vec3 &R,
                              const Vec3 &dRdx, const Vec3 &dRdy, float *result);

    /// Get information about the given texture.  Return true if found
    /// and the data has been put in *data.  Return false if the texture
    /// doesn't exist, doesn't have the requested data, if the data
    /// doesn't match the type requested. or some other failure.
    virtual bool get_texture_info (ustring filename, int subimage,
                                   ustring dataname, TypeDesc datatype,
                                   void *data);


    /// Lookup nearest points in a point cloud. It will search for points
    /// around the given center within the specified radius. A list of indices
    /// is returned so the programmer can later retrieve attributes with
    /// pointcloud_get. The indices array is mandatory, but distances can be NULL.
    /// If a derivs_offset > 0 is given, derivatives will be computed for
    /// distances (when provided).
    ///
    /// Return the number of points found, always < max_points
    virtual int pointcloud_search (ShaderGlobals *sg,
                                   ustring filename, const OSL::Vec3 &center,
                                   float radius, int max_points, bool sort,
                                   size_t *out_indices,
                                   float *out_distances, int derivs_offset) = 0;

    /// Retrieve an attribute for an index list. The result is another array
    /// of the requested type stored in out_data.
    ///
    /// Return 1 if the attribute is found, 0 otherwise.
    virtual int pointcloud_get (ustring filename, size_t *indices, int count,
                                ustring attr_name, TypeDesc attr_type,
                                void *out_data) = 0;

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

private:
    TextureSystem *m_texturesys;   // For default texture implementation
};


OSL_NAMESPACE_EXIT

#endif /* OSLEXEC_H */
