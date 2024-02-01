// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once


#include <OSL/encodedtypes.h>
#include <OSL/oslconfig.h>
#include <OSL/variant.h>


OSL_NAMESPACE_ENTER

class RendererServices;
template<int WidthT> class BatchedRendererServices;
class ShadingContext;
struct ShaderGlobals;
class ShaderGroup;

// Tags for polymorphic dispatch
template<int SimdWidthT> class WidthOf {};


/// Opaque pointer to whatever the renderer uses to represent a
/// (potentially motion-blurred) coordinate transformation.
typedef const void* TransformationPtr;


// Callbacks for closure creation
typedef void (*PrepareClosureFunc)(RendererServices*, int id, void* data);
typedef void (*SetupClosureFunc)(RendererServices*, int id, void* data);

enum class AttributeSpecBuiltinArg {
    ShaderGlobalsPointer,  // void* (TODO: ideally ShaderGlobals*)
    ShadeIndex,            // int
    Derivatives,           // bool
    Type,                  // TypeDesc_pod
    ArrayIndex,            // int, Always zero for non-indexed array lookups.
    IsArrayLookup,         // bool
    ObjectName,            // const char* (TODO: change to ustringhash)
    AttributeName,         // const char* (TODO: change to ustringhash)
};

using AttributeSpecArg    = ArgVariant<AttributeSpecBuiltinArg>;
using AttributeGetterSpec = FunctionSpec<AttributeSpecArg>;

// Turn off warnings about unused params for this file, since we have lots
// of declarations with stub function bodies.
OSL_PRAGMA_WARNING_PUSH
OSL_GCC_PRAGMA(GCC diagnostic ignored "-Wunused-parameter")



/// RendererServices defines an abstract interface through which a
/// renderer may provide callback to the ShadingSystem.
class OSLEXECPUBLIC RendererServices {
    // Keep interface in sync with rs_free_function.h
public:
    typedef TextureSystem::TextureHandle TextureHandle;
    typedef TextureSystem::Perthread TexturePerthread;


    RendererServices(TextureSystem* texsys = NULL);
    virtual ~RendererServices() {}

    /// Given the name of a 'feature', return whether this RendererServices
    /// supports it. Feature names include:
    ///    "OptiX"
    ///    "build_attribute_getter"
    ///
    /// This allows some customization of JIT generated code based on the
    /// facilities and features of a particular renderer. It also allows
    /// future expansion of RendererServices methods (with trivial default
    /// implementations) without requiring every renderer implementation to
    /// support it, as long as the OSL runtime only uses that feature if the
    /// supports("feature") says it's present, thus preserving source
    /// compatibility.
    virtual int supports(string_view feature) const { return false; }

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation at the given time.  Return true if ok, false
    /// on error.
    virtual bool get_matrix(ShaderGlobals* sg, Matrix44& result,
                            TransformationPtr xform, float time)
    {
        return false;
    }

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation at the given time.  Return true if ok, false on
    /// error.  The default implementation is to use get_matrix and
    /// invert it, but a particular renderer may have a better technique
    /// and overload the implementation.
    virtual bool get_inverse_matrix(ShaderGlobals* sg, Matrix44& result,
                                    TransformationPtr xform, float time);

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation.  Return true if ok, false on error.  Since no
    /// time value is given, also return false if the transformation may
    /// be time-varying.
    virtual bool get_matrix(ShaderGlobals* sg, Matrix44& result,
                            TransformationPtr xform)
    {
        return false;
    }

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation.  Return true if ok, false on error.  Since no
    /// time value is given, also return false if the transformation may
    /// be time-varying.  The default implementation is to use
    /// get_matrix and invert it, but a particular renderer may have a
    /// better technique and overload the implementation.
    virtual bool get_inverse_matrix(ShaderGlobals* sg, Matrix44& result,
                                    TransformationPtr xform);

    /// Get the 4x4 matrix that transforms points from the named
    /// 'from' coordinate system to "common" space at the given time.
    /// Returns true if ok, false if the named matrix is not known.
    virtual bool get_matrix(ShaderGlobals* sg, Matrix44& result,
                            ustringhash from, float time);

    /// Get the 4x4 matrix that transforms points from "common" space to
    /// the named 'to' coordinate system to at the given time.  The
    /// default implementation is to use get_matrix and invert it, but a
    /// particular renderer may have a better technique and overload the
    /// implementation.
    virtual bool get_inverse_matrix(ShaderGlobals* sg, Matrix44& result,
                                    ustringhash to, float time);

    /// Get the 4x4 matrix that transforms 'from' to "common" space.
    /// Since there is no time value passed, return false if the
    /// transformation may be time-varying (as well as if it's not found
    /// at all).
    virtual bool get_matrix(ShaderGlobals* sg, Matrix44& result,
                            ustringhash from);

    /// Get the 4x4 matrix that transforms points from "common" space to
    /// the named 'to' coordinate system.  Since there is no time value
    /// passed, return false if the transformation may be time-varying
    /// (as well as if it's not found at all).  The default
    /// implementation is to use get_matrix and invert it, but a
    /// particular renderer may have a better technique and overload the
    /// implementation.
    virtual bool get_inverse_matrix(ShaderGlobals* sg, Matrix44& result,
                                    ustringhash to);

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
    virtual bool transform_points(ShaderGlobals* sg, ustringhash from,
                                  ustringhash to, float time, const Vec3* Pin,
                                  Vec3* Pout, int npoints,
                                  TypeDesc::VECSEMANTICS vectype);

    /// Report errors, warnings, printf, and fprintf.
    /// Fmtlib style format specifier is used (vs. printf style)
    /// Arguments are represented as EncodedTypes (encodedtypes.h) and
    /// packed into an arg_values buffer.  OSL::decode_message converts these
    /// arguments into a std::string for renderer's handle to use as they please.
    /// For device compatibility, the format specifier and any string arguments
    /// are passed as ustringhash's.
    /// Default implementation decodes the messages and fowards them to the
    /// ShadingContext onto the ShadyingSystem's ErrorHandler.
    /// It is recomended to override and make use of the
    /// journal buffer to record everything to a buffer than can be post
    /// processed as needed vs. going through the ShadingSystem ErrorHandler.
    virtual void errorfmt(OSL::ShaderGlobals* sg,
                          OSL::ustringhash fmt_specification, int32_t arg_count,
                          const EncodedType* arg_types,
                          uint32_t arg_values_size, uint8_t* arg_values);

    virtual void warningfmt(OSL::ShaderGlobals* sg,
                            OSL::ustringhash fmt_specification,
                            int32_t arg_count, const EncodedType* arg_types,
                            uint32_t arg_values_size, uint8_t* arg_values);

    virtual void printfmt(OSL::ShaderGlobals* sg,
                          OSL::ustringhash fmt_specification, int32_t arg_count,
                          const EncodedType* arg_types,
                          uint32_t arg_values_size, uint8_t* arg_values);

    virtual void filefmt(OSL::ShaderGlobals* sg, OSL::ustringhash filename_hash,
                         OSL::ustringhash fmt_specification, int32_t arg_count,
                         const EncodedType* arg_types, uint32_t arg_values_size,
                         uint8_t* arg_values);

    /// Builds a free function to provide a value for a given attribute.
    /// This occurs at shader compile time, not at execution time.
    ///
    /// @param group
    ///     The shader group currently requesting the attribute.
    ///
    /// @param is_object_lookup
    ///     True if an object name was specified, even if the value is not
    ///     known at compile time.
    ///
    /// @param object_name
    ///     The object name, or nullptr if the value is not specified or it
    ///     is not known at compile time.
    ///
    /// @param attribute_name
    ///     The attribute name, or nullptr if the value is not known at
    ///     compile time.
    ///
    /// @param is_array_lookup
    ///     True if the attribute lookup provides an index.
    ///
    /// @param array_index
    ///     The array index, or nullptr if the value is not specified or it
    ///     is not known at compile time.
    ///
    /// @param type
    ///     The type of the value being requested.
    ///
    /// @param derivatives
    ///     True if derivatives are also being requested.
    ///
    /// @param spec
    ///     The built attribute getter. An empty function name is interpreted
    ///     as a missing attribute.
    ///
    virtual void
    build_attribute_getter(const ShaderGroup& group, bool is_object_lookup,
                           const ustring* object_name,
                           const ustring* attribute_name, bool is_array_lookup,
                           const int* array_index, TypeDesc type,
                           bool derivatives, AttributeGetterSpec& spec);

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
    virtual bool get_attribute(ShaderGlobals* sg, bool derivatives,
                               ustringhash object, TypeDesc type,
                               ustringhash name, void* val);

    /// Similar to get_attribute();  this method will return the 'index'
    /// element of an attribute array.
    virtual bool get_array_attribute(ShaderGlobals* sg, bool derivatives,
                                     ustringhash object, TypeDesc type,
                                     ustringhash name, int index, void* val);

    /// Get the named user-data from the current object and write it into
    /// 'val'. If derivatives is true, the derivatives should be written into val
    /// as well. Return false if no user-data with the given name and type was
    /// found.
    virtual bool get_userdata(bool derivatives, ustringhash name, TypeDesc type,
                              ShaderGlobals* sg, void* val);

    /// Given the name of a texture, return an opaque handle that can be used
    /// with texture calls to avoid the name lookups. The `options`, if not
    /// null, may be used in renderer-specific ways to specialize a handle
    /// based on certain texture option choices.
    virtual TextureHandle*
    get_texture_handle(ustring filename, ShadingContext* context,
                       const TextureOpt* options = nullptr);
    virtual TextureHandle*
    get_texture_handle(ustringhash filename, ShadingContext* context,
                       const TextureOpt* options = nullptr);

    /// Return true if the texture handle (previously returned by
    /// get_texture_handle()) is a valid texture that can be subsequently
    /// read or sampled.
    virtual bool good(TextureHandle* texture_handle);

    /// Return true if the texture handle (previously returned by
    /// get_texture_handle()) is udim
    virtual bool is_udim(TextureHandle* texture_handle);

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
    virtual bool texture(ustringhash filename, TextureHandle* texture_handle,
                         TexturePerthread* texture_thread_info,
                         TextureOpt& options, ShaderGlobals* sg, float s,
                         float t, float dsdx, float dtdx, float dsdy,
                         float dtdy, int nchannels, float* result,
                         float* dresultds, float* dresultdt,
                         ustringhash* errormessage);

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
    virtual bool texture3d(ustringhash filename, TextureHandle* texture_handle,
                           TexturePerthread* texture_thread_info,
                           TextureOpt& options, ShaderGlobals* sg,
                           const Vec3& P, const Vec3& dPdx, const Vec3& dPdy,
                           const Vec3& dPdz, int nchannels, float* result,
                           float* dresultds, float* dresultdt, float* dresultdr,
                           ustringhash* errormessage);

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
    virtual bool environment(ustringhash filename,
                             TextureHandle* texture_handle,
                             TexturePerthread* texture_thread_info,
                             TextureOpt& options, ShaderGlobals* sg,
                             const Vec3& R, const Vec3& dRdx, const Vec3& dRdy,
                             int nchannels, float* result, float* dresultds,
                             float* dresultdt, ustringhash* errormessage);

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
    virtual bool get_texture_info(ustringhash filename,
                                  TextureHandle* texture_handle,
                                  TexturePerthread* texture_thread_info,
                                  ShaderGlobals* sg, int subimage,
                                  ustringhash dataname, TypeDesc datatype,
                                  void* data, ustringhash* errormessage);

    virtual bool
    get_texture_info(ustringhash filename, TextureHandle* texture_handle,
                     float s, float t, TexturePerthread* texture_thread_info,
                     ShaderGlobals* sg, int subimage, ustringhash dataname,
                     TypeDesc datatype, void* data, ustringhash* errormessage);


    /// Lookup nearest points in a point cloud. It will search for
    /// points around the given center within the specified radius. A
    /// list of indices is returned so the programmer can later retrieve
    /// attributes with pointcloud_get. The indices array is mandatory,
    /// but distances can be NULL.  If a derivs_offset > 0 is given,
    /// derivatives will be computed for distances (when provided).
    ///
    /// Return the number of points found, always < max_points
    virtual int pointcloud_search(ShaderGlobals* sg, ustringhash filename,
                                  const OSL::Vec3& center, float radius,
                                  int max_points, bool sort,
                                  size_t* out_indices, float* out_distances,
                                  int derivs_offset);

    /// Retrieve an attribute for an index list. The result is another array
    /// of the requested type stored in out_data.
    ///
    /// Return 1 if the attribute is found, 0 otherwise.
    virtual int pointcloud_get(ShaderGlobals* sg, ustringhash filename,
                               size_t* indices, int count,
                               ustringhash attr_name, TypeDesc attr_type,
                               void* out_data);

    /// Write a point to the named pointcloud, which will be saved
    /// at the end of the frame.  Return true if everything is ok,
    /// false if there was an error.
    virtual bool pointcloud_write(ShaderGlobals* sg, ustringhash filename,
                                  const OSL::Vec3& pos, int nattribs,
                                  const ustringrep* names,
                                  const TypeDesc* types, const void** data);

    /// Options for the trace call.
    struct TraceOpt {
        float mindist;        ///< ignore hits closer than this
        float maxdist;        ///< ignore hits farther than this
        bool shade;           ///< whether to shade what is hit
        ustringrep traceset;  ///< named trace set
        TraceOpt() : mindist(0.0f), maxdist(1.0e30), shade(false) {}

        enum class LLVMMemberIndex {
            mindist = 0,
            maxdist,
            shade,
            traceset,
            count
        };
    };

    /// Immediately trace a ray from P in the direction R.  Return true
    /// if anything hit, otherwise false.
    virtual bool trace(TraceOpt& options, ShaderGlobals* sg, const OSL::Vec3& P,
                       const OSL::Vec3& dPdx, const OSL::Vec3& dPdy,
                       const OSL::Vec3& R, const OSL::Vec3& dRdx,
                       const OSL::Vec3& dRdy);

    /// Get the named message from the renderer and if found then
    /// write it into 'val'.  Otherwise, return false.  This is only
    /// called for "sourced" messages, not ordinary intra-group messages.
    virtual bool getmessage(ShaderGlobals* sg, ustringhash source,
                            ustringhash name, TypeDesc type, void* val,
                            bool derivatives);

    /// Return a pointer to the texture system (if available).
    virtual TextureSystem* texturesys() const;

    /// Allocate `size` bytes of memory on the device that will execute the
    /// shaders. (Equivalent to malloc() on the CPU.)
    virtual void* device_alloc(size_t size)
    {
        return nullptr;
        // Note: for an OptiX-based renderer, this method should be overriden
        // with something like:
        //
        //     void* dptr;
        //     auto r = cudaMalloc(&dptr, size);
        //     return r == cudaSuccess ? dptr : nullptr;
    }

    /// Free a previous allocation (by `device_alloc()`) on the device that
    /// will execute the shaders. (Equivalent to free() on the CPU.)
    virtual void device_free(void* ptr)
    {
        // Note: for an OptiX-based renderer, this method should be overriden
        // with something like:
        //
        //     cudaFree(ptr);
    }

    /// Copy `size` bytes from location `src_host` on the host/CPU (the
    /// machine making this call) into location `dst_device` on the device
    /// executing shaders. (Equivalent to `memcpy(dst, src, size)` on the
    /// CPU.)
    virtual void* copy_to_device(void* dst_device, const void* src_host,
                                 size_t size)
    {
        return nullptr;
        // Note: for an OptiX-based renderer, this method should be overriden
        // with something like:
        //
        //     auto r = cudaMemcpy(dst_device, src_host, size,
        //                         cudaMemcpyHostToDevice);
        //     return dst_device;
    }

    /// Options we use for noise calls.
    struct NoiseOpt {
        int anisotropic;
        int do_filter;
        Vec3 direction;
        float bandwidth;
        float impulses;
        NoiseOpt()
            : anisotropic(0)
            , do_filter(true)
            , direction(1.0f, 0.0f, 0.0f)
            , bandwidth(1.0f)
            , impulses(16.0f)
        {
        }
    };

    /// A renderer may choose to support batched execution by providing pointers
    /// to objects satisfying the BatchedRendererServices<WidthOf<#>> interface
    /// for specific batch sizes.
    /// Unless overridden, a nullptr is returned.
    virtual BatchedRendererServices<16>* batched(WidthOf<16>);
    virtual BatchedRendererServices<8>* batched(WidthOf<8>);

protected:
    TextureSystem* m_texturesys;  // A place to hold a TextureSystem
};


OSL_PRAGMA_WARNING_POP
OSL_NAMESPACE_EXIT
