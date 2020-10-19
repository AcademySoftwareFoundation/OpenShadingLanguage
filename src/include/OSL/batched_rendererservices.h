// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>
#include <OSL/batched_texture.h>
#include <OSL/rendererservices.h>

#include <OpenImageIO/ustring.h>
#if 1

OSL_NAMESPACE_ENTER

// Users MUST derive from BatchedRendererServicesBase,
// DO NOT derive from BatchedRendererServices (it won't let you).
template<int WidthT>
class OSLEXECPUBLIC BatchedRendererServicesBase;


template<int WidthT>
class OSLEXECPUBLIC BatchedRendererServices {
public:
    typedef TextureSystem::Perthread TexturePerthread;

    static constexpr int width = WidthT;

    OSL_USING_DATA_WIDTH(WidthT);

private:
    friend class BatchedRendererServicesBase<WidthT>;
    // Private constructor to force
    // derivation from BatchedRendererServicesBase
    BatchedRendererServices (TextureSystem *texsys=NULL);
public:

    virtual ~BatchedRendererServices () { }

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation at the given time.  Return true if ok, false
    /// on error.
    virtual Mask get_matrix (BatchedShaderGlobals *bsg, Masked<Matrix44> result,
            Wide<const TransformationPtr> xform, Wide<const float> time) = 0;

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation at the given time.  Return true if ok, false on
    /// error.  The default implementation is to use get_matrix and
    /// invert it, but a particular renderer may have a better technique
    /// and overload the implementation.
    virtual Mask get_inverse_matrix (BatchedShaderGlobals *bsg, Masked<Matrix44> result,
            Wide<const TransformationPtr> xform, Wide<const float> time);
    bool is_overridden_get_inverse_matrix_WmWxWf () const { return m_is_overridden_get_inverse_matrix_WmWxWf; }

    /// Get the 4x4 matrix that transforms points from the named
    /// 'from' coordinate system to "common" space at the given time.
    /// Returns true if ok, false if the named matrix is not known.
    virtual Mask get_matrix (BatchedShaderGlobals *bsg, Masked<Matrix44> result,
                             ustring from, Wide<const float> time) = 0;
    virtual Mask get_matrix (BatchedShaderGlobals *bsg, Masked<Matrix44> result,
                             Wide<const ustring> from, Wide<const float> time);
    bool is_overridden_get_matrix_WmWsWf () const { return m_is_overridden_get_matrix_WmWsWf; }


    /// Get the 4x4 matrix that transforms points from "common" space to
    /// the named 'to' coordinate system to at the given time.  The
    /// default implementation is to use get_matrix and invert it, but a
    /// particular renderer may have a better technique and overload the
    /// implementation.
    virtual Mask get_inverse_matrix (BatchedShaderGlobals *bsg, Masked<Matrix44> result,
                                     ustring to, Wide<const float> time);
    bool is_overridden_get_inverse_matrix_WmsWf () const { return m_is_overridden_get_inverse_matrix_WmsWf; }
    virtual Mask get_inverse_matrix (BatchedShaderGlobals *bsg, Masked<Matrix44> result,
                                     Wide<const ustring> to, Wide<const float> time);
    bool is_overridden_get_inverse_matrix_WmWsWf () const { return m_is_overridden_get_inverse_matrix_WmWsWf; }

#if 0 // non linear transformations are currently unsupported, but could be added if needed

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
    virtual bool transform_points (BatchedShaderGlobals *bsg,
                                   ustring from, ustring to, float time,
                                   const Vec3 *Pin, Vec3 *Pout, int npoints,
                                   TypeDesc::VECSEMANTICS vectype)
        { return false; }

#endif

    /// Identify if the the named attribute from the renderer can be treated as uniform
    /// accross all batches to the shader.  We assume all attributes are varying unless
    /// identified as uniform by the renderer.  NOTE:  To enable constant folding of
    // an attribute value, it must be uniform and retrievable with a NULL BatchedShaderGlobals
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

    virtual Mask get_attribute (BatchedShaderGlobals *bsg,
                                ustring object, ustring name,
                                MaskedData val) = 0;
    
    /// Similar to get_attribute();  this method will fetch the 'index'
    /// element of an attribute array.
    virtual Mask get_array_attribute (BatchedShaderGlobals *bsg,
                                      ustring object,
                                      ustring name, int index, MaskedData val) = 0;

    virtual bool get_attribute_uniform (BatchedShaderGlobals *bsg,
                                ustring object, ustring name, RefData val) = 0;

    /// Similar to get_attribute();  this method will fetch the 'index'
    /// element of an attribute array.
    virtual bool get_array_attribute_uniform (BatchedShaderGlobals *bsg,
                                      ustring object,
                                      ustring name, int index, RefData val) = 0;
    
    /// Get multiple named user-data from the current object and write them into
    /// 'val'. If derivatives is true, the derivatives should be written into val
    /// as well. It is assumed the results are varying and returns Mask 
    // with its bit set to off if no user-data with the given name and type was
    /// found.
#ifdef OSL_EXPERIMENTAL_BIND_USER_DATA_WITH_LAYERNAME
    virtual Mask get_userdata (ustring name, ustring layername,
                               BatchedShaderGlobals *bsg, MaskedData val) = 0;
#else
    virtual Mask get_userdata (ustring name, 
                               BatchedShaderGlobals *bsg, MaskedData val) = 0;
#endif

    // Currently texture handles are serviced by the non-batched
    // RendererServices interface.  If necessary, customized ones could
    // be added here.  Decided to wait until the use case arises
    
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
    ///
    virtual Mask texture(ustring filename, TextureSystem::TextureHandle *texture_handle,
                                  TextureSystem::Perthread *texture_thread_info,
                                  const BatchedTextureOptions &options, BatchedShaderGlobals *bsg,
                                  Wide<const float> s, Wide<const float> t,
                                  Wide<const float> dsdx, Wide<const float> dtdx,
                                  Wide<const float> dsdy, Wide<const float> dtdy,
                                  BatchedTextureOutputs& outputs);
    bool is_overridden_texture () const { return m_is_overridden_texture; }


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
    virtual Mask texture3d (ustring filename, TextureSystem::TextureHandle *texture_handle,
                            TextureSystem::Perthread *texture_thread_info,
                            const BatchedTextureOptions &options, BatchedShaderGlobals *bsg,
                            Wide<const Vec3> P, Wide<const Vec3> dPdx, Wide<const Vec3> dPdy,
                            Wide<const Vec3> dPdz, BatchedTextureOutputs& outputs);
    bool is_overridden_texture3d () const { return m_is_overridden_texture3d; }


#if 0
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
    virtual bool environment (ustring filename, TextureSystem::TextureHandle *texture_handle,
                              TextureSystem::Perthread *texture_thread_info,
                              TextureOpt &options, ShaderGlobals *sg,
                              const Vec3 &R, const Vec3 &dRdx, const Vec3 &dRdy,
                              int nchannels, float *result,
                              float *dresultds, float *dresultdt,
                              ustring *errormessage);
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
    virtual Mask get_texture_info (BatchedShaderGlobals *bsg,
                                   TexturePerthread *texture_thread_info,
                                   Wide<const ustring> filename,
                                   // We do not need to support texture handle for varying data.
                                   int subimage,
                                   ustring dataname,
                                   MaskedData val);

    virtual bool get_texture_info_uniform (BatchedShaderGlobals *bsg,
                                           TexturePerthread *texture_thread_info,
                                           ustring filename,
                                           TextureSystem::TextureHandle *texture_handle,
                                           int subimage,
                                           ustring dataname,
                                           RefData val);

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
    virtual void trace (TraceOpt &options,  BatchedShaderGlobals *bsg, Masked<int> result,
                            Wide<const Vec3> P, Wide<const Vec3> dPdx,
                            Wide<const Vec3> dPdy, Wide<const Vec3> R,
                            Wide<const Vec3> dRdx, Wide<const Vec3> dRdy)
    {
        for (int lane = 0; lane<WidthT; ++lane)
        {
            result[lane] = 0;
        }
    }

    virtual void getmessage (BatchedShaderGlobals *bsg, Masked<int> result,
                             ustring source, ustring name, MaskedData val) {
        // Currently this code path should only be followed when source == "trace"
        OSL_DASSERT(result.mask() == val.mask());
        for (int lane = 0; lane<WidthT; ++lane)
        {
            result[lane] = 0;
        }
    }





#if 0 // Incomplete, could be added
    /// Lookup nearest points in a point cloud. It will search for
    /// points around the given center within the specified radius. A
    /// list of indices is returned so the programmer can later retrieve
    /// attributes with pointcloud_get. The indices array is mandatory,
    /// but distances can be NULL.  If a derivs_offset > 0 is given,
    /// derivatives will be computed for distances (when provided).
    ///
    /// Return the number of points found, always < max_points
    virtual int pointcloud_search (BatchedShaderGlobals *bsg,
                                   ustring filename, const OSL::Vec3 &center,
                                   float radius, int max_points, bool sort,
                                   size_t *out_indices,
                                   float *out_distances, int derivs_offset);

    /// Retrieve an attribute for an index list. The result is another array
    /// of the requested type stored in out_data.
    ///
    /// Return 1 if the attribute is found, 0 otherwise.
    virtual int pointcloud_get (BatchedShaderGlobals *bsg,
                                ustring filename, size_t *indices, int count,
                                ustring attr_name, TypeDesc attr_type,
                                void *out_data);

    /// Write a point to the named pointcloud, which will be saved
    /// at the end of the frame.  Return true if everything is ok,
    /// false if there was an error.
    virtual bool pointcloud_write (BatchedShaderGlobals *bsg,
                                   ustring filename, const OSL::Vec3 &pos,
                                   int nattribs, const ustring *names,
                                   const TypeDesc *types,
                                   const void **data);

#endif

    /// Return a pointer to the texture system (if available).
    virtual TextureSystem *texturesys () const;


protected:
    TextureSystem *m_texturesys;   // A place to hold a TextureSystem

    // Implementation detail:  for any virtual methods whose default
    // implementation exists in a target specific library,
    // we must track if the user has overridden them in a derived class.
    // BatchedRendererServicesBase will do this for us automatically.
    bool m_is_overridden_get_matrix_WmWsWf;
    bool m_is_overridden_get_inverse_matrix_WmWxWf;
    bool m_is_overridden_get_inverse_matrix_WmsWf;
    bool m_is_overridden_get_inverse_matrix_WmWsWf;
    bool m_is_overridden_texture;
    bool m_is_overridden_texture3d;
};



///////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION BELOW
///////////////////////////////////////////////////////////////////////////////

namespace {

    // Ideally we would use "is_detected" from library fundamentals TS v2 and C++17
    // Until OSL moves to require such features and compilers support of SFINAE improves,
    // we will have our own simplified versions

    // Tests for existence of specific function names with generic parameters on a type
    // NOTE: using the comma operator in the decltype to allow SFINAE to disallow
    // it from consideration when the function doesn't exist and to evaluate to
    // std::true_type when it does
    template<typename T, typename... ParameterList>
    using check_get_matrix = decltype(std::declval<T>().get_matrix(std::declval<ParameterList>()...), std::true_type());

    template<typename T, typename... ParameterList>
    using check_get_inverse_matrix = decltype(std::declval<T>().get_inverse_matrix(std::declval<ParameterList>()...), std::true_type());

    template<typename T, typename... ParameterList>
    using check_texture = decltype(std::declval<T>().texture(std::declval<ParameterList>()...), std::true_type());

    template<typename T, typename... ParameterList>
    using check_texture3d = decltype(std::declval<T>().texture3d(std::declval<ParameterList>()...), std::true_type());

    template <template<class> class CheckerT, typename T>
    struct SFINAE_checker
    {
        template <typename, typename> static std::false_type check(...);
        // NOTE: Trailing return type uses SFINAE
        template <typename DerivedT, typename P> static auto check(P * p)->CheckerT<DerivedT>;

        struct Dummy {};
        typedef decltype(check<T, Dummy>(nullptr)) type;
        static constexpr bool value = std::is_same<std::true_type, type>::value;
    };

    template <template<class> class CheckerT, typename T>
    inline constexpr bool has_method() { return SFINAE_checker<CheckerT, T>::value; }

} // anonymous namespace




// NOTE:  for methods that we want to have default implementations optimized in
// target specific libraries, a virtual function call won't work.  A virtual
// function requires that all method addresses be resolved for the V-Table
// to be populated.  As the implementation we want to call exist in dynamically
// loaded shared libraries that will not work.
//
// Instead our target specific libraries will query the BatchedRendererServices
// to see if a method has been overriden.  If the method is not overriden
// we will just call the virtual function on BatchedRendererServices.
// Otherwise it will call the a default version that exists only
// inside the optimized target specific library.
template<int WidthT>
class OSLEXECPUBLIC BatchedRendererServicesBase
: protected BatchedRendererServices<WidthT>
// To detect if derived classes have overloaded
// functions, we need to make them protected so
// that DerivedT::method will cause
// SFINAE (Substitution Failure Is Not An Error)
// if the method is NOT overriden.
{
private:
    typedef BatchedRendererServices<WidthT> Base;
public:
    OSL_USING_DATA_WIDTH(WidthT);

private:
    // Making methods with default implementations in target specific
    // libraries private to prevent a derivation from
    // chain calling down to them.
    using Base::get_inverse_matrix;
    using Base::texture;
    using Base::texture3d;

protected:
    struct Detector {

        template<typename T>
        using get_matrix_WmWsWf = check_get_matrix<T, BatchedShaderGlobals *, Masked<Matrix44>, Wide<const ustring>, Wide<const float>>;

        template<typename T>
        using get_inverse_matrix_WmWxWf = check_get_inverse_matrix<T, BatchedShaderGlobals *, Masked<Matrix44>, Wide<const TransformationPtr>, Wide<const float>>;
        template<typename T>
        using get_inverse_matrix_WmsWf = check_get_inverse_matrix<T, BatchedShaderGlobals *, Masked<Matrix44>, ustring, Wide<const float>>;
        template<typename T>
        using get_inverse_matrix_WmWsWf = check_get_inverse_matrix<T, BatchedShaderGlobals *, Masked<Matrix44>, Wide<const ustring>, Wide<const float>>;

        template<typename T>
        using texture = check_texture<T, ustring, TextureSystem::TextureHandle *,
            TextureSystem::Perthread *,
            const BatchedTextureOptions &, BatchedShaderGlobals *,
            const Wide<const float> &, const Wide<const float> &,
            const Wide<const float> &, const Wide<const float> &,
            const Wide<const float> &, const Wide<const float> &,
            BatchedTextureOutputs&>;

        template<typename T>
        using texture3d = check_texture3d<T, ustring, TextureSystem::TextureHandle *,
                TextureSystem::Perthread *,
                const BatchedTextureOptions &, BatchedShaderGlobals *,
                Wide<const Vec3>, Wide<const Vec3>, Wide<const Vec3>,
                Wide<const Vec3>, BatchedTextureOutputs&>;


        // Ensure our tests for detecting function's existence work on BatchedRendererServices
        static_assert(has_method<get_matrix_WmWsWf, Base>(), "Must keep get_matrix_WmWsWf parameters in sync with the test for its existence on a derived class");
        static_assert(has_method<get_inverse_matrix_WmWxWf, Base>(), "Must keep get_inverse_matrix parameters in sync with the test for its existence on a derived class");
        static_assert(has_method<get_inverse_matrix_WmsWf, Base>(), "Must keep get_inverse_matrix parameters in sync with the test for its existence on a derived class");
        static_assert(has_method<get_inverse_matrix_WmWsWf, Base>(), "Must keep get_inverse_matrix parameters in sync with the test for its existence on a derived class");
        static_assert(has_method<texture, Base>(), "Must keep texture parameters in sync with the test for its existence on a derived class");
        static_assert(has_method<texture3d, Base>(), "Must keep texture3d parameters in sync with the test for its existence on a derived class");

        // Ensure that none of these methods are accessible by BatchedRendererServicesBase
        static_assert(!has_method<get_matrix_WmWsWf, BatchedRendererServicesBase>(), "get_matrix is not supposed to be accessible from BatchedRendererServicesBase, make sure inheritance chain is protected");
        static_assert(!has_method<get_inverse_matrix_WmWxWf, BatchedRendererServicesBase>(), "get_inverse_matrix is not supposed to be accessible from BatchedRendererServicesBase, make sure inheritance chain is protected");
        static_assert(!has_method<get_inverse_matrix_WmsWf, BatchedRendererServicesBase>(), "get_inverse_matrix is not supposed to be accessible from BatchedRendererServicesBase, make sure inheritance chain is protected");
        static_assert(!has_method<get_inverse_matrix_WmWsWf, BatchedRendererServicesBase>(), "get_inverse_matrix is not supposed to be accessible from BatchedRendererServicesBase, make sure inheritance chain is protected");
        static_assert(!has_method<texture, BatchedRendererServicesBase>(), "texture is not supposed to be accessible from BatchedRendererServicesBase, make sure inheritance chain is protected");
        static_assert(!has_method<texture3d, BatchedRendererServicesBase>(), "texture3d is not supposed to be accessible from BatchedRendererServicesBase, make sure inheritance chain is protected");

        // If the renderer uses a class hierarchy for its derivation, then
        // each class's constructor should call
        // Detector::detectOverrides(this, this);
        // so that we can test that specific type for overrides
        template<typename DerivedT>
        static void detectOverrides (DerivedT *, Base* base)
        {
            static_assert(std::is_base_of<BatchedRendererServicesBase, DerivedT>::value, "Only valid for derived classes");
            static_assert(!std::is_same<BatchedRendererServicesBase, DerivedT>::value, "Only valid for derived classes");

            // If the method is accessible by DerivedT, then it must have been
            // overridden at some point in the class hierarchy from BatchedRendererServicesBase
            if (has_method<get_matrix_WmWsWf, DerivedT>()) base->m_is_overridden_get_matrix_WmWsWf = true;
            if (has_method<get_inverse_matrix_WmWxWf, DerivedT>()) base->m_is_overridden_get_inverse_matrix_WmWxWf = true;
            if (has_method<get_inverse_matrix_WmsWf, DerivedT>()) base->m_is_overridden_get_inverse_matrix_WmsWf  = true;
            if (has_method<get_inverse_matrix_WmWsWf, DerivedT>()) base->m_is_overridden_get_inverse_matrix_WmWsWf = true;
            if (has_method<texture, DerivedT>()) base->m_is_overridden_texture = true;
            if (has_method<texture3d, DerivedT>()) base->m_is_overridden_texture3d = true;
        }
    };
public:

    // Only expose this templated constructor to ensure that at least the next type in
    // the hierarchy is tested for overrides
    template<typename DerivedT>
    OSL_FORCEINLINE BatchedRendererServicesBase (DerivedT *derived, TextureSystem *texsys=NULL)
    : BatchedRendererServices<WidthT>(texsys)
    {
        Detector::detectOverrides(derived, this);
    }
};

OSL_NAMESPACE_EXIT
#endif
