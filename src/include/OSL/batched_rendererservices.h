// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/batched_shaderglobals.h>
#include <OSL/batched_texture.h>
#include <OSL/oslconfig.h>
#include <OSL/rendererservices.h>

#include <OpenImageIO/ustring.h>

OSL_NAMESPACE_ENTER

// Implementation detail:  virtual methods that have a corresponding
// virtual bool is_overridden_* method
// will not be called unless is_overridden_*() returns true;
// This requirement allows the codegenerator to use target specific
// default implementations instead of virtual function calls

// See <OSL/wide.h> for detailed documentation and implementation,
// here is a brief summary of Structure of Array data layout wrappers
//
// Wide<T> contains a T value for WidthT lanes and is accessed
// with [lane] where lane in [0-(WidthT-1)]
//
// Masked<T> bundles a T value for WidthT lanes with a bitmask representing
// the active lanes.  So you don't see a Mask or other representation of active
// lanes passed separately, it should be bundled with the data it applies to
//
// MaskedData bundles any type of value for WidthT lanes with a bitmask
// representing the active lanes.  MaskData can be invalid, may have derivs,
// and underlying data type can be queried or tested.  IE:
//     void myFunction(MaskedData<16> any) {
//         if (Masked<Vec3>::is(any) {
//             Masked<Vec3> vecVal(any);
//
// RefData bundles any type of value.  RefData can be invalid, may have derivs,
// and underlying data type can be queried or tested.  IE:
//     void myFunction(RefData any) {
//         if (Ref<int[2]>::is(any) {
//             Ref<int[2]> pairVal(any);

template<int WidthT> class OSLEXECPUBLIC BatchedRendererServices {
public:
    typedef TextureSystem::Perthread TexturePerthread;

    static constexpr int width = WidthT;

    // To avoid having the specify the WidthT parameter
    // to every templated wrapper, we can create class
    // scoped aliases for this class's WidthT
    OSL_USING_DATA_WIDTH(WidthT);

    BatchedRendererServices(TextureSystem* texsys = NULL);
    virtual ~BatchedRendererServices() {}

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation at the given time.  Return a Mask with lanes set to
    /// true if ok, false on error.
    virtual Mask get_matrix(BatchedShaderGlobals* bsg, Masked<Matrix44> wresult,
                            Wide<const TransformationPtr> wxform,
                            Wide<const float> wtime)
    {
        return Mask(false);
    }

    /// Get the 4x4 matrix that transforms by the specified
    /// transformation at the given time.  Return a Mask with lanes set to true
    /// if ok, false on error.  The default implementation is to use get_matrix and
    /// invert it, but a particular renderer may have a better technique
    /// and overload the implementation.
    virtual Mask get_inverse_matrix(BatchedShaderGlobals* bsg,
                                    Masked<Matrix44> wresult,
                                    Wide<const TransformationPtr> wxform,
                                    Wide<const float> wtime);
    virtual bool is_overridden_get_inverse_matrix_WmWxWf() const = 0;

    /// Get the 4x4 matrix that transforms points from the named
    /// 'from' coordinate system to "common" space at the given time.
    /// Return a Mask with lanes set to  true if ok, false if the named matrix
    /// is not known.
    virtual Mask get_matrix(BatchedShaderGlobals* bsg, Masked<Matrix44> wresult,
                            ustringhash from, Wide<const float> wtime)
    {
        return Mask(false);
    }
    virtual Mask get_matrix(BatchedShaderGlobals* bsg, Masked<Matrix44> result,
                            Wide<const ustringhash> wfrom,
                            Wide<const float> wtime);
    virtual bool is_overridden_get_matrix_WmWsWf() const = 0;


    /// Get the 4x4 matrix that transforms points from "common" space to
    /// the named 'to' coordinate system to at the given time.  The
    /// default implementation is to use get_matrix and invert it, but a
    /// particular renderer may have a better technique and overload the
    /// implementation.
    virtual Mask get_inverse_matrix(BatchedShaderGlobals* bsg,
                                    Masked<Matrix44> wresult, ustringhash to,
                                    Wide<const float> wtime);
    virtual bool is_overridden_get_inverse_matrix_WmsWf() const = 0;
    virtual Mask get_inverse_matrix(BatchedShaderGlobals* bsg,
                                    Masked<Matrix44> wresult,
                                    Wide<const ustringhash> wto,
                                    Wide<const float> wtime);
    virtual bool is_overridden_get_inverse_matrix_WmWsWf() const = 0;

    // non linear transformations are currently unsupported,
    // but could be added if needed

    /// Identify if the the named attribute from the renderer can be treated as uniform
    /// across all batches to the shader.  We assume all attributes are varying unless
    /// identified as uniform by the renderer.  NOTE:  To enable constant folding of
    /// an attribute value, it must be uniform and retrievable with
    /// a NULL BatchedShaderGlobals
    virtual bool is_attribute_uniform(ustring object, ustring name)
    {
        return false;
    }

    /// Get the named attribute from the renderer and if found then
    /// write it into 'val'.  Otherwise, return false.  If no object is
    /// specified (object == ustring()), then the renderer should search *first*
    /// for the attribute on the currently shaded object, and next, if
    /// unsuccessful, on the currently shaded "scene".
    virtual Mask get_attribute(BatchedShaderGlobals* bsg, ustringhash object,
                               ustringhash name, MaskedData wval)
    {
        return Mask(false);
    }

    /// Similar to get_attribute();  this method will fetch the 'index'
    /// element of an attribute array.
    virtual Mask get_array_attribute(BatchedShaderGlobals* bsg,
                                     ustringhash object, ustringhash name,
                                     int index, MaskedData wval)
    {
        return Mask(false);
    }

    virtual bool get_attribute_uniform(BatchedShaderGlobals* bsg,
                                       ustringhash object, ustringhash name,
                                       RefData val)
    {
        return false;
    }

    /// Similar to get_attribute();  this method will fetch the 'index'
    /// element of an attribute array.
    virtual bool get_array_attribute_uniform(BatchedShaderGlobals* bsg,
                                             ustringhash object,
                                             ustringhash name, int index,
                                             RefData val)
    {
        return false;
    }

    /// Get multiple named user-data from the current object and write them into
    /// 'val'. If derivatives is true, the derivatives should be written into val
    /// as well. It is assumed the results are varying and returns Mask
    // with its bit set to off if no user-data with the given name and type was
    /// found.
    virtual Mask get_userdata(ustringhash name, BatchedShaderGlobals* bsg,
                              MaskedData wval)
    {
        return Mask(false);
    }

    // Currently texture handles are serviced by the non-batched
    // RendererServices interface.  If necessary, customized ones could
    // be added here.  Decided to wait until the use case arises

    /// Texturing is setup for a single filename/handle and
    /// BatchedTextureOptions (containing uniform and varying sections)
    /// and supports varying texture coordinates and derivatives.

    /// BatchedTextureOutputs (defined in batched_texture.h) encapsulates a
    /// Masked result + derivs, Masked alpha channel + derivs, and Masked
    /// errormessage. Primary purpose is to allow a single Mask instance to
    /// be shared among the multiple Masked accessors.

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
    /// Return Mask with lanes set to true if the file is found and could be
    /// opened, otherwise return false.
    ///
    /// If the errormessage is NULL, this method is expected to
    /// handle the errors fully, including forwarding them to the renderer
    /// or shading system. If errormessage is non-NULL, any resulting error
    /// messages (in case of failure, when the function returns false) will
    /// be stored there, leaving it up to the caller/shader to handle the
    /// error.
    ///
    virtual Mask
    texture(ustringhash filename, TextureSystem::TextureHandle* texture_handle,
            TextureSystem::Perthread* texture_thread_info,
            const BatchedTextureOptions& options, BatchedShaderGlobals* bsg,
            Wide<const float> ws, Wide<const float> wt, Wide<const float> wdsdx,
            Wide<const float> wdtdx, Wide<const float> wdsdy,
            Wide<const float> wdtdy, BatchedTextureOutputs& outputs);
    virtual bool is_overridden_texture() const = 0;


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
    /// Return a Mask with lanes set to true if the file is found and could
    /// be opened, otherwise return false.
    ///
    /// If the errormessage parameter is NULL, this method is expected to
    /// handle the errors fully, including forwarding them to the renderer
    /// or shading system. If errormessage is non-NULL, any resulting error
    /// messages (in case of failure, when the function returns false) will
    /// be stored there, leaving it up to the caller/shader to handle the
    /// error.
    virtual Mask texture3d(ustringhash filename,
                           TextureSystem::TextureHandle* texture_handle,
                           TextureSystem::Perthread* texture_thread_info,
                           const BatchedTextureOptions& options,
                           BatchedShaderGlobals* bsg, Wide<const Vec3> wP,
                           Wide<const Vec3> wdPdx, Wide<const Vec3> wdPdy,
                           Wide<const Vec3> wdPdz,
                           BatchedTextureOutputs& outputs);
    virtual bool is_overridden_texture3d() const = 0;


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
    /// Return a Mask with lanes set to true if the file is found and could
    /// be opened, otherwise return false.
    ///
    /// If the errormessage parameter is NULL, this method is expected to
    /// handle the errors fully, including forwarding them to the renderer
    /// or shading system. If errormessage is non-NULL, any resulting error
    /// messages (in case of failure, when the function returns false) will
    /// be stored there, leaving it up to the caller/shader to handle the
    /// error.
    virtual Mask environment(ustringhash filename,
                             TextureSystem::TextureHandle* texture_handle,
                             TextureSystem::Perthread* texture_thread_info,
                             const BatchedTextureOptions& options,
                             BatchedShaderGlobals* bsg, Wide<const Vec3> wR,
                             Wide<const Vec3> wdRdx, Wide<const Vec3> wdRdy,
                             BatchedTextureOutputs& outputs);
    virtual bool is_overridden_environment() const = 0;



    /// Get information about the given texture.  Return a Mask with lanes
    /// set to true if found and the data has been put in *data.  Mask lanes
    /// set to false if the texture doesn't exist, doesn't have the requested
    /// data, if the data doesn't match the type requested. or some other
    /// failure.
    ///
    /// The filename will always be passed, and it's ok for the renderer
    /// implementation to use only that (and in fact should be prepared to
    /// deal with texture_handle and texture_thread_info being NULL). But
    /// sometimes OSL can figure out the texture handle or thread info also
    /// and may pass them as non-NULL, in which case the renderer may (if it
    /// can) use that extra information to perform a less expensive texture
    /// lookup.

    virtual TextureSystem::TextureHandle* resolve_udim_uniform(
        BatchedShaderGlobals* bsg, TexturePerthread* texture_thread_info,
        ustringhash filename, TextureSystem::TextureHandle* texture_handle,
        float S, float T);

    virtual void resolve_udim(BatchedShaderGlobals* bsg,
                              TexturePerthread* texture_thread_info,
                              ustringhash filename,
                              TextureSystem::TextureHandle* texture_handle,
                              Wide<const float> wS, Wide<const float> wT,
                              Masked<TextureSystem::TextureHandle*> wresult);

    // Assumes any UDIM has been resolved already
    virtual bool get_texture_info_uniform(
        BatchedShaderGlobals* bsg, TexturePerthread* texture_thread_info,
        ustringhash filename, TextureSystem::TextureHandle* texture_handle,
        int subimage, ustringhash dataname, RefData val);


    /// Lookup nearest points in a point cloud. It will search for
    /// points around the given center within the specified radius. A
    /// list of indices is returned so the programmer can later retrieve
    /// attributes with pointcloud_get. The indices array is mandatory,
    /// but distances MaskedData can be invalid (empty), may or may not
    /// have derivs, will always be float[], and contains the mask to use
    /// for the operations.
    ///
    /// out_num_points will contain the number of points found, always < max_points

    // To enable sharing of single mask with multiple outputs we use
    // a class to encapsulate multiple wide pointers with a single mask.
    // Because the Masked accessors will be constructed by the helper methods
    // once inlined compilers will see the same mask_value is used for both.
    class PointCloudSearchResults {
        void* m_wnum_points;
        void* m_windices;
        int m_indices_array_length;
        void* m_wdistances;
        int m_distances_array_length;
        bool m_distances_have_derivs;
        Mask m_mask;

    public:
        PointCloudSearchResults(void* wnum_points, void* windices,
                                int indices_array_length, void* wdistances,
                                int distances_array_length,
                                int distances_have_derivs, int mask_value)
            : m_wnum_points(wnum_points)
            , m_windices(windices)
            , m_indices_array_length(indices_array_length)
            , m_wdistances(wdistances)
            , m_distances_array_length(distances_array_length)
            , m_distances_have_derivs(distances_have_derivs)
            , m_mask(mask_value)
        {
        }

        OSL_FORCEINLINE Mask mask() const { return m_mask; }

        // NOTE:  Always assign to a local object before calling an
        // array subscript operator [].  Bugs can occur if the []
        // is called on a temporary object as a reference to a
        // temporary object would be created.
        // DO NOT DO THIS:
        //     auto out_num_points = results.wnum_points()[lane];
        // instead
        //     auto wnum_points = results.wnum_points();
        //     auto out_num_points = wnum_points[lane];

        OSL_FORCEINLINE Masked<int> wnum_points() const
        {
            return Masked<int>(m_wnum_points, m_mask);
        }

        OSL_FORCEINLINE Masked<int[]> windices() const
        {
            return Masked<int[]>(m_windices, m_indices_array_length, m_mask);
        }

        OSL_FORCEINLINE bool has_distances() const
        {
            return m_wdistances != nullptr;
        }

        OSL_FORCEINLINE Masked<float[]> wdistances() const
        {
            return Masked<float[]>(m_wdistances, m_distances_array_length,
                                   m_mask);
        }

        OSL_FORCEINLINE bool distances_have_derivs() const
        {
            return m_distances_have_derivs;
        }

        OSL_FORCEINLINE MaskedDx<float[]> wdistancesDx() const
        {
            return MaskedDx<float[]>(m_wdistances, m_distances_array_length,
                                     m_mask);
        }

        OSL_FORCEINLINE MaskedDy<float[]> wdistancesDy() const
        {
            return MaskedDy<float[]>(m_wdistances, m_distances_array_length,
                                     m_mask);
        }
    };


    virtual void pointcloud_search(BatchedShaderGlobals* bsg,
                                   ustringhash filename, const void* wcenter,
                                   Wide<const float> wradius, int max_points,
                                   bool sort, PointCloudSearchResults& results);
    virtual bool is_overridden_pointcloud_search() const = 0;


    virtual Mask pointcloud_get(BatchedShaderGlobals* bsg, ustringhash filename,
                                Wide<const int[]> windices,
                                Wide<const int> wnum_points,
                                ustringhash attr_name, MaskedData wout_data);
    virtual bool is_overridden_pointcloud_get() const = 0;


    virtual Mask
    pointcloud_write(BatchedShaderGlobals* bsg, ustringhash filename,
                     Wide<const OSL::Vec3> wpos, int nattribs,
                     const ustring* attr_names, const TypeDesc* attr_types,
                     const void** pointers_to_wide_attr_value, Mask mask);
    virtual bool is_overridden_pointcloud_write() const = 0;

    /// Options for the trace call.
    using TraceOpt = RendererServices::TraceOpt;

    /// Immediately trace a ray from P in the direction R.  Return true
    /// if anything hit, otherwise false.
    virtual void trace(TraceOpt& options, BatchedShaderGlobals* bsg,
                       Masked<int> result, Wide<const Vec3> wP,
                       Wide<const Vec3> wdPdx, Wide<const Vec3> wdPdy,
                       Wide<const Vec3> wR, Wide<const Vec3> wdRdx,
                       Wide<const Vec3> wdRdy);

    virtual void getmessage(BatchedShaderGlobals* bsg, Masked<int> wresult,
                            ustringhash source, ustringhash name,
                            MaskedData wval);

    // pointcloud_search is T.B.D.

    /// Return a pointer to the texture system (if available).
    virtual TextureSystem* texturesys() const;

protected:
    TextureSystem* m_texturesys;  // A place to hold a TextureSystem
};


OSL_NAMESPACE_EXIT
