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
        = 0;

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
                            ustring from, Wide<const float> wtime)
        = 0;
    virtual Mask get_matrix(BatchedShaderGlobals* bsg, Masked<Matrix44> result,
                            Wide<const ustring> wfrom, Wide<const float> wtime);
    virtual bool is_overridden_get_matrix_WmWsWf() const = 0;


    /// Get the 4x4 matrix that transforms points from "common" space to
    /// the named 'to' coordinate system to at the given time.  The
    /// default implementation is to use get_matrix and invert it, but a
    /// particular renderer may have a better technique and overload the
    /// implementation.
    virtual Mask get_inverse_matrix(BatchedShaderGlobals* bsg,
                                    Masked<Matrix44> wresult, ustring to,
                                    Wide<const float> wtime);
    virtual bool is_overridden_get_inverse_matrix_WmsWf() const = 0;
    virtual Mask get_inverse_matrix(BatchedShaderGlobals* bsg,
                                    Masked<Matrix44> wresult,
                                    Wide<const ustring> wto,
                                    Wide<const float> wtime);
    virtual bool is_overridden_get_inverse_matrix_WmWsWf() const = 0;

    // non linear transformations are currently unsupported,
    // but could be added if needed

    /// Identify if the the named attribute from the renderer can be treated as uniform
    /// across all batches to the shader.  We assume all attributes are varying unless
    /// identified as uniform by the renderer.  NOTE:  To enable constant folding of
    // an attribute value, it must be uniform and retrievable with a NULL BatchedShaderGlobals
    virtual bool is_attribute_uniform(ustring object, ustring name) = 0;

    /// Get the named attribute from the renderer and if found then
    /// write it into 'val'.  Otherwise, return false.  If no object is
    /// specified (object == ustring()), then the renderer should search *first*
    /// for the attribute on the currently shaded object, and next, if
    /// unsuccessful, on the currently shaded "scene".
    virtual Mask get_attribute(BatchedShaderGlobals* bsg, ustring object,
                               ustring name, MaskedData wval)
        = 0;

    /// Similar to get_attribute();  this method will fetch the 'index'
    /// element of an attribute array.
    virtual Mask get_array_attribute(BatchedShaderGlobals* bsg, ustring object,
                                     ustring name, int index, MaskedData wval)
        = 0;

    virtual bool get_attribute_uniform(BatchedShaderGlobals* bsg,
                                       ustring object, ustring name,
                                       RefData val)
        = 0;

    /// Similar to get_attribute();  this method will fetch the 'index'
    /// element of an attribute array.
    virtual bool get_array_attribute_uniform(BatchedShaderGlobals* bsg,
                                             ustring object, ustring name,
                                             int index, RefData val)
        = 0;

    /// Get multiple named user-data from the current object and write them into
    /// 'val'. If derivatives is true, the derivatives should be written into val
    /// as well. It is assumed the results are varying and returns Mask
    // with its bit set to off if no user-data with the given name and type was
    /// found.
    virtual Mask get_userdata(ustring name, BatchedShaderGlobals* bsg,
                              MaskedData wval)
        = 0;

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
    texture(ustring filename, TextureSystem::TextureHandle* texture_handle,
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
    virtual Mask texture3d(ustring filename,
                           TextureSystem::TextureHandle* texture_handle,
                           TextureSystem::Perthread* texture_thread_info,
                           const BatchedTextureOptions& options,
                           BatchedShaderGlobals* bsg, Wide<const Vec3> wP,
                           Wide<const Vec3> wdPdx, Wide<const Vec3> wdPdy,
                           Wide<const Vec3> wdPdz,
                           BatchedTextureOutputs& outputs);
    virtual bool is_overridden_texture3d() const = 0;


    // Filtered environment lookup for a single point is T.B.D.

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

    // Should shader's fully avoid non-constant strings we could remove this
    // method which supports Wide<const ustring> wfilename.  Or replace
    // this virtual method with a library side loop that calls the uniform version
    virtual Mask get_texture_info(
        BatchedShaderGlobals* bsg, TexturePerthread* texture_thread_info,
        Wide<const ustring> wfilename,
        // We do not need to support texture handle for varying data.
        int subimage, ustring dataname, MaskedData val);

    virtual bool get_texture_info_uniform(
        BatchedShaderGlobals* bsg, TexturePerthread* texture_thread_info,
        ustring filename, TextureSystem::TextureHandle* texture_handle,
        int subimage, ustring dataname, RefData val);

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
                            ustring source, ustring name, MaskedData wval);

    // pointcloud_search is T.B.D.

    /// Return a pointer to the texture system (if available).
    virtual TextureSystem* texturesys() const;

protected:
    TextureSystem* m_texturesys;  // A place to hold a TextureSystem
};


OSL_NAMESPACE_EXIT
