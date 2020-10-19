// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#pragma once

#include <map>
#include <memory>
#include <unordered_map>

#include <OpenImageIO/ustring.h>
#include <OpenImageIO/imagebuf.h>

#include <OSL/oslexec.h>
OSL_NAMESPACE_ENTER


void register_closures(OSL::ShadingSystem* shadingsys);

class SimpleRenderer;

template<int WidthT>
class BatchedSimpleRenderer : private BatchedRendererServicesBase<WidthT>
{
    // Because conversion to inaccessible base class is not allowed,
    // SimpleRenderer must be a friend to downcast BatchedSimpleRenderer
    // through the private/protected inheritance chain to BatchedRendererServices.
    friend class SimpleRenderer;
public:
    BatchedSimpleRenderer(SimpleRenderer &sr);
    virtual ~BatchedSimpleRenderer();

    // Because the base class is dependent on this class' template parameter
    // and C++ Standard 14.6.2/3 states "the base class scope is not examined
    // during unqualified name lookup" we must manually forward types/templates
    // from the Base (or use fully qualified names).  We choose to forward
    // with the 'using' declaration.
    OSL_USING_DATA_WIDTH(WidthT);
    using typename BatchedRendererServices<WidthT>::TraceOpt;

    Mask get_matrix (BatchedShaderGlobals *bsg,
                             Masked<Matrix44> result,
                             Wide<const TransformationPtr> xform,
                             Wide<const float> time) override;
    Mask get_matrix (BatchedShaderGlobals *bsg,
                             Masked<Matrix44> result,
                             ustring from,
                             Wide<const float> time) override;
    Mask get_matrix (BatchedShaderGlobals *bsg,
                             Masked<Matrix44> result,
                             Wide<const ustring> from,
                             Wide<const float> time) override;

    void trace (TraceOpt &options,  BatchedShaderGlobals *bsg, Masked<int> result,
            Wide<const Vec3> P, Wide<const Vec3> dPdx,
            Wide<const Vec3> dPdy, Wide<const Vec3> R,
            Wide<const Vec3> dRdx, Wide<const Vec3> dRdy) override;

    void getmessage (BatchedShaderGlobals *bsg, Masked<int> result,
                                 ustring source, ustring name, MaskedData val) override;
private:
    template<typename RAccessorT>
    OSL_FORCEINLINE bool impl_get_inverse_matrix (
        RAccessorT & result,
        ustring to) const;

public:
    Mask get_inverse_matrix (BatchedShaderGlobals *bsg, Masked<Matrix44> result,
                             ustring to, Wide<const float> time) override;
    Mask get_inverse_matrix (BatchedShaderGlobals *bsg, Masked<Matrix44> result,
                             Wide<const ustring> to, Wide<const float> time) override;


    bool is_attribute_uniform(ustring object, ustring name) override;

    Mask get_array_attribute (BatchedShaderGlobals *bsg,
                                      ustring object, ustring name,
                                      int index, MaskedData amd) override;

    Mask get_attribute (BatchedShaderGlobals *bsg, ustring object,
                                ustring name, MaskedData amd) override;


    bool get_array_attribute_uniform (BatchedShaderGlobals *bsg,
                                      ustring object, ustring name,
                                      int index, RefData val) override;

    bool get_attribute_uniform (BatchedShaderGlobals *bsg, ustring object,
                                ustring name, RefData val) override;

#ifdef OSL_EXPERIMENTAL_BIND_USER_DATA_WITH_LAYERNAME
    Mask get_userdata (ustring name, ustring layername,
                               BatchedShaderGlobals *bsg, MaskedData val) override;
#else
    Mask get_userdata (ustring name,
                               BatchedShaderGlobals *bsg, MaskedData val) override;
#endif

#if __OSL_MOCK_OVERRIDE_TEXTURE // Just to test if override detection works
    virtual Mask texture(ustring filename, TextureSystem::TextureHandle  *texture_handle,
            TextureSystem::Perthread *texture_thread_info,
                                  const BatchedTextureOptions &options, BatchedShaderGlobals *bsg,
                                  Wide<const float> s, Wide<const float> t,
                                  Wide<const float> dsdx, Wide<const float> dtdx,
                                  Wide<const float> dsdy, Wide<const float> dtdy,
                                  BatchedTextureOutputs& outputs) override { return Mask(true); }
#endif

#if __OSL_MOCK_OVERRIDE_TEXTURE_3D // Just to test if override detection works
    virtual Mask texture3d (ustring filename, TextureSystem::TextureHandle *texture_handle,
                            TextureSystem::Perthread *texture_thread_info,
                            const BatchedTextureOptions &options, BatchedShaderGlobals *bsg,
                            Wide<const Vec3> P, Wide<const Vec3> dPdx, Wide<const Vec3> dPdy,
                            Wide<const Vec3> dPdz, BatchedTextureOutputs& outputs) override { return Mask(true); }
#endif

private:
    SimpleRenderer &m_sr;
    std::unordered_set<ustring, ustringHash> m_uniform_objects;
    std::unordered_set<ustring, ustringHash> m_uniform_attributes;

    // Attribute and userdata retrieval -- for fast dispatch, use a hash
    // table to map attribute names to functions that retrieve them. We
    // imagine this to be fairly quick, but for a performance-critical
    // renderer, we would encourage benchmarking various methods and
    // alternate data structures.
    typedef bool (SimpleRenderer::*WideAttrGetter)(ustring object, ustring name, MaskedData val);
    typedef std::unordered_map<ustring, WideAttrGetter, ustringHash> WideAttrGetterMap;
    WideAttrGetterMap m_wide_attr_getters;
};

OSL_NAMESPACE_EXIT
