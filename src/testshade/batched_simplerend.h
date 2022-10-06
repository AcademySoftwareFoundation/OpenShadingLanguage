// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/ustring.h>

#include <OSL/oslconfig.h>

#include <OSL/batched_rendererservices.h>

OSL_NAMESPACE_ENTER

class SimpleRenderer;

template<int WidthT>
class BatchedSimpleRenderer : public BatchedRendererServices<WidthT> {
public:
    explicit BatchedSimpleRenderer(SimpleRenderer& sr);
    virtual ~BatchedSimpleRenderer();

    OSL_USING_DATA_WIDTH(WidthT);
    using typename BatchedRendererServices<WidthT>::TraceOpt;

    Mask get_matrix(BatchedShaderGlobals* bsg, Masked<Matrix44> result,
                    Wide<const TransformationPtr> xform,
                    Wide<const float> time) override;
    bool is_overridden_get_inverse_matrix_WmWxWf() const override
    {
        return false;
    }

    Mask get_matrix(BatchedShaderGlobals* bsg, Masked<Matrix44> result,
                    ustringhash from, Wide<const float> time) override;
    Mask get_matrix(BatchedShaderGlobals* bsg, Masked<Matrix44> result,
                    Wide<const ustringhash> from,
                    Wide<const float> time) override;
    bool is_overridden_get_matrix_WmWsWf() const override { return true; }

private:
    template<typename RAccessorT>
    OSL_FORCEINLINE bool impl_get_inverse_matrix(RAccessorT& result,
                                                 ustringhash to) const;

public:
    Mask get_inverse_matrix(BatchedShaderGlobals* bsg, Masked<Matrix44> result,
                            ustringhash to, Wide<const float> time) override;
    bool is_overridden_get_inverse_matrix_WmsWf() const override
    {
        return true;
    }
    Mask get_inverse_matrix(BatchedShaderGlobals* bsg, Masked<Matrix44> result,
                            Wide<const ustringhash> to,
                            Wide<const float> time) override;
    bool is_overridden_get_inverse_matrix_WmWsWf() const override
    {
        return true;
    }


    bool is_attribute_uniform(ustring object, ustring name) override;

    Mask get_array_attribute(BatchedShaderGlobals* bsg, ustringhash object,
                             ustringhash name, int index,
                             MaskedData amd) override;

    Mask get_attribute(BatchedShaderGlobals* bsg, ustringhash object,
                       ustringhash name, MaskedData amd) override;


    bool get_array_attribute_uniform(BatchedShaderGlobals* bsg,
                                     ustringhash object, ustringhash name,
                                     int index, RefData val) override;

    bool get_attribute_uniform(BatchedShaderGlobals* bsg, ustringhash object,
                               ustringhash name, RefData val) override;

    Mask get_userdata(ustringhash name, BatchedShaderGlobals* bsg,
                      MaskedData val) override;

    bool is_overridden_texture() const override { return false; }
    bool is_overridden_texture3d() const override { return false; }
    bool is_overridden_environment() const override { return false; }
    bool is_overridden_pointcloud_search() const override { return false; }
    bool is_overridden_pointcloud_get() const override { return false; }
    bool is_overridden_pointcloud_write() const override { return false; }

    void trace(TraceOpt& options, BatchedShaderGlobals* bsg, Masked<int> result,
               Wide<const Vec3> P, Wide<const Vec3> dPdx, Wide<const Vec3> dPdy,
               Wide<const Vec3> R, Wide<const Vec3> dRdx,
               Wide<const Vec3> dRdy) override;

    void getmessage(BatchedShaderGlobals* bsg, Masked<int> result,
                    ustringhash source, ustringhash name,
                    MaskedData val) override;

private:
    SimpleRenderer& m_sr;
    std::unordered_set<ustring> m_uniform_objects;
    std::unordered_set<ustring> m_uniform_attributes;

    // Attribute and userdata retrieval -- for fast dispatch, use a hash
    // table to map attribute names to functions that retrieve them. We
    // imagine this to be fairly quick, but for a performance-critical
    // renderer, we would encourage benchmarking various methods and
    // alternate data structures.
    typedef bool (BatchedSimpleRenderer::*VaryingAttrGetter)(ustringhash object,
                                                             ustringhash name,
                                                             MaskedData val);
    typedef std::unordered_map<ustringhash, VaryingAttrGetter>
        VaryingAttrGetterMap;
    VaryingAttrGetterMap m_varying_attr_getters;

    typedef bool (BatchedSimpleRenderer::*UniformAttrGetter)(ustringhash object,
                                                             ustringhash name,
                                                             RefData val);
    typedef std::unordered_map<ustringhash, UniformAttrGetter>
        UniformAttrGetterMap;
    UniformAttrGetterMap m_uniform_attr_getters;

    // Attribute getters
    template<typename RefOrMaskedT>
    bool get_osl_version(ustringhash object, ustringhash name,
                         RefOrMaskedT data);
    template<typename RefOrMaskedT>
    bool get_camera_resolution(ustringhash object, ustringhash name,
                               RefOrMaskedT data);
    template<typename RefOrMaskedT>
    bool get_camera_projection(ustringhash object, ustringhash name,
                               RefOrMaskedT data);
    template<typename RefOrMaskedT>
    bool get_camera_fov(ustringhash object, ustringhash name,
                        RefOrMaskedT data);
    template<typename RefOrMaskedT>
    bool get_camera_pixelaspect(ustringhash object, ustringhash name,
                                RefOrMaskedT data);
    template<typename RefOrMaskedT>
    bool get_camera_clip(ustringhash object, ustringhash name,
                         RefOrMaskedT data);
    template<typename RefOrMaskedT>
    bool get_camera_clip_near(ustringhash object, ustringhash name,
                              RefOrMaskedT data);
    template<typename RefOrMaskedT>
    bool get_camera_clip_far(ustringhash object, ustringhash name,
                             RefOrMaskedT data);
    template<typename RefOrMaskedT>
    bool get_camera_shutter(ustringhash object, ustringhash name,
                            RefOrMaskedT data);
    template<typename RefOrMaskedT>
    bool get_camera_shutter_open(ustringhash object, ustringhash name,
                                 RefOrMaskedT data);
    template<typename RefOrMaskedT>
    bool get_camera_shutter_close(ustringhash object, ustringhash name,
                                  RefOrMaskedT data);
    template<typename RefOrMaskedT>
    bool get_camera_screen_window(ustringhash object, ustringhash name,
                                  RefOrMaskedT data);
};

OSL_NAMESPACE_EXIT
