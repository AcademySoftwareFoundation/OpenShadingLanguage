// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include "batched_simplerend.h"
#include "simplerend.h"

using namespace OSL;

OSL_NAMESPACE_ENTER

struct UniqueStringCache {
    UniqueStringCache()
        : camera("camera")
        , screen("screen")
        , NDC("NDC")
        , raster("raster")
        , perspective("perspective")
        , s("s")
        , t("t")
        , lookupTable("lookupTable")
        , blahblah("blahblah")
        , options("options")
        , global("global")
        , camera_resolution("camera:resolution")
        , camera_projection("camera:projection")
        , camera_pixelaspect("camera:pixelaspect")
        , camera_screen_window("camera:screen_window")
        , camera_fov("camera:fov")
        , camera_clip("camera:clip")
        , camera_clip_near("camera:clip_near")
        , camera_clip_far("camera:clip_far")
        , camera_shutter("camera:shutter")
        , camera_shutter_open("camera:shutter_open")
        , camera_shutter_close("camera:shutter_close")
    {
    }

    ustring camera;
    ustring screen;
    ustring NDC;
    ustring raster;
    ustring perspective;
    ustring s;
    ustring t;
    ustring lookupTable;
    ustring blahblah;
    ustring options;
    ustring global;
    ustring camera_resolution;
    ustring camera_projection;
    ustring camera_pixelaspect;
    ustring camera_screen_window;
    ustring camera_fov;
    ustring camera_clip;
    ustring camera_clip_near;
    ustring camera_clip_far;
    ustring camera_shutter;
    ustring camera_shutter_open;
    ustring camera_shutter_close;
};

// Lazily construct UniqueStringCache to avoid static construction issues of a global
const UniqueStringCache&
ucache()
{
    static UniqueStringCache unique_string_cache;
    return unique_string_cache;
}


template<int WidthT>
BatchedSimpleRenderer<WidthT>::BatchedSimpleRenderer(SimpleRenderer& sr)
    : BatchedRendererServices<WidthT>(sr.texturesys()), m_sr(sr)
{
    m_uniform_objects.insert(ucache().global);
    m_uniform_objects.insert(ucache().options);

    m_varying_attr_getters[ucache().camera_resolution]
        = &BatchedSimpleRenderer::get_camera_resolution<MaskedData>;
    m_varying_attr_getters[ucache().camera_projection]
        = &BatchedSimpleRenderer::get_camera_projection<MaskedData>;
    m_varying_attr_getters[ucache().camera_pixelaspect]
        = &BatchedSimpleRenderer::get_camera_pixelaspect<MaskedData>;
    m_varying_attr_getters[ucache().camera_screen_window]
        = &BatchedSimpleRenderer::get_camera_screen_window<MaskedData>;
    m_varying_attr_getters[ucache().camera_fov]
        = &BatchedSimpleRenderer::get_camera_fov<MaskedData>;
    m_varying_attr_getters[ucache().camera_clip]
        = &BatchedSimpleRenderer::get_camera_clip<MaskedData>;
    m_varying_attr_getters[ucache().camera_clip_near]
        = &BatchedSimpleRenderer::get_camera_clip_near<MaskedData>;
    m_varying_attr_getters[ucache().camera_clip_far]
        = &BatchedSimpleRenderer::get_camera_clip_far<MaskedData>;
    m_varying_attr_getters[ucache().camera_shutter]
        = &BatchedSimpleRenderer::get_camera_shutter<MaskedData>;
    m_varying_attr_getters[ucache().camera_shutter_open]
        = &BatchedSimpleRenderer::get_camera_shutter_open<MaskedData>;
    m_varying_attr_getters[ucache().camera_shutter_close]
        = &BatchedSimpleRenderer::get_camera_shutter_close<MaskedData>;

    m_uniform_attr_getters[ucache().camera_resolution]
        = &BatchedSimpleRenderer::get_camera_resolution<RefData>;
    m_uniform_attr_getters[ucache().camera_projection]
        = &BatchedSimpleRenderer::get_camera_projection<RefData>;
    m_uniform_attr_getters[ucache().camera_pixelaspect]
        = &BatchedSimpleRenderer::get_camera_pixelaspect<RefData>;
    m_uniform_attr_getters[ucache().camera_screen_window]
        = &BatchedSimpleRenderer::get_camera_screen_window<RefData>;
    m_uniform_attr_getters[ucache().camera_fov]
        = &BatchedSimpleRenderer::get_camera_fov<RefData>;
    m_uniform_attr_getters[ucache().camera_clip]
        = &BatchedSimpleRenderer::get_camera_clip<RefData>;
    m_uniform_attr_getters[ucache().camera_clip_near]
        = &BatchedSimpleRenderer::get_camera_clip_near<RefData>;
    m_uniform_attr_getters[ucache().camera_clip_far]
        = &BatchedSimpleRenderer::get_camera_clip_far<RefData>;
    m_uniform_attr_getters[ucache().camera_shutter]
        = &BatchedSimpleRenderer::get_camera_shutter<RefData>;
    m_uniform_attr_getters[ucache().camera_shutter_open]
        = &BatchedSimpleRenderer::get_camera_shutter_open<RefData>;
    m_uniform_attr_getters[ucache().camera_shutter_close]
        = &BatchedSimpleRenderer::get_camera_shutter_close<RefData>;
}

template<int WidthT> BatchedSimpleRenderer<WidthT>::~BatchedSimpleRenderer() {}


template<int WidthT>
typename BatchedSimpleRenderer<WidthT>::Mask
BatchedSimpleRenderer<WidthT>::get_matrix(BatchedShaderGlobals* bsg,
                                          Masked<Matrix44> result,
                                          Wide<const TransformationPtr> xform,
                                          Wide<const float> time)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    //result = *reinterpret_cast<const Matrix44*>(xform);

    OSL_FORCEINLINE_BLOCK
    {
        TransformationPtr uniform_xform = xform[0];

#if 0
        // In general, one can't assume that the transformation is uniform
        const Matrix44 uniformTransform = *reinterpret_cast<const Matrix44*>(uniform_xform);
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
        for(int lane=0; lane < WidthT; ++lane) {
            if (OSL_LIKELY((uniform_xform == xform[lane]))) {
                result[lane] = uniformTransform;
            } else {
                TransformationPtr lane_xform = xform[lane];
                const Matrix44 & transformFromShaderGlobals = *reinterpret_cast<const Matrix44*>(lane_xform);
                result[lane] = transformFromShaderGlobals;
            }
        }
#else
        // But this is "testshade" and we know we only have one object, so lets just
        // use that fact
        const Matrix44 uniformTransform = *reinterpret_cast<const Matrix44*>(
            uniform_xform);

        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
        for (int lane = 0; lane < WidthT; ++lane) {
#    if __INTEL_COMPILER >= 1900
            // Used load + blend + store instead of masked store to temporarily work around
            // an icc19u5 issue when automatic ISA dispatch is used causing scatters to be generated
            Matrix44 m = result[lane];
            if (result.mask()[lane]) {
                m = uniformTransform;
            }
            result[ActiveLane(lane)] = m;
#    else
            result[lane] = uniformTransform;
#    endif
        }
#endif
    }


    return Mask(true);
}

template<int WidthT>
typename BatchedSimpleRenderer<WidthT>::Mask
BatchedSimpleRenderer<WidthT>::get_matrix(BatchedShaderGlobals* /*bsg*/,
                                          Masked<Matrix44> wresult,
                                          ustring from,
                                          Wide<const float> /*wtime*/)
{
    auto found = m_sr.m_named_xforms.find(from);
    if (found != m_sr.m_named_xforms.end()) {
        const Matrix44& uniformTransform = *(found->second);

        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
        for (int lane = 0; lane < WidthT; ++lane) {
            wresult[lane] = uniformTransform;
        }

        return Mask(true);
    } else {
        return Mask(false);
    }
}

template<int WidthT>
typename BatchedSimpleRenderer<WidthT>::Mask
BatchedSimpleRenderer<WidthT>::get_matrix(BatchedShaderGlobals* /*bsg*/,
                                          Masked<Matrix44> wresult,
                                          Wide<const ustring> wfrom,
                                          Wide<const float> /*wtime*/)
{
    Mask succeeded(false);
    wresult.mask().template foreach<1 /*MinOccupancyT*/>(
        [&](ActiveLane lane) -> void {
            ustring from = wfrom[lane];
            auto found   = m_sr.m_named_xforms.find(from);
            if (found != m_sr.m_named_xforms.end()) {
                const Matrix44& transform = *(found->second);
                wresult[lane]             = transform;
                succeeded.set_on(lane);
            }
        });
    return succeeded;
}

template<int WidthT>
template<typename RAccessorT>
bool
BatchedSimpleRenderer<WidthT>::impl_get_inverse_matrix(RAccessorT& result,
                                                       ustring to) const
{
    if (to == ucache().camera || to == ucache().screen || to == ucache().NDC
        || to == ucache().raster) {
        Matrix44 M = m_sr.m_world_to_camera;
        if (to == ucache().screen || to == ucache().NDC
            || to == ucache().raster) {
            float depthrange = (double)m_sr.m_yon - (double)m_sr.m_hither;
            if (m_sr.m_projection == ucache().perspective) {
                float tanhalffov = tanf(0.5f * m_sr.m_fov * M_PI / 180.0);
                Matrix44 camera_to_screen(1 / tanhalffov, 0, 0, 0, 0,
                                          1 / tanhalffov, 0, 0, 0, 0,
                                          m_sr.m_yon / depthrange, 1, 0, 0,
                                          -m_sr.m_yon * m_sr.m_hither
                                              / depthrange,
                                          0);
                M = M * camera_to_screen;
            } else {
                Matrix44 camera_to_screen(1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                          1 / depthrange, 0, 0, 0,
                                          -m_sr.m_hither / depthrange, 1);
                M = M * camera_to_screen;
            }
            if (to == ucache().NDC || to == ucache().raster) {
                float screenleft = -1.0, screenwidth = 2.0;
                float screenbottom = -1.0, screenheight = 2.0;
                Matrix44 screen_to_ndc(1 / screenwidth, 0, 0, 0, 0,
                                       1 / screenheight, 0, 0, 0, 0, 1, 0,
                                       -screenleft / screenwidth,
                                       -screenbottom / screenheight, 0, 1);
                M = M * screen_to_ndc;
                if (to == ucache().raster) {
                    Matrix44 ndc_to_raster(m_sr.m_xres, 0, 0, 0, 0, m_sr.m_yres,
                                           0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
                    M = M * ndc_to_raster;
                }
            }
        }

        result = M;

        return true;
    }

    auto found = m_sr.m_named_xforms.find(to);
    if (found != m_sr.m_named_xforms.end()) {
        Matrix44 M = *(found->second);
        M.invert();

        result = M;
        return true;
    } else {
        return false;
    }
}

template<int WidthT>
typename BatchedSimpleRenderer<WidthT>::Mask
BatchedSimpleRenderer<WidthT>::get_inverse_matrix(BatchedShaderGlobals* bsg,
                                                  Masked<Matrix44> result,
                                                  ustring to,
                                                  Wide<const float> time)
{
    Matrix44 scalar_result;
    bool success = impl_get_inverse_matrix(scalar_result, to);

    if (success) {
        OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
        for (int i = 0; i < WidthT; ++i) {
            result[i] = scalar_result;
        }
        return result.mask();
    }
    return Mask(false);
}

template<int WidthT>
typename BatchedSimpleRenderer<WidthT>::Mask
BatchedSimpleRenderer<WidthT>::get_inverse_matrix(BatchedShaderGlobals* bsg,
                                                  Masked<Matrix44> wResult,
                                                  Wide<const ustring> wTo,
                                                  Wide<const float> wTime)

{
    Mask status(false);

    for (int i = 0; i < WidthT; ++i) {
        if (wResult.mask()[i]) {
            ustring to  = wTo[i];
            auto result = wResult[i];
            if (impl_get_inverse_matrix(result, to)) {
                status.set_on(i);
            }
        }
    }
    return status;
}


template<int WidthT>
bool
BatchedSimpleRenderer<WidthT>::is_attribute_uniform(ustring object,
                                                    ustring name)
{
    if (m_uniform_objects.find(object) != m_uniform_objects.end())
        return true;

    if ((!object.empty())
        && m_uniform_attributes.find(name) != m_uniform_attributes.end())
        return true;

    return false;
}

template<int WidthT>
void
BatchedSimpleRenderer<WidthT>::trace(
    TraceOpt& options, BatchedShaderGlobals* bsg, Masked<int> result,
    Wide<const Vec3> P, Wide<const Vec3> dPdx, Wide<const Vec3> dPdy,
    Wide<const Vec3> R, Wide<const Vec3> dRdx, Wide<const Vec3> dRdy)
{
    for (int lane = 0; lane < WidthT; ++lane) {
        Vec3 point_lane = P[lane];
        Vec3 dir_lane   = R[lane];

        float dot_val = point_lane.dot(dir_lane);

        if ((bsg->varying.u[lane] / dot_val) > 0.75) {
            result[lane] = 1;
        }

        else {
            result[lane] = 0;
        }
    }
}

template<int WidthT>
void
BatchedSimpleRenderer<WidthT>::getmessage(BatchedShaderGlobals* bsg,
                                          Masked<int> result, ustring source,
                                          ustring name, MaskedData val)
{
    for (int lane = 0; lane < WidthT; ++lane) {
        if (bsg->varying.u[lane] > 0.75) {
            result[lane] = 1;
        }

        else {
            result[lane] = 0;
        }
    }
}



template<int WidthT>
typename BatchedSimpleRenderer<WidthT>::Mask
BatchedSimpleRenderer<WidthT>::get_array_attribute(BatchedShaderGlobals* bsg,
                                                   ustring object, ustring name,
                                                   int index, MaskedData val)
{
    // Normally the common_get_attribute would be for is_attribute_uniform() == true only
    // However a name that has a uniform answer could be part of a varying name inside
    // the shader.  So we still have to check for wide versions of our getter
    // functions
    auto g = m_varying_attr_getters.find(name);
    if (g != m_varying_attr_getters.end()) {
        auto getter  = g->second;
        bool success = (this->*(getter))(object, name, val);
        return val.mask() & Mask(success);
    }



    if (object == nullptr && name == ucache().blahblah) {
        if (Masked<float>::is(val)) {
            Masked<float> out(val);
            for (int i = 0; i < WidthT; ++i) {
#if 0
                // Example of how to test the lane of the proxy
                // if you want to skip expensive code on the right
                // hand side of the assignment, such as a function call
                if(out[i].is_on()) {
                    out[i] = 1.0f -  Vec3(bsg->varying.P[i]).x;
                }
#else
                // Masking is silently handled by the assignment operator
                // of the proxy out[i]
                out[i] = 1.0f - Vec3(bsg->varying.P[i]).x;
#endif
            }
            return val.mask();
        } else if (Masked<Vec3>::is(val)) {
            Masked<Vec3> out(val);
            for (int i = 0; i < WidthT; ++i) {
                out[i] = Vec3(1.0f) - bsg->varying.P[i];
            }
            return val.mask();
        }
    }

    if (object == nullptr && name == ucache().lookupTable) {
        if (Masked<float[]>::is(val)) {
            Masked<float[]> out(val);
            for (int lane_index = 0; lane_index < WidthT; ++lane_index) {
                auto lut = out[lane_index];
                for (int i = 0; i < lut.length(); ++i) {
                    lut[i] = 1.0 - float(i) / float(lut.length());
                }
            }
            return Mask(true);
        }
    }


    if (object == nullptr && name == "not_a_color") {
        if (Masked<float[3]>::is(val)) {
            Masked<float[3]> out(val);

            float valArray[3] = { 0.0f, 0.5f, 1.0f };

            for (int i = 0; i < WidthT; ++i) {
                // NOTE: out[i] proxy handles pulling values out of the
                // the local array and distributing to the
                // correct wide arrays, including testing
                // the mask during assignment
                out[i] = valArray;
            }


#if 0  // Show that array elements can be extracted from the proxy out[i]
            for(int i=0; i < WidthT; ++i) {
                if(out[i].is_on()) {
                    float valArray[3];
                    // Illegal to return an array by value
                    //     valArray = out[i];
                    // but we can use the [arrayindex] operator on
                    // the proxy out[i] to get access to underlying data
                    valArray[0] = out[i][0];
                    valArray[1] = out[i][1];
                    valArray[2] = out[i][2];

                    std::cout << "valArray[0] = " << valArray[0] << std::endl;
                    std::cout << "valArray[1] = " << valArray[1] << std::endl;
                    std::cout << "valArray[2] = " << valArray[2] << std::endl;
                }
            }
#endif
            return val.mask();
        }
    }

    // If no named attribute was found, allow userdata to bind to the
    // attribute request.
    if (object.empty() && index == -1)
        return get_userdata(name, bsg, val);

    return Mask(false);
}

template<int WidthT>
typename BatchedSimpleRenderer<WidthT>::Mask
BatchedSimpleRenderer<WidthT>::get_attribute(BatchedShaderGlobals* bsg,
                                             ustring object, ustring name,
                                             MaskedData val)
{
    return get_array_attribute(bsg, object, name, -1, val);
}



template<int WidthT>
bool
BatchedSimpleRenderer<WidthT>::get_array_attribute_uniform(
    BatchedShaderGlobals* bsg, ustring object, ustring name, int index,
    RefData val)
{
    ASSERT(!name.empty());

    auto g = m_uniform_attr_getters.find(name);
    if (g != m_uniform_attr_getters.end()) {
        auto getter = g->second;
        return (this->*(getter))(object, name, val);
    }

    // In order to test getattribute(), respond positively to
    // "options"/"blahblah"
    if (object == ucache().options && name == ucache().blahblah
        && Ref<float>::is(val)) {
        (Ref<float>(val)) = 3.14159f;
        return true;
    }


    if (/*object == nullptr &&*/ name == ucache().lookupTable) {
#if 0
        // Old way of checking typedesc and arraylength
        if (val.type().is_array() && val.type().basetype == TypeDesc::FLOAT && val.type().aggregate == TypeDesc::SCALAR)
        {
            float * lt = reinterpret_cast<float *>(val.ptr());
            for(int i=0; i < val.type().arraylen; ++i)
            {
                lt[i] = 1.0 - float(i)/float(val.type().arraylen);
            }
            return true;
        }
#endif
        // New way of checking for array of a given type
        if (Ref<float[]>::is(val)) {
            Ref<float[]> lut(val);
            for (int i = 0; i < lut.length(); ++i) {
                lut[i] = 1.0 - float(i) / float(lut.length());
            }
            return true;
        }
    }
    // NOTE: we do not bother calling through to get_userdata
    // The only way to get inside this call was to of had the
    // is_attribute_uniform return true.  It is a logic bug on the renderer
    // if it claims user data is uniform.
    // TODO: validate the above comment
    return false;
}

template<int WidthT>
bool
BatchedSimpleRenderer<WidthT>::get_attribute_uniform(BatchedShaderGlobals* bsg,
                                                     ustring object,
                                                     ustring name, RefData val)
{
    return get_array_attribute_uniform(bsg, object, name, -1, val);
}

template<int WidthT>
typename BatchedSimpleRenderer<WidthT>::Mask
BatchedSimpleRenderer<WidthT>::get_userdata(ustring name,
                                            BatchedShaderGlobals* bsg,
                                            MaskedData val)
{
    // Just to illustrate how this works, respect s and t userdata, filled
    // in with the uv coordinates.  In a real renderer, it would probably
    // look up something specific to the primitive, rather than have hard-
    // coded names.

    // For testing of interactions with default values
    // may not provide data for all lanes
    if (name == ucache().s && Masked<float>::is(val)) {
        Masked<float> out(val);
        for (int i = 0; i < WidthT; ++i) {
            // NOTE: assigning to out[i] will mask by itself
            // this check is just to show how you could do it if you
            // wanted to skip executing right hand side of assignment
            if (out[i].is_on()) {
                out[i] = bsg->varying.u[i];
            }
        }
        if (val.has_derivs()) {
            MaskedDx<float> out_dx(val);
            for (int i = 0; i < WidthT; ++i) {
                out_dx[i] = bsg->varying.dudx[i];
            }

            MaskedDy<float> out_dy(val);
            for (int i = 0; i < WidthT; ++i) {
                out_dy[i] = bsg->varying.dudy[i];
            }
        }

        return val.mask();
    }
    if (name == ucache().t && Masked<float>::is(val)) {
        Masked<float> out(val);
        for (int i = 0; i < WidthT; ++i) {
            out[i] = bsg->varying.v[i];
        }
        if (val.has_derivs()) {
            MaskedDx<float> out_dx(val);
            for (int i = 0; i < WidthT; ++i) {
                out_dx[i] = bsg->varying.dvdx[i];
            }

            MaskedDy<float> out_dy(val);
            for (int i = 0; i < WidthT; ++i) {
                out_dy[i] = bsg->varying.dvdy[i];
            }
        }

        return val.mask();
    }

#ifdef __OSL_DEBUG_MISSING_USER_DATA
    static std::unordered_set<ustring> missingUserData;
    if (missingUserData.find(name) == missingUserData.end()) {
        std::cout << "Missing user data for " << name << std::endl;
        missingUserData.insert(name);
    }
#endif

    return Mask(false);
}

namespace {  // anonymous
template<typename DataT>
OSL_FORCEINLINE bool
assign_and_zero_derivs(RefData data, const DataT& val)
{
    if (Ref<DataT>::is(data)) {
        (Ref<DataT>(data)) = val;
        if (data.has_derivs()) {
            // Could have passed in explicit dx & dy,
            // but they were all 0's, so
            // Zero initialize derivatives
            DataT zero           = {};
            (RefDx<DataT>(data)) = zero;
            (RefDy<DataT>(data)) = zero;
        }
        return true;
    }
    return false;
}

template<typename DataT, int WidthT>
OSL_FORCEINLINE bool
assign_and_zero_derivs(MaskedData<WidthT> data, const DataT& val)
{
    if (Masked<DataT, WidthT>::is(data)) {
        assign_all(Masked<DataT, WidthT>(data), val);
        if (data.has_derivs()) {
            // Could have passed in explicit dx & dy,
            // but they were all 0's, so
            // Zero initialize derivatives
            DataT zero = {};
            assign_all(MaskedDx<DataT, WidthT>(data), zero);
            assign_all(MaskedDy<DataT, WidthT>(data), zero);
        }
        return true;
    }
    return false;
}
}  // anonymous namespace

template<int WidthT>
template<typename RefOrMaskedT>
bool
BatchedSimpleRenderer<WidthT>::get_osl_version(ustring /*object*/,
                                               ustring /*name*/,
                                               RefOrMaskedT data)
{
    return assign_and_zero_derivs(data, int(OSL_VERSION));
}

template<int WidthT>
template<typename RefOrMaskedT>
bool
BatchedSimpleRenderer<WidthT>::get_camera_resolution(ustring /*object*/,
                                                     ustring /*name*/,
                                                     RefOrMaskedT data)
{
    int res[2] = { m_sr.m_xres, m_sr.m_yres };
    return assign_and_zero_derivs(data, res);
}

template<int WidthT>
template<typename RefOrMaskedT>
bool
BatchedSimpleRenderer<WidthT>::get_camera_projection(ustring /*object*/,
                                                     ustring /*name*/,
                                                     RefOrMaskedT data)
{
    return assign_and_zero_derivs(data, m_sr.m_projection);
}


template<int WidthT>
template<typename RefOrMaskedT>
bool
BatchedSimpleRenderer<WidthT>::get_camera_fov(ustring /*object*/,
                                              ustring /*name*/,
                                              RefOrMaskedT data)
{
    // N.B. in a real renderer, this may be time-dependent
    return assign_and_zero_derivs(data, m_sr.m_fov);
}


template<int WidthT>
template<typename RefOrMaskedT>
bool
BatchedSimpleRenderer<WidthT>::get_camera_pixelaspect(ustring /*object*/,
                                                      ustring /*name*/,
                                                      RefOrMaskedT data)
{
    return assign_and_zero_derivs(data, m_sr.m_pixelaspect);
}


template<int WidthT>
template<typename RefOrMaskedT>
bool
BatchedSimpleRenderer<WidthT>::get_camera_clip(ustring /*object*/,
                                               ustring /*name*/,
                                               RefOrMaskedT data)
{
    float clip[2] = { m_sr.m_hither, m_sr.m_yon };
    return assign_and_zero_derivs(data, clip);
}


template<int WidthT>
template<typename RefOrMaskedT>
bool
BatchedSimpleRenderer<WidthT>::get_camera_clip_near(ustring /*object*/,
                                                    ustring /*name*/,
                                                    RefOrMaskedT data)
{
    return assign_and_zero_derivs(data, m_sr.m_hither);
}


template<int WidthT>
template<typename RefOrMaskedT>
bool
BatchedSimpleRenderer<WidthT>::get_camera_clip_far(ustring /*object*/,
                                                   ustring /*name*/,
                                                   RefOrMaskedT data)
{
    return assign_and_zero_derivs(data, m_sr.m_yon);
}



template<int WidthT>
template<typename RefOrMaskedT>
bool
BatchedSimpleRenderer<WidthT>::get_camera_shutter(ustring /*object*/,
                                                  ustring /*name*/,
                                                  RefOrMaskedT data)
{
    return assign_and_zero_derivs(data, m_sr.m_shutter);
}


template<int WidthT>
template<typename RefOrMaskedT>
bool
BatchedSimpleRenderer<WidthT>::get_camera_shutter_open(ustring /*object*/,
                                                       ustring /*name*/,
                                                       RefOrMaskedT data)
{
    return assign_and_zero_derivs(data, m_sr.m_shutter[0]);
}


template<int WidthT>
template<typename RefOrMaskedT>
bool
BatchedSimpleRenderer<WidthT>::get_camera_shutter_close(ustring /*object*/,
                                                        ustring /*name*/,
                                                        RefOrMaskedT data)
{
    return assign_and_zero_derivs(data, m_sr.m_shutter[1]);
}


template<int WidthT>
template<typename RefOrMaskedT>
bool
BatchedSimpleRenderer<WidthT>::get_camera_screen_window(ustring /*object*/,
                                                        ustring /*name*/,
                                                        RefOrMaskedT data)
{
    return assign_and_zero_derivs(data, m_sr.m_screen_window);
}


// Explicitly instantiate BatchedSimpleRenderer template
template class BatchedSimpleRenderer<16>;
template class BatchedSimpleRenderer<8>;


OSL_NAMESPACE_EXIT
