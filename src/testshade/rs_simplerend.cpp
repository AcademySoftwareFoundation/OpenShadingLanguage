// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifndef OSL_HOST_RS_BITCODE
#    error OSL_HOST_RS_BITCODE must be defined by your build system.
#endif

#include <OSL/fmt_util.h>
#include <OSL/journal.h>
#include <OSL/rendererservices.h>
#include <OSL/rs_free_function.h>

#include "render_state.h"

// Extern all global string variables used by free function renderer services.
// NOTE:  C linkage with a "RS_" prefix is used to allow for unmangled
// non-colliding global symbol names, so its easier to pass them to
// OSL::register_JIT_Global(name, addr) for host execution
// NOTE:  the STRING_PARAMS macro adapts to OSL_HOST_RS_BITCODE
// to utilize the RS_ prefix.  RS_ prefixed versions of all OSL::Strings
// instances have been created by rs_free_function.h, so the same STRING_PARAMS
// macro can be used for renderer service or OSL strings.
#define RS_STRDECL(str, var_name) extern "C" OSL::ustring RS_##var_name;
#include "rs_strdecls.h"
#undef RS_STRDECL


// Keep free functions in sync with virtual function based SimpleRenderer.

OSL_RSOP bool
rs_get_matrix_xform_time(OSL::OpaqueExecContextPtr /*ec*/,
                         OSL::Matrix44& result, OSL::TransformationPtr xform,
                         float /*time*/)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    auto ptr = reinterpret_cast<OSL::Matrix44 const*>(xform);
    result   = *ptr;
    return true;
}

OSL_RSOP bool
rs_get_inverse_matrix_xform_time(OSL::OpaqueExecContextPtr ec,
                                 OSL::Matrix44& result,
                                 OSL::TransformationPtr xform, float time)
{
    bool ok = rs_get_matrix_xform_time(ec, result, xform, time);
    if (ok) {
        result.invert();
    }
    return ok;
}

OSL_RSOP bool
rs_get_matrix_space_time(OSL::OpaqueExecContextPtr /*ec*/,
                         OSL::Matrix44& result, OSL::StringParam from,
                         float /*time*/)
{
    if (from == STRING_PARAMS(myspace)) {
        OSL::Matrix44 Mmyspace;
        Mmyspace.scale(OSL::Vec3(1.0, 2.0, 1.0));
        result = Mmyspace;
        return true;
    } else {
        return false;
    }
}

OSL_RSOP bool
rs_get_inverse_matrix_space_time(OSL::OpaqueExecContextPtr ec,
                                 OSL::Matrix44& result, OSL::StringParam to,
                                 float time)
{
    using OSL::Matrix44;

    auto rs = OSL::get_rs<RenderState>(ec);
    if (to == STRING_PARAMS(camera) || to == STRING_PARAMS(screen)
        || to == STRING_PARAMS(NDC) || to == STRING_PARAMS(raster)) {
        Matrix44 M { rs->world_to_camera };

        if (to == STRING_PARAMS(screen) || to == STRING_PARAMS(NDC)
            || to == STRING_PARAMS(raster)) {
            float depthrange = (double)rs->yon - (double)rs->hither;
            const auto& proj = rs->projection;

            if (proj == STRING_PARAMS(perspective)) {
                float tanhalffov = OIIO::fast_tan(0.5f * rs->fov * M_PI
                                                  / 180.0);
                // clang-format off
                Matrix44 camera_to_screen (1/tanhalffov, 0, 0, 0,
                                           0, 1/tanhalffov, 0, 0,
                                           0, 0, rs->yon/depthrange, 1,
                                           0, 0, -(rs->yon*rs->hither)/depthrange, 0);
                // clang-format on
                M = M * camera_to_screen;
            } else {
                // clang-format off
                Matrix44 camera_to_screen (1, 0, 0, 0,
                                           0, 1, 0, 0,
                                           0, 0, 1/depthrange, 0,
                                           0, 0, -(rs->hither)/depthrange, 1);
                // clang-format on
                M = M * camera_to_screen;
            }
            if (to == STRING_PARAMS(NDC) || to == STRING_PARAMS(raster)) {
                float screenleft = -1.0, screenwidth = 2.0;
                float screenbottom = -1.0, screenheight = 2.0;
                // clang-format off
                Matrix44 screen_to_ndc (1/screenwidth, 0, 0, 0,
                                        0, 1/screenheight, 0, 0,
                                        0, 0, 1, 0,
                                        -screenleft/screenwidth, -screenbottom/screenheight, 0, 1);
                // clang-format on
                M = M * screen_to_ndc;
                if (to == STRING_PARAMS(raster)) {
                    // clang-format off
                    Matrix44 ndc_to_raster (rs->xres, 0, 0, 0,
                                            0, rs->yres, 0, 0,
                                            0, 0, 1, 0,
                                            0, 0, 0, 1);
                    M = M * ndc_to_raster;
                    // clang-format on
                }
            }
        }

        result = M;
        return true;
    } else {
        bool ok = rs_get_matrix_space_time(ec, result, to, time);
        if (ok) {
            result.invert();
        }

        return ok;
    }
}

OSL_RSOP bool
rs_get_matrix_xform(OSL::OpaqueExecContextPtr /*ec*/, OSL::Matrix44& result,
                    OSL::TransformationPtr xform)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    auto ptr = reinterpret_cast<OSL::Matrix44 const*>(xform);
    result   = *ptr;
    return true;
}

OSL_RSOP bool
rs_get_inverse_matrix_xform(OSL::OpaqueExecContextPtr ec, OSL::Matrix44& result,
                            OSL::TransformationPtr xform)
{
    bool ok = rs_get_matrix_xform(ec, result, xform);
    if (ok) {
        result.invert();
    }

    return ok;
}

OSL_RSOP bool
rs_get_matrix_space(OSL::OpaqueExecContextPtr /*ec*/, OSL::Matrix44& /*result*/,
                    OSL::StringParam from)
{
    if (from == STRING_PARAMS(myspace)) {
        return true;
    } else {
        return false;
    }
}

OSL_RSOP bool
rs_get_inverse_matrix_space(OSL::OpaqueExecContextPtr ec, OSL::Matrix44& result,
                            OSL::StringParam to)
{
    bool ok = rs_get_matrix_space(ec, result, to);
    if (ok) {
        result.invert();
    }
    return ok;
}

OSL_RSOP bool
rs_transform_points(OSL::OpaqueExecContextPtr /*ec*/, OSL::StringParam /*from*/,
                    OSL::StringParam /*to*/, float /*time*/,
                    const OSL::Vec3* /*Pin*/, OSL::Vec3* /*Pout*/,
                    int /*npoints*/, OSL::TypeDesc::VECSEMANTICS /*vectype*/)
{
    return false;
}

OSL_RSOP bool
rs_get_attribute_constant_string(OSL::StringParam value, void* result)
{
    reinterpret_cast<OSL::StringParam*>(result)[0] = value;
    return true;
}

OSL_RSOP bool
rs_get_attribute_constant_int(int value, void* result)
{
    reinterpret_cast<int*>(result)[0] = value;
    return true;
}

OSL_RSOP bool
rs_get_attribute_constant_int2(int value1, int value2, void* result)
{
    reinterpret_cast<int*>(result)[0] = value1;
    reinterpret_cast<int*>(result)[1] = value2;
    return true;
}

OSL_RSOP bool
rs_get_attribute_constant_int3(int value1, int value2, int value3, void* result)
{
    reinterpret_cast<int*>(result)[0] = value1;
    reinterpret_cast<int*>(result)[1] = value2;
    reinterpret_cast<int*>(result)[2] = value3;
    return true;
}

OSL_RSOP bool
rs_get_attribute_constant_int4(int value1, int value2, int value3, int value4,
                               void* result)
{
    reinterpret_cast<int*>(result)[0] = value1;
    reinterpret_cast<int*>(result)[1] = value2;
    reinterpret_cast<int*>(result)[2] = value3;
    reinterpret_cast<int*>(result)[3] = value4;
    return true;
}

OSL_RSOP bool
rs_get_attribute_constant_float(float value, bool derivatives, void* result)
{
    reinterpret_cast<float*>(result)[0] = value;
    if (derivatives) {
        reinterpret_cast<float*>(result)[1] = 0.f;
        reinterpret_cast<float*>(result)[2] = 0.f;
    }
    return true;
}

OSL_RSOP bool
rs_get_attribute_constant_float2(float value1, float value2, bool derivatives,
                                 void* result)
{
    reinterpret_cast<float*>(result)[0] = value1;
    reinterpret_cast<float*>(result)[1] = value2;
    if (derivatives) {
        reinterpret_cast<float*>(result)[2] = 0.f;
        reinterpret_cast<float*>(result)[3] = 0.f;
        reinterpret_cast<float*>(result)[4] = 0.f;
        reinterpret_cast<float*>(result)[5] = 0.f;
    }
    return true;
}

OSL_RSOP bool
rs_get_attribute_constant_float3(float value1, float value2, float value3,
                                 bool derivatives, void* result)
{
    reinterpret_cast<float*>(result)[0] = value1;
    reinterpret_cast<float*>(result)[1] = value2;
    reinterpret_cast<float*>(result)[2] = value3;
    if (derivatives) {
        reinterpret_cast<float*>(result)[3] = 0.f;
        reinterpret_cast<float*>(result)[4] = 0.f;
        reinterpret_cast<float*>(result)[5] = 0.f;
        reinterpret_cast<float*>(result)[6] = 0.f;
        reinterpret_cast<float*>(result)[7] = 0.f;
        reinterpret_cast<float*>(result)[8] = 0.f;
    }
    return true;
}

OSL_RSOP bool
rs_get_attribute_constant_float4(float value1, float value2, float value3,
                                 float value4, bool derivatives, void* result)
{
    reinterpret_cast<float*>(result)[0] = value1;
    reinterpret_cast<float*>(result)[1] = value2;
    reinterpret_cast<float*>(result)[2] = value3;
    reinterpret_cast<float*>(result)[3] = value4;
    if (derivatives) {
        reinterpret_cast<float*>(result)[4]  = 0.f;
        reinterpret_cast<float*>(result)[5]  = 0.f;
        reinterpret_cast<float*>(result)[6]  = 0.f;
        reinterpret_cast<float*>(result)[7]  = 0.f;
        reinterpret_cast<float*>(result)[8]  = 0.f;
        reinterpret_cast<float*>(result)[9]  = 0.f;
        reinterpret_cast<float*>(result)[10] = 0.f;
        reinterpret_cast<float*>(result)[11] = 0.f;
    }
    return true;
}

OSL_RSOP bool
rs_get_shade_index(void* _sg, void* result)
{
    reinterpret_cast<int*>(result)[0] = OSL::get_shade_index(_sg);
    return true;
}

OSL_RSOP bool
rs_get_attribute(void* _sg, const char* _object, const char* _name,
                 OSL::TypeDesc_pod _type, bool derivatives, int index,
                 void* result)
{
    OSL::ShaderGlobals* sg        = reinterpret_cast<OSL::ShaderGlobals*>(_sg);
    const OSL::StringParam object = OSL::bitcast<OSL::ustringrep>(_object);
    const OSL::StringParam name   = OSL::bitcast<OSL::ustringrep>(_name);
    const OSL::TypeDesc type      = OSL::TypeDesc_from(_type);

    const RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);

    // The many branches in the code below handle the case where we don't know
    // the attribute name at compile time. In the case it is known, dead-code
    // elimination should optimize this to only the relevant branch.
    if (name == STRING_PARAMS(osl_version) && type == OSL::TypeInt)
        return rs_get_attribute_constant_int(OSL_VERSION, result);
    if (name == STRING_PARAMS(camera_resolution)
        && type == OSL::TypeDesc(OSL::TypeDesc::INT, 2))
        return rs_get_attribute_constant_int2(rs->xres, rs->yres, result);
    if (name == STRING_PARAMS(camera_projection) && type == OSL::TypeString)
        return rs_get_attribute_constant_string(rs->projection, result);
    if (name == STRING_PARAMS(camera_pixelaspect) && type == OSL::TypeFloat)
        return rs_get_attribute_constant_float(rs->pixelaspect, derivatives,
                                               result);
    if (name == STRING_PARAMS(camera_screen_window)
        && type == OSL::TypeDesc(OSL::TypeDesc::FLOAT, 4))
        return rs_get_attribute_constant_float4(rs->screen_window[0],
                                                rs->screen_window[1],
                                                rs->screen_window[2],
                                                rs->screen_window[3],
                                                derivatives, result);
    if (name == STRING_PARAMS(camera_fov) && type == OSL::TypeFloat)
        return rs_get_attribute_constant_float(rs->fov, derivatives, result);
    if (name == STRING_PARAMS(camera_clip)
        && type == OSL::TypeDesc(OSL::TypeDesc::FLOAT, 2))
        return rs_get_attribute_constant_float2(rs->hither, rs->yon,
                                                derivatives, result);
    if (name == STRING_PARAMS(camera_clip_near) && type == OSL::TypeFloat)
        return rs_get_attribute_constant_float(rs->hither, derivatives, result);
    if (name == STRING_PARAMS(camera_clip_far) && type == OSL::TypeFloat)
        return rs_get_attribute_constant_float(rs->yon, derivatives, result);
    if (name == STRING_PARAMS(camera_shutter)
        && type == OSL::TypeDesc(OSL::TypeDesc::FLOAT, 2))
        return rs_get_attribute_constant_float2(rs->shutter[0], rs->shutter[1],
                                                derivatives, result);
    if (name == STRING_PARAMS(camera_shutter_open) && type == OSL::TypeFloat)
        return rs_get_attribute_constant_float(rs->shutter[0], derivatives,
                                               result);
    if (name == STRING_PARAMS(camera_shutter_close) && type == OSL::TypeFloat)
        return rs_get_attribute_constant_float(rs->shutter[1], derivatives,
                                               result);

    if (name == STRING_PARAMS(shading_index) && type == OSL::TypeInt)
        return rs_get_shade_index(sg, result);

    if (object == STRING_PARAMS(options) && name == STRING_PARAMS(blahblah)
        && type == OSL::TypeFloat)
        return rs_get_attribute_constant_float(3.14159f, derivatives, result);

    if (object.empty())
        return sg->renderer->get_userdata(derivatives, name, type, sg, result);

    return false;
}

OSL_RSOP void
rs_errorfmt(OSL::OpaqueExecContextPtr ec, OSL::ustringhash fmt_specification,
            int32_t arg_count, const OSL::EncodedType* argTypes,
            uint32_t argValuesSize, uint8_t* argValues)
{
    auto rs = OSL::get_rs<RenderState>(ec);

    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_errorfmt(OSL::get_thread_index(ec), OSL::get_shade_index(ec),
                       fmt_specification, arg_count, argTypes, argValuesSize,
                       argValues);
}

OSL_RSOP void
rs_warningfmt(OSL::OpaqueExecContextPtr ec, OSL::ustringhash fmt_specification,
              int32_t arg_count, const OSL::EncodedType* argTypes,
              uint32_t argValuesSize, uint8_t* argValues)
{
    auto rs = OSL::get_rs<RenderState>(ec);

    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_warningfmt(OSL::get_max_warnings_per_thread(ec),
                         OSL::get_thread_index(ec), OSL::get_shade_index(ec),
                         fmt_specification, arg_count, argTypes, argValuesSize,
                         argValues);
}


OSL_RSOP void
rs_printfmt(OSL::OpaqueExecContextPtr ec, OSL::ustringhash fmt_specification,
            int32_t arg_count, const OSL::EncodedType* argTypes,
            uint32_t argValuesSize, uint8_t* argValues)
{
    auto rs = OSL::get_rs<RenderState>(ec);

    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_printfmt(OSL::get_thread_index(ec), OSL::get_shade_index(ec),
                       fmt_specification, arg_count, argTypes, argValuesSize,
                       argValues);
}


OSL_RSOP void
rs_filefmt(OSL::OpaqueExecContextPtr ec, OSL::ustringhash filename_hash,
           OSL::ustringhash fmt_specification, int32_t arg_count,
           const OSL::EncodedType* argTypes, uint32_t argValuesSize,
           uint8_t* argValues)
{
    auto rs = OSL::get_rs<RenderState>(ec);

    OSL::journal::Writer jw { rs->journal_buffer };
    jw.record_filefmt(OSL::get_thread_index(ec), OSL::get_shade_index(ec),
                      filename_hash, fmt_specification, arg_count, argTypes,
                      argValuesSize, argValues);
}
