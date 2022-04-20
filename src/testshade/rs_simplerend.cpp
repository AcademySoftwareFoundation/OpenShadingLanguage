// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifndef OSL_HOST_RS_BITCODE
#  error OSL_HOST_RS_BITCODE must be defined by your build system.
#endif

#include <OSL/rs_free_function.h>

#include "render_state.h"

// Extern all global string variables used by free function renderer services.
// NOTE:  C linkage with a "RS_" prefix is used to allow for unmangled
// non-colliding global symbol names, so its easier to pass them to
// OSL::register_JIT_Global(name, addr) for host execution
// NOTE:  the STRING_PARAMS macro adapts to OSL_HOST_RS_BITCODE 
// to utilize the RS_ prefix.  RS_ prefixed versions of all OSL::Strings
// intances have been created by rs_free_function.h, so the same STRING_PARAMS
// macro can be used for renderer service or OSL strings.
#define RS_STRDECL(str, var_name) extern "C" OSL::ustring RS_##var_name; 
#include "rs_strdecls.h" 
#undef RS_STRDECL


// Keep free functions in sync with virtual function based SimpleRenderer.

OSL_RSOP bool
rs_get_matrix_xform_time(OSL::ShaderGlobals* /*sg*/, OSL::Matrix44& result,
                         OSL::TransformationPtr xform, float /*time*/)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    auto ptr = reinterpret_cast<OSL::Matrix44 const*>(xform);
    result   = *ptr;
    return true;
}

OSL_RSOP bool
rs_get_inverse_matrix_xform_time(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                                 OSL::TransformationPtr xform, float time)
{
    bool ok = rs_get_matrix_xform_time(sg, result, xform, time);
    if (ok) {
        result.invert();
    }
    return ok;
}

OSL_RSOP bool
rs_get_matrix_space_time(OSL::ShaderGlobals* /*sg*/, OSL::Matrix44& result,
                         OSL::StringParam from, float /*time*/)
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
rs_get_inverse_matrix_space_time(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                                 OSL::StringParam to, float time)
{
    using OSL::Matrix44;

    RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);
    if (to == STRING_PARAMS(camera) || to == STRING_PARAMS(screen)
        || to == STRING_PARAMS(NDC) || to == STRING_PARAMS(raster)) {
        Matrix44 M { rs->world_to_camera };

        if (to == STRING_PARAMS(screen) || to == STRING_PARAMS(NDC)
            || to == STRING_PARAMS(raster)) {
            float depthrange = (double)rs->yon - (double)rs->hither;
            //OSL::StringParam proj{rs->projection.m_chars};
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
        bool ok = rs_get_matrix_space_time(sg, result, to, time);
        if (ok) {
            result.invert();
        }

        return ok;
    }
}

OSL_RSOP bool
rs_get_matrix_xform(OSL::ShaderGlobals* /*sg*/, OSL::Matrix44& result,
                    OSL::TransformationPtr xform)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    auto ptr = reinterpret_cast<OSL::Matrix44 const*>(xform);
    result   = *ptr;
    return true;
}

OSL_RSOP bool
rs_get_inverse_matrix_xform(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OSL::TransformationPtr xform)
{
    bool ok = rs_get_matrix_xform(sg, result, xform);
    if (ok) {
        result.invert();
    }

    return ok;
}

OSL_RSOP bool
rs_get_matrix_space(OSL::ShaderGlobals* /*sg*/, OSL::Matrix44& /*result*/,
                    OSL::StringParam from)
{
    if (from == STRING_PARAMS(myspace)) {
        return true;
    } else {
        return false;
    }
}

OSL_RSOP bool
rs_get_inverse_matrix_space(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OSL::StringParam to)
{
    bool ok = rs_get_matrix_space(sg, result, to);
    if (ok) {
        result.invert();
    }
    return ok;
}

OSL_RSOP bool
rs_transform_points(OSL::ShaderGlobals* /*sg*/, OSL::StringParam /*from*/,
                    OSL::StringParam /*to*/, float /*time*/,
                    const OSL::Vec3* /*Pin*/, OSL::Vec3* /*Pout*/,
                    int /*npoints*/, OSL::TypeDesc::VECSEMANTICS /*vectype*/)
{
    return false;
}
