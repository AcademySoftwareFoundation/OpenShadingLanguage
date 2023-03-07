// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifndef OSL_HOST_RS_BITCODE
#    error OSL_HOST_RS_BITCODE must be defined by your build system.
#endif

#include <OSL/rs_free_function.h>
#include <OSL/journal.h>

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
rs_get_matrix_xform_time(OpaqueExecContextPtr exec_ctx/*OSL::ShaderGlobals* */ /*sg*/, OSL::Matrix44& result,
                         OSL::TransformationPtr xform, float /*time*/)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    auto ptr = reinterpret_cast<OSL::Matrix44 const*>(xform);
    result   = *ptr;
    return true;
}

OSL_RSOP bool
rs_get_inverse_matrix_xform_time(/*OSL::ShaderGlobals* sg*/ OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                                 OSL::TransformationPtr xform, float time)
{
    //auto sg = osl_get_sg(exec_ptr);
    bool ok = rs_get_matrix_xform_time(exec_ctx/*sg*/, result, xform, time);
    if (ok) {
        result.invert();
    }
    return ok;
}

OSL_RSOP bool
rs_get_matrix_space_time(OpaqueExecContextPtr exec_ctx/*OSL::ShaderGlobals* *//*sg*/, OSL::Matrix44& result,
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
rs_get_inverse_matrix_space_time(/*OSL::ShaderGlobals* sg*/OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                                 OSL::StringParam to, float time)
{
    using OSL::Matrix44;

    //RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);
    auto rs = osl_get_rs<RenderState>(exec_ctx);
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
        bool ok = rs_get_matrix_space_time(exec_ctx/*sg*/, result, to, time);
        if (ok) {
            result.invert();
        }

        return ok;
    }
}

OSL_RSOP bool
rs_get_matrix_xform(OpaqueExecContextPtr exec_ctx /*OSL::ShaderGlobals* *//*sg*/, OSL::Matrix44& result,
                    OSL::TransformationPtr xform)
{
    // SimpleRenderer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    auto ptr = reinterpret_cast<OSL::Matrix44 const*>(xform);
    result   = *ptr;
    return true;
}

OSL_RSOP bool
rs_get_inverse_matrix_xform(/*OSL::ShaderGlobals* sg*/ OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                            OSL::TransformationPtr xform)
{
    bool ok = rs_get_matrix_xform(exec_ctx, result, xform);
    if (ok) {
        result.invert();
    }

    return ok;
}

OSL_RSOP bool
rs_get_matrix_space(OpaqueExecContextPtr exec_ctx/*OSL::ShaderGlobals* *//*sg*/, OSL::Matrix44& /*result*/,
                    OSL::StringParam from)
{
    if (from == STRING_PARAMS(myspace)) {
        return true;
    } else {
        return false;
    }
}

OSL_RSOP bool
rs_get_inverse_matrix_space(/*OSL::ShaderGlobals* sg*/ OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                            OSL::StringParam to)
{
    bool ok = rs_get_matrix_space(/*sg*/exec_ctx, result, to);
    if (ok) {
        result.invert();
    }
    return ok;
}

OSL_RSOP bool
rs_transform_points(/*OSL::ShaderGlobals* */ OpaqueExecContextPtr exec_ctx/*sg*/, OSL::StringParam /*from*/,
                    OSL::StringParam /*to*/, float /*time*/,
                    const OSL::Vec3* /*Pin*/, OSL::Vec3* /*Pout*/,
                    int /*npoints*/, OSL::TypeDesc::VECSEMANTICS /*vectype*/)
{
    return false;
}



OSL_RSOP void
rs_errorfmt_dummy(/*OSL::ShaderGlobals* sg*/ OpaqueExecContextPtr exec_ctx)
{
    auto rs = osl_get_rs<RenderState>(exec_ctx);
  

    OSL::journal::Writer jw{rs->journal_buffer};
    //jw.record_errorfmt(sg->thread_index, sg->shade_index, rs_fmt_specification, arg_count, argTypes, argValuesSize, argValues);
}

OSL_RSOP void
rs_errorfmt(/*OSL::ShaderGlobals* sg*/ OpaqueExecContextPtr exec_ctx, 
            OSL::ustringhash fmt_specification, 
            //OSL::ustringhash fmt_specification, 
            int32_t arg_count, 
            const EncodedType *argTypes, 
            uint32_t argValuesSize, 
            uint8_t *argValues)
{
    //RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);
    auto rs = osl_get_rs<RenderState>(exec_ctx);
    auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
    // auto shade_index = osl_get_shade_index(exec_ctx);
    // auto thread_index = osl_get_thread_index(exec_ctx);

    //OSL::ustringhash rs_fmt_specification = OSL::ustringhash_from(fmt_specification);
//    OSL::ustringhash rs_fmt_specification{fmt_specification};

    OSL::journal::Writer jw{rs->journal_buffer};
    jw.record_errorfmt(sg->thread_index, sg->shade_index, fmt_specification, arg_count, argTypes, argValuesSize, argValues);
}

OSL_RSOP void
rs_warningfmt(/*OSL::ShaderGlobals* sg*/ OpaqueExecContextPtr exec_ctx, 
            OSL::ustringhash fmt_specification, 
            int32_t arg_count, 
            const EncodedType *argTypes, 
            uint32_t argValuesSize, 
            uint8_t *argValues)
{
    //RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);
   // auto rs = osl_get_rs<RenderState>(exec_ctx);
    //OSL::ShaderGlobals* sg = reinterpret_cast<OSL::ShaderGlobals*>(exec_ctx);
    

    auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);

    RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate); //SM: Becomes unreachable if I dont cast like so
   
   // OSL::ustringhash rs_fmt_specification = OSL::ustringhash_from(fmt_specification);
    //  std::cout<<"************************"<<std::endl;
    // std::cout<<"RS dummy: "<<rs->dummy<<std::endl;//SM: printed
    // std::cout<<"************************"<<std::endl;
    //if(rs == nullptr) {std::cout<<"RS null"<<std::endl;}
    //std::cout<<"RS dummy var: "<<rs->dummy<<std::endl;
    OSL::journal::Writer jw{rs->journal_buffer};
    
    jw.record_warningfmt(sg->thread_index, sg->shade_index, fmt_specification, arg_count, argTypes, argValuesSize, argValues);
    
}

OSL_RSOP void
rs_printffmt(/*OSL::ShaderGlobals* sg*/ OpaqueExecContextPtr exec_ctx, 
            OSL::ustringhash fmt_specification, 
            int32_t arg_count, 
            const EncodedType *argTypes, 
            uint32_t argValuesSize, 
            uint8_t *argValues)
{
    //RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);
    auto rs = osl_get_rs<RenderState>(exec_ctx);
    auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
    //OSL::ustringhash rs_fmt_specification = OSL::ustringhash_from(fmt_specification);

    OSL::journal::Writer jw{rs->journal_buffer};
    jw.record_printfmt(sg->thread_index, sg->shade_index, fmt_specification, arg_count, argTypes, argValuesSize, argValues);
}

OSL_RSOP void
rs_filefmt(/*OSL::ShaderGlobals* sg*/ OpaqueExecContextPtr exec_ctx, 
            OSL::ustringhash filename_hash, 
            OSL::ustringhash fmt_specification, 
            int32_t arg_count, 
            const EncodedType *argTypes, 
            uint32_t argValuesSize, 
            uint8_t *argValues)
{
    //RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);
    auto rs = osl_get_rs<RenderState>(exec_ctx);
    auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
    OSL::ustringhash rs_fmt_specification = OSL::ustringhash_from(fmt_specification);
    OSL::ustringhash rs_filename_hash = OSL::ustringhash_from(filename_hash);

    OSL::journal::Writer jw{rs->journal_buffer};
    jw.record_filefmt(sg->thread_index, sg->shade_index, filename_hash, fmt_specification, arg_count, argTypes, argValuesSize, argValues);
}


// OSL_RSOP void
// rs_messagefmt(OSL::ShaderGlobals* sg, 
//             OSL::ustringhash fmt_specification, 
//             int32_t arg_count, 
//             const EncodedType *argTypes, 
//             uint32_t argValuesSize, 
//             uint8_t *argValues)
// {
//     RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);

//     journal::Writer jw{rs->journal_buffer};
//     jw.record_message(sg->thread_index, sg->shade_index, fmt_specification, arg_count, argTypes, argValuesSize, argValues);

// }

// OSL_RSOP void
// rs_infofmt(OSL::ShaderGlobals* sg, 
//             OSL::ustringhash fmt_specification, 
//             int32_t arg_count, 
//             const EncodedType *argTypes, 
//             uint32_t argValuesSize, 
//             uint8_t *argValues)
// {
//     RenderState* rs = reinterpret_cast<RenderState*>(sg->renderstate);

//     OSL::journal::Writer jw{rs->journal_buffer};
//     jw.record_info(sg->thread_index, sg->shade_index, fmt_specification, arg_count, argTypes, argValuesSize, argValues);
// }

