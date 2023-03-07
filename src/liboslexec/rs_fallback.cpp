// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/rs_free_function.h>

#include <OSL/rendererservices.h>

#include <OSL/journal.h>



// Host only fallback implementation of free function renderer services that
// simply forward the calls to the existing virtual RendererServices methods

OSL_RSOP bool
rs_get_matrix_xform_time(/*OSL::ShaderGlobals* sg*/OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                         OSL::TransformationPtr from, float time)
{
    auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
    return sg->renderer->get_matrix(sg, result, from, time);
}

OSL_RSOP bool
rs_get_inverse_matrix_xform_time(/*OSL::ShaderGlobals* sg*/OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                                 OSL::TransformationPtr xform, float time)
{
    auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
    return sg->renderer->get_inverse_matrix(sg, result, xform, time);
}

OSL_RSOP bool
rs_get_matrix_space_time(/*OSL::ShaderGlobals* sg*/OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                         OSL::StringParam from, float time)
{
    auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
    return sg->renderer->get_matrix(sg, result, from, time);
}

OSL_RSOP bool
rs_get_inverse_matrix_space_time(/*OSL::ShaderGlobals* sg*/OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                                 OSL::StringParam to, float time)
{
    auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
    return sg->renderer->get_inverse_matrix(sg, result, to, time);
}

OSL_RSOP bool
rs_get_matrix_xform(/*OSL::ShaderGlobals* sg*/OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                    OSL::TransformationPtr xform)
{
    auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
    return sg->renderer->get_matrix(sg, result, xform);
}

OSL_RSOP bool
rs_get_inverse_matrix_xform(/*OSL::ShaderGlobals* sg*/OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                            OSL::TransformationPtr xform)
{
    auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
    return sg->renderer->get_inverse_matrix(sg, result, xform);
}

OSL_RSOP bool
rs_get_matrix_space(/*OSL::ShaderGlobals* sg*/OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                    OSL::StringParam from)
{
    auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
    return sg->renderer->get_matrix(sg, result, from);
}

OSL_RSOP bool
rs_get_inverse_matrix_space(/*OSL::ShaderGlobals* sg*/OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                            OSL::StringParam to)
{
    auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
    return sg->renderer->get_inverse_matrix(sg, result, to);
}

OSL_RSOP bool
rs_transform_points(/*OSL::ShaderGlobals* sg*/OpaqueExecContextPtr exec_ctx, OSL::StringParam from,
                    OSL::StringParam to, float time, const OSL::Vec3* Pin,
                    OSL::Vec3* Pout, int npoints,
                    OSL::TypeDesc::VECSEMANTICS vectype)
{
    auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
    return sg->renderer->transform_points(sg, from, to, time, Pin, Pout,
                                          npoints, vectype);
}

OSL_RSOP void
rs_errorfmt_dummy(/*OSL::ShaderGlobals* sg*/OpaqueExecContextPtr exec_ctx)
{
   
}
OSL_RSOP void
rs_errorfmt(/*OSL::ShaderGlobals* sg*/OpaqueExecContextPtr exec_ctx, 
            OSL::ustringhash fmt_specification, 
            int32_t count, 
            const EncodedType *argTypes, 
            uint32_t argValuesSize, 
            uint8_t *argValues)
{
   
   auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
   sg->renderer->errorfmt(sg, fmt_specification, count, 
                        argTypes, argValuesSize, 
                        argValues);
}

OSL_RSOP void
rs_warningfmt(/*OSL::ShaderGlobals* sg*/OpaqueExecContextPtr exec_ctx, 
            OSL::ustringhash fmt_specification, 
            int32_t count, 
            const EncodedType *argTypes, 
            uint32_t argValuesSize, 
            uint8_t *argValues)
{   
    auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
    sg->renderer->warningfmt(sg, fmt_specification, count, 
                        argTypes, argValuesSize, 
                        argValues);
}



OSL_RSOP void
rs_printfmt(/*OSL::ShaderGlobals* sg*/OpaqueExecContextPtr exec_ctx, 
            OSL::ustringhash fmt_specification, 
            int32_t count, 
            const EncodedType *argTypes, 
            uint32_t argValuesSize, 
            uint8_t *argValues)
{
   auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
   sg->renderer->printfmt(sg, fmt_specification, count, 
                        argTypes, argValuesSize, 
                        argValues);
}

OSL_RSOP void
rs_filefmt(/*OSL::ShaderGlobals* sg*/OpaqueExecContextPtr exec_ctx,
            OSL::ustringhash filename, 
            OSL::ustringhash fmt_specification, 
            int32_t count, 
            const EncodedType *argTypes, 
            uint32_t argValuesSize, 
            uint8_t *argValues)
{

   auto sg = osl_get_sg<OSL::ShaderGlobals>(exec_ctx);
   sg->renderer->filefmt(sg, filename, fmt_specification, count, 
                        argTypes, argValuesSize, 
                        argValues);
}


