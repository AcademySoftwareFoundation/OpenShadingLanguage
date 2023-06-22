// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/fmt_util.h>

#include <OSL/rendererservices.h>

#include <OSL/journal.h>


// Fallback is to reroute calls back through the virtual function
// based RendererServices from ShaderGlobals.
// We are intentially hiding ShaderGlobals and RendererServices from
// user supplied render service free functions as they should not
// allowed access to them.
// However to implement the fallback, we allow such access only right here

// Intentially private to this file
namespace {
inline OSL::ShaderGlobals*
get_sg(OSL::OpaqueExecContextPtr cptr)
{
    return reinterpret_cast<OSL::ShaderGlobals*>(cptr);
}
}  // namespace


// Host only fallback implementation of free function renderer services that
// simply forward the calls to the existing virtual RendererServices methods

OSL_RSOP bool
rs_get_matrix_xform_time(OSL::OpaqueExecContextPtr exec_ctx,
                         OSL::Matrix44& result, OSL::TransformationPtr from,
                         float time)
{
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_matrix(sg, result, from, time);
}

OSL_RSOP bool
rs_get_inverse_matrix_xform_time(OSL::OpaqueExecContextPtr exec_ctx,
                                 OSL::Matrix44& result,
                                 OSL::TransformationPtr xform, float time)
{
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_inverse_matrix(sg, result, xform, time);
}

OSL_RSOP bool
rs_get_matrix_space_time(OSL::OpaqueExecContextPtr exec_ctx,
                         OSL::Matrix44& result, OSL::StringParam from,
                         float time)
{
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_matrix(sg, result, from, time);
}

OSL_RSOP bool
rs_get_inverse_matrix_space_time(OSL::OpaqueExecContextPtr exec_ctx,
                                 OSL::Matrix44& result, OSL::StringParam to,
                                 float time)
{
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_inverse_matrix(sg, result, to, time);
}

OSL_RSOP bool
rs_get_matrix_xform(OSL::OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                    OSL::TransformationPtr xform)
{
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_matrix(sg, result, xform);
}

OSL_RSOP bool
rs_get_inverse_matrix_xform(OSL::OpaqueExecContextPtr exec_ctx,
                            OSL::Matrix44& result, OSL::TransformationPtr xform)
{
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_inverse_matrix(sg, result, xform);
}

OSL_RSOP bool
rs_get_matrix_space(OSL::OpaqueExecContextPtr exec_ctx, OSL::Matrix44& result,
                    OSL::StringParam from)
{
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_matrix(sg, result, from);
}

OSL_RSOP bool
rs_get_inverse_matrix_space(OSL::OpaqueExecContextPtr exec_ctx,
                            OSL::Matrix44& result, OSL::StringParam to)
{
    auto sg = get_sg(exec_ctx);
    return sg->renderer->get_inverse_matrix(sg, result, to);
}

OSL_RSOP bool
rs_transform_points(OSL::OpaqueExecContextPtr exec_ctx, OSL::StringParam from,
                    OSL::StringParam to, float time, const OSL::Vec3* Pin,
                    OSL::Vec3* Pout, int npoints,
                    OSL::TypeDesc::VECSEMANTICS vectype)
{
    auto sg = get_sg(exec_ctx);
    return sg->renderer->transform_points(sg, from, to, time, Pin, Pout,
                                          npoints, vectype);
}

OSL_RSOP void
rs_errorfmt(OSL::OpaqueExecContextPtr exec_ctx,
            OSL::ustringhash fmt_specification, int32_t count,
            const OSL::EncodedType* argTypes, uint32_t argValuesSize,
            uint8_t* argValues)
{
    auto sg = get_sg(exec_ctx);
    sg->renderer->errorfmt(sg, fmt_specification, count, argTypes,
                           argValuesSize, argValues);
}

OSL_RSOP void
rs_warningfmt(OSL::OpaqueExecContextPtr exec_ctx,
              OSL::ustringhash fmt_specification, int32_t count,
              const OSL::EncodedType* argTypes, uint32_t argValuesSize,
              uint8_t* argValues)
{
    auto sg = get_sg(exec_ctx);
    sg->renderer->warningfmt(sg, fmt_specification, count, argTypes,
                             argValuesSize, argValues);
}

OSL_RSOP void
rs_printfmt(OSL::OpaqueExecContextPtr exec_ctx,
            OSL::ustringhash fmt_specification, int32_t count,
            const OSL::EncodedType* argTypes, uint32_t argValuesSize,
            uint8_t* argValues)
{
    auto sg = get_sg(exec_ctx);
    sg->renderer->printfmt(sg, fmt_specification, count, argTypes,
                           argValuesSize, argValues);
}

OSL_RSOP void
rs_filefmt(OSL::OpaqueExecContextPtr exec_ctx, OSL::ustringhash filename,
           OSL::ustringhash fmt_specification, int32_t count,
           const OSL::EncodedType* argTypes, uint32_t argValuesSize,
           uint8_t* argValues)
{
    auto sg = get_sg(exec_ctx);
    sg->renderer->filefmt(sg, filename, fmt_specification, count, argTypes,
                          argValuesSize, argValues);
}
