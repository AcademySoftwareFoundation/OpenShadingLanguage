// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/rs_free_function.h>

#include <OSL/rendererservices.h>

// Host only fallback implementation of free function renderer services that
// simply forward the calls to the existing virtual RendererServices methods

OSL_RSOP bool
rs_get_matrix_xform_time(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                         OSL::TransformationPtr from, float time)
{
    return sg->renderer->get_matrix(sg, result, from, time);
}

OSL_RSOP bool
rs_get_inverse_matrix_xform_time(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                                 OSL::TransformationPtr xform, float time)
{
    return sg->renderer->get_inverse_matrix(sg, result, xform, time);
}

OSL_RSOP bool
rs_get_matrix_space_time(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                         OSL::StringParam from, float time)
{
    return sg->renderer->get_matrix(sg, result, from, time);
}

OSL_RSOP bool
rs_get_inverse_matrix_space_time(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                                 OSL::StringParam to, float time)
{
    return sg->renderer->get_inverse_matrix(sg, result, to, time);
}

OSL_RSOP bool
rs_get_matrix_xform(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                    OSL::TransformationPtr xform)
{
    return sg->renderer->get_matrix(sg, result, xform);
}

OSL_RSOP bool
rs_get_inverse_matrix_xform(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OSL::TransformationPtr xform)
{
    return sg->renderer->get_inverse_matrix(sg, result, xform);
}

OSL_RSOP bool
rs_get_matrix_space(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                    OSL::StringParam from)
{
    return sg->renderer->get_matrix(sg, result, from);
}

OSL_RSOP bool
rs_get_inverse_matrix_space(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OSL::StringParam to)
{
    return sg->renderer->get_inverse_matrix(sg, result, to);
}

OSL_RSOP bool
rs_transform_points(OSL::ShaderGlobals* sg, OSL::StringParam from,
                    OSL::StringParam to, float time, const OSL::Vec3* Pin,
                    OSL::Vec3* Pout, int npoints,
                    OSL::TypeDesc::VECSEMANTICS vectype)
{
    return sg->renderer->transform_points(sg, from, to, time, Pin, Pout,
                                          npoints, vectype);
}
