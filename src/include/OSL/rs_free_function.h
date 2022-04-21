// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/oslconfig.h>

// Make all the strings defined in OSL
// available to the renderer services free functions
OSL_NAMESPACE_ENTER
namespace Strings {
#define STRDECL(str, var_name) OSLEXECPUBLIC extern const ustring var_name;
#include <OSL/strdecls.h>
#undef STRDECL
};  // namespace Strings
OSL_NAMESPACE_EXIT

#ifdef OSL_HOST_RS_BITCODE
extern "C" {
#    define STRDECL(str, var_name) \
        const OSL::ustring& RS_##var_name = OSL::Strings::var_name;
#    include <OSL/strdecls.h>
#    undef STRDECL
}
#endif

#include <OSL/device_string.h>
#include <OSL/shaderglobals.h>

// Prefix for OSL shade op declarations.
// "C" linkage (no C++ name mangling) and local visibility
#define OSL_RSOP extern "C"

// We are choosing to use unique names encoding parameters directly
// as opposed to using overloaded functions.

// Keep free functions in sync with virtual function based RendererServices.

/// Get the 4x4 matrix that transforms by the specified
/// transformation at the given time.  Return true if ok, false
/// on error.
OSL_RSOP bool
rs_get_matrix_xform_time(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                         OSL::TransformationPtr from, float time);

/// Get the 4x4 matrix that transforms by the specified
/// transformation at the given time.  Return true if ok, false on
/// error.  Suggested implementation is to use rs_get_matrix_xform_time and
/// invert it, but a particular renderer may have a better technique.
OSL_RSOP bool
rs_get_inverse_matrix_xform_time(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                                 OSL::TransformationPtr xform, float time);

/// Get the 4x4 matrix that transforms points from the named
/// 'from' coordinate system to "common" space at the given time.
/// Returns true if ok, false if the named matrix is not known.
OSL_RSOP bool
rs_get_matrix_space_time(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                         OSL::StringParam from, float time);

/// Get the 4x4 matrix that transforms points from "common" space to
/// the named 'to' coordinate system to at the given time.  Suggested
/// implementation is to use rs_get_matrix_space_time and invert it, but a
/// particular renderer may have a better technique.
OSL_RSOP bool
rs_get_inverse_matrix_space_time(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                                 OSL::StringParam to, float time);

/// Get the 4x4 matrix that transforms by the specified
/// transformation.  Return true if ok, false on error.  Since no
/// time value is given, also return false if the transformation may
/// be time-varying.
OSL_RSOP bool
rs_get_matrix_xform(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                    OSL::TransformationPtr xform);

/// Get the 4x4 matrix that transforms by the specified
/// transformation.  Return true if ok, false on error.  Since no
/// time value is given, also return false if the transformation may
/// be time-varying.  Suggested implementation is to use
/// rs_get_matrix_xform and invert it, but a particular renderer may have a
/// better technique.
OSL_RSOP bool
rs_get_inverse_matrix_xform(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OSL::TransformationPtr xform);

/// Get the 4x4 matrix that transforms 'from' to "common" space.
/// Since there is no time value passed, return false if the
/// transformation may be time-varying (as well as if it's not found
/// at all).
OSL_RSOP bool
rs_get_matrix_space(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                    OSL::StringParam from);

/// Get the 4x4 matrix that transforms points from "common" space to
/// the named 'to' coordinate system.  Since there is no time value
/// passed, return false if the transformation may be time-varying
/// (as well as if it's not found at all).  Suggested
/// implementation is to use rs_get_matrix_space and invert it, but a
/// particular renderer may have a better technique.
OSL_RSOP bool
rs_get_inverse_matrix_space(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OSL::StringParam to);

/// Transform points Pin[0..npoints-1] in named coordinate system
/// 'from' into 'to' coordinates, storing the result in Pout[] using
/// the specified vector semantic (POINT, VECTOR, NORMAL).  The
/// function returns true if the renderer correctly transformed the
/// points, false if it failed (for example, because it did not know
/// the name of one of the coordinate systems).  Suggested implementation
/// is simply to make appropriate calls to rs_get_matrix_space and
/// get_inverse_matrix_space.  The existence of this function is to allow
/// some renderers to provide transformations that cannot be
/// expressed by a 4x4 matrix.
///
/// Note, the virtual function based RendererServices::transform_points
/// provides additional functionality/modes that are only used during
/// code generation to detect if it required to even call rs_transform_points
/// during evaluation of the shader.  There is no need for the free
/// function version to handle these additional modes.
///
/// Note to implementations: just return 'false'
/// if there isn't a special nonlinear transformation between the
/// two spaces.
OSL_RSOP bool
rs_transform_points(OSL::ShaderGlobals* sg, OSL::StringParam from,
                    OSL::StringParam to, float time, const OSL::Vec3* Pin,
                    OSL::Vec3* Pout, int npoints,
                    OSL::TypeDesc::VECSEMANTICS vectype);