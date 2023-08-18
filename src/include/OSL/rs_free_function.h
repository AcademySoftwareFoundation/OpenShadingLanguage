// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/encodedtypes.h>
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
// Must be re-enterant; Expect to be called from multiple threads

/// Get the 4x4 matrix that transforms by the specified
/// transformation at the given time.  Return true if ok, false
/// on error.
OSL_RSOP bool
rs_get_matrix_xform_time(OSL::OpaqueExecContextPtr oec, OSL::Matrix44& result,
                         OSL::TransformationPtr from, float time);

/// Get the 4x4 matrix that transforms by the specified
/// transformation at the given time.  Return true if ok, false on
/// error.  Suggested implementation is to use rs_get_matrix_xform_time and
/// invert it, but a particular renderer may have a better technique.
OSL_RSOP bool
rs_get_inverse_matrix_xform_time(OSL::OpaqueExecContextPtr oec,
                                 OSL::Matrix44& result,
                                 OSL::TransformationPtr xform, float time);

/// Get the 4x4 matrix that transforms points from the named
/// 'from' coordinate system to "common" space at the given time.
/// Returns true if ok, false if the named matrix is not known.
OSL_RSOP bool
rs_get_matrix_space_time(OSL::OpaqueExecContextPtr oec, OSL::Matrix44& result,
                         OSL::StringParam from, float time);

/// Get the 4x4 matrix that transforms points from "common" space to
/// the named 'to' coordinate system to at the given time.  Suggested
/// implementation is to use rs_get_matrix_space_time and invert it, but a
/// particular renderer may have a better technique.
OSL_RSOP bool
rs_get_inverse_matrix_space_time(OSL::OpaqueExecContextPtr oec,
                                 OSL::Matrix44& result, OSL::StringParam to,
                                 float time);

/// Get the 4x4 matrix that transforms by the specified
/// transformation.  Return true if ok, false on error.  Since no
/// time value is given, also return false if the transformation may
/// be time-varying.
OSL_RSOP bool
rs_get_matrix_xform(OSL::OpaqueExecContextPtr oec, OSL::Matrix44& result,
                    OSL::TransformationPtr xform);

/// Get the 4x4 matrix that transforms by the specified
/// transformation.  Return true if ok, false on error.  Since no
/// time value is given, also return false if the transformation may
/// be time-varying.  Suggested implementation is to use
/// rs_get_matrix_xform and invert it, but a particular renderer may have a
/// better technique.
OSL_RSOP bool
rs_get_inverse_matrix_xform(OSL::OpaqueExecContextPtr oec,
                            OSL::Matrix44& result,
                            OSL::TransformationPtr xform);

/// Get the 4x4 matrix that transforms 'from' to "common" space.
/// Since there is no time value passed, return false if the
/// transformation may be time-varying (as well as if it's not found
/// at all).
OSL_RSOP bool
rs_get_matrix_space(OSL::OpaqueExecContextPtr oec, OSL::Matrix44& result,
                    OSL::StringParam from);

/// Get the 4x4 matrix that transforms points from "common" space to
/// the named 'to' coordinate system.  Since there is no time value
/// passed, return false if the transformation may be time-varying
/// (as well as if it's not found at all).  Suggested
/// implementation is to use rs_get_matrix_space and invert it, but a
/// particular renderer may have a better technique.
OSL_RSOP bool
rs_get_inverse_matrix_space(OSL::OpaqueExecContextPtr oec,
                            OSL::Matrix44& result, OSL::StringParam to);

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
rs_transform_points(OSL::OpaqueExecContextPtr oec, OSL::StringParam from,
                    OSL::StringParam to, float time, const OSL::Vec3* Pin,
                    OSL::Vec3* Pout, int npoints,
                    OSL::TypeDesc::VECSEMANTICS vectype);


/// Report errors, warnings, printf, and fprintf.
/// Fmtlib style format specifier is used (vs. printf style)
/// Arguments are represented as EncodedTypes (encodedtypes.h) and
/// packed into an arg_values buffer.  OSL::decode_message converts these
/// arguments into a std::string for renderer's handle to use as they please.
/// For device compatibility, the format specifier and any string arguments
/// are passed as ustringhash's.
/// It is recomended to override and make use of the journal buffer to record
/// everything to a buffer than can be post processed as needed.
OSL_RSOP void
rs_errorfmt(OSL::OpaqueExecContextPtr oec, OSL::ustringhash fmt_specification,
            int32_t count, const OSL::EncodedType* argTypes,
            uint32_t argValuesSize, uint8_t* argValues);

OSL_RSOP void
rs_filefmt(OSL::OpaqueExecContextPtr oec, OSL::ustringhash filname_hash,
           OSL::ustringhash fmt_specification, int32_t count,
           const OSL::EncodedType* argTypes, uint32_t argValuesSize,
           uint8_t* argValues);

OSL_RSOP void
rs_printfmt(OSL::OpaqueExecContextPtr oec, OSL::ustringhash fmt_specification,
            int32_t count, const OSL::EncodedType* argTypes,
            uint32_t argValuesSize, uint8_t* argValues);


OSL_RSOP void
rs_warningfmt(OSL::OpaqueExecContextPtr oec, OSL::ustringhash fmt_specification,
              int32_t count, const OSL::EncodedType* argTypes,
              uint32_t argValuesSize, uint8_t* argValues);


#if 0
// C++ helpers to accept variable arguements and encode messages to be sent
// through renderer service free functions
OSL_NAMESPACE_ENTER

namespace pvt {
// PackedArgs is similar to tuple but packs its data back to back
// in memory layout, which is what we need to build up payload
// to the fmt reporting system
template<int IndexT, typename TypeT> struct PackedArg {
    explicit PackedArg(const TypeT& a_value) : m_value(a_value) {}
    TypeT m_value;
} __attribute__((packed, aligned(1)));

template<typename IntSequenceT, typename... TypeListT> struct PackedArgsBase;
// Specialize to extract a parameter pack of the IntegerSquence
// so it can be expanded alongside the TypeListT parameter pack
template<int... IntegerListT, typename... TypeListT>
struct PackedArgsBase<std::integer_sequence<int, IntegerListT...>, TypeListT...>
    : public PackedArg<IntegerListT, TypeListT>... {
    explicit PackedArgsBase(const TypeListT&... a_values)
        // multiple inheritance of individual components
        // uniquely identified by the <Integer,Type> combo
        : PackedArg<IntegerListT, TypeListT>(a_values)...
    {
    }
} __attribute__((packed, aligned(1)));

template<typename... TypeListT> struct PackedArgs {
    typedef std::make_integer_sequence<int, sizeof...(TypeListT)>
        IndexSequenceType;
    PackedArgsBase<IndexSequenceType, TypeListT...> m_components;

    explicit PackedArgs(const TypeListT&... a_values)
        : m_components(a_values...)
    {
    }
};
}  // namespace pvt



template<typename FilenameT, typename SpecifierT, typename... ArgListT>
void
filefmt(OpaqueExecContextPtr oec, const FilenameT& filename_hash,
        const SpecifierT& fmt_specification, ArgListT... args)
{
    constexpr int32_t count = sizeof...(args);
    constexpr OSL::EncodedType argTypes[]
        = { pvt::TypeEncoder<ArgListT>::value... };
    pvt::PackedArgs<typename pvt::TypeEncoder<ArgListT>::DataType...> argValues {
        pvt::TypeEncoder<ArgListT>::Encode(args)...
    };

    rs_filefmt(oec, OSL::ustringhash { filename_hash },
               OSL::ustringhash { fmt_specification }, count,
               (count == 0) ? nullptr : argTypes,
               static_cast<uint32_t>((count == 0) ? 0 : sizeof(argValues)),
               (count == 0) ? nullptr : reinterpret_cast<uint8_t*>(&argValues));
}

template<typename SpecifierT, typename... ArgListT>
void
printfmt(OpaqueExecContextPtr oec, const SpecifierT& fmt_specification,
         ArgListT... args)
{
    constexpr int32_t count = sizeof...(args);
    constexpr OSL::EncodedType argTypes[]
        = { pvt::TypeEncoder<ArgListT>::value... };
    pvt::PackedArgs<typename pvt::TypeEncoder<ArgListT>::DataType...> argValues {
        pvt::TypeEncoder<ArgListT>::Encode(args)...
    };

    rs_printfmt(oec, OSL::ustringhash { fmt_specification }, count,
                (count == 0) ? nullptr : argTypes,
                static_cast<uint32_t>((count == 0) ? 0 : sizeof(argValues)),
                (count == 0) ? nullptr
                             : reinterpret_cast<uint8_t*>(&argValues));
}

template<typename SpecifierT, typename... ArgListT>
void
errorfmt(OpaqueExecContextPtr oec, const SpecifierT& fmt_specification,
         ArgListT... args)
{
    constexpr int32_t count = sizeof...(args);
    constexpr OSL::EncodedType argTypes[]
        = { pvt::TypeEncoder<ArgListT>::value... };
    pvt::PackedArgs<typename pvt::TypeEncoder<ArgListT>::DataType...> argValues {
        pvt::TypeEncoder<ArgListT>::Encode(args)...
    };

    rs_errorfmt(oec, OSL::ustringhash { fmt_specification }, count,
                (count == 0) ? nullptr : argTypes,
                static_cast<uint32_t>((count == 0) ? 0 : sizeof(argValues)),
                (count == 0) ? nullptr
                             : reinterpret_cast<uint8_t*>(&argValues));
}

template<typename SpecifierT, typename... ArgListT>
void
warningfmt(OpaqueExecContextPtr oec, const SpecifierT& fmt_specification,
           ArgListT... args)
{
    constexpr int32_t count = sizeof...(args);
    constexpr OSL::EncodedType argTypes[]
        = { pvt::TypeEncoder<ArgListT>::value... };
    pvt::PackedArgs<typename pvt::TypeEncoder<ArgListT>::DataType...> argValues {
        pvt::TypeEncoder<ArgListT>::Encode(args)...
    };

    rs_warningfmt(oec, OSL::ustringhash { fmt_specification }, count,
                  (count == 0) ? nullptr : argTypes,
                  static_cast<uint32_t>((count == 0) ? 0 : sizeof(argValues)),
                  (count == 0) ? nullptr
                               : reinterpret_cast<uint8_t*>(&argValues));
}


OSL_NAMESPACE_EXIT
#endif