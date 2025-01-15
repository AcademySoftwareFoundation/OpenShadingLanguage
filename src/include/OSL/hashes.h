// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>
#include <OpenImageIO/detail/farmhash.h>
#include <OpenImageIO/oiioversion.h>

// USAGE NOTES:
//
// To define a "standard" DeviceString, add a STRDECL to <OSL/strdecls.h>
// specifying the string literal and the name to use for the variable.


OSL_NAMESPACE_BEGIN

namespace pvt {

constexpr inline size_t
pvtstrlen(const char* s)
{
    if (s == nullptr)
        return 0;
    size_t len = 0;
    while (s[len] != 0)
        len++;
    return len;
}

}  // namespace pvt

// The string_view(const char *) is only constexpr for c++17
// which would prevent OIIO::Strutil::strhash from being
// constexpr for c++14.
// workaround by using local version here with a private
// constexpr strlen
constexpr inline size_t
strhash(const char* s)
{
    size_t len = pvt::pvtstrlen(s);
#ifdef OIIO_USTRING_SAFE_HASH
    return ustring::strhash(string_view(s, len));
#else
    return OIIO::Strutil::strhash(string_view(s, len));
#endif
}

// Template to ensure the hash is evaluated at compile time.
template<size_t V> static constexpr size_t HashConstEval = V;
#define OSL_HASHIFY(unquoted_string) \
    HashConstEval<OSL::strhash(__OSL_STRINGIFY(unquoted_string))>

namespace {  // Scope Hashes variables to just this translation unit
namespace Hashes {
#ifdef __CUDA_ARCH__  // TODO: restrict to CUDA version < 11.4, otherwise the contexpr should work
#    define STRDECL(str, var_name) \
        __device__ const OSL::ustringhash var_name(OSL::strhash(str));
#else
#    define STRDECL(str, var_name) \
        constexpr OSL::ustringhash var_name(OSL::strhash(str));
#endif
#include <OSL/strdecls.h>
#undef STRDECL
};  // namespace Hashes
}  // unnamed namespace



OSL_NAMESPACE_END
