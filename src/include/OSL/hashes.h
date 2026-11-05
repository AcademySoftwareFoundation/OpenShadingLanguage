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
// nvcc needs the __device__ variables: its device code cannot reference a constexpr host
// variable of class type (CUDA 13.2 still rejects it). Clang can use the constexpr variables
// in device code, and it would emit the __device__ ones with external linkage (LLVM 23), which
// makes llvm-link fail with "symbol multiply defined" when it links the bitcode of several
// source files. nvcc preprocesses the device code with its host compiler, which defines
// __clang__ when it is clang, so test for nvcc with __NVCC__.
#if defined(__CUDA_ARCH__) && defined(__NVCC__)
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
