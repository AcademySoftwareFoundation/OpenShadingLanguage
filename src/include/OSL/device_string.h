// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>

// USAGE NOTES:
//
// To define a "standard" DeviceString, add a STRDECL to <OSL/strdecls.h>
// specifying the string literal and the name to use for the variable.


OSL_NAMESPACE_ENTER


// In OptiX 7+:
// Strings are not stored on the GPU at all.  Strings are represented by
// their hash values which are computed at compile-time.  Because of this,
// we have no access to either the strings' contents or their lengths.

struct DeviceString {
#ifdef __CUDA_ARCH__
    size_t m_chars;
#else
    const char* m_chars;
#endif

#ifdef __CUDA_ARCH__
    OSL_HOSTDEVICE DeviceString() {}
    OSL_HOSTDEVICE DeviceString(uint64_t i) : m_chars(i) {}
#endif

    OSL_HOSTDEVICE uint64_t hash() const
    {
#ifdef __CUDA_ARCH__
        return m_chars;
#else
        return *(uint64_t*)(m_chars - sizeof(uint64_t) - sizeof(uint64_t));
#endif
    }

    // In OptiX 7 we don't store the string's length. Make this a compile
    // time error.
#ifndef __CUDA_ARCH__
    OSL_HOSTDEVICE uint64_t length() const
    {
        return *(uint64_t*)(m_chars - sizeof(uint64_t));
    }
#endif

    // In OptiX 7 we can't return the string's contents. Make this a compile
    // time error.
#ifndef __CUDA_ARCH__
    OSL_HOSTDEVICE const char* c_str() const { return m_chars; }
#endif

    OSL_HOSTDEVICE bool operator==(const DeviceString& other) const
    {
        return m_chars == other.m_chars;
    }

    OSL_HOSTDEVICE bool operator!=(const DeviceString& other) const
    {
        return m_chars != other.m_chars;
    }

#ifdef __CUDA_ARCH__

    OSL_HOSTDEVICE bool operator==(const size_t other) const
    {
        return hash() == other;
    }

    OSL_HOSTDEVICE bool operator!=(const size_t other) const
    {
        return hash() != other;
    }

    OSL_HOSTDEVICE operator size_t() const { return hash(); }

#endif
};


// Choose the right cast for string parameters depending on the target. The
// macro is the same as the USTR macro defined in oslexec_pvt.h when compiling
// for the host.
#ifndef __CUDA_ARCH__
#    define HDSTR(cstr) (*((ustring*)&cstr))
#else
#    define HDSTR(cstr) (*((OSL::DeviceString*)&cstr))
#endif


// When compiling shadeops C++ sources for CUDA devices, we need to use
// DeviceString instead of ustring for some input parameters, so we use this
// typedef to select the correct type depending on the target.
#ifndef __CUDA_ARCH__
typedef ustring StringParam;
#else
typedef DeviceString StringParam;
#endif


#ifdef __CUDA_ARCH__
namespace DeviceStrings {
#    define STRDECL(str, var_name) \
        extern __device__ OSL_NAMESPACE::DeviceString var_name;
#    undef STRDECL
}  // namespace DeviceStrings
#else
#    ifdef OSL_HOST_RS_BITCODE
#        define STRING_PARAMS(x) RS_##x
#    else
#        define STRING_PARAMS(x) StringParams::x
#    endif
#endif


OSL_NAMESPACE_EXIT


#ifndef __CUDA_ARCH__
namespace StringParams = OSL_NAMESPACE::Strings;
#endif
