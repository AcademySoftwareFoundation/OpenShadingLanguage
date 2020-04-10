// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>


// USAGE NOTES:
//
// To define a "standard" DeviceString, add a STRDECL to <OSL/strdecls.h>
// specifying the string literal and the name to use for the variable.


OSL_NAMESPACE_ENTER


// Strings are stored in a block of CUDA device memory. String variables hold a
// pointer to the start of the char array for each string. Each canonical string
// has a unique entry in the table, so two strings can be tested for equality by
// comparing their addresses.
//
// As a convenience, the ustring hash and the length of the string are also
// stored in the table, in the 16 bytes preceding the characters.

struct DeviceString {
    const char* m_chars;

    OSL_HOSTDEVICE uint64_t hash() const
    {
        return *(uint64_t*)(m_chars - sizeof(uint64_t) - sizeof(uint64_t));
    }

    OSL_HOSTDEVICE uint64_t length() const
    {
        return *(uint64_t*)(m_chars - sizeof(uint64_t));
    }

    OSL_HOSTDEVICE const char* c_str() const
    {
        return m_chars;
    }

    OSL_HOSTDEVICE bool operator== (const DeviceString& other) const
    {
        return m_chars == other.m_chars;
    }

    OSL_HOSTDEVICE bool operator!= (const DeviceString& other) const
    {
        return m_chars != other.m_chars;
    }
};


// Choose the right cast for string parameters depending on the target. The
// macro is the same as the USTR macro defined in oslexec_pvt.h when compiling
// for the host.
#ifndef __CUDA_ARCH__
# define HDSTR(cstr) (*((ustring *)&cstr))
#else
# define HDSTR(cstr) (*(OSL::DeviceString*)&cstr)
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
#define STRDECL(str,var_name)                       \
    extern __device__ OSL_NAMESPACE::DeviceString var_name;
#include <OSL/strdecls.h>
#undef STRDECL
}
#endif


OSL_NAMESPACE_EXIT


#ifdef __CUDA_ARCH__
namespace StringParams = OSL_NAMESPACE::DeviceStrings;
#else
namespace StringParams = OSL_NAMESPACE::Strings;
#endif
