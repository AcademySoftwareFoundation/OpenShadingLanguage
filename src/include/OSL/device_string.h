/*
Copyright (c) 2009-2018 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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

    OSL_HOSTDEVICE uint64_t hash()
    {
        return *(uint64_t*)(m_chars - sizeof(uint64_t) - sizeof(uint64_t));
    }

    OSL_HOSTDEVICE uint64_t length()
    {
        return *(uint64_t*)(m_chars - sizeof(uint64_t));
    }

    OSL_HOSTDEVICE const char* c_str()
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


// Handy re-casting macro
#define DEVSTR(cstr) (*(OSL::DeviceString*)&cstr)


#ifdef __CUDA_ARCH__
namespace DeviceStrings {
#define STRDECL(str,var_name)                   \
    extern __device__ DeviceString var_name;
#include <OSL/strdecls.h>
#undef STRDECL
}
#endif


OSL_NAMESPACE_EXIT
