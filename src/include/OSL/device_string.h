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


#if 0
struct DeviceString {
    uint64_t addr;

    OSL_HOSTDEVICE const char* c_str() const
    {
        return reinterpret_cast<const char*>(addr);
    }

    OSL_HOSTDEVICE bool operator== (const DeviceString& other) const
    {
        return addr == other.addr;
    }

    OSL_HOSTDEVICE bool operator!= (const DeviceString& other) const
    {
        return addr != other.addr;
    }
};


// Cast a raw DeviceString pointer to this type to access the hash and length
// more conveniently.
struct StrRep {
    uint64_t    hash;
    uint64_t    len;
    const char* chars;
};
#endif


// A device-side representation of an OIIO::ustring. This is assuming
// that the ustrings are being allocated in Unified Memory that is shared
// between the GPU and the host.
struct UstringDevice {
    const char* m_chars;

#if 0
    OSL_HOSTDEVICE
    UstringDevice () { m_chars = nullptr; }

    OSL_HOSTDEVICE
    UstringDevice (void* ptr) { m_chars = (const char*) ptr; }

    OSL_HOSTDEVICE
    UstringDevice (uint64_t addr) { m_chars = (const char*) addr; }
#endif

    OSL_HOSTDEVICE const char *c_str () const {
        return m_chars;
    }

    OSL_HOSTDEVICE size_t length (void) const {
        if (! m_chars)
            return 0;
        const TableRepView *rep = ((const TableRepView *)m_chars) - 1;
        return rep->length;
    }

    OSL_HOSTDEVICE size_t hash (void) const {
        if (! m_chars)
            return 0;
        const TableRepView *rep = ((const TableRepView *)m_chars) - 1;
        return rep->hashed;
    }

    OSL_HOSTDEVICE bool operator== (const UstringDevice& other) const
    {
        return m_chars == other.m_chars;
    }

    OSL_HOSTDEVICE bool operator!= (const UstringDevice other) const
    {
        return m_chars != other.m_chars;
    }

    // A simplified view of a host-side TableRepMap entry, designed to place
    // the hash, length, and C-string at the correct offsets.
    struct TableRepView {
        size_t     hashed;
        const char pad0[32] = {0};
        size_t     length;
        size_t     dummy_capacity;
        int        dummy_refcount;

        OSL_HOSTDEVICE  TableRepView () { }
        OSL_HOSTDEVICE ~TableRepView () { }

        OSL_HOSTDEVICE const char *c_str () const {
            return (const char *)(this + 1);
        }
    };
};


OSL_NAMESPACE_EXIT


#ifdef __CUDA_ARCH__
namespace DeviceStrings {
#define STRDECL(str,var_name)                   \
    extern __device__ OSL::UstringDevice var_name;
#include <OSL/strdecls.h>
#undef STRDECL
}
#endif
