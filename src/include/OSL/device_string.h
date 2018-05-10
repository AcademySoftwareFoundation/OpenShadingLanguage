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

OSL_NAMESPACE_ENTER


// For the universe of "standard" strings, we want to fix the tags so that OSL
// library code (shadeops, noise gen, etc.), compiled shaders, and the renderer
// can agree on the tag associated with each "canonical" string.
enum StringTags: uint64_t {
    EMPTY_STRING = 0,
    TEST_STRING,
    UNKNOWN_STRING = ~0u
};


struct device_string {
    OSL_HOSTDEVICE uint64_t tag ()
    {
        return m_tag;
    }

    OSL_HOSTDEVICE const char* c_str () const
    {
        return m_ptr;
    }

    OSL_HOSTDEVICE bool operator== (const device_string& other) const
    {
        return m_tag == other.m_tag;
    }

    OSL_HOSTDEVICE bool operator!= (const device_string& other) const
    {
        return m_tag != other.m_tag;
    }

    uint64_t    m_tag;
    const char* m_ptr;
};


OSL_NAMESPACE_EXIT
