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


// A device_string is a special string representation intended for use on GPUs,
// which do not readily support ustrings.
//
// Two device_strings can be tested for equality simply by comparing their m_tag
// fields; m_chars is available for cases where the actual string contents are
// required (e.g., for printf).
//
// For comparisons between device_strings to be valid, OSL library code
// (shadeops, noise gen, etc.), compiled shaders, and renderer application code
// must all agree on the tag for each string.
//
// The other design goal is that compiled shaders containing strings can be
// cached, because tags are consistent from run to run.
//
// For "standard" strings (in particular, members of the 'Strings' namespace in
// liboslexec/oslexec_pvt.h), we declare fixed tags to make it easier to write
// CUDA code that uses them.
//
// Strings that do not need to be shared between liboslexec and the renderer,
// but which have special significance in the renderer (such as custom closure
// parameters), can be registered with the shading system at runtime via
// register_string_tag().
//
// For other strings, the tag is the ustring hash.
//
// A pre-declared device_string is listed:
//
// 1) In the StringTags enum in this file. This file can be included in your
//    OptiX/CUDA renderer to make the tags available.
//
// 2) In the definitions in liboslexec/device_string.cpp. These are linked with
//    the other 'shadeops' sources (opnoise.cpp, etc) and made available as
//    global symbols to executing shaders.
//
// 3) In ShadingSystemImpl::setup_string_tags(), which registers the enum value
//    as the tag for that ustring. This lets libsolexec create a device_string
//    with the appropriate tag during shader compilation.


OSL_NAMESPACE_ENTER


struct device_string {
    OSL_HOSTDEVICE uint64_t tag () const
    {
        return m_tag;
    }

    OSL_HOSTDEVICE const char* c_str () const
    {
        return m_chars;
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
    const char* m_chars;
};


enum StringTags: uint64_t {
#define STRDECL(str,var_name) \
    var_name,
#include <OSL/strdecls.h>
#undef STRDECL
    NUM_TAGS,
    UNKNOWN_STRING = ~0u
};


OSL_NAMESPACE_EXIT
