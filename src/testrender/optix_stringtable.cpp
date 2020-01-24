/*
Copyright (c) 2009-2019 Sony Pictures Imageworks Inc., et al.
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
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOTSS
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

#include "optix_stringtable.h"

using OIIO::ustring;

OSL_NAMESPACE_ENTER


OptiXStringTable::OptiXStringTable(optix::Context ctx)
    : m_ptr (nullptr),
      m_size (1 << 16),
      m_offset (0),
      m_optix_ctx (ctx)
{
}



OptiXStringTable::~OptiXStringTable()
{
    freetable();
}



void
OptiXStringTable::freetable()
{
    if (m_ptr)
        OSL::cudaFree (m_ptr);
    m_ptr = nullptr;
}



void OptiXStringTable::init (OSL::optix::Context ctx OSL_MAYBE_UNUSED)
{
#ifdef OSL_USE_OPTIX
    OSL_ASSERT (! m_ptr && "StringTable should only be initialized once");
    m_optix_ctx = ctx;

    OSL_ASSERT ((m_optix_ctx->getEnabledDeviceCount() == 1) &&
            "Only one CUDA device is currently supported");

    OSL::cudaMalloc (reinterpret_cast<void**>(&m_ptr), (m_size));

    // Add the statically-declared strings to the table, and create OptiX
    // variables for them in the OSL::DeviceStrings namespace.
    //
    // The names of the variables created here must match the extern variables
    // declared in OSL/device_string.h for OptiX's variable scoping mechanisms
    // to work.

#define STRDECL(str,var_name)                                           \
    addString (ustring(str), ustring(OSL_NAMESPACE_STRING "::DeviceStrings::" #var_name));
#include <OSL/strdecls.h>
#undef STRDECL
#endif
}


uint64_t OptiXStringTable::addString (ustring str OSL_MAYBE_UNUSED,
                                      ustring var_name OSL_MAYBE_UNUSED)
{
#ifdef OSL_USE_OPTIX
    OSL_ASSERT (m_ptr && "StringTable has not been initialized");

    // The strings are laid out in the table as a struct:
    //
    //   struct TableRep {
    //       size_t len;
    //       size_t hash;
    //       char   str[len+1];
    //   };

    // Compute the size of the entry before adding it to the table 
    size_t size = sizeof(size_t) + sizeof(size_t) + str.size() + 1;
    if (((m_offset + size) >= m_size)) {
        reallocTable();
    }

    // It should be hard to trigger this assert, unless the table size is
    // very small and the string is very large.
    OSL_ASSERT (m_offset + size <= m_size && "String table allocation error");

    int offset = getOffset(str.string());
    if (offset < 0) {
        // Place the hash and length of the string before the characters
        size_t hash = str.hash();
        cudaMemcpy (m_ptr + m_offset, (void*)&hash, sizeof(size_t), cudaMemcpyHostToDevice);
        m_offset += sizeof(size_t);

        size_t len = str.length();
        cudaMemcpy (m_ptr + m_offset, (void*)&len, sizeof(size_t), cudaMemcpyHostToDevice);
        m_offset += sizeof(size_t);

        offset = m_offset;
        m_offset_map [str] = offset;
        m_name_map   [str] = var_name;

        // Copy the raw characters to the table
        cudaMemcpy (m_ptr + m_offset, str.c_str(), str.size() + 1, cudaMemcpyHostToDevice);
        m_offset += str.size() + 1;

        // Align the offset for the next entry to 8-byte boundaries
        m_offset = (m_offset + 0x7u) & ~0x7u;
    }

    uint64_t addr = reinterpret_cast<uint64_t>(m_ptr + offset);

    // Optionally create an OptiX variable for the string. It's not necessary to
    // create a variable for strings that do not appear by name in compiled code
    // (in either the OSL library functions or in the renderer).
    if (! var_name.empty()) {
        m_optix_ctx [var_name.string()]->setUserData (8, &addr);
    }

    return addr;
#else
    return 0;
#endif
}


int OptiXStringTable::getOffset (const std::string& str) const
{
    auto it = m_offset_map.find (ustring(str));
    return (it != m_offset_map.end()) ? it->second : -1;
}


void OptiXStringTable::reallocTable()
{
#ifdef OSL_USE_OPTIX
    OSL_ASSERT ((m_optix_ctx->getEnabledDeviceCount() == 1) &&
                "Only one CUDA device is currently supported");

    m_size *= 2;
    OSL::cudaFree (m_ptr);
    OSL::cudaMalloc (reinterpret_cast<void**>(&m_ptr), (m_size));

    // The offsets need to be recomputed
    m_offset = 0;
    m_offset_map.clear();

    // Add the string collection to the newly-allocated memory
    for (auto& entry : m_name_map) {
        addString (entry.first, entry.second);
    }
#endif
}

OSL_NAMESPACE_EXIT
