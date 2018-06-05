/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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

#pragma once

#include <map>
#include <memory>
#include <unordered_map>
#include <OpenImageIO/ustring.h>
#include <OSL/oslexec.h>

#include <cuda_runtime_api.h>
#include <optix_world.h>

OSL_NAMESPACE_ENTER


class StringTable {

    typedef std::pair<std::string, uint64_t> string_pair;

public:
    StringTable()
        : m_ptr          (nullptr),
          m_offset       (0),
          m_needs_update (false)
    {
    }

    ~StringTable()
    {
        if (m_ptr) {
            cudaFree (m_ptr);
        }
    }

    void init (optix::Context ctx)
    {
        m_optix_ctx = ctx;

        // Allocate 64KB in CUDA Unified Memory for the raw string table.
        // TODO: Make the table size a template parameter?
        cudaMallocManaged (reinterpret_cast<void**>(&m_ptr), (1<<16));

        // Create OptiX Buffer objects for the string table and an array of
        // offsets into that buffer
        m_str_buf     = m_optix_ctx->createBuffer (RT_BUFFER_INPUT);
        m_offsets_buf = m_optix_ctx->createBuffer (RT_BUFFER_INPUT, RT_FORMAT_INT);

        // Bind the string table to the OptiX Buffer
        m_str_buf->setDevicePointer (0, m_ptr);

        m_optix_ctx["str_table"  ]->set (m_str_buf);
        m_optix_ctx["str_offsets"]->set (m_offsets_buf);

        // Add all of the "standard" strings declared in strdecls.h to the
        // string table
        unsigned long long addr;
#define STRDECL(str,var_name)                                   \
        addr = addString (str);                                 \
        ctx["DeviceStrings::"#var_name]->setUserData(8, &addr);
#include <OSL/strdecls.h>
#undef STRDECL

        updateOffsets();
    }


    // Add a string to the table (if it hasn't already been added), and return
    // its global address.
    uint64_t addString (const std::string& str, const std::string& var_name="")
    {
        int offset = getOffset(str);
        if (offset < 0) {
            offset = m_offset;
            m_strings.emplace_back (str, m_offset);
            memcpy (m_ptr + m_offset, str.c_str(), str.size() + 1);
            m_offset += str.size() + 1;
            m_needs_update = true;
        }

        uint64_t addr = reinterpret_cast<uint64_t>(m_ptr + offset);

        if (! var_name.empty()) {
            m_string_vars.emplace_back (var_name, addr);
        }

        return addr;
    }


    // If a string has already been added to the table, return its offset in the
    // char array; otherwise, return -1.
    int getOffset (const std::string& str) const
    {
        auto it = std::find_if (
            m_strings.begin(), m_strings.end(),
            [&](const string_pair& val) {
                return val.first == str;
            });

        return (it != m_strings.end()) ? it->second : -1;
    }


    // Update the buffer of string offsets. This should be called before launch
    // if any string has been added since the last update.
    void updateOffsets()
    {
        if (m_needs_update) {
            m_offsets_buf->setSize(m_strings.size());

            int* offsets_ptr = reinterpret_cast<int*>(m_offsets_buf->map());
            for (size_t idx = 0; idx < m_strings.size(); ++idx) {
                memcpy (offsets_ptr + idx, &m_strings[idx].second, sizeof(int));
            }

            m_offsets_buf->unmap();
            m_needs_update = false;
        }
    }


    // For newly-added strings, we need to set the OptiX variables with the
    // addresses of the strings.
    void updateStringAddrs()
    {
        for (size_t idx = 0; idx < m_string_vars.size(); ++idx) {
            uint64_t addr = m_string_vars[idx].second;
            m_optix_ctx[m_string_vars[idx].first]->setUserData(sizeof(uint64_t),
                                                               &addr);
        }
    }

private:
    // A raw char array containing the concatenation of all strings added to the
    // table, allocated in CUDA Unified Memory.
    char*                    m_ptr;

    // The offset in the char array at which the next string will be added.
    int                      m_offset;

    // The collection of strings added so far, and their corresponding offsets.
    std::vector<string_pair> m_strings;


    // The collection of name/initializer pairs that needs to be updated before
    // the next launch.
    std::vector<string_pair> m_string_vars;

    // OptiXy stuff
    optix::Context           m_optix_ctx;
    optix::Buffer            m_str_buf;
    optix::Buffer            m_offsets_buf;

    // Dirty flag to signal when the offsets buffer needs to be updated.
    bool                     m_needs_update;
};


class OptixRenderer : public RendererServices
{
public:
    // Just use 4x4 matrix for transformations
    typedef Matrix44 Transformation;

    OptixRenderer () { }
    ~OptixRenderer () { }

    void init_string_table(optix::Context ctx)
    {
        m_str_table.init(ctx);
    }

    void update_string_table()
    {
        m_str_table.updateOffsets();
        m_str_table.updateStringAddrs();
    }

    virtual int supports (string_view feature) const
    {
        if (feature == "OptiX") {
            return true;
        }

        return false;
    }


    uint64_t register_string (const std::string& str, const std::string& var_name)
    {
        return m_str_table.addString(str, var_name);
    }


    // Function stubs
    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             TransformationPtr xform,
                             float time)
    {
        return 0;
    }

    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             ustring from, float time)
    {
        return 0;
    }

    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             TransformationPtr xform)
    {
        return 0;
    }

    virtual bool get_matrix (ShaderGlobals *sg, Matrix44 &result,
                             ustring from)
    {
        return 0;
    }

    virtual bool get_inverse_matrix (ShaderGlobals *sg, Matrix44 &result,
                                     ustring to, float time)
    {
        return 0;
    }


    virtual bool get_array_attribute (ShaderGlobals *sg, bool derivatives,
                                      ustring object, TypeDesc type, ustring name,
                                      int index, void *val )
    {
        return 0;
    }

    virtual bool get_attribute (ShaderGlobals *sg, bool derivatives, ustring object,
                                TypeDesc type, ustring name, void *val)
    {
        return 0;
    }

    virtual bool get_userdata (bool derivatives, ustring name, TypeDesc type,
                               ShaderGlobals *sg, void *val)
    {
        return 0;
    }


    StringTable m_str_table;
};


OSL_NAMESPACE_EXIT
