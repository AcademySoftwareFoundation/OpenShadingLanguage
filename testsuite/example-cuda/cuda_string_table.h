// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#pragma once

#include <map>
#include <unordered_map>
#include <utility>

#include <cuda_runtime_api.h>

#include <OSL/oslexec.h>
#include <OpenImageIO/ustring.h>

typedef std::unordered_map<OIIO::ustring, int64_t, OIIO::ustringHash>
    StringTableMap;

// The CudaStringTable manages a block of CUDA device memory designated
// to hold all of the string constants that a shader might access during
// execution.
//
// Any string that needs to be visible on the device needs to be added using the
// addString function.
//
// Note: this is basically just a copy of the one from testrender
class CudaStringTable {
public:
    CudaStringTable() : m_ptr(nullptr), m_size(1 << 16), m_offset(0) {}

    ~CudaStringTable() { freetable(); }

    // Allocate CUDA device memory for the raw string table and add the
    // "standard" strings declared in strdecls.h.
    void init()
    {
        OSL_ASSERT(!m_ptr && "StringTable should only be initialized once");

        cudaMalloc(reinterpret_cast<void**>(&m_ptr), (m_size));

        // Add the statically-declared strings to the table, and create OptiX
        // variables for them in the OSL::DeviceStrings namespace.
        //
        // The names of the variables created here must match the extern
        // variables declared in OSL/device_string.h for OptiX's variable
        // scoping mechanisms to work.

#define STRDECL(str, var_name)                   \
    addString(OIIO::ustring(str),                \
              OIIO::ustring(OSL_NAMESPACE_STRING \
                            "::DeviceStrings::" #var_name));
#include <OSL/strdecls.h>
#undef STRDECL
    }

    const StringTableMap& contents() const { return m_addr_table; }

    // Add a string to the table (if it hasn't already been added), and return
    // its address in device memory. Also, create an OptiX variable to hold the
    // address of each string variable, through which the strings will be
    // accessed during execution.
    //
    // Creating variables for the strings -- rather than using the string
    // addresses directly -- is necessary for keeping the generated PTX stable
    // from run to run, regardless of the order in which strings are added to
    // the table. This helps make the PTX more cacheable.
    //
    // This function should not be called when a kernel that could potentially
    // access the table is running, since it has the potential to invalidate
    // pointers if reallocation is triggered.
    uint64_t addString(OIIO::ustring str, OIIO::ustring var_name)
    {
        OSL_ASSERT(m_ptr && "StringTable has not been initialized");

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
        OSL_ASSERT(m_offset + size <= m_size
                   && "String table allocation error");

        int offset = getOffset(str.string());
        if (offset < 0) {
            // Place the hash and length of the string before the characters
            size_t hash = str.hash();
            cudaMemcpy(m_ptr + m_offset, (void*)&hash, sizeof(size_t),
                       cudaMemcpyHostToDevice);
            m_offset += sizeof(size_t);

            size_t len = str.length();
            cudaMemcpy(m_ptr + m_offset, (void*)&len, sizeof(size_t),
                       cudaMemcpyHostToDevice);
            m_offset += sizeof(size_t);

            offset            = m_offset;
            m_offset_map[str] = offset;
            m_name_map[str]   = var_name;

            // Copy the raw characters to the table
            cudaMemcpy(m_ptr + m_offset, str.c_str(), str.size() + 1,
                       cudaMemcpyHostToDevice);
            m_offset += str.size() + 1;

            // Align the offset for the next entry to 8-byte boundaries
            m_offset = (m_offset + 0x7u) & ~0x7u;
        } else if (!var_name.empty()) {
            // update what str points to
            m_name_map[str] = var_name;
        }

        uint64_t addr = reinterpret_cast<uint64_t>(m_ptr + offset);

        // Optionally create an OptiX variable for the string. It's not
        // necessary to create a variable for strings that do not appear by name
        // in compiled code (in either the OSL library functions or in the
        // renderer).
        if (!var_name.empty())
            m_addr_table[var_name] = addr;
        return addr;
    }

    void freetable()
    {
        if (m_ptr)
            cudaFree(m_ptr);
        m_ptr = nullptr;
    }

private:
    // If a string has already been added to the table, return its offset in the
    // char array; otherwise, return -1.
    int getOffset(const std::string& str) const
    {
        auto it = m_offset_map.find(OIIO::ustring(str));
        return (it != m_offset_map.end()) ? it->second : -1;
    }

    // Free the previous allocation, allocate a new block of GPU memory of twice
    // the size, copy the string contents into the new allocation, and update
    // the OptiX variables that hold the string addresses.
    void reallocTable()
    {
        m_size *= 2;
        cudaFree(m_ptr);
        cudaMalloc(reinterpret_cast<void**>(&m_ptr), (m_size));

        // The offsets need to be recomputed
        m_offset = 0;
        m_offset_map.clear();

        // Add the string collection to the newly-allocated memory
        for (auto& entry : m_name_map) {
            addString(entry.first, entry.second);
        }
    }

    // A byte array containing the concatenation of all strings added to the
    // table, allocated in CUDA device memory. The hash value and length of each
    // string are stored in the 16 bytes preceding the raw characters.
    char* m_ptr;

    // The size of the table in bytes.
    size_t m_size;

    // The offset in the char array at which the next string will be added.
    int m_offset;

    // The memory offsets associated with each canonical string.
    std::map<OIIO::ustring, int> m_offset_map;

    // The variable names associated with each canonical string.
    std::map<OIIO::ustring, OIIO::ustring> m_name_map;

    StringTableMap m_addr_table;
};
