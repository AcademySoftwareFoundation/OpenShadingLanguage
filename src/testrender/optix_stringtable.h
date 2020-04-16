// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#pragma once

#include <map>
#include <utility>

#include <OpenImageIO/ustring.h>
#include <OSL/oslexec.h>
#include "optix_compat.h"

OSL_NAMESPACE_ENTER


// The OptiXStringTable manages a block of CUDA device memory designated
// to hold all of the string constants that a shader might access during
// execution.
//
// Any string that needs to be visible on the device needs to be added using the
// addString function.
class OptiXStringTable {
public:
    OptiXStringTable(optix::Context ctx = nullptr);

    ~OptiXStringTable();

    // Allocate CUDA device memory for the raw string table and add the
    // "standard" strings declared in strdecls.h.
    void init (optix::Context ctx);

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
    uint64_t addString (OIIO::ustring str, OIIO::ustring var_name);

    void freetable();

private:
    // If a string has already been added to the table, return its offset in the
    // char array; otherwise, return -1.
    int getOffset (const std::string& str) const;

    // Free the previous allocation, allocate a new block of GPU memory of twice
    // the size, copy the string contents into the new allocation, and upate the
    // OptiX variables that hold the string addresses.
    void reallocTable();

    // A byte array containing the concatenation of all strings added to the
    // table, allocated in CUDA device memory. The hash value and length of each
    // string are stored in the 16 bytes preceding the raw characters.
    char*          m_ptr;

    // The size of the table in bytes.
    size_t         m_size;

    // The offset in the char array at which the next string will be added.
    int            m_offset;

    // A handle on the OptiX Context to use when creating global variables.
    optix::Context m_optix_ctx;

    // The memory offsets associated with each canonical string.
    std::map<OIIO::ustring,int>           m_offset_map;

    // The variable names associated with each canonical string.
    std::map<OIIO::ustring,OIIO::ustring> m_name_map;
};

OSL_NAMESPACE_EXIT
