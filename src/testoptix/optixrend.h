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
#include <OpenImageIO/ustring.h>
#include <OSL/oslexec.h>

#include <cuda_runtime_api.h>
#include <optix_world.h>


OSL_NAMESPACE_ENTER


// TODO: Make sure this works with multiple GPUs
class StringTable {
public:
    StringTable() { }
    ~StringTable() { }


    // Allocate CUDA Unified Memory for the raw string table and add the
    // "standard" strings declared in strdecls.h.
    void init (optix::Context ctx)
    {
        m_optix_ctx = ctx;

#define STRDECL(str,var_name)                           \
        addString (str, "DeviceStrings::"#var_name);
#include <OSL/strdecls.h>
#undef STRDECL

        ustring test_string("this is my test string");
        uint64_t addr = (uint64_t)test_string.c_str();
        m_optix_ctx["test_string_addr"]->setUserData (8, &addr);

        // Make sure that ustrings are being stored in Unified Memory
        cudaPointerAttributes attr;
        cudaPointerGetAttributes (&attr, test_string.c_str());

        // Shotgun blast to make sure ustrings are being placed in UM
        ASSERT (attr.isManaged && attr.devicePointer && attr.hostPointer &&
                "ustrings are not being allocated in CUDA Unified Memory!");
    }


    // Add a string to the table (if it hasn't already been added), and return
    // its global address.
    //
    // Also, create an OptiX variable to hold the address of each string
    // variable.
    uint64_t addString (const std::string& str, const std::string& var_name="")
    {
        ustring ustr (str);
        uint64_t addr = reinterpret_cast<uint64_t>(ustr.c_str());

        if (! var_name.empty()) {
            m_optix_ctx[var_name]->setUserData (8, &addr);
#if 0
            std::cout << "Creating var for " << str
                      << " with name " << var_name
                      << " and addr " << (void*) addr << std::endl;
#endif
        }

        return addr;
    }

private:
    // The collection of strings added so far, and their addresses.
    std::map<ustring,uint64_t> m_string_map;

    // A handle on the OptiX Context to use when creating global variables.
    optix::Context             m_optix_ctx;
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
        m_str_table.init (ctx);
    }

    uint64_t register_string (const std::string& str, const std::string& var_name)
    {
        return m_str_table.addString (str, var_name);
    }

    virtual int supports (string_view feature) const
    {
        if (feature == "OptiX") {
            return true;
        }

        return false;
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
