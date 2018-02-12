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


The code in this file is based somewhat on code released by NVIDIA as
part of Gelato (specifically, gsoargs.h).  That code had the following
copyright notice:

   Copyright 2004 NVIDIA Corporation.  All Rights Reserved.

and was distributed under BSD licensing terms identical to the
Sony Pictures Imageworks terms, above.
*/


#pragma once

#include <string>
#include <vector>

#include <OSL/oslconfig.h>

OSL_NAMESPACE_ENTER

class ShaderGroup;    // opaque class for now

namespace pvt {
    class OSOReaderQuery;   // Just so OSLQuery can friend OSLReaderQuery
};



/// <doc OSLQuery>
///                  **OSLQuery Documentation**
///
///
/// Introduction and Tutorial
/// =========================
///
/// `OSLQuery` is a class that lets an application interrogate a
/// compiled shader for information about its parameters.
///
/// The shader may be an already-compiled shader file on disk (i.e. a
/// `.oso` file), or the `.oso` equivalent in a string, or the binary
/// representation used by the OSL `ShaderSystem` runtime (as a pointer
/// to a `ShaderGroup`).  For example,
///
/// ~~~
///     OSLQuery oslquery ("polished_oak");
/// ~~~
///
/// It's then easy to retrieve a specific parameter:
///
/// ~~~
///     int nparams = oslquery.nparams();  // number of params
///
///     const OSLQuery::Parameter *param;
///     p = oslquery.getparam (i);            // by index (0..nparams-1)
///     p = oslquery.getparam ("woodcolor");  // by name
/// ~~~
///
/// And the `Parameter` structure will hold all the information you need
/// about that paramter. For example:
///
/// ~~~
///     std::cout << "Parameter " << p->name
///               << " is type " << p->type << "\n"
/// ~~~
///
/// You can find out if the parameter is a closure, an output parameter,
/// etc. You can also find out its default values, which are stored in
/// vector fields `idefault`, `fdefault`, and `sdefault` depending on
/// whether the types is based on int, float, or string, respectively.
///
///
/// OSLQuery class API Reference
/// ============================

class OSLQUERYPUBLIC OSLQuery {
public:
    /// Parameter helper structure
    /// --------------------------
    /// `Parameter` holds all the information about a single shader
    /// parameter.
    /// <code>
    struct Parameter {
        ustring name;                    //< name
        TypeDesc type;                   //< data type
        bool isoutput = false;           //< is it an output param?
        bool validdefault = false;       //< false if there's no default val
        bool varlenarray = false;        //< is it a varying-length array?
        bool isstruct = false;           //< is it a structure?
        bool isclosure = false;          //< is it a closure?
        void *data = nullptr;            //< pointer to data
        std::vector<int> idefault;       //< default int values
        std::vector<float> fdefault;     //< default float values
        std::vector<ustring> sdefault;   //< default string values
        std::vector<ustring> spacename;  //< space name for matrices and
                                         //<  triples, for each array elem.
        std::vector<ustring> fields;     //< Names of this struct's fields
        ustring structname;              //< Name of the struct
        std::vector<Parameter> metadata; //< Meta-data about the param

        Parameter () {}
        Parameter (const Parameter& src);
        Parameter (Parameter&& src);
        const Parameter& operator= (const Parameter &);
        const Parameter& operator= (Parameter &&);
    };
    /// </code>

    /// OSLQuery methods
    /// ----------------

    OSLQuery ();
    ///< Construct an uninitialized OSLQuery. It will not hold any
    /// information about a shader until `open()` is called.

    OSLQuery (string_view shadername,
              string_view searchpath = string_view()) {
        open (shadername, searchpath);
    }
    ///< Construct an OSLQuery and open a compiled shader from a disk file.
    /// The `shadername` may be either the name of the `.oso` file, or the
    /// name of the shader. The optional `searchpath` paramter gives a
    /// colon-separated list of directories to search for compiled shaders.

    OSLQuery (const ShaderGroup *group, int layernum) {
        init (group, layernum);
    }
    ///< Construct an OSLQuery and initialize it with an existing
    /// `ShaderGroup` (which must have been built using the `ShadingSystem`
    /// runtime API for OSL.

    ~OSLQuery ();
    ///< Clean up and destruct the `OSLQuery`.

    bool open (string_view shadername,
               string_view searchpath = string_view());
    ///< For an uninitialized `OSLQuery` object, initialize it with info on
    /// the named shader with optional searcphath.  Return true for success,
    /// false if the shader could not be found or opened properly.

    bool open_bytecode (string_view buffer);
    ///< Get info on the shader from it's compiled bytecode (i.e., like the
    /// contents of an `.oso` file, but in a string).  Return `true` for
    /// success, false if the shader could not be found or opened properly.
    ///
    /// This is meant to be called from an app which caches bytecodes from
    /// it's own side and wants to get shader info on runtime without
    /// creating a temporary file.

    bool init (const ShaderGroup *group, int layernum);
    ///< Meant to be called at runtime from an app with a full ShadingSystem,
    /// fill out an OSLQuery structure for the given layer of the group.
    /// This is much faster than using open() to read it from an oso file on
    /// disk.

    const ustring shadertype (void) const { return m_shadertypename; }
    ///< Return the shader type: "surface", "displacement", "volume",
    /// "light", or "shader" (for generic shaders).

    const ustring shadername (void) const { return m_shadername; }
    ///< Get the name of the shader.

    size_t nparams (void) const { return (int) m_params.size(); }
    ///< How many parameters does the shader have

    const Parameter *getparam (size_t i) const;
    const Parameter *getparam (const std::string &name) const;
    const Parameter *getparam (ustring name) const;
    ///< Retrieve a parameter, either by index or by name. Return nullptr if
    /// the index is out of range, or if the named parameter is not found.

    const std::vector<Parameter> &metadata (void) const { return m_meta; }
    ///< Retrieve a reference to the metadata about the shader.

    ///> Return error string, empty if there was no error, and reset the
    /// error string.
    std::string geterror (bool clear_error = true) {
        std::string e = m_error;
        if (clear_error)
            m_error.clear ();
        return e;
    }

private:
    ustring m_shadername;              //< Name of shader
    ustring m_shadertypename;          //< Type of shader
    mutable std::string m_error;       //< Error message
    std::vector<Parameter> m_params;   //< Params to the shader
    std::vector<Parameter> m_meta;     //< Meta-data about the shader
    friend class pvt::OSOReaderQuery;

    // Internal error reporting routine, with printf-like arguments.
    template<typename... Args>
    inline void error (string_view fmt, const Args&... args) const {
        append_error(OIIO::Strutil::format (fmt, args...));
    }

    void append_error (const std::string& message) const {
        if (m_error.size())
            m_error += '\n';
        m_error += message;
    }
};




////////// Implementation

inline const OSLQuery::Parameter*
OSLQuery::getparam (size_t i) const
{
    if (i >= nparams())
        return nullptr;
    return &(m_params[i]);
}


inline const OSLQuery::Parameter*
OSLQuery::getparam (const std::string &name) const
{
    for (size_t i = 0;  i < nparams();  ++i)
        if (m_params[i].name == name)
            return &(m_params[i]);
    return nullptr;
}


inline const OSLQuery::Parameter*
OSLQuery::getparam (ustring name) const
{
    for (size_t i = 0;  i < nparams();  ++i)
        if (m_params[i].name == name)
            return &(m_params[i]);
    return nullptr;
}


// more documentation
/// <inc oslinfo_source>

OSL_NAMESPACE_EXIT
