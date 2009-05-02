/*****************************************************************************
 *
 *             Copyright (c) 2009 Sony Pictures Imageworks, Inc.
 *                            All rights reserved.
 *
 *  This material contains the confidential and proprietary information
 *  of Sony Pictures Imageworks, Inc. and may not be disclosed, copied or
 *  duplicated in any form, electronic or hardcopy, in whole or in part,
 *  without the express prior written consent of Sony Pictures Imageworks,
 *  Inc. This copyright notice does not imply publication.
 *
 *****************************************************************************/

////////////////////////////////////////////////////////////////////////////
// This code is based somewhat on code released by NVIDIA as part of
// Gelato (specifically, gsoargs.h).  That code had the following
// copyright notice:
//
// Copyright 2004 NVIDIA Corporation.  All Rights Reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// * Neither the name of NVIDIA nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// (This is the Modified BSD License)
/////////////////////////////////////////////////////////////////////////////


#ifndef OSLQUERY_H
#define OSLQUERY_H

#include <string>
#include <vector>

#include "OpenImageIO/export.h"
#include "OpenImageIO/typedesc.h"


namespace OSL {

namespace pvt {
    class OSOReaderQuery;   // Just so OSLQuery can friend OSLReaderQuery
};



class DLLPUBLIC OSLQuery {
public:
    /// Parameter holds all the information about a single shader parameter.
    ///
    struct Parameter {
        std::string name;                ///< name
        TypeDesc type;                   ///< data type
        bool isoutput;                   ///< is it an output param?
        bool validdefault;               ///< false if there's no default val
        bool varlenarray;                ///< is it a varying-length array?
        bool isstruct;                   ///< is it a structure?
        bool isclosure;                  ///< is it a closure?
        std::vector<int> idefault;       ///< default int values
        std::vector<float> fdefault;     ///< default float values
        std::vector<std::string> sdefault;   ///< default string values
        std::vector<std::string> spacename;  ///< space name for matrices and
                                             ///<  triples, for each array elem.
        std::vector<Parameter> metadata; ///< Meta-data about the param
        Parameter ()
            : isoutput(false), validdefault(false), varlenarray(false),
              isstruct(false), isclosure(false)
        { }
    };

    OSLQuery ();
    ~OSLQuery ();

    /// Get info on the named shader with optional searcphath.  Return
    /// true for success, false if the shader could not be found or
    /// opened properly.
    bool open (const std::string &shadername,
               const std::string &searchpath=std::string());

    /// Return the shader type: "surface", "displacement", "volume",
    /// "light", or "shader" (for generic shaders).
    const std::string &shadertype (void) const { return m_shadertype; }

    /// Get the name of the shader.
    ///
    const std::string &shadername (void) const { return m_shadername; }

    /// How many parameters does the shader have?
    ///
    size_t nparams (void) const { return (int) m_params.size(); }

    /// Retrieve a parameter, either by index or by name.  Return NULL if the
    /// index is out of range, or if the named parameter is not found.
    const Parameter *getparam (size_t i) const {
        if (i < 0 || i >= nparams())
            return NULL;
        return &(m_params[i]);
    }
    const Parameter *getparam (const std::string &name) const {
        for (size_t i = 0;  i < nparams();  ++i)
            if (m_params[i].name == name)
                return &(m_params[i]);
        return NULL;
    }

    /// Retrieve a reference to the metadata about the shader.
    ///
    const std::vector<Parameter> &metadata (void) const { return m_meta; }

    /// Return error string, empty if there was no error, and reset the
    /// error string.
    std::string error (void) {
        std::string e = m_error;
        m_error.clear ();
        return e;
    }

private:
    std::string m_shadername;          ///< Name of shader
    std::string m_shadertype;          ///< Type of shader
    std::string m_error;               ///< Error message
    std::vector<Parameter> m_params;   ///< Params to the shader
    std::vector<Parameter> m_meta;     ///< Meta-data about the shader
    friend class pvt::OSOReaderQuery;
};


}; /* end namespace OSL */


#endif /* OSLQUERY_H */
