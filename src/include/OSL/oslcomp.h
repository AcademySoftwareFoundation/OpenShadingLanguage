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
*/

#pragma once

#include <OSL/oslconfig.h>

OSL_NAMESPACE_ENTER


namespace pvt {
    class OSLCompilerImpl;
}



class OSLCOMPPUBLIC OSLCompiler {
public:
    OSL_DEPRECATED("Directly construct or new an OSLCompiler")
    static OSLCompiler *create ();

    OSLCompiler (ErrorHandler *errhandler=NULL);
    ~OSLCompiler ();

    /// Compile the given file, using the list of command-line options.
    /// Return true if ok, false if the compile failed.
    bool compile (string_view filename,
                  const std::vector<std::string> &options,
                  string_view stdoslpath = string_view());

    /// Compile the given source code buffer, using the list of command-line
    /// options, placing the resulting "oso" in osobuffer. Return true if
    /// ok, false if the compile failed.
    bool compile_buffer (string_view sourcecode, std::string &osobuffer,
                         const std::vector<std::string> &options,
                         string_view stdoslpath = string_view());

    /// Return the name of our compiled output (must be called after
    /// compile()).
    string_view output_filename () const;

private:
    pvt::OSLCompilerImpl *m_impl;
};



OSL_NAMESPACE_EXIT
