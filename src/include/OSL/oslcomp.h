// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>

OSL_NAMESPACE_ENTER


namespace pvt {
class OSLCompilerImpl;
}



class OSLCOMPPUBLIC OSLCompiler {
public:
    OSLCompiler(ErrorHandler* errhandler = NULL);
    ~OSLCompiler();

    /// Compile the given file, using the list of command-line options. The
    /// stdoslpath parameter provides a custom path for finding stdosl.h.
    /// Return true if ok, false if the compile failed.
    bool compile(string_view filename, const std::vector<std::string>& options,
                 string_view stdoslpath = string_view());

    /// Compile the given source code buffer, using the list of command-line
    /// options, placing the resulting "oso" in osobuffer. The stdoslpath
    /// parameter provides a custom path for finding stdosl.h. The filename
    /// optionally provides a name for the buffer, used for error reporting
    /// (the compile() from file method would have used the name of the
    /// actual file for this purpose). Return true if ok, false if the
    /// compile failed.
    bool compile_buffer(string_view sourcecode, std::string& osobuffer,
                        const std::vector<std::string>& options,
                        string_view stdoslpath = string_view(),
                        string_view filename   = string_view());

    /// Return the name of our compiled output (must be called after
    /// compile()).
    string_view output_filename() const;

private:
    pvt::OSLCompilerImpl* m_impl;
};



OSL_NAMESPACE_EXIT
