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

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of string functions
/// such as format, concat, printf, etc.
///
/////////////////////////////////////////////////////////////////////////

#include <cstdarg>

#include <OpenImageIO/strutil.h>
#include <OpenImageIO/fmath.h>

#include "oslexec_pvt.h"

#include <boost/regex.hpp>

#define USTR(cstr) (*((ustring *)&cstr))

OSL_NAMESPACE_ENTER
namespace pvt {


// This symbol is strictly to force linkage of this file when building
// static library.
int opstring_cpp_dummy = 1;


// Heavy lifting of OSL regex operations.
OSL_SHADEOP int
osl_regex_impl2 (OSL::ShadingContext *ctx, ustring subject_,
                 int *results, int nresults, ustring pattern,
                 int fullmatch)
{
    const std::string &subject (subject_.string());
    boost::match_results<std::string::const_iterator> mresults;
    const boost::regex &regex (ctx->find_regex (pattern));
    if (nresults > 0) {
        std::string::const_iterator start = subject.begin();
        int res = fullmatch ? boost::regex_match (subject, mresults, regex)
                            : boost::regex_search (subject, mresults, regex);
        int *m = (int *)results;
        for (int r = 0;  r < nresults;  ++r) {
            if (r/2 < (int)mresults.size()) {
                if ((r & 1) == 0)
                    m[r] = mresults[r/2].first - start;
                else
                    m[r] = mresults[r/2].second - start;
            } else {
                m[r] = pattern.length();
            }
        }
        return res;
    } else {
        return fullmatch ? boost::regex_match (subject, regex)
                         : boost::regex_search (subject, regex);
    }
}

OSL_SHADEOP const char *
osl_format (const char* format_str, ...)
{
    va_list args;
    va_start (args, format_str);
    std::string s = Strutil::vformat (format_str, args);
    va_end (args);
    return ustring(s).c_str();
}


OSL_SHADEOP void
osl_printf (ShaderGlobals *sg, const char* format_str, ...)
{
    va_list args;
    va_start (args, format_str);
#if 0
    // Make super sure we know we are excuting LLVM-generated code!
    std::string newfmt = std::string("llvm: ") + format_str;
    format_str = newfmt.c_str();
#endif
    std::string s = Strutil::vformat (format_str, args);
    va_end (args);
    sg->context->shadingsys().message (s);
}


OSL_SHADEOP void
osl_error (ShaderGlobals *sg, const char* format_str, ...)
{
    va_list args;
    va_start (args, format_str);
    std::string s = Strutil::vformat (format_str, args);
    va_end (args);
    sg->context->shadingsys().error (s);
}


OSL_SHADEOP void
osl_warning (ShaderGlobals *sg, const char* format_str, ...)
{
    va_list args;
    va_start (args, format_str);
    std::string s = Strutil::vformat (format_str, args);
    va_end (args);
    sg->context->shadingsys().warning (s);
}



OSL_SHADEOP int
osl_split (const char *str, ustring *results, const char *sep,
           int maxsplit, int resultslen)
{
    maxsplit = OIIO::clamp (maxsplit, 0, resultslen);
    std::vector<std::string> splits;
    Strutil::split (USTR(str).string(), splits, USTR(sep).string(), maxsplit);
    int n = std::min (maxsplit, (int)splits.size());
    for (int i = 0;  i < n;  ++i)
        results[i] = ustring(splits[i]);
    return n;
}


} // end namespace pvt
OSL_NAMESPACE_EXIT
