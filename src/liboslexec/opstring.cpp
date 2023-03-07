// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of string functions
/// such as format, concat, printf, etc.
///
/////////////////////////////////////////////////////////////////////////

#include <OSL/rs_free_function.h>
#include <cstdarg>


#include <OpenImageIO/fmath.h>
#include <OpenImageIO/strutil.h>


#include "oslexec_pvt.h"


OSL_NAMESPACE_ENTER
namespace pvt {


// Only define 2-arg version of concat, sort it out upstream
OSL_SHADEOP const char*
osl_concat_sss(const char* s, const char* t)
{
    size_t sl  = USTR(s).length();
    size_t tl  = USTR(t).length();
    size_t len = sl + tl;
    std::unique_ptr<char[]> heap_buf;
    char local_buf[256];
    char* buf = local_buf;
    if (len > sizeof(local_buf)) {
        heap_buf.reset(new char[len]);
        buf = heap_buf.get();
    }
    memcpy(buf, s, sl);
    memcpy(buf + sl, t, tl);
    return ustring(buf, len).c_str();
}

OSL_SHADEOP int
osl_strlen_is(const char* s)
{
    return (int)USTR(s).length();
}

OSL_SHADEOP int
osl_hash_is(const char* s)
{
    return (int)USTR(s).hash();
}

OSL_SHADEOP int
osl_getchar_isi(const char* str, int index)
{
    return str && unsigned(index) < USTR(str).length() ? str[index] : 0;
}


OSL_SHADEOP int
osl_startswith_iss(const char* s_, const char* substr_)
{
    ustring substr(USTR(substr_));
    size_t substr_len = substr.length();
    if (substr_len == 0)  // empty substr always matches
        return 1;
    ustring s(USTR(s_));
    size_t s_len = s.length();
    if (substr_len > s_len)  // longer needle than haystack can't
        return 0;            // match (including empty s)
    return strncmp(s.c_str(), substr.c_str(), substr_len) == 0;
}

OSL_SHADEOP int
osl_endswith_iss(const char* s_, const char* substr_)
{
    ustring substr(USTR(substr_));
    size_t substr_len = substr.length();
    if (substr_len == 0)  // empty substr always matches
        return 1;
    ustring s(USTR(s_));
    size_t s_len = s.length();
    if (substr_len > s_len)  // longer needle than haystack can't
        return 0;            // match (including empty s)
    return strncmp(s.c_str() + s_len - substr_len, substr.c_str(), substr_len)
           == 0;
}

OSL_SHADEOP int
osl_stoi_is(const char* str)
{
    return str ? Strutil::from_string<int>(str) : 0;
}

OSL_SHADEOP float
osl_stof_fs(const char* str)
{
    return str ? Strutil::from_string<float>(str) : 0.0f;
}

OSL_SHADEOP const char*
osl_substr_ssii(const char* s_, int start, int length)
{
    ustring s(USTR(s_));
    int slen = int(s.length());
    if (slen == 0)
        return NULL;  // No substring of empty string
    int b = start;
    if (b < 0)
        b += slen;
    b = Imath::clamp(b, 0, slen);
    return ustring(s, b, Imath::clamp(length, 0, slen)).c_str();
}


OSL_SHADEOP int
osl_regex_impl(void* sg_, const char* subject_, void* results, int nresults,
               const char* pattern, int fullmatch)
{
    ShaderGlobals* sg   = (ShaderGlobals*)sg_;
    ShadingContext* ctx = sg->context;
    const std::string& subject(ustring::from_unique(subject_).string());
    std::match_results<std::string::const_iterator> mresults;
    const std::regex& regex(ctx->find_regex(USTR(pattern)));
    if (nresults > 0) {
        std::string::const_iterator start = subject.begin();
        int res = fullmatch ? std::regex_match(subject, mresults, regex)
                            : std::regex_search(subject, mresults, regex);
        int* m  = (int*)results;
        for (int r = 0; r < nresults; ++r) {
            if (r / 2 < (int)mresults.size()) {
                if ((r & 1) == 0)
                    m[r] = mresults[r / 2].first - start;
                else
                    m[r] = mresults[r / 2].second - start;
            } else {
                m[r] = USTR(pattern).length();
            }
        }
        return res;
    } else {
        return fullmatch ? std::regex_match(subject, regex)
                         : std::regex_search(subject, regex);
    }
}

// Shims to convert llvm gen to rs free function C++ parameter types
// and forward on calls to re free functions.
// TODO: moveto opgen.cpp
OSL_RSOP void
osl_gen_errorfmt(/*OSL::ShaderGlobals* sg*/ OpaqueExecContextPtr exec_ctx, 
            OSL::ustringhash_pod fmt_specification, 
            int32_t arg_count, 
            const EncodedType *argTypes, 
            uint32_t argValuesSize, 
            uint8_t *argValues)
{

     OSL::ustringhash rs_fmt_specification = OSL::ustringhash_from(fmt_specification);
     rs_errorfmt(exec_ctx, rs_fmt_specification, 
           arg_count, 
             argTypes, 
             argValuesSize, 
             argValues);
 }


OSL_SHADEOP const char*
osl_format(const char* format_str, ...)
{
    va_list args;
    va_start(args, format_str);
    std::string s = Strutil::vsprintf(format_str, args);
    va_end(args);
    return ustring(s).c_str();
}

//#if 0
//Intent is to remove this and call the renderer service function directly in future PR
OSL_SHADEOP void
osl_printf(ShaderGlobals* sg, const char* format_str, ...)
{
    // Until llvm_gen directly calls rs_printfmt, 
    // we will need to perform the formating here 
    // as renderer only accepts the fmt specification
    va_list args;
    va_start(args, format_str);
    std::string s = Strutil::vsprintf(format_str, args);
    va_end(args);
    osl_printfmt(sg, OSL::ustringhash(s));
}

//Intent is to remove this and call the renderer service function directly in future PR
OSL_SHADEOP void
osl_error(ShaderGlobals* sg, const char* format_str, ...)
{
    // Until llvm_gen directly calls rs_errorfmt, 
    // we will need to perform the formating here 
    // as renderer only accepts the fmt specification
    va_list args;
    va_start(args, format_str);
    std::string s = Strutil::vsprintf(format_str, args);
    va_end(args);
    osl_errorfmt(sg, OSL::ustringhash(s));

}

//Intent is to remove this and call the renderer service function directly in future PR
OSL_SHADEOP void
osl_warning(ShaderGlobals* sg, const char* format_str, ...)
{
    ShadingStateUniform* ssu = (ShadingStateUniform*)sg->shadingStateUniform;
    if(ssu->m_allow_warnings){
        // Until llvm_gen directly calls rs_warningfmt, 
        // we will need to perform the formating here 
        // as renderer only accepts the fmt specification
        va_list args;
        va_start(args, format_str);
        std::string s = Strutil::vsprintf(format_str, args);
        va_end(args);
        osl_warningfmt(sg, OSL::ustringhash(s));
    }
}


//Intent is to remove this and call the renderer service function directly in future PR
OSL_SHADEOP void
osl_fprintf(ShaderGlobals* sg, const char* filename, const char* format_str,
            ...)
{
    // Until llvm_gen directly calls rs_filefmt, 
    // we will need to perform the formating here 
    // as renderer only accepts the fmt specification
    va_list args;
    va_start(args, format_str);
    std::string s = Strutil::vsprintf(format_str, args);
    va_end(args);
    osl_filefmt(sg,OSL::ustringhash(filename), OSL::ustringhash(s));
}
//#endif



OSL_SHADEOP int
osl_split(const char* str, ustring* results, const char* sep, int maxsplit,
          int resultslen)
{
    maxsplit = OIIO::clamp(maxsplit, 0, resultslen);
    std::vector<std::string> splits;
    Strutil::split(USTR(str).string(), splits, USTR(sep).string(), maxsplit);
    int n = std::min(maxsplit, (int)splits.size());
    for (int i = 0; i < n; ++i)
        results[i] = ustring(splits[i]);
    return n;
}


}  // end namespace pvt
OSL_NAMESPACE_EXIT
