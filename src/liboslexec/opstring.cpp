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

#include <cstdarg>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/fmath.h>
#include <OpenImageIO/strutil.h>

#include <OSL/fmt_util.h>

#include "oslexec_pvt.h"


OSL_NAMESPACE_BEGIN
namespace pvt {


// Only define 2-arg version of concat, sort it out upstream
OSL_SHADEOP ustringhash_pod
osl_concat_sss(ustringhash_pod s_, ustringhash_pod t_)
{
    ustringhash s_uh = ustringhash_from(s_);
    ustringhash t_uh = ustringhash_from(t_);

    ustring s = ustring_from(s_uh);
    ustring t = ustring_from(t_uh);

    size_t sl  = s.size();
    size_t tl  = t.size();
    size_t len = sl + tl;
    std::unique_ptr<char[]> heap_buf;
    char local_buf[256];
    char* buf = local_buf;
    if (len > sizeof(local_buf)) {
        heap_buf.reset(new char[len]);
        buf = heap_buf.get();
    }
    memcpy(buf, s.c_str(), sl);
    memcpy(buf + sl, t.c_str(), tl);
    ustring result(buf, len);

    return result.uhash().hash();
}

OSL_SHADEOP int
osl_strlen_is(ustringhash_pod s_)
{
    auto s = ustring_from(s_);
    return (int)s.length();
}

OSL_SHADEOP int
osl_hash_is(ustringhash_pod s_)
{
    auto s = ustring_from(s_);
    return (int)s.hash();
}

OSL_SHADEOP int
osl_getchar_isi(ustringhash_pod str_, int index)
{
    auto str = ustring_from(str_);
    return str.data() && unsigned(index) < str.length() ? str[index] : 0;
}


OSL_SHADEOP int
osl_startswith_iss(ustringhash_pod s_, ustringhash_pod substr_)
{
    auto substr       = ustring_from(substr_);
    size_t substr_len = substr.length();
    if (substr_len == 0)  // empty substr always matches
        return 1;
    auto s       = ustring_from(s_);
    size_t s_len = s.length();
    if (substr_len > s_len)  // longer needle than haystack can't
        return 0;            // match (including empty s)
    return strncmp(s.c_str(), substr.c_str(), substr_len) == 0;
}

OSL_SHADEOP int
osl_endswith_iss(ustringhash_pod s_, ustringhash_pod substr_)
{
    auto substr       = ustring_from(substr_);
    size_t substr_len = substr.length();
    if (substr_len == 0)  // empty substr always matches
        return 1;
    auto s       = ustring_from(s_);
    size_t s_len = s.length();
    if (substr_len > s_len)  // longer needle than haystack can't
        return 0;            // match (including empty s)
    return strncmp(s.c_str() + s_len - substr_len, substr.c_str(), substr_len)
           == 0;
}

OSL_SHADEOP int
osl_stoi_is(ustringhash_pod str_)
{
    auto str = ustring_from(str_);
    return str.data() ? Strutil::from_string<int>(str) : 0;
}

OSL_SHADEOP float
osl_stof_fs(ustringhash_pod str_)
{
    auto str = ustring_from(str_);
    return str.data() ? Strutil::from_string<float>(str) : 0.0f;
}

OSL_SHADEOP ustringhash_pod
osl_substr_ssii(ustringhash_pod s_, int start, int length)
{
    auto s   = ustring_from(s_);
    int slen = int(s.length());
    if (slen == 0)
        return ustringhash_pod();  // No substring of empty string
    int b = start;
    if (b < 0)
        b += slen;
    b = Imath::clamp(b, 0, slen);
    return ustringhash_from(ustring(s, b, Imath::clamp(length, 0, slen))).hash();
}


OSL_SHADEOP int
osl_regex_impl(void* sg_, ustringhash_pod subject_, void* results, int nresults,
               ustringhash_pod pattern_, int fullmatch)
{
    ShaderGlobals* sg        = (ShaderGlobals*)sg_;
    ShadingContext* ctx      = sg->context;
    ustringhash subject_hash = ustringhash_from(subject_);
    ustring subject_ustr     = ustring_from(subject_hash);
    const std::string& subject(subject_ustr.string());
    ustringhash pattern_hash = ustringhash_from(pattern_);
    ustring pattern          = ustring_from(pattern_hash);
    std::match_results<std::string::const_iterator> mresults;
    const std::regex& regex(ctx->find_regex(pattern));
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
                m[r] = pattern.length();
            }
        }
        return res;
    } else {
        return fullmatch ? std::regex_match(subject, regex)
                         : std::regex_search(subject, regex);
    }
}

// TODO: transition format to from llvm_gen_printf_legacy
//       to llvm_gen_print_fmt by providing an osl_gen_formatfmt here
OSL_SHADEOP ustringhash_pod
osl_format(ustringhash_pod format_str_, ...)
{
    auto format_str = ustring_from(format_str_);
    va_list args;
    va_start(args, format_str_);
    std::string s = Strutil::vsprintf(format_str.c_str(), args);
    va_end(args);
    return ustring(s).hash();
}



OSL_SHADEOP int
osl_split(ustringhash_pod str_, ustringhash_pod* results, ustringhash_pod sep_,
          int maxsplit, int resultslen)
{
    auto str = ustring_from(str_);
    auto sep = ustring_from(sep_);
    maxsplit = OIIO::clamp(maxsplit, 0, resultslen);
    std::vector<std::string> splits;
    Strutil::split(str.string(), splits, sep.string(), maxsplit);
    int n = std::min(maxsplit, (int)splits.size());
    for (int i = 0; i < n; ++i)
        results[i] = ustring(splits[i]).uhash().hash();
    return n;
}



////////
// The osl_printf, osl_error, osl_warning, and osl_fprintf are deprecated but
// the stubs are needed for now to prevent breaking OptiX-based renderers who
// aren't quite ready to refactor around the journaling print family of
// functions. They eventually can be removed when we're happy that all the
// compliant renderers have adapted.

OSL_SHADEOP void
osl_printf(ShaderGlobals* sg, OSL::ustringhash_pod format_str, ...)
{
}

OSL_SHADEOP void
osl_error(ShaderGlobals* sg, OSL::ustringhash_pod format_str, ...)
{
}

OSL_SHADEOP void
osl_warning(ShaderGlobals* sg, OSL::ustringhash_pod format_str, ...)
{
}

OSL_SHADEOP void
osl_fprintf(ShaderGlobals* /*sg*/, OSL::ustringhash_pod filename,
            OSL::ustringhash_pod format_str, ...)
{
}

////////


OSL_RSOP OSL::ustringhash_pod
osl_formatfmt(OpaqueExecContextPtr exec_ctx,
              OSL::ustringhash_pod fmt_specification, int32_t arg_count,
              void* arg_types, uint32_t arg_values_size, uint8_t* arg_values)
{
    auto encoded_types = reinterpret_cast<const EncodedType*>(arg_types);

    std::string decoded_str;
    OSL::decode_message(fmt_specification, arg_count, encoded_types, arg_values,
                        decoded_str);
    return ustring(decoded_str).hash();
}


}  // end namespace pvt
OSL_NAMESPACE_END
