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

#include <OSL/oslconfig.h>

#include <OSL/batched_shaderglobals.h>
#include <OSL/wide.h>

#include <OpenImageIO/fmath.h>
#include <OpenImageIO/strutil.h>

#include "oslexec_pvt.h"

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

#include "define_opname_macros.h"


OSL_BATCHOP void
__OSL_MASKED_OP3(concat, Ws, Ws, Ws)(void* wr_, void* ws_, void* wt_,
                                     unsigned int mask_value)
{
    Wide<const ustring> wS(ws_);
    Wide<const ustring> wT(wt_);
    Masked<ustring> wR(wr_, Mask(mask_value));

    char local_buf[256];
    std::unique_ptr<char[]> heap_buf;
    size_t heap_buf_len = 0;

    // Must check mask before dereferencing s or t
    // as they are undefined when masked off
    wR.mask().foreach (
        [=, &local_buf, &heap_buf, &heap_buf_len](ActiveLane lane) -> void {
            ustring s  = wS[lane];
            ustring t  = wT[lane];
            size_t sl  = s.length();
            size_t tl  = t.length();
            size_t len = sl + tl;
            char* buf  = local_buf;
            if (len > sizeof(local_buf)) {
                if (len > heap_buf_len) {
                    heap_buf.reset(new char[len]);
                    heap_buf_len = len;
                }
                buf = heap_buf.get();
            }
            memcpy(buf, s.c_str(), sl);
            memcpy(buf + sl, t.c_str(), tl);
            wR[lane] = ustring(buf, len);
        });
}



OSL_BATCHOP void
__OSL_MASKED_OP2(strlen, Wi, Ws)(void* wr_, void* ws_, unsigned int mask_value)
{
    Wide<const ustring> wS(ws_);
    Masked<int> wR(wr_, Mask(mask_value));

    OSL_FORCEINLINE_BLOCK
    {
#if (!OSL_CLANG_VERSION || OSL_INTEL_CLASSIC_COMPILER_VERSION)
        // Clang 11 generated SIMD crashes at runtime
        // TODO: investigate clang crash when vectorizing
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
#endif
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            ustring s = wS[lane];
            if (wR.mask()[lane]) {
                wR[ActiveLane(lane)] = (int)s.length();
            }
        }
    }
}


OSL_BATCHOP void
__OSL_MASKED_OP2(hash, Wi, Ws)(void* wr_, void* ws_, unsigned int mask_value)
{
    Wide<const ustring> wS(ws_);
    Masked<int> wR(wr_, Mask(mask_value));

    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            ustring s = wS[lane];
            if (wR.mask()[lane]) {
                wR[ActiveLane(lane)] = (int)s.hash();
            }
        }
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP3(getchar, Wi, Ws, Wi)(void* wr_, void* ws_, void* wi_,
                                      unsigned int mask_value)
{
    Wide<const ustring> wS(ws_);
    Wide<const int> wI(wi_);
    Masked<int> wR(wr_, Mask(mask_value));

#if 1  // SIMD version may not be profitable, need to benchmark to confirm
    wR.mask().foreach ([=](ActiveLane lane) -> void {
        ustring str = wS[lane];
        int index   = wI[lane];
        wR[lane]    = ((!str.empty()) && unsigned(index) < str.length())
                          ? str[index]
                          : 0;
    });
#else
    OSL_FORCEINLINE_BLOCK
    {
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            ustring str = wS[lane];
            int index   = wI[lane];
            if (wR.mask()[lane]) {
                char rchar = 0;

                // The length implementation returns 0 when str.empty()
                if (unsigned(index) < str.length()) {
                    rchar = str[index];
                }
                wR[ActiveLane(lane)] = rchar;
            }
        }
    }
#endif
}



// TODO: duplicated in opstring.cpp, move to common header (or not)
static OSL_FORCEINLINE int
startswith_iss_impl(ustring s, ustring substr)
{
    size_t substr_len = substr.length();
    if (substr_len == 0)  // empty substr always matches
        return 1;
    size_t s_len = s.length();
    if (substr_len > s_len)  // longer needle than haystack can't
        return 0;            // match (including empty s)
    return strncmp(s.c_str(), substr.c_str(), substr_len) == 0;
}



OSL_BATCHOP void
__OSL_MASKED_OP3(startswith, Wi, Ws, Ws)(void* wr_, void* ws_, void* wsubs_,
                                         unsigned int mask_value)
{
    Wide<const ustring> wS(ws_);
    Wide<const ustring> wSubs(wsubs_);
    Masked<int> wR(wr_, Mask(mask_value));

    wR.mask().foreach ([=](ActiveLane lane) -> void {
        ustring substr = wSubs[lane];
        ustring s      = wS[lane];
        wR[lane]       = startswith_iss_impl(s, substr);
    });
}



// TODO: duplicated in opstring.cpp, move to common header (or not)
static OSL_FORCEINLINE int
endswith_iss_impl(ustring s, ustring substr)
{
    size_t substr_len = substr.length();
    if (substr_len == 0)  // empty substr always matches
        return 1;
    size_t s_len = s.length();
    if (substr_len > s_len)  // longer needle than haystack can't
        return 0;            // match (including empty s)
    return strncmp(s.c_str() + s_len - substr_len, substr.c_str(), substr_len)
           == 0;
}


OSL_BATCHOP void
__OSL_MASKED_OP3(endswith, Wi, Ws, Ws)(void* wr_, void* ws_, void* wsubs_,
                                       unsigned int mask_value)
{
    Wide<const ustring> wS(ws_);
    Wide<const ustring> wSubs(wsubs_);
    Masked<int> wR(wr_, Mask(mask_value));

    wR.mask().foreach ([=](ActiveLane lane) -> void {
        ustring substr = wSubs[lane];
        ustring s      = wS[lane];
        wR[lane]       = endswith_iss_impl(s, substr);
    });
}



OSL_BATCHOP void
__OSL_MASKED_OP2(stoi, Wi, Ws)(void* wint_ptr, void* wstr_ptr,
                               unsigned int mask_value)
{
    Wide<const ustring> wstr(wstr_ptr);
    Masked<int> wR(wint_ptr, Mask(mask_value));

    // Avoid cost of strtol if lane is masked off
    // Also the value of str for a masked off lane could be
    // invalid/undefined and not safe to call strtol on.
    wR.mask().foreach ([=](ActiveLane lane) -> void {
        const char* str = unproxy(wstr[lane]).c_str();
        // TODO: Suspect we could implement SIMD friendly version
        // that is more efficient than the library call
        wR[lane] = str ? OIIO::Strutil::from_string<int>(str) : 0;
    });
}



OSL_BATCHOP void
__OSL_MASKED_OP2(stof, Wf, Ws)(void* wr_, void* ws_, unsigned int mask_value)
{
    Wide<const ustring> wS(ws_);
    Masked<float> wR(wr_, Mask(mask_value));

    // Avoid cost of strtof if lane is masked off
    // Also the value of str for a masked off lane could be
    // invalid/undefined and not safe to call strtod on.
    wR.mask().foreach ([=](ActiveLane lane) -> void {
        const char* str = unproxy(wS[lane]).c_str();
        // TODO: Suspect we could implement SIMD friendly version
        // that is more efficient than the library call
        wR[lane] = str ? OIIO::Strutil::from_string<float>(str) : 0.0f;
    });
}



// TODO: duplicated in opstring.cpp, move to common header (or not)
static OSL_FORCEINLINE ustring
substr_ssii_impl(ustring s, int start, int length)
{
    int slen = int(s.length());
    if (slen == 0) {
        return ustring(NULL);  // No substring of empty string
    }
    int b = start;
    if (b < 0)
        b += slen;
    b = Imath::clamp(b, 0, slen);
    return ustring(s, b, Imath::clamp(length, 0, slen));
}



OSL_BATCHOP void
__OSL_MASKED_OP4(substr, Ws, Ws, Wi, Wi)(void* wr_, void* ws_, void* wstart_,
                                         void* wlength_,
                                         unsigned int mask_value)
{
    Wide<const ustring> wS(ws_);
    Wide<const int> wL(wlength_);
    Wide<const int> wSt(wstart_);
    Masked<ustring> wR(wr_, Mask(mask_value));

    wR.mask().foreach ([=](ActiveLane lane) -> void {
        ustring s  = wS[lane];
        int start  = wSt[lane];
        int length = wL[lane];
        wR[lane]   = substr_ssii_impl(s, start, length);
    });
}



OSL_BATCHOP int
__OSL_OP(regex_impl)(void* bsg_, const char* subject_, void* results,
                     int nresults, const char* pattern, int fullmatch)
{
    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ShadingContext* ctx = bsg->uniform.context;

    // TODO:  probably could share implementation from here with osl_regex_impl from opstring.cpp
    OSL_ASSERT(ustring::is_unique(subject_));
    OSL_ASSERT(ustring::is_unique(pattern));

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
        return fullmatch ? regex_match(subject, regex)
                         : regex_search(subject, regex);
    }
}



OSL_BATCHOP void
__OSL_MASKED_OP(regex_impl)(void* bsg_, void* wsuccess_ptr, void* wsubject_ptr,
                            void* wresults_ptr, int nresults,
                            void* wpattern_ptr, int fullmatch,
                            unsigned int mask_value)
{
    auto* bsg           = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    ShadingContext* ctx = bsg->uniform.context;

    Mask mask(mask_value);
    OSL_ASSERT(mask.any_on());

    Masked<int> wsuccess(wsuccess_ptr, mask);
    Masked<int[]> wresults(wresults_ptr, nresults, mask);
    Wide<const ustring> wsubject(wsubject_ptr);
    Wide<const ustring> wpattern(wpattern_ptr);

    mask.foreach ([=](ActiveLane lane) -> void {
        ustring usubject = wsubject[lane];
        ustring pattern  = wpattern[lane];
        OSL_ASSERT(ustring::is_unique(usubject.c_str()));
        OSL_ASSERT(ustring::is_unique(pattern.c_str()));

        auto results = wresults[lane];

        const std::string& subject = usubject.string();
        std::match_results<std::string::const_iterator> mresults;
        const std::regex& regex(ctx->find_regex(pattern));
        if (nresults > 0) {
            std::string::const_iterator start = subject.begin();
            int res = fullmatch ? std::regex_match(subject, mresults, regex)
                                : std::regex_search(subject, mresults, regex);
            for (int r = 0; r < nresults; ++r) {
                if (r / 2 < (int)mresults.size()) {
                    if ((r & 1) == 0)
                        results[r] = mresults[r / 2].first - start;
                    else
                        results[r] = mresults[r / 2].second - start;
                } else {
                    results[r] = pattern.length();
                }
            }
            wsuccess[lane] = res;
        } else {
            wsuccess[lane] = fullmatch ? regex_match(subject, regex)
                                       : regex_search(subject, regex);
        }
    });
}



OSL_BATCHOP void
__OSL_OP(format)(void* wide_output, unsigned int mask_value,
                 const char* format_str, ...)
{
    va_list args;
    va_start(args, format_str);
    std::string s = Strutil::vsprintf(format_str, args);
    va_end(args);

    ustring result = ustring(s);
    Masked<ustring> wOut(wide_output, Mask(mask_value));

    OSL::assign_all(wOut, result);
}



OSL_BATCHOP void
__OSL_OP(printf)(BatchedShaderGlobals* bsg, unsigned int mask_value,
                 const char* format_str, ...)
{
    Mask mask(mask_value);
    // Not strictly necessary, but using to ensure "current" logic that conditionals should skip
    // any code block with no active lanes, this could be changed in future
    OSL_ASSERT(mask.any_on());

    va_list args;
    va_start(args, format_str);
    std::string s = Strutil::vsprintf(format_str, args);
    va_end(args);

    bsg->uniform.context->record_error(
        ErrorHandler::EH_MESSAGE, s,
        static_cast<OSL::Mask<MaxSupportedSimdLaneCount>>(mask));
}



OSL_BATCHOP void
__OSL_OP(error)(BatchedShaderGlobals* bsg, unsigned int mask_value,
                const char* format_str, ...)
{
    Mask mask(mask_value);
    // Not strictly necessary, but using to ensure "current" logic that conditionals should skip
    // any code block with no active lanes, this could be changed in future
    OSL_ASSERT(mask.any_on());

    va_list args;
    va_start(args, format_str);
    std::string s = Strutil::vsprintf(format_str, args);
    va_end(args);

    bsg->uniform.context->record_error(
        ErrorHandler::EH_ERROR, s,
        static_cast<OSL::Mask<MaxSupportedSimdLaneCount>>(mask));
}



OSL_BATCHOP void
__OSL_OP(warning)(BatchedShaderGlobals* bsg, unsigned int mask_value,
                  const char* format_str, ...)
{
    if (bsg->uniform.context->allow_warnings()) {
        Mask mask(mask_value);
        // Not strictly necessary, but using to ensure "current" logic that conditionals should skip
        // any code block with no active lanes, this could be changed in future
        OSL_ASSERT(mask.any_on());

        va_list args;
        va_start(args, format_str);
        std::string s = Strutil::vsprintf(format_str, args);
        va_end(args);

        bsg->uniform.context->record_error(
            ErrorHandler::EH_WARNING, s,
            static_cast<OSL::Mask<MaxSupportedSimdLaneCount>>(mask));
    }
}



OSL_BATCHOP void
__OSL_OP(fprintf)(BatchedShaderGlobals* bsg, unsigned int mask_value,
                  const char* filename, const char* format_str, ...)
{
    OSL_ASSERT(bsg != nullptr);
    OSL_ASSERT(filename != nullptr);
    OSL_ASSERT(format_str != nullptr);
    Mask mask(mask_value);
    // Not strictly necessary, but using to ensure "current" logic that conditionals should skip
    // any code block with no active lanes, this could be changed in future
    OSL_ASSERT(mask.any_on());

    va_list args;
    va_start(args, format_str);
    std::string s = Strutil::vsprintf(format_str, args);
    va_end(args);

    bsg->uniform.context->record_to_file(
        USTR(filename), s,
        static_cast<OSL::Mask<MaxSupportedSimdLaneCount>>(mask));
}



OSL_BATCHOP void
__OSL_MASKED_OP(split)(void* wresults,
                       /*stores n*/ void* wstr, /*input string*/
                       void* wresult_string,    /*resulting split string*/
                       void* wsep,              /*sep*/
                       void* wmaxsplit,         /*void *wrlen,*/
                       int resultslen, unsigned int mask_value)
{
    Wide<const ustring> wS(wstr);

    Masked<int> wR(wresults, Mask(mask_value));  //length of split array
    Masked<ustring[]> wRString(wresult_string, resultslen,
                               Mask(mask_value));  //Split string

    Wide<const ustring> wSep(wsep);
    Wide<const int> wMaxSplit(wmaxsplit);

    wR.mask().foreach ([=](ActiveLane lane) -> void {
        int maxsplit       = wMaxSplit[lane];
        ustring str        = wS[lane];
        ustring sep        = wSep[lane];
        auto resultStrings = wRString[lane];

        maxsplit = OIIO::clamp(maxsplit, 0, resultslen);
        std::vector<std::string> splits;
        OIIO::Strutil::split(str.string(), splits, sep.string(), maxsplit);
        int n    = std::min(maxsplit, (int)splits.size());
        wR[lane] = n;  //Length of split array
        for (int i = 0; i < n; ++i) {
            resultStrings[i] = ustring(splits[i]);
        }
    });

}  //split ends


}  // end namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#include "undef_opname_macros.h"
