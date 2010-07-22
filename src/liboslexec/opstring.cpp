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

#include "oslops.h"
#include "oslexec_pvt.h"

#include <OpenEXR/ImathFun.h>


// Heavy lifting of OSL regex operations.
extern "C" int
osl_regex_impl2 (OSL::pvt::ShadingContext *ctx, ustring subject_,
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



#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {


static std::string
format_args (ShadingExecution *exec, const char *format,
             int nargs, const int *args, int whichpoint)
{
    if (! format || ! format[0])
        return std::string ();

    std::string s;
    int arg = 0;   // Which arg we're on
    while (*format) {
        if (*format == '%') {
            if (format[1] == '%') {
                // '%%' is a literal '%'
                s += '%';
                format += 2;  // skip both percentages
                continue;
            }
            const char *oldfmt = format;  // mark beginning of format
            while (*format && *format != 'c' && *format != 'd' && 
                   *format != 'f' && *format != 'g' && *format != 'i' &&
                   *format != 'm' && *format != 'n' && *format != 'p' &&
                   *format != 's' && *format != 'v')
                ++format;
            ++format; // Also eat the format char
            std::string ourformat (oldfmt, format);  // straddle the format
            if (arg >= nargs) {
                // FIXME -- send this error through the exec
                exec->error ("Mismatch between format string and arguments");
                continue;
            }
            // Doctor it to fix mismatches between format and data
            Symbol &sym (exec->sym(args[arg]));
            TypeDesc simpletype (sym.typespec().simpletype());
            char code = ourformat[ourformat.length()-1];
            if (sym.typespec().is_closure() && code != 's') {
                ourformat[ourformat.length()-1] = 's';
            } else if (simpletype.basetype == TypeDesc::FLOAT &&
                    code != 'f' && code != 'g') {
                ourformat[ourformat.length()-1] = 'g';
            } else if (simpletype.basetype == TypeDesc::INT && code != 'd') {
                ourformat[ourformat.length()-1] = 'd';
            } else if (simpletype.basetype == TypeDesc::STRING && code != 's') {
                ourformat[ourformat.length()-1] = 's';
            }
            s += exec->format_symbol (ourformat, sym, whichpoint);
            ++arg;
        } else if (*format == '\\') {
            // Escape sequence
            ++format;  // skip the backslash
            switch (*format) {
            case 'n' : s += '\n';     break;
            case 'r' : s += '\r';     break;
            case 't' : s += '\t';     break;
            default:   s += *format;  break;  // Catches '\\' also!
            }
            ++format;
        } else {
            // Everything else -- just copy the character and advance
            s += *format++;
        }
    }
    return s;
}



DECLOP (OP_printf)
{
    ASSERT (nargs >= 1);
    Symbol &Format (exec->sym (args[0]));
    ASSERT (Format.typespec().is_string ());
    VaryingRef<ustring> format ((ustring *)Format.data(), Format.step());
    bool varying = format.is_varying ();
    for (int i = 1;  i < nargs;  ++i)
        varying |= exec->sym(args[i]).is_varying ();
    if (! varying) {
        std::string s = format_args (exec, format[0].c_str(),
                                     nargs-1, args+1, 0);
        exec->message ("%s", s.c_str());
    } else {
        SHADE_LOOP_BEGIN
            std::string s = format_args (exec, format[i].c_str(),
                                         nargs-1, args+1, i);
            exec->message ("%s", s.c_str());
        SHADE_LOOP_END
    }
}



DECLOP (OP_error)
{
    ASSERT (nargs >= 1);
    Symbol &Format (exec->sym (args[0]));
    ASSERT (Format.typespec().is_string ());
    VaryingRef<ustring> format ((ustring *)Format.data(), Format.step());
    bool varying = format.is_varying ();
    for (int i = 1;  i < nargs;  ++i)
        varying |= exec->sym(args[i]).is_varying ();
    if (! varying) {
        std::string s = format_args (exec, format[0].c_str(),
                                     nargs-1, args+1, 0);
        exec->error ("Shader error [%s]: %s",
                     exec->shadername().c_str(), s.c_str());
    } else {
        SHADE_LOOP_BEGIN
            std::string s = format_args (exec, format[i].c_str(),
                                         nargs-1, args+1, i);
            exec->error ("Shader error [%s]: %s",
                         exec->shadername().c_str(), s.c_str());
        SHADE_LOOP_END
    }
}



DECLOP (OP_warning)
{
    ASSERT (nargs >= 1);
    Symbol &Format (exec->sym (args[0]));
    ASSERT (Format.typespec().is_string ());
    VaryingRef<ustring> format ((ustring *)Format.data(), Format.step());
    bool varying = format.is_varying ();
    for (int i = 1;  i < nargs;  ++i)
        varying |= exec->sym(args[i]).is_varying ();
    if (! varying) {
        std::string s = format_args (exec, format[0].c_str(),
                                     nargs-1, args+1, 0);
        exec->warning ("Shader warning [%s]: %s",
                       exec->shadername().c_str(), s.c_str());
    } else {
        SHADE_LOOP_BEGIN
            std::string s = format_args (exec, format[i].c_str(),
                                         nargs-1, args+1, i);
            exec->warning ("Shader warning [%s]: %s",
                           exec->shadername().c_str(), s.c_str());
        SHADE_LOOP_END
    }
}



DECLOP (OP_format)
{
    ASSERT (nargs >= 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Format (exec->sym (args[1]));
    ASSERT (Result.typespec().is_string ());
    ASSERT (Format.typespec().is_string ());

    // Adjust the result's uniform/varying status
    bool varying = Format.is_varying ();
    for (int i = 2;  i < nargs;  ++i)
        varying |= exec->sym(args[i]).is_varying ();
    exec->adjust_varying (Result, varying);

    VaryingRef<ustring> result ((ustring *)Result.data(), Result.step());
    VaryingRef<ustring> format ((ustring *)Format.data(), Format.step());
    if (Result.is_uniform()) {
        result[0] = format_args (exec, format[0].c_str(),
                                 nargs-2, args+2, 0);
    } else {
        SHADE_LOOP_BEGIN
            result[i] = format_args (exec, format[i].c_str(),
                                     nargs-2, args+2, i);
        SHADE_LOOP_END
    }
}



DECLOP (OP_concat)
{
    ASSERT (nargs >= 1);
    Symbol &Result (exec->sym (args[0]));
    ASSERT (Result.typespec().is_string ());

    // Adjust the result's uniform/varying status and construct our
    // format string.
    bool varying = false;
    std::string format;
    for (int i = 1;  i < nargs;  ++i) {
        varying |= exec->sym(args[i]).is_varying ();
        format += std::string ("%s");
    }
    exec->adjust_varying (Result, varying);

    VaryingRef<ustring> result ((ustring *)Result.data(), Result.step());
    if (Result.is_uniform()) {
        result[0] = ustring (format_args (exec, format.c_str(),
                                          nargs-1, args+1, 0));
    } else {
        SHADE_LOOP_BEGIN
            result[i] = ustring (format_args (exec, format.c_str(),
                                              nargs-1, args+1, i));
        SHADE_LOOP_END
    }
}



DECLOP (OP_strlen)
{
    DASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &S (exec->sym (args[1]));
    DASSERT (Result.typespec().is_int() && S.typespec().is_string());

    exec->adjust_varying (Result, S.is_varying(), false /* can't alias */);

    VaryingRef<int> result ((int *)Result.data(), Result.step());
    VaryingRef<ustring> s ((ustring *)S.data(), S.step());
    if (Result.is_uniform()) {
        result[0] = s[0].length ();
    } else {
        SHADE_LOOP_BEGIN
            result[i] = s[i].length ();
        SHADE_LOOP_END
    }
}



DECLOP (OP_startswith)
{
    DASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &S (exec->sym (args[1]));
    Symbol &Substr (exec->sym (args[2]));
    DASSERT (Result.typespec().is_int() && S.typespec().is_string() &&
             Substr.typespec().is_string());

    exec->adjust_varying (Result, S.is_varying() | Substr.is_varying(),
                          false /* can't alias */);

    VaryingRef<int> result ((int *)Result.data(), Result.step());
    VaryingRef<ustring> s ((ustring *)S.data(), S.step());
    VaryingRef<ustring> substr ((ustring *)Substr.data(), Substr.step());
    if (Result.is_uniform()) {
        int c = strncmp (s[0].c_str(), substr[0].c_str(), substr[0].size());
        result[0] = (c == 0);
    } else {
        SHADE_LOOP_BEGIN
            int c = strncmp (s[i].c_str(), substr[i].c_str(), substr[i].size());
            result[i] = (c == 0);
        SHADE_LOOP_END
    }
}



DECLOP (OP_endswith)
{
    DASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &S (exec->sym (args[1]));
    Symbol &Substr (exec->sym (args[2]));
    DASSERT (Result.typespec().is_int() && S.typespec().is_string() &&
             Substr.typespec().is_string());

    exec->adjust_varying (Result, S.is_varying() | Substr.is_varying(),
                          false /* can't alias */);

    VaryingRef<int> result ((int *)Result.data(), Result.step());
    VaryingRef<ustring> s ((ustring *)S.data(), S.step());
    VaryingRef<ustring> substr ((ustring *)Substr.data(), Substr.step());
    if (Result.is_uniform()) {
        size_t len = substr[0].length ();
        if (len > s[0].length())
            result[0] = 0;
        else {
            int c = strncmp (s[0].c_str() + s[0].length() - len,
                             substr[0].c_str(), substr[0].size());
            result[0] = (c == 0);
        }
    } else {
        SHADE_LOOP_BEGIN
            size_t len = substr[i].length ();
            if (len > s[i].length())
                result[i] = 0;
            else {
                int c = strncmp (s[i].c_str() + s[i].length() - len,
                                 substr[i].c_str(), substr[i].size());
                result[i] = (c == 0);
            }
        SHADE_LOOP_END
    }
}



DECLOP (OP_substr)
{
    DASSERT (nargs == 4);
    Symbol &Result (exec->sym (args[0]));
    Symbol &S (exec->sym (args[1]));
    Symbol &Start (exec->sym (args[2]));
    Symbol &Length (exec->sym (args[3]));
    DASSERT (Result.typespec().is_string() && S.typespec().is_string() &&
             Start.typespec().is_int() && Length.typespec().is_int());

    bool varying = S.is_varying() | Start.is_varying() | Length.is_varying();
    exec->adjust_varying (Result, varying);

    VaryingRef<ustring> result ((ustring *)Result.data(), Result.step());
    VaryingRef<ustring> s ((ustring *)S.data(), S.step());
    VaryingRef<int> start ((int *)Start.data(), Start.step());
    VaryingRef<int> length ((int *)Length.data(), Length.step());
    if (Result.is_uniform()) {
        const ustring &str (s[0]);
        int b = start[0];
        if (b < 0)
            b += str.length();
        b = Imath::clamp (b, 0, (int)str.length());
        int len = Imath::clamp (length[0], 0, (int)str.length());
        result[0] = ustring (s[0], b, len);
    } else {
        SHADE_LOOP_BEGIN
            const ustring &str (s[i]);
            int b = start[i];
            if (b < 0)
                b += str.length();
            b = Imath::clamp (b, 0, (int)str.length());
            int len = Imath::clamp (length[i], 0, (int)str.length());
            result[i] = ustring (s[i], b, len);
        SHADE_LOOP_END
    }
}



// This template function does regex_match or regex_search, with or
// without match results being saved, and will be specialized for one of
// those 4 cases by template expansion.
template <bool fullmatch, bool do_match_results>
DECLOP (regex_search_specialized)
{
    Symbol &Result (exec->sym (args[0]));
    Symbol &Subject (exec->sym (args[1]));
    Symbol &Match (exec->sym (args[2]));
    TypeDesc matchtype = Match.typespec().simpletype();
    Symbol &Pattern (exec->sym (args[2+do_match_results]));
    DASSERT (Result.typespec().is_int() && Subject.typespec().is_string() &&
             Pattern.typespec().is_string());
    DASSERT (!do_match_results || 
             (Match.typespec().is_array() &&
              Match.typespec().elementtype().is_int()));
    bool varying = Subject.is_varying() | Pattern.is_varying();
    exec->adjust_varying (Result, varying);
    if (do_match_results) {
        ASSERT (matchtype.arraylen && matchtype.elementtype() == TypeDesc::INT);
        exec->adjust_varying (Match, varying);
    }

    VaryingRef<int> result ((int *)Result.data(), Result.step());
    VaryingRef<ustring> subject ((ustring *)Subject.data(), Subject.step());
    VaryingRef<ustring> pattern ((ustring *)Pattern.data(), Pattern.step());
    ustring last_pattern;
    boost::match_results<std::string::const_iterator> mresults;
    const boost::regex *regex = NULL;
    SHADE_LOOP_BEGIN
        if (! regex || pattern[i] != last_pattern) {
            regex = &exec->context()->find_regex (pattern[i]);
            last_pattern = pattern[i];
        }
        if (do_match_results && matchtype.arraylen > 0) {
            std::string::const_iterator start = subject[i].string().begin();
            result[i] = fullmatch ? 
                boost::regex_match (subject[i].string(), mresults, *regex) :
                boost::regex_search (subject[i].string(), mresults, *regex);
            int *m = (int *)((char *)Match.data() + i*Match.step());
            for (int r = 0;  r < matchtype.arraylen;  ++r) {
                if (r/2 < (int)mresults.size()) {
                    if ((r & 1) == 0)
                        m[r] = mresults[r/2].first - start;
                    else
                        m[r] = mresults[r/2].second - start;
                } else {
                    m[r] = pattern[i].length();
                }
            }
        } else {
            result[i] = fullmatch ? regex_match (subject[i].c_str(), *regex)
                : regex_search (subject[i].c_str(), *regex);
        }
        if (! Result.is_varying())
            SHADE_LOOP_EXIT
    SHADE_LOOP_END
}



DECLOP (OP_regex_search)
{
    ASSERT (nargs == 3 || nargs == 4);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Subject (exec->sym (args[1]));
    bool do_match_results = (nargs == 4);
    Symbol &Match (exec->sym (args[2]));
    Symbol &Pattern (exec->sym (args[2+do_match_results]));
    ASSERT (Result.typespec().is_int() && Subject.typespec().is_string() &&
            Pattern.typespec().is_string());
    ASSERT (!do_match_results || 
            (Match.typespec().is_array() &&
             Match.typespec().elementtype().is_int()));

    OpImpl impl = NULL;
    if (do_match_results)
        impl = regex_search_specialized<false, true>;
    else
        impl = regex_search_specialized<false, false>;
    impl (exec, nargs, args);
    // Use the specialized one for next time!  Never have to check the
    // types or do the other sanity checks again.
    // FIXME -- is this thread-safe?
    exec->op().implementation (impl);
}



DECLOP (OP_regex_match)
{
    ASSERT (nargs == 3 || nargs == 4);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Subject (exec->sym (args[1]));
    bool do_match_results = (nargs == 4);
    Symbol &Match (exec->sym (args[2]));
    Symbol &Pattern (exec->sym (args[2+do_match_results]));
    ASSERT (Result.typespec().is_int() && Subject.typespec().is_string() &&
            Pattern.typespec().is_string());
    ASSERT (!do_match_results || 
            (Match.typespec().is_array() &&
             Match.typespec().elementtype().is_int()));

    OpImpl impl = NULL;
    if (do_match_results)
        impl = regex_search_specialized<true, true>;
    else
        impl = regex_search_specialized<true, false>;
    impl (exec, nargs, args);
    // Use the specialized one for next time!  Never have to check the
    // types or do the other sanity checks again.
    // FIXME -- is this thread-safe?
    exec->op().implementation (impl);
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
