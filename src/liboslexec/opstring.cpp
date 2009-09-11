/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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
                std::cerr << "Mismatch between format string and arguments";
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
    if (exec->debug())
        std::cout << "printf!\n";
    ASSERT (nargs >= 1);
    Symbol &Format (exec->sym (args[0]));
    ASSERT (Format.typespec().is_string ());
    VaryingRef<ustring> format ((ustring *)Format.data(), Format.step());
    bool varying = format.is_varying ();
    for (int i = 1;  i < nargs;  ++i)
        varying |= exec->sym(args[i]).is_varying ();
    if (varying) {
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i]) {
                std::string s = format_args (exec, format[i].c_str(),
                                             nargs-1, args+1, i);
                std::cout << s;
                // FIXME -- go through the exec's error mechanism
            }
    } else {
        // Uniform case
        std::string s = format_args (exec, (*format).c_str(),
                                     nargs-1, args+1, 0);
        std::cout << s;
        // FIXME -- go through the exec's error mechanism
    }
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
