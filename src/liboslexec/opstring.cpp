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

#include "oslops.h"
#include "oslexec_pvt.h"


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
            s += exec->format_symbol (ourformat, exec->sym(args[arg++]), whichpoint);
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
