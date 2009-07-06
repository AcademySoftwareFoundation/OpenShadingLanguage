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

#include <iostream>

#include "oslexec_pvt.h"
#include "oslops.h"

#include "OpenImageIO/varyingref.h"


namespace OSL {
namespace pvt {



// Heavy lifting of 'assign'
template <class R, class S>
static void
specialized_assign (R *result, int resultstep,
                    S *src, int srcstep,
                    const Runflag *runflags, int beginpoint, int endpoint)
{
    if (! resultstep) {
        // Result (and src) are uniform
        *result = *src;
    } else {
        VaryingRef<R> Result (result, resultstep);
        VaryingRef<S> Src (src, srcstep);
        for (int i = beginpoint;  i < endpoint;  ++i)
            if (runflags[i])
                result[i] = src[i];
    }
    std::cerr << "After assignment, new values are:\n";
    for (int i = beginpoint;  i < endpoint;  ++i)
        std::cerr << "\t" << i << ": " << (result[i]) << "\n";

}



DECLOP (OP_assign)
{
    std::cerr << "Executing assign!\n";
    ASSERT (nargs == 2);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Src (exec->sym (args[1]));
    std::cerr << "  Result is " << Result.typespec().string() 
              << " " << Result.mangled() << " @ " << (void *)Result.data() << "\n";
    std::cerr << "  Src is " << Src.typespec().string() 
              << " " << Src.mangled() << " @ " << (void*)Src.data() << "\n";
    ASSERT (! Result.typespec().is_closure() &&
            ! Result.typespec().is_structure() &&
            ! Result.typespec().is_array());   // Not yet
    ASSERT (! Src.typespec().is_closure() &&
            ! Src.typespec().is_structure() &&
            ! Src.typespec().is_array());   // Not yet
    if (Result.typespec().is_float()) {
        if (Src.typespec().is_float()) {
            specialized_assign ((float *)Result.data(), Result.step(),
                                (float *)Src.data(), Src.step(),
                                runflags, beginpoint, endpoint);
            return;
        } if (Src.typespec().is_int()) {
            
        } else {

        }
    }
    ASSERT (0 && "Assignment types can't be handled");
}



}; // namespace pvt
}; // namespace OSL
