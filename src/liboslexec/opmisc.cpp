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

#include "oslops.h"
#include "oslexec_pvt.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {



DECLOP (OP_nop)
{
}



DECLOP (OP_missing)
{
    exec->error ("Missing op '%s'!", exec->op().opname().c_str());
}



DECLOP (OP_end)
{
}



// 'useparam' is called prior to use of parameters, and is the trigger
// for lazy evaluation of earlier layers that may be connected to the
// parameter.  Its arguments may consist of more than one parameter that
// needs its value to be established.
DECLOP (OP_useparam)
{
    DASSERT (nargs > 0);
    // For each argument to useparam, which names a parameter, initialize
    // it if it has not already been done.
    for (int a = 0;  a < nargs;  ++a) {
        Symbol &Param (exec->sym (args[a]));
        DASSERT (Param.symtype() == SymTypeParam || 
                 Param.symtype() == SymTypeOutputParam);
        // std::cerr << "useparam " << Param.name() << "\n";
        if (! Param.initialized()) {
            // std::cerr << "  not yet initialized!\n";
            exec->bind_initialize_param (Param, args[a]);
        }
    }
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
