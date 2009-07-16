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

#include <iostream>

#include "oslexec_pvt.h"
#include "oslops.h"

#include "OpenImageIO/varyingref.h"
#include "OpenImageIO/sysutil.h"


namespace OSL {
namespace pvt {


namespace {



};  // End anonymous namespace




DECLOP (OP_if)
{
    ASSERT (nargs == 1);
    Symbol &Condition (exec->sym (args[0]));
    ASSERT (Condition.typespec().is_int());
    VaryingRef<int> condition ((int *)Condition.data(), Condition.step());

    Opcode &op (exec->op());
    std::cerr << "if!  " << Condition.mangled() << ' ' << op.jump(0) << ' ' << op.jump(1) << "\n";

    // Determine if it's a "uniform if"
    bool uniform = Condition.is_uniform ();
    for (int i = beginpoint+1;  uniform && i < endpoint;  ++i)
        if (runflags[i] && condition[i] != condition[beginpoint]) {
            uniform = false;
            break;
        }

    // FIXME -- if there's potentially a 'break' or 'continue' inside
    // this conditional, we need to treat it as varying.

    if (uniform) {
        // Uniform condition -- don't need new runflags
        if (condition[beginpoint]) {
            // Condition is true -- execute the true clause.
            // But if there is no else clause, do nothing (!) and the
            // normal execution will just take care of itself
            if (op.jump(1) != op.jump(0)) {
                exec->run (exec->ip()+1, op.jump(0));  // Run the true clause
                exec->ip (op.jump(1) - 1);             // Skip the false clause
            }
        } else {
            // Condition is false -- just a jump to 'else' and keep going
            exec->ip (op.jump(0) - 1);
        }
        return;
    }

    // From here on, varying condition or potential break/continue at play

    // Generate new true and false runflags based on the condition
    Runflag *true_runflags = ALLOCA (Runflag, exec->npoints());
    memcpy (true_runflags, runflags, exec->npoints() * sizeof(Runflag));
    Runflag *false_runflags = ALLOCA (Runflag, exec->npoints());
    memcpy (true_runflags, runflags, exec->npoints() * sizeof(Runflag));
    for (int i = beginpoint;  i < endpoint;  ++i) {
        if (runflags[i]) {
            if (condition[i])
                false_runflags[i] = RunflagOff;
            else
                true_runflags[i] = RunflagOff;
        }
    }

    // True clause
    exec->push_runflags (true_runflags, beginpoint, endpoint);
    exec->run (exec->ip() + 1, op.jump(0));
    exec->pop_runflags ();

    // False clause
    if (op.jump(0) < op.jump(1)) {
        exec->push_runflags (false_runflags, beginpoint, endpoint);
        exec->run (op.jump(0), op.jump(1));
        exec->pop_runflags ();
    }

    // Jump to after the if (remember that the interpreter loop will
    // increment the ip one more time, so back up one.
    exec->ip (op.jump(1) - 1);

    // FIXME -- we may need to call new_runflag_range here if, during
    // execution, we may have hit a 'break' or 'continue'.
}


}; // namespace pvt
}; // namespace OSL
