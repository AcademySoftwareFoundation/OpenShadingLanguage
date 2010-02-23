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
/// Shader interpreter implementation of control flow statements
/// such as 'if', 'for', etc.
///
/////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "oslexec_pvt.h"
#include "oslops.h"

#include "OpenImageIO/varyingref.h"
#include "OpenImageIO/sysutil.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {


DECLOP (OP_if)
{
    ASSERT (nargs == 1);
    Symbol &Condition (exec->sym (args[0]));
    ASSERT (Condition.typespec().is_int());
    VaryingRef<int> condition ((int *)Condition.data(), Condition.step());
    Opcode &op (exec->op());

    // Determine if it's a "uniform if"
    bool uniform = Condition.is_uniform ();
    if (! uniform) {
        // Second chance -- what if the condition is varying, but the
        // results are the same at all points?
        uniform = true;
        int beginpoint = exec->beginpoint();
        SHADE_LOOP_BEGIN
            if (condition[i] != condition[beginpoint]) {
                uniform = false;
                break;
            }
        SHADE_LOOP_END
    }

    // FIXME -- if there's potentially a 'break' or 'continue' inside
    // this conditional, we need to treat it as varying.

    if (uniform) {
        // Uniform condition -- don't need new runflags
        if (condition[exec->beginpoint()]) {
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
#if USE_RUNFLAGS
    Runflag *true_runflags = ALLOCA (Runflag, exec->npoints());
    memcpy (true_runflags, exec->runstate().runflags, exec->npoints() * sizeof(Runflag));
    Runflag *false_runflags = ALLOCA (Runflag, exec->npoints());
    memcpy (false_runflags, exec->runstate().runflags, exec->npoints() * sizeof(Runflag));
    int *true_indices = NULL;
    int *false_indices = NULL;
    int ntrue_indices = 0, nfalse_indices = 0;
    SHADE_LOOP_BEGIN
        if (condition[i])
            false_runflags[i] = RunflagOff;
        else
            true_runflags[i] = RunflagOff;
    SHADE_LOOP_END
#elif USE_RUNINDICES
    Runflag *true_runflags = NULL;
    Runflag *false_runflags = NULL;
    RunIndex *true_indices = ALLOCA (RunIndex, exec->runstate().nindices);
    RunIndex *false_indices = ALLOCA (RunIndex, exec->runstate().nindices);
    int ntrue_indices = 0, nfalse_indices = 0;
    SHADE_LOOP_BEGIN
        if (condition[i])
            true_indices[ntrue_indices++] = i;
        else
            false_indices[nfalse_indices++] = i;
    SHADE_LOOP_END
#elif USE_RUNSPANS
    Runflag *true_runflags = NULL;
    Runflag *false_runflags = NULL;
    RunIndex *true_indices = ALLOCA (RunIndex, 2*(exec->endpoint()-exec->beginpoint()));
    true_indices[0] = 0;
    RunIndex *false_indices = ALLOCA (RunIndex, 2*(exec->endpoint()-exec->beginpoint()));
    false_indices[0] = 0;
    int ntrue_indices = 0, nfalse_indices = 0;
    spans_runflags_to_spans (exec->runstate().indices,
                             exec->runstate().nindices, condition,
                             true_indices, ntrue_indices, 1);
    spans_runflags_to_spans (exec->runstate().indices,
                             exec->runstate().nindices, condition,
                             false_indices, nfalse_indices, 0);
#endif

    exec->enter_conditional ();

    // True clause
    exec->push_runflags (true_runflags, exec->runstate().beginpoint, exec->runstate().endpoint,
                         true_indices, ntrue_indices);
    exec->run (exec->ip() + 1, op.jump(0));
    exec->pop_runflags ();

    // False clause
    if (op.jump(0) < op.jump(1)) {
        exec->push_runflags (false_runflags, exec->runstate().beginpoint, exec->runstate().endpoint,
                             false_indices, nfalse_indices);
        exec->run (op.jump(0), op.jump(1));
        exec->pop_runflags ();
    }

    // Jump to after the if (remember that the interpreter loop will
    // increment the ip one more time, so back up one.
    exec->ip (op.jump(1) - 1);
    exec->exit_conditional ();

    // FIXME -- we may need to call new_runflag_range here if, during
    // execution, we may have hit a 'break' or 'continue'.
}



DECLOP (OP_for)
{
    ASSERT (nargs == 1);
    Symbol &Condition (exec->sym (args[0]));
    ASSERT (Condition.typespec().is_int());
    Opcode &op (exec->op());

    // Jump addresses
    int startinit = exec->ip() + 1;
    int startcondition = op.jump (0);
    int startbody = op.jump (1);
    int startiterate = op.jump (2);
    int done = op.jump (3);

    // Execute the initialization
    if (startinit < startcondition)
        exec->run (startinit, startcondition);

#if USE_RUNFLAGS
    Runflag *true_runflags = ALLOCA (Runflag, exec->npoints());
    memcpy (true_runflags, exec->runstate().runflags,
            exec->npoints() * sizeof(Runflag));
    Runflag *old_runflags = ALLOCA (Runflag, exec->npoints());
    int *true_indices = NULL;
    bool all_on = true;
#else // both indices and spans...
    Runflag *true_runflags = NULL;
#if USE_RUNINDICES
    int *true_indices = ALLOCA (RunIndex, exec->runstate().nindices);
#else
    int *true_indices = ALLOCA (RunIndex, exec->endpoint()-exec->beginpoint());
    // N.B. Max number is if every other point is on, and we need 2 entries
    // per span.  More points can only decrease the number of spans!
#endif
    memcpy (true_indices, exec->runstate().indices,
            exec->runstate().nindices * sizeof(RunIndex));
    int *old_indices = ALLOCA (RunIndex, exec->runstate().nindices);
    int old_nindices = exec->runstate().nindices;
#endif
    bool pushed_runstate = false;  // Did we push runflags already?

    while (1) {
        // Execute the condition
        exec->run (startcondition, startbody);

        // Determine if it's a "uniform if"
        bool uniform = Condition.is_uniform ();
        VaryingRef<int> condition ((int *)Condition.data(), Condition.step());

        // FIXME -- if there's potentially a 'break' or 'continue' inside
        // this loop, we need to treat it as varying.

        if (uniform /* and no possible break/continue */) {
            // Uniform condition -- don't need new runflags
            if (condition[exec->runstate().beginpoint])
                exec->run (startbody, startiterate);  // Run the body
            else
                break;   // break out of the loop

        } else {
            // From here on, varying condition or potential
            // break/continue at play

            // FIXME -- we should check if the condition is true
            // everywhere, and there isn't a break/continue in the loop,
            // we don't need to make new runflags.

            // Generate new runflags based on the condition
            bool diverged = false;  // Have we turned any points off?
            bool all_off = true;  // Are all points turned off?
            int nindices = 0;
#if USE_RUNFLAGS
            // Save runflags
            memcpy (old_runflags, true_runflags, exec->npoints() * sizeof(Runflag));
            // Examine the condition, turn off runflags where cond was false
            SHADE_LOOP_RUNFLAGS_BEGIN (old_runflags, 
                                       exec->runstate().beginpoint,
                                       exec->runstate().endpoint)
                if (condition[i]) {
                    all_off = false;  // this point is still on
                } else {
                    // this point has turned off on this iteration
                    true_runflags[i] = RunflagOff;
                    all_on = false;
                    diverged = true;
                }
            SHADE_LOOP_END
#elif USE_RUNINDICES
            old_nindices = exec->runstate().nindices;
            memcpy (old_indices, true_indices, old_nindices * sizeof(RunIndex));
            SHADE_LOOP_INDICES_BEGIN (old_indices, old_nindices)
                if (condition[i]) {
                    true_indices[nindices++] = i;
                    all_off = false;
                } else {
                    diverged = true;
                }
            SHADE_LOOP_END
#elif USE_RUNSPANS
            old_nindices = exec->runstate().nindices;
            memcpy (old_indices, true_indices, old_nindices * sizeof(RunIndex));
            nindices = 0;
            diverged |= spans_runflags_to_spans (old_indices, old_nindices,
                                                 condition,
                                                 true_indices, nindices);
            all_off = (nindices == 0);
#endif

            if (all_off)
                break;     // No points left on

            if (pushed_runstate) {
                exec->pop_runflags ();
                exec->exit_conditional ();
                pushed_runstate = false;
            }

            if (diverged) {
                exec->push_runflags (true_runflags, exec->runstate().beginpoint,
                                     exec->runstate().endpoint,
                                     true_indices, nindices);
                exec->enter_conditional ();
                pushed_runstate = true;
            }

            // Execute the body
            exec->run (startbody, startiterate);

            // FIXME -- we may need to call new_runflag_range here if, during
            // execution, we may have hit a 'break' or 'continue'.

        }
        if (startiterate < done)
            exec->run (startiterate, done);
    }

    if (pushed_runstate) {
        exec->pop_runflags ();
        exec->exit_conditional ();
    }

    // Skip to after the loop
    exec->ip (done-1);
}



DECLOP (OP_dowhile)
{
    ASSERT (nargs == 1);
    Symbol &Condition (exec->sym (args[0]));
    ASSERT (Condition.typespec().is_int());
    Opcode &op (exec->op());

    // Jump addresses
    int startinit = exec->ip() + 1;
    int startcondition = op.jump (0);
    int startbody = op.jump (1);
    int startiterate = op.jump (2);
    int done = op.jump (3);

    // Execute the initialization
    if (startinit < startcondition)
        exec->run (startinit, startcondition);

#if USE_RUNFLAGS
    Runflag *true_runflags = ALLOCA (Runflag, exec->npoints());
    memcpy (true_runflags, exec->runstate().runflags,
            exec->npoints() * sizeof(Runflag));
    Runflag *old_runflags = ALLOCA (Runflag, exec->npoints());
    int *true_indices = NULL;
    bool all_on = true;
#else /* good for indices and spans */
    Runflag *true_runflags = NULL;
#if USE_RUNINDICES
    int *true_indices = ALLOCA (RunIndex, exec->runstate().nindices);
#else
    int *true_indices = ALLOCA (RunIndex, exec->endpoint()-exec->beginpoint());
    // N.B. Max number is if every other point is on, and we need 2 entries
    // per span.  More points can only decrease the number of spans!
#endif
    memcpy (true_indices, exec->runstate().indices,
            exec->runstate().nindices * sizeof(RunIndex));
    int *old_indices = ALLOCA (RunIndex, exec->runstate().nindices);
    int old_nindices = exec->runstate().nindices;
#endif
    bool pushed_runstate = false;  // Did we push runflags already?

    while (1) {
        // Execute the body
        exec->run (startbody, startiterate);
            
        // Execute the condition
        exec->run (startcondition, startbody);

        // Determine if it's a "uniform if"
        bool uniform = Condition.is_uniform ();
        VaryingRef<int> condition ((int *)Condition.data(), Condition.step());

        // FIXME -- if there's potentially a 'break' or 'continue' inside
        // this loop, we need to treat it as varying.

        if (uniform) {
            // Uniform condition -- don't need new runflags
            if (condition[exec->runstate().beginpoint])
                continue;    // All true, back to the beginning
            else
                break;       // All false, break out of the loop
        } else {
            // From here on, varying condition or potential
            // break/continue at play

            // FIXME -- we should check if the condition is true
            // everywhere, and there isn't a break/continue in the loop,
            // we don't need to make new runflags.

            // Generate new runflags based on the condition
            bool diverged = false;  // Have we turned any points off?
            bool all_off = true;  // Are all points turned off?
            int nindices = 0;
#if USE_RUNFLAGS
            // Save runflags
            memcpy (old_runflags, true_runflags, exec->npoints() * sizeof(Runflag));
            // Examine the condition, turn off runflags where cond was false
            SHADE_LOOP_RUNFLAGS_BEGIN (old_runflags, 
                                       exec->runstate().beginpoint,
                                       exec->runstate().endpoint)
                if (condition[i]) {
                    all_off = false;  // this point is still on
                } else {
                    // this point has turned off on this iteration
                    true_runflags[i] = RunflagOff;
                    all_on = false;
                    diverged = true;
                }
            SHADE_LOOP_END
#elif USE_RUNINDICES
            old_nindices = exec->runstate().nindices;
            memcpy (old_indices, true_indices, old_nindices * sizeof(RunIndex));
            SHADE_LOOP_INDICES_BEGIN (old_indices, old_nindices)
                if (condition[i]) {
                    true_indices[nindices++] = i;
                    all_off = false;
                } else {
                    diverged = true;
                }
            SHADE_LOOP_END
#elif USE_RUNSPANS
            old_nindices = exec->runstate().nindices;
            memcpy (old_indices, true_indices, old_nindices * sizeof(RunIndex));
            diverged |= spans_runflags_to_spans (old_indices, old_nindices,
                                                 condition,
                                                 true_indices, nindices);
            all_off = (nindices == 0);
#endif

            if (all_off)
                break;     // No points left on

            if (pushed_runstate) {
                exec->pop_runflags ();
                exec->exit_conditional ();
                pushed_runstate = false;
            }

            if (diverged) {
                exec->push_runflags (true_runflags, exec->runstate().beginpoint,
                                     exec->runstate().endpoint,
                                     true_indices, nindices);
                exec->enter_conditional ();
                pushed_runstate = true;
            }

            // FIXME -- we may need to call new_runflag_range here if, during
            // execution, we may have hit a 'break' or 'continue'.
        }
    }

    if (pushed_runstate) {
        exec->pop_runflags ();
        exec->exit_conditional ();
    }

    // Skip to after the loop
    exec->ip (done-1);
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
