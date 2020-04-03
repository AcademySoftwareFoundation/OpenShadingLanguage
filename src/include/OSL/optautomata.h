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

#pragma once

#include <OpenImageIO/ustring.h>

#include "oslcontainers.h"

OSL_NAMESPACE_ENTER

class DfAutomata;


/// Optimized compact version of DfAutomata
///
/// Apparently hash maps suck in speed for our transition tables. This
/// is a fast compact equivalent of the DfAutomata designed for read
/// only operations.
///
class DfOptimizedAutomata
{
    public:

        void compileFrom(const DfAutomata &dfautomata);

        int getTransition(int state, ustring symbol)const
        {
            const State &mystate = m_states[state];
            const Transition *begin = &m_trans[mystate.begin_trans];
            const Transition *end = begin + mystate.ntrans;
            while (begin < end) { // binary search
                const Transition *middle = begin + ((end - begin)>>1);
                if (symbol.data() < middle->symbol.data())
                    end = middle;
                else if (middle->symbol.data() < symbol.data())
                    begin = middle + 1;
                else // match
                    return middle->state;
            }
            return mystate.wildcard_trans;
        }

        void * const * getRules(int state, int &count)const
        {
            count = m_states[state].nrules;
            return &m_rules[m_states[state].begin_rules];
        }

    protected:
        struct State
        {
            unsigned int begin_trans;
            unsigned int ntrans;
            unsigned int begin_rules;
            unsigned int nrules;
            int wildcard_trans;
        };
        struct Transition
        {
            // we use this only for sorting
            static bool trans_comp (const Transition &a, const Transition &b);
            ustring symbol;
            int      state;
        };
        pvt::vector<Transition> m_trans;
        pvt::vector<void *>     m_rules;
        pvt::vector<State>      m_states;
};

OSL_NAMESPACE_EXIT
