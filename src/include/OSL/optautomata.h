// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OpenImageIO/ustring.h>

#include <OSL/export.h>
#include <OSL/oslversion.h>

#include <vector>

OSL_NAMESPACE_ENTER

class DfAutomata;


/// Optimized compact version of DfAutomata
///
/// Apparently hash maps suck in speed for our transition tables. This
/// is a fast compact equivalent of the DfAutomata designed for read
/// only operations.
///
class OSLEXECPUBLIC DfOptimizedAutomata {
public:
    void compileFrom(const DfAutomata& dfautomata);

    int getTransition(int state, OIIO::ustring symbol) const
    {
        const State& mystate    = m_states[state];
        const Transition* begin = &m_trans[mystate.begin_trans];
        const Transition* end   = begin + mystate.ntrans;
        while (begin < end) {  // binary search
            const Transition* middle = begin + ((end - begin) >> 1);
            if (symbol.data() < middle->symbol.data())
                end = middle;
            else if (middle->symbol.data() < symbol.data())
                begin = middle + 1;
            else  // match
                return middle->state;
        }
        return mystate.wildcard_trans;
    }

    void* const* getRules(int state, int& count) const
    {
        count = m_states[state].nrules;
        return &m_rules[m_states[state].begin_rules];
    }

protected:
    struct State {
        unsigned int begin_trans;
        unsigned int ntrans;
        unsigned int begin_rules;
        unsigned int nrules;
        int wildcard_trans;
    };
    struct Transition {
        // we use this only for sorting
        static bool trans_comp(const Transition& a, const Transition& b);
        OIIO::ustring symbol;
        int state;
    };
    std::vector<Transition> m_trans;
    std::vector<void*> m_rules;
    std::vector<State> m_states;
};

OSL_NAMESPACE_EXIT
