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

#include "OSL/accum.h"
#include "OSL/oslclosure.h"
#include "lpeparse.h"
#include <OpenImageIO/dassert.h>


OSL_NAMESPACE_ENTER



void
AovOutput::flush(void *flush_data)
{
    if (!aov)
        return;
    if (neg_color) {
        color.setValue(1.0f - color.x, 1.0f - color.y, 1.0f - color.z);
        has_color = true;
    }
    if (neg_alpha) {
        alpha = 1.0f - alpha;
        has_alpha = true;
    }

    aov->write(flush_data, color, alpha, has_color, has_alpha);
}



void
AccumRule::accum(const Color3 &color, std::vector<AovOutput> &outputs)const
{
    if (m_save_to_alpha) {
        outputs[m_outidx].alpha += (color.x + color.y + color.z) * 1.0f/3.0f;
        outputs[m_outidx].has_alpha = true;
    } else {
        outputs[m_outidx].color += color;
        outputs[m_outidx].has_color = true;
    }
}




AccumAutomata::~AccumAutomata()
{
    for (std::list<lpexp::Rule *>::iterator i = m_rules.begin(); i != m_rules.end(); ++i)
        delete *i;
}



AccumRule *
AccumAutomata::addRule(const char *pattern, int outidx, bool toalpha)
{
    // First parse the lpexp and see if it fails
    Parser parser(&m_user_events, &m_user_scatterings);
    LPexp *e = parser.parse(pattern);
    if (parser.error()) {
        std::cerr << "[pathexp] Parse error" << parser.getErrorMsg() << " at char " << parser.getErrorPos() << std::endl;
        delete e;
        return NULL;
    }
    m_accumrules.push_back (AccumRule (outidx, toalpha));
    // it is a list, so as long as we don't remove it from there, the pointer is valid
    void *rule = (void *)&(m_accumrules.back());
    m_rules.push_back (new lpexp::Rule (e, rule));
    return &(m_accumrules.back());
}



void
AccumAutomata::compile()
{
    NdfAutomata ndfautomata;
    for (std::list<lpexp::Rule *>::const_iterator i = m_rules.begin(); i != m_rules.end(); ++i) {
        (*i)->genAuto(ndfautomata);
        delete *i;
    }
    // Nuke the compiled regexps, we don't need them anymore
    m_rules.clear();
    DfAutomata dfautomata;
    ndfautoToDfauto(ndfautomata, dfautomata);
    m_dfoptautomata.compileFrom(dfautomata);
}



void
AccumAutomata::accum(int state, const Color3 &color, std::vector<AovOutput> &outputs)const
{
    // get the rules field, the underlying type is a std::vector
    int nrules = 0;
    void * const * rules = getRulesInState(state, nrules);
    // Iterate the vector
    for (int i = 0; i < nrules; ++i)
        // Let the accumulator rule do its job
        ((AccumRule *)(rules[i]))->accum(color, outputs);
}



Accumulator::Accumulator(const AccumAutomata *accauto):m_accum_automata(accauto)
{
    const std::list<AccumRule> &rules = m_accum_automata->getRuleList();
    // Make sure we have as many outputs as the rules need
    int maxouts = 0;
    for (std::list<AccumRule>::const_iterator i =  rules.begin(); i != rules.end(); ++i)
        maxouts = i->getOutputIndex() > maxouts ? i->getOutputIndex() : maxouts;
    m_outputs.resize(maxouts+1);

    // 0 is our initial state always
    m_state = 0;
}



void
Accumulator::setAov(int outidx, Aov *aov, bool neg_color, bool neg_alpha)
{
    ASSERT (0 <= outidx && outidx < (int) m_outputs.size());
    m_outputs[outidx].aov = aov;
    m_outputs[outidx].neg_color = neg_color;
    m_outputs[outidx].neg_alpha = neg_alpha;
}



void
Accumulator::pushState()
{
    ASSERT (m_state >= 0);
    m_stack.push(m_state);
}



void
Accumulator::popState()
{
    ASSERT (m_stack.size());
    m_state = m_stack.top();
    m_stack.pop();
}



void
Accumulator::move(ustring symbol)
{
    if (m_state >= 0)
        m_state = m_accum_automata->getTransition(m_state, symbol);
}



void
Accumulator::move(const ustring *symbols)
{
    while (m_state >= 0 && symbols && *symbols != Labels::NONE)
        m_state = m_accum_automata->getTransition(m_state, *(symbols++));
}



void
Accumulator::move(ustring event, ustring scatt, const ustring *custom, ustring stop)
{
    if (m_state >= 0)
        m_state = m_accum_automata->getTransition(m_state, event);
    if (m_state >= 0)
        m_state = m_accum_automata->getTransition(m_state, scatt);
    while (m_state >= 0 && custom && *custom != Labels::NONE)
        m_state = m_accum_automata->getTransition(m_state, *(custom++));
    if (m_state >= 0)
        m_state = m_accum_automata->getTransition(m_state, stop);
}



void
Accumulator::begin()
{
    for (size_t i = 0; i < m_outputs.size(); ++i)
        m_outputs[i].reset();
}



void
Accumulator::end(void *flush_data)
{
    for (size_t i = 0; i < m_outputs.size(); ++i)
        m_outputs[i].flush(flush_data);
}

OSL_NAMESPACE_EXIT
