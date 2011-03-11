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

#include "lpexp.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {



lpexp::FirstLast
lpexp::Cat::genAuto(NdfAutomata &automata)const
{
    NdfAutomata::State * first = NULL;
    NdfAutomata::State * last = NULL;
    // Sequentially create the states for the expressions and link them all by
    // lambda transitions. Making the begin state of the first one our begin, and the
    // end state of the last one our end
    for (std::list<LPexp *>::const_iterator i = m_children.begin(); i != m_children.end(); ++i) {
        FirstLast fl = (*i)->genAuto(automata);
        if (!first)
            first = fl.first;
        else
            // This is not the first of the list, so link it from the previous
            // one end state
            last->addTransition(lambda, fl.first);
        last = fl.second;
    }
    return FirstLast(first, last);
}



void
lpexp::Cat::append(LPexp *lpexp)
{
    m_children.push_back(lpexp);
}



lpexp::Cat::~Cat()
{
    for (std::list<LPexp *>::iterator i = m_children.begin(); i != m_children.end(); ++i)
        delete *i;
}



lpexp::LPexp *
lpexp::Cat::clone()const
{
    Cat *newcat = new Cat();
    for (std::list<LPexp *>::const_iterator i = m_children.begin(); i != m_children.end(); ++i)
        newcat->append((*i)->clone());
    return newcat;
}



lpexp::FirstLast
lpexp::Symbol::genAuto(NdfAutomata &automata)const
{
    // Easiest lpexp ever. Two new states, than join the first to
    // the second with the symbol we got
    NdfAutomata::State *begin = automata.newState();
    NdfAutomata::State *end    = automata.newState();
    begin->addTransition(m_sym, end);
    return FirstLast(begin, end);
}



lpexp::FirstLast
lpexp::Wildexp::genAuto(NdfAutomata &automata)const
{
    // Same as the Symbol lpexp but with a wildcard insted of a symbol
    NdfAutomata::State *begin = automata.newState();
    NdfAutomata::State *end    = automata.newState();
    begin->addWildcardTransition(new Wildcard(m_wildcard), end);
    return FirstLast(begin, end);
}



lpexp::FirstLast
lpexp::Orlist::genAuto(NdfAutomata &automata)const
{
    // Cat was like a serial circuit and this is a parallel one. We need
    // two new states begin and end
    NdfAutomata::State *begin = automata.newState();
    NdfAutomata::State *end = automata.newState();
    for (std::list<LPexp *>::const_iterator i = m_children.begin(); i != m_children.end(); ++i) {
        // And then for every child we create its part of automata and link our begin to its
        // begin and its end to our end with lambda transitions
        FirstLast fl = (*i)->genAuto(automata);
        begin->addTransition(lambda, fl.first);
        fl.second->addTransition(lambda, end);
    }
    return FirstLast(begin, end);
}



void
lpexp::Orlist::append(LPexp *lpexp)
{
    m_children.push_back(lpexp);
}



lpexp::Orlist::~Orlist()
{
    for (std::list<LPexp *>::iterator i = m_children.begin(); i != m_children.end(); ++i)
        delete *i;
}



lpexp::LPexp *
lpexp::Orlist::clone()const
{
    Orlist *newor = new Orlist();
    for (std::list<LPexp *>::const_iterator i = m_children.begin(); i != m_children.end(); ++i)
        newor->append((*i)->clone());
    return newor;
}



lpexp::FirstLast
lpexp::Repeat::genAuto(NdfAutomata &automata)const
{
    NdfAutomata::State *begin = automata.newState();
    NdfAutomata::State *end = automata.newState();
    FirstLast fl = m_child->genAuto(automata);
    begin->addTransition(lambda, fl.first);
    fl.second->addTransition(lambda, end);
    // Easy, make its begin and end states almost the same with
    // lambda transitions so it can repeat for ever
    begin->addTransition(lambda, end);
    end->addTransition(lambda, begin);
    return FirstLast(begin, end);
}



lpexp::FirstLast
lpexp::NRepeat::genAuto(NdfAutomata &automata)const
{
    NdfAutomata::State *first = NULL;
    NdfAutomata::State *last  = NULL;
    int i;
    // This is a bit trickier. For {m.n} we first make a concatenation of
    // the child expression m times
    for (i = 0; i < m_min; ++i) {
        FirstLast fl = m_child->genAuto(automata);
        if (!first)
            first = fl.first;
        else
            last->addTransition(lambda, fl.first);
        last = fl.second;
    }
    // And then n - m aditional movements. But we make them optional using
    // lambda transitions
    if (!last && i < m_max)
        first = last = automata.newState();
    for (; i < m_max; ++i) {
        FirstLast fl = m_child->genAuto(automata);
        last->addTransition(lambda, fl.first);
        // Since this repetitions are optional, put a bypass with lambda
        last->addTransition(lambda, fl.second);
        last = fl.second;
    }
    return FirstLast(first, last);
}



void
lpexp::Rule::genAuto(NdfAutomata &automata)const
{
    // First generate the actual automata
    FirstLast fl = m_child->genAuto(automata);
    // now, put the rule in the last state (making it a final state)
    fl.second->setRule(m_rule);
    // And then make its begin state accessible from the master initial state
    // of the automata so it becomes initial too
    automata.getInitial()->addTransition(lambda, fl.first);
}

}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
