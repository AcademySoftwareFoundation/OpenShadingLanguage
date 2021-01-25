// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "lpexp.h"


OSL_NAMESPACE_ENTER




lpexp::FirstLast
lpexp::Cat::genAuto(NdfAutomata &automata)const
{
    NdfAutomata::State * first = NULL;
    NdfAutomata::State * last = NULL;
    // Sequentially create the states for the expressions and link them all by
    // lambda transitions. Making the begin state of the first one our begin, and the
    // end state of the last one our end
    for (auto child : m_children) {
        FirstLast fl = child->genAuto(automata);
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
    for (auto& child : m_children)
        delete child;
}



lpexp::LPexp *
lpexp::Cat::clone()const
{
    Cat *newcat = new Cat();
    for (auto child : m_children)
        newcat->append(child->clone());
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
    for (auto child : m_children) {
        // And then for every child we create its part of automata and link our begin to its
        // begin and its end to our end with lambda transitions
        FirstLast fl = child->genAuto(automata);
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
    for (auto& child : m_children)
        delete child;
}



lpexp::LPexp *
lpexp::Orlist::clone()const
{
    Orlist *newor = new Orlist();
    for (auto child : m_children)
        newor->append(child->clone());
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
    // And then n - m additional movements. But we make them optional using
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


OSL_NAMESPACE_EXIT
