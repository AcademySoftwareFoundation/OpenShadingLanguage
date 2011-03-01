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

#include "automata.h"
#include "optautomata.h"
#include <algorithm>
#include <cstdio>


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {

#ifdef OIIO_NAMESPACE
namespace Strutil = OIIO::Strutil;
#endif

#ifdef _MSC_VER
#define snprintf sprintf_s
#endif

ustring lambda("__lambda__");

void
NdfAutomata::State::getTransitions(ustring symbol, IntSet &out_states)const
{
    SymbolToIntList::const_iterator s = m_symbol_trans.find(symbol);
    if (s != m_symbol_trans.end())
        for (IntSet::const_iterator i = s->second.begin(); i != s->second.end(); ++i)
            out_states.insert(*i);
    if (m_wildcard && m_wildcard->matches(symbol))
        out_states.insert(m_wildcard_trans);
}


static IntSet _emptyset;

std::pair <IntSet::const_iterator, IntSet::const_iterator>
NdfAutomata::State::getLambdaTransitions ()const
{
    std::pair <IntSet::const_iterator, IntSet::const_iterator> res;
    SymbolToIntList::const_iterator s = m_symbol_trans.find(lambda);
    if (s != m_symbol_trans.end()) {
        res.first = s->second.begin();
        res.second = s->second.end();
    }
    else
        // We use a static empty list to return an empty range
        res.first = res.second = _emptyset.end();
    return res;
}



void
NdfAutomata::State::addTransition(ustring symbol, NdfAutomata::State *state)
{
    m_symbol_trans[symbol].insert(state->m_id);
}



void
NdfAutomata::State::addWildcardTransition(Wildcard *wildcard, NdfAutomata::State *state)
{
    if (m_wildcard)
        std::cerr << "[pathexp] redefining wildcard transition" << std::endl;
    m_wildcard = wildcard;
    m_wildcard_trans = state->m_id;
}



std::string
NdfAutomata::State::tostr()const
{
    std::string s = "";
    // output the transitions
    for (SymbolToIntList::const_iterator i = m_symbol_trans.begin(); i != m_symbol_trans.end(); ++i) {
        ustring sym = i->first;
        const IntSet &dest = i->second;
        if (s.size())
            s += " ";
        if (sym == lambda)
            s += "@";
        else
            s += sym.c_str();
        s += ":{";
        for (IntSet::const_iterator j = dest.begin(); j != dest.end(); ++j) {
            if (s[s.size()-1] != '{')
                s += ", ";
            s += Strutil::format("%d", *j);
        }
        s += "}";
    }
    // In case there is a wildcard transition ...
    if (m_wildcard) {
        if (s.size())
            s += " ";
        // No symbols in the black list, print just .
        if (m_wildcard->m_minus.empty())
            s += ".:";
        else {
            // Standard regexp notation [^abcd]
            s += "[^";
            for (SymbolSet::const_iterator i = m_wildcard->m_minus.begin(); i != m_wildcard->m_minus.end(); ++i) {
                if (!i->c_str())
                    s += "_";
                else
                    s += i->c_str();
            }
            s += "]:";
        }
        s += Strutil::format("%d", m_wildcard_trans);
    }
    // and finally the rule if we have it
    if (m_rule) {
        s += " | ";
        s += Strutil::format("%lx", (long unsigned int)m_rule);
    }
    return s;
}



NdfAutomata::State *
NdfAutomata::newState()
{
    m_states.push_back(new State(m_states.size()));
    return m_states.back();
}



void
NdfAutomata::symbolsFrom(const IntSet &states, SymbolSet &out_symbols, Wildcard *&wildcard)const
{
    for (IntSet::const_iterator i = states.begin(); i != states.end(); ++i) {
        const State *state = m_states[*i];
        // For every state we have to go thorugh all the symbols in the transition table
        // m_symbol_trans and add them to the output
        for (SymbolToIntList::const_iterator j = state->m_symbol_trans.begin(); j != state->m_symbol_trans.end(); ++j)
            if (j->first != lambda)
                out_symbols.insert(j->first);
        if (state->m_wildcard) {
            if (!wildcard)
                wildcard = new Wildcard();
            // And if the state has a wildcard movement, we need to add its exclusion list
            // to our new computed exclusion list. So we guarantee that the returned wildcard
            // matches will be contained in all the wildcards out of this set
            wildcard->m_minus.insert(state->m_wildcard->m_minus.begin(), state->m_wildcard->m_minus.end());
        }
    }
    if (wildcard) {
        // We have to make sure that all the symbols covered by the wildcards
        // are either covered by our wildcard or in out_symbols set
        for (IntSet::const_iterator i = states.begin(); i != states.end(); ++i) {
            const State *state = m_states[*i];
            if (state->m_wildcard)
                for (SymbolSet::const_iterator j = wildcard->m_minus.begin(); j != wildcard->m_minus.end(); ++j)
                    if (state->m_wildcard->matches(*j))
                        out_symbols.insert(*j);
        }
        // And don't forget about the symbols which are already in the transitions
        wildcard->m_minus.insert(out_symbols.begin(), out_symbols.end());
    }
}



void
NdfAutomata::transitionsFrom(const IntSet &states, ustring symbol, IntSet &out_states)const
{
    for (IntSet::const_iterator i = states.begin(); i != states.end(); ++i)
        // remember getTransitions is not destructive with out_states, it just adds stuff
        m_states[*i]->getTransitions(symbol, out_states);

    lambdaClosure(out_states);
}



void
NdfAutomata::wildcardTransitionsFrom(const IntSet &states, IntSet &out_states)const
{
    for (IntSet::const_iterator i = states.begin(); i != states.end(); ++i) {
        const State *state = m_states[*i];
        if (state->m_wildcard)
            out_states.insert(state->m_wildcard_trans);
    }
    lambdaClosure(out_states);
}



void
NdfAutomata::lambdaClosure(IntSet &states)const
{
    // This algorithm basically keeps expanding the set until no new states appear
    // to avoid checking over and over the same states we keep a frontier pair of sets
    // so we only expand newly discovered states
    std::vector<int> frontier, discovered;
    // First iterate all the states in the given set
    // and see what lambda transitions are there
    for (IntSet::const_iterator i = states.begin(); i != states.end(); ++i) {
        const State *state = m_states[*i];
        std::pair <IntSet::const_iterator, IntSet::const_iterator> lr;
        // iterate all lambda transitions for this state
        for (lr = state->getLambdaTransitions(); lr.first != lr.second; lr.first++) {
            // Add them to the set, and if they were not already there add to the
            // frontier
            std::pair<IntSet::iterator, bool> rec = states.insert(*(lr.first));
            if (rec.second) // newly added
                frontier.push_back(*(lr.first));
        }
    }
    // frontier becomes last discovered
    frontier.swap(discovered); // swap discovered and frontier
    while (discovered.size()) { // as long as there are new found states
        frontier.clear();
        // we do the same as in the above loop but with discovered instead of states
        for (std::vector<int>::iterator i = discovered.begin(); i != discovered.end(); ++i) {
            const State *state = m_states[*i];
            std::pair <IntSet::const_iterator, IntSet::const_iterator> lr;
            for (lr = state->getLambdaTransitions(); lr.first != lr.second; lr.first++) {
                std::pair<IntSet::iterator, bool> rec = states.insert(*(lr.first));
                if (rec.second)
                    frontier.push_back(*(lr.first));
            }

        }
        // again frontier becomes last discovered
        frontier.swap(discovered); // swap discovered and frontier
    }
}



std::string
NdfAutomata::tostr()const
{
    std::string s;
    for (size_t i = 0; i < m_states.size(); ++i) {
        char temp[32];
        snprintf(temp, 32, "%ld : ", i);
        s += temp + m_states[i]->tostr() + "\n";
    }
    return s;
}



NdfAutomata::~NdfAutomata()
{
    for (std::vector<State *>::iterator i = m_states.begin(); i != m_states.end(); ++i)
        delete *i;
}



void keyFromStateSet(const IntSet &states, StateSetKey &out_key)
{
    out_key.clear(); // just in case
    for (IntSet::const_iterator i = states.begin(); i != states.end(); ++i)
        out_key.push_back(*i);
    // Sort the ids so we make sure the vector is unique for each set
    sort(out_key.begin(), out_key.end());
}



int
DfAutomata::State::getTransition(ustring symbol)const
{
    SymbolToInt::const_iterator i = m_symbol_trans.find(symbol);
    if (i == m_symbol_trans.end())
        // in case there is a wildcard (!= -1) and the symbol is not
        // tagged as -1 (not found), then follow the wildcard
        return m_wildcard_trans;
    else
        // it already has -1 if it is in the wildcard's black list
        return i->second;
}



void
DfAutomata::State::addTransition(ustring symbol, DfAutomata::State *state)
{
    SymbolToInt::value_type value(symbol, state->m_id);
    std::pair<SymbolToInt::iterator, bool> place = m_symbol_trans.insert(value);
    if (!place.second)
        std::cerr << "[pathexp] overwriting a transition in a DF automata" << std::endl;
}



void
DfAutomata::State::addWildcardTransition(Wildcard *wildcard, DfAutomata::State *state)
{
    for (SymbolSet::const_iterator i = wildcard->m_minus.begin(); i != wildcard->m_minus.end(); ++i)
        // optimized storage, if it is not already in the transition table, tag it with -1
        if (m_symbol_trans.find(*i) == m_symbol_trans.end())
            m_symbol_trans[*i] = -1;
    m_wildcard_trans = state->m_id;
    delete wildcard;
}



void
DfAutomata::State::removeUselessTransitions()
{
    if (m_wildcard_trans >= 0) {
        std::list<SymbolToInt::iterator> toremove;
        for (SymbolToInt::iterator i = m_symbol_trans.begin(); i != m_symbol_trans.end(); ++i)
            // If there is a transition to the same state as the wildcard, we better nuke it
            // and just add that symbol to the wildcard be removing it from the map itself
            if (i->second == m_wildcard_trans)
                toremove.push_back(i);
        for (std::list<SymbolToInt::iterator>::iterator i = toremove.begin(); i != toremove.end(); ++i)
            m_symbol_trans.erase(*i);
    }
}



std::string
DfAutomata::State::tostr()const
{
    std::string s = "";
    // normal transitions
    for (SymbolToInt::const_iterator i = m_symbol_trans.begin(); i != m_symbol_trans.end(); ++i) {
        ustring sym = i->first;
        int dest = i->second;
        if (s.size())
            s += " ";
        if (sym == lambda)
            s += "@";
        else
            s += sym.c_str();
        s += ":";
        s += Strutil::format("%d", dest);
    }
    // wildcard
    if (m_wildcard_trans >= 0) {
        if (s.size())
            s += " ";
        if (m_symbol_trans.empty())
            s += ".:";
        else {
            s += "[^";
            for (SymbolToInt::const_iterator i = m_symbol_trans.begin(); i != m_symbol_trans.end(); ++i) {
                if (!i->first.c_str())
                    s += "_";
                else
                    s += i->first.c_str();
            }
            s += "}:";
        }
        s += Strutil::format("%d", m_wildcard_trans);
    }
    // and the rules
    if (m_rules.size()) {
        s += " | [";
        for (RuleSet::const_iterator i = m_rules.begin(); i != m_rules.end(); ++i) {
            if (s[s.size()-1] != '[')
                s += ", ";
            s += Strutil::format("%lx", (long unsigned int)*i);
        }
        s += "]";
    }
    return s;
}



DfAutomata::State *
DfAutomata::newState()
{
    m_states.push_back(new State(m_states.size()));
    return m_states.back();
}



std::string
DfAutomata::tostr()const
{
    std::string s;
    for (size_t i = 0; i < m_states.size(); ++i) {
        char temp[32];
        snprintf(temp, 32, "%ld : ", i);
        s += temp + m_states[i]->tostr() + "\n";
    }
    return s;
}



bool
DfAutomata::equivalent(const State *dfstateA, const State *dfstateB)
{
    // early exit if the size of the tables is different
    if (dfstateA->m_symbol_trans.size() != dfstateB->m_symbol_trans.size())
        return false;
    // The pointed state by both transitions have to be the same or any of dfstateA and dfstateB
    int destA = (dfstateA->m_wildcard_trans == dfstateA->getId() || dfstateA->m_wildcard_trans == dfstateB->getId()) ? -2 : dfstateA->m_wildcard_trans;
    int destB = (dfstateB->m_wildcard_trans == dfstateA->getId() || dfstateB->m_wildcard_trans == dfstateB->getId()) ? -2 : dfstateB->m_wildcard_trans;
    if (destA != destB)
        return false;
    // Rules have to be the same
    if (dfstateA->m_rules != dfstateB->m_rules)
        return false;
    for (SymbolToInt::const_iterator i = dfstateA->m_symbol_trans.begin(); i != dfstateA->m_symbol_trans.end(); ++i) {
        SymbolToInt::const_iterator other = dfstateB->m_symbol_trans.find(i->first);
        if (other == dfstateB->m_symbol_trans.end())
            return false;
        // The pointed state by both transitions have to be the same or any of dfstateA and dfstateB
        int destA = (i->second == dfstateA->getId() || i->second == dfstateB->getId()) ? -2 : i->second;
        int destB = (other->second == dfstateA->getId() || other->second == dfstateB->getId()) ? -2 : other->second;
        // when they are -1 is because they are in the wildcard black list, anyway they have to match so ...
        if (destA != destB)
            return false;
    }
    // if everything passed, they are equivalent, congratulations.
    return true;
}



void
DfAutomata::removeEquivalentStates()
{
    std::vector<State *> newstatelist;
    HashIntInt newfromold;

    // First go through all states and delete all those
    // that are equivalent with a previous one
    for (size_t i = 0; i < m_states.size(); ++i) {
        if (!m_states[i]) // it has already been removed
            continue;
        // create a new state id from newstatelist.size()
        // move the pointer there and register the translation
        int newstate = newfromold[i] = newstatelist.size();
        newstatelist.push_back(m_states[i]);
        for (size_t j = i + 1; j < m_states.size(); ++j)
            if (m_states[j] && equivalent(m_states[i], m_states[j])) {
                // put in the record that this state will be known as
                // the one in newstate from now on
                newfromold[j] = newstate;
                delete m_states[j];
                m_states[j] = NULL;
            }
    }
    // Everything has been moved now, but we still have to fix the
    // transitions so they point to the right states!
    for (size_t i = 0; i < newstatelist.size(); ++i) {
        State *state = newstatelist[i];
        for (SymbolToInt::iterator j = state->m_symbol_trans.begin(); j != state->m_symbol_trans.end(); ++j) {
            if (j->second != -1) { // if it is -1 it is just in the wildcards black list
                // Get the new state that maps to the oldstate pointed by the transition
                HashIntInt::const_iterator trans = newfromold.find(j->second);
                if (trans != newfromold.end())
                    j->second = trans->second;
                else
                    std::cerr << "[pathexp] broken translation list between states" << std::endl;
            }
        }
        // Do the same with the wildcard
        if (state->m_wildcard_trans >=0) {
            HashIntInt::const_iterator trans = newfromold.find(state->m_wildcard_trans);
            if (trans != newfromold.end())
                state->m_wildcard_trans = trans->second;
            else
                std::cerr << "[pathexp] broken translation list between states" << std::endl;
        }
    }
    // switch to the new hopefully reduced state vector
    m_states = newstatelist;
}



void
DfAutomata::removeUselessTransitions()
{
    for (size_t i = 0; i < m_states.size(); ++i)
        m_states[i]->removeUselessTransitions();
}



void
DfAutomata::clear()
{
    for (std::vector<State *>::iterator i = m_states.begin(); i != m_states.end(); ++i)
        delete *i;
    m_states.clear();
}



DfAutomata::~DfAutomata()
{
    clear();
}



DfAutomata::State *
StateSetRecord::ensureState(const IntSet &newstates, std::list<StateSetRecord::Discovery> &discovered)
{
    // create the key
    StateSetKey newkey;
    keyFromStateSet(newstates, newkey);
    // check if it is there
    StateSetMap::const_iterator i = m_key_to_dfstate.find(newkey);
    if (i != m_key_to_dfstate.end())
        return i->second;
    else {
        // if not in our records create a new DF state
        DfAutomata::State *tstate = m_dfautomata.newState();
        getRulesFromSet(tstate, m_ndfautomata, newstates);
        m_key_to_dfstate[newkey] = tstate;
        // Add the discovery to the list so it will be explored
        discovered.push_back(Discovery(tstate, newstates));
        return tstate;
    }
}



void
StateSetRecord::getRulesFromSet(DfAutomata::State *dfstate, const NdfAutomata &ndfautomata, const IntSet &ndfstates)
{
    for (IntSet::const_iterator i = ndfstates.begin(); i != ndfstates.end(); ++i) {
        const NdfAutomata::State *ndfstate = ndfautomata.getState(*i);
        if (ndfstate->getRule())
            dfstate->addRule(ndfstate->getRule());
    }
}



void
ndfautoToDfauto(const NdfAutomata &ndfautomata, DfAutomata &dfautomata)
{
    std::list<StateSetRecord::Discovery> toexplore, discovered;
    // our initial state is the lambda closure
    // of the initial state in the NDF automata
    IntSet initial;
    initial.insert(0);
    ndfautomata.lambdaClosure(initial);
    StateSetRecord record(ndfautomata, dfautomata);
    // register the initial state
    record.ensureState(initial, toexplore);
    while (toexplore.size()) {
        // new states that we may find when calculating transitions
        // make sure it is empty
        discovered.clear();
        for (std::list<StateSetRecord::Discovery>::iterator i = toexplore.begin(); i != toexplore.end(); ++i) {
            // get the available symbols to move from this state
            // set (originalset) in the original automata. Plus
            // a wildcard movement that is guaranteed to match all
            // the wildcard transitions in the set (if any)
            SymbolSet symbols;
            Wildcard *wildcard = NULL;
            ndfautomata.symbolsFrom(i->second, symbols, wildcard);
            for (SymbolSet::iterator j = symbols.begin(); j != symbols.end(); ++j) {
                IntSet newstates;
                // get all the states reachable with this symbol
                ndfautomata.transitionsFrom(i->second, *j, newstates);
                // build or recover the associated DF state
                DfAutomata::State *next_state = record.ensureState(newstates, discovered);
                // and store a transition
                i->first->addTransition(*j, next_state);
            }
            if (wildcard) {
                IntSet newstates;
                // we know they all match whatever ours match
                ndfautomata.wildcardTransitionsFrom(i->second, newstates);
                // build or recover the associated DF state
                DfAutomata::State *next_state = record.ensureState(newstates, discovered);
                // and store a transition
                i->first->addWildcardTransition(wildcard, next_state);
            }
        }
        // swap toexplore and discovered
        toexplore.swap(discovered);
    }
    // final optimizations
    dfautomata.removeEquivalentStates();
    dfautomata.removeUselessTransitions();
}



bool
DfOptimizedAutomata::Transition::trans_comp (const DfOptimizedAutomata::Transition &a, const DfOptimizedAutomata::Transition &b)
{
    return a.symbol.data() < b.symbol.data();
}



void
DfOptimizedAutomata::compileFrom(const DfAutomata &dfautomata)
{
    m_states.resize(dfautomata.m_states.size());
    size_t totaltrans = 0;
    size_t totalrules = 0;
    for (size_t s = 0; s < m_states.size(); ++s) {
        totaltrans += dfautomata.m_states[s]->m_symbol_trans.size();
        totalrules += dfautomata.m_states[s]->m_rules.size();
    }
    m_trans.resize(totaltrans);
    m_rules.resize(totalrules);
    size_t trans_offset = 0;
    size_t rules_offset = 0;
    for (size_t s = 0; s < m_states.size(); ++s) {
        m_states[s].begin_trans = trans_offset;
        m_states[s].begin_rules = rules_offset;
        for (SymbolToInt::const_iterator i = dfautomata.m_states[s]->m_symbol_trans.begin();
              i != dfautomata.m_states[s]->m_symbol_trans.end(); ++i, ++trans_offset) {
            m_trans[trans_offset].symbol = i->first;
            m_trans[trans_offset].state = i->second;
        }
        for (RuleSet::const_iterator i = dfautomata.m_states[s]->m_rules.begin();
              i != dfautomata.m_states[s]->m_rules.end(); ++i, ++rules_offset)
            m_rules[rules_offset] = *i;
        m_states[s].ntrans = dfautomata.m_states[s]->m_symbol_trans.size();
        m_states[s].nrules = dfautomata.m_states[s]->m_rules.size();
        std::sort(m_trans.begin() + m_states[s].begin_trans, m_trans.begin() + m_states[s].begin_trans + m_states[s].ntrans,
                     DfOptimizedAutomata::Transition::trans_comp);
        m_states[s].wildcard_trans = dfautomata.m_states[s]->m_wildcard_trans;
    }
}


}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
