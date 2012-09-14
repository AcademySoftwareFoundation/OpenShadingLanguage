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

#ifndef AUTOMATA_H
#define AUTOMATA_H

#include <set>
#include <map>
#include <list>
#include <vector>

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

#include "oslconfig.h"


OSL_NAMESPACE_ENTER


// General container for all symbol sets

// General container for integer sets
typedef std::set<int> IntSet; // probably faster to test for equality, unions and so

typedef boost::unordered_set<ustring, ustringHash> SymbolSet;
// This is for the transition table used in DfAutomata::State
typedef boost::unordered_map<ustring, int, ustringHash> SymbolToInt;
// And this is for the transition table in NdfAutomata which
// has several movements for each symbol
typedef boost::unordered_map<ustring, IntSet, ustringHash> SymbolToIntList;
typedef boost::unordered_map<int, int> HashIntInt;

// For the rules in the deterministic states, we don't need a real set
// cause when converting from the NDF automata we will never find the same
// rule twice
typedef std::vector<void *> RuleSet;

// The lambda symbol (empty word)
extern ustring lambda;


/// This struct represent a wildcard (for wildcard transitions)
/// but it is NOT the transition itself. Light path expressions also
/// use this class to create the transitions. A wildcard matches
/// any symbol which is not listed in its black list (m_minus).
struct Wildcard {
    Wildcard() {};
    Wildcard(SymbolSet &minus):m_minus(minus) {};

    bool matches(ustring symbol)const
    {
        // true if the given symbol is not in m_minus
        return (m_minus.find(symbol) == m_minus.end());
    };

    // Black list of unincluded symbols
    SymbolSet m_minus;
};

/// Non Deterministic Finite Automata
//
/// This class represents and automata with multiple transitions per
/// symbol and allowed lambda transitions too. Light path expressions
/// build this automata easily. It is later converted to a DF automata.
class NdfAutomata {
    public:

        // Basic state for NDF automata
        class State {
            friend class NdfAutomata;
            public:

                State(int id)
                {
                    m_id = id;
                    m_wildcard_trans = -1;
                    m_wildcard = NULL;
                    m_rule = NULL;
                };
                ~State() { delete m_wildcard; };

                /// Get the set of state id's reachable by the given symbol
                ///
                /// It doesn't clean the given result set, so you can use this
                /// function to accumulate states.
                void getTransitions (ustring symbol, IntSet &out_states)const;

                /// Get all the lambda transitions
                ///
                /// Returns the state collection as a pair of iterators [begin, end)
                std::pair <IntSet::const_iterator, IntSet::const_iterator>
                    getLambdaTransitions ()const;

                /// Add a standar transition (also valid for lambda)
                void addTransition (ustring symbol, State *state);
                /// Add a wildcard transition
                ///
                /// Note that the state is going to take ownership of the wildcard
                /// pointer. So it should be newly allocated on heap and you don't
                /// have to delete ir
                void addWildcardTransition(Wildcard *wildcard, State *state);

                /// For final states, this sets the associated rule. Which
                /// can be any kind of pointer
                void setRule(void *rule) { m_rule = rule; };

                void *getRule()const { return m_rule; };
                int  getId()const { return m_id; };

                /// For debuging purposes
                std::string tostr()const;

            protected:

                // State id (i.e index in the states vector)
                int m_id;
                // Transitions by single symbols
                SymbolToIntList m_symbol_trans;
                // In case m_wildcard is not NULL this holds the destination
                // of the wildcard transition
                int m_wildcard_trans;
                // Wildcard (NULL if no wildcard transition present)
                Wildcard *m_wildcard;
                // Associated rule for final states, NULL if not final
                void *m_rule;
        };



        NdfAutomata() { newState(); };
        ~NdfAutomata();

        const State *getInitial()const { return m_states[0]; };
        State *getInitial() { return m_states[0]; };

        /// Creates a new state qith a valid id
        State *newState();

        const State *getState(int i)const { return m_states[i]; };
        size_t size()const { return m_states.size(); };

        /// return all the symbols that lead to other states starting
        /// in any of the states in the given set. Also if there are
        /// wildcard transitions, give back a new wildcard that guarantees
        /// that its matches match all the wildcards in the set.
        /// So the symbols returned in out_symbols do NOT overlap those that
        /// match the returned wildcard (if present). And the union of out_symbols
        /// and those matched by the wildcard, are all the valid transitions from
        /// the given state set
        void symbolsFrom(const IntSet &states, SymbolSet &out_symbols, Wildcard *&wildcard)const;

        /// Get the set of states that are reachable from the given state set using the given symbol
        void transitionsFrom(const IntSet &states, ustring symbol, IntSet &out_states)const;
        /// Get the set of states that are reachable from the given state set by lambda
        void wildcardTransitionsFrom(const IntSet &states, IntSet &out_states)const;

        /// Perform a lambda closure of a state set
        ///
        /// In other words, complete the given set so it includes all the aditional
        /// states that are reachable by the lambda symbol
        void lambdaClosure(IntSet &states)const;

        /// for debuging purposes
        std::string tostr()const;

    protected:
        /// Vector of states
        std::vector<State *> m_states;
};



// We need to make a set of sets of integers (states). For doing that
// we need a unique key for a single set, which is going to be the list
// of state ids sorted by value. And this is the type that is going to hold
// that. Legal to use in a std::set
typedef std::vector<int> StateSetKey;

// Compute the unique key for the given set of states
void keyFromStateSet(const IntSet &states, StateSetKey &out_key);



/// Deterministic Finite Automata
///
/// This is a ready to use for parsing finite state automata where every
/// symbol takes you to a single next state at most. It allows for linear
/// time parsing of light path expressions
class DfAutomata {
    friend class DfOptimizedAutomata;
    public:
        /// Simple state for a deterministic automata
        class State {
            friend class DfAutomata;
            friend class DfOptimizedAutomata;
            public:

                State(int id)
                {
                    m_id = id;
                    m_wildcard_trans = -1;
                }

                /// Get the transition for a symbol
                //
                /// Returns -1 if no transitions with that symbol. That means the
                /// symbol is not recognized at this point of the automata
                int getTransition(ustring symbol)const;

                // Same semantics as in NdfAutomata::State
                void addTransition(ustring symbol, State *state);
                // WARNING: this has an optimized representation and has to be called
                // always AFTER all the normal transitions have been added
                void addWildcardTransition(Wildcard *wildcard, State *state);

                /// Optimize transitions
                ///
                /// Sometimes the existing wildcard can be extended to wrap some of
                /// the single symbol transitions. This function performs that optimization
                void removeUselessTransitions();

                void addRule(void *rule) { m_rules.push_back(rule); };
                const RuleSet &getRules()const { return m_rules; };
                int  getId()const { return m_id; };
                std::string tostr()const;

            protected:

                int m_id;
                // Only one transition per symbol here
                SymbolToInt m_symbol_trans;
                int m_wildcard_trans;
                // A final state here might contain several final states
                // from the original NDF automata. Therefore, we need a list
                // of rules here
                RuleSet m_rules;
        };


        DfAutomata() {};
        ~DfAutomata();

        State *newState();
        const State *getState(int i)const { return m_states[i]; };
        size_t size()const { return m_states.size(); };
        // In case somebody wants to reuse the same object and recreate
        // an automata, we provide this method
        void clear();

        /// Colapse all the equivalent states into single ones
        void removeEquivalentStates();
        /// Go through all the states and perform removeUselessTransitions
        /// method call on them
        void removeUselessTransitions();

        /// For debuging purposes
        std::string tostr()const;

    protected:

        /// Return true if two states are equivalent
        ///
        /// Two states are equivalent if their transition tables lead
        /// to the same states for the same symbols
        bool equivalent(const State *dfstateA, const State *dfstateB);

        // State vector with the automata
        std::vector<State *> m_states;
};



/// Helper class for the ndfautoToDfauto algorithm. It keeps a record
/// of state sets that have already been found to be new states of the
/// deterministic automata
class StateSetRecord {
    public:
        StateSetRecord(const NdfAutomata &ndfautomata, DfAutomata &dfautomata):
            m_ndfautomata(ndfautomata), m_dfautomata(dfautomata)
            {};

        // A new found state is defined by the deterministic state created
        // for it and the state(int) set in the original automata
        typedef std::pair<DfAutomata::State *, IntSet> Discovery;
        // The type that will index our new created states indexed by the set key
        typedef std::map<StateSetKey, DfAutomata::State *> StateSetMap;

        /// Take a state set and build a new df state (or return existing one)
        /// Also, if it was newly created, append it to the discovered list so we
        /// can iterate over it later
        DfAutomata::State *ensureState(const IntSet &newstates, std::list<Discovery> &discovered);

    private:

        /// Gather all the rules from the original automata in the given sets (if any)
        /// and put them in the dfstate rule set
        void getRulesFromSet(DfAutomata::State *dfstate, const NdfAutomata &ndfautomata, const IntSet &ndfstates);

        const NdfAutomata &m_ndfautomata;
        DfAutomata &m_dfautomata;
        StateSetMap m_key_to_dfstate;
};



/// NDF to DF automata convertion
///
/// This function is the most important pice of the whole process. It takes
/// a non-deterministic finite automata and computes an equivalent deterministic
/// one. It is equivalente in the sense that they  both recognize the same language
void ndfautoToDfauto(const NdfAutomata &ndfautomata, DfAutomata &dfautomata);

OSL_NAMESPACE_EXIT

#endif // AUTOMATA_H
