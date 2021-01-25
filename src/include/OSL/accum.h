// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/optautomata.h>
#include <OSL/oslconfig.h>
#include <list>
#include <stack>

OSL_NAMESPACE_ENTER

class Aov {
public:
    virtual void write(void* flush_data, Color3& color, float alpha,
                       bool has_color, bool has_alpha)
        = 0;
};

/// AOV slot where the integrator will write to
///
/// This is the end point of the output before going to
/// the AOV. These objects are owned by the Accumulator
/// class and there is going to be exactly one for each
/// active AOV. These objects should be reset before the
/// integration ray walk for a pixel and flushed at the end
struct AovOutput {
    AovOutput() : neg_color(false), neg_alpha(false), aov(NULL) {};

    // Accumulated values
    Color3 color;
    float alpha;
    // whether there has been some value added to color
    bool has_color;
    // whether there has been some value added to alpha
    bool has_alpha;
    // It is also possible to "invert" values before flushing
    bool neg_color;
    bool neg_alpha;
    // The abstract AOV to send the data to
    Aov* aov;

    // Reset the accumulated value to start a new integration
    void reset()
    {
        color.setValue(0, 0, 0);
        alpha     = 0.0f;
        has_alpha = has_color = false;
    };
    /// Sends the color information to the AOV
    void flush(void* flush_data);
};



/// Rule mapping a pattern to an AOV
///
/// This is the entity being linked from the automata. At any state, if
/// it is final, you will find pointers to these objects. They already know
/// what output to use and how. Multiple rules can point to the same AOV.
class OSLEXECPUBLIC AccumRule {
public:
    /// Create a rule for accumulating results to an AOV
    ///
    ///  \param out_idx      output index that this rule should be writing to
    ///  \param only_source When accumulating don't apply the viewer filter, just the
    ///                            source value
    ///  \param toalpha      Convert value and pack to the alpha channel of the AOV
    ///  \param neg            Invert (1 - v) the value before sending to the AOV
    ///
    AccumRule(int outidx, bool toalpha)
        : m_outidx(outidx), m_save_to_alpha(toalpha) {};

    /// Called to accumulate from AccumAutomata. It will select the right ouput from
    /// the given vector based in the AOV index number (they are guaranteed to match)
    void accum(const Color3& color, std::vector<AovOutput>& outputs) const;

    // This link information is actually not used inside of this class for other thing
    // than to keep track of who links who and in what way. Everything is used at the end
    // from AovOutput
    bool toAlpha() const { return m_save_to_alpha; };
    int getOutputIndex() const { return m_outidx; };

private:
    int m_outidx;
    bool m_save_to_alpha;
};


namespace lpexp {
class Rule;
}

/// The accumulation automata
///
/// It consists of a basic DF automata from optautomata.h and a list
/// of rules which are linked from the DFA final states. It is constant
/// along the render process, doesn't keep any state.
class OSLEXECPUBLIC AccumAutomata {
public:
    ~AccumAutomata();

    /// Support the given symbol as event tag on lpe expressions
    void addEventType(ustring symbol) { m_user_events.push_back(symbol); };
    /// Support the given symbol as scattering tag on lpe expressions
    void addScatteringType(ustring symbol)
    {
        m_user_scatterings.push_back(symbol);
    };

    /// Add a single rule for rendering outputs
    ///
    ///    \param pattern         The light path expression to be mapped to the new rule
    ///    \param outidx          Index of the target output
    ///    \param toalpha         Whether to map this rule to the alpha value of the AOV
    ///    \param neg              Whether to invert the involved AOV at the end of the render
    ///
    AccumRule* addRule(const char* pattern, int outidx, bool toalpha = false);

    /// Once all the desired rules have been added, compile the automata
    void compile();

    /// Performs an accumulation in the given outputs vector if any rule is activated in the given state
    void accum(int state, const Color3& color,
               std::vector<AovOutput>& outputs) const;

    /// Get an specific transition
    int getTransition(int state, ustring symbol) const
    {
        return m_dfoptautomata.getTransition(state, symbol);
    };

    /// The rule list is for public use in read-only, so Accumulator knows what AOVS are we using
    const std::list<AccumRule>& getRuleList() const { return m_accumrules; };

    /// Get the rules for a given state
    void* const* getRulesInState(int state, int& count) const
    {
        return m_dfoptautomata.getRules(state, count);
    };

private:
    // Compiled lpexp's we save while creating the rules with addRule.
    // It gets nuked after you call compile()
    std::list<lpexp::Rule*> m_rules;
    // The famous so called DF automata
    DfOptimizedAutomata m_dfoptautomata;
    // List of rules linked as void * from the automata's states
    std::list<AccumRule> m_accumrules;
    // Custom symbols to support on expressions as events
    std::vector<ustring> m_user_events;
    // Custom symbols to support on expressions as scattering
    std::vector<ustring> m_user_scatterings;
};



/// State sensitive render accumulator
///
/// This is the actual object used for accumulation from the
/// integrator functions during the light walk. Knows what state
/// we are at and keeps record of the accumulated values (AovOutput)
///
class OSLEXECPUBLIC Accumulator {
public:
    Accumulator(const AccumAutomata* accauto);

    void setAov(int outidx, Aov* aov, bool neg_color, bool neg_alpha);

    /// If the machine is broken no result will be stored, you can cut the branch
    bool broken() const { return m_state < 0; }

    void pushState();
    void popState();

    /// Push a single label
    void move(ustring symbol);

    /// Push a NONE terminated array of labels
    void move(const ustring* symbols);

    /// very commonly we push all labels, this helps reducing code. custom can be NULL
    /// and although the last label is always stop, we leave the argument to make the
    /// code show that there is a STOP label there.
    void move(ustring event, ustring scatt, const ustring* custom,
              ustring stop);

    /// Check if a given movement is possible without breaking the automata.
    /// Leaves the state untouched
    bool test(ustring dir, ustring sca, const ustring* custom, ustring stop)
    {
        pushState();
        move(dir, sca, custom, stop);
        bool active = !broken();
        popState();
        return active;
    }



    /// Clears all the outputs to start integrating
    void begin();

    /// finishes and flushes the outputs to the sample store
    void end(void* flush_data);


    /// Send a result to whatever rules might be active in the current state
    void accum(const Color3& color)
    {
        if (m_state >= 0)
            m_accum_automata->accum(m_state, color, m_outputs);
    };

    const AovOutput& getOutput(int idx) const { return m_outputs[idx]; };

private:
    // A reference to the stateless automata that can be shared between multiple
    // threads
    const AccumAutomata* m_accum_automata;
    // The output array has as many entries as AOV's enabled, and they share
    // the same index so m_outputs[aov->getIndex()].aov == aov for AOV's linked
    // by rules and NULL for the rest
    std::vector<AovOutput> m_outputs;
    // Current state stack, this is state information
    std::stack<int> m_stack;
    // And the current state
    int m_state;
};


OSL_NAMESPACE_EXIT
