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

#ifndef LEPARSE_H
#define LEPARSE_H

#include "lpexp.h"


OSL_NAMESPACE_ENTER



using lpexp::LPexp;


/// Light path expression parser
///
/// As most hand written descendant parsers, it is a bit messy in the implementation.
/// But the interface more or less resembles the grammar of the language:
///
///    symbol    := <char> | 'string'
///    listsym  := <lambda> | listsym sym
///    listexp  := <lambda> | listexp exp
///    group     := '<' exp exp exp listexp '>'
///    catexp    := ( listexp )
///    sorexp    := exp '|' exp
///    borexp    := [ listsym ]
///    wildexp  := . | [^ listsym ]
///    repeat    := exp *
///    repeat+  := exp +
///    mnrepeat := exp {<int>} | exp {<int>,} | exp {<int>,<int>}
///    exp        := wildexp | borexp | sorexp | catexp | group | repeat | repeat+ | mnrepeat
///    topexp    := exp | topexp exp
///
class Parser
{
    public:

        Parser();

        /// Parse a string and return the resulting light path expression tree or NULL if failed
        LPexp *parse(const char *text);

        /// Check for error in the last parsed string
        bool error()const { return m_error.size() > 0; };
        /// Get the error string
        const char *getErrorMsg()const { return m_error.c_str(); };
        /// Get the position of the string where the error appeared
        int getErrorPos()const { return m_pos; };

    private:

        /// Current char being parsed
        char head()const { return m_text[m_pos]; };
        /// Any input (including head) left?
        bool hasInput()const { return m_text.size() > m_pos; };
        /// Go to the next char in the string
        void next() { m_pos++; };

        /// build the complete pattern for a pathtracing stop
        ///
        /// That means the three basic label match expressions, plus those for the custom labels (if any),
        /// plus an additional wildcard to eat extra custom labels at the end, and finally the stop mark.
        /// So you provide the regexps for the basic three labels:
        ///
        ///      \param etype         Event type (CFVL)
        ///      \param dir            Direction (RD)
        ///      \param scatter      Scattering type (DGSs)
        ///
        /// That of course can be wildcards or any other expression willing to match them. And then you can
        /// provide extra expressions to match the custom labels:
        ///
        ///      \param custom        Custom label regexps
        ///
        /// And the function will automatically add a "[^stop_mark]*stop_mark" at the end.
        ///
        LPexp *buildStop(LPexp *etype, LPexp *scatter, const std::list<LPexp*> &custom);
        /// Gicen that a symbol is ready in head() to parse, parse it
        LPexp *parseSymbol();
        /// Gicen that a symbol is ready in head() to parse, parse it as a ustring
        /// and report it was a custom symbol in the iscustom flag
        ustring parseRawSymbol(bool &iscustom);
        /// Given that the begining of a concatenation of regexps is ready to parse, parse it
        /// and it can be optionally be enclosed in parentheis ()
        LPexp *parseCat();
        /// Given that a fully qualified group like <.RD'custom'> is ready to parse, parse it
        LPexp *parseGroup();
        /// Given that a ^abcde] (note missing [) is ready to parse, parse it
        LPexp *parseNegor();
        /// Given that [abcde] is ready to parse, parse it, but if it finds that it was actually
        /// [^abcd], fall back to parseNegor
        LPexp *parseOrlist();
        /// Given that a range like {5,7} or {2,} or {3} is ready to parse, parse it and return
        /// the range. Second number being -1 if if was {2,} and equals the first if it was {3}
        std::pair<int, int> parseRange();
        /// Take an already parsed lpexp and parse its possible modifier (*={}) if present and
        /// return the new lpexp
        LPexp *parseModifier(LPexp *e);
        /// Generic parse whatever comes next (just one item)
        LPexp *_parse();

        // error string
        std::string    m_error;
        // True if we are actually parsing a group <>, since otherwise everything gets
        // automatically converted to a group, this prevents that happening
        bool             m_ingroup;
        // maps each basic label to its expected possition in the appearance order, for instance
        // the direction label can't be in the fisrt pos of a group. This way we know where to put the
        // expression when the user writes just S, which translates to <..S>
        SymbolToInt m_label_position;
        // The set of the basic labels
        SymbolSet    m_basic_labels;
        // The black list for our wildcards, that have to exclude always the stop mark
        SymbolSet    m_minus_stop;

        // Current text being parsed
        std::string    m_text;
        // Current position in the text for head()
        size_t          m_pos;
};


OSL_NAMESPACE_EXIT

#endif // LEPARSE_H
