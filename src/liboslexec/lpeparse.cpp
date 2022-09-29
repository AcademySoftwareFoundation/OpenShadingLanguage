// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "lpeparse.h"
#include <OSL/oslclosure.h>


OSL_NAMESPACE_ENTER



static ustring udot(".");

Parser::Parser(const std::vector<ustring>* user_events,
               const std::vector<ustring>* user_scatterings)
{
    m_ingroup = false;
    m_error   = "";

    m_label_position[Labels::CAMERA]     = 0;
    m_label_position[Labels::LIGHT]      = 0;
    m_label_position[Labels::BACKGROUND] = 0;
    m_label_position[Labels::VOLUME]     = 0;
    m_label_position[Labels::TRANSMIT]   = 0;
    m_label_position[Labels::REFLECT]    = 0;
    m_label_position[Labels::OBJECT]     = 0;
    m_label_position[Labels::DIFFUSE]    = 1;
    m_label_position[Labels::GLOSSY]     = 1;
    m_label_position[Labels::SINGULAR]   = 1;
    m_label_position[Labels::STRAIGHT]   = 1;

    m_basic_labels.insert(Labels::CAMERA);
    m_basic_labels.insert(Labels::LIGHT);
    m_basic_labels.insert(Labels::BACKGROUND);
    m_basic_labels.insert(Labels::VOLUME);
    m_basic_labels.insert(Labels::TRANSMIT);
    m_basic_labels.insert(Labels::REFLECT);
    m_basic_labels.insert(Labels::OBJECT);
    m_basic_labels.insert(Labels::DIFFUSE);
    m_basic_labels.insert(Labels::GLOSSY);
    m_basic_labels.insert(Labels::SINGULAR);
    m_basic_labels.insert(Labels::STRAIGHT);
    m_basic_labels.insert(Labels::NONE);
    m_basic_labels.insert(Labels::STOP);

    m_minus_stop.insert(Labels::STOP);

    if (user_events)
        for (auto&& e : *user_events) {
            m_label_position[e] = 0;
            m_basic_labels.insert(e);
        }
    if (user_scatterings)
        for (auto&& s : *user_scatterings) {
            m_label_position[s] = 1;
            m_basic_labels.insert(s);
        }
}


LPexp*
Parser::buildStop(LPexp* etype, LPexp* scatter, const std::list<LPexp*>& custom)
{
    lpexp::Cat* cat = new lpexp::Cat();
    cat->append(etype);
    cat->append(scatter);
    for (const auto& p : custom)
        cat->append(p);

    if (custom.size() < 5)
        cat->append(new lpexp::Repeat(new lpexp::Wildexp(m_basic_labels)));
    cat->append(new lpexp::Symbol(Labels::STOP));
    return cat;
}



LPexp*
Parser::parseSymbol()
{
    bool iscustom = false;
    ustring sym   = parseRawSymbol(iscustom);
    if (m_ingroup) {
        if (sym == udot)
            return new lpexp::Wildexp(m_minus_stop);
        else
            return new lpexp::Symbol(sym);
    } else {
        if (iscustom) {
            std::list<LPexp*> custom;
            custom.push_back(new lpexp::Symbol(sym));
            return buildStop(new lpexp::Wildexp(m_minus_stop),
                             new lpexp::Wildexp(m_minus_stop), custom);
        } else {
            LPexp* basics[2] = { NULL, NULL };
            if (sym != ".") {
                SymbolToInt::const_iterator i = m_label_position.find(sym);
                if (i == m_label_position.end()) {
                    m_error = fmtformat("Unrecognized basic label: {}", sym);
                    return nullptr;
                }
                int pos     = i->second;
                basics[pos] = new lpexp::Symbol(sym);
            }
            for (int k = 0; k < 2; ++k)
                if (!basics[k])
                    basics[k] = new lpexp::Wildexp(m_minus_stop);
            std::list<LPexp*> empty;
            return buildStop(basics[0], basics[1], empty);
        }
    }
}



ustring
Parser::parseRawSymbol(bool& iscustom)
{
    std::string sym;
    if (head() == '\'') {
        next();
        while (hasInput() && head() != '\'') {
            sym += head();
            next();
        }
        if (!hasInput()) {
            m_error = "Reached end of line looking for ' to end a literal";
            return Labels::NONE;
        }
        next();
        iscustom = true;
    } else {
        sym += head();
        next();
        iscustom = false;
    }
    // hacky alias for NONE label
    if (!iscustom && sym == "x")
        return Labels::NONE;
    return ustring(sym);
}



LPexp*
Parser::parseCat()
{
    //lpexp::Cat *cat = new lpexp::Cat();
    std::vector<LPexp*> explist;
    char endchar;
    if (head() == '(') {
        next();
        endchar = ')';
    } else
        endchar = 0;

    while (hasInput() && head() != endchar) {
        if (head() == '|') {
            if (!explist.size()) {
                m_error = "No left expression to or with |";
                for (auto& exp : explist)
                    delete exp;
                return nullptr;
            }
            next();
            LPexp* e = _parse();
            if (error()) {
                delete e;
                for (auto& exp : explist)
                    delete exp;
                return nullptr;
            }
            if (explist.back()->getType() == lpexp::OR)
                ((lpexp::Orlist*)explist.back())->append(e);
            else {
                lpexp::Orlist* orexp = new lpexp::Orlist();
                orexp->append(explist.back());
                orexp->append(e);
                explist[explist.size() - 1] = orexp;
            }
        } else {
            LPexp* e = _parse();
            if (error()) {
                delete e;
                for (auto& exp : explist)
                    delete exp;
                return nullptr;
            }
            explist.push_back(e);
        }
    }
    if (hasInput() && head() == endchar)
        next();
    else if (endchar != 0) {
        m_error = "Reached end of line looking for )";
        for (auto& exp : explist)
            delete exp;
        return nullptr;
    }
    lpexp::Cat* cat = new lpexp::Cat();
    for (auto& exp : explist)
        cat->append(exp);
    return cat;
}



LPexp*
Parser::parseGroup()
{
    OSL_DASSERT(head() == '<');
    if (m_ingroup) {
        m_error = "No groups allowed inside of groups";
        return nullptr;
    }
    int basicpos     = 0;
    LPexp* basics[2] = { nullptr, nullptr };
    std::list<LPexp*> custom;
#define THROWAWAY()            \
    do {                       \
        delete basics[0];      \
        delete basics[1];      \
        for (auto& p : custom) \
            delete p;          \
        m_ingroup = false;     \
        return nullptr;        \
    } while (0)

    m_ingroup = true;
    next();
    while (hasInput() && head() != '>') {
        LPexp* e = _parse();
        if (error()) {
            delete e;
            THROWAWAY();
        }
        if (basicpos < 2)
            basics[basicpos++] = e;
        else
            custom.push_back(e);
    }

    if (!hasInput()) {
        m_error = "Reached end of line looking for > to end a group";
        THROWAWAY();
    }
    next();
    m_ingroup = false;
    for (; basicpos < 2; ++basicpos)
        basics[basicpos] = new lpexp::Wildexp(m_minus_stop);
    return buildStop(basics[0], basics[1], custom);
}



LPexp*
Parser::parseNegor()
{
    OSL_DASSERT(head() == '^');
    SymbolSet symlist;
    symlist.insert(Labels::STOP);  // never allowed
    int pos = -1;
    next();
    while (hasInput() && head() != ']') {
        bool iscustom = false;
        ustring sym   = parseRawSymbol(iscustom);
        if (error())
            return nullptr;
        symlist.insert(sym);
        if (iscustom) {
            if (symlist.size() > 2 && pos != -1)
                std::cerr
                    << "[pathexp] you are mixing labels of different type in [...]"
                    << std::endl;
            pos = -1;
        } else {
            SymbolToInt::const_iterator found = m_label_position.find(sym);
            if (found == m_label_position.end()) {
                m_error = "Unrecognized basic label";
                return nullptr;
            }
            if (symlist.size() > 2 && found->second != pos)
                std::cerr
                    << "[pathexp] you are mixing labels of different type in [...]"
                    << std::endl;
            pos = found->second;
        }
    }
    if (!hasInput()) {
        m_error
            = "Reached end of line looking for ] to end an negative or list'";
        return nullptr;
    }
    if (symlist.size() < 2) {
        m_error = "Empty or list [^] not allowed";
        return nullptr;
    }
    next();
    lpexp::Wildexp* wildcard = new lpexp::Wildexp(symlist);
    if (m_ingroup)
        return wildcard;
    else {
        std::list<LPexp*> custom;
        if (pos < 0) {  // is a custom label
            custom.push_back(wildcard);
            return buildStop(new lpexp::Wildexp(m_minus_stop),
                             new lpexp::Wildexp(m_minus_stop), custom);
        } else {
            LPexp* w2 = new lpexp::Wildexp(m_minus_stop);
            return buildStop(pos == 0 ? wildcard : w2, pos == 0 ? w2 : wildcard,
                             custom);
        }
    }
}



LPexp*
Parser::parseOrlist()
{
    OSL_DASSERT(head() == '[');
    next();
    if (hasInput() && head() == '^')
        return parseNegor();
    else {
        lpexp::Orlist* orlist = new lpexp::Orlist();
        while (hasInput() && head() != ']') {
            LPexp* e = _parse();
            if (error()) {
                delete e;
                delete orlist;
                return nullptr;
            }
            orlist->append(e);
        }
        if (!hasInput()) {
            m_error = "Reached end of line looking for ] to end an or list";
            delete orlist;
            return nullptr;
        }
        next();
        return orlist;
    }
}



std::pair<int, int>
Parser::parseRange()
{
    OSL_DASSERT(head() == '{');
    next();
    std::string firstnum = "";
    while (hasInput() && '0' <= head() && head() <= '9') {
        firstnum += head();
        next();
    }
    std::string secondnum = "";
    if (hasInput() && head() == ',') {
        next();
        while (hasInput() && '0' <= head() && head() <= '9') {
            secondnum += head();
            next();
        }
        if (!secondnum.size())
            secondnum = "-1";
    }
    if (!hasInput() || head() != '}' || !firstnum.size()) {
        m_error = "Bad {} range definition";
        return std::pair<int, int>(-1, -1);
    }
    next();
    if (secondnum.size())
        return std::pair<int, int>(OIIO::Strutil::stoi(firstnum),
                                   OIIO::Strutil::stoi(secondnum));
    else
        return std::pair<int, int>(OIIO::Strutil::stoi(firstnum),
                                   OIIO::Strutil::stoi(firstnum));
}



LPexp*
Parser::parseModifier(LPexp* e)
{
    if (hasInput()) {
        if (m_ingroup && (head() == '*' || head() == '{' || head() == '+')) {
            m_error = std::string("Repetitions not allowed inside '<...>'");
            return nullptr;
        }
        if (head() == '*') {
            next();
            return new lpexp::Repeat(e);
        } else if (head() == '{') {
            std::pair<int, int> range = parseRange();
            if (error())
                return nullptr;
            if (range.second < 0) {
                lpexp::Cat* cat = new lpexp::Cat();
                cat->append(new lpexp::NRepeat(e, range.first, range.first));
                cat->append(new lpexp::Repeat(e->clone()));
                return cat;
            } else
                return new lpexp::NRepeat(e, range.first, range.second);
        } else if (head() == '+') {
            next();
            lpexp::Cat* cat = new lpexp::Cat();
            cat->append(e);
            cat->append(new lpexp::Repeat(e->clone()));
            return cat;
        } else
            return e;
    } else
        return e;
}



LPexp*
Parser::_parse()
{
    LPexp* e;
    if (head() == '(')
        e = parseCat();
    else if (head() == '[')
        e = parseOrlist();
    else if (head() == '<')
        e = parseGroup();
    else
        e = parseSymbol();
    if (error())
        return nullptr;
    return parseModifier(e);
}



LPexp*
Parser::parse(const char* text)
{
    m_error   = "";
    m_text    = text;
    m_pos     = 0;
    m_ingroup = false;
    if (hasInput())
        return parseCat();
    else
        return nullptr;
}


OSL_NAMESPACE_EXIT
