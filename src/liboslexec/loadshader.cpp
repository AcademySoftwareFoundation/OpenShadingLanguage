/*****************************************************************************
 *
 *             Copyright (c) 2009 Sony Pictures Imageworks, Inc.
 *                            All rights reserved.
 *
 *  This material contains the confidential and proprietary information
 *  of Sony Pictures Imageworks, Inc. and may not be disclosed, copied or
 *  duplicated in any form, electronic or hardcopy, in whole or in part,
 *  without the express prior written consent of Sony Pictures Imageworks,
 *  Inc. This copyright notice does not imply publication.
 *
 *****************************************************************************/


#include <vector>
#include <string>
#include <cstdio>

#include <boost/algorithm/string.hpp>

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"

#include "oslexec_pvt.h"
#include "osoreader.h"




namespace OSL {

namespace pvt {   // OSL::pvt


/// Custom subclass of OSOReader that provide callbacks that set all the
/// right fields in the ShaderMaster.
class OSOReaderToMaster : public OSOReader
{
public:
    OSOReaderToMaster ()
        : m_master (new ShaderMaster), m_reading_instruction(false)
      { }
    virtual ~OSOReaderToMaster () { }
    virtual void version (const char *specid, float version) { }
    virtual void shader (const char *shadertype, const char *name);
    virtual void symbol (SymType symtype, TypeSpec typespec, const char *name);
    virtual void symdefault (int def);
    virtual void symdefault (float def);
    virtual void symdefault (const char *def);
    virtual void hint (const char *hintstring);
    virtual void codemarker (const char *name);
    virtual void instruction (int label, const char *opcode);
    virtual void instruction_arg (const char *name);
    virtual void instruction_jump (int target);
    virtual void instruction_end ();

    ShaderMaster::ref master () const { return m_master; }

private:
    ShaderMaster::ref m_master;
    size_t m_firstarg;
    size_t m_nargs;
    bool m_reading_instruction;
    ustring m_sourcefile;
    int m_sourceline;
};



void
OSOReaderToMaster::shader (const char *shadertype, const char *name)
{
    m_master->m_shadername = ustring(name);
    m_master->m_shadertype = shadertype_from_name (shadertype);
}



void
OSOReaderToMaster::symbol (SymType symtype, TypeSpec typespec, const char *name)
{
    Symbol sym (ustring(name), typespec, symtype);
    if (typespec.simpletype().basetype == TypeDesc::FLOAT)
        sym.m_dataoffset = (int) m_master->m_fdefaults.size();
    else if (typespec.simpletype().basetype == TypeDesc::INT)
        sym.m_dataoffset = (int) m_master->m_idefaults.size();
    else if (typespec.simpletype().basetype == TypeDesc::STRING)
        sym.m_dataoffset = (int) m_master->m_sdefaults.size();
    else {
        ASSERT (0 && "unexpected type");
    }
    m_master->m_symbols.push_back (sym);
}



void
OSOReaderToMaster::symdefault (int def)
{
    ASSERT (m_master->m_symbols.size() && "symdefault but no sym");
    Symbol &sym (m_master->m_symbols.back());
    if (sym.typespec().simpletype().basetype == TypeDesc::FLOAT)
        m_master->m_fdefaults.push_back ((float)def);
    else if (sym.typespec().simpletype().basetype == TypeDesc::INT)
        m_master->m_idefaults.push_back (def);
    else {
        ASSERT (0 && "unexpected type");
    }
}



void
OSOReaderToMaster::symdefault (float def)
{
    ASSERT (m_master->m_symbols.size() && "symdefault but no sym");
    Symbol &sym (m_master->m_symbols.back());
    if (sym.typespec().simpletype().basetype == TypeDesc::FLOAT)
        m_master->m_fdefaults.push_back (def);
    else {
        ASSERT (0 && "unexpected type");
    }
}



void
OSOReaderToMaster::symdefault (const char *def)
{
    ASSERT (m_master->m_symbols.size() && "symdefault but no sym");
    Symbol &sym (m_master->m_symbols.back());
    if (sym.typespec().simpletype().basetype == TypeDesc::STRING)
        m_master->m_sdefaults.push_back (ustring(def));
    else {
        ASSERT (0 && "unexpected type");
    }
}



// If the string 'source' begins with 'pattern', erase the pattern from
// the start of source and return true.  Otherwise, do not alter source
// and return false.
inline bool
extract_prefix (std::string &source, const std::string &pattern)
{
    if (boost::starts_with (source, pattern)) {
        source.erase (0, pattern.length());
        return true;
    }
    return false;
}



// Return the prefix of source that doesn't contain any characters in
// 'stop', erase that prefix from source (up to and including the stop
// character.  Also, the returned string is trimmed of leading and trailing
// spaces if 'do_trim' is true.
static std::string
readuntil (std::string &source, const std::string &stop, bool do_trim=false)
{
    size_t e = source.find_first_of (stop);
    std::string r (source, 0, e);
    source.erase (0, e == source.npos ? e : e+1);
    if (do_trim)
        boost::trim (r);
    return r;
}



void
OSOReaderToMaster::hint (const char *hintstring)
{
    std::string h (hintstring);
    if (extract_prefix (h, "%filename{\"")) {
        m_sourcefile = readuntil (h, "\"");
        return;
    }
    if (extract_prefix (h, "%line{")) {
        m_sourceline = atoi (h.c_str());
        return;
    }
}



void
OSOReaderToMaster::codemarker (const char *name)
{
#if 0
    m_reading_param = false;
#endif
    m_sourcefile.clear();
}



void
OSOReaderToMaster::instruction (int label, const char *opcode)
{
    Opcode op (ustring(opcode), ustring(""));
    m_master->m_ops.push_back (op);
    m_firstarg = m_master->m_args.size();
    m_nargs = 0;
    m_reading_instruction = true;
}



void
OSOReaderToMaster::instruction_arg (const char *name)
{
    ustring argname (name);
    for (size_t i = 0;  i < m_master->m_symbols.size();  ++i) {
        if (m_master->m_symbols[i].name() == argname) {
            m_master->m_args.push_back (i);
            ++m_nargs;
            return;
        }
    }
    // ERROR! -- FIXME
//    m_master->m_args.push_back (0);  // FIXME
    std::cerr << "(unknown arg " << name << ") ";
}



void
OSOReaderToMaster::instruction_jump (int target)
{
    m_master->m_ops.back().add_jump (target);
}



void
OSOReaderToMaster::instruction_end ()
{
    m_master->m_ops.back().set_args (m_firstarg, m_nargs);
    m_master->m_ops.back().source (m_sourcefile, m_sourceline);
    m_reading_instruction = false;
}



ShaderMaster::ref
ShadingSystemImpl::loadshader (const char *name)
{
    OSOReaderToMaster oso;
    std::string filename = name;   // FIXME -- do search, etc.
    bool ok = oso.parse (filename);
    return ok ? oso.master() : NULL;
}



void
ShaderMaster::print ()
{
    std::cout << "Shader " << m_shadername << " type=" 
              << shadertypename(m_shadertype) << "\n";
    std::cout << "  symbols:\n";
    for (size_t i = 0;  i < m_symbols.size();  ++i) {
        const Symbol &s (m_symbols[i]);
        std::cout << "    " << s.typespec().string() << " " << s.name()
                  << "\n";
    }
    std::cout << "  int defaults:\n    ";
    for (size_t i = 0;  i < m_idefaults.size();  ++i)
        std::cout << m_idefaults[i] << ' ';
    std::cout << "\n";
    std::cout << "  float defaults:\n    ";
    for (size_t i = 0;  i < m_fdefaults.size();  ++i)
        std::cout << m_fdefaults[i] << ' ';
    std::cout << "\n";
    std::cout << "  string defaults:\n    ";
    for (size_t i = 0;  i < m_sdefaults.size();  ++i)
        std::cout << "\"" << m_sdefaults[i] << "\" ";
    std::cout << "\n";
    std::cout << "  code:\n";
    for (size_t i = 0;  i < m_ops.size();  ++i) {
        std::cout << "    " << i << ": " << m_ops[i].opname();
        for (size_t a = 0;  a < m_ops[i].nargs();  ++a)
            std::cout << " " << m_symbols[m_args[m_ops[i].firstarg()+a]].name();
        for (size_t j = 0;  j < Opcode::max_jumps;  ++j)
            if (m_ops[i].jump(j) >= 0)
                std::cout << " " << m_ops[i].jump(j);
        if (m_ops[i].sourcefile())
            std::cout << "\t(" << m_ops[i].sourcefile() << ":" 
                      << m_ops[i].sourceline() << ")";
        std::cout << "\n";
    }
}


}; // namespace pvt
}; // namespace OSL
