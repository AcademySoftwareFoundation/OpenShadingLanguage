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
    OSOReaderToMaster () : m_master (new ShaderMaster) { }
    virtual ~OSOReaderToMaster () { }
    virtual void version (const char *specid, float version) { }
    virtual void shader (const char *shadertype, const char *name);
    virtual void symbol (SymType symtype, TypeSpec typespec, const char *name);
    virtual void symdefault (int def);
    virtual void symdefault (float def);
    virtual void symdefault (const char *def);
    virtual void hint (const char *hintstring);
    virtual void codemarker (const char *name);
    virtual void instruction (int label, const char *opcode) { }
    virtual void instruction_arg (const char *name) { }
    virtual void instruction_jump (int target) { }

    ShaderMaster::ref master () const { return m_master; }

private:
    ShaderMaster::ref m_master;
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



void
OSOReaderToMaster::hint (const char *hintstring)
{
#if 0
    if (m_reading_param && ! strncmp (hintstring, "%meta{", 6)) {
        hintstring += 6;
        // std::cerr << "  Metadata '" << hintstring << "'\n";
        std::string type = readuntil (&hintstring, ',', '}');
        std::string name = readuntil (&hintstring, ',', '}');
        // std::cerr << "    " << name << " : " << type << "\n";
        OSLToMaster::Parameter p;
        p.name = name;
        p.type = string_to_type (type.c_str());
        if (p.type.basetype == TypeDesc::STRING) {
            while (*hintstring == ' ')
                ++hintstring;
            while (hintstring[0] == '\"') {
                ++hintstring;
                p.sdefault.push_back (readuntil (&hintstring, '\"'));
            }
        } else if (p.type.basetype == TypeDesc::INT) {
            while (*hintstring == ' ')
                ++hintstring;
            while (*hintstring && *hintstring != '}') {
                p.idefault.push_back (atoi (hintstring));
                readuntil (&hintstring, ',', '}');
            }
        } else if (p.type.basetype == TypeDesc::FLOAT) {
            while (*hintstring == ' ')
                ++hintstring;
            while (*hintstring && *hintstring != '}') {
                p.fdefault.push_back (atof (hintstring));
                readuntil (&hintstring, ',', '}');
            }
        }
        m_query.m_params[m_query.nparams()-1].metadata.push_back (p);
    }
    // std::cerr << "Hint '" << hintstring << "'\n";
#endif
}



void
OSOReaderToMaster::codemarker (const char *name)
{
#if 0
    m_reading_param = false;
#endif
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
}


}; // namespace pvt
}; // namespace OSL
