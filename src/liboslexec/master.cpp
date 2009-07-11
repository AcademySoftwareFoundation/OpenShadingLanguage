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
#include <limits>

#include <boost/foreach.hpp>

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"

#include "oslexec_pvt.h"
#include "oslops.h"



namespace OSL {

namespace pvt {   // OSL::pvt


int
ShaderMaster::findsymbol (ustring name) const
{
    for (size_t i = 0;  i < m_symbols.size();  ++i)
        if (m_symbols[i].name() == name)
            return (int)i;
    return -1;
}



int
ShaderMaster::findparam (ustring name) const
{
    for (int i = m_firstparam;  i <= m_lastparam;  ++i)
        if (m_symbols[i].name() == name)
            return i;
    return -1;
}



void
ShaderMaster::resolve_defaults ()
{
    m_firstparam = std::numeric_limits<int>::max();
    m_lastparam = -1;
    int i = 0;
    BOOST_FOREACH (Symbol &s, m_symbols) {
        if (s.symtype() == SymTypeParam || s.symtype() == SymTypeOutputParam) {
            if (m_firstparam > i)
                m_firstparam = i;
            m_lastparam = i;
            if (s.dataoffset() >= 0) {
                if (s.typespec().simpletype().basetype == TypeDesc::INT)
                    s.data (&(m_idefaults[s.dataoffset()]));
                else if (s.typespec().simpletype().basetype == TypeDesc::FLOAT)
                    s.data (&(m_fdefaults[s.dataoffset()]));
                else if (s.typespec().simpletype().basetype == TypeDesc::STRING)
                    s.data (&(m_sdefaults[s.dataoffset()]));
            }
        }
        if (s.symtype() == SymTypeConst) {
            if (s.dataoffset() >= 0) {
                if (s.typespec().simpletype().basetype == TypeDesc::INT)
                    s.data (&(m_iconsts[s.dataoffset()]));
                else if (s.typespec().simpletype().basetype == TypeDesc::FLOAT)
                    s.data (&(m_fconsts[s.dataoffset()]));
                else if (s.typespec().simpletype().basetype == TypeDesc::STRING)
                    s.data (&(m_sconsts[s.dataoffset()]));
            }
        }
        ++i;
    }
}



void
ShaderMaster::print ()
{
    std::cout << "Shader " << m_shadername << " type=" 
              << shadertypename(m_shadertype) << "\n";
    std::cout << "  path = " << m_osofilename << "\n";
    std::cout << "  symbols:\n";
    for (size_t i = 0;  i < m_symbols.size();  ++i) {
        const Symbol &s (m_symbols[i]);
        std::cout << "    " << i << ": " << s.typespec().string() 
                  << " " << s.name() << "\n";
    }
    std::cout << "  int consts:\n    ";
    for (size_t i = 0;  i < m_iconsts.size();  ++i)
        std::cout << m_iconsts[i] << ' ';
    std::cout << "\n";
    std::cout << "  float consts:\n    ";
    for (size_t i = 0;  i < m_fconsts.size();  ++i)
        std::cout << m_fconsts[i] << ' ';
    std::cout << "\n";
    std::cout << "  string consts:\n    ";
    for (size_t i = 0;  i < m_sconsts.size();  ++i)
        std::cout << "\"" << m_sconsts[i] << "\" ";
    std::cout << "\n";
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



void
ShaderMaster::resolve_ops ()
{
    BOOST_FOREACH (Opcode &op, m_ops) {
        // FIXME -- replace this hard-coded crap with a hash table or
        // something.
        if (shadingsys().debug())
            std::cout << "resolving " << op.opname() << "\n";
        if (op.opname() == "add")
            op.implementation (OP_add);
        else if (op.opname() == "assign")
            op.implementation (OP_assign);
        else if (op.opname() == "end")
            op.implementation (OP_end);
        else if (op.opname() == "sub")
            op.implementation (OP_sub);
        else
            op.implementation (OP_missing);
    }
}


}; // namespace pvt
}; // namespace OSL
