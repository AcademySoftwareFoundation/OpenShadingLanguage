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
#include "OpenImageIO/filesystem.h"

#include "oslexec_pvt.h"
#include "osoreader.h"




namespace OSL {

namespace pvt {   // OSL::pvt


void
ShaderMaster::print ()
{
    std::cout << "Shader " << m_shadername << " type=" 
              << shadertypename(m_shadertype) << "\n";
    std::cout << "  path = " << m_osofilename << "\n";
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
