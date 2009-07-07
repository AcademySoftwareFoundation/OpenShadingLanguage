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

#include <boost/foreach.hpp>

#include "OpenImageIO/dassert.h"

#include "oslexec_pvt.h"



namespace OSL {

namespace pvt {   // OSL::pvt


ShaderInstance::ShaderInstance (ShaderMaster::ref master,
                                const char *layername) 
    : m_master(master), m_symbols(m_master->m_symbols),
      m_layername(layername), m_heapsize(0)
{
    calc_heapsize ();
}



void
ShaderInstance::parameters (const std::vector<ParamRef> &params)
{
    m_iparams = m_master->m_idefaults;
    m_fparams = m_master->m_fdefaults;
    m_sparams = m_master->m_sdefaults;
    m_symbols = m_master->m_symbols;
    BOOST_FOREACH (const ParamRef &p, params) {
        if (shadingsys().debug())
            std::cout << " PARAMETER " << p.name() << ' ' << p.type().c_str() << "\n";
        int i = m_master->findparam (p.name());
        if (i >= 0) {
            if (shadingsys().debug())
                std::cout << "    found " << i << "\n";
#if 0
            if (s.typespec().simpletype().basetype == TypeDesc::INT) {
                s.data (&(m_iparams[s.dataoffset()]));
            } else if (s.typespec().simpletype().basetype == TypeDesc::FLOAT) {
                s.data (&(m_fparams[s.dataoffset()]));
            } else if (s.typespec().simpletype().basetype == TypeDesc::STRING) {
                s.data (&(m_sparams [s.dataoffset()]));
            }
//          std::cerr << "    sym " << s.name() << " offset " << s.dataoffset()
//                    << " address " << (void *)s.data() << "\n";
#endif
        }
    }
}



size_t
ShaderInstance::calc_heapsize ()
{
    if (shadingsys().debug())
        std::cout << "calc_heapsize on " << m_master->shadername() << "\n";
    m_heapsize = 0;
    BOOST_FOREACH (const Symbol &s, m_symbols) {
        // std::cout << "  sym " << s.mangled() << "\n";

        // Skip if the symbol is a type that doesn't need heap space
        if (s.symtype() == SymTypeConst || s.symtype() == SymTypeGlobal)
            continue;

        const TypeSpec &t (s.typespec());
        size_t size = 0;
        if (t.is_closure()) {
            // FIXME
        } else if (t.is_structure()) {
            // FIXME
        } else {
            size = t.simpletype().size();
        }
        // Round up to multipe of 4 bytes
        size = (size+3) & (~3);
        m_heapsize += size;
        // FIXME -- have a ShadingSystem method in a central place that
        // computes heap size for all types
    }
    if (shadingsys().debug())
        std::cout << " Heap needed " << m_heapsize << "\n";
    return m_heapsize;
}



}; // namespace pvt
}; // namespace OSL
