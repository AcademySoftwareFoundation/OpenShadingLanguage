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
      m_layername(layername)
{
}



void
ShaderInstance::parameters (const std::vector<ParamRef> &params)
{
    m_symbols = m_master->m_symbols;
    BOOST_FOREACH (const ParamRef &p, params) {
        std::cout << " PARAMETER " << p.name() << ' ' << p.type().c_str() << "\n";
        int i = m_master->findparam (p.name());
        if (i >= 0) {
            std::cerr << "    found " << i << "\n";
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


}; // namespace pvt
}; // namespace OSL
