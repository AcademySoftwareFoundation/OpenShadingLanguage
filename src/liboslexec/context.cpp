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


ShadingContext::ShadingContext (ShadingSystemImpl &shadingsys) 
    : m_shadingsys(shadingsys), m_attribs(NULL),
      m_globals(NULL)
{
    m_shadingsys.m_stat_contexts += 1;
}



ShadingContext::~ShadingContext ()
{
    m_shadingsys.m_stat_contexts -= 1;
}



void
ShadingContext::bind (int n, ShadingAttribState &sas, ShaderGlobals &sg)
{
    std::cerr << "bind " << (void *)this << " with " << n << " points\n";
    m_attribs = &sas;
    m_globals = &sg;
    m_npoints = n;
    m_nlights = 0;
    m_curlight = -1;
    m_curuse = ShadUseUnknown;

    // FIXME -- allocate enough space on the heap

    // Calculate number of layers we need for each use
    for (int i = 0;  i < ShadUseLast;  ++i) {
        m_nlayers[i] = m_attribs->m_shaders[i].nlayers ();
        std::cerr << "  " << m_nlayers[i] << " layers of " << shaderusename((ShaderUse)i) << "\n";
    }
}



void
ShadingContext::execute (ShaderUse use, Runflag *rf)
{
    // FIXME -- timers/stats

    std::cerr << "execute " << (void *)this 
              << " as " << shaderusename(use) << "\n";
    m_curuse = use;
    ASSERT (use == ShadUseSurface);  // FIXME

    // Get a handy ref to the network for this shader use
    ShaderNetwork &network (m_attribs->m_shaders[use]);
    int nlayers = network.nlayers ();

    // Get a handy ref to the array of ShadeExec layer for this shade use,
    // and make sure it's big enough for the number of layers we have.
    ExecutionLayers &execlayers (m_exec[use]);
    if (nlayers > execlayers.size())
        execlayers.resize (nlayers);

    for (int layer = 0;  layer < nlayers;  ++layer) {
        execlayers[layer].bind (this, use, layer, network[layer]);
        // FIXME -- for now, we're executing layers unconditionally.
        // Eventually, we only want to execut them here if they have
        // side effects (including generating final renderer outputs).
        // Layers without side effects should be executed lazily, only
        // as their outputs are needed by other layers.
        execlayers[layer].run (rf);
    }
}


}; // namespace pvt
}; // namespace OSL
