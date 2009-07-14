/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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

#include <vector>
#include <string>
#include <cstdio>

#include <boost/foreach.hpp>

#include <OpenImageIO/dassert.h>

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
    if (shadingsys().debug())
        std::cout << "bind " << (void *)this << " with " << n << " points\n";
    m_attribs = &sas;
    m_globals = &sg;
    m_npoints = n;
    m_nlights = 0;
    m_curlight = -1;
    m_curuse = ShadUseUnknown;
    m_heap_allotted = 0;

    // Allocate enough space on the heap
    size_t heap_size_needed = m_npoints * sas.heapsize ();
    heap_size_needed += m_npoints * m_shadingsys.m_global_heap_total;
    if (shadingsys().debug())
        std::cout << "  need heap " << heap_size_needed << " vs " << m_heap.size() << "\n";
    if (heap_size_needed > m_heap.size()) {
        if (shadingsys().debug())
            std::cout << "  ShadingContext " << (void *)this 
                      << " growing heap to " << heap_size_needed << "\n";
        m_heap.resize (heap_size_needed);
    }
    // Zero out everything in the heap
    memset (&m_heap[0], 0, m_heap.size());

    // Calculate number of layers we need for each use
    for (int i = 0;  i < ShadUseLast;  ++i) {
        m_nlayers[i] = m_attribs->shadergroup ((ShaderUse)i).nlayers ();
        if (shadingsys().debug())
            std::cout << "  " << m_nlayers[i] << " layers of " << shaderusename((ShaderUse)i) << "\n";
    }
}



void
ShadingContext::execute (ShaderUse use, Runflag *rf)
{
    // FIXME -- timers/stats

    if (shadingsys().debug())
        std::cout << "execute " << (void *)this 
                  << " as " << shaderusename(use) << "\n";
    m_curuse = use;
    ASSERT (use == ShadUseSurface);  // FIXME

    // Get a handy ref to the shader group for this shader use
    ShaderGroup &sgroup (m_attribs->shadergroup (use));
    int nlayers = sgroup.nlayers ();

    // Get a handy ref to the array of ShadeExec layer for this shade use,
    // and make sure it's big enough for the number of layers we have.
    ExecutionLayers &execlayers (m_exec[use]);
    if (nlayers > execlayers.size())
        execlayers.resize (nlayers);

    for (int layer = 0;  layer < nlayers;  ++layer) {
        execlayers[layer].bind (this, use, layer, sgroup[layer]);
        // FIXME -- for now, we're executing layers unconditionally.
        // Eventually, we only want to execut them here if they have
        // side effects (including generating final renderer outputs).
        // Layers without side effects should be executed lazily, only
        // as their outputs are needed by other layers.
        execlayers[layer].run (rf);
    }
}



Symbol *
ShadingContext::symbol (ShaderUse use, ustring name)
{
    for (int layer = (int)m_exec[use].size()-1;  layer >= 0;  --layer) {
        Symbol *sym = m_exec[use][layer].symbol (name);
        if (sym)
            return sym;
    }
    return NULL;
}


}; // namespace pvt
}; // namespace OSL
