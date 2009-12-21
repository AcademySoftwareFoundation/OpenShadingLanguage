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
        shadingsys().info ("bind %p with %d points", (void *)this, n);
    m_attribs = &sas;
    m_globals = &sg;
    m_npoints = n;
    m_nlights = 0;
    m_curlight = -1;
    m_curuse = ShadUseUnknown;
    m_heap_allotted = 0;
    m_closures_allotted = 0;

    // Allocate enough space on the heap
    size_t heap_size_needed = m_npoints * sas.heapsize () + sas.heapround ();
    // FIXME: the next statement is totally bogus, yet harmless.
    heap_size_needed += m_npoints * m_shadingsys.m_global_heap_total;
    if (shadingsys().debug())
        shadingsys().info ("  need heap %llu vs %llu",
                           (unsigned long long)heap_size_needed,
                           (unsigned long long)m_heap.size());
    if (heap_size_needed > m_heap.size()) {
        if (shadingsys().debug())
            shadingsys().info ("  ShadingContext %p growing heap to %llu",
                               this, (unsigned long long) heap_size_needed);
        m_heap.resize (heap_size_needed);
    }
    // Zero out the heap memory we will be using
    memset (&m_heap[0], 0, heap_size_needed);

    // Set up closure storage
    size_t closures_needed = m_npoints * sas.numclosures ();
    if (shadingsys().debug())
        shadingsys().info ("  need closures %d vs %llu", closures_needed,
                           (unsigned long long) m_closures.size());
    if (closures_needed > m_closures.size()) {
        if (shadingsys().debug())
            shadingsys().info ("  ShadingContext %p growing closures to %llu",
                               this, (unsigned long long)closures_needed);
        m_closures.resize (closures_needed);
    }
    // Zero out the closures
    for (size_t i = 0;  i < closures_needed;  ++i)
        m_closures[i].clear ();

    // Calculate number of layers we need for each use
    for (int i = 0;  i < ShadUseLast;  ++i) {
        m_nlayers[i] = m_attribs->shadergroup ((ShaderUse)i).nlayers ();
        if (shadingsys().debug())
            shadingsys().info ("  %d layers of %s", m_nlayers[i],
                               shaderusename((ShaderUse)i));
    }
}



void
ShadingContext::execute (ShaderUse use, Runflag *rf)
{
    // FIXME -- timers/stats

    if (shadingsys().debug())
        shadingsys().info ("execute %p as %s", this, shaderusename(use));
    m_curuse = use;
    ASSERT (use == ShadUseSurface);  // FIXME

    // Get a handy ref to the shader group for this shader use
    ShaderGroup &sgroup (m_attribs->shadergroup (use));
    size_t nlayers = sgroup.nlayers ();

    // Get a handy ref to the array of ShadeExec layer for this shade use,
    // and make sure it's big enough for the number of layers we have.
    ExecutionLayers &execlayers (m_exec[use]);
    if (nlayers > execlayers.size())
        execlayers.resize (nlayers);

    for (size_t layer = 0;  layer < nlayers;  ++layer)
        execlayers[layer].unbind ();

    m_lazy_evals = 0;
    int uncond_evals = 0;
    for (size_t layer = 0;  layer < nlayers;  ++layer) {
        ShadingExecution &exec (execlayers[layer]);
        ShaderInstance *inst = sgroup[layer];
        exec.bind (this, use, layer, inst);
        // Only execute layers that write globals (or, in the future,
        // have other side effects?) or the last layer of the sequence.
        if (inst->writes_globals() || layer == nlayers-1 ||
                ! m_shadingsys.m_lazylayers) {
            // exec.bind (this, use, layer, inst);
            exec.run (rf);
            ++uncond_evals;
        }
        // FIXME -- is it possible to also only bind when needed?  Or is
        // there some reason why that won't work?
    }

    shadingsys().m_layers_executed_uncond += uncond_evals;
    shadingsys().m_layers_executed_lazy += m_lazy_evals;
    shadingsys().m_layers_executed_never += nlayers - uncond_evals - m_lazy_evals;
}



Symbol *
ShadingContext::symbol (ShaderUse use, ustring name)
{
    size_t nlayers = m_nlayers[use];
   
    ASSERT(nlayers <= m_exec[use].size()); 

    for (int layer = (int)nlayers-1;  layer >= 0;  --layer) {
        Symbol *sym = m_exec[use][layer].symbol (name);
        if (sym)
            return sym;
    }
    return NULL;
}



const boost::regex &
ShadingContext::find_regex (ustring r)
{
    std::map<ustring,boost::regex>::const_iterator found;
    found = m_regex_map.find (r);
    if (found != m_regex_map.end())
        return found->second;
    // otherwise, it wasn't found, add it
    m_regex_map[r].assign (r.c_str());
    m_shadingsys.m_stat_regexes += 1;
    // std::cerr << "Made new regex for " << r << "\n";
    return m_regex_map[r];
}


}; // namespace pvt
}; // namespace OSL
