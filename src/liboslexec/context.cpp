/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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
#include <OpenImageIO/sysutil.h>

#include "oslexec_pvt.h"
#include "oslops.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

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
    m_curuse = ShadUseUnknown;
    m_heap_allotted = 0;
    m_closures_allotted = 0;

    // Optimize if we haven't already
    for (int i = 0;  i < ShadUseLast;  ++i) {
        ShaderGroup &group (m_attribs->shadergroup ((ShaderUse)i));
        if (! group.optimized())
            shadingsys().optimize_group (sas, group);
    }

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
    if (shadingsys().m_clearmemory)
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

    // Clear the message blackboard
    m_messages.clear ();
    m_closure_msgs.clear ();

    // Calculate number of layers we need for each use
    for (int i = 0;  i < ShadUseLast;  ++i) {
        m_nlayers[i] = m_attribs->shadergroup ((ShaderUse)i).nlayers ();
        if (shadingsys().debug())
            shadingsys().info ("  %d layers of %s", m_nlayers[i],
                               shaderusename((ShaderUse)i));
    }
}



void
ShadingContext::execute (ShaderUse use, Runflag *rf, int *ind, int nind)
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
    if (nlayers > execlayers.size()) {
        size_t oldlayers = execlayers.size();
        execlayers.resize (nlayers);
        // Initialize the new layers
        for (  ;  oldlayers < nlayers;  ++oldlayers)
            execlayers[oldlayers].init (this, use, oldlayers);
    }

    for (size_t layer = 0;  layer < nlayers;  ++layer)
        execlayers[layer].prebind ();

    // Make space for new runflags
#if USE_RUNFLAGS
    Runflag *runflags = rf;
    int *indices = NULL;
    int nindices = 0;
    if (rf) {
        // Passed runflags -- done!
    } else if (ind) {
        runflags = ALLOCA (Runflag, m_npoints);
        // Passed indices -- need to convert to runflags
        memset (runflags, RunflagOff, m_npoints*sizeof(Runflag));
        for (int i = 0;  i < nind;  ++i)
            runflags[indices[i]] = RunflagOn;
    } else {
        runflags = ALLOCA (Runflag, m_npoints);
        // If not passed runflags, make new ones
        for (int i = 0;  i < m_npoints;  ++i)
            runflags[i] = RunflagOn;
    }
#elif USE_RUNINDICES
    Runflag *runflags = rf;
    int *indices = ALLOCA (int, m_npoints);
    int nindices = nind;
    if (ind) {
        memcpy (indices, ind, nind*sizeof(indices[0]));
    } else if (rf) {
        // Passed runflags -- convert those to indices
        for (int i = 0;  i < m_npoints;  ++i)
            if (rf[i])
                indices[nindices++] = i;
    } else {
        // If not passed either, make new ones
        nindices = m_npoints;
        for (int i = 0;  i < nindices;  ++i)
            indices[i] = i;
    }
#elif USE_RUNSPANS
    Runflag *runflags = rf;
    int *indices = NULL;
    int nindices = nind;
    if (ind) {
        // NOTE: this assumes indices were passed in spans format
        indices = ALLOCA (int, nind);
        memcpy (indices, ind, nind*sizeof(indices[0]));
    } else if (rf) {
        // Passed runflags -- convert those to spans
        indices = ALLOCA (int, m_npoints*2);
        nindices = 0;
        runflags_to_spans (rf, 0, m_npoints, indices, nindices);
    } else {
        // If not passed either, make new ones
        indices = ALLOCA (int, 2);  // max space we could need
        nindices = 2;
        indices[0] = 0;
        indices[1] = m_npoints;
    }
#endif

    m_lazy_evals = 0;
    m_rebinds = 0;
    m_binds = 0;
    m_paramstobind = 0;
    m_paramsbound = 0;
    m_instructions_run = 0;
    int uncond_evals = 0;
    for (size_t layer = 0;  layer < nlayers;  ++layer) {
        ShadingExecution &exec (execlayers[layer]);
        ShaderInstance *inst = sgroup[layer];
        // Only execute layers that write globals (or, in the future,
        // have other side effects?) or the last layer of the sequence.
        if (! inst->run_lazily()) {
#if 0
            std::cerr << "Running layer " << layer << ' ' << inst->layername()
                      << ' ' << inst->master()->shadername() << "\n";
#endif
            exec.run (runflags, indices, nindices);
            ++uncond_evals;
        }
//        else std::cerr << "skip layer " << layer << "\n";
        // FIXME -- is it possible to also only bind when needed?  Or is
        // there some reason why that won't work?
    }

    shadingsys().m_layers_executed_uncond += uncond_evals;
    shadingsys().m_layers_executed_lazy += m_lazy_evals;
    shadingsys().m_layers_executed_never += nlayers - uncond_evals - m_lazy_evals;
    shadingsys().m_stat_rebinds += m_rebinds;
    shadingsys().m_stat_binds += m_binds;
    shadingsys().m_stat_paramstobind += m_paramstobind;
    shadingsys().m_stat_paramsbound += m_paramsbound;
    shadingsys().m_stat_instructions_run += m_instructions_run;
#ifdef DEBUG_ADJUST_VARYING
    for (size_t layer = 0;  layer < nlayers;  ++layer) {
        ShadingExecution &exec (execlayers[layer]);
        shadingsys().m_adjust_calls += exec.m_adjust_calls;
        shadingsys().m_keep_varying += exec.m_keep_varying;
        shadingsys().m_keep_uniform += exec.m_keep_uniform;
        shadingsys().m_make_varying += exec.m_make_varying;
        shadingsys().m_make_uniform += exec.m_make_uniform;
    }
#endif

    int biggest_Ci = 0;
    int biggest_Ci_i = 0;
#if USE_RUNFLAGS
    SHADE_LOOP_RUNFLAGS_BEGIN (runflags, 0, m_npoints)
#elif USE_RUNINDICES
    SHADE_LOOP_INDICES_BEGIN (indices, nindices)
#elif USE_RUNSPANS
    SHADE_LOOP_SPANS_BEGIN (indices, nindices)
#endif
        int n = m_globals->Ci[i]->ncomponents();
        if (biggest_Ci < n)
            biggest_Ci_i = i;
        biggest_Ci = std::max (biggest_Ci, n);
        shadingsys().m_stat_total_Ci_components += n;
        bool has_small = false;
        for (int j = 0;  j < n;  ++j) {
            const Imath::Color3<float> &w (m_globals->Ci[i]->weight(j));
            if (w[0] < 0.0001 && w[1] < 0.0001 && w[2] < 0.0001) {
                shadingsys().m_stat_small_Ci_components += 1;
                has_small = true;
            }
            if (w[0] == 0.0 && w[1] == 0.0 && w[2] == 0.0)
                shadingsys().m_stat_zero_Ci_components += 1;
        }
        shadingsys().m_stat_Ci_has_small_components += has_small;
        shadingsys().m_stat_total_Cis += 1;
    SHADE_LOOP_END
    if (biggest_Ci > shadingsys().m_stat_biggest_Ci) {
        spin_lock lock (shadingsys().m_stat_mutex);
        shadingsys().m_stat_biggest_Ci = std::max ((int)shadingsys().m_stat_biggest_Ci, biggest_Ci);
        std::stringstream stream;
        stream << *(m_globals->Ci[biggest_Ci_i]);
        shadingsys().info ("Big Ci = %s\n", stream.str().c_str());
    }
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
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
