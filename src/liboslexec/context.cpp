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
    : m_shadingsys(shadingsys), m_renderer(m_shadingsys.renderer()),
      m_attribs(NULL), m_globals(NULL)
{
    m_shadingsys.m_stat_contexts += 1;
}



ShadingContext::~ShadingContext ()
{
    m_shadingsys.m_stat_contexts -= 1;
}



void
ShadingContext::execute_llvm (ShaderUse use, Runflag *rf, int *ind, int nind)
{
    ShaderGroup &sgroup (attribs()->shadergroup (use));

    RunLLVMGroupFunc run_func = sgroup.llvm_compiled_version();
    ASSERT (run_func); 

    // Client supplied ShaderGlobals.  Pipe fit to SSG with lots of 
    // extra copying.
    ASSERT (globals());

    // Ignore runflags for now
    SingleShaderGlobal my_sg;
    ShaderGlobals& sg = *(globals());
    size_t groupdata_size = sgroup.llvm_groupdata_size();
    DASSERT(groupdata_size * m_npoints <= m_heap.size());
#if USE_RUNFLAGS
    for (int i = 0; i < m_npoints; i++) {
        if (rf[i]) {
#elif USE_RUNINDICES
    for (int ind_ = 0;  ind_ < nind;  ++ind_) { {
        int i = ind[ind_];
#elif USE_RUNSPANS
    for (int nspans_ = nind/2; nspans_; --nspans_, ind += 2) {
        for (int i = ind[0];  i < ind[1];  ++i) {
#else
            { int i=0; ASSERT(0 && "not runflags, indices, or spans!");
#endif
            static Vec3 vzero (0,0,0);
            my_sg.P = sg.P[i];
            my_sg.dPdx = sg.dPdx[i];
            my_sg.dPdy = sg.dPdy[i];
            my_sg.I = sg.I.is_null() ? vzero : sg.I[i];
            my_sg.dIdx = sg.dIdx.is_null() ? vzero : sg.dIdx[i];
            my_sg.dIdy = sg.dIdy.is_null() ? vzero : sg.dIdy[i];
            my_sg.N = sg.N[i];
            my_sg.Ng = sg.Ng[i];
            my_sg.u = sg.u[i];
            my_sg.v = sg.v[i];
            my_sg.dudx = sg.dudx[i];
            my_sg.dudy = sg.dudy[i];
            my_sg.dvdx = sg.dvdx[i];
            my_sg.dvdy = sg.dvdy[i];
            my_sg.dPdu = sg.dPdu[i];
            my_sg.dPdv = sg.dPdv[i];
            my_sg.time = sg.time[i];
            my_sg.dtime = sg.dtime.is_null() ? 0.0f : sg.dtime[i];
            my_sg.dPdtime = sg.dtime.is_null() ? vzero : sg.dPdtime[i];
            my_sg.Ps = sg.Ps.is_null() ? vzero : sg.Ps[i];
            my_sg.dPsdx = sg.dPsdx.is_null() ? vzero : sg.dPsdx[i];
            my_sg.dPsdy = sg.dPsdy.is_null() ? vzero : sg.dPsdy[i];
            my_sg.renderstate = sg.renderstate[i];
            my_sg.context = this;
            my_sg.object2common = sg.object2common[i];
            my_sg.shader2common = sg.shader2common[i];
            my_sg.Ci = sg.Ci[i];
            my_sg.surfacearea = sg.surfacearea.is_null() ? 1.0f : sg.surfacearea[i];
            my_sg.iscameraray = sg.iscameraray;
            my_sg.isshadowray = sg.isshadowray;
            my_sg.isdiffuseray = sg.isdiffuseray;
            my_sg.isglossyray = sg.isglossyray;
            my_sg.flipHandedness = sg.flipHandedness;
            my_sg.backfacing = sg.backfacing;
            run_func (&my_sg, &m_heap[groupdata_size * i]);

            sg.Ci[i] = my_sg.Ci;
//            if (use == ShadUseDisplacement) {
            // FIXME -- should only do this extra work for disp shaders,
            // but at the moment we only use ShadUseSurface, even for disp!
                sg.P[i] = my_sg.P;
                sg.dPdx[i] = my_sg.dPdx;
                sg.dPdy[i] = my_sg.dPdy;
                sg.N[i] = my_sg.N;
//            }
        }
    }
}



bool
ShadingContext::prepare_execution (ShaderUse use, ShadingAttribState &sas,
                                   int npoints)
{
    DASSERT (use == ShadUseSurface);  // FIXME

    m_curuse = use;
    m_attribs = &sas;
    m_npoints = npoints;
    m_closures_allotted = 0;

    // Optimize if we haven't already
    ShaderGroup &sgroup (sas.shadergroup (use));
    if (sgroup.nlayers()) {
        sgroup.start_running (npoints);
        if (! sgroup.optimized()) {
            shadingsys().optimize_group (sas, sgroup);
        }
    } else {
       // empty shader - nothing to do!
       return false; 
    }

    // Allocate enough space on the heap
    size_t heap_size_needed = sgroup.llvm_groupdata_size() * m_npoints;
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
    m_closure_pool.clear();

    // Clear the message blackboard
    m_messages.clear ();

    return true;
}



void
ShadingContext::execute (ShaderUse use, ShadingAttribState &sas,
                         SingleShaderGlobal &ssg)
{
    if (! prepare_execution (use, sas, 1))
        return;

    m_globals = NULL;
    ShaderGroup &sgroup (m_attribs->shadergroup (use));
    DASSERT (sgroup.llvm_compiled_version());
    DASSERT (sgroup.llvm_groupdata_size() <= m_heap.size());
    ssg.context = this;
    ssg.Ci = NULL;
    RunLLVMGroupFunc run_func = sgroup.llvm_compiled_version();
    run_func (&ssg, &m_heap[0]);
}



void
ShadingContext::execute (ShaderUse use, int n, ShadingAttribState &sas,
                         ShaderGlobals &sg,
                         Runflag *rf, int *ind, int nind)
{
    // FIXME -- timers/stats
    if (shadingsys().debug())
        shadingsys().info ("execute context %p as %s for %d points", this, shaderusename(use), n);

    if (! prepare_execution (use, sas, n))
        return;

    m_globals = &sg;

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

    execute_llvm (use, runflags, indices, nindices);
}



Symbol *
ShadingContext::symbol (ShaderUse use, ustring name)
{
    ShaderGroup &sgroup (attribs()->shadergroup (use));
    int nlayers = sgroup.nlayers ();
    if (sgroup.llvm_compiled_version()) {
        for (int layer = nlayers-1;  layer >= 0;  --layer) {
            int symidx = sgroup[layer]->findsymbol (name);
            if (symidx >= 0)
                return sgroup[layer]->symbol (symidx);
        }
    }
    return NULL;
}



void *
ShadingContext::symbol_data (Symbol &sym, int gridpoint)
{
    ShaderGroup &sgroup (attribs()->shadergroup ((ShaderUse)m_curuse));
    if (sgroup.llvm_compiled_version()) {
        size_t offset = sgroup.llvm_groupdata_size() * gridpoint;
        offset += sym.dataoffset();
        return &m_heap[offset];
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
