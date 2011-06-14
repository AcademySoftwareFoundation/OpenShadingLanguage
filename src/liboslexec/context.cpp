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
#include <boost/regex.hpp>

#include <OpenImageIO/dassert.h>
#include <OpenImageIO/sysutil.h>
#include <OpenImageIO/timer.h>
#include <OpenImageIO/thread.h>

#include "oslexec_pvt.h"
#include "oslops.h"

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {

namespace pvt {   // OSL::pvt

using OIIO::Timer;


ShadingContext::ShadingContext (ShadingSystemImpl &shadingsys) 
    : m_shadingsys(shadingsys), m_renderer(m_shadingsys.renderer()),
      m_attribs(NULL), m_dictionary(NULL), m_next_failed_attrib(0)
{
    m_shadingsys.m_stat_contexts += 1;
}



ShadingContext::~ShadingContext ()
{
    m_shadingsys.m_stat_contexts -= 1;
    for (RegexMap::iterator it = m_regex_map.begin(); it != m_regex_map.end(); ++it) {
      delete it->second;
    }
    free_dict_resources ();
}



bool
ShadingContext::prepare_execution (ShaderUse use, ShadingAttribState &sas)
{
    DASSERT (use == ShadUseSurface);  // FIXME

    m_curuse = use;
    m_attribs = &sas;
    m_closures_allotted = 0;

    // Optimize if we haven't already
    ShaderGroup &sgroup (sas.shadergroup (use));
    if (sgroup.nlayers()) {
        sgroup.start_running ();
        if (! sgroup.optimized()) {
            shadingsys().optimize_group (sas, sgroup);
        }
    } else {
       // empty shader - nothing to do!
       return false; 
    }

    // Allocate enough space on the heap
    size_t heap_size_needed = sgroup.llvm_groupdata_size();
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
                         ShaderGlobals &ssg)
{
    if (! prepare_execution (use, sas))
        return;

    ShaderGroup &sgroup (m_attribs->shadergroup (use));
    DASSERT (sgroup.llvm_compiled_version());
    DASSERT (sgroup.llvm_groupdata_size() <= m_heap.size());
    ssg.context = this;
    ssg.Ci = NULL;
    RunLLVMGroupFunc run_func = sgroup.llvm_compiled_version();
    run_func (&ssg, &m_heap[0]);
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
    RegexMap::const_iterator found = m_regex_map.find (r);
    if (found != m_regex_map.end())
        return *found->second;
    // otherwise, it wasn't found, add it
    m_regex_map[r] = new boost::regex(r.c_str());
    m_shadingsys.m_stat_regexes += 1;
    // std::cerr << "Made new regex for " << r << "\n";
    return *m_regex_map[r];
}



bool
ShadingContext::osl_get_attribute (void *renderstate, void *objdata,
                                   int dest_derivs,
                                   ustring obj_name, ustring attr_name,
                                   int array_lookup, int index,
                                   TypeDesc attr_type, void *attr_dest)
{
    Timer timer;
    bool ok;

    for (int i = 0;  i < FAILED_ATTRIBS;  ++i) {
        if ((obj_name || m_failed_attribs[i].objdata == objdata) &&
            m_failed_attribs[i].attr_name == attr_name &&
            m_failed_attribs[i].obj_name == obj_name &&
            m_failed_attribs[i].array_lookup == array_lookup &&
            m_failed_attribs[i].index == index &&
            m_failed_attribs[i].objdata) {
            double time = timer();
            shadingsys().m_stat_getattribute_time += time;
            shadingsys().m_stat_getattribute_fail_time += time;
            shadingsys().m_stat_getattribute_calls += 1;
            return false;
        }
    }

    if (array_lookup)
        ok = renderer()->get_array_attribute (renderstate, dest_derivs,
                                              obj_name, attr_type,
                                              attr_name, index, attr_dest);
    else
        ok = renderer()->get_attribute (renderstate, dest_derivs,
                                        obj_name, attr_type,
                                        attr_name, attr_dest);
    if (!ok) {
        int i = m_next_failed_attrib;
        m_failed_attribs[i].objdata = objdata;
        m_failed_attribs[i].obj_name = obj_name;
        m_failed_attribs[i].attr_name = attr_name;
        m_failed_attribs[i].array_lookup = array_lookup;
        m_failed_attribs[i].index = index;
        m_next_failed_attrib = (i == FAILED_ATTRIBS-1) ? 0 : (i+1);
    }

    double time = timer();
    shadingsys().m_stat_getattribute_time += time;
    if (!ok)
        shadingsys().m_stat_getattribute_fail_time += time;
    shadingsys().m_stat_getattribute_calls += 1;
//    std::cout << "getattribute! '" << obj_name << "' " << attr_name << ' ' << attr_type.c_str() << " ok=" << ok << ", objdata was " << objdata << "\n";
    return ok;
}


}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
