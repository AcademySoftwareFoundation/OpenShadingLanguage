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

OSL_NAMESPACE_ENTER

static mutex buffered_errors_mutex;



ShadingContext::ShadingContext (ShadingSystemImpl &shadingsys,
                                PerThreadInfo *threadinfo)
    : m_shadingsys(shadingsys), m_renderer(m_shadingsys.renderer()),
      m_attribs(NULL), m_max_warnings(shadingsys.max_warnings_per_thread()), m_dictionary(NULL), m_next_failed_attrib(0)
{
    m_shadingsys.m_stat_contexts += 1;
    m_threadinfo = threadinfo ? threadinfo : shadingsys.get_perthread_info ();
    m_texture_thread_info = shadingsys.texturesys()->get_perthread_info ();
}



ShadingContext::~ShadingContext ()
{
    process_errors ();
    m_shadingsys.m_stat_contexts -= 1;
    for (RegexMap::iterator it = m_regex_map.begin(); it != m_regex_map.end(); ++it) {
      delete it->second;
    }
    free_dict_resources ();
}



bool
ShadingContext::execute (ShaderGroup &sgroup, ShaderGlobals &ssg, bool run)
{
    m_texture_thread_info = m_shadingsys.texturesys()->get_perthread_info ();
    m_attribs = &sgroup;

    // Optimize if we haven't already
    if (sgroup.nlayers()) {
        sgroup.start_running ();
        if (! sgroup.optimized()) {
            shadingsys().optimize_group (sgroup);
            if (shadingsys().m_greedyjit && shadingsys().m_groups_to_compile_count) {
                // If we are greedily JITing, optimize/JIT everything now
                shadingsys().optimize_all_groups ();
            }
        }
        if (sgroup.does_nothing())
            return false;
    } else {
       // empty shader - nothing to do!
       return false;
    }

    int profile = shadingsys().m_profile;
    OIIO::Timer timer (profile);

    // Allocate enough space on the heap
    size_t heap_size_needed = sgroup.llvm_groupdata_size();
    if (heap_size_needed > m_heap.size()) {
        if (shadingsys().debug())
            info ("  ShadingContext %p growing heap to %llu",
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

    // Clear miscellaneous scratch space
    m_scratch_pool.clear ();

    // Zero out stats for this execution
    clear_runtime_stats ();

    if (run) {
        ssg.context = this;
        ssg.renderer = renderer();
        ssg.Ci = NULL;
        RunLLVMGroupFunc run_func = sgroup.llvm_compiled_version();
        DASSERT (run_func);
        DASSERT (sgroup.llvm_groupdata_size() <= m_heap.size());
        run_func (&ssg, &m_heap[0]);
    }

    // Process any queued up error messages, warnings, printfs from shaders
    process_errors ();

    if (profile) {
        record_runtime_stats ();   // Transfer runtime stats to the shadingsys
        long long ticks = timer.ticks();
        shadingsys().m_stat_total_shading_time_ticks += ticks;
        sgroup.m_stat_total_shading_time_ticks += ticks;
    }

    return true;
}



void
ShadingContext::record_error (ErrorHandler::ErrCode code,
                              const std::string &text) const
{
    m_buffered_errors.push_back (ErrorItem(code,text));
    // If we aren't buffering, just process immediately
    if (! shadingsys().m_buffer_printf)
        process_errors ();
}



void
ShadingContext::process_errors () const
{
    size_t nerrors = m_buffered_errors.size();
    if (! nerrors)
        return;

    // Use a mutex to make sure output from different threads stays
    // together, at least for one shader invocation, rather than being
    // interleaved with other threads.
    lock_guard lock (buffered_errors_mutex);

    for (size_t i = 0;  i < nerrors;  ++i) {
        switch (m_buffered_errors[i].first) {
        case ErrorHandler::EH_MESSAGE :
        case ErrorHandler::EH_DEBUG :
           shadingsys().message (m_buffered_errors[i].second);
            break;
        case ErrorHandler::EH_INFO :
            shadingsys().info (m_buffered_errors[i].second);
            break;
        case ErrorHandler::EH_WARNING :
            shadingsys().warning (m_buffered_errors[i].second);
            break;
        case ErrorHandler::EH_ERROR :
        case ErrorHandler::EH_SEVERE :
            shadingsys().error (m_buffered_errors[i].second);
            break;
        default:
            break;
        }
    }
    m_buffered_errors.clear();
}



Symbol *
ShadingContext::symbol (ustring layername, ustring symbolname)
{
    ShaderGroup &sgroup (*attribs());
    int nlayers = sgroup.nlayers ();
    if (sgroup.llvm_compiled_version()) {
        for (int layer = nlayers-1;  layer >= 0;  --layer) {
            ShaderInstance *inst (sgroup[layer]);
            if (layername.size() && layername != inst->layername())
                continue;  // They asked for a specific layer and this isn't it
            int symidx = inst->findsymbol (symbolname);
            if (symidx >= 0)
                return inst->symbol (symidx);
        }
    }
    return NULL;
}



void *
ShadingContext::symbol_data (Symbol &sym)
{
    ShaderGroup &sgroup (*attribs());
    if (! sgroup.llvm_compiled_version())
        return NULL;   // can't retrieve symbol if we didn't JIT and runit

    if (sym.dataoffset() >= 0 && (int)m_heap.size() > sym.dataoffset()) {
        // lives on the heap
        return &m_heap[sym.dataoffset()];
    }

    // doesn't live on the heap
    if ((sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam) &&
        (sym.valuesource() == Symbol::DefaultVal || sym.valuesource() == Symbol::InstanceVal)) {
        ASSERT (sym.data());
        return sym.data() ? sym.data() : NULL;
    }

    return NULL;  // not something we can retrieve
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
ShadingContext::osl_get_attribute (ShaderGlobals *sg, void *objdata,
                                   int dest_derivs,
                                   ustring obj_name, ustring attr_name,
                                   int array_lookup, int index,
                                   TypeDesc attr_type, void *attr_dest)
{
#if 0
    // Change the #if's below if you want to
    OIIO::Timer timer;
#endif
    bool ok;

    for (int i = 0;  i < FAILED_ATTRIBS;  ++i) {
        if ((obj_name || m_failed_attribs[i].objdata == objdata) &&
            m_failed_attribs[i].attr_name == attr_name &&
            m_failed_attribs[i].obj_name == obj_name &&
            m_failed_attribs[i].attr_type == attr_type &&
            m_failed_attribs[i].array_lookup == array_lookup &&
            m_failed_attribs[i].index == index &&
            m_failed_attribs[i].objdata) {
#if 0
            double time = timer();
            shadingsys().m_stat_getattribute_time += time;
            shadingsys().m_stat_getattribute_fail_time += time;
            shadingsys().m_stat_getattribute_calls += 1;
#endif
            return false;
        }
    }

    if (array_lookup)
        ok = renderer()->get_array_attribute (sg, dest_derivs,
                                              obj_name, attr_type,
                                              attr_name, index, attr_dest);
    else
        ok = renderer()->get_attribute (sg, dest_derivs,
                                        obj_name, attr_type,
                                        attr_name, attr_dest);
    if (!ok) {
        int i = m_next_failed_attrib;
        m_failed_attribs[i].objdata = objdata;
        m_failed_attribs[i].obj_name = obj_name;
        m_failed_attribs[i].attr_name = attr_name;
        m_failed_attribs[i].attr_type = attr_type;
        m_failed_attribs[i].array_lookup = array_lookup;
        m_failed_attribs[i].index = index;
        m_next_failed_attrib = (i == FAILED_ATTRIBS-1) ? 0 : (i+1);
    }

#if 0
    double time = timer();
    shadingsys().m_stat_getattribute_time += time;
    if (!ok)
        shadingsys().m_stat_getattribute_fail_time += time;
    shadingsys().m_stat_getattribute_calls += 1;
#endif
//    std::cout << "getattribute! '" << obj_name << "' " << attr_name << ' ' << attr_type.c_str() << " ok=" << ok << ", objdata was " << objdata << "\n";
    return ok;
}



OSL_SHADEOP void
osl_incr_layers_executed (ShaderGlobals *sg)
{
    ShadingContext *ctx = (ShadingContext *)sg->context;
    ctx->incr_layers_executed ();
}


OSL_NAMESPACE_EXIT
