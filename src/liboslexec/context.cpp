// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include <vector>
#include <string>
#include <cstdio>

#include <OpenImageIO/sysutil.h>
#include <OpenImageIO/timer.h>
#include <OpenImageIO/thread.h>

#include "oslexec_pvt.h"

OSL_NAMESPACE_ENTER

static mutex buffered_errors_mutex;



ShadingContext::ShadingContext (ShadingSystemImpl &shadingsys,
                                PerThreadInfo *threadinfo)
    : m_shadingsys(shadingsys), m_renderer(m_shadingsys.renderer()),
      m_group(NULL), m_max_warnings(shadingsys.max_warnings_per_thread()), m_dictionary(NULL)
{
    m_shadingsys.m_stat_contexts += 1;
    m_threadinfo = threadinfo ? threadinfo : shadingsys.get_perthread_info ();
    m_texture_thread_info = NULL;
}



ShadingContext::~ShadingContext ()
{
    process_errors ();
    m_shadingsys.m_stat_contexts -= 1;
    free_dict_resources ();
}



bool
ShadingContext::execute_init (ShaderGroup &sgroup, ShaderGlobals &ssg, bool run)
{
    if (m_group)
        execute_cleanup ();
    m_group = &sgroup;
    m_ticks = 0;

    // Optimize if we haven't already
    if (sgroup.nlayers()) {
        sgroup.start_running ();
        if (! sgroup.optimized()) {
            auto ctx = shadingsys().get_context(thread_info());
            shadingsys().optimize_group (sgroup, ctx);
            if (shadingsys().m_greedyjit && shadingsys().m_groups_to_compile_count) {
                // If we are greedily JITing, optimize/JIT everything now
                shadingsys().optimize_all_groups ();
            }
            shadingsys().release_context(ctx);
        }
        if (sgroup.does_nothing())
            return false;
    } else {
       // empty shader - nothing to do!
       return false;
    }

    int profile = shadingsys().m_profile;
    OIIO::Timer timer (profile ? OIIO::Timer::StartNow : OIIO::Timer::DontStartNow);

    // Allocate enough space on the heap
    size_t heap_size_needed = sgroup.llvm_groupdata_size();
    reserve_heap(heap_size_needed);
    // Zero out the heap memory we will be using
    if (shadingsys().m_clearmemory)
        memset (m_heap.get(), 0, heap_size_needed);

    // Set up closure storage
    m_closure_pool.clear();

    // Clear the message blackboard
    m_messages.clear ();

    // Clear miscellaneous scratch space
    m_scratch_pool.clear ();

    // Zero out stats for this execution
    clear_runtime_stats ();

    if (run) {
        RunLLVMGroupFunc run_func = sgroup.llvm_compiled_init();
        if (!run_func)
            return false;
        ssg.context = this;
        ssg.renderer = renderer();
        ssg.Ci = NULL;
        run_func (&ssg, m_heap.get());
    }

    if (profile)
        m_ticks += timer.ticks();
    return true;
}



bool
ShadingContext::execute_layer (ShaderGlobals &ssg, int layernumber)
{
    if (!group() || group()->nlayers() == 0 || group()->does_nothing())
        return false;
    OSL_DASSERT(ssg.context == this && ssg.renderer == renderer());

    int profile = shadingsys().m_profile;
    OIIO::Timer timer (profile ? OIIO::Timer::StartNow : OIIO::Timer::DontStartNow);

    RunLLVMGroupFunc run_func = group()->llvm_compiled_layer (layernumber);
    if (! run_func)
        return false;

    run_func (&ssg, m_heap.get());

    if (profile)
        m_ticks += timer.ticks();

    return true;
}



bool
ShadingContext::execute_cleanup ()
{
    if (! group()) {
        errorf("execute_cleanup called again on a cleaned-up context");
        return false;
    }

    // Process any queued up error messages, warnings, printfs from shaders
    process_errors ();

    if (shadingsys().m_profile) {
        record_runtime_stats ();   // Transfer runtime stats to the shadingsys
        shadingsys().m_stat_total_shading_time_ticks += m_ticks;
        group()->m_stat_total_shading_time_ticks += m_ticks;
    }

    return true;
}



bool
ShadingContext::execute (ShaderGroup &sgroup, ShaderGlobals &ssg, bool run)
{
    int n = sgroup.m_exec_repeat;
    Vec3 Psave, Nsave;   // for repeats
    bool repeat = (n > 1);
    if (repeat) {
        // If we're going to repeat more than once, we need to save any
        // globals that might get modified.
        Psave = ssg.P;
        Nsave = ssg.N;
        if (! run)
            n = 1;
    }

    bool result = true;
    while (1) {
        if (! execute_init (sgroup, ssg, run))
            return false;
        if (run && n)
            execute_layer (ssg, group()->nlayers()-1);
        result = execute_cleanup ();
        if (--n < 1)
            break;   // done
        if (repeat) {
            // Going around for another pass... restore things as best as we
            // can.
            ssg.P = Psave;
            ssg.N = Nsave;
            ssg.Ci = NULL;
        }
    }
    return result;
}



void
ShadingContext::record_error (ErrorHandler::ErrCode code,
                              const std::string &text) const
{
    m_buffered_errors.emplace_back(code,text);
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



const Symbol *
ShadingContext::symbol (ustring layername, ustring symbolname) const
{
    return group()->find_symbol (layername, symbolname);
}



const void *
ShadingContext::symbol_data (const Symbol &sym) const
{
    const ShaderGroup &sgroup (*group());
    if (! sgroup.optimized())
        return NULL;   // can't retrieve symbol if we didn't optimize it

    if (sym.dataoffset() >= 0 && (int)m_heapsize > sym.dataoffset()) {
        // lives on the heap
        return m_heap.get() + sym.dataoffset();
    }

    // doesn't live on the heap
    if ((sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam) &&
        (sym.valuesource() == Symbol::DefaultVal || sym.valuesource() == Symbol::InstanceVal)) {
        return sym.data();
    }

    return NULL;  // not something we can retrieve
}



const regex &
ShadingContext::find_regex (ustring r)
{
    RegexMap::const_iterator found = m_regex_map.find (r);
    if (found != m_regex_map.end())
        return *found->second;
    // otherwise, it wasn't found, add it
    m_regex_map[r].reset (new regex(r.c_str()));
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

    if (array_lookup)
        ok = renderer()->get_array_attribute (sg, dest_derivs,
                                              obj_name, attr_type,
                                              attr_name, index, attr_dest);
    else
        ok = renderer()->get_attribute (sg, dest_derivs,
                                        obj_name, attr_type,
                                        attr_name, attr_dest);

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
