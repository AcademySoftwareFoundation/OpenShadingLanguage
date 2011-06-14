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

#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>

#include "oslexec_pvt.h"
#include "genclosure.h"
#include "llvm_headers.h"

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"
#include "OpenImageIO/filesystem.h"
#ifdef OIIO_NAMESPACE
namespace Filesystem = OIIO::Filesystem;
#endif

using namespace OSL;
using namespace OSL::pvt;

// avoid naming conflict with MSVC macro
#ifdef RGB
#undef RGB
#endif

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {


ShadingSystem *
ShadingSystem::create (RendererServices *renderer,
                       TextureSystem *texturesystem,
                       ErrorHandler *err)
{
    // If client didn't supply an error handler, just use the default
    // one that echoes to the terminal.
    if (! err) {
        err = & ErrorHandler::default_handler ();
        ASSERT (err != NULL && "Can't create default ErrorHandler");
    }

    // Doesn't need a shared cache
    ShadingSystemImpl *ts = new ShadingSystemImpl (renderer, texturesystem, err);
#ifdef DEBUG
    err->info ("creating new ShadingSystem %p", (void *)ts);
#endif
    return ts;
}



void
ShadingSystem::destroy (ShadingSystem *x)
{
    delete (ShadingSystemImpl *) x;
}



ShadingSystem::ShadingSystem ()
{
}



ShadingSystem::~ShadingSystem ()
{
}



namespace Strings {

// Define static ustring symbols for very fast comparison
ustring camera ("camera"), common ("common");
ustring object ("object"), shader ("shader");
ustring rgb ("rgb"), RGB ("RGB");
ustring hsv ("hsv"), hsl ("hsl"), YIQ ("YIQ"), xyz ("xyz");
ustring null ("null"), default_("default");
ustring label ("label");
ustring sidedness ("sidedness"), front ("front"), back ("back"), both ("both");
ustring P ("P"), I ("I"), N ("N"), Ng ("Ng");
ustring dPdu ("dPdu"), dPdv ("dPdv"), u ("u"), v ("v"), Ps ("Ps");
ustring time ("time"), dtime ("dtime"), dPdtime ("dPdtime");
ustring Ci ("Ci");
ustring width ("width"), swidth ("swidth"), twidth ("twidth"), rwidth ("rwidth");
ustring blur ("blur"), sblur ("sblur"), tblur ("tblur"), rblur ("rblur");
ustring wrap ("wrap"), swrap ("swrap"), twrap ("twrap"), rwrap ("rwrap");
ustring black ("black"), clamp ("clamp");
ustring periodic ("periodic"), mirror ("mirror");
ustring firstchannel ("firstchannel"), fill ("fill"), alpha ("alpha");
ustring interp("interp"), closest("closest"), linear("linear");
ustring cubic("cubic"), smartbicubic("smartbicubic");
};



namespace pvt {   // OSL::pvt



ShadingSystemImpl::ShadingSystemImpl (RendererServices *renderer,
                                      TextureSystem *texturesystem,
                                      ErrorHandler *err)
    : m_renderer(renderer), m_texturesys(texturesystem), m_err(err),
      m_statslevel (0), m_debug (false), m_lazylayers (true),
      m_lazyglobals (false),
      m_clearmemory (false), m_rebind (false), m_debugnan (false),
      m_lockgeom_default (false), m_strict_messages(true),
      m_range_checking(true),
      m_optimize (1),
      m_llvm_debug(false),
      m_commonspace_synonym("world"),
      m_in_group (false),
      m_global_heap_total (0),
      m_stat_opt_locking_time(0), m_stat_specialization_time(0),
      m_stat_total_llvm_time(0),
      m_stat_llvm_setup_time(0), m_stat_llvm_irgen_time(0),
      m_stat_llvm_opt_time(0), m_stat_llvm_jit_time(0),
      m_llvm_context (NULL),
      m_llvm_module (NULL),
      m_llvm_exec (NULL),
      m_llvm_jitmm (NULL)
{
    m_stat_shaders_loaded = 0;
    m_stat_shaders_requested = 0;
    m_stat_groups = 0;
    m_stat_groupinstances = 0;
    m_stat_regexes = 0;
    m_layers_executed_uncond = 0;
    m_layers_executed_lazy = 0;
    m_layers_executed_never = 0;
    m_stat_binds = 0;
    m_stat_rebinds = 0;
    m_stat_paramstobind = 0;
    m_stat_paramsbound = 0;
    m_stat_instructions_run = 0;
    m_stat_total_syms = 0;
    m_stat_syms_with_derivs = 0;
    m_stat_optimization_time = 0;
    m_stat_getattribute_time = 0;
    m_stat_getattribute_fail_time = 0;
    m_stat_getattribute_calls = 0;

    init_global_heap_offsets ();

    // If client didn't supply an error handler, just use the default
    // one that echoes to the terminal.
    if (! m_err) {
        m_err = & ErrorHandler::default_handler ();
    }

#if 0
    // If client didn't supply renderer services, create a default one
    if (! m_renderer) {
        m_renderer = NULL;
        ASSERT (m_renderer);
    }
#endif

    // If client didn't supply a texture system, create a new one
    if (! m_texturesys) {
        m_texturesys = TextureSystem::create (true /* shared */);
        ASSERT (m_texturesys);
        // Make some good guesses about default options
        m_texturesys->attribute ("automip",  1);
        m_texturesys->attribute ("autotile", 64);
    }

    // Alternate way of turning on LLVM debug mode (temporary/experimental)
    const char *llvm_debug_env = getenv ("OSL_LLVM_DEBUG");
    if (llvm_debug_env && *llvm_debug_env)
        m_llvm_debug = atoi(llvm_debug_env);

    // Initialize a default set of raytype names.  A particular renderer
    // can override this, add custom names, or change the bits around,
    // if this default ordering is not to its liking.
    static const char *raytypes[] = {
        /*1*/ "camera", /*2*/ "shadow", /*4*/ "reflection", /*8*/ "refraction",
        /*16*/ "diffuse", /*32*/ "glossy"
    };
    const int nraytypes = sizeof(raytypes)/sizeof(raytypes[0]);
    attribute ("raytypes", TypeDesc(TypeDesc::STRING,nraytypes), raytypes);
}



void
ShadingSystemImpl::register_closure(const char *name, int id, const ClosureParam *params, int size,
                                    PrepareClosureFunc prepare, SetupClosureFunc setup, CompareClosureFunc compare)
{
    for (int i = 0; params && params[i].type != TypeDesc(); ++i) {
        if (params[i].key == NULL && params[i].type.size() != (size_t)params[i].field_size) {
            error ("Parameter %d of '%s' closure is assigned to a field of incompatible size", i + 1, name);
            return;
        }
    }
    m_closure_registry.register_closure(name, id, params, size, prepare, setup, compare);
}



ShadingSystemImpl::~ShadingSystemImpl ()
{
    printstats ();
    // N.B. just let m_texsys go -- if we asked for one to be created,
    // we asked for a shared one.

    delete m_llvm_exec;
    // NOTE(boulos): Deleting the execution engine should in theory
    // clean up the module. Calling delete on the module here results
    // in a crash (suggesting theory meets practice).

    // delete m_llvm_module;

    delete m_llvm_context;

    // Delete the retained JIT memory manager
    delete m_llvm_jitmm;

    // FIXME(boulos): According to the docs, we should also call
    // llvm_shutdown once we're done. However, ~ShadingSystemImpl
    // seems like the wrong place for this since in a multi-threaded
    // implementation we might destroy this impl while having others
    // outstanding. I'll leave this as a fixme for now.

    //llvm::llvm_shutdown();
}



bool
ShadingSystemImpl::attribute (const std::string &name, TypeDesc type,
                              const void *val)
{
    lock_guard guard (m_mutex);  // Thread safety
    if (name == "searchpath:shader" && type == TypeDesc::STRING) {
        m_searchpath = std::string (*(const char **)val);
        Filesystem::searchpath_split (m_searchpath, m_searchpath_dirs);
        return true;
    }
    if (name == "statistics:level" && type == TypeDesc::INT) {
        m_statslevel = *(const int *)val;
        return true;
    }
    if (name == "debug" && type == TypeDesc::INT) {
        m_debug = *(const int *)val;
        return true;
    }
    if (name == "lazylayers" && type == TypeDesc::INT) {
        m_lazylayers = *(const int *)val;
        return true;
    }
    if (name == "lazyglobals" && type == TypeDesc::INT) {
        m_lazyglobals = *(const int *)val;
        return true;
    }
    if (name == "clearmemory" && type == TypeDesc::INT) {
        m_clearmemory = *(const int *)val;
        return true;
    }
    if (name == "rebind" && type == TypeDesc::INT) {
        m_rebind = *(const int *)val;
        return true;
    }
    if (name == "debugnan" && type == TypeDesc::INT) {
        m_debugnan = *(const int *)val;
        return true;
    }
    if (name == "lockgeom" && type == TypeDesc::INT) {
        m_lockgeom_default = *(const int *)val;
        return true;
    }
    if (name == "optimize" && type == TypeDesc::INT) {
        m_optimize = *(const int *)val;
        return true;
    }
    if (name == "llvm_debug" && type == TypeDesc::INT) {
        m_llvm_debug = *(const int *)val;
        return true;
    }
    if (name == "strict_messages" && type == TypeDesc::INT) {
        m_strict_messages = *(const int *)val;
        return true;
    }
    if (name == "range_checking" && type == TypeDesc::INT) {
        m_range_checking = *(const int *)val;
        return true;
    }
    if (name == "commonspace" && type == TypeDesc::STRING) {
        m_commonspace_synonym = ustring (*(const char **)val);
        return true;
    }
    if (name == "raytypes" && type.basetype == TypeDesc::STRING) {
        ASSERT (type.numelements() <= 32 &&
                "ShaderGlobals.raytype is an int, max of 32 raytypes");
        m_raytypes.clear ();
        for (size_t i = 0;  i < type.numelements();  ++i)
            m_raytypes.push_back (ustring(((const char **)val)[i]));
        return true;
    }
    return false;
}



bool
ShadingSystemImpl::getattribute (const std::string &name, TypeDesc type,
                                 void *val)
{
#define ATTR_DECODE(_name,_ctype,_src)                                  \
    if (name == _name && type == OIIO::BaseTypeFromC<_ctype>::value) {  \
        *(_ctype *)(val) = (_ctype)(_src);                              \
        return true;                                                    \
    }

    lock_guard guard (m_mutex);  // Thread safety
    if (name == "searchpath:shader" && type == TypeDesc::STRING) {
        *(const char **)val = m_searchpath.c_str();
        return true;
    }
    ATTR_DECODE ("statistics:level", int, m_statslevel);
    ATTR_DECODE ("debug", int, m_debug);
    ATTR_DECODE ("lazylayers", int, m_lazylayers);
    ATTR_DECODE ("lazyglobals", int, m_lazyglobals);
    ATTR_DECODE ("clearmemory", int, m_clearmemory);
    ATTR_DECODE ("rebind", int, m_rebind);
    ATTR_DECODE ("debugnan", int, m_debugnan);
    ATTR_DECODE ("lockgeom", int, m_lockgeom_default);
    ATTR_DECODE ("optimize", int, m_optimize);
    ATTR_DECODE ("llvm_debug", int, m_llvm_debug);
    ATTR_DECODE ("strict_messages", int, m_strict_messages);
    ATTR_DECODE ("range_checking", int, m_range_checking);
    ATTR_DECODE ("stat:masters", int, m_stat_shaders_loaded);
    ATTR_DECODE ("stat:groups", int, m_stat_groups);
    ATTR_DECODE ("stat:instances", int, m_stat_groupinstances);
    ATTR_DECODE ("stat:memory_current", long long, m_stat_memory.current());
    ATTR_DECODE ("stat:memory_peak", long long, m_stat_memory.peak());
    ATTR_DECODE ("stat:optimization_time", float, m_stat_optimization_time);
    
    return false;
#undef ATTR_DECODE
}



void
ShadingSystemImpl::error (const char *format, ...)
{
    va_list ap;
    va_start (ap, format);
    std::string msg = Strutil::vformat (format, ap);
    error (msg);
    va_end (ap);
}



void
ShadingSystemImpl::warning (const char *format, ...)
{
    va_list ap;
    va_start (ap, format);
    std::string msg = Strutil::vformat (format, ap);
    warning (msg);
    va_end (ap);
}



void
ShadingSystemImpl::info (const char *format, ...)
{
    va_list ap;
    va_start (ap, format);
    std::string msg = Strutil::vformat (format, ap);
    info (msg);
    va_end (ap);
}



void
ShadingSystemImpl::message (const char *format, ...)
{
    va_list ap;
    va_start (ap, format);
    std::string msg = Strutil::vformat (format, ap);
    message (msg);
    va_end (ap);
}



void
ShadingSystemImpl::error (const std::string &msg)
{
    lock_guard guard (m_errmutex);
    int n = 0;
    BOOST_FOREACH (std::string &s, m_errseen) {
        if (s == msg)
            return;
        ++n;
    }
    if (n >= m_errseenmax)
        m_errseen.pop_front ();
    m_errseen.push_back (msg);
    m_err->error (msg);
}



void
ShadingSystemImpl::warning (const std::string &msg)
{
    lock_guard guard (m_errmutex);
    int n = 0;
    BOOST_FOREACH (std::string &s, m_warnseen) {
        if (s == msg)
            return;
        ++n;
    }
    if (n >= m_errseenmax)
        m_warnseen.pop_front ();
    m_warnseen.push_back (msg);
    m_err->warning (msg);
}



void
ShadingSystemImpl::info (const std::string &msg)
{
    m_err->info (msg);
}



void
ShadingSystemImpl::message (const std::string &msg)
{
    m_err->message (msg);
}



std::string
ShadingSystemImpl::getstats (int level) const
{
    if (level <= 0)
        return "";
    std::ostringstream out;
    out << "OSL ShadingSystem statistics (" << (void*)this << ")\n";
    if (m_stat_shaders_requested == 0) {
        out << "  No shaders requested\n";
        return out.str();
    }
    out << "  Shaders:\n";
    out << "    Requested: " << m_stat_shaders_requested << "\n";
    out << "    Loaded:    " << m_stat_shaders_loaded << "\n";
    out << "    Masters:   " << m_stat_shaders_loaded << "\n";
    out << "    Instances: " << m_stat_instances << "\n";
    out << "  Shading contexts: " << m_stat_contexts << "\n";
    out << "  Shading groups:   " << m_stat_groups << "\n";
    out << "    Total instances in all groups: " << m_stat_groupinstances << "\n";
    float iperg = (float)m_stat_groupinstances/std::max(m_stat_groups,1);
    out << "    Avg instances per group: " 
        << Strutil::format ("%.1f", iperg) << "\n";

    long long totalexec = m_layers_executed_uncond + m_layers_executed_lazy +
                          m_layers_executed_never;
    out << Strutil::format ("  Total layers run: %10lld\n", totalexec);
    double inv_totalexec = 1.0 / std::max (totalexec, 1LL);  // prevent div by 0
    out << Strutil::format ("    Unconditional:  %10lld  (%.1f%%)\n",
                            (long long)m_layers_executed_uncond,
                            (100.0*m_layers_executed_uncond) * inv_totalexec);
    out << Strutil::format ("    On demand:      %10lld  (%.1f%%)\n",
                            (long long)m_layers_executed_lazy,
                            (100.0*m_layers_executed_lazy) * inv_totalexec);
    out << Strutil::format ("    Skipped:        %10lld  (%.1f%%)\n",
                            (long long)m_layers_executed_never,
                            (100.0*m_layers_executed_never) * inv_totalexec);

    out << Strutil::format ("  Binds:  %lld\n",
                            (long long)m_stat_binds);
    out << Strutil::format ("  Rebinds:  %lld / %lld  (%.1f%%)\n",
                            (long long)m_stat_rebinds, totalexec,
                            (100.0*m_stat_rebinds) * inv_totalexec);
    out << Strutil::format ("  Params bound:  %lld / %lld  (%.1f%%)\n",
                            (long long)m_stat_paramsbound,
                            (long long)m_stat_paramstobind,
                            (100.0*m_stat_paramsbound)/std::max((int)m_stat_paramstobind, 1));
    out << Strutil::format ("  Total instructions run:  %lld\n",
                            (long long)m_stat_instructions_run);
    out << Strutil::format ("  Derivatives needed on %d / %d symbols (%.1f%%)\n",
                            (int)m_stat_syms_with_derivs, (int)m_stat_total_syms,
                            (100.0*(int)m_stat_syms_with_derivs)/std::max((int)m_stat_total_syms,1));
    out << "  Runtime optimization cost: "
        << Strutil::timeintervalformat (m_stat_optimization_time, 2) << "\n";
    out << "    locking:                   "
        << Strutil::timeintervalformat (m_stat_opt_locking_time, 2) << "\n";
    out << "    runtime specialization:    "
        << Strutil::timeintervalformat (m_stat_specialization_time, 2) << "\n";
    if (m_stat_total_llvm_time > 0.0) {
        out << "    LLVM setup:                "
            << Strutil::timeintervalformat (m_stat_llvm_setup_time, 2) << "\n";
        out << "    LLVM IR gen:               "
            << Strutil::timeintervalformat (m_stat_llvm_irgen_time, 2) << "\n";
        out << "    LLVM optimize:             "
            << Strutil::timeintervalformat (m_stat_llvm_opt_time, 2) << "\n";
        out << "    LLVM JIT:                  "
            << Strutil::timeintervalformat (m_stat_llvm_jit_time, 2) << "\n";
    }

    out << "  Regex's compiled: " << m_stat_regexes << "\n";
    if (m_stat_getattribute_calls) {
        out << "  getattribute calls: " << m_stat_getattribute_calls << " ("
            << Strutil::timeintervalformat (m_stat_getattribute_time, 2) << ")\n";
        out << "     (fail time "
            << Strutil::timeintervalformat (m_stat_getattribute_fail_time, 2) << ")\n";
    }
    out << "  Memory total: " << m_stat_memory.memstat() << '\n';
    out << "    Master memory: " << m_stat_mem_master.memstat() << '\n';
    out << "        Master ops:            " << m_stat_mem_master_ops.memstat() << '\n';
    out << "        Master args:           " << m_stat_mem_master_args.memstat() << '\n';
    out << "        Master syms:           " << m_stat_mem_master_syms.memstat() << '\n';
    out << "        Master defaults:       " << m_stat_mem_master_defaults.memstat() << '\n';
    out << "        Master consts:         " << m_stat_mem_master_consts.memstat() << '\n';
    out << "    Instance memory: " << m_stat_mem_inst.memstat() << '\n';
    out << "        Instance ops:          " << m_stat_mem_inst_ops.memstat() << '\n';
    out << "        Instance args:         " << m_stat_mem_inst_args.memstat() << '\n';
    out << "        Instance syms:         " << m_stat_mem_inst_syms.memstat() << '\n';
    out << "        Instance param values: " << m_stat_mem_inst_paramvals.memstat() << '\n';
    out << "        Instance connections:  " << m_stat_mem_inst_connections.memstat() << '\n';

    if (m_llvm_jitmm) {
        size_t jitmem = m_llvm_jitmm->GetDefaultCodeSlabSize() * m_llvm_jitmm->GetNumCodeSlabs()
            + m_llvm_jitmm->GetDefaultDataSlabSize() * m_llvm_jitmm->GetNumDataSlabs()
            + m_llvm_jitmm->GetDefaultStubSlabSize() * m_llvm_jitmm->GetNumStubSlabs();
        out << "    LLVM JIT memory: " << Strutil::memformat(jitmem) << '\n';
    }

    return out.str();
}



void
ShadingSystemImpl::printstats () const
{
    if (m_statslevel == 0)
        return;
    m_err->message (getstats (m_statslevel));
}



bool
ShadingSystemImpl::Parameter (const char *name, TypeDesc t, const void *val)
{
    // We work very hard not to do extra copies of the data.  First,
    // grow the pending list by one (empty) slot...
    m_pending_params.resize (m_pending_params.size() + 1);
    // ...then initialize it in place
    m_pending_params.back().init (name, t, 1, val);
    return true;
}



bool
ShadingSystemImpl::ShaderGroupBegin (void)
{
    if (m_in_group) {
        error ("Nested ShaderGroupBegin() calls");
        return false;
    }
    m_in_group = true;
    m_group_use = ShadUseUnknown;
    return true;
}



bool
ShadingSystemImpl::ShaderGroupEnd (void)
{
    if (! m_in_group) {
        error ("ShaderGroupEnd() was called without ShaderGroupBegin()");
        return false;
    }

    // Mark the layers that can be run lazily
    if (m_group_use != ShadUseUnknown) {
        ShaderGroup &sgroup (m_curattrib->shadergroup (m_group_use));
        size_t nlayers = sgroup.nlayers ();
        for (size_t layer = 0;  layer < nlayers;  ++layer) {
            ShaderInstance *inst = sgroup[layer];
            if (! inst)
                continue;
            if (m_lazylayers) {
                // lazylayers option turned on: unconditionally run shaders
                // with no outgoing connections ("root" nodes, including the
                // last in the group) or shaders that alter global variables
                // (unless 'lazyglobals' is turned on).
                if (m_lazyglobals)
                    inst->run_lazily (inst->outgoing_connections());
                else
                    inst->run_lazily (inst->outgoing_connections() &&
                                      ! inst->writes_globals());
#if 0
                // Suggested warning below... but are there use cases where
                // people want these to run (because they will extract the
                // results they want from output params)?
                if (! inst->outgoing_connections() && ! inst->writes_globals())
                    warning ("Layer \"%s\" (shader %s) will run even though it appears to have no used results",
                             inst->layername().c_str(), inst->shadername().c_str());
#endif
            } else {
                // lazylayers option turned off: never run lazily
                inst->run_lazily (false);
            }
        }
    }

    m_in_group = false;
    m_group_use = ShadUseUnknown;
    return true;
}



bool
ShadingSystemImpl::Shader (const char *shaderusage,
                           const char *shadername,
                           const char *layername)
{
    // Make sure we have a current attrib state
    if (! m_curattrib)
        m_curattrib.reset (new ShadingAttribState);

    ShaderMaster::ref master = loadshader (shadername);
    if (! master) {
        error ("Could not find shader \"%s\"", shadername);
        return false;
    }

    ShaderUse use = shaderuse_from_name (shaderusage);
    if (use == ShadUseUnknown) {
        error ("Unknown shader usage \"%s\"", shaderusage);
        return false;
    }

    // If somebody is already hanging onto the shader state, clone it before
    // we modify it.
    if (! m_curattrib.unique ()) {
        ShadingAttribStateRef newstate (new ShadingAttribState (*m_curattrib));
        m_curattrib = newstate;
    }

    ShaderInstanceRef instance (new ShaderInstance (master, layername));
    instance->parameters (m_pending_params);
    m_pending_params.clear ();

    ShaderGroup &shadergroup (m_curattrib->shadergroup (use));
    if (! m_in_group || m_group_use == ShadUseUnknown) {
        // A singleton, or the first in a group
        shadergroup.clear ();
        m_stat_groups += 1;
    }
    if (m_in_group) {
        if (m_group_use == ShadUseUnknown) {  // First shader in group
            m_group_use = use;
        } else if (use != m_group_use) {
            error ("Shader usage \"%s\" does not match current group (%s)",
                   shaderusage, shaderusename (m_group_use));
            return false;
        }
    }

    shadergroup.append (instance);
    m_curattrib->changed_shaders ();
    m_stat_groupinstances += 1;

    // FIXME -- check for duplicate layer name within the group?

    return true;
}



bool
ShadingSystemImpl::ConnectShaders (const char *srclayer, const char *srcparam,
                                   const char *dstlayer, const char *dstparam)
{
    // Basic sanity checks -- make sure it's a legal time to call
    // ConnectShaders, and that the layer and parameter names are not empty.
    if (! m_in_group) {
        error ("ConnectShaders can only be called within ShaderGroupBegin/End");
        return false;
    }
    if (!srclayer || !srclayer[0] || !srcparam || !srcparam[0]) {
        error ("ConnectShaders: badly formed source layer/parameter");
        return false;
    }
    if (!dstlayer || !dstlayer[0] || !dstparam || !dstparam[0]) {
        error ("ConnectShaders: badly formed destination layer/parameter");
        return false;
    }

    // Decode the layers, finding the indices within our group and
    // pointers to the instances.  Error and return if they are not found,
    // or if it's not connecting an earlier src to a later dst.
    ShaderInstance *srcinst, *dstinst;
    int srcinstindex = find_named_layer_in_group (ustring(srclayer), srcinst);
    int dstinstindex = find_named_layer_in_group (ustring(dstlayer), dstinst);
    if (! srcinst) {
        error ("ConnectShaders: source layer \"%s\" not found", srclayer);
        return false;
    }
    if (! dstinst) {
        error ("ConnectShaders: destination layer \"%s\" not found", dstlayer);
        return false;
    }
    if (dstinstindex <= srcinstindex) {
        error ("ConnectShaders: destination layer must follow source layer\n");
        return false;
    }

    // Decode the parameter names, find their symbols in their
    // respective layers, and also decode requrest to attach specific
    // array elements or color/vector channels.
    ConnectedParam srccon = decode_connected_param(srcparam, srclayer, srcinst);
    ConnectedParam dstcon = decode_connected_param(dstparam, dstlayer, dstinst);
    if (! (srccon.valid() && dstcon.valid())) {
        
        return false;
    }

    if (srccon.type.is_structure() && dstcon.type.is_structure() &&
            equivalent (srccon.type, dstcon.type)) {
        // If the connection is whole struct-to-struct (and they are
        // structs with equivalent data layout), implement it underneath
        // as connections between their respective fields.
        StructSpec *srcstruct = srccon.type.structspec();
        StructSpec *dststruct = dstcon.type.structspec();
        for (size_t i = 0;  i < srcstruct->numfields();  ++i) {
            std::string s = Strutil::format("%s.%s", srcparam, srcstruct->field(i).name.c_str());
            std::string d = Strutil::format("%s.%s", dstparam, dststruct->field(i).name.c_str());
            ConnectShaders (srclayer, s.c_str(), dstlayer, d.c_str());
        }
        return true;
    }

    if (! assignable (dstcon.type, srccon.type)) {
        error ("ConnectShaders: cannot connect a %s (%s) to a %s (%s)",
               srccon.type.c_str(), srcparam, dstcon.type.c_str(), dstparam);
        return false;
    }

    dstinst->add_connection (srcinstindex, srccon, dstcon);
    dstinst->symbol(dstcon.param)->valuesource (Symbol::ConnectedVal);
    srcinst->outgoing_connections (true);

    if (debug())
        m_err->message ("ConnectShaders %s %s -> %s %s\n",
                        srclayer, srcparam, dstlayer, dstparam);

    return true;
}



ShadingAttribStateRef
ShadingSystemImpl::state () const
{
    return m_curattrib;
}



void
ShadingSystemImpl::clear_state ()
{
    m_curattrib.reset (new ShadingAttribState);
}




void*
ShadingSystemImpl::create_thread_info()
{
    return (void*) new PerThreadInfo();
}




void
ShadingSystemImpl::destroy_thread_info(void* thread_info)
{
    delete ((PerThreadInfo*) thread_info);
}



ShadingContext *
ShadingSystemImpl::get_context (void* thread_info)
{
    PerThreadInfo *threadinfo = thread_info == NULL ? get_perthread_info () : (PerThreadInfo*) thread_info;
    if (threadinfo->context_pool.empty()) {
        return new ShadingContext (*this);
    } else {
        return threadinfo->pop_context ();
    }
}



void
ShadingSystemImpl::release_context (ShadingContext *sc, void* thread_info)
{
    PerThreadInfo *threadinfo = thread_info == NULL ? get_perthread_info () : (PerThreadInfo*) thread_info;
    threadinfo->context_pool.push (sc);
}



ShadingContext *
ShadingSystemImpl::PerThreadInfo::pop_context ()
{
    ShadingContext *sc = context_pool.top ();
    context_pool.pop ();
    return sc;
}



ShadingSystemImpl::PerThreadInfo::~PerThreadInfo ()
{
    while (! context_pool.empty())
        delete pop_context ();
}



int
ShadingSystemImpl::find_named_layer_in_group (ustring layername,
                                              ShaderInstance * &inst)
{
    inst = NULL;
    if (m_group_use >= ShadUseUnknown)
        return -1;
    ShaderGroup &group (m_curattrib->shadergroup (m_group_use));
    for (int i = 0;  i < group.nlayers();  ++i) {
        if (group[i]->layername() == layername) {
            inst = group[i];
            return i;
        }
    }
    return -1;
}



ConnectedParam
ShadingSystemImpl::decode_connected_param (const char *connectionname,
                                   const char *layername, ShaderInstance *inst)
{
    ConnectedParam c;  // initializes to "invalid"

    // Look for a bracket in the "parameter name"
    const char *bracket = strchr (connectionname, '[');
    // Grab just the part of the param name up to the bracket
    ustring param (connectionname, 0,
                   bracket ? size_t(bracket-connectionname) : ustring::npos);

    // Search for the param with that name, fail if not found
    c.param = inst->findsymbol (param);
    if (c.param < 0) {
        error ("ConnectShaders: \"%s\" is not a parameter or global of layer \"%s\" (shader \"%s\")",
               param.c_str(), layername, inst->shadername().c_str());
        return c;
    }

    Symbol *sym = inst->symbol (c.param);
    ASSERT (sym);

    // Only params, output params, and globals are legal for connections
    if (! (sym->symtype() == SymTypeParam ||
           sym->symtype() == SymTypeOutputParam ||
           sym->symtype() == SymTypeGlobal)) {
        error ("ConnectShaders: \"%s\" is not a parameter or global of layer \"%s\" (shader \"%s\")",
               param.c_str(), layername, inst->shadername().c_str());
        c.param = -1;  // mark as invalid
        return c;
    }

    c.type = sym->typespec();

    if (bracket && c.type.arraylength()) {
        // There was at least one set of brackets that appears to be
        // selecting an array element.
        c.arrayindex = atoi (bracket+1);
        if (c.arrayindex >= c.type.arraylength()) {
            error ("ConnectShaders: cannot request array element %s from a %s",
                   connectionname, c.type.c_str());
            c.arrayindex = c.type.arraylength() - 1;  // clamp it
        }
        c.type.make_array (0);              // chop to the element type
        c.offset += c.type.simpletype().size() * c.arrayindex;
        bracket = strchr (bracket+1, '[');  // skip to next bracket
    }

    if (bracket && ! c.type.is_closure() &&
            c.type.aggregate() != TypeDesc::SCALAR) {
        // There was at least one set of brackets that appears to be 
        // selecting a color/vector component.
        c.channel = atoi (bracket+1);
        if (c.channel >= (int)c.type.aggregate()) {
            error ("ConnectShaders: cannot request component %s from a %s",
                   connectionname, c.type.c_str());
            c.channel = (int)c.type.aggregate() - 1;  // clamp it
        }
        // chop to just the scalar part
        c.type = TypeSpec ((TypeDesc::BASETYPE)c.type.simpletype().basetype);
        c.offset += c.type.simpletype().size() * c.channel;
        bracket = strchr (bracket+1, '[');     // skip to next bracket
    }

    // Deal with left over brackets
    if (bracket) {
        // Still a leftover bracket, no idea what to do about that
        error ("ConnectShaders: don't know how to connect '%s' when \"%s\" is a \"%s\"",
               connectionname, param.c_str(), c.type.c_str());
        c.param = -1;  // mark as invalid
    }
    return c;
}



void
ShadingSystemImpl::init_global_heap_offsets ()
{
    lock_guard lock (m_mutex);
    if (m_global_heap_total > 0)
        return;   // Already initialized

    // FIXME -- these are all wrong (but luckily unused).  They don't 
    // properly consider npoints, let alone derivs or alignment.  Ugh.
    // Leave here for right now, anticipate future deletion or fixing.
    const int triple_size = sizeof (Vec3);
    m_global_heap_offsets[ustring("P")] = m_global_heap_total;
    m_global_heap_total += triple_size;
    m_global_heap_offsets[ustring("I")] = m_global_heap_total;
    m_global_heap_total += triple_size;
    m_global_heap_offsets[ustring("N")] = m_global_heap_total;
    m_global_heap_total += triple_size;
    m_global_heap_offsets[ustring("Ng")] = m_global_heap_total;
    m_global_heap_total += triple_size;
    m_global_heap_offsets[ustring("dPdu")] = m_global_heap_total;
    m_global_heap_total += triple_size;
    m_global_heap_offsets[ustring("dPdv")] = m_global_heap_total;
    m_global_heap_total += triple_size;
    m_global_heap_offsets[ustring("dPdtime")] = m_global_heap_total;
    m_global_heap_total += triple_size;
    m_global_heap_offsets[ustring("Ps")] = m_global_heap_total;
    m_global_heap_total += triple_size;

    m_global_heap_offsets[ustring("u")] = m_global_heap_total;
    m_global_heap_total += sizeof (float);
    m_global_heap_offsets[ustring("v")] = m_global_heap_total;
    m_global_heap_total += sizeof (float);
    m_global_heap_offsets[ustring("time")] = m_global_heap_total;
    m_global_heap_total += sizeof (float);
    m_global_heap_offsets[ustring("dtime")] = m_global_heap_total;
    m_global_heap_total += sizeof (float);
}



int
ShadingSystemImpl::global_heap_offset (ustring name)
{
    // FIXME -- these are all wrong.  I don't know what I was thinking
    // when I wrote this.  But luckily, it also doesn't seem to be used
    // now.
    ASSERT (0);
    std::map<ustring,int>::const_iterator f = m_global_heap_offsets.find (name);
    return f != m_global_heap_offsets.end() ? f->second : -1;
}



int
ShadingSystemImpl::raytype_bit (ustring name)
{
    for (size_t i = 0, e = m_raytypes.size();  i < e;  ++i)
        if (name == m_raytypes[i])
            return (1 << i);
    return 0;  // not found
}




void ClosureRegistry::register_closure(const char *name, int id, const ClosureParam *params, int size,
                                       PrepareClosureFunc prepare, SetupClosureFunc setup, CompareClosureFunc compare)
{
    if (m_closure_table.size() <= (size_t)id)
        m_closure_table.resize(id + 1);
    ClosureEntry &entry = m_closure_table[id];
    entry.id = id;
    entry.name = name;
    entry.nformal = 0;
    entry.nkeyword = 0;
    for (int i = 0; params && params[i].type != TypeDesc(); ++i) {
        entry.params.push_back(params[i]);
        if (params[i].key == NULL)
            entry.nformal ++;
        else
            entry.nkeyword ++;
    }
    entry.struct_size = size;
    entry.prepare = prepare;
    entry.setup = setup;
    entry.compare = compare;
    m_closure_name_to_id[ustring(name)] = id;
}



const ClosureRegistry::ClosureEntry *ClosureRegistry::get_entry(ustring name)const
{
    std::map<ustring, int>::const_iterator i = m_closure_name_to_id.find(name);
    if (i != m_closure_name_to_id.end())
    {
        ASSERT((size_t)i->second < m_closure_table.size());
        return &m_closure_table[i->second];
    }
    else
        return NULL;
}



}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
