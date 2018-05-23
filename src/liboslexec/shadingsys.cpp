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
#include <fstream>
#include <cstdlib>
#include <mutex>

#include "oslexec_pvt.h"
#include "OSL/genclosure.h"
#include "backendllvm_wide.h"
#include "backendllvm.h"
#include "OSL/oslquery.h"

#include <OpenImageIO/strutil.h>
#include <OpenImageIO/dassert.h>
#include <OpenImageIO/thread.h>
#include <OpenImageIO/timer.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/optparser.h>
#include <OpenImageIO/fmath.h>

using namespace OSL;
using namespace OSL::pvt;

// avoid naming conflict with MSVC macro
#ifdef RGB
#undef RGB
#endif

OSL_NAMESPACE_ENTER



ShadingSystem::ShadingSystem (RendererServices *renderer,
                              TextureSystem *texturesystem,
                              ErrorHandler *err)
    : m_impl (NULL)
{
    if (! err) {
        err = & ErrorHandler::default_handler ();
        ASSERT (err != NULL && "Can't create default ErrorHandler");
    }
    m_impl = new ShadingSystemImpl (renderer, texturesystem, err);
#ifndef NDEBUG
    err->info ("creating new ShadingSystem %p", (void *)this);
#endif
}



ShadingSystem::~ShadingSystem ()
{
    delete m_impl;
}



bool
ShadingSystem::attribute (string_view name, TypeDesc type, const void *val)
{
    return m_impl->attribute (name, type, val);
}



bool
ShadingSystem::attribute (ShaderGroup *group, string_view name,
                          TypeDesc type, const void *val)
{
    return m_impl->attribute (group, name, type, val);
}



bool
ShadingSystem::getattribute (string_view name, TypeDesc type, void *val)
{
    return m_impl->getattribute (name, type, val);
}



bool
ShadingSystem::getattribute (ShaderGroup *group, string_view name,
                             TypeDesc type, void *val)
{
    return m_impl->getattribute (group, name, type, val);
}



bool
ShadingSystem::LoadMemoryCompiledShader (string_view shadername,
                                         string_view buffer)
{
    return m_impl->LoadMemoryCompiledShader (shadername, buffer);
}



ShaderGroupRef
ShadingSystem::ShaderGroupBegin (string_view groupname)
{
    return m_impl->ShaderGroupBegin (groupname);
}



ShaderGroupRef
ShadingSystem::ShaderGroupBegin (string_view groupname, string_view usage,
                                 string_view groupspec)
{
    return m_impl->ShaderGroupBegin (groupname, usage, groupspec);
}



bool
ShadingSystem::ShaderGroupEnd (void)
{
    return m_impl->ShaderGroupEnd();
}



bool
ShadingSystem::Parameter (string_view name, TypeDesc t, const void *val)
{
    return m_impl->Parameter (name, t, val);
}



bool
ShadingSystem::Parameter (string_view name, TypeDesc t, const void *val,
                          bool lockgeom)
{
    return m_impl->Parameter (name, t, val, lockgeom);
}



bool
ShadingSystem::Shader (string_view shaderusage, string_view shadername,
                       string_view layername)
{
    return m_impl->Shader (shaderusage, shadername, layername);
}



bool
ShadingSystem::ConnectShaders (string_view srclayer, string_view srcparam,
                               string_view dstlayer, string_view dstparam)
{
    return m_impl->ConnectShaders (srclayer, srcparam, dstlayer, dstparam);
}



bool
ShadingSystem::ReParameter (ShaderGroup &group, string_view layername,
                            string_view paramname, TypeDesc type,
                            const void *val)
{
    return m_impl->ReParameter (group, layername, paramname, type, val);
}



PerThreadInfo *
ShadingSystem::create_thread_info ()
{
    return m_impl->create_thread_info();
}



void
ShadingSystem::destroy_thread_info (PerThreadInfo *threadinfo)
{
    return m_impl->destroy_thread_info (threadinfo);
}



ShadingContext *
ShadingSystem::get_context (PerThreadInfo *threadinfo,
                            TextureSystem::Perthread *texture_threadinfo)
{
    return m_impl->get_context (threadinfo, texture_threadinfo);
}



void
ShadingSystem::release_context (ShadingContext *ctx)
{
    return m_impl->release_context (ctx);
}



bool
ShadingSystem::execute (ShadingContext *ctx, ShaderGroup &group,
                        ShaderGlobals &globals, bool run)
{
    return m_impl->execute (ctx, group, globals, run);
}



bool
ShadingSystem::execute (ShadingContext &ctx, ShaderGroup &group,
                        ShaderGlobals &globals, bool run)
{
    return m_impl->execute (&ctx, group, globals, run);
}


bool
ShadingSystem::execute_batch (ShadingContext *ctx, ShaderGroup &group,
		ShaderGlobalsBatch &globals_batch, bool run)
{
    return m_impl->execute_batch (ctx, group, globals_batch, run);
}


bool
ShadingSystem::execute_init (ShadingContext &ctx, ShaderGroup &group,
                             ShaderGlobals &globals, bool run)
{
    return ctx.execute_init (group, globals, run);
}

bool
ShadingSystem::execute_batch_init (ShadingContext &ctx, ShaderGroup &group,
		ShaderGlobalsBatch &globals_batch, bool run)
{
    return ctx.execute_batch_init (group, globals_batch, run);
}




bool
ShadingSystem::execute_layer (ShadingContext &ctx, ShaderGlobals &globals,
                              int layernumber)
{
    return ctx.execute_layer (globals, layernumber);
}

bool
ShadingSystem::execute_batch_layer (ShadingContext &ctx, ShaderGlobalsBatch &globals_batch,
                              int layernumber)
{
    return ctx.execute_batch_layer (globals_batch, layernumber);
}


bool
ShadingSystem::execute_layer (ShadingContext &ctx, ShaderGlobals &globals,
                              ustring layername)
{
    int layernumber = find_layer (*ctx.group(), layername);
    return layernumber >= 0 ? ctx.execute_layer (globals, layernumber) : false;
}

bool
ShadingSystem::execute_batch_layer (ShadingContext &ctx, ShaderGlobalsBatch &globals_batch,
                              ustring layername)
{
    int layernumber = find_layer (*ctx.group(), layername);
    return layernumber >= 0 ? ctx.execute_batch_layer (globals_batch, layernumber) : false;
}


bool
ShadingSystem::execute_layer (ShadingContext &ctx, ShaderGlobals &globals,
                              const ShaderSymbol *symbol)
{
    ASSERT (symbol);
    const Symbol *sym = reinterpret_cast<const Symbol *>(symbol);
    int layernumber = sym->layer();
    return layernumber >= 0 ? ctx.execute_layer (globals, layernumber) : false;
}

bool
ShadingSystem::execute_batch_layer (ShadingContext &ctx, ShaderGlobalsBatch &globals_batch,
                              const ShaderSymbol *symbol)
{
    ASSERT (symbol);
    const Symbol *sym = reinterpret_cast<const Symbol *>(symbol);
    int layernumber = sym->layer();
    return layernumber >= 0 ? ctx.execute_batch_layer (globals_batch, layernumber) : false;
}


bool
ShadingSystem::execute_cleanup (ShadingContext &ctx)
{
    return ctx.execute_cleanup ();
}



int
ShadingSystem::find_layer (const ShaderGroup &group, ustring layername) const
{
    return group.find_layer (layername);
}



const void*
ShadingSystem::get_symbol (const ShadingContext &ctx, ustring layername,
                           ustring symbolname, TypeDesc &type) const
{
    const ShaderSymbol *sym = find_symbol (*ctx.group(), layername,
                                           symbolname);
    if (sym) {
        type = symbol_typedesc (sym);
        return symbol_address (ctx, sym);
    }
    return NULL;
}



const void*
ShadingSystem::get_symbol (const ShadingContext &ctx,
                           ustring symbolname, TypeDesc &type) const
{
    ustring layername;
    size_t dot = symbolname.find('.');
    if (dot != ustring::npos) {
        // If the name contains a dot, it's intended to be layer.symbol
        layername = ustring (symbolname, 0, dot);
        symbolname = ustring (symbolname, dot+1);
    }
    return get_symbol (ctx, layername, symbolname, type);
}



const ShaderSymbol*
ShadingSystem::find_symbol (const ShaderGroup &group, ustring layername,
                            ustring symbolname) const
{
    if (! group.optimized())
        return NULL;   // has to be post-optimized
    return (const ShaderSymbol *) group.find_symbol (layername, symbolname);
}



const ShaderSymbol*
ShadingSystem::find_symbol (const ShaderGroup &group, ustring symbolname) const
{
    ustring layername;
    size_t dot = symbolname.find('.');
    if (dot != ustring::npos) {
        // If the name contains a dot, it's intended to be layer.symbol
        layername = ustring (symbolname, 0, dot);
        symbolname = ustring (symbolname, dot+1);
    }
    return find_symbol (group, layername, symbolname);
}



TypeDesc
ShadingSystem::symbol_typedesc (const ShaderSymbol *sym) const
{
    return sym ? ((const Symbol *)sym)->typespec().simpletype() : TypeDesc();
}



const void*
ShadingSystem::symbol_address (const ShadingContext &ctx,
                               const ShaderSymbol *sym) const
{
    return sym ? ctx.symbol_data (*(const Symbol *)sym) : NULL;
}



std::string
ShadingSystem::getstats (int level) const
{
    return m_impl->getstats (level);
}



void
ShadingSystem::register_closure (string_view name, int id,
                                 const ClosureParam *params,
                                 PrepareClosureFunc prepare,
                                 SetupClosureFunc setup,
                                 int alignment)
{
    return m_impl->register_closure (name, id, params, prepare, setup, alignment);
}



bool
ShadingSystem::query_closure (const char **name, int *id,
                              const ClosureParam **params)
{
    return m_impl->query_closure (name, id, params);
}



int
ShadingSystem::raytype_bit (ustring name)
{
    return m_impl->raytype_bit (name);
}



void
ShadingSystem::optimize_all_groups (int nthreads)
{
    return m_impl->optimize_all_groups (nthreads);
}

void
ShadingSystem::jit_all_groups (int nthreads)
{
    return m_impl->jit_all_groups (nthreads);
}

void
ShadingSystem::batched_jit_all_groups (int nthreads)
{
    return m_impl->batched_jit_all_groups (nthreads);
}


TextureSystem *
ShadingSystem::texturesys () const
{
    return m_impl->texturesys();
}



RendererServices *
ShadingSystem::renderer () const
{
    return m_impl->renderer();
}



bool
ShadingSystem::archive_shadergroup (ShaderGroup *group, string_view filename)
{
    return m_impl->archive_shadergroup (group, filename);
}



void
ShadingSystem::optimize_group (ShaderGroup *group)
{
    optimize_group (group, 0, 0);   // No knowledge of the ray flags
}



void
ShadingSystem::optimize_group (ShaderGroup *group,
                               int raytypes_on, int raytypes_off)
{
    ASSERT (group);
    m_impl->optimize_group (*group, raytypes_on, raytypes_off);
}

void
ShadingSystem::jit_group (ShaderGroup *group)
{
    ASSERT (group);
    m_impl->jit_group (*group);	
}

void
ShadingSystem::batched_jit_group (ShaderGroup *group)
{
    ASSERT (group);
    m_impl->batched_jit_group (*group);	
}



static TypeDesc TypeFloatArray2 (TypeDesc::FLOAT, 2);



bool
ShadingSystem::convert_value (void *dst, TypeDesc dsttype,
                              const void *src, TypeDesc srctype)
{
    int tmp_int;
    if (srctype == TypeDesc::UINT8) {
        // uint8 src: Up-convert the source to int
        if (src) {
            tmp_int = *(const unsigned char *)src;
            src = &tmp_int;
        }
        srctype = TypeDesc::TypeInt;
    }

    float tmp_float;
    if (srctype == TypeDesc::TypeInt && dsttype.basetype == TypeDesc::FLOAT) {
        // int -> float-based : up-convert the source to float
        if (src) {
            tmp_float = (float) (*(const int *)src);
            src = &tmp_float;
        }
        srctype = TypeDesc::TypeFloat;
    }

    // Just copy equivalent types
    if (equivalent (dsttype, srctype)) {
        if (dst && src)
            memcpy (dst, src, dsttype.size());
        return true;
    }

    if (srctype == TypeDesc::TypeFloat) {
        // float->triple conversion
        if (equivalent(dsttype, TypeDesc::TypePoint)) {
            if (dst && src) {
                float f = *(const float *)src;
                ((OSL::Vec3 *)dst)->setValue (f, f, f);
            }
            return true;
        }
        // float->int
        if (dsttype == TypeDesc::TypeInt) {
            if (dst && src)
                *(int *)dst = (int) *(const float *)src;
            return true;
        }
        // float->float[2]
        if (dsttype == TypeFloatArray2) {
            if (dst && src) {
                float f = *(const float *)src;
                ((float *)dst)[0] = f;
                ((float *)dst)[1] = f;
            }
            return true;
        }
        return false; // Unsupported conversion
    }

    // float[2] -> triple
    if (srctype == TypeFloatArray2 && equivalent(dsttype, TypeDesc::TypePoint)) {
        if (dst && src) {
            float f0 = ((const float *)src)[0];
            float f1 = ((const float *)src)[1];
            ((OSL::Vec3 *)dst)->setValue (f0, f1, 0.0f);
        }
        return true;
    }

    return false;   // Unsupported conversion
}



PerThreadInfo::PerThreadInfo ()
{
}



PerThreadInfo::~PerThreadInfo ()
{
    while (! context_pool.empty())
        delete pop_context ();
}



ShadingContext *
PerThreadInfo::pop_context ()
{
    ShadingContext *sc = context_pool.top ();
    context_pool.pop ();
    return sc;
}





namespace Strings {

// Define static ustring symbols for very fast comparison
ustring camera ("camera"), common ("common");
ustring object ("object"), shader ("shader");
ustring screen ("screen"), NDC ("NDC");
ustring rgb ("rgb"), RGB ("RGB");
ustring hsv ("hsv"), hsl ("hsl"), YIQ ("YIQ");
ustring XYZ ("XYZ"), xyz ("xyz"), xyY("xyY");
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
ustring errormessage ("errormessage");
ustring interp("interp"), closest("closest"), linear("linear");
ustring cubic("cubic"), smartcubic("smartcubic");
ustring perlin("perlin"), uperlin("uperlin");
ustring noise("noise"), snoise("snoise");
ustring cell("cell"), cellnoise("cellnoise"), pcellnoise("pcellnoise");
ustring pnoise("pnoise"), psnoise("psnoise");
ustring genericnoise("genericnoise"), genericpnoise("genericpnoise");
ustring gabor("gabor"), gabornoise("gabornoise"), gaborpnoise("gaborpnoise");
ustring simplex("simplex"), usimplex("usimplex");
ustring simplexnoise("simplexnoise"), usimplexnoise("usimplexnoise");
ustring anisotropic("anisotropic"), direction("direction");
ustring do_filter("do_filter"), bandwidth("bandwidth"), impulses("impulses");
ustring op_dowhile("dowhile"), op_for("for"), op_while("while");
ustring op_exit("exit");
ustring subimage("subimage"), subimagename("subimagename");
ustring missingcolor("missingcolor"), missingalpha("missingalpha");
ustring end("end"), useparam("useparam");
ustring uninitialized_string("!!!uninitialized!!!");
ustring unull("unull");
ustring raytype("raytype");
ustring color("color"), point("point"), vector("vector"), normal("normal");
ustring matrix("matrix");
ustring unknown ("unknown");
ustring _emptystring_ ("");
}; // namespace Strings



namespace pvt {   // OSL::pvt


ShadingSystemImpl::ShadingSystemImpl (RendererServices *renderer,
                                      TextureSystem *texturesystem,
                                      ErrorHandler *err)
    : m_renderer(renderer), m_texturesys(texturesystem), m_err(err),
      m_statslevel (0), m_lazylayers (true),
      m_lazyglobals (true), m_lazyunconnected(true),
      m_lazy_userdata(false), m_userdata_isconnected(false),
      m_clearmemory (false), m_debugnan (false), m_debug_uninit(false),
      m_lockgeom_default (true), m_strict_messages(true),
      m_range_checking(true),
      m_unknown_coordsys_error(true), m_connection_error(true),
      m_greedyjit(false), m_countlayerexecs(false),
      m_max_warnings_per_thread(100),
      m_profile(0),
      m_optimize(2),
      m_opt_simplify_param(true), m_opt_constant_fold(true),
      m_opt_stale_assign(true), m_opt_elide_useless_ops(true),
      m_opt_elide_unconnected_outputs(true),
      m_opt_peephole(true), m_opt_coalesce_temps(true),
      m_opt_assign(true), m_opt_mix(true),
      m_opt_merge_instances(1), m_opt_merge_instances_with_userdata(true),
      m_opt_fold_getattribute(true),
      m_opt_middleman(true), m_opt_texture_handle(true),
      m_opt_seed_bblock_aliases(true),
      m_optimize_nondebug(false),
      m_opt_passes(10),
      m_llvm_optimize(0),
      m_debug(0), m_llvm_debug(0),
      m_llvm_debug_layers(0), m_llvm_debug_ops(0),
      m_llvm_debugging_symbols(0),
      m_llvm_profiling_events(0),
      m_commonspace_synonym("world"),
      m_colorspace("Rec709"),
      m_max_local_mem_KB(2048),
      m_compile_report(false),
      m_buffer_printf(true),
      m_no_noise(false),
      m_no_pointcloud(false),
      m_force_derivs(false),
      m_exec_repeat(1),
      m_in_group (false),
      m_stat_opt_locking_time(0), m_stat_specialization_time(0),
      m_stat_total_llvm_time(0),
      m_stat_llvm_setup_time(0), m_stat_llvm_irgen_time(0),
      m_stat_llvm_opt_time(0), m_stat_llvm_jit_time(0),
      m_stat_inst_merge_time(0),
      m_stat_max_llvm_local_mem(0)
{
    m_stat_shaders_loaded = 0;
    m_stat_shaders_requested = 0;
    m_stat_groups = 0;
    m_stat_groupinstances = 0;
    m_stat_instances_compiled = 0;
    m_stat_groups_compiled = 0;
    m_stat_empty_instances = 0;
    m_stat_merged_inst = 0;
    m_stat_merged_inst_opt = 0;
    m_stat_empty_groups = 0;
    m_stat_regexes = 0;
    m_stat_preopt_syms = 0;
    m_stat_postopt_syms = 0;
    m_stat_syms_with_derivs = 0;
    m_stat_preopt_ops = 0;
    m_stat_postopt_ops = 0;
    m_stat_middlemen_eliminated = 0;
    m_stat_const_connections = 0;
    m_stat_global_connections = 0;
    m_stat_tex_calls_codegened = 0;
    m_stat_tex_calls_as_handles = 0;
    m_stat_master_load_time = 0;
    m_stat_optimization_time = 0;
    m_stat_getattribute_time = 0;
    m_stat_getattribute_fail_time = 0;
    m_stat_getattribute_calls = 0;
    m_stat_get_userdata_calls = 0;
    m_stat_noise_calls = 0;
    m_stat_pointcloud_searches = 0;
    m_stat_pointcloud_searches_total_results = 0;
    m_stat_pointcloud_max_results = 0;
    m_stat_pointcloud_failures = 0;
    m_stat_pointcloud_gets = 0;
    m_stat_pointcloud_writes = 0;
    m_stat_layers_executed = 0;
    m_stat_total_shading_time_ticks = 0;

    m_groups_to_compile_count = 0;
    m_threads_currently_compiling = 0;

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

    // If client didn't supply a texture system, use the one already held
    // by the renderer (if it returns one).
    if (! m_texturesys)
        m_texturesys = renderer->texturesys();

    // If we still don't have a texture system, create a new one
    if (! m_texturesys) {
#if OSL_NO_DEFAULT_TEXTURESYSTEM
        // This build option instructs OSL to never create a TextureSystem
        // itself. (Most likely reason: this build of OSL is for a renderer
        // that replaces OIIO's TextureSystem with its own, and therefore
        // wouldn't want to accidentally make an OIIO one here.
        ASSERT (0 && "ShadingSystem was not passed a working TextureSystem*");
#else
        m_texturesys = TextureSystem::create (true /* shared */);
        ASSERT (m_texturesys);
        // Make some good guesses about default options
        m_texturesys->attribute ("automip",  1);
        m_texturesys->attribute ("autotile", 64);
#endif
    }

    // Alternate way of turning on LLVM debug mode (temporary/experimental)
    const char *llvm_debug_env = getenv ("OSL_LLVM_DEBUG");
    if (llvm_debug_env && *llvm_debug_env)
        m_llvm_debug = atoi(llvm_debug_env);

    // Alternate way of generating LLVM debugging symbols (temporary/experimental)
    const char *llvm_debugging_symbols_env = getenv ("OSL_LLVM_DEBUGGING_SYMBOLS");
    if (llvm_debugging_symbols_env && *llvm_debugging_symbols_env)
        m_llvm_debugging_symbols = atoi(llvm_debugging_symbols_env);

    // Alternate way of generating LLVM profiling events (temporary/experimental)
    const char *llvm_profiling_events_env = getenv ("OSL_LLVM_PROFILING_EVENTS");
    if (llvm_profiling_events_env && *llvm_profiling_events_env)
        m_llvm_profiling_events = atoi(llvm_profiling_events_env);

    // Initialize a default set of raytype names.  A particular renderer
    // can override this, add custom names, or change the bits around,
    // if this default ordering is not to its liking.
    static const char *raytypes[] = {
        /*1*/ "camera", /*2*/ "shadow", /*4*/ "reflection", /*8*/ "refraction",
        /*16*/ "diffuse", /*32*/ "glossy", /*64*/ "subsurface",
        /*128*/ "displacement"
    };
    const int nraytypes = sizeof(raytypes)/sizeof(raytypes[0]);
    attribute ("raytypes", TypeDesc(TypeDesc::STRING,nraytypes), raytypes);

    attribute ("colorspace", TypeDesc::STRING, &m_colorspace);

    // Allow environment variable to override default options
    const char *options = getenv ("OSL_OPTIONS");
    if (options)
        attribute ("options", TypeDesc::STRING, &options);

    setup_op_descriptors ();
}



static void
shading_system_setup_op_descriptors (ShadingSystemImpl::OpDescriptorMap& op_descriptor)
{
#define OP2(alias,name,ll,fold,simp,flag)                                \
    extern bool llvm_gen_##ll (BackendLLVMWide &rop, int opnum);             \
    extern bool llvm_gen_##ll (BackendLLVM &rop, int opnum);             \
    extern int  constfold_##fold (RuntimeOptimizer &rop, int opnum);     \
    op_descriptor[ustring(#alias)] = OpDescriptor(#name, llvm_gen_##ll, llvm_gen_##ll,  \
                                                  constfold_##fold, simp, flag);
#define OP(name,ll,fold,simp,flag) OP2(name,name,ll,fold,simp,flag)
#define TEX OpDescriptor::Tex
#define LLVM_INLINED OpDescriptor::LLVMInlined

    // name          llvmgen              folder         simple     flags
    OP (aassign,     aassign,             aassign,       false,     0);
    OP (abs,         generic,             abs,           true,      0 /*LLVM_INLINED*/);
    OP (acos,        generic,             acos,          true,      0);
    OP (add,         add,                 add,           true,      0);
    OP (and,         andor,               and,           true,      0);
    OP (area,        area,                deriv,         true,      0);
    OP (aref,        aref,                aref,          true,      0);
    OP (arraycopy,   arraycopy,           none,          false,     0);
    OP (arraylength, arraylength,         arraylength,   true,      0);
    OP (asin,        generic,             asin,          true,      0);
    OP (assign,      assign,              none,          true,      0);
    OP (atan,        generic,             none,          true,      0);
    OP (atan2,       generic,             none,          true,      0);
    OP (backfacing,  get_simple_SG_field, none,          true,      0);
    OP (bitand,      bitwise_binary_op,   bitand,        true,      0);
    OP (bitor,       bitwise_binary_op,   bitor,         true,      0);
    OP (blackbody,   blackbody,           none,          true,      0);
    OP (break,       loopmod_op,          none,          false,     0);
    OP (calculatenormal, calculatenormal, none,          true,      0);
    OP (ceil,        generic,             ceil,          true,      0);
    OP (cellnoise,   noise,               noise,         true,      0);
    OP (clamp,       clamp,               clamp,         true,      0);
    OP (closure,     closure,             none,          true,      0);
    OP (color,       construct_color,     triple,        true,      0);
    OP (compassign,  compassign,          compassign,    false,     0);
    OP (compl,       unary_op,            compl,         true,      0);
    OP (compref,     compref,             compref,       true,      0);
    OP (concat,      generic,             concat,        true,      0);
    OP (continue,    loopmod_op,          none,          false,     0);
    OP (cos,         generic,             cos,           true,      0);
    OP (cosh,        generic,             none,          true,      0);
    OP (cross,       generic,             none,          true,      0);
    OP (degrees,     generic,             degrees,       true,      0);
    OP (determinant, generic,             none,          true,      0);
    OP (dict_find,   dict_find,           none,          false,     0);
    OP (dict_next,   dict_next,           none,          false,     0);
    OP (dict_value,  dict_value,          none,          false,     0);
    OP (distance,    generic,             none,          true,      0);
    OP (div,         div,                 div,           true,      0);
    OP (dot,         generic,             dot,           true,      0);
    OP (Dx,          DxDy,                deriv,         true,      0);
    OP (Dy,          DxDy,                deriv,         true,      0);
    OP (Dz,          Dz,                  deriv,         true,      0);
    OP (dowhile,     loop_op,             none,          false,     0);
    OP (end,         end,                 none,          false,     0);
    OP (endswith,    generic,             endswith,      true,      0);
    OP (environment, environment,         none,          true,      TEX);
    OP (eq,          compare_op,          eq,            true,      0);
    OP (erf,         generic,             erf,           true,      0);
    OP (erfc,        generic,             erfc,          true,      0);
    OP (error,       printf,              none,          false,     0);
    OP (exit,        return,              none,          false,     0);
    OP (exp,         generic,             exp,           true,      0);
    OP (exp2,        generic,             exp2,          true,      0);
    OP (expm1,       generic,             expm1,         true,      0);
    OP (fabs,        generic,             abs,           true,      0);
    OP (filterwidth, filterwidth,         deriv,         true,      0);
    OP (floor,       generic,             floor,         true,      0  /*LLVM_INLINED*/);
    OP (fmod,        modulus,             none,          true,      0);
    OP (for,         loop_op,             none,          false,     0);
    OP (format,      printf,              format,        true,      0);
    OP (functioncall, functioncall,       functioncall,  false,     0);
    OP (functioncall_nr,functioncall_nr,  none,          false,     0);
    OP (ge,          compare_op,          ge,            true,      0);
    OP (getattribute, getattribute,       getattribute,  false,     0);
    OP (getchar,      generic,            getchar,       true,      0);
    OP (getmatrix,   getmatrix,           getmatrix,     false,     0);
    OP (getmessage,  getmessage,          getmessage,    false,     0);
    OP (gettextureinfo, gettextureinfo,   gettextureinfo,false,     TEX);
    OP (gt,          compare_op,          gt,            true,      0);
    OP (hash,        generic,             hash,          true,      0);
    OP (if,          if,                  if,            false,     0);
    OP (inversesqrt, generic,             inversesqrt,   true,      0);
    OP (isconnected, generic,             none,          true,      0);
    OP (isconstant,  isconstant,          isconstant,    true,      0);
    OP (isfinite,    generic,             none,          true,      0);
    OP (isinf,       generic,             none,          true,      0);
    OP (isnan,       generic,             none,          true,      0);
    OP (le,          compare_op,          le,            true,      0);
    OP (length,      generic,             none,          true,      0);
    OP (log,         generic,             log,           true,      0);
    OP (log10,       generic,             log10,         true,      0);
    OP (log2,        generic,             log2,          true,      0);
    OP (logb,        generic,             logb,          true,      0);
    OP (lt,          compare_op,          lt,            true,      0);
    OP (luminance,   luminance,           none,          true,      0);
    OP (matrix,      matrix,              matrix,        true,      0);
    OP (max,         minmax,              max,           true,      0);
    OP (mxcompassign, mxcompassign,       mxcompassign,  false,     0);
    OP (mxcompref,   mxcompref,           none,          true,      0);
    OP (min,         minmax,              min,           true,      0);
    OP (mix,         mix,                 mix,           true,      0);
    OP (mod,         modulus,             none,          true,      0);
    OP (mul,         mul,                 mul,           true,      0);
    OP (neg,         neg,                 neg,           true,      0);
    OP (neq,         compare_op,          neq,           true,      0);
    OP (noise,       noise,               noise,         true,      0);
    OP (nop,         nop,                 none,          true,      0);
    OP (normal,      construct_triple,    triple,        true,      0);
    OP (normalize,   generic,             normalize,     true,      0);
    OP (or,          andor,               or,            true,      0);
    OP (pnoise,      noise,               noise,         true,      0);
    OP (point,       construct_triple,    triple,        true,      0);
    OP (pointcloud_search, pointcloud_search, pointcloud_search,
                                                         false,     TEX);
    OP (pointcloud_get, pointcloud_get,   pointcloud_get,false,     TEX);
    OP (pointcloud_write, pointcloud_write, none,        false,     0);
    OP (pow,         generic,             pow,           true,      0);
    OP (printf,      printf,              none,          false,     0);
    OP (psnoise,     noise,               noise,         true,      0);
    OP (radians,     generic,             radians,       true,      0);
    OP (raytype,     raytype,             raytype,       true,      0);
    OP (regex_match, regex,               none,          false,     0);
    OP (regex_search, regex,              regex_search,  false,     0);
    OP (return,      return,              none,          false,     0);
    OP (round,       generic,             none,          true,      0);
    OP (select,      select,              select,        true,      0);
    OP (setmessage,  setmessage,          setmessage,    false,     0);
    OP (shl,         bitwise_binary_op,   none,          true,      0);
    OP (shr,         bitwise_binary_op,   none,          true,      0);
    OP (sign,        generic,             none,          true,      0);
    OP (sin,         generic,             sin,           true,      0);
    OP (sincos,      sincos,              sincos,        false,     0);
    OP (sinh,        generic,             none,          true,      0);
    OP (smoothstep,  generic,             none,          true,      0 /*LLVM_INLINED*/);
    OP (snoise,      noise,               noise,         true,      0);
    OP (spline,      spline,              none,          true,      0);
    OP (splineinverse, spline,            none,          true,      0);
    OP (split,       split,               split,         false,     0);
    OP (sqrt,        generic,             sqrt,          true,      0 /*LLVM_INLINED*/);
    OP (startswith,  generic,             none,          true,      0);
    OP (step,        generic,             none,          true,      0);
    OP (stof,        generic,             stof,          true,      0);
    OP (stoi,        generic,             stoi,          true,      0);
    OP (strlen,      generic,             strlen,        true,      0);
    OP2(strtof,stof, generic,             stof,          true,      0);
    OP2(strtoi,stoi, generic,             stoi,          true,      0);
    OP (sub,         sub,                 sub,           true,      0);
    OP (substr,      generic,             substr,        true,      0);
    OP (surfacearea, get_simple_SG_field, none,          true,      0);
    OP (tan,         generic,             none,          true,      0);
    OP (tanh,        generic,             none,          true,      0);
    OP (texture,     texture,             texture,       true,      TEX);
    OP (texture3d,   texture3d,           none,          true,      TEX);
    OP (trace,       trace,               none,          false,     0);
    OP (transform,   transform,           transform,     true,      0);
    OP (transformn,  transform,           transform,     true,      0);
    OP (transformv,  transform,           transform,     true,      0);
    OP (transpose,   generic,             none,          true,      0);
    OP (trunc,       generic,             none,          true,      0);
    OP (useparam,    useparam,            useparam,      false,     0);
    OP (vector,      construct_triple,    triple,        true,      0);
    OP (warning,     printf,              warning,       false,     0);
    OP (wavelength_color, blackbody,      none,          true,      0);
    OP (while,       loop_op,             none,          false,     0);
    OP (xor,         bitwise_binary_op,   xor,           true,      0);
#undef OP
#undef TEX
}



void
ShadingSystemImpl::setup_op_descriptors ()
{
    // This is not a class member function to avoid namespace issues
    // with function declarations in the function body, when building
    // with visual studio.
    shading_system_setup_op_descriptors(m_op_descriptor);
}



void
ShadingSystemImpl::register_closure (string_view name, int id,
                                     const ClosureParam *params,
                                     PrepareClosureFunc prepare,
                                     SetupClosureFunc setup,
                                     int alignment)
{
    for (int i = 0; params && params[i].type != TypeDesc(); ++i) {
        if (params[i].key == NULL && params[i].type.size() != (size_t)params[i].field_size) {
            error ("Parameter %d of '%s' closure is assigned to a field of incompatible size", i + 1, name);
            return;
        }
    }
    m_closure_registry.register_closure(name, id, params, prepare, setup, alignment);
}



bool
ShadingSystemImpl::query_closure(const char **name, int *id,
                                 const ClosureParam **params)
{
    ASSERT(name || id);
    const ClosureRegistry::ClosureEntry *entry =
        (name && *name) ? m_closure_registry.get_entry(ustring(*name))
                        : m_closure_registry.get_entry(*id);
    if (!entry)
        return false;

    if (name)
        *name   = entry->name.c_str();
    if (id)
        *id     = entry->id;
    if (params)
        *params = &entry->params[0];

    return true;
}



ShadingSystemImpl::~ShadingSystemImpl ()
{
	// ignoring multithreaded locking, 
	// why would we be destroying this if there are other
	// threads pointing to it
    size_t ngroups = m_all_shader_groups.size();
    for (size_t i = 0;  i < ngroups;  ++i) {
    	if (ShaderGroupRef g = m_all_shader_groups[i].lock()) {
			if (!g->jitted() || !g->batch_jitted()) {
				// As we are now lazier in jitting and need to keep the OSL IR
				// around in case we want to create a batched JIT or vice versa
				// we may have OSL IR to cleanup
				group_post_jit_cleanup(*g);
			}
    	}
    }
	
    printstats ();
    // N.B. just let m_texsys go -- if we asked for one to be created,
    // we asked for a shared one.

    // FIXME(boulos): According to the docs, we should also call
    // llvm_shutdown once we're done. However, ~ShadingSystemImpl
    // seems like the wrong place for this since in a multi-threaded
    // implementation we might destroy this impl while having others
    // outstanding. I'll leave this as a fixme for now.

    //llvm::llvm_shutdown();
}



bool
ShadingSystemImpl::attribute (string_view name, TypeDesc type,
                              const void *val)
{
#define ATTR_SET(_name,_ctype,_dst)                                     \
    if (name == _name && type == OIIO::BaseTypeFromC<_ctype>::value) {  \
        _dst = *(_ctype *)(val);                                        \
        return true;                                                    \
    }
#define ATTR_SET_STRING(_name,_dst)                                     \
    if (name == _name && type == TypeDesc::STRING) {                    \
        _dst = ustring (*(const char **)val);                           \
        return true;                                                    \
    }

    if (name == "options" && type == TypeDesc::STRING) {
        return OIIO::optparser (*this, *(const char **)val);
    }

    lock_guard guard (m_mutex);  // Thread safety
    ATTR_SET ("statistics:level", int, m_statslevel);
    ATTR_SET ("debug", int, m_debug);
    ATTR_SET ("lazylayers", int, m_lazylayers);
    ATTR_SET ("lazyglobals", int, m_lazyglobals);
    ATTR_SET ("lazyunconnected", int, m_lazyunconnected);
    ATTR_SET ("lazy_userdata", int, m_lazy_userdata);
    ATTR_SET ("userdata_isconnected", int, m_userdata_isconnected);
    ATTR_SET ("clearmemory", int, m_clearmemory);
    ATTR_SET ("debug_nan", int, m_debugnan);
    ATTR_SET ("debugnan", int, m_debugnan);  // back-compatible alias
    ATTR_SET ("debug_uninit", int, m_debug_uninit);
    ATTR_SET ("lockgeom", int, m_lockgeom_default);
    ATTR_SET ("profile", int, m_profile);
    ATTR_SET ("optimize", int, m_optimize);
    ATTR_SET ("opt_simplify_param", int, m_opt_simplify_param);
    ATTR_SET ("opt_constant_fold", int, m_opt_constant_fold);
    ATTR_SET ("opt_stale_assign", int, m_opt_stale_assign);
    ATTR_SET ("opt_elide_useless_ops", int, m_opt_elide_useless_ops);
    ATTR_SET ("opt_elide_unconnected_outputs", int, m_opt_elide_unconnected_outputs);
    ATTR_SET ("opt_peephole", int, m_opt_peephole);
    ATTR_SET ("opt_coalesce_temps", int, m_opt_coalesce_temps);
    ATTR_SET ("opt_assign", int, m_opt_assign);
    ATTR_SET ("opt_mix", int, m_opt_mix);
    ATTR_SET ("opt_merge_instances", int, m_opt_merge_instances);
    ATTR_SET ("opt_merge_instances_with_userdata", int, m_opt_merge_instances_with_userdata);
    ATTR_SET ("opt_fold_getattribute", int, m_opt_fold_getattribute);
    ATTR_SET ("opt_middleman", int, m_opt_middleman);
    ATTR_SET ("opt_texture_handle", int, m_opt_texture_handle);
    ATTR_SET ("opt_seed_bblock_aliases", int, m_opt_seed_bblock_aliases);
    ATTR_SET ("opt_passes", int, m_opt_passes);
    ATTR_SET ("optimize_nondebug", int, m_optimize_nondebug);
    ATTR_SET ("llvm_optimize", int, m_llvm_optimize);
    ATTR_SET ("llvm_debug", int, m_llvm_debug);
    ATTR_SET ("llvm_debug_layers", int, m_llvm_debug_layers);
    ATTR_SET ("llvm_debug_ops", int, m_llvm_debug_ops);
    ATTR_SET ("llvm_debugging_symbols", int, m_llvm_debugging_symbols);
    ATTR_SET ("llvm_profiling_events", int, m_llvm_profiling_events);
    ATTR_SET ("strict_messages", int, m_strict_messages);
    ATTR_SET ("range_checking", int, m_range_checking);
    ATTR_SET ("unknown_coordsys_error", int, m_unknown_coordsys_error);
    ATTR_SET ("connection_error", int, m_connection_error);
    ATTR_SET ("greedyjit", int, m_greedyjit);
    ATTR_SET ("countlayerexecs", int, m_countlayerexecs);
    ATTR_SET ("max_warnings_per_thread", int, m_max_warnings_per_thread);
    ATTR_SET ("max_local_mem_KB", int, m_max_local_mem_KB);
    ATTR_SET ("compile_report", int, m_compile_report);
    ATTR_SET ("buffer_printf", int, m_buffer_printf);
    ATTR_SET ("no_noise", int, m_no_noise);
    ATTR_SET ("no_pointcloud", int, m_no_pointcloud);
    ATTR_SET ("force_derivs", int, m_force_derivs);
    ATTR_SET ("exec_repeat", int, m_exec_repeat);
    ATTR_SET_STRING ("commonspace", m_commonspace_synonym);
    ATTR_SET_STRING ("debug_groupname", m_debug_groupname);
    ATTR_SET_STRING ("debug_layername", m_debug_layername);
    ATTR_SET_STRING ("opt_layername", m_opt_layername);
    ATTR_SET_STRING ("only_groupname", m_only_groupname);
    ATTR_SET_STRING ("archive_groupname", m_archive_groupname);
    ATTR_SET_STRING ("archive_filename", m_archive_filename);

    // cases for special handling
    if (name == "searchpath:shader" && type == TypeDesc::STRING) {
        m_searchpath = std::string (*(const char **)val);
        OIIO::Filesystem::searchpath_split (m_searchpath, m_searchpath_dirs);
        return true;
    }
    if (name == "colorspace" && type == TypeDesc::STRING) {
        ustring c = ustring (*(const char **)val);
        if (set_colorspace (m_colorspace))
            m_colorspace = c;
        else
            error ("Unknown color space \"%s\"", c.c_str());
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
    if (name == "renderer_outputs" && type.basetype == TypeDesc::STRING) {
        m_renderer_outputs.clear ();
        for (size_t i = 0;  i < type.numelements();  ++i)
            m_renderer_outputs.push_back (ustring(((const char **)val)[i]));
        return true;
    }
    return false;
#undef ATTR_SET
#undef ATTR_SET_STRING
}



bool
ShadingSystemImpl::getattribute (string_view name, TypeDesc type,
                                 void *val)
{
#define ATTR_DECODE(_name,_ctype,_src)                                  \
    if (name == _name && type == OIIO::BaseTypeFromC<_ctype>::value) {  \
        *(_ctype *)(val) = (_ctype)(_src);                              \
        return true;                                                    \
    }
#define ATTR_DECODE_STRING(_name,_src)                                  \
    if (name == _name && type == TypeDesc::STRING) {                    \
        *(const char **)(val) = _src.c_str();                           \
        return true;                                                    \
    }

    lock_guard guard (m_mutex);  // Thread safety
    ATTR_DECODE_STRING ("searchpath:shader", m_searchpath);
    ATTR_DECODE ("statistics:level", int, m_statslevel);
    ATTR_DECODE ("lazylayers", int, m_lazylayers);
    ATTR_DECODE ("lazyglobals", int, m_lazyglobals);
    ATTR_DECODE ("lazyunconnected", int, m_lazyunconnected);
    ATTR_DECODE ("lazy_userdata", int, m_lazy_userdata);
    ATTR_DECODE ("userdata_isconnected", int, m_userdata_isconnected);
    ATTR_DECODE ("clearmemory", int, m_clearmemory);
    ATTR_DECODE ("debug_nan", int, m_debugnan);
    ATTR_DECODE ("debugnan", int, m_debugnan);  // back-compatible alias
    ATTR_DECODE ("debug_uninit", int, m_debug_uninit);
    ATTR_DECODE ("lockgeom", int, m_lockgeom_default);
    ATTR_DECODE ("profile", int, m_profile);
    ATTR_DECODE ("optimize", int, m_optimize);
    ATTR_DECODE ("opt_simplify_param", int, m_opt_simplify_param);
    ATTR_DECODE ("opt_constant_fold", int, m_opt_constant_fold);
    ATTR_DECODE ("opt_stale_assign", int, m_opt_stale_assign);
    ATTR_DECODE ("opt_elide_useless_ops", int, m_opt_elide_useless_ops);
    ATTR_DECODE ("opt_elide_unconnected_outputs", int, m_opt_elide_unconnected_outputs);
    ATTR_DECODE ("opt_peephole", int, m_opt_peephole);
    ATTR_DECODE ("opt_coalesce_temps", int, m_opt_coalesce_temps);
    ATTR_DECODE ("opt_assign", int, m_opt_assign);
    ATTR_DECODE ("opt_mix", int, m_opt_mix);
    ATTR_DECODE ("opt_merge_instances", int, m_opt_merge_instances);
    ATTR_DECODE ("opt_merge_instances_with_userdata", int, m_opt_merge_instances_with_userdata);
    ATTR_DECODE ("opt_fold_getattribute", int, m_opt_fold_getattribute);
    ATTR_DECODE ("opt_middleman", int, m_opt_middleman);
    ATTR_DECODE ("opt_texture_handle", int, m_opt_texture_handle);
    ATTR_DECODE ("opt_seed_bblock_aliases", int, m_opt_seed_bblock_aliases);
    ATTR_DECODE ("opt_passes", int, m_opt_passes);
    ATTR_DECODE ("optimize_nondebug", int, m_optimize_nondebug);
    ATTR_DECODE ("llvm_optimize", int, m_llvm_optimize);
    ATTR_DECODE ("debug", int, m_debug);
    ATTR_DECODE ("llvm_debug", int, m_llvm_debug);
    ATTR_DECODE ("llvm_debug_layers", int, m_llvm_debug_layers);
    ATTR_DECODE ("llvm_debug_ops", int, m_llvm_debug_ops);
    ATTR_DECODE ("llvm_debugging_symbols", int, m_llvm_debugging_symbols);
    ATTR_DECODE ("llvm_profiling_events", int, m_llvm_profiling_events);
    ATTR_DECODE ("strict_messages", int, m_strict_messages);
    ATTR_DECODE ("range_checking", int, m_range_checking);
    ATTR_DECODE ("unknown_coordsys_error", int, m_unknown_coordsys_error);
    ATTR_DECODE ("connection_error", int, m_connection_error);
    ATTR_DECODE ("greedyjit", int, m_greedyjit);
    ATTR_DECODE ("countlayerexecs", int, m_countlayerexecs);
    ATTR_DECODE ("max_warnings_per_thread", int, m_max_warnings_per_thread);
    ATTR_DECODE_STRING ("commonspace", m_commonspace_synonym);
    ATTR_DECODE_STRING ("colorspace", m_colorspace);
    ATTR_DECODE_STRING ("debug_groupname", m_debug_groupname);
    ATTR_DECODE_STRING ("debug_layername", m_debug_layername);
    ATTR_DECODE_STRING ("opt_layername", m_opt_layername);
    ATTR_DECODE_STRING ("only_groupname", m_only_groupname);
    ATTR_DECODE_STRING ("archive_groupname", m_archive_groupname);
    ATTR_DECODE_STRING ("archive_filename", m_archive_filename);
    ATTR_DECODE ("max_local_mem_KB", int, m_max_local_mem_KB);
    ATTR_DECODE ("compile_report", int, m_compile_report);
    ATTR_DECODE ("buffer_printf", int, m_buffer_printf);
    ATTR_DECODE ("no_noise", int, m_no_noise);
    ATTR_DECODE ("no_pointcloud", int, m_no_pointcloud);
    ATTR_DECODE ("force_derivs", int, m_force_derivs);
    ATTR_DECODE ("exec_repeat", int, m_exec_repeat);

    ATTR_DECODE ("stat:masters", int, m_stat_shaders_loaded);
    ATTR_DECODE ("stat:groups", int, m_stat_groups);
    ATTR_DECODE ("stat:instances_compiled", int, m_stat_instances_compiled);
    ATTR_DECODE ("stat:groups_compiled", int, m_stat_groups_compiled);
    ATTR_DECODE ("stat:empty_instances", int, m_stat_empty_instances);
    ATTR_DECODE ("stat:merged_inst", int, m_stat_merged_inst);
    ATTR_DECODE ("stat:merged_inst_opt", int, m_stat_merged_inst_opt);
    ATTR_DECODE ("stat:empty_groups", int, m_stat_empty_groups);
    ATTR_DECODE ("stat:instances", int, m_stat_groupinstances);
    ATTR_DECODE ("stat:regexes", int, m_stat_regexes);
    ATTR_DECODE ("stat:preopt_syms", int, m_stat_preopt_syms);
    ATTR_DECODE ("stat:postopt_syms", int, m_stat_postopt_syms);
    ATTR_DECODE ("stat:syms_with_derivs", int, m_stat_syms_with_derivs);
    ATTR_DECODE ("stat:preopt_ops", int, m_stat_preopt_ops);
    ATTR_DECODE ("stat:postopt_ops", int, m_stat_postopt_ops);
    ATTR_DECODE ("stat:middlemen_eliminated", int, m_stat_middlemen_eliminated);
    ATTR_DECODE ("stat:const_connections", int, m_stat_const_connections);
    ATTR_DECODE ("stat:global_connections", int, m_stat_global_connections);
    ATTR_DECODE ("stat:tex_calls_codegened", int, m_stat_tex_calls_codegened);
    ATTR_DECODE ("stat:tex_calls_as_handles", int, m_stat_tex_calls_as_handles);
    ATTR_DECODE ("stat:master_load_time", float, m_stat_master_load_time);
    ATTR_DECODE ("stat:optimization_time", float, m_stat_optimization_time);
    ATTR_DECODE ("stat:opt_locking_time", float, m_stat_opt_locking_time);
    ATTR_DECODE ("stat:specialization_time", float, m_stat_specialization_time);
    ATTR_DECODE ("stat:total_llvm_time", float, m_stat_total_llvm_time);
    ATTR_DECODE ("stat:llvm_setup_time", float, m_stat_llvm_setup_time);
    ATTR_DECODE ("stat:llvm_irgen_time", float, m_stat_llvm_irgen_time);
    ATTR_DECODE ("stat:llvm_opt_time", float, m_stat_llvm_opt_time);
    ATTR_DECODE ("stat:llvm_jit_time", float, m_stat_llvm_jit_time);
    ATTR_DECODE ("stat:inst_merge_time", float, m_stat_inst_merge_time);
    ATTR_DECODE ("stat:getattribute_calls", long long, m_stat_getattribute_calls);
    ATTR_DECODE ("stat:get_userdata_calls", long long, m_stat_get_userdata_calls);
    ATTR_DECODE ("stat:noise_calls", long long, m_stat_noise_calls);
    ATTR_DECODE ("stat:pointcloud_searches", long long, m_stat_pointcloud_searches);
    ATTR_DECODE ("stat:pointcloud_gets", long long, m_stat_pointcloud_gets);
    ATTR_DECODE ("stat:pointcloud_writes", long long, m_stat_pointcloud_writes);
    ATTR_DECODE ("stat:pointcloud_searches_total_results", long long, m_stat_pointcloud_searches_total_results);
    ATTR_DECODE ("stat:pointcloud_max_results", int, m_stat_pointcloud_max_results);
    ATTR_DECODE ("stat:pointcloud_failures", int, m_stat_pointcloud_failures);
    ATTR_DECODE ("stat:memory_current", long long, m_stat_memory.current());
    ATTR_DECODE ("stat:memory_peak", long long, m_stat_memory.peak());
    ATTR_DECODE ("stat:mem_master_current", long long, m_stat_mem_master.current());
    ATTR_DECODE ("stat:mem_master_peak", long long, m_stat_mem_master.peak());
    ATTR_DECODE ("stat:mem_master_ops_current", long long, m_stat_mem_master_ops.current());
    ATTR_DECODE ("stat:mem_master_ops_peak", long long, m_stat_mem_master_ops.peak());
    ATTR_DECODE ("stat:mem_master_args_current", long long, m_stat_mem_master_args.current());
    ATTR_DECODE ("stat:mem_master_args_peak", long long, m_stat_mem_master_args.peak());
    ATTR_DECODE ("stat:mem_master_syms_current", long long, m_stat_mem_master_syms.current());
    ATTR_DECODE ("stat:mem_master_syms_peak", long long, m_stat_mem_master_syms.peak());
    ATTR_DECODE ("stat:mem_master_defaults_current", long long, m_stat_mem_master_defaults.current());
    ATTR_DECODE ("stat:mem_master_defaults_peak", long long, m_stat_mem_master_defaults.peak());
    ATTR_DECODE ("stat:mem_master_consts_current", long long, m_stat_mem_master_consts.current());
    ATTR_DECODE ("stat:mem_master_consts_peak", long long, m_stat_mem_master_consts.peak());
    ATTR_DECODE ("stat:mem_inst_current", long long, m_stat_mem_inst.current());
    ATTR_DECODE ("stat:mem_inst_peak", long long, m_stat_mem_inst.peak());
    ATTR_DECODE ("stat:mem_inst_syms_current", long long, m_stat_mem_inst_syms.current());
    ATTR_DECODE ("stat:mem_inst_syms_peak", long long, m_stat_mem_inst_syms.peak());
    ATTR_DECODE ("stat:mem_inst_paramvals_current", long long, m_stat_mem_inst_paramvals.current());
    ATTR_DECODE ("stat:mem_inst_paramvals_peak", long long, m_stat_mem_inst_paramvals.peak());
    ATTR_DECODE ("stat:mem_inst_connections_current", long long, m_stat_mem_inst_connections.current());
    ATTR_DECODE ("stat:mem_inst_connections_peak", long long, m_stat_mem_inst_connections.peak());

    return false;
#undef ATTR_DECODE
#undef ATTR_DECODE_STRING
}



bool
ShadingSystemImpl::attribute (ShaderGroup *group, string_view name,
                              TypeDesc type, const void *val)
{
    // No current group attributes to set
    if (! group)
        return attribute (name, type, val);
    lock_guard lock (group->m_mutex);
    if (name == "renderer_outputs" && type.basetype == TypeDesc::STRING) {
        group->m_renderer_outputs.clear ();
        for (size_t i = 0;  i < type.numelements();  ++i)
            group->m_renderer_outputs.push_back (ustring(((const char **)val)[i]));
        return true;
    }
    if (name == "entry_layers" && type.basetype == TypeDesc::STRING) {
        group->clear_entry_layers ();
        for (int i = 0;  i < (int)type.numelements();  ++i)
            group->mark_entry_layer (ustring(((const char **)val)[i]));
        return true;
    }
    if (name == "exec_repeat" && type == TypeDesc::TypeInt) {
        group->m_exec_repeat = *(const int *)val;
        return true;
    }
    if (name == "groupname" && type == TypeDesc::TypeString) {
        group->name (ustring(((const char **)val)[0]));
        return true;
    }
    return false;
}



bool
ShadingSystemImpl::getattribute (ShaderGroup *group, string_view name,
                                 TypeDesc type, void *val)
{
    if (! group)
        return false;
    if (name == "groupname" && type == TypeDesc::TypeString) {
        *(ustring *)val = group->name();
        return true;
    }
    if (name == "num_layers" && type == TypeDesc::TypeInt) {
        *(int *)val = group->nlayers();
        return true;
    }
    if (name == "layer_names" && type.basetype == TypeDesc::STRING) {
        size_t n = std::min (type.numelements(), (size_t)group->nlayers());
        for (size_t i = 0;  i < n;  ++i)
            ((ustring *)val)[i] = (*group)[i]->layername();
        return true;
    }
    if (name == "num_renderer_outputs" && type.basetype == TypeDesc::INT) {
        *(int *)val = (int) group->m_renderer_outputs.size();
        return true;
    }
    if (name == "renderer_outputs" && type.basetype == TypeDesc::STRING) {
        size_t n = std::min (type.numelements(), group->m_renderer_outputs.size());
        for (size_t i = 0;  i < n;  ++i)
            ((ustring *)val)[i] = group->m_renderer_outputs[i];
        for (size_t i = n;  i < type.numelements();  ++i)
            ((ustring *)val)[i] = ustring();
        return true;
    }
    if (name == "raytype_queries" && type.basetype == TypeDesc::INT) {
        *(int *)val = group->raytype_queries();
        return true;
    }
    if (name == "num_entry_layers" && type.basetype == TypeDesc::INT) {
        int n = 0;
        for (int i = 0;  i < group->nlayers();  ++i)
            n += group->layer(i)->entry_layer();
        *(int *)val = n;
        return true;
    }
    if (name == "entry_layers" && type.basetype == TypeDesc::STRING) {
        size_t n = 0;
        for (size_t i = 0;  i < (size_t)group->nlayers() && i < type.numelements();  ++i)
            if (group->layer(i)->entry_layer())
                ((ustring *)val)[n++] = (*group)[i]->layername();
        for (size_t i = n;  i < type.numelements();  ++i)
            ((ustring *)val)[i] = ustring();
        return true;
    }
    if (name == "num_textures_needed" && type == TypeDesc::TypeInt) {
        if (! group->optimized())
            optimize_group (*group);
        *(int *)val = (int)group->m_textures_needed.size();
        return true;
    }
    if (name == "textures_needed" && type.basetype == TypeDesc::PTR) {
        if (! group->optimized())
            optimize_group (*group);
        size_t n = group->m_textures_needed.size();
        *(ustring **)val = n ? &group->m_textures_needed[0] : NULL;
        return true;
    }
    if (name == "unknown_textures_needed" && type == TypeDesc::TypeInt) {
        if (! group->optimized())
            optimize_group (*group);
        *(int *)val = (int)group->m_unknown_textures_needed;
        return true;
    }

    if (name == "num_closures_needed" && type == TypeDesc::TypeInt) {
        if (! group->optimized())
            optimize_group (*group);
        *(int *)val = (int)group->m_closures_needed.size();
        return true;
    }
    if (name == "closures_needed" && type.basetype == TypeDesc::PTR) {
        if (! group->optimized())
            optimize_group (*group);
        size_t n = group->m_closures_needed.size();
        *(ustring **)val = n ? &group->m_closures_needed[0] : NULL;
        return true;
    }
    if (name == "unknown_closures_needed" && type == TypeDesc::TypeInt) {
        if (! group->optimized())
            optimize_group (*group);
        *(int *)val = (int)group->m_unknown_closures_needed;
        return true;
    }

    if (name == "num_globals_needed" && type == TypeDesc::TypeInt) {
        if (! group->optimized())
            optimize_group (*group);
        *(int *)val = (int)group->m_globals_needed.size();
        return true;
    }
    if (name == "globals_needed" && type.basetype == TypeDesc::PTR) {
        if (! group->optimized())
            optimize_group (*group);
        size_t n = group->m_globals_needed.size();
        *(ustring **)val = n ? &group->m_globals_needed[0] : NULL;
        return true;
    }

    if (name == "num_userdata" && type == TypeDesc::TypeInt) {
        if (! group->optimized())
            optimize_group (*group);
        *(int *)val = (int)group->m_userdata_names.size();
        return true;
    }
    if (name == "userdata_names" && type.basetype == TypeDesc::PTR) {
        if (! group->optimized())
            optimize_group (*group);
        size_t n = group->m_userdata_names.size();
        *(ustring **)val = n ? &group->m_userdata_names[0] : NULL;
        return true;
    }
    if (name == "userdata_types" && type.basetype == TypeDesc::PTR) {
        if (! group->optimized())
            optimize_group (*group);
        size_t n = group->m_userdata_types.size();
        *(TypeDesc **)val = n ? &group->m_userdata_types[0] : NULL;
        return true;
    }
    if (name == "userdata_offsets" && type.basetype == TypeDesc::PTR) {
        if (! group->optimized())
            optimize_group (*group);
        size_t n = group->m_userdata_offsets.size();
        *(int **)val = n ? &group->m_userdata_offsets[0] : NULL;
        return true;
    }
    if (name == "userdata_derivs" && type.basetype == TypeDesc::PTR) {
        if (! group->optimized())
            optimize_group (*group);
        size_t n = group->m_userdata_derivs.size();
        *(char **)val = n ? &group->m_userdata_derivs[0] : NULL;
        return true;
    }
    if (name == "pickle" && type == TypeDesc::STRING) {
        *(ustring *)val = ustring(group->serialize());
        return true;
    }

    if (name == "num_attributes_needed" && type == TypeDesc::TypeInt) {
        if (! group->optimized())
            optimize_group (*group);
        *(int *)val = (int)group->m_attributes_needed.size();
        return true;
    }
    if (name == "attributes_needed" && type.basetype == TypeDesc::PTR) {
        if (! group->optimized())
            optimize_group (*group);
        size_t n = group->m_attributes_needed.size();
        *(ustring **)val = n ? &group->m_attributes_needed[0] : NULL;
        return true;
    }
    if (name == "attribute_scopes" && type.basetype == TypeDesc::PTR) {
        if (! group->optimized())
            optimize_group (*group);
        size_t n = group->m_attribute_scopes.size();
        *(ustring **)val = n ? &group->m_attribute_scopes[0] : NULL;
        return true;
    }
    if (name == "unknown_attributes_needed" && type == TypeDesc::TypeInt) {
        if (! group->optimized())
            optimize_group (*group);
        *(int *)val = (int)group->m_unknown_attributes_needed;
        return true;
    }
    if (name == "exec_repeat" && type == TypeDesc::TypeInt) {
        *(int *)val = group->m_exec_repeat;
        return true;
    }

    return false;
}



void
ShadingSystemImpl::error (const std::string &msg) const
{
    lock_guard guard (m_errmutex);
    int n = 0;
    for (auto&& s : m_errseen) {
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
ShadingSystemImpl::warning (const std::string &msg) const
{
    lock_guard guard (m_errmutex);
    int n = 0;
    for (auto&& s : m_warnseen) {
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
ShadingSystemImpl::info (const std::string &msg) const
{
    lock_guard guard (m_errmutex);
    m_err->info (msg);
}



void
ShadingSystemImpl::message (const std::string &msg) const
{
    lock_guard guard (m_errmutex);
    m_err->message (msg);
}



void
ShadingSystemImpl::pointcloud_stats (int search, int get, int results,
                                     int writes)
{
    spin_lock lock (m_stat_mutex);
    m_stat_pointcloud_searches += search;
    m_stat_pointcloud_gets += get;
    m_stat_pointcloud_searches_total_results += results;
    if (search && ! results)
        ++m_stat_pointcloud_failures;
    m_stat_pointcloud_max_results = std::max (m_stat_pointcloud_max_results,
                                              results);
    m_stat_pointcloud_writes += writes;
}



namespace {
typedef std::pair<ustring,long long> GroupTimeVal;
struct group_time_compare { // So looking forward to C++11 lambdas!
    bool operator() (const GroupTimeVal &a, const GroupTimeVal &b) {
        return a.second > b.second;
    }
};
}



std::string
ShadingSystemImpl::getstats (int level) const
{
    if (level <= 0)
        return "";
    std::ostringstream out;
    out << "OSL ShadingSystem statistics (" << (void*)this;
    out << ") ver " << OSL_LIBRARY_VERSION_STRING
        << ", LLVM " << OSL_LLVM_FULL_VERSION << "\n";
    if (m_stat_shaders_requested == 0) {
        out << "  No shaders requested\n";
        return out.str();
    }

    std::string opt;
#define BOOLOPT(name) opt += Strutil::format(#name "=%d ", m_##name)
#define INTOPT(name) opt += Strutil::format(#name "=%d ", m_##name)
#define STROPT(name) if (m_##name.size()) opt += Strutil::format(#name "=\"%s\" ", m_##name)
    INTOPT (optimize);
    INTOPT (llvm_optimize);
    INTOPT (debug);
    INTOPT (profile);
    INTOPT (llvm_debug);
    BOOLOPT (llvm_debug_layers);
    BOOLOPT (llvm_debug_ops);
    BOOLOPT (lazylayers);
    BOOLOPT (lazyglobals);
    BOOLOPT (lazyunconnected);
    BOOLOPT (lazy_userdata);
    BOOLOPT (userdata_isconnected);
    BOOLOPT (clearmemory);
    BOOLOPT (debugnan);
    BOOLOPT (debug_uninit);
    BOOLOPT (lockgeom_default);
    BOOLOPT (strict_messages);
    BOOLOPT (range_checking);
    BOOLOPT (greedyjit);
    BOOLOPT (countlayerexecs);
    BOOLOPT (opt_simplify_param);
    BOOLOPT (opt_constant_fold);
    BOOLOPT (opt_stale_assign);
    BOOLOPT (opt_elide_useless_ops);
    BOOLOPT (opt_elide_unconnected_outputs);
    BOOLOPT (opt_peephole);
    BOOLOPT (opt_coalesce_temps);
    BOOLOPT (opt_assign);
    BOOLOPT (opt_mix);
    INTOPT  (opt_merge_instances);
    BOOLOPT (opt_merge_instances_with_userdata);
    BOOLOPT (opt_fold_getattribute);
    BOOLOPT (opt_middleman);
    BOOLOPT (opt_texture_handle);
    BOOLOPT (opt_seed_bblock_aliases);
    INTOPT  (opt_passes);
    INTOPT (no_noise);
    INTOPT (no_pointcloud);
    INTOPT (force_derivs);
    INTOPT (exec_repeat);
    STROPT (debug_groupname);
    STROPT (debug_layername);
    STROPT (archive_groupname);
    STROPT (archive_filename);
#undef BOOLOPT
#undef INTOPT
#undef STROPT
    out << "  Options:  " << Strutil::wordwrap(opt, 75, 12) << "\n";

    out << "  Shaders:\n";
    out << "    Requested: " << m_stat_shaders_requested << "\n";
    out << "    Loaded:    " << m_stat_shaders_loaded << "\n";
    out << "    Masters:   " << m_stat_shaders_loaded << "\n";
    out << "    Instances: " << m_stat_instances << "\n";
    out << "  Time loading masters: "
        << Strutil::timeintervalformat (m_stat_master_load_time, 2) << "\n";
    out << "  Shading groups:   " << m_stat_groups << "\n";
    out << "    Total instances in all groups: " << m_stat_groupinstances << "\n";
    float iperg = (float)m_stat_groupinstances/std::max(m_stat_groups,1);
    out << "    Avg instances per group: "
        << Strutil::format ("%.1f", iperg) << "\n";
    out << "  Shading contexts: " << m_stat_contexts << "\n";
    if (m_countlayerexecs)
        out << "  Total layers executed: " << m_stat_layers_executed << "\n";

#if 0
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

#endif

    out << "  Compiled " << m_stat_groups_compiled << " groups, "
        << m_stat_instances_compiled << " instances\n";
    out << "  Merged " << (m_stat_merged_inst+m_stat_merged_inst_opt)
        << " instances (" << m_stat_merged_inst << " initial, "
        << m_stat_merged_inst_opt << " after opt) in "
        << Strutil::timeintervalformat (m_stat_inst_merge_time, 2) << "\n";
    if (m_stat_instances_compiled > 0)
        out << "  After optimization, " << m_stat_empty_instances
            << " empty instances ("
            << (int)(100.0f*m_stat_empty_instances/m_stat_instances_compiled) << "%)\n";
    if (m_stat_groups_compiled > 0)
        out << "  After optimization, " << m_stat_empty_groups << " empty groups ("
            << (int)(100.0f*m_stat_empty_groups/m_stat_groups_compiled)<< "%)\n";
    if (m_stat_instances_compiled > 0 || m_stat_groups_compiled > 0) {
        out << Strutil::format ("  Optimized %llu ops to %llu (%.1f%%)\n",
                                (long long)m_stat_preopt_ops,
                                (long long)m_stat_postopt_ops,
                                100.0*(double(m_stat_postopt_ops)/double(std::max(1,(int)m_stat_preopt_ops))-1.0));
        out << Strutil::format ("  Optimized %llu symbols to %llu (%.1f%%)\n",
                                (long long)m_stat_preopt_syms,
                                (long long)m_stat_postopt_syms,
                                100.0*(double(m_stat_postopt_syms)/double(std::max(1,(int)m_stat_preopt_syms))-1.0));
    }
    out << Strutil::format ("  Constant connections eliminated: %d\n",
                            (int)m_stat_const_connections);
    out << Strutil::format ("  Global connections eliminated: %d\n",
                            (int)m_stat_global_connections);
    out << Strutil::format ("  Middlemen eliminated: %d\n",
                            (int)m_stat_middlemen_eliminated);
    out << Strutil::format ("  Derivatives needed on %d / %d symbols (%.1f%%)\n",
                            (int)m_stat_syms_with_derivs, (int)m_stat_postopt_syms,
                            (100.0*(int)m_stat_syms_with_derivs)/std::max((int)m_stat_postopt_syms,1));
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

    out << "  Texture calls compiled: "
        << (int)m_stat_tex_calls_codegened
        << " (" << (int)m_stat_tex_calls_as_handles << " used handles)\n";
    out << "  Regex's compiled: " << m_stat_regexes << "\n";
    out << "  Largest generated function local memory size: "
        << m_stat_max_llvm_local_mem/1024 << " KB\n";
    if (m_stat_getattribute_calls) {
        out << "  getattribute calls: " << m_stat_getattribute_calls << " ("
            << Strutil::timeintervalformat (m_stat_getattribute_time, 2) << ")\n";
        out << "     (fail time "
            << Strutil::timeintervalformat (m_stat_getattribute_fail_time, 2) << ")\n";
    }
    out << "  Number of get_userdata calls: " << m_stat_get_userdata_calls << "\n";
    if (profile() > 1)
        out << "  Number of noise calls: " << m_stat_noise_calls << "\n";
    if (m_stat_pointcloud_searches || m_stat_pointcloud_writes) {
        out << "  Pointcloud operations:\n";
        out << "    pointcloud_search calls: " << m_stat_pointcloud_searches << "\n";
        out << "      max query results: " << m_stat_pointcloud_max_results << "\n";
        double avg = m_stat_pointcloud_searches ?
            (double)m_stat_pointcloud_searches_total_results/(double)m_stat_pointcloud_searches : 0.0;
        out << "      average query results: " << Strutil::format ("%.1f", avg) << "\n";
        out << "      failures: " << m_stat_pointcloud_failures << "\n";
        out << "    pointcloud_get calls: " << m_stat_pointcloud_gets << "\n";
        out << "    pointcloud_write calls: " << m_stat_pointcloud_writes << "\n";
    }
    out << "  Memory total: " << m_stat_memory.memstat() << '\n';
    out << "    Master memory: " << m_stat_mem_master.memstat() << '\n';
    out << "        Master ops:            " << m_stat_mem_master_ops.memstat() << '\n';
    out << "        Master args:           " << m_stat_mem_master_args.memstat() << '\n';
    out << "        Master syms:           " << m_stat_mem_master_syms.memstat() << '\n';
    out << "        Master defaults:       " << m_stat_mem_master_defaults.memstat() << '\n';
    out << "        Master consts:         " << m_stat_mem_master_consts.memstat() << '\n';
    out << "    Instance memory: " << m_stat_mem_inst.memstat() << '\n';
    out << "        Instance syms:         " << m_stat_mem_inst_syms.memstat() << '\n';
    out << "        Instance param values: " << m_stat_mem_inst_paramvals.memstat() << '\n';
    out << "        Instance connections:  " << m_stat_mem_inst_connections.memstat() << '\n';

    size_t jitmem = LLVM_Util::total_jit_memory_held();
    out << "    LLVM JIT memory: " << Strutil::memformat(jitmem) << '\n';

    if (m_profile) {
        out << "  Execution profile:\n";
        out << "    Total shader execution time: "
            << Strutil::timeintervalformat(OIIO::Timer::seconds(m_stat_total_shading_time_ticks), 2)
            << " (sum of all threads)\n";
        // Account for times of any groups that haven't yet been destroyed
        {
            spin_lock lock (m_all_shader_groups_mutex);
            for (size_t i = 0, e = m_all_shader_groups.size(); i < e; ++i) {
                if (ShaderGroupRef g = m_all_shader_groups[i].lock()) {
                    long long ticks = g->m_stat_total_shading_time_ticks;
                    m_group_profile_times[g->name()] += ticks;
                    g->m_stat_total_shading_time_ticks -= ticks;
                }
            }
        }
        {
            spin_lock lock (m_stat_mutex);
            std::vector<GroupTimeVal> grouptimes;
            for (std::map<ustring,long long>::const_iterator m = m_group_profile_times.begin();
                 m != m_group_profile_times.end(); ++m) {
                grouptimes.push_back (GroupTimeVal(m->first, m->second));
            }
            std::sort (grouptimes.begin(), grouptimes.end(), group_time_compare());
            if (grouptimes.size() > 5)
                grouptimes.resize (5);
            if (grouptimes.size())
                out << "    Most expensive shader groups:\n";
            for (std::vector<GroupTimeVal>::const_iterator i = grouptimes.begin();
                     i != grouptimes.end(); ++i) {
                out << "      " << Strutil::timeintervalformat(OIIO::Timer::seconds(i->second),2) 
                    << ' ' << (i->first.size() ? i->first.c_str() : "<unnamed group>") << "\n";
            }
        }

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
ShadingSystemImpl::Parameter (string_view name, TypeDesc t, const void *val,
                              bool lockgeom)
{
    // We work very hard not to do extra copies of the data.  First,
    // grow the pending list by one (empty) slot...
    m_pending_params.grow();
    // ...then initialize it in place
    m_pending_params.back().init (name, t, 1, val);
    // If we have a possible geometric override (lockgeom=false), set the
    // param's interpolation to VERTEX rather than the default CONSTANT.
    if (lockgeom == false)
        m_pending_params.back().interp (OIIO::ParamValue::INTERP_VERTEX);
    return true;
}



bool
ShadingSystemImpl::Parameter (string_view name, TypeDesc t, const void *val)
{
    return Parameter (name, t, val, true);
}



ShaderGroupRef
ShadingSystemImpl::ShaderGroupBegin (string_view groupname)
{
    if (m_in_group) {
        error ("Nested ShaderGroupBegin() calls");
        return ShaderGroupRef();
    }
    m_in_group = true;
    m_group_use = ShadUseUnknown;
    m_curgroup.reset (new ShaderGroup(groupname));
    m_curgroup->m_exec_repeat = m_exec_repeat;
    return m_curgroup;
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
        int nlayers = m_curgroup->nlayers ();
        for (int layer = 0;  layer < nlayers;  ++layer) {
            ShaderInstance *inst = (*m_curgroup)[layer];
            if (! inst)
                continue;
            inst->last_layer (layer == nlayers-1);
        }

        // Merge instances now if they really want it bad, otherwise wait
        // until we optimize the group.
        if (m_opt_merge_instances >= 2)
            merge_instances (*m_curgroup);
    }

    // Merge the raytype_queries of all the individual layers
    m_curgroup->m_raytype_queries = 0;
    for (int layer = 0, n = m_curgroup->nlayers();  layer < n;  ++layer) {
        ASSERT ((*m_curgroup)[layer]);
        if (ShaderInstance *inst = (*m_curgroup)[layer])
            m_curgroup->m_raytype_queries |= inst->master()->raytype_queries();
    }
    // std::cout << "Group " << m_curgroup->name() << " ray query bits "
    //         << m_curgroup->m_raytype_queries << "\n";

    {
        // Record the group in the SS's census of all extant groups
        spin_lock lock (m_all_shader_groups_mutex);
        m_all_shader_groups.push_back (m_curgroup);
        ++m_groups_to_compile_count;
    }

    m_in_group = false;
    m_group_use = ShadUseUnknown;

    ustring groupname = m_curgroup->name();
    if (groupname.size() && groupname == m_archive_groupname) {
        std::string filename = m_archive_filename.string();
        if (! filename.size())
            filename = OIIO::Filesystem::filename (groupname.string()) + ".tar.gz";
        archive_shadergroup (m_curgroup.get(), filename);
    }
    return true;
}



bool
ShadingSystemImpl::Shader (string_view shaderusage,
                           string_view shadername,
                           string_view layername)
{
    // Make sure we have a current attrib state
    bool singleton = (! m_curgroup);
    if (singleton)
        ShaderGroupBegin ("");

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

    ShaderInstanceRef instance (new ShaderInstance (master, layername));
    instance->parameters(m_pending_params);
    m_pending_params.clear ();

    if (singleton || m_group_use == ShadUseUnknown) {
        // A singleton, or the first in a group
        m_curgroup->clear ();
        m_stat_groups += 1;
    }
    if (! singleton) {
        if (m_group_use == ShadUseUnknown) {  // First shader in group
            m_group_use = use;
        } else if (use != m_group_use) {
            error ("Shader usage \"%s\" does not match current group (%s)",
                   shaderusage, shaderusename (m_group_use));
            return false;
        }
    }

    m_curgroup->append (instance);
    m_stat_groupinstances += 1;

    // FIXME -- check for duplicate layer name within the group?

    return true;
}



bool
ShadingSystemImpl::ConnectShaders (string_view srclayer, string_view srcparam,
                                   string_view dstlayer, string_view dstparam)
{
    // Basic sanity checks -- make sure it's a legal time to call
    // ConnectShaders, and that the layer and parameter names are not empty.
    if (! m_in_group) {
        error ("ConnectShaders can only be called within ShaderGroupBegin/End");
        return false;
    }
    if (! srclayer.size() || ! srcparam.size()) {
        error ("ConnectShaders: badly formed source layer/parameter");
        return false;
    }
    if (! dstlayer.size() || ! dstparam.size()) {
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
        error ("ConnectShaders: destination layer must follow source layer (tried to connect %s.%s -> %s.%s)\n", srclayer, srcparam, dstlayer, dstparam);
        return false;
    }

    // Decode the parameter names, find their symbols in their
    // respective layers, and also decode requrest to attach specific
    // array elements or color/vector channels.
    ConnectedParam srccon = decode_connected_param(srcparam, srclayer, srcinst);
    ConnectedParam dstcon = decode_connected_param(dstparam, dstlayer, dstinst);
    if (! (srccon.valid() && dstcon.valid())) {
        if (connection_error())
            error ("ConnectShaders: cannot connect a %s (%s) to a %s (%s), invalid connection",
                   srccon.type, srcparam, dstcon.type, dstparam);
        else
            warning ("ConnectShaders: cannot connect a %s (%s) to a %s (%s), invalid connection",
                     srccon.type, srcparam, dstcon.type, dstparam);
        return false;
    }

    if (srccon.type.is_structure() && dstcon.type.is_structure() &&
            equivalent (srccon.type, dstcon.type)) {
        // If the connection is whole struct-to-struct (and they are
        // structs with equivalent data layout), implement it underneath
        // as connections between their respective fields.
        StructSpec *srcstruct = srccon.type.structspec();
        StructSpec *dststruct = dstcon.type.structspec();
        for (size_t i = 0;  i < (size_t)srcstruct->numfields();  ++i) {
            std::string s = Strutil::format("%s.%s", srcparam, srcstruct->field(i).name);
            std::string d = Strutil::format("%s.%s", dstparam, dststruct->field(i).name);
            ConnectShaders (srclayer, s, dstlayer, d);
        }
        return true;
    }

    if (! assignable (dstcon.type, srccon.type)) {
        if (connection_error())
            error ("ConnectShaders: cannot connect a %s (%s) to a %s (%s)",
                   srccon.type.c_str(), srcparam, dstcon.type.c_str(), dstparam);
        else
            warning ("ConnectShaders: cannot connect a %s (%s) to a %s (%s)",
                     srccon.type.c_str(), srcparam, dstcon.type.c_str(), dstparam);
        return false;
    }

    dstinst->add_connection (srcinstindex, srccon, dstcon);
    dstinst->instoverride(dstcon.param)->valuesource (Symbol::ConnectedVal);
    srcinst->instoverride(srccon.param)->connected_down (true);
    srcinst->outgoing_connections (true);

    // if (debug())
    //     message ("ConnectShaders %s %s -> %s %s\n",
    //              srclayer, srcparam, dstlayer, dstparam);

    return true;
}



ShaderGroupRef
ShadingSystemImpl::ShaderGroupBegin (string_view groupname,
                                     string_view usage,
                                     string_view groupspec)
{
    ShaderGroupRef g = ShaderGroupBegin (groupname);
    bool err = false;
    std::string errdesc;
    std::vector<int> intvals;
    std::vector<float> floatvals;
    std::vector<ustring> stringvals;
    string_view p = groupspec;   // parse view
    // std::cout << "!!!!!\n---\n" << groupspec << "\n---\n\n";
    while (p.size()) {
        Strutil::skip_whitespace (p);
        if (! p.size())
            break;
        while (Strutil::parse_char (p, ';'))  // skip blank statements
            ;
        string_view keyword = Strutil::parse_word (p);

        if (keyword == "shader") {
            string_view shadername = Strutil::parse_identifier (p);
            Strutil::skip_whitespace (p);
            string_view layername = Strutil::parse_until (p, " \t\r\n,;");
            Shader (usage, shadername, layername);
            Strutil::parse_char (p, ';') || Strutil::parse_char (p, ',');
            Strutil::skip_whitespace (p);
            continue;
        }

        if (keyword == "connect") {
            Strutil::skip_whitespace (p);
            string_view lay1 = Strutil::parse_until (p, " \t\r\n.");
            Strutil::parse_char (p, '.');
            string_view param1 = Strutil::parse_until (p, " \t\r\n,;");
            Strutil::skip_whitespace (p);
            string_view lay2 = Strutil::parse_until (p, " \t\r\n.");
            Strutil::parse_char (p, '.');
            string_view param2 = Strutil::parse_until (p, " \t\r\n,;");
            ConnectShaders (lay1, param1, lay2, param2);
            Strutil::parse_char (p, ';') || Strutil::parse_char (p, ',');
            Strutil::skip_whitespace (p);
            continue;
        }

        // Remaining case -- it should be declaring a parameter.
        string_view typestring;
        if (keyword == "param") {
            typestring = Strutil::parse_word (p);
        } else if (TypeDesc(keyword.str().c_str()) != TypeDesc::UNKNOWN) {
            // compatibility: let the 'param' keyword be optional, if it's
            // obvious that it's a type name.
            typestring = keyword;
        } else {
            err = true;
            errdesc = Strutil::format ("Unknown statement (expected 'param', "
                                       "'shader', or 'connect'): \"%s\"",
                                       keyword);
            break;
        }
        TypeDesc type;
        if (typestring == "int")
            type = TypeDesc::TypeInt;
        else if (typestring == "float")
            type = TypeDesc::TypeFloat;
        else if (typestring == "color")
            type = TypeDesc::TypeColor;
        else if (typestring == "point")
            type = TypeDesc::TypePoint;
        else if (typestring == "vector")
            type = TypeDesc::TypeVector;
        else if (typestring == "normal")
            type = TypeDesc::TypeNormal;
        else if (typestring == "matrix")
            type = TypeDesc::TypeMatrix;
        else if (typestring == "string")
            type = TypeDesc::TypeString;
        else {
            err = true;
            errdesc = Strutil::format ("Unknown type: %s", typestring);
            break;  // error
        }
        if (Strutil::parse_char (p, '[')) {
            int arraylen = -1;
            Strutil::parse_int (p, arraylen);
            Strutil::parse_char (p, ']');
            type.arraylen = arraylen;
        }
        std::string paramname_string;
        while (1) {
            paramname_string += Strutil::parse_identifier (p);
            Strutil::skip_whitespace (p);
            if (Strutil::parse_char (p, '.')) {
                paramname_string += ".";
            } else {
                break;
            }
        }
        string_view paramname (paramname_string);
        int lockgeom = true;
        // For speed, reserve space. Note that for "unsized" arrays, we only
        // preallocate 1 slot and let it grow as needed. That's ok. For
        // everything else, we will reserve the right amount up front.
        int vals_to_preallocate = type.is_unsized_array()
                                ? 1 : type.numelements() * type.aggregate;
        // Stop parsing values when we hit the limit based on the
        // declaration.
        int max_vals = type.is_unsized_array() ? 1<<28 : vals_to_preallocate;
        if (type.basetype == TypeDesc::INT) {
            intvals.clear ();
            intvals.reserve (vals_to_preallocate);
            int i;
            for (i = 0; i < max_vals; ++i) {
                int val = 0;
                if (Strutil::parse_int (p, val))
                    intvals.push_back (val);
                else
                    break;
            }
            if (type.is_unsized_array()) {
                // For unsized arrays, now set the size based on how many
                // values we actually read.
                type.arraylen = std::max (1, i/type.aggregate);
            }
            // Zero-pad if we parsed fewer values than we needed
            intvals.resize (type.numelements()*type.aggregate, 0);
            ASSERT (int(type.numelements())*type.aggregate == int(intvals.size()));
        } else if (type.basetype == TypeDesc::FLOAT) {
            floatvals.clear ();
            floatvals.reserve (vals_to_preallocate);
            int i;
            for (i = 0; i < max_vals; ++i) {
                float val = 0;
                if (Strutil::parse_float (p, val))
                    floatvals.push_back (val);
                else
                    break;
            }
            if (type.is_unsized_array()) {
                // For unsized arrays, now set the size based on how many
                // values we actually read.
                type.arraylen = std::max (1, i/type.aggregate);
            }
            // Zero-pad if we parsed fewer values than we needed
            floatvals.resize (type.numelements()*type.aggregate, 0);
            ASSERT (int(type.numelements())*type.aggregate == int(floatvals.size()));
        } else if (type.basetype == TypeDesc::STRING) {
            stringvals.clear ();
            stringvals.reserve (vals_to_preallocate);
            int i;
            for (i = 0; i < max_vals; ++i) {
                std::string unescaped;
                string_view s;
                Strutil::skip_whitespace (p);
                if (p.size() && p[0] == '\"') {
                    if (! Strutil::parse_string (p, s))
                        break;
                    unescaped = Strutil::unescape_chars (s);
                    s = unescaped;
                }
                else {
                    s = Strutil::parse_until (p, " \t\r\n;");
                    if (s.size() == 0)
                        break;
                }
                stringvals.push_back (ustring(s));
            }
            if (type.is_unsized_array()) {
                // For unsized arrays, now set the size based on how many
                // values we actually read.
                type.arraylen = std::max (1, i/type.aggregate);
            }
            // Zero-pad if we parsed fewer values than we needed
            stringvals.resize (type.numelements()*type.aggregate, ustring());
            ASSERT (int(type.numelements())*type.aggregate == int(stringvals.size()));
        }

        if (Strutil::parse_prefix (p, "[[")) {  // hints
            do {
                Strutil::skip_whitespace (p);
                string_view hint_typename = Strutil::parse_word (p);
                string_view hint_name = Strutil::parse_identifier (p);
                TypeDesc hint_type (hint_typename.str().c_str());
                if (! hint_name.size() || hint_type == TypeDesc::UNKNOWN) {
                    err = true;
                    errdesc = "malformed hint";
                    break;
                }
                if (! Strutil::parse_char (p, '=')) {
                    err = true;
                    errdesc = "hint expected value";
                    break;
                }
                if (hint_name == "lockgeom" && hint_type == TypeDesc::INT) {
                    if (! Strutil::parse_int (p, lockgeom)) {
                        err = true;
                        errdesc = Strutil::format ("hint %s expected int value", hint_name);
                        break;
                    }
                } else {
                    err = true;
                    errdesc = Strutil::format ("unknown hint '%s %s'",
                                               hint_type, hint_name);
                    break;
                }
            } while (Strutil::parse_char (p, ','));
            if (err)
                break;
            if (! Strutil::parse_prefix (p, "]]")) {
                err = true;
                errdesc = "malformed hint";
                break;
            }
        }

        if (type.basetype == TypeDesc::INT) {
            Parameter (paramname, type, &intvals[0], lockgeom);
        } else if (type.basetype == TypeDesc::FLOAT) {
            Parameter (paramname, type, &floatvals[0], lockgeom);
        } else if (type.basetype == TypeDesc::STRING) {
            Parameter (paramname, type, &stringvals[0], lockgeom);
        }

        Strutil::skip_whitespace (p);
        if (! p.size())
            break;

        if (Strutil::parse_char (p, ';') || Strutil::parse_char (p, ','))
            continue;  // next command

        Strutil::parse_until_char (p, ';');
        if (! Strutil::parse_char (p, ';')) {
            err = true;
            errdesc = "semicolon expected";
        }
    }

    if (err) {
        if (! groupname.size())
            groupname = "<unknown>";
        size_t offset = p.data() - groupspec.data();
        size_t begin_stmt = std::min (groupspec.find_last_of (';', offset),
                                      groupspec.find_last_of (',', offset));
        size_t end_stmt = groupspec.find_first_of (';', begin_stmt+1);
        string_view statement = groupspec.substr (begin_stmt+1, end_stmt-begin_stmt);
        error ("ShaderGroupBegin: error parsing group description: %s\n"
               "        group: \"%s\"\n"
               "        problem might be here: %s\n",
               errdesc, groupname, statement);
        if (debug())
            info ("Broken group was:\n---%s\n---\n", groupspec);
        return ShaderGroupRef();
    }

    return g;
}



std::string
ShadingSystemImpl::serialize_group (ShaderGroup *group)
{
    return group->serialize ();
}



bool
ShadingSystemImpl::ReParameter (ShaderGroup &group, string_view layername_,
                                string_view paramname,
                                TypeDesc type, const void *val)
{
    // Find the named layer
    ustring layername (layername_);
    ShaderInstance *layer = NULL;
    for (int i = 0, e = group.nlayers();  i < e;  ++i) {
        if (group[i]->layername() == layername) {
            layer = group[i];
            break;
        }
    }
    if (! layer)
        return false;   // could not find the named layer

    // Find the named parameter within the layer
    int paramindex = layer->findparam (ustring(paramname));
    if (paramindex < 0)
        return false;   // could not find the named parameter
    Symbol *sym = layer->symbol (paramindex);
    ASSERT (sym != NULL);

    // Check for mismatch versus previously-declared type
    if (!equivalent(sym->typespec(), type))
        return false;

    // Can't change param value if the group has already been optimized,
    // unless that parameter is marked lockgeom=0.
    if (group.optimized() && sym->lockgeom())
        return false;

    // Do the deed
    memcpy (sym->data(), val, type.size());
    return true;
}



PerThreadInfo *
ShadingSystemImpl::create_thread_info()
{
    return new PerThreadInfo;
}



void
ShadingSystemImpl::destroy_thread_info (PerThreadInfo *threadinfo)
{
    delete threadinfo;
}



ShadingContext *
ShadingSystemImpl::get_context (PerThreadInfo *threadinfo,
                                TextureSystem::Perthread *texture_threadinfo)
{
    if (! threadinfo)
        threadinfo = get_perthread_info ();
    ShadingContext *ctx = threadinfo->context_pool.empty()
                          ? new ShadingContext (*this, threadinfo)
                          : threadinfo->pop_context ();
    ctx->texture_thread_info (texture_threadinfo);
    return ctx;
}



void
ShadingSystemImpl::release_context (ShadingContext *ctx)
{
    if (! ctx)
        return;
    ctx->process_errors ();
    ctx->thread_info()->context_pool.push (ctx);
}



bool
ShadingSystemImpl::execute (ShadingContext *ctx, ShaderGroup &group,
                            ShaderGlobals &ssg, bool run)
{
    bool free_context = false;
    if (! ctx) {
        ctx = get_context();
        free_context = true;
    }
    bool result = ctx->execute (group, ssg, run);
    if (free_context)
        release_context (ctx);
    return result;
}


bool
ShadingSystemImpl::execute_batch (ShadingContext *ctx, ShaderGroup &group,
							ShaderGlobalsBatch &sgb, bool run)
{
//	std::cout << "execute_batch = ";
//	sgb.dump();
//	std::cout << std::endl;
    bool free_context = false;
    if (! ctx) {
        ctx = get_context();
        free_context = true;
    }
    bool result = ctx->execute_batch (group, sgb, run);
    if (free_context)
        release_context (ctx);
    return result;
}
	

const void *
ShadingSystemImpl::get_symbol (ShadingContext &ctx, ustring layername,
                               ustring symbolname, TypeDesc &type)
{
    const Symbol *sym = ctx.symbol (layername, symbolname);
    if (sym) {
        type = sym->typespec().simpletype();
        return ctx.symbol_data (*sym);
    } else {
        return NULL;
    }
}



int
ShadingSystemImpl::find_named_layer_in_group (ustring layername,
                                              ShaderInstance * &inst)
{
    inst = NULL;
    if (m_group_use >= ShadUseUnknown)
        return -1;
    ShaderGroup &group (*m_curgroup);
    for (int i = 0;  i < group.nlayers();  ++i) {
        if (group[i]->layername() == layername) {
            inst = group[i];
            return i;
        }
    }
    return -1;
}



ConnectedParam
ShadingSystemImpl::decode_connected_param (string_view connectionname,
                                   string_view layername, ShaderInstance *inst)
{
    ConnectedParam c;  // initializes to "invalid"

    // Look for a bracket in the "parameter name"
    size_t bracketpos = connectionname.find ('[');
    const char *bracket = bracketpos == string_view::npos ? NULL
                                   : connectionname.data()+bracketpos;
    // Grab just the part of the param name up to the bracket
    ustring param (connectionname, 0, bracketpos);

    // Search for the param with that name, fail if not found
    c.param = inst->findsymbol (param);
    if (c.param < 0) {
        if (connection_error())
            error ("ConnectShaders: \"%s\" is not a parameter or global of layer \"%s\" (shader \"%s\")",
                   param.c_str(), layername, inst->shadername().c_str());
        else
            warning ("ConnectShaders: \"%s\" is not a parameter or global of layer \"%s\" (shader \"%s\")",
                     param.c_str(), layername, inst->shadername().c_str());
        return c;
    }

    const Symbol *sym = inst->mastersymbol (c.param);
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

    if (bracket && c.type.is_array()) {
        // There was at least one set of brackets that appears to be
        // selecting an array element.
        c.arrayindex = atoi (bracket+1);
        if (c.arrayindex >= c.type.arraylength()) {
            error ("ConnectShaders: cannot request array element %s from a %s",
                   connectionname, c.type.c_str());
            c.arrayindex = c.type.arraylength() - 1;  // clamp it
        }
        c.type.make_array (0);              // chop to the element type
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



int
ShadingSystemImpl::raytype_bit (ustring name)
{
    for (size_t i = 0, e = m_raytypes.size();  i < e;  ++i)
        if (name == m_raytypes[i])
            return (1 << i);
    return 0;  // not found
}




bool
ShadingSystemImpl::is_renderer_output (ustring layername, ustring paramname,
                                       ShaderGroup *group) const
{
    if (group) {
        const std::vector<ustring> &aovs (group->m_renderer_outputs);
        if (std::find (aovs.begin(), aovs.end(), paramname) != aovs.end())
            return true;
        // Try "layer.name"
        ustring name2 = ustring::format ("%s.%s", layername, paramname);
        if (std::find (aovs.begin(), aovs.end(), name2) != aovs.end())
            return true;
    }
    const std::vector<ustring> &aovs (m_renderer_outputs);
    if (std::find (aovs.begin(), aovs.end(), paramname) != aovs.end())
        return true;
    ustring name2 = ustring::format ("%s.%s", layername, paramname);
    if (std::find (aovs.begin(), aovs.end(), name2) != aovs.end())
        return true;
    return false;
}



void
ShadingSystemImpl::group_post_jit_cleanup (ShaderGroup &group)
{
	OSL_DEV_ONLY(std::cout << "ShadingSystemImpl::group_post_jit_cleanup (ShaderGroup &group)" << std::endl);
    // Once we're generated the IR, we really don't need the ops and args,
    // and we only need the syms that include the params.
    off_t symmem = 0;
    size_t connectionmem = 0;
    for (int layer = 0;  layer < group.nlayers();  ++layer) {
        ShaderInstance *inst = group[layer];
        // We no longer needs ops and args -- create empty vectors and
        // swap with the ones in the instance.
        OpcodeVec emptyops;
        inst->ops().swap (emptyops);
        std::vector<int> emptyargs;
        inst->args().swap (emptyargs);
        if (inst->unused()) {
            // If we'll never use the layer, we don't need the syms at all
            SymbolVec nosyms;
            std::swap (inst->symbols(), nosyms);
            symmem += vectorbytes(nosyms);
            // also don't need the connection info any more
            connectionmem += (off_t) inst->clear_connections ();
        }
    }
    {
        // adjust memory stats
        spin_lock lock (m_stat_mutex);
        m_stat_mem_inst_syms -= symmem;
        m_stat_mem_inst_connections -= connectionmem;
        m_stat_mem_inst -= symmem + connectionmem;
        m_stat_memory -= symmem + connectionmem;
    }
}



void
ShadingSystemImpl::optimize_group (ShaderGroup &group,
                                   int raytypes_on, int raytypes_off, PerThreadInfo *threadinfo)
{
    if (group.optimized())
        return;    // already optimized

    OIIO::Timer timer;
    lock_guard lock (group.m_mutex);
    if (group.optimized()) {
        // The group was somehow optimized by another thread between the
        // time we checked group.optimized() and now that we have the lock.
        // Nothing to do but record how long we waited for the lock.
        spin_lock stat_lock (m_stat_mutex);
        double t = timer();
        m_stat_optimization_time += t;
        m_stat_opt_locking_time += t;
        return;
    }

    if (m_only_groupname && m_only_groupname != group.name()) {
        // For debugging purposes, we are requested to compile only one
        // shader group, and this is not it.  Mark it as does_nothing,
        // and also as optimized so nobody locks on it again, and record
        // how long we waited for the lock.
        group.does_nothing (true);
        group.m_optimized = true;
        spin_lock stat_lock (m_stat_mutex);
        double t = timer();
        m_stat_optimization_time += t;
        m_stat_opt_locking_time += t;
        return;
    }

    OSL_DEV_ONLY(std::cout << "ShadingSystemImpl::optimize_group" << std::endl);
    
    double locking_time = timer();

    ShadingContext *ctx = get_context (threadinfo);
    RuntimeOptimizer rop (*this, group, ctx);
    rop.set_raytypes (raytypes_on, raytypes_off);
    rop.run ();

    // Copy some info recorted by the RuntimeOptimizer into the group
    group.m_unknown_textures_needed = rop.m_unknown_textures_needed;
    for (auto&& f : rop.m_textures_needed)
        group.m_textures_needed.push_back (f);
    group.m_unknown_closures_needed = rop.m_unknown_closures_needed;
    for (auto&& f : rop.m_closures_needed)
        group.m_closures_needed.push_back (f);
    for (auto&& f : rop.m_globals_needed)
        group.m_globals_needed.push_back (f);
    size_t num_userdata = rop.m_userdata_needed.size();
    group.m_userdata_names.reserve (num_userdata);
    group.m_userdata_types.reserve (num_userdata);
    group.m_userdata_offsets.resize (num_userdata, 0);
    group.m_userdata_derivs.reserve (num_userdata);
    for (auto&& n : rop.m_userdata_needed) {
        group.m_userdata_names.push_back (n.name);
        group.m_userdata_types.push_back (n.type);
        group.m_userdata_derivs.push_back (n.derivs);
    }
    group.m_unknown_attributes_needed = rop.m_unknown_attributes_needed;
    for (auto&& f : rop.m_attributes_needed) {
        group.m_attributes_needed.push_back (f.name);
        group.m_attribute_scopes.push_back (f.scope);
    }

    release_context (ctx);

    group.m_optimized = true;
    spin_lock stat_lock (m_stat_mutex);
    m_stat_optimization_time += timer();
    m_stat_opt_locking_time += locking_time + rop.m_stat_opt_locking_time;
    m_stat_specialization_time += rop.m_stat_specialization_time;
}

void
ShadingSystemImpl::jit_group (ShaderGroup &group, PerThreadInfo *threadinfo)
{
    if (group.jitted())
        return;    // already optimized
    
    if (!group.optimized())
        optimize_group (group,
                        0, // raytypes_on
                        0, // raytypes_off
                        threadinfo);

    OIIO::Timer timer;
    // TODO: we could have separate mutexes for jit vs. batched_jit
    // choose to keep it simple to start with
    lock_guard lock (group.m_mutex);
    if (group.jitted()) {
        // The group was somehow jitted by another thread between the
        // time we checked group.jitted() and now that we have the lock.
        // Nothing to do (expect maybe record how long we waited for the lock).
        spin_lock stat_lock (m_stat_mutex);
        double t = timer();
        m_stat_optimization_time += t;
        m_stat_opt_locking_time += t;
        return;
    }
	
    ShadingContext *ctx = get_context (threadinfo);
    BackendLLVM lljitter (*this, group, ctx);
    lljitter.run ();

	// Keep OSL instructions around in case someone 
    // wants the batch version jitted
    if (group.batch_jitted()) {
    	group_post_jit_cleanup (group);
    }

    release_context (ctx);

    group.m_jitted = true;
    spin_lock stat_lock (m_stat_mutex);   
    m_stat_total_llvm_time += lljitter.m_stat_total_llvm_time;
    m_stat_llvm_setup_time += lljitter.m_stat_llvm_setup_time;
    m_stat_llvm_irgen_time += lljitter.m_stat_llvm_irgen_time;
    m_stat_llvm_opt_time += lljitter.m_stat_llvm_opt_time;
    m_stat_llvm_jit_time += lljitter.m_stat_llvm_jit_time;
    m_stat_max_llvm_local_mem = std::max (m_stat_max_llvm_local_mem,
                                          lljitter.m_llvm_local_mem);
    
    // TODO: not sure how to count these given batched vs. not
    m_stat_groups_compiled += 1;
    m_stat_instances_compiled += group.nlayers();
    m_groups_to_compile_count -= 1;	
}

void
ShadingSystemImpl::batched_jit_group (ShaderGroup &group, PerThreadInfo *threadinfo)
{    
    if (group.batch_jitted())
        return;    // already optimized
    
    if (!group.optimized())
        optimize_group (group,
                        0, // raytypes_on
                        0, // raytypes_off
                        threadinfo);

    OIIO::Timer timer;
    // TODO: we could have separate mutexes for jit vs. batched_jit
    // choose to keep it simple to start with
    lock_guard lock (group.m_mutex);
    if (group.batch_jitted()) {
        // The group was somehow batch_jitted by another thread between the
        // time we checked group.batch_jitted() and now that we have the lock.
        // Nothing to do (expect maybe record how long we waited for the lock).
        spin_lock stat_lock (m_stat_mutex);
        double t = timer();
        m_stat_optimization_time += t;
        m_stat_opt_locking_time += t;
        return;
    }
	
    ShadingContext *ctx = get_context (threadinfo);
    BackendLLVMWide lljitter (*this, group, ctx);
    lljitter.run ();

	// Keep OSL instructions around in case someone 
    // wants the scalar version jitted
    if (group.jitted()) {
    	group_post_jit_cleanup (group);
    }

    release_context (ctx);

    group.m_batch_jitted = true;
    spin_lock stat_lock (m_stat_mutex);   
    m_stat_total_llvm_time += lljitter.m_stat_total_llvm_time;
    m_stat_llvm_setup_time += lljitter.m_stat_llvm_setup_time;
    m_stat_llvm_irgen_time += lljitter.m_stat_llvm_irgen_time;
    m_stat_llvm_opt_time += lljitter.m_stat_llvm_opt_time;
    m_stat_llvm_jit_time += lljitter.m_stat_llvm_jit_time;
    m_stat_max_llvm_local_mem = std::max (m_stat_max_llvm_local_mem,
                                          lljitter.m_llvm_local_mem);
    
    // TODO: not sure how to count these given batched vs. not
    m_stat_groups_compiled += 1;
    m_stat_instances_compiled += group.nlayers();
    m_groups_to_compile_count -= 1;	
}



static void optimize_all_groups_wrapper (ShadingSystemImpl *ss, int mythread, int totalthreads)
{
    ss->optimize_all_groups (1, mythread, totalthreads);
}

static void jit_all_groups_wrapper (ShadingSystemImpl *ss, int mythread, int totalthreads)
{
    ss->jit_all_groups (1, mythread, totalthreads);
}

static void batched_jit_all_groups_wrapper (ShadingSystemImpl *ss, int mythread, int totalthreads)
{
    ss->batched_jit_all_groups (1, mythread, totalthreads);
}



void
ShadingSystemImpl::optimize_all_groups (int nthreads, int mythread, int totalthreads)
{
    // Spawn a bunch of threads to do this in parallel -- just call this
    // routine again (with threads=1) for each thread.
    if (nthreads < 1)  // threads <= 0 means use all hardware available
        nthreads = std::min ((int)std::thread::hardware_concurrency(),
                             (int)m_groups_to_compile_count);
    if (nthreads > 1) {
        if (m_threads_currently_compiling)
            return;   // never mind, somebody else spawned the JIT threads
        OIIO::thread_group threads;
        m_threads_currently_compiling += nthreads;
        for (int t = 0;  t < nthreads;  ++t)
            threads.add_thread (new std::thread (optimize_all_groups_wrapper, this, t, nthreads));
        threads.join_all ();
        m_threads_currently_compiling -= nthreads;
        return;
    }

    // And here's the single thread case
    size_t ngroups = 0;
    {
        spin_lock lock (m_all_shader_groups_mutex);
        ngroups = m_all_shader_groups.size();
    }
    for (size_t i = 0;  i < ngroups;  ++i) {
        // Assign to threads based on mod of totalthreads
        if ((i % totalthreads) == (unsigned)mythread) {
            ShaderGroupRef group;
            {
                spin_lock lock (m_all_shader_groups_mutex);
                group = m_all_shader_groups[i].lock();
            }
            if (group)
                optimize_group (*group);
        }
    }
}

void
ShadingSystemImpl::jit_all_groups (int nthreads, int mythread, int totalthreads)
{
    // Spawn a bunch of threads to do this in parallel -- just call this
    // routine again (with threads=1) for each thread.
    if (nthreads < 1)  // threads <= 0 means use all hardware available
        nthreads = std::min ((int)std::thread::hardware_concurrency(),
                             (int)m_groups_to_compile_count);
    if (nthreads > 1) {
        if (m_threads_currently_compiling)
            return;   // never mind, somebody else spawned the JIT threads
        OIIO::thread_group threads;
        m_threads_currently_compiling += nthreads;
        for (int t = 0;  t < nthreads;  ++t)
            threads.add_thread (new std::thread (jit_all_groups_wrapper, this, t, nthreads));
        threads.join_all ();
        m_threads_currently_compiling -= nthreads;
        return;
    }

    // And here's the single thread case
    size_t ngroups = 0;
    {
        spin_lock lock (m_all_shader_groups_mutex);
        ngroups = m_all_shader_groups.size();
    }
    for (size_t i = 0;  i < ngroups;  ++i) {
        // Assign to threads based on mod of totalthreads
        if ((i % totalthreads) == (unsigned)mythread) {
            ShaderGroupRef group;
            {
                spin_lock lock (m_all_shader_groups_mutex);
                group = m_all_shader_groups[i].lock();
            }
            if (group)
                jit_group (*group);
        }
    }
}

void
ShadingSystemImpl::batched_jit_all_groups (int nthreads, int mythread, int totalthreads)
{
    // Spawn a bunch of threads to do this in parallel -- just call this
    // routine again (with threads=1) for each thread.
    if (nthreads < 1)  // threads <= 0 means use all hardware available
        nthreads = std::min ((int)std::thread::hardware_concurrency(),
                             (int)m_groups_to_compile_count);
    if (nthreads > 1) {
        if (m_threads_currently_compiling)
            return;   // never mind, somebody else spawned the JIT threads
        OIIO::thread_group threads;
        m_threads_currently_compiling += nthreads;
        for (int t = 0;  t < nthreads;  ++t)
            threads.add_thread (new std::thread (batched_jit_all_groups_wrapper, this, t, nthreads));
        threads.join_all ();
        m_threads_currently_compiling -= nthreads;
        return;
    }

    // And here's the single thread case
    size_t ngroups = 0;
    {
        spin_lock lock (m_all_shader_groups_mutex);
        ngroups = m_all_shader_groups.size();
    }
    for (size_t i = 0;  i < ngroups;  ++i) {
        // Assign to threads based on mod of totalthreads
        if ((i % totalthreads) == (unsigned)mythread) {
            ShaderGroupRef group;
            {
                spin_lock lock (m_all_shader_groups_mutex);
                group = m_all_shader_groups[i].lock();
            }
            if (group)
                batched_jit_group (*group);
        }
    }
}


int
ShadingSystemImpl::merge_instances (ShaderGroup &group, bool post_opt)
{
    // Look through the shader group for pairs of nodes/layers that
    // actually do exactly the same thing, and eliminate one of the
    // rundantant shaders, carefully rewiring all its outgoing
    // connections to later layers to refer to the one we keep.
    //
    // It turns out that in practice, it's not uncommon to have
    // duplicate nodes.  For example, some materials are "layered" --
    // like a character skin shader that has separate sub-networks for
    // skin, oil, wetness, and so on -- and those different sub-nets
    // often reference the same texture maps or noise functions by
    // repetition.  Yes, ideally, the redundancies would be eliminated
    // before they were fed to the renderer, but in practice that's hard
    // and for many scenes we get substantial savings of time (mostly
    // because of reduced texture calls) and instance memory by finding
    // these redundancies automatically.  The amount of savings is quite
    // scene dependent, as well as probably very dependent on the
    // general shading and lookdev approach of the studio.  But it was
    // very helpful for us in many cases.
    //
    // The basic loop below looks very inefficient, O(n^2) in number of
    // instances in the group. But it's really not -- a few seconds (sum
    // of all threads) for even our very complex scenes. This is because
    // most potential pairs have a very fast rejection case if they are
    // not using the same master.  Since there's no appreciable cost to
    // the brute force approach, it seems silly to have a complex scheme
    // to try to reduce the number of pairings.

    if (! m_opt_merge_instances || optimize() < 1)
        return 0;

    OIIO::Timer timer;          // Time we spend looking for and doing merges
    int merges = 0;             // number of merges we do
    size_t connectionmem = 0;   // Connection memory we free
    int nlayers = group.nlayers();

    // Need to quickly make sure userdata_params is up to date before any
    // mergeability tests.
    for (int layer = 0;  layer < nlayers;  ++layer)
        if (! group[layer]->unused())
            group[layer]->evaluate_writes_globals_and_userdata_params ();

    // Loop over all layers...
    for (int a = 0;  a < nlayers-1;  ++a) {
        if (group[a]->unused())    // Don't merge a layer that's not used
            continue;
        // Check all later layers...
        for (int b = a+1;  b < nlayers;  ++b) {
            if (group[b]->unused())    // Don't merge a layer that's not used
                continue;
            if (b == nlayers-1)   // Don't merge the last layer -- causes
                continue;         // many tears because it's the group entry

            // Now we have two used layers, a and b, to examine.
            // See if they are mergeable (identical).  All the heavy
            // lifting is done by ShaderInstance::mergeable().
            if (! group[a]->mergeable (*group[b], group))
                continue;

            // The two nodes a and b are mergeable, so merge them.
            ShaderInstance *A = group[a];
            ShaderInstance *B = group[b];
            ++merges;

            // We'll keep A, get rid of B.  For all layers later than B,
            // check its incoming connections and replace all references
            // to B with references to A.
            for (int j = b+1;  j < nlayers;  ++j) {
                ShaderInstance *inst = group[j];
                if (inst->unused())  // don't bother if it's unused
                    continue;
                for (int c = 0, ce = inst->nconnections();  c < ce;  ++c) {
                    Connection &con = inst->connection(c);
                    if (con.srclayer == b) {
                        con.srclayer = a;
                        A->outgoing_connections (true);
                        if (A->symbols().size() && B->symbols().size()) {
                            ASSERT (A->symbol(con.src.param)->name() ==
                                    B->symbol(con.src.param)->name());
                        }
                    }
                }
            }

            // Mark parameters of B as no longer connected
            for (int p = B->firstparam();  p < B->lastparam();  ++p) {
                if (B->symbols().size())
                    B->symbol(p)->connected_down(false);
                if (B->m_instoverrides.size())
                    B->instoverride(p)->connected_down(false);
            }
            // B won't be used, so mark it as having no outgoing
            // connections and clear its incoming connections (which are
            // no longer used).
            ASSERT (B->merged_unused() == false);
            B->outgoing_connections (false);
            connectionmem += B->clear_connections ();
            B->m_merged_unused = true;
            ASSERT (B->unused());
        }
    }

    {
        // Adjust stats
        spin_lock lock (m_stat_mutex);
        m_stat_mem_inst_connections -= connectionmem;
        m_stat_mem_inst -= connectionmem;
        m_stat_memory -= connectionmem;
        if (post_opt)
            m_stat_merged_inst_opt += merges;
        else
            m_stat_merged_inst += merges;
        m_stat_inst_merge_time += timer();
    }

    return merges;
}



bool
ShadingSystemImpl::archive_shadergroup (ShaderGroup *group, string_view filename)
{
    std::string filename_base = OIIO::Filesystem::filename(filename);
    std::string extension;
    for (std::string e = OIIO::Filesystem::extension(filename);
         e.size() && filename.size();
         e = OIIO::Filesystem::extension(filename)) {
        extension = e + extension;
        filename.remove_suffix (e.size());
    }
    if (extension.size() < 2 || extension[0] != '.') {
        error ("archive_shadergroup: invalid filename \"%s\"", filename);
        return false;
    }
    filename_base.erase (filename_base.size() - extension.size());

    std::string pattern = OIIO::Filesystem::temp_directory_path() + "/OSL-%%%%-%%%%";
    if (! pattern.size()) {
        error ("archive_shadergroup: Could not find a temp directory");
        return false;
    }
    std::string tmpdir = OIIO::Filesystem::unique_path(pattern);
    if (! pattern.size()) {
        error ("archive_shadergroup: Could not find a temp filename");
        return false;
    }
    std::string errmessage;
    bool dir_ok = OIIO::Filesystem::create_directory (tmpdir, errmessage);
    if (! dir_ok) {
        error ("archive_shadergroup: Could not create temp directory %s (%s)",
               tmpdir, errmessage);
        return false;
    }

    bool ok = true;
    std::string groupfilename = tmpdir + "/shadergroup";
    std::ofstream groupfile;
    OIIO::Filesystem::open(groupfile, groupfilename);
    if (groupfile.good()) {
        groupfile << group->serialize();
        groupfile.close ();
    } else {
        error ("archive_shadergroup: Could not open shadergroup file");
        ok = false;
    }

    std::string filename_list = "shadergroup";
    {
        std::lock_guard<ShaderGroup> lock (*group);
        std::set<std::string> entries;   // to avoid duplicates
        for (int i = 0, nl = group->nlayers(); i < nl; ++i) {
            std::string osofile = (*group)[i]->master()->osofilename();
            std::string osoname = OIIO::Filesystem::filename (osofile);
            if (entries.find(osoname) == entries.end()) {
                entries.insert (osoname);
                std::string localfile = tmpdir + "/" + osoname;
                OIIO::Filesystem::copy (osofile, localfile);
                filename_list += " " + osoname;
            }
        }
    }

    if (extension == ".tar" || extension == ".tar.gz" || extension == ".tgz") {
        std::string z = Strutil::ends_with (extension, "gz") ? "-z" : "";
        std::string cmd = Strutil::format ("tar -c %s -C %s -f %s%s %s",
                                           z, tmpdir, filename, extension,
                                           filename_list);
        // std::cout << "Command =\n" << cmd << "\n";
        if (system (cmd.c_str()) != 0) {
            error ("archive_shadergroup: executing tar command failed");
            ok = false;
        }

    } else if (extension == ".zip") {
        std::string cmd = Strutil::format ("zip -q %s%s %s",
                                           filename, extension,
                                           filename_list);
        // std::cout << "Command =\n" << cmd << "\n";
        if (system (cmd.c_str()) != 0) {
            error ("archive_shadergroup: executing zip command failed");
            ok = false;
        }
    } else {
        error ("archive_shadergroup: no archiving/compressing command");
        ok = false;
    }

    OIIO::Filesystem::remove_all (tmpdir);

    return ok;
}



void
ClosureRegistry::register_closure (string_view name, int id,
                                   const ClosureParam *params,
                                   PrepareClosureFunc prepare,
                                   SetupClosureFunc setup,
                                   int alignment)
{
    if (m_closure_table.size() <= (size_t)id)
        m_closure_table.resize(id + 1);
    ClosureEntry &entry = m_closure_table[id];
    entry.id = id;
    entry.name = name;
    entry.nformal = 0;
    entry.nkeyword = 0;
    entry.struct_size = 0; /* params could be NULL */
    for (int i = 0; params; ++i) {
        /* always push so the end marker is there */
        entry.params.push_back(params[i]);
        if (params[i].type == TypeDesc()) {
            entry.struct_size = params[i].offset;
            break;
        }
        if (params[i].key == NULL)
            entry.nformal ++;
        else
            entry.nkeyword ++;
    }
    entry.prepare = prepare;
    entry.setup = setup;
    entry.alignment = alignment;
    m_closure_name_to_id[ustring(name)] = id;
}



const ClosureRegistry::ClosureEntry *
ClosureRegistry::get_entry(ustring name) const
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
OSL_NAMESPACE_EXIT



bool
OSL::OSLQuery::init (const ShaderGroup *group, int layernum)
{
    geterror();   // clear the error, we're newly initializing
    if (! group) {
        error ("No group pointer supplied.");
        return false;
    }
    if (layernum < 0 || layernum >= group->nlayers()) {
        error ("Invalid layer number %d (valid indices: 0-%d).",
               layernum, group->nlayers()-1);
        return false;
    }

    const ShaderMaster *master = (*group)[layernum]->master();
    m_shadername = master->shadername();
    m_shadertypename = master->shadertypename();
    m_params.clear();
    if (int nparams = master->num_params()) {
        m_params.resize (nparams);
        for (int i = 0;  i < nparams;  ++i) {
            const Symbol *sym = master->symbol (i);
            Parameter &p (m_params[i]);
            p.name = sym->name().string();
            const TypeSpec &ts (sym->typespec());
            p.type = ts.simpletype();
            p.isoutput = (sym->symtype() == SymTypeOutputParam);
            p.varlenarray = ts.is_unsized_array();
            p.isstruct = ts.is_structure() || ts.is_structure_array();
            p.isclosure = ts.is_closure_based();
            p.data = sym->data();
            // In this mode, we don't fill in idefault, fdefault, sdefault,
            // or spacename.
            p.idefault.clear();
            p.fdefault.clear();
            p.sdefault.clear();
            p.spacename.clear();
            p.fields.clear();  // don't bother filling this out
            if (StructSpec *ss = ts.structspec()) {
                p.structname = ss->name().string();
                for (size_t i = 0, e = ss->numfields();  i < e;  ++i)
                    p.fields.push_back (ss->field(i).name);
            } else {
                p.structname.clear();
            }
            p.metadata.clear();   // FIXME?
            p.validdefault = (p.data != NULL);
        }
    }

    m_meta.clear();   // no metadata available at this point

    return false;
}



// vals points to a symbol with a total of ncomps floats (ncomps ==
// aggregate*arraylen).  If has_derivs is true, it's actually 3 times
// that length, the main values then the derivatives.  We want to check
// for nans in vals[firstcheck..firstcheck+nchecks-1], and also in the
// derivatives if present.  Note that if firstcheck==0 and nchecks==ncomps,
// we are checking the entire contents of the symbol.  More restrictive
// firstcheck,nchecks are used to check just one element of an array.
OSL_SHADEOP void
osl_naninf_check (int ncomps, const void *vals_, int has_derivs,
                  void *sg, const void *sourcefile, int sourceline,
                  void *symbolname, int firstcheck, int nchecks,
                  const void *opname)
{
    ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
    const float *vals = (const float *)vals_;
    for (int d = 0;  d < (has_derivs ? 3 : 1);  ++d) {
        for (int c = firstcheck, e = c+nchecks; c < e;  ++c) {
            int i = d*ncomps + c;
            if (! OIIO::isfinite(vals[i])) {
                ctx->error ("Detected %g value in %s%s at %s:%d (op %s)",
                            vals[i], d > 0 ? "the derivatives of " : "",
                            USTR(symbolname), USTR(sourcefile), sourceline,
                            USTR(opname));
                return;
            }
        }
    }
}

// vals points to a symbol with a total of ncomps floats (ncomps ==
// aggregate*arraylen).  If has_derivs is true, it's actually 3 times
// that length, the main values then the derivatives.  We want to check
// for nans in vals[firstcheck..firstcheck+nchecks-1], and also in the
// derivatives if present.  Note that if firstcheck==0 and nchecks==ncomps,
// we are checking the entire contents of the symbol.  More restrictive
// firstcheck,nchecks are used to check just one element of an array.
OSL_SHADEOP void
osl_naninf_check_batched (
                  int ncomps, const void *vals_, int has_derivs,
                  void *sgb, const void *sourcefile, int sourceline,
                  void *symbolname, int firstcheck, int nchecks,
                  const void *opname)
{
    ShadingContext *ctx = (ShadingContext *)((ShaderGlobalsBatch *)sgb)->uniform().context;
    const float *vals = (const float *)vals_;
    for (int d = 0;  d < (has_derivs ? 3 : 1);  ++d) {
        for (int c = firstcheck, e = c+nchecks; c < e;  ++c) {
            int i = d*ncomps + c;
            if (! OIIO::isfinite(vals[i])) {
                ctx->error ("Detected %g value in %s%s at %s:%d (op %s)",
                            vals[i], d > 0 ? "the derivatives of " : "",
                            USTR(symbolname), USTR(sourcefile), sourceline,
                            USTR(opname));
                return;
            }
        }
    }
}

// Wide vals + mask, but uniform index
OSL_SHADEOP void
osl_naninf_check_u_offset_masked (int mask_value,
                  int ncomps, const void *vals_, int has_derivs,
                  void *sgb, const void *sourcefile, int sourceline,
                  void *symbolname, int firstcheck, int nchecks,
                  const void *opname)
{
    ShadingContext *ctx = (ShadingContext *)((ShaderGlobalsBatch *)sgb)->uniform().context;
    const float *vals = (const float *)vals_;
    const Mask mask(mask_value);
    for (int d = 0;  d < (has_derivs ? 3 : 1);  ++d) {
        for (int c = firstcheck, e = c+nchecks; c < e;  ++c) {
            int i = d*ncomps + c;
            for(int lane = 0; lane < SimdLaneCount; ++lane) {
                if (mask[lane]) {
                    if (! OIIO::isfinite(vals[i*SimdLaneCount + lane])) {
                        ctx->error ("Detected %g value in %s%s at %s:%d (op %s) batch lane:%d",
                                    vals[i*SimdLaneCount + lane], d > 0 ? "the derivatives of " : "",
                                    USTR(symbolname), USTR(sourcefile), sourceline,
                                    USTR(opname), lane);
                        return;
                    }
                }
            }
        }
    }
}

// Wide vals + mask + varying index
OSL_SHADEOP void
osl_naninf_check_w16_offset_masked (int mask_value,
                  int ncomps, const void *vals_, int has_derivs,
                  void *sgb, const void *sourcefile, int sourceline,
                  void *symbolname, const void * wide_offsets_ptr,
                  int nchecks, const void *opname)
{
    ShadingContext *ctx = (ShadingContext *)((ShaderGlobalsBatch *)sgb)->uniform().context;
    ConstWideAccessor<int> wOffsets(wide_offsets_ptr);
    const Mask mask(mask_value);

    const float *vals = (const float *)vals_;
    for (int d = 0;  d < (has_derivs ? 3 : 1);  ++d) {
        for(int lane = 0; lane < SimdLaneCount; ++lane) {
            if (mask[lane]) {
                int firstcheck = wOffsets[lane];
                for (int c = firstcheck, e = c+nchecks; c < e;  ++c) {
                    int i = d*ncomps + c;
                    if (! OIIO::isfinite(vals[i*SimdLaneCount + lane])) {
                        ctx->error ("Detected %g value in %s%s at %s:%d (op %s) batch lane:%d",
                                    vals[i*SimdLaneCount + lane], d > 0 ? "the derivatives of " : "",
                                    USTR(symbolname), USTR(sourcefile), sourceline,
                                    USTR(opname), lane);
                        return;
                    }
                }
            }
        }
    }
}

// vals points to the data of a float-, int-, or string-based symbol.
// (described by typedesc).  We want to check
// vals[firstcheck..firstcheck+nchecks-1] for floats that are NaN , or
// ints that are -MAXINT, or strings that are "!!!uninitialized!!!"
// which would indicate that the value is uninitialized if
// 'debug_uninit' is turned on.  Note that if firstcheck==0 and
// nchecks==ncomps, we are checking the entire contents of the symbol.
// More restrictive firstcheck,nchecks are used to check just one
// element of an array.
OSL_SHADEOP void
osl_uninit_check (long long typedesc_, void *vals_,
                  void *sg, const void *sourcefile, int sourceline,
                  const char *groupname, int layer, const char *layername,
                  const char *shadername,
                  int opnum, const char *opname, int argnum,
                  void *symbolname, int firstcheck, int nchecks)
{
    TypeDesc typedesc = TYPEDESC(typedesc_);
    ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
    bool uninit = false;
    if (typedesc.basetype == TypeDesc::FLOAT) {
        float *vals = (float *)vals_;
        for (int c = firstcheck, e = firstcheck+nchecks; c < e;  ++c)
            if (!OIIO::isfinite(vals[c])) {
                uninit = true;
                vals[c] = 0;
            }
    }
    if (typedesc.basetype == TypeDesc::INT) {
        int *vals = (int *)vals_;
        for (int c = firstcheck, e = firstcheck+nchecks; c < e;  ++c)
            if (vals[c] == std::numeric_limits<int>::min()) {
                uninit = true;
                vals[c] = 0;
            }
    }
    if (typedesc.basetype == TypeDesc::STRING) {
        ustring *vals = (ustring *)vals_;
        for (int c = firstcheck, e = firstcheck+nchecks; c < e;  ++c)
            if (vals[c] == Strings::uninitialized_string) {
                uninit = true;
                vals[c] = ustring();
            }
    }
    if (uninit) {
        ctx->error ("Detected possible use of uninitialized value in %s %s at %s:%d (group %s, layer %d %s, shader %s, op %d '%s', arg %d)",
                    typedesc, USTR(symbolname), USTR(sourcefile), sourceline,
                    (groupname && groupname[0]) ? groupname: "<unnamed group>",
                    layer, (layername && layername[0]) ? layername : "<unnamed layer>",
                    shadername, opnum, USTR(opname), argnum);
    }
}

OSL_SHADEOP void
osl_uninit_check_u_values_u_offset_batched (long long typedesc_, void *vals_,
                  void *sgb, const void *sourcefile, int sourceline,
                  const char *groupname, int layer, const char *layername,
                  const char *shadername,
                  int opnum, const char *opname, int argnum,
                  void *symbolname, int firstcheck, int nchecks)
{
    TypeDesc typedesc = TYPEDESC(typedesc_);
    ShadingContext *ctx = (ShadingContext *)((ShaderGlobalsBatch *)sgb)->uniform().context;
    bool uninit = false;
    if (typedesc.basetype == TypeDesc::FLOAT) {
        float *vals = (float *)vals_;
        for (int c = firstcheck, e = firstcheck+nchecks; c < e;  ++c)
            if (!OIIO::isfinite(vals[c])) {
                uninit = true;
                vals[c] = 0;
            }
    }
    if (typedesc.basetype == TypeDesc::INT) {
        int *vals = (int *)vals_;
        for (int c = firstcheck, e = firstcheck+nchecks; c < e;  ++c)
            if (vals[c] == std::numeric_limits<int>::min()) {
                uninit = true;
                vals[c] = 0;
            }
    }
    if (typedesc.basetype == TypeDesc::STRING) {
        ustring *vals = (ustring *)vals_;
        for (int c = firstcheck, e = firstcheck+nchecks; c < e;  ++c)
            if (vals[c] == Strings::uninitialized_string) {
                uninit = true;
                vals[c] = ustring();
            }
    }
    if (uninit) {
        ctx->error ("Detected possible use of uninitialized value in %s %s at %s:%d (group %s, layer %d %s, shader %s, op %d '%s', arg %d)",
                    typedesc, USTR(symbolname), USTR(sourcefile), sourceline,
                    (groupname && groupname[0]) ? groupname: "<unnamed group>",
                    layer, (layername && layername[0]) ? layername : "<unnamed layer>",
                    shadername, opnum, USTR(opname), argnum);
    }
}

OSL_SHADEOP void
osl_uninit_check_w16_values_u_offset_masked (int mask_value,
                  long long typedesc_, void *vals_,
                  void *sgb, const void *sourcefile, int sourceline,
                  const char *groupname, int layer, const char *layername,
                  const char *shadername,
                  int opnum, const char *opname, int argnum,
                  void *symbolname, int firstcheck, int nchecks)
{
    TypeDesc typedesc = TYPEDESC(typedesc_);
    ShadingContext *ctx = (ShadingContext *)((ShaderGlobalsBatch *)sgb)->uniform().context;
    const Mask mask(mask_value);

    //std::cout << "osl_uninit_check_w16_values_u_offset_masked="<< mask_value << std::endl;
    Mask lanes_uninit(false);

    if (typedesc.basetype == TypeDesc::FLOAT) {
        float *vals = (float *)vals_;
        for (int c = firstcheck, e = firstcheck+nchecks; c < e;  ++c)
            for(int lane = 0; lane < SimdLaneCount; ++lane) {
                if (mask[lane]) {
                    if (!OIIO::isfinite(vals[c*SimdLaneCount + lane])) {
                        lanes_uninit.set_on(lane);
                        vals[c*SimdLaneCount + lane] = 0;
                    }
                }
            }
    }
    if (typedesc.basetype == TypeDesc::INT) {
        int *vals = (int *)vals_;
        for (int c = firstcheck, e = firstcheck+nchecks; c < e;  ++c)
            for(int lane = 0; lane < SimdLaneCount; ++lane) {
                if (mask[lane]) {
                    if (vals[c*SimdLaneCount + lane] == std::numeric_limits<int>::min()) {
                        lanes_uninit.set_on(lane);
                        vals[c*SimdLaneCount + lane] = 0;
                    }
                }
            }
    }
    if (typedesc.basetype == TypeDesc::STRING) {
        ustring *vals = (ustring *)vals_;
        for (int c = firstcheck, e = firstcheck+nchecks; c < e;  ++c)
            for(int lane = 0; lane < SimdLaneCount; ++lane) {
                if (mask[lane]) {
                    if (vals[c*SimdLaneCount + lane] == Strings::uninitialized_string) {
                        lanes_uninit.set_on(lane);
                        vals[c*SimdLaneCount + lane] = ustring();
                    }
                }
            }
    }
    if (lanes_uninit.any_on()) {
        ctx->error ("Detected possible use of uninitialized value in %s %s at %s:%d (group %s, layer %d %s, shader %s, op %d '%s', arg %d) for lanes(%x) of batch",
                    typedesc, USTR(symbolname), USTR(sourcefile), sourceline,
                    (groupname && groupname[0]) ? groupname: "<unnamed group>",
                    layer, (layername && layername[0]) ? layername : "<unnamed layer>",
                    shadername, opnum, USTR(opname), argnum, lanes_uninit.value());
    }
}

OSL_SHADEOP void
osl_uninit_check_u_values_w16_offset_masked (int mask_value,
                  long long typedesc_, void *vals_,
                  void *sgb, const void *sourcefile, int sourceline,
                  const char *groupname, int layer, const char *layername,
                  const char *shadername,
                  int opnum, const char *opname, int argnum,
                  void *symbolname, const void * wide_offsets_ptr, int nchecks)
{
    TypeDesc typedesc = TYPEDESC(typedesc_);
    ShadingContext *ctx = (ShadingContext *)((ShaderGlobalsBatch *)sgb)->uniform().context;
    ConstWideAccessor<int> wOffsets(wide_offsets_ptr);
    const Mask mask(mask_value);
    //std::cout << "osl_uninit_check_u_values_w16_offset_masked="<< mask_value << std::endl;
    Mask lanes_uninit(false);
    if (typedesc.basetype == TypeDesc::FLOAT) {
        float *vals = (float *)vals_;
        for(int lane = 0; lane < SimdLaneCount; ++lane) {
            if (mask[lane]) {
                int firstcheck = wOffsets[lane];
                for (int c = firstcheck, e = firstcheck+nchecks; c < e;  ++c)
                    if (!OIIO::isfinite(vals[c])) {
                        lanes_uninit.set_on(lane);
                        vals[c] = 0;
                    }
            }
        }
    }
    if (typedesc.basetype == TypeDesc::INT) {
        int *vals = (int *)vals_;
        for(int lane = 0; lane < SimdLaneCount; ++lane) {
            if (mask[lane]) {
                int firstcheck = wOffsets[lane];
                for (int c = firstcheck, e = firstcheck+nchecks; c < e;  ++c)
                    if (vals[c] == std::numeric_limits<int>::min()) {
                        lanes_uninit.set_on(lane);
                        vals[c] = 0;
                    }
            }
        }
    }
    if (typedesc.basetype == TypeDesc::STRING) {
        ustring *vals = (ustring *)vals_;
        for(int lane = 0; lane < SimdLaneCount; ++lane) {
            if (mask[lane]) {
                int firstcheck = wOffsets[lane];
                for (int c = firstcheck, e = firstcheck+nchecks; c < e;  ++c)
                    if (vals[c] == Strings::uninitialized_string) {
                        lanes_uninit.set_on(lane);
                        vals[c] = ustring();
                    }
            }
        }
    }

    if (lanes_uninit.any_on()) {
        ctx->error ("Detected possible use of uninitialized value in %s %s at %s:%d (group %s, layer %d %s, shader %s, op %d '%s', arg %d) for lanes(%x) of batch",
                    typedesc, USTR(symbolname), USTR(sourcefile), sourceline,
                    (groupname && groupname[0]) ? groupname: "<unnamed group>",
                    layer, (layername && layername[0]) ? layername : "<unnamed layer>",
                    shadername, opnum, USTR(opname), argnum, lanes_uninit.value());
    }
}
OSL_SHADEOP void
osl_uninit_check_w16_values_w16_offset_masked (int mask_value,
                  long long typedesc_, void *vals_,
                  void *sgb, const void *sourcefile, int sourceline,
                  const char *groupname, int layer, const char *layername,
                  const char *shadername,
                  int opnum, const char *opname, int argnum,
                  void *symbolname, const void * wide_offsets_ptr, int nchecks)
{
    TypeDesc typedesc = TYPEDESC(typedesc_);
    ShadingContext *ctx = (ShadingContext *)((ShaderGlobalsBatch *)sgb)->uniform().context;
    ConstWideAccessor<int> wOffsets(wide_offsets_ptr);
    const Mask mask(mask_value);
    //std::cout << "osl_uninit_check_w16_values_w16_offset_masked="<< mask_value << std::endl;
    Mask lanes_uninit(false);
    if (typedesc.basetype == TypeDesc::FLOAT) {
        float *vals = (float *)vals_;
        for(int lane = 0; lane < SimdLaneCount; ++lane) {
            if (mask[lane]) {
                int firstcheck = wOffsets[lane];
                for (int c = firstcheck, e = firstcheck+nchecks; c < e;  ++c)
                    if (!OIIO::isfinite(vals[c*SimdLaneCount + lane])) {
                        lanes_uninit.set_on(lane);
                        vals[c*SimdLaneCount + lane] = 0;
                    }
            }
        }
    }
    if (typedesc.basetype == TypeDesc::INT) {
        int *vals = (int *)vals_;
        for(int lane = 0; lane < SimdLaneCount; ++lane) {
            if (mask[lane]) {
                int firstcheck = wOffsets[lane];
                for (int c = firstcheck, e = firstcheck+nchecks; c < e;  ++c)
                    if (vals[c*SimdLaneCount + lane] == std::numeric_limits<int>::min()) {
                        lanes_uninit.set_on(lane);
                        vals[c*SimdLaneCount + lane] = 0;
                    }
            }
        }
    }
    if (typedesc.basetype == TypeDesc::STRING) {
        ustring *vals = (ustring *)vals_;
        for(int lane = 0; lane < SimdLaneCount; ++lane) {
            if (mask[lane]) {
                int firstcheck = wOffsets[lane];
                for (int c = firstcheck, e = firstcheck+nchecks; c < e;  ++c)
                    if (vals[c*SimdLaneCount + lane] == Strings::uninitialized_string) {
                        lanes_uninit.set_on(lane);
                        vals[c*SimdLaneCount + lane] = ustring();
                    }
            }
        }
    }

    if (lanes_uninit.any_on()) {
        ctx->error ("Detected possible use of uninitialized value in %s %s at %s:%d (group %s, layer %d %s, shader %s, op %d '%s', arg %d) for lanes(%x) of batch",
                    typedesc, USTR(symbolname), USTR(sourcefile), sourceline,
                    (groupname && groupname[0]) ? groupname: "<unnamed group>",
                    layer, (layername && layername[0]) ? layername : "<unnamed layer>",
                    shadername, opnum, USTR(opname), argnum, lanes_uninit.value());
    }
}

OSL_SHADEOP int
osl_range_check (int indexvalue, int length, const char *symname,
                 void *sg, const void *sourcefile, int sourceline,
                 const char *groupname, int layer, const char *layername,
                 const char *shadername)
{
    if (indexvalue < 0 || indexvalue >= length) {
        ShadingContext *ctx = (ShadingContext *)((ShaderGlobals *)sg)->context;
        ctx->error ("Index [%d] out of range %s[0..%d]: %s:%d"
                    " (group %s, layer %d %s, shader %s)",
                    indexvalue, USTR(symname), length-1,
                    USTR(sourcefile), sourceline,
                    (groupname && groupname[0]) ? groupname : "<unnamed group>", layer,
                    (layername && layername[0]) ? layername : "<unnamed layer>",
                    USTR(shadername));
        if (indexvalue >= length)
            indexvalue = length-1;
        else
            indexvalue = 0;
    }
    return indexvalue;
}

OSL_SHADEOP int
osl_range_check_batched (int indexvalue, int length, const char *symname,
                 void *sgb, const void *sourcefile, int sourceline,
                 const char *groupname, int layer, const char *layername,
                 const char *shadername)
{
    if (indexvalue < 0 || indexvalue >= length) {
        ShadingContext *ctx = (ShadingContext *)((ShaderGlobalsBatch *)sgb)->uniform().context;
        ctx->error ("Index [%d] out of range %s[0..%d]: %s:%d"
                    " (group %s, layer %d %s, shader %s)",
                    indexvalue, USTR(symname), length-1,
                    USTR(sourcefile), sourceline,
                    (groupname && groupname[0]) ? groupname : "<unnamed group>", layer,
                    (layername && layername[0]) ? layername : "<unnamed layer>",
                    USTR(shadername));
        if (indexvalue >= length)
            indexvalue = length-1;
        else
            indexvalue = 0;
    }
    return indexvalue;
}

OSL_SHADEOP void
osl_range_check_masked (void * wide_indexvalue, int mask_value, int length, const char *symname,
                 void *sgb, const void *sourcefile, int sourceline,
                 const char *groupname, int layer, const char *layername,
                 const char *shadername)
{
    MaskedAccessor<int> wIndexValue(wide_indexvalue, Mask(mask_value));
    for(int lane = 0; lane < SimdLaneCount; ++lane) {
        if (wIndexValue.mask()[lane]) {
            int indexvalue = wIndexValue[lane];
            if (indexvalue < 0 || indexvalue >= length) {
                ShadingContext *ctx = (ShadingContext *)((ShaderGlobalsBatch *)sgb)->uniform().context;
                ctx->error ("Index [%d] out of range %s[0..%d]: %s:%d"
                            " (group %s, layer %d %s, shader %s)",
                            indexvalue, USTR(symname), length-1,
                            USTR(sourcefile), sourceline,
                            (groupname && groupname[0]) ? groupname : "<unnamed group>", layer,
                            (layername && layername[0]) ? layername : "<unnamed layer>",
                            USTR(shadername));
                if (indexvalue >= length)
                    indexvalue = length-1;
                else
                    indexvalue = 0;
                // modify index value so it is not out of bounds
                wIndexValue[lane] = indexvalue;
            }
        }
    }
}



// Asked if the raytype is a name we can't know until mid-shader.
OSL_SHADEOP int osl_raytype_name (void *sg_, void *name)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    int bit = sg->context->shadingsys().raytype_bit (USTR(name));
    return (sg->raytype & bit) != 0;
}


OSL_SHADEOP int osl_get_attribute(void *sg_,
                             int   dest_derivs,
                             void *obj_name_,
                             void *attr_name_,
                             int   array_lookup,
                             int   index,
                             const void *attr_type,
                             void *attr_dest)
{
    ShaderGlobals *sg   = (ShaderGlobals *)sg_;
    const ustring &obj_name  = USTR(obj_name_);
    const ustring &attr_name = USTR(attr_name_);

    return sg->context->osl_get_attribute (sg, sg->objdata,
                                           dest_derivs, obj_name, attr_name,
                                           array_lookup, index,
                                           *(const TypeDesc *)attr_type,
                                           attr_dest);
}

OSL_SHADEOP int osl_get_attribute_batched(void *sgb_,
                                           int   dest_derivs,
                                           void *obj_name_,
                                           void *attr_name_,
                                           int   array_lookup,
                                           int   index,
                                           const void *attr_type,
                                           void *wide_attr_dest,
                                           int mask_)
{
    Mask mask(mask_);
    ASSERT(mask.any_on());

    ShaderGlobalsBatch *sgb   = reinterpret_cast<ShaderGlobalsBatch *>(sgb_);
    const ustring &obj_name  = USTR(obj_name_);
    const ustring &attr_name = USTR(attr_name_);

    Mask retVal = sgb->uniform().context->osl_get_attribute_batched (sgb, sgb->uniform().objdata,
                                                       dest_derivs, obj_name, attr_name,
                                                       array_lookup, index,
                                                       *(const TypeDesc *)attr_type,
                                                       wide_attr_dest, mask);
    
    return retVal.value();
}

OSL_SHADEOP int osl_get_attribute_w16attr_name_batched(void *sgb_,
                                           int   dest_derivs,
                                           void *obj_name_,
                                           void *wattr_name_,
                                           int   array_lookup,
                                           int   index,
                                           const void *attr_type,
                                           void *wide_attr_dest,
                                           int mask_)
{
    Mask mask(mask_);
    ASSERT(mask.any_on());

    ShaderGlobalsBatch *sgb   = reinterpret_cast<ShaderGlobalsBatch *>(sgb_);
    const ustring &obj_name  = USTR(obj_name_);
    ConstWideAccessor<ustring> wAttrName(wattr_name_);


    Mask retVal(false);

    // We have a varying attribute name.
    // Lets find all the lanes with the same values and
    // make a call for each unique attr_name
    Mask uninspectedMask(mask);
    for(int inspectLane=0; inspectLane < mask.width; ++inspectLane)
    {
    	if (uninspectedMask[inspectLane]) {
    		const ustring attr_name = wAttrName[inspectLane];
    		// Identify any remaining lanes that might have the same attribute name
    		Mask lanesWithSameAttrName(false);
    		lanesWithSameAttrName.set_on(inspectLane);
    		for (int otherLane = inspectLane+1; otherLane < mask.width; ++otherLane)
    		{
    			const ustring otherAttrName = wAttrName[otherLane];
    			if (uninspectedMask[otherLane] && attr_name == otherAttrName) {
    				lanesWithSameAttrName.set_on(otherLane);
    			}
    		}

    	    Mask lanesPopulated = sgb->uniform().context->osl_get_attribute_batched (sgb, sgb->uniform().objdata,
    	                                                       dest_derivs, obj_name, attr_name,
    	                                                       array_lookup, index,
    	                                                       *(const TypeDesc *)attr_type,
    	                                                       wide_attr_dest, lanesWithSameAttrName);
    		uninspectedMask &= ~lanesWithSameAttrName;
    		retVal |= lanesPopulated;
    	}
    }

    return retVal.value();
}



OSL_SHADEOP bool osl_get_attribute_batched_uniform(void *sgb_,
                                           int   dest_derivs,
                                           void *obj_name_,
                                           void *attr_name_,
                                           int   array_lookup,
                                           int   index,
                                           const void *attr_type,
                                           void *attr_dest)
{
//    Mask mask(mask_);
//    // TODO: LLVM could check this before calling this function
//    if (mask.all_off()) {
//        return 0;
//    }
    
    ShaderGlobalsBatch *sgb   = reinterpret_cast<ShaderGlobalsBatch *>(sgb_);
    const ustring &obj_name  = USTR(obj_name_);
    const ustring &attr_name = USTR(attr_name_);

    bool success = sgb->uniform().context->osl_get_attribute_batched_uniform(
    		                                           sgb, sgb->uniform().objdata,
                                                       dest_derivs, obj_name, attr_name,
                                                       array_lookup, index,
                                                       *(const TypeDesc *)attr_type,
                                                       attr_dest);
    return success;
}

#ifdef OSL_EXPERIMENTAL_BIND_USER_DATA_WITH_LAYERNAME
OSL_SHADEOP int
osl_bind_interpolated_param (void *sg_, const void *name, const void *layername,
                             long long type, int userdata_has_derivs,
                             void *userdata_data, int symbol_has_derivs,
                             void *symbol_data, int symbol_data_size,
                             char *userdata_initialized, int userdata_index)
{
    // XXX: Disable the userdata cache for now, it does not correctly
    //      handle layername caching. Ideally userdata_initialized
    //      would be keyed from layername as well as name. -WLW
    //char status = *userdata_initialized;
    char status = 0;

    if (status == 0) {
        // First time retrieving this userdata
        ShaderGlobals *sg = (ShaderGlobals *)sg_;
        bool ok = sg->renderer->get_userdata (userdata_has_derivs, USTR(name),
                                              USTR(layername), TYPEDESC(type),
                                              sg, userdata_data);
        // printf ("Binding %s %s : index %d, ok = %d\n", name,
        //         TYPEDESC(type).c_str(),userdata_index, ok);
        *userdata_initialized = status = 1 + ok;  // 1 = not found, 2 = found
        sg->context->incr_get_userdata_calls ();
    }
    if (status == 2) {
        // If userdata was present, copy it to the shader variable
        memcpy (symbol_data, userdata_data, symbol_data_size);
        return 1;
    }
    return 0;  // no such user data
}
#else
OSL_SHADEOP int
osl_bind_interpolated_param (void *sg_, const void *name, long long type,
                             int userdata_has_derivs, void *userdata_data,
                             int symbol_has_derivs, void *symbol_data,
                             int symbol_data_size,
                             char *userdata_initialized, int userdata_index)
{
    char status = *userdata_initialized;
    if (status == 0) {
        // First time retrieving this userdata
        ShaderGlobals *sg = (ShaderGlobals *)sg_;
        bool ok = sg->renderer->get_userdata (userdata_has_derivs, USTR(name),
                                              TYPEDESC(type),
                                              sg, userdata_data);
        // printf ("Binding %s %s : index %d, ok = %d\n", name,
        //         TYPEDESC(type).c_str(),userdata_index, ok);
        *userdata_initialized = status = 1 + ok;  // 1 = not found, 2 = found
        sg->context->incr_get_userdata_calls ();
    }
    if (status == 2) {
        // If userdata was present, copy it to the shader variable
        memcpy (symbol_data, userdata_data, symbol_data_size);
        return 1;
    }
    return 0;  // no such user data
}
#endif

#ifdef OSL_EXPERIMENTAL_BIND_USER_DATA_WITH_LAYERNAME
OSL_SHADEOP int
osl_bind_interpolated_param_wide (void *sgb_, const void *name, const void *layername, long long type,
                             int userdata_has_derivs, void *userdata_data,
                             int symbol_has_derivs, void *symbol_data,
                             int symbol_data_size,
                             unsigned int *userdata_initialized, int userdata_index, int mask_value)
{
    // Top bit indicate if we have checked for user data yet or not
    // the bottom half is a mask of which lanes successfully retrieved
    // user data
    // XXX: Disable the userdata cache for now, it does not correctly
    //      handle layername caching. Ideally userdata_initialized
    //      would be keyed from layername as well as name. -SAF
    // int status = (*userdata_initialized)>>31;
    int status = 0;
    if (status == 0) {
        // First time retrieving this userdata
        ShaderGlobalsBatch *sgb   = reinterpret_cast<ShaderGlobalsBatch *>(sgb_);
        MaskedDataRef userDest(TYPEDESC(type), userdata_has_derivs, Mask(mask_value), userdata_data);
        Mask foundUserData = sgb->uniform().renderer->batched()->get_userdata (USTR(name),
                USTR(layername), sgb, userDest);
        // printf ("Binding [%s] %s %s : index %d, ok = %d\n", layername, name,
        //         TYPEDESC(type).c_str(),userdata_index, foundUserData.value());

        *userdata_initialized = (1<<31) | foundUserData.value();
        sgb->uniform().context->incr_get_userdata_calls ();
    }
    DASSERT((*userdata_initialized)>>31 == 1);
    Mask foundUserData(*userdata_initialized & 0x7FFFFFFF);
    if (foundUserData.any_on()) {
        // If userdata was present, copy it to the shader variable
        // Don't bother masking as any lanes without user data
        // will be overwritten by init ops or by default value
        memcpy (symbol_data, userdata_data, symbol_data_size);
    }

    return foundUserData.value();
}
#else
OSL_SHADEOP int
osl_bind_interpolated_param_wide (void *sgb_, const void *name, long long type,
                             int userdata_has_derivs, void *userdata_data,
                             int symbol_has_derivs, void *symbol_data,
                             int symbol_data_size,
                             unsigned int *userdata_initialized, int userdata_index, int mask_value)
{
    // Top bit indicate if we have checked for user data yet or not
    // the bottom half is a mask of which lanes successfully retrieved 
	// user data 
	int status = (*userdata_initialized)>>31;
    if (status == 0) {
        // First time retrieving this userdata
        ShaderGlobalsBatch *sgb   = reinterpret_cast<ShaderGlobalsBatch *>(sgb_);  
        MaskedDataRef userDest(TYPEDESC(type), userdata_has_derivs, Mask(mask_value), userdata_data);
        Mask foundUserData = sgb->uniform().renderer->batched()->get_userdata (USTR(name),
                sgb, userDest);
                                              
        // printf ("Binding %s %s : index %d, ok = %d\n", name,
        //         TYPEDESC(type).c_str(),userdata_index, foundUserData.value());
        
        *userdata_initialized = (1<<31) | foundUserData.value();
        sgb->uniform().context->incr_get_userdata_calls ();
    }
    DASSERT((*userdata_initialized)>>31 == 1);
    Mask foundUserData(*userdata_initialized & 0x7FFFFFFF);
    if (foundUserData.any_on()) {
        // If userdata was present, copy it to the shader variable
    	// Don't bother masking as any lanes without user data
    	// will be overwritten by init ops or by default value
        memcpy (symbol_data, userdata_data, symbol_data_size);
    }    
    
    return foundUserData.value();  
}
#endif
