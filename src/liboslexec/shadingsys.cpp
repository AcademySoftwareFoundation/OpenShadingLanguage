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
#include <OSL/genclosure.h>
#include "backendllvm.h"
#include <OSL/oslquery.h>

#include <OpenImageIO/strutil.h>
#include <OpenImageIO/dassert.h>
#include <OpenImageIO/thread.h>
#include <OpenImageIO/timer.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/optparser.h>
#include <OpenImageIO/fmath.h>

#include "opcolor.h"

using namespace OSL;
using namespace OSL::pvt;

// avoid naming conflicts with MSVC macros
#ifdef _MSC_VER
 #undef RGB
 // We use some of the iso646.h macro names later on in this file. For
 // some compilers (MSVS, I'm looking at you) this is trouble. I don't know
 // how or why that header would have been included here, but it did for at
 // least one person, so shut off those macros so they don't cause trouble.
 #undef and
 #undef or
 #undef xor
 #undef compl
 #undef bitand
 #undef bitor
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
ShadingSystem::ShaderGroupEnd (ShaderGroup& group)
{
    return m_impl->ShaderGroupEnd(group);
}


bool
ShadingSystem::ShaderGroupEnd (void)
{
    return m_impl->ShaderGroupEnd();
}



bool
ShadingSystem::Parameter (ShaderGroup& group, string_view name, TypeDesc t,
                          const void *val, bool lockgeom)
{
    return m_impl->Parameter (group, name, t, val, lockgeom);
}



bool
ShadingSystem::Parameter (string_view name, TypeDesc t, const void *val,
                          bool lockgeom)
{
    return m_impl->Parameter (name, t, val, lockgeom);
}



bool
ShadingSystem::Shader (ShaderGroup& group, string_view shaderusage,
                       string_view shadername, string_view layername)
{
    return m_impl->Shader (group, shaderusage, shadername, layername);
}



bool
ShadingSystem::Shader (string_view shaderusage, string_view shadername,
                       string_view layername)
{
    return m_impl->Shader (shaderusage, shadername, layername);
}



bool
ShadingSystem::ConnectShaders (ShaderGroup& group,
                               string_view srclayer, string_view srcparam,
                               string_view dstlayer, string_view dstparam)
{
    return m_impl->ConnectShaders (group, srclayer, srcparam,
                                   dstlayer, dstparam);
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
ShadingSystem::execute (ShadingContext &ctx, ShaderGroup &group,
                        ShaderGlobals &globals, bool run)
{
    return m_impl->execute (ctx, group, globals, run);
}



// DEPRECATED(2.0)
bool
ShadingSystem::execute (ShadingContext *ctx, ShaderGroup &group,
                        ShaderGlobals &globals, bool run)
{
    return m_impl->execute (ctx, group, globals, run);
}



bool
ShadingSystem::execute_init (ShadingContext &ctx, ShaderGroup &group,
                             ShaderGlobals &globals, bool run)
{
    return ctx.execute_init (group, globals, run);
}



bool
ShadingSystem::execute_layer (ShadingContext &ctx, ShaderGlobals &globals,
                              int layernumber)
{
    return ctx.execute_layer (globals, layernumber);
}



bool
ShadingSystem::execute_layer (ShadingContext &ctx, ShaderGlobals &globals,
                              ustring layername)
{
    int layernumber = find_layer (*ctx.group(), layername);
    return layernumber >= 0 ? ctx.execute_layer (globals, layernumber) : false;
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
                                 SetupClosureFunc setup)
{
    return m_impl->register_closure (name, id, params, prepare, setup);
}



bool
ShadingSystem::query_closure (const char **name, int *id,
                              const ClosureParam **params)
{
    return m_impl->query_closure (name, id, params);
}



static cspan< std::pair<ustring,SGBits> >
sgbit_table ()
{
    static const std::pair<ustring,SGBits> table[] = {
        { ustring("P"),       SGBits::P },
        { ustring("I"),       SGBits::I },
        { ustring("N"),       SGBits::N },
        { ustring("Ng"),      SGBits::Ng },
        { ustring("u"),       SGBits::u },
        { ustring("v"),       SGBits::v },
        { ustring("dPdu"),    SGBits::dPdu },
        { ustring("dPdv"),    SGBits::dPdv },
        { ustring("time"),    SGBits::time },
        { ustring("dtime"),   SGBits::dtime },
        { ustring("dPdtime"), SGBits::dPdtime },
        { ustring("Ps"),      SGBits::Ps },
        { ustring("Ci"),      SGBits::Ci }
    };
    return cspan<std::pair<ustring,SGBits>>(table);
}



SGBits
ShadingSystem::globals_bit (ustring name)
{
    for (auto t : sgbit_table()) {
        if (name == t.first)
            return t.second;
    }
    return SGBits::None;
}



ustring
ShadingSystem::globals_name (SGBits bit)
{
    for (auto t : sgbit_table()) {
        if (bit == t.second)
            return t.first;
    }
    return ustring();
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
    if (!group) {
        m_impl->error ("archive_shadergroup: passed nullptr as group");
        return false;
    }
    return m_impl->archive_shadergroup (*group, filename);
}


bool
ShadingSystem::archive_shadergroup (ShaderGroup& group, string_view filename)
{
    return m_impl->archive_shadergroup (group, filename);
}


void
ShadingSystem::set_raytypes (ShaderGroup *group, int raytypes_on, int raytypes_off)
{
    DASSERT (group);
    group->set_raytypes(raytypes_on, raytypes_off);
}


void
ShadingSystem::optimize_group (ShaderGroup *group, ShadingContext *ctx)
{
    DASSERT (group);
    m_impl->optimize_group (*group, ctx);
}



void
ShadingSystem::optimize_group (ShaderGroup *group,
                               int raytypes_on, int raytypes_off,
                               ShadingContext *ctx)
{
    // convenience function for backwards compatibility
    set_raytypes (group, raytypes_on, raytypes_off);
    optimize_group (group, ctx);
}



static TypeDesc TypeFloatArray2 (TypeDesc::FLOAT, 2);
static TypeDesc TypeFloatArray3 (TypeDesc::FLOAT, 3);
static TypeDesc TypeFloatArray4 (TypeDesc::FLOAT, 4);



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
            memmove (dst, src, dsttype.size());
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
        // float->float[4]
        if (dsttype == TypeFloatArray4) {
            if (dst && src) {
                float f = *(const float *)src;
                ((float *)dst)[0] = f;
                ((float *)dst)[1] = f;
                ((float *)dst)[2] = f;
                ((float *)dst)[3] = f;
            }
            return true;
        }
        return false; // Unsupported conversion
    }

    // float[3] -> triple
    if ((srctype == TypeFloatArray3 && equivalent(dsttype, TypeDesc::TypePoint)) ||
        (dsttype == TypeFloatArray3 && equivalent(srctype, TypeDesc::TypePoint))) {
        if (dst && src)
            memmove (dst, src, dsttype.size());
        return true;
    }

    // float[4] -> vec4
    if ((srctype == TypeFloatArray4 && equivalent(dsttype, TypeDesc::TypeFloat4)) ||
        (dsttype == TypeFloatArray4 && equivalent(srctype, TypeDesc::TypeFloat4))) {
        if (dst && src)
            memmove (dst, src, dsttype.size());
        return true;
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
#define STRDECL(str,var_name) const ustring var_name(str);
#include <OSL/strdecls.h>
#undef STRDECL
}



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
      m_error_repeats(false),
      m_range_checking(true),
      m_unknown_coordsys_error(true), m_connection_error(true),
      m_greedyjit(false), m_countlayerexecs(false),
      m_relaxed_param_typecheck(false),
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
      m_llvm_output_bitcode(0),
      m_commonspace_synonym("world"),
      m_max_local_mem_KB(2048),
      m_compile_report(false),
      m_buffer_printf(true),
      m_no_noise(false),
      m_no_pointcloud(false),
      m_force_derivs(false),
      m_allow_shader_replacement(false),
      m_exec_repeat(1),
      m_opt_warnings(0),
      m_gpu_opt_error(0),
      m_colorspace("Rec709"),
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

    // Allow environment variable to override default options
    const char *options = getenv ("OSL_OPTIONS");
    if (options)
        attribute ("options", TypeDesc::STRING, &options);

    setup_op_descriptors ();

    colorsystem().set_colorspace(m_colorspace);
    ASSERT(colorsystem().set_colorspace(m_colorspace) && "Invalid colorspace");
}



static void
shading_system_setup_op_descriptors (ShadingSystemImpl::OpDescriptorMap& op_descriptor)
{
#define OP2(alias,name,ll,fold,simp,flag)                                \
    extern bool llvm_gen_##ll (BackendLLVM &rop, int opnum);             \
    extern int  constfold_##fold (RuntimeOptimizer &rop, int opnum);     \
    op_descriptor[ustring(#alias)] = OpDescriptor(#name, llvm_gen_##ll,  \
                                                  constfold_##fold, simp, flag);
#define OP(name,ll,fold,simp,flag) OP2(name,name,ll,fold,simp,flag)
#define TEX OpDescriptor::Tex
#define SIDE OpDescriptor::SideEffects

    // name          llvmgen              folder         simple     flags
    OP (aassign,     aassign,             aassign,       false,     0);
    OP (abs,         generic,             abs,           true,      0);
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
    OP (error,       printf,              none,          false,     SIDE);
    OP (exit,        return,              none,          false,     0);
    OP (exp,         generic,             exp,           true,      0);
    OP (exp2,        generic,             exp2,          true,      0);
    OP (expm1,       generic,             expm1,         true,      0);
    OP (fabs,        generic,             abs,           true,      0);
    OP (filterwidth, filterwidth,         deriv,         true,      0);
    OP (floor,       generic,             floor,         true,      0);
    OP (fmod,        modulus,             none,          true,      0);
    OP (for,         loop_op,             none,          false,     0);
    OP (format,      printf,              format,        true,      0);
    OP (fprintf,     printf,              none,          false,     SIDE);
    OP (functioncall, functioncall,       functioncall,  false,     0);
    OP (ge,          compare_op,          ge,            true,      0);
    OP (getattribute, getattribute,       getattribute,  false,     0);
    OP (getchar,      generic,            getchar,       true,      0);
    OP (getmatrix,   getmatrix,           getmatrix,     false,     0);
    OP (getmessage,  getmessage,          getmessage,    false,     0);
    OP (gettextureinfo, gettextureinfo,   gettextureinfo,false,     TEX);
    OP (gt,          compare_op,          gt,            true,      0);
    OP (hash,        generic,             hash,          true,      0);
    OP (hashnoise,   noise,               noise,         true,      0);
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
    OP (mod,         modulus,             mod,           true,      0);
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
    OP (pointcloud_write, pointcloud_write, none,        false,     SIDE);
    OP (pow,         generic,             pow,           true,      0);
    OP (printf,      printf,              none,          false,     SIDE);
    OP (psnoise,     noise,               noise,         true,      0);
    OP (radians,     generic,             radians,       true,      0);
    OP (raytype,     raytype,             raytype,       true,      0);
    OP (regex_match, regex,               none,          false,     0);
    OP (regex_search, regex,              regex_search,  false,     0);
    OP (return,      return,              none,          false,     0);
    OP (round,       generic,             none,          true,      0);
    OP (select,      select,              select,        true,      0);
    OP (setmessage,  setmessage,          setmessage,    false,     SIDE);
    OP (shl,         bitwise_binary_op,   none,          true,      0);
    OP (shr,         bitwise_binary_op,   none,          true,      0);
    OP (sign,        generic,             none,          true,      0);
    OP (sin,         generic,             sin,           true,      0);
    OP (sincos,      sincos,              sincos,        false,     0);
    OP (sinh,        generic,             none,          true,      0);
    OP (smoothstep,  generic,             none,          true,      0);
    OP (snoise,      noise,               noise,         true,      0);
    OP (spline,      spline,              none,          true,      0);
    OP (splineinverse, spline,            none,          true,      0);
    OP (split,       split,               split,         false,     0);
    OP (sqrt,        generic,             sqrt,          true,      0);
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
    OP (trace,       trace,               none,          false,     SIDE);
    OP (transform,   transform,           transform,     true,      0);
    OP (transformc,  transformc,          transformc,    true,      0);
    OP (transformn,  transform,           transform,     true,      0);
    OP (transformv,  transform,           transform,     true,      0);
    OP (transpose,   generic,             none,          true,      0);
    OP (trunc,       generic,             none,          true,      0);
    OP (useparam,    useparam,            useparam,      false,     0);
    OP (vector,      construct_triple,    triple,        true,      0);
    OP (warning,     printf,              warning,       false,     SIDE);
    OP (wavelength_color, blackbody,      none,          true,      0);
    OP (while,       loop_op,             none,          false,     0);
    OP (xor,         bitwise_binary_op,   xor,           true,      0);
#undef OP
#undef TEX
#undef SIDE
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
                                     SetupClosureFunc setup)
{
    for (int i = 0; params && params[i].type != TypeDesc(); ++i) {
        if (params[i].key == NULL && params[i].type.size() != (size_t)params[i].field_size) {
            error ("Parameter %d of '%s' closure is assigned to a field of incompatible size", i + 1, name);
            return;
        }
    }
    m_closure_registry.register_closure(name, id, params, prepare, setup);
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
    ATTR_SET ("llvm_output_bitcode", int, m_llvm_output_bitcode);
    ATTR_SET ("strict_messages", int, m_strict_messages);
    ATTR_SET ("range_checking", int, m_range_checking);
    ATTR_SET ("unknown_coordsys_error", int, m_unknown_coordsys_error);
    ATTR_SET ("connection_error", int, m_connection_error);
    ATTR_SET ("greedyjit", int, m_greedyjit);
    ATTR_SET ("relaxed_param_typecheck", int, m_relaxed_param_typecheck);
    ATTR_SET ("countlayerexecs", int, m_countlayerexecs);
    ATTR_SET ("max_warnings_per_thread", int, m_max_warnings_per_thread);
    ATTR_SET ("max_local_mem_KB", int, m_max_local_mem_KB);
    ATTR_SET ("compile_report", int, m_compile_report);
    ATTR_SET ("buffer_printf", int, m_buffer_printf);
    ATTR_SET ("no_noise", int, m_no_noise);
    ATTR_SET ("no_pointcloud", int, m_no_pointcloud);
    ATTR_SET ("force_derivs", int, m_force_derivs);
    ATTR_SET ("allow_shader_replacement", int, m_allow_shader_replacement);
    ATTR_SET ("exec_repeat", int, m_exec_repeat);
    ATTR_SET ("opt_warnings", int, m_opt_warnings);
    ATTR_SET ("gpu_opt_error", int, m_gpu_opt_error);
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
        if (colorsystem().set_colorspace(c))
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
            m_raytypes.emplace_back(((const char **)val)[i]);
        return true;
    }
    if (name == "renderer_outputs" && type.basetype == TypeDesc::STRING) {
        m_renderer_outputs.clear ();
        for (size_t i = 0;  i < type.numelements();  ++i)
            m_renderer_outputs.emplace_back(((const char **)val)[i]);
        return true;
    }
    if (name == "lib_bitcode" && type.basetype == TypeDesc::UINT8) {
        if (type.arraylen < 0) {
            error ("Invalid bitcode size: %d", type.arraylen);
            return false;
        }
        m_lib_bitcode.clear();
        if (type.arraylen) {
            const char* bytes = static_cast<const char*>(val);
            std::copy(bytes, bytes + type.arraylen,
                      back_inserter(m_lib_bitcode));
        }
        return true;
    }
    if (name == "error_repeats") {
        // Special case: setting error_repeats also clears the "previously
        // seen" error and warning lists.
        m_errseen.clear();
        m_warnseen.clear();
        ATTR_SET ("error_repeats", int, m_error_repeats);
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
    ATTR_DECODE ("llvm_output_bitcode", int, m_llvm_output_bitcode);
    ATTR_DECODE ("strict_messages", int, m_strict_messages);
    ATTR_DECODE ("error_repeats", int, m_error_repeats);
    ATTR_DECODE ("range_checking", int, m_range_checking);
    ATTR_DECODE ("unknown_coordsys_error", int, m_unknown_coordsys_error);
    ATTR_DECODE ("connection_error", int, m_connection_error);
    ATTR_DECODE ("greedyjit", int, m_greedyjit);
    ATTR_DECODE ("countlayerexecs", int, m_countlayerexecs);
    ATTR_DECODE ("relaxed_param_typecheck", int, m_relaxed_param_typecheck);
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
    ATTR_DECODE ("allow_shader_replacement", int, m_allow_shader_replacement);
    ATTR_DECODE ("exec_repeat", int, m_exec_repeat);
    ATTR_DECODE ("opt_warnings", int, m_opt_warnings);
    ATTR_DECODE ("gpu_opt_error", int, m_gpu_opt_error);

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

    if (name == "colorsystem" && type.basetype == TypeDesc::PTR) {
        *(void**)val = &colorsystem();
        return true;
    }

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
            group->m_renderer_outputs.emplace_back(((const char **)val)[i]);
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
    if (name == "group_init_name" && type.basetype == TypeDesc::STRING) {
#ifdef OIIO_HAS_SPRINTF
        *(ustring *)val = ustring::sprintf ("group_%d_init", group->id());
#else
        *(ustring *)val = ustring::format ("group_%d_init", group->id());
#endif
        return true;
    }
    if (name == "group_entry_name" && type.basetype == TypeDesc::STRING) {
        int nlayers = group->nlayers ();
        ShaderInstance *inst = (*group)[nlayers-1];
        // This formuation mirrors OSOProcessorBase::layer_function_name()
#ifdef OIIO_HAS_SPRINTF
        *(ustring *)val = ustring::sprintf ("%s_%s_%d", group->name(),
                                           inst->layername(), inst->id());
#else
        *(ustring *)val = ustring::format ("%s_%s_%d", group->name(),
                                           inst->layername(), inst->id());
#endif
        return true;
    }
    if (name == "layer_osofiles" && type.basetype == TypeDesc::STRING) {
        size_t n = std::min (type.numelements(), (size_t)group->nlayers());
        for (size_t i = 0;  i < n;  ++i)
            ((ustring *)val)[i] =(*group)[i]->master()->osofilename();
        return true;
    }
    if (name == "pickle" && type == TypeDesc::STRING) {
        *(ustring *)val = ustring(group->serialize());
        return true;
    }
    if (name == "exec_repeat" && type == TypeDesc::TypeInt) {
        *(int *)val = group->m_exec_repeat;
        return true;
    }
    if (name == "ptx_compiled_version" && type.basetype == TypeDesc::PTR) {
        bool exists = !group->m_llvm_ptx_compiled_version.empty();
        *(std::string *)val = exists ? group->m_llvm_ptx_compiled_version : "";
        return true;
    }

    // All the remaining attributes require the group to already be
    // optimized.
    if (! group->optimized()) {
        auto threadinfo = create_thread_info();
        auto ctx = get_context(threadinfo);
        optimize_group (*group, ctx);
        release_context(ctx);
        destroy_thread_info (threadinfo);
    }

    if (name == "num_textures_needed" && type == TypeDesc::TypeInt) {
        *(int *)val = (int)group->m_textures_needed.size();
        return true;
    }
    if (name == "textures_needed" && type.basetype == TypeDesc::PTR) {
        size_t n = group->m_textures_needed.size();
        *(ustring **)val = n ? &group->m_textures_needed[0] : NULL;
        return true;
    }
    if (name == "unknown_textures_needed" && type == TypeDesc::TypeInt) {
        *(int *)val = (int)group->m_unknown_textures_needed;
        return true;
    }

    if (name == "num_closures_needed" && type == TypeDesc::TypeInt) {
        *(int *)val = (int)group->m_closures_needed.size();
        return true;
    }
    if (name == "closures_needed" && type.basetype == TypeDesc::PTR) {
        size_t n = group->m_closures_needed.size();
        *(ustring **)val = n ? &group->m_closures_needed[0] : NULL;
        return true;
    }
    if (name == "unknown_closures_needed" && type == TypeDesc::TypeInt) {
        *(int *)val = (int)group->m_unknown_closures_needed;
        return true;
    }

    if (name == "num_globals_needed" && type == TypeDesc::TypeInt) {
        *(int *)val = (int)group->m_globals_needed.size();
        return true;
    }
    if (name == "globals_needed" && type.basetype == TypeDesc::PTR) {
        size_t n = group->m_globals_needed.size();
        *(ustring **)val = n ? &group->m_globals_needed[0] : NULL;
        return true;
    }
    if (name == "globals_read" && type.basetype == TypeDesc::INT) {
        *(int *)val = group->m_globals_read;
        return true;
    }
    if (name == "globals_write" && type.basetype == TypeDesc::INT) {
        *(int *)val = group->m_globals_write;
        return true;
    }

    if (name == "num_userdata" && type == TypeDesc::TypeInt) {
        *(int *)val = (int)group->m_userdata_names.size();
        return true;
    }
    if (name == "userdata_names" && type.basetype == TypeDesc::PTR) {
        size_t n = group->m_userdata_names.size();
        *(ustring **)val = n ? &group->m_userdata_names[0] : NULL;
        return true;
    }
    if (name == "userdata_types" && type.basetype == TypeDesc::PTR) {
        size_t n = group->m_userdata_types.size();
        *(TypeDesc **)val = n ? &group->m_userdata_types[0] : NULL;
        return true;
    }
    if (name == "userdata_offsets" && type.basetype == TypeDesc::PTR) {
        size_t n = group->m_userdata_offsets.size();
        *(int **)val = n ? &group->m_userdata_offsets[0] : NULL;
        return true;
    }
    if (name == "userdata_derivs" && type.basetype == TypeDesc::PTR) {
        size_t n = group->m_userdata_derivs.size();
        *(char **)val = n ? &group->m_userdata_derivs[0] : NULL;
        return true;
    }
    if (name == "num_attributes_needed" && type == TypeDesc::TypeInt) {
        *(int *)val = (int)group->m_attributes_needed.size();
        return true;
    }
    if (name == "attributes_needed" && type.basetype == TypeDesc::PTR) {
        size_t n = group->m_attributes_needed.size();
        *(ustring **)val = n ? &group->m_attributes_needed[0] : NULL;
        return true;
    }
    if (name == "attribute_scopes" && type.basetype == TypeDesc::PTR) {
        size_t n = group->m_attribute_scopes.size();
        *(ustring **)val = n ? &group->m_attribute_scopes[0] : NULL;
        return true;
    }
    if (name == "unknown_attributes_needed" && type == TypeDesc::TypeInt) {
        *(int *)val = (int)group->m_unknown_attributes_needed;
        return true;
    }
    if (name == "group_id" && type == TypeDesc::TypeInt) {
        *(int *)val = (int) group->id();
        return true;
    }

    // Additional atttributes useful to OptiX-based renderers
    if (name == "userdata_layers" && type.basetype == TypeDesc::PTR) {
        size_t n = group->m_userdata_layers.size();
        *(int **)val = n ? &group->m_userdata_layers[0] : NULL;
        return true;
    }
    if (name == "userdata_init_vals" && type.basetype == TypeDesc::PTR) {
        size_t n = group->m_userdata_init_vals.size();
        *(void **)val = n ? &group->m_userdata_init_vals[0] : NULL;
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
        if (s == msg && !m_error_repeats)
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
        if (s == msg && !m_error_repeats)
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
    out.imbue (std::locale::classic());  // force C locale
    out << "OSL ShadingSystem statistics (" << (void*)this;
    out << ") ver " << OSL_LIBRARY_VERSION_STRING
        << ", LLVM " << OSL_LLVM_FULL_VERSION << "\n";
    if (m_stat_shaders_requested == 0 && m_stat_shaders_loaded == 0) {
        out << "  No shaders requested or loaded\n";
        return out.str();
    }

    std::string opt;
#define BOOLOPT(name) opt += Strutil::sprintf(#name "=%d ", m_##name)
#define INTOPT(name) opt += Strutil::sprintf(#name "=%d ", m_##name)
#define STROPT(name) if (m_##name.size()) opt += Strutil::sprintf(#name "=\"%s\" ", m_##name)
    INTOPT (optimize);
    INTOPT (llvm_optimize);
    INTOPT (debug);
    INTOPT (profile);
    INTOPT (llvm_debug);
    BOOLOPT (llvm_debug_layers);
    BOOLOPT (llvm_debug_ops);
    BOOLOPT (llvm_output_bitcode);
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
    BOOLOPT (error_repeats);
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
    INTOPT (allow_shader_replacement);
    INTOPT (exec_repeat);
    INTOPT (opt_warnings);
    INTOPT (gpu_opt_error);
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
    float iperg = (float)m_stat_groupinstances/std::max((int)m_stat_groups,1);
    out << "    Avg instances per group: "
        << Strutil::sprintf ("%.1f", iperg) << "\n";
    out << "  Shading contexts: " << m_stat_contexts << "\n";
    if (m_countlayerexecs)
        out << "  Total layers executed: " << m_stat_layers_executed << "\n";

#if 0
    long long totalexec = m_layers_executed_uncond + m_layers_executed_lazy +
                          m_layers_executed_never;
    out << Strutil::sprintf ("  Total layers run: %10lld\n", totalexec);
    double inv_totalexec = 1.0 / std::max (totalexec, 1LL);  // prevent div by 0
    out << Strutil::sprintf ("    Unconditional:  %10lld  (%.1f%%)\n",
                            (long long)m_layers_executed_uncond,
                            (100.0*m_layers_executed_uncond) * inv_totalexec);
    out << Strutil::sprintf ("    On demand:      %10lld  (%.1f%%)\n",
                            (long long)m_layers_executed_lazy,
                            (100.0*m_layers_executed_lazy) * inv_totalexec);
    out << Strutil::sprintf ("    Skipped:        %10lld  (%.1f%%)\n",
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
        out << Strutil::sprintf ("  Optimized %llu ops to %llu (%.1f%%)\n",
                                (long long)m_stat_preopt_ops,
                                (long long)m_stat_postopt_ops,
                                100.0*(double(m_stat_postopt_ops)/double(std::max(1,(int)m_stat_preopt_ops))-1.0));
        out << Strutil::sprintf ("  Optimized %llu symbols to %llu (%.1f%%)\n",
                                (long long)m_stat_preopt_syms,
                                (long long)m_stat_postopt_syms,
                                100.0*(double(m_stat_postopt_syms)/double(std::max(1,(int)m_stat_preopt_syms))-1.0));
    }
    out << Strutil::sprintf ("  Constant connections eliminated: %d\n",
                            (int)m_stat_const_connections);
    out << Strutil::sprintf ("  Global connections eliminated: %d\n",
                            (int)m_stat_global_connections);
    out << Strutil::sprintf ("  Middlemen eliminated: %d\n",
                            (int)m_stat_middlemen_eliminated);
    out << Strutil::sprintf ("  Derivatives needed on %d / %d symbols (%.1f%%)\n",
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
        out << "      average query results: " << Strutil::sprintf ("%.1f", avg) << "\n";
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
            for (auto&& grp : m_all_shader_groups) {
                if (ShaderGroupRef g = grp.lock()) {
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
                grouptimes.emplace_back(m->first, m->second);
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
    return Parameter (*m_curgroup, name, t, val, lockgeom);
}



bool
ShadingSystemImpl::Parameter (ShaderGroup& group, string_view name,
                              TypeDesc t, const void *val, bool lockgeom)
{
    // We work very hard not to do extra copies of the data.  First,
    // grow the pending list by one (empty) slot...
    group.m_pending_params.grow();
    // ...then initialize it in place
    group.m_pending_params.back().init (name, t, 1, val);
    // If we have a possible geometric override (lockgeom=false), set the
    // param's interpolation to VERTEX rather than the default CONSTANT.
    if (lockgeom == false)
        group.m_pending_params.back().interp (OIIO::ParamValue::INTERP_VERTEX);
    return true;
}



ShaderGroupRef
ShadingSystemImpl::ShaderGroupBegin (string_view groupname)
{
    ShaderGroupRef group (new ShaderGroup(groupname));
    group->m_exec_repeat = m_exec_repeat;
    {
        // Record the group in the SS's census of all extant groups
        spin_lock lock (m_all_shader_groups_mutex);
        m_all_shader_groups.push_back (group);
        ++m_groups_to_compile_count;
        m_curgroup = group;
    }
    return group;
}



bool
ShadingSystemImpl::ShaderGroupEnd (void)
{
    if (! m_curgroup) {
        error ("ShaderGroupEnd() was called without ShaderGroupBegin()");
        return false;
    }
    bool ok = ShaderGroupEnd (*m_curgroup);
    m_curgroup.reset();  // no currently active group
    return ok;
}



bool
ShadingSystemImpl::ShaderGroupEnd (ShaderGroup& group)
{
    // Lock just in case we do something not thread-safe within
    // ShaderGroupEnd. This may be overly cautious, but unless it shows
    // up as a major bottleneck, I'm inclined to play it safe.
    lock_guard lock (m_mutex);

    // Mark the layers that can be run lazily
    if (! group.m_group_use.empty()) {
        int nlayers = group.nlayers ();
        for (int layer = 0;  layer < nlayers;  ++layer) {
            ShaderInstance *inst = group[layer];
            if (! inst)
                continue;
            inst->last_layer (layer == nlayers-1);
        }

        // Merge instances now if they really want it bad, otherwise wait
        // until we optimize the group.
        if (m_opt_merge_instances >= 2)
            merge_instances (group);
    }

    // Merge the raytype_queries of all the individual layers
    group.m_raytype_queries = 0;
    for (int layer = 0, n = group.nlayers(); layer < n; ++layer) {
        ASSERT (group[layer]);
        if (ShaderInstance *inst = group[layer])
            group.m_raytype_queries |= inst->master()->raytype_queries();
    }
    // std::cout << "Group " << group.name() << " ray query bits "
    //         << group.m_raytype_queries << "\n";

    ustring groupname = group.name();
    if (groupname.size() && groupname == m_archive_groupname) {
        std::string filename = m_archive_filename.string();
        if (! filename.size())
            filename = OIIO::Filesystem::filename (groupname.string()) + ".tar.gz";
        archive_shadergroup (group, filename);
    }

    group.m_complete = true;
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

    return Shader (*m_curgroup, shaderusage, shadername, layername);
}



bool
ShadingSystemImpl::Shader (ShaderGroup& group, string_view shaderusage,
                           string_view shadername, string_view layername)
{
    ShaderMaster::ref master = loadshader (shadername);
    if (! master) {
        error ("Could not find shader \"%s\"\n"
               "        group: %s",
               shadername, group.name());
        return false;
    }

    if (shaderusage.empty()) {
        error ("Shader usage required\n"
               "        group: %s",
               shadername, group.name());
        return false;
    }

    // If a layer name was not supplied, make one up.
    std::string local_layername;
    if (layername.empty()) {
        local_layername = OIIO::Strutil::sprintf ("%s_%d", master->shadername(),
                                                 group.nlayers());
        layername = string_view (local_layername);
    }

    ShaderInstanceRef instance (new ShaderInstance (master, layername));
    instance->parameters (group.m_pending_params);
    group.m_pending_params.clear ();
    group.m_pending_params.shrink_to_fit ();

    if (group.m_group_use.empty()) {
        // First in a group
        group.clear ();
        m_stat_groups += 1;
        group.m_group_use = shaderusage;
    } else if (shaderusage != group.m_group_use) {
        error ("Shader usage \"%s\" does not match current group (%s)\n"
               "        group: %s",
               shaderusage, group.m_group_use, group.name());
        return false;
    }

    group.append (instance);
    m_stat_groupinstances += 1;

    // FIXME -- check for duplicate layer name within the group?

    return true;
}



bool
ShadingSystemImpl::ConnectShaders (string_view srclayer, string_view srcparam,
                                   string_view dstlayer, string_view dstparam)
{
    if (! m_curgroup) {
        error ("ConnectShaders can only be called within ShaderGroupBegin/End");
        return false;
    }
    return ConnectShaders (*m_curgroup, srclayer, srcparam, dstlayer, dstparam);
}



bool
ShadingSystemImpl::ConnectShaders (ShaderGroup& group,
                                   string_view srclayer, string_view srcparam,
                                   string_view dstlayer, string_view dstparam)
{
    // Basic sanity checks
    // ConnectShaders, and that the layer and parameter names are not empty.
    if (! srclayer.size() || ! srcparam.size()) {
        error ("ConnectShaders: badly formed source layer/parameter\n"
               "        group: %s", group.name());
        return false;
    }
    if (! dstlayer.size() || ! dstparam.size()) {
        error ("ConnectShaders: badly formed destination layer/parameter\n"
               "        group: %s", group.name());
        return false;
    }

    // Decode the layers, finding the indices within our group and
    // pointers to the instances.  Error and return if they are not found,
    // or if it's not connecting an earlier src to a later dst.
    ShaderInstance *srcinst, *dstinst;
    int srcinstindex = find_named_layer_in_group (group, ustring(srclayer), srcinst);
    int dstinstindex = find_named_layer_in_group (group, ustring(dstlayer), dstinst);
    if (! srcinst) {
        error ("ConnectShaders: source layer \"%s\" not found\n"
               "        group: %s", srclayer, group.name());
        return false;
    }
    if (! dstinst) {
        error ("ConnectShaders: destination layer \"%s\" not found\n"
               "        group: %s", dstlayer, group.name());
        return false;
    }
    if (dstinstindex <= srcinstindex) {
        error ("ConnectShaders: destination layer must follow source layer (tried to connect %s.%s -> %s.%s)\n"
               "        group: %s", srclayer, srcparam, dstlayer, dstparam,
               group.name());
        return false;
    }

    // Decode the parameter names, find their symbols in their
    // respective layers, and also decode requrest to attach specific
    // array elements or color/vector channels.
    ConnectedParam srccon = decode_connected_param(srcparam, srclayer, srcinst);
    ConnectedParam dstcon = decode_connected_param(dstparam, dstlayer, dstinst);
    if (! (srccon.valid() && dstcon.valid())) {
        if (connection_error())
            error ("ConnectShaders: cannot connect a %s (%s) to a %s (%s), invalid connection\n"
                   "        group: %s",
                   srccon.type, srcparam, dstcon.type, dstparam, group.name());
        else
            warning ("ConnectShaders: cannot connect a %s (%s) to a %s (%s), invalid connection\n"
                     "        group: %s",
                     srccon.type, srcparam, dstcon.type, dstparam, group.name());
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
            std::string s = Strutil::sprintf("%s.%s", srcparam, srcstruct->field(i).name);
            std::string d = Strutil::sprintf("%s.%s", dstparam, dststruct->field(i).name);
            ConnectShaders (group, srclayer, s, dstlayer, d);
        }
        return true;
    }

    if (! assignable (dstcon.type, srccon.type)) {
        if (connection_error())
            error ("ConnectShaders: cannot connect a %s (%s) to a %s (%s)\n"
                   "        group: %s",
                   srccon.type, srcparam, dstcon.type, dstparam, group.name());
        else
            warning ("ConnectShaders: cannot connect a %s (%s) to a %s (%s)\n"
                     "        group: %s",
                     srccon.type, srcparam, dstcon.type, dstparam, group.name());
        return false;
    }

    const Symbol *dstsym = dstinst->mastersymbol(dstcon.param);
    ASSERT (dstsym);
    if (dstsym && !dstsym->allowconnect()) {
        std::string name = dstlayer.size() ? Strutil::sprintf("%s.%s", dstlayer, dstparam)
                                           : std::string(dstparam);
        error ("ConnectShaders: cannot connect to %s because it has metadata allowconnect=0\n"
               "        group: %s", name, group.name());
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
    string_view errstatement;
    std::vector<int> intvals;
    std::vector<float> floatvals;
    std::vector<ustring> stringvals;
    string_view p = groupspec;   // parse view
    // std::cout << "!!!!!\n---\n" << groupspec << "\n---\n\n";
    while (p.size()) {
        string_view pstart = p;  // save where we were for error reporting
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
            bool ok = Shader (usage, shadername, layername);
            if (!ok) {
                errstatement = pstart;
                err = true;
                break;
            }
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
            bool ok = ConnectShaders (lay1, param1, lay2, param2);
            if (!ok) {
                errstatement = pstart;
                err = true;
                break;
            }
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
            errdesc = Strutil::sprintf ("Unknown statement (expected 'param', "
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
            errdesc = Strutil::sprintf ("Unknown type: %s", typestring);
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
        int lockgeom = m_lockgeom_default;
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
                stringvals.emplace_back(s);
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
                        errdesc = Strutil::sprintf ("hint %s expected int value", hint_name);
                        break;
                    }
                } else {
                    err = true;
                    errdesc = Strutil::sprintf ("unknown hint '%s %s'",
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

        bool ok = true;
        if (type.basetype == TypeDesc::INT) {
            ok = Parameter (paramname, type, &intvals[0], lockgeom);
        } else if (type.basetype == TypeDesc::FLOAT) {
            ok = Parameter (paramname, type, &floatvals[0], lockgeom);
        } else if (type.basetype == TypeDesc::STRING) {
            ok = Parameter (paramname, type, &stringvals[0], lockgeom);
        }
        if (!ok) {
            errstatement = pstart;
            err = true;
            break;
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
        std::string msg = Strutil::format (
                "ShaderGroupBegin: error parsing group description: %s\n"
                "        group: %s",
                errdesc, g->name());
        if (errstatement.empty()) {
            size_t offset = p.data() - groupspec.data();
            size_t begin_stmt = std::min (groupspec.find_last_of (';', offset),
                                          groupspec.find_last_of (',', offset));
            size_t end_stmt = groupspec.find_first_of (';', begin_stmt+1);
            errstatement = groupspec.substr (begin_stmt+1, end_stmt-begin_stmt);
        }
        if (errstatement.size())
            msg += Strutil::format ("\n        problem might be here: %s",
                                    errstatement);
        error ("%s", msg);
        if (debug())
            info ("Broken group was:\n---%s\n---\n", groupspec);
        return ShaderGroupRef();
    }

    return g;
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
    if (!sym) {
        // Can have a paramindex >= 0, but no symbol when it's a master-symbol
        DASSERT(layer->mastersymbol(paramindex) && "No symbol for paramindex");
        return false;
    }

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
    if (! threadinfo) {
#if OSL_VERSION < 20200
        threadinfo = get_perthread_info ();
        warning ("ShadingSystem::get_context called without a PerThreadInfo");
#else
        error ("ShadingSystem::get_context called without a PerThreadInfo");
        return nullptr;
#endif
    }
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
ShadingSystemImpl::execute (ShadingContext &ctx, ShaderGroup &group,
                            ShaderGlobals &ssg, bool run)
{
    return ctx.execute (group, ssg, run);
}



// Deprecated
bool
ShadingSystemImpl::execute (ShadingContext *ctx, ShaderGroup &group,
                            ShaderGlobals &ssg, bool run)
{
    bool free_context = false;
    OSL::PerThreadInfo *thread_info = nullptr;
    if (! ctx) {
        thread_info = create_thread_info();
        ctx = get_context(thread_info);
        free_context = true;
    }
    bool result = ctx->execute (group, ssg, run);
    if (free_context) {
        release_context(ctx);
        destroy_thread_info(thread_info);
    }
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
ShadingSystemImpl::find_named_layer_in_group (ShaderGroup& group,
                                              ustring layername,
                                              ShaderInstance * &inst)
{
    inst = NULL;
    if (group.m_group_use.empty())
        return -1;
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
    // Grab just the part of the param name up to the bracket
    ustring param (connectionname, 0, bracketpos);
    string_view cname_remaining = connectionname.substr (bracketpos);

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

    if (! cname_remaining.empty() && c.type.is_array()) {
        // There was at least one set of brackets that appears to be
        // selecting an array element.
        int index = 0;
        if (! (Strutil::parse_char (cname_remaining, '[') &&
               Strutil::parse_int  (cname_remaining, index) &&
               Strutil::parse_char (cname_remaining, ']'))) {
            error ("ConnectShaders: malformed parameter \"%s\"", connectionname);
            c.param = -1;  // mark as invalid
            return c;
        }
        c.arrayindex = index;
        if (c.arrayindex >= c.type.arraylength()) {
            error ("ConnectShaders: cannot request array element %s from a %s",
                   connectionname, c.type.c_str());
            c.arrayindex = c.type.arraylength() - 1;  // clamp it
        }
        c.type.make_array (0);              // chop to the element type
        Strutil::skip_whitespace (cname_remaining); // skip to next bracket
    }

    if (! cname_remaining.empty() && cname_remaining.front() == '[' &&
          ! c.type.is_closure() && c.type.aggregate() != TypeDesc::SCALAR) {
        // There was at least one set of brackets that appears to be
        // selecting a color/vector component.
        int index = 0;
        if (! (Strutil::parse_char (cname_remaining, '[') &&
               Strutil::parse_int  (cname_remaining, index) &&
               Strutil::parse_char (cname_remaining, ']'))) {
            error ("ConnectShaders: malformed parameter \"%s\"", connectionname);
            c.param = -1;  // mark as invalid
            return c;
        }
        c.channel = index;
        if (c.channel >= (int)c.type.aggregate()) {
            error ("ConnectShaders: cannot request component %s from a %s",
                   connectionname, c.type.c_str());
            c.channel = (int)c.type.aggregate() - 1;  // clamp it
        }
        // chop to just the scalar part
        c.type = TypeSpec ((TypeDesc::BASETYPE)c.type.simpletype().basetype);
        Strutil::skip_whitespace (cname_remaining);
    }

    // Deal with left over nonsense or unsupported param designations
    if (! cname_remaining.empty()) {
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
        if (aovs.size() > 0) {
            if (std::find(aovs.begin(), aovs.end(), paramname) != aovs.end())
                return true;
            // Try "layer.name"
            ustring name2 = ustring::format("%s.%s", layername, paramname);
            if (std::find(aovs.begin(), aovs.end(), name2) != aovs.end())
                return true;
        }
    }
    const std::vector<ustring> &aovs (m_renderer_outputs);
    if (aovs.size() > 0) {
        if (std::find(aovs.begin(), aovs.end(), paramname) != aovs.end())
            return true;
        ustring name2 = ustring::format("%s.%s", layername, paramname);
        if (std::find(aovs.begin(), aovs.end(), name2) != aovs.end())
            return true;
    }
    return false;
}



void
ShadingSystemImpl::group_post_jit_cleanup (ShaderGroup &group)
{
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
ShadingSystemImpl::optimize_group (ShaderGroup &group, ShadingContext *ctx)
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

    if (!m_only_groupname.empty() && m_only_groupname != group.name()) {
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

    double locking_time = timer();

    bool ctx_allocated = false;
    PerThreadInfo *thread_info = nullptr;
    if (! ctx) {
        thread_info = create_thread_info();
        ctx = get_context(thread_info);
        ctx_allocated = true;
    }
    RuntimeOptimizer rop (*this, group, ctx);
    rop.run ();
    rop.police_failed_optimizations();

    // Copy some info recorded by the RuntimeOptimizer into the group
    group.m_unknown_textures_needed = rop.m_unknown_textures_needed;
    for (auto&& f : rop.m_textures_needed)
        group.m_textures_needed.push_back (f);
    group.m_unknown_closures_needed = rop.m_unknown_closures_needed;
    for (auto&& f : rop.m_closures_needed)
        group.m_closures_needed.push_back (f);
    for (auto&& f : rop.m_globals_needed)
        group.m_globals_needed.push_back (f);
    group.m_globals_read = rop.m_globals_read;
    group.m_globals_write = rop.m_globals_write;
    size_t num_userdata = rop.m_userdata_needed.size();
    group.m_userdata_names.reserve (num_userdata);
    group.m_userdata_types.reserve (num_userdata);
    group.m_userdata_offsets.resize (num_userdata, 0);
    group.m_userdata_derivs.reserve (num_userdata);
    group.m_userdata_layers.reserve (num_userdata);
    group.m_userdata_init_vals.reserve (num_userdata);
    for (auto&& n : rop.m_userdata_needed) {
        group.m_userdata_names.push_back (n.name);
        group.m_userdata_types.push_back (n.type);
        group.m_userdata_derivs.push_back (n.derivs);
        group.m_userdata_layers.push_back (n.layer_num);
        group.m_userdata_init_vals.push_back (n.data);
    }
    group.m_unknown_attributes_needed = rop.m_unknown_attributes_needed;
    for (auto&& f : rop.m_attributes_needed) {
        group.m_attributes_needed.push_back (f.name);
        group.m_attribute_scopes.push_back (f.scope);
    }

    BackendLLVM lljitter (*this, group, ctx);
    lljitter.run ();

    group_post_jit_cleanup (group);

    if (ctx_allocated) {
        release_context(ctx);
        destroy_thread_info(thread_info);
    }

    group.m_optimized = true;
    spin_lock stat_lock (m_stat_mutex);
    m_stat_optimization_time += timer();
    m_stat_opt_locking_time += locking_time + rop.m_stat_opt_locking_time;
    m_stat_specialization_time += rop.m_stat_specialization_time;
    m_stat_total_llvm_time += lljitter.m_stat_total_llvm_time;
    m_stat_llvm_setup_time += lljitter.m_stat_llvm_setup_time;
    m_stat_llvm_irgen_time += lljitter.m_stat_llvm_irgen_time;
    m_stat_llvm_opt_time += lljitter.m_stat_llvm_opt_time;
    m_stat_llvm_jit_time += lljitter.m_stat_llvm_jit_time;
    m_stat_max_llvm_local_mem = std::max (m_stat_max_llvm_local_mem,
                                          lljitter.m_llvm_local_mem);
    m_stat_groups_compiled += 1;
    m_stat_instances_compiled += group.nlayers();
    m_groups_to_compile_count -= 1;
}



static void optimize_all_groups_wrapper (ShadingSystemImpl *ss, int mythread, int totalthreads)
{
    ss->optimize_all_groups (1, mythread, totalthreads);
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
    PerThreadInfo* threadinfo = create_thread_info();
    ShadingContext* ctx = get_context(threadinfo);
    for (size_t i = 0;  i < ngroups;  ++i) {
        // Assign to threads based on mod of totalthreads
        if ((i % totalthreads) == (unsigned)mythread) {
            ShaderGroupRef group;
            {
                spin_lock lock (m_all_shader_groups_mutex);
                group = m_all_shader_groups[i].lock();
            }
            if (group && group->m_complete)
                optimize_group (*group, ctx);
        }
    }
    release_context(ctx);
    destroy_thread_info(threadinfo);
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



#if OIIO_HAS_COLORPROCESSOR

OIIO::ColorProcessorHandle
OCIOColorSystem::load_transform (StringParam fromspace, StringParam tospace)
{
    if (fromspace != m_last_colorproc_fromspace ||
        tospace != m_last_colorproc_tospace) {
        m_last_colorproc = m_colorconfig.createColorProcessor (fromspace, tospace);
        m_last_colorproc_fromspace = fromspace;
        m_last_colorproc_tospace = tospace;
    }
    return m_last_colorproc;
}

#endif



template <> bool
ShadingSystemImpl::ocio_transform (StringParam fromspace, StringParam tospace,
                                   const Color3& C, Color3& Cout) {
#if OIIO_HAS_COLORPROCESSOR
    OIIO::ColorProcessorHandle cp;
    {
        lock_guard lock (m_mutex);
        cp = m_ocio_system.load_transform(fromspace, tospace);
    }
    if (cp) {
        Cout = C;
        cp->apply ((float *)&Cout);
        return true;
    }
#endif
    return false;
}



template <> bool
ShadingSystemImpl::ocio_transform (StringParam fromspace, StringParam tospace,
                                   const Dual2<Color3>& C, Dual2<Color3>& Cout) {
#if OIIO_HAS_COLORPROCESSOR
    OIIO::ColorProcessorHandle cp;
    {
        lock_guard lock (m_mutex);
        cp = m_ocio_system.load_transform(fromspace, tospace);
    }

    if (cp) {
        // Use finite differencing to approximate the derivative. Make 3
        // color values to convert.
        const float eps = 0.001f;
        Color3 CC[3] = { C.val(), C.val() + eps*C.dx(), C.val() + eps*C.dy() };
        cp->apply ((float *)&CC, 3, 1, 3, sizeof(float), sizeof(Color3), 0);
        Cout.set (CC[0],
                  (CC[1] - CC[0]) * (1.0f / eps),
                  (CC[2] - CC[0]) * (1.0f / eps));
        return true;
    }
#endif
    return false;
}



bool
ShadingSystemImpl::archive_shadergroup (ShaderGroup& group, string_view filename)
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
        groupfile << group.serialize();
        groupfile.close ();
    } else {
        error ("archive_shadergroup: Could not open shadergroup file");
        ok = false;
    }

    std::string filename_list = "shadergroup";
    {
        std::lock_guard<ShaderGroup> lock (group);
        std::set<std::string> entries;   // to avoid duplicates
        for (int i = 0, nl = group.nlayers(); i < nl; ++i) {
            std::string osofile = group[i]->master()->osofilename();
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
        std::string cmd = Strutil::sprintf ("tar -c %s -C %s -f %s%s %s",
                                           z, tmpdir, filename, extension,
                                           filename_list);
        // std::cout << "Command =\n" << cmd << "\n";
        if (system (cmd.c_str()) != 0) {
            error ("archive_shadergroup: executing tar command failed");
            ok = false;
        }

    } else if (extension == ".zip") {
        std::string cmd = Strutil::sprintf ("zip -q %s%s %s",
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
                                   SetupClosureFunc setup)
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
            /* CLOSURE_FINISH_PARAM stashes the real struct alignement here
             * make sure that the closure struct doesn't want more alignment than ClosureComponent
             * because we will be allocating the real struct inside it. */
            ASSERT_MSG(params[i].field_size <= int(alignof(ClosureComponent)),
                "Closure %s wants alignment of %d which is larger than that of ClosureComponent",
                name.c_str(),
                params[i].field_size);
            break;
        }
        if (params[i].key == nullptr)
            entry.nformal ++;
        else
            entry.nkeyword ++;
    }
    entry.prepare = prepare;
    entry.setup = setup;
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
            int n = int (p.type.numelements() * p.type.aggregate);
            if (p.type.basetype == TypeDesc::INT) {
                for (int i = 0; i < n; ++i)
                    p.idefault.push_back (sym->get_int(i));
            }
            if (p.type.basetype == TypeDesc::FLOAT) {
                for (int i = 0; i < n; ++i)
                    p.fdefault.push_back (sym->get_float(i));
            }
            if (p.type.basetype == TypeDesc::STRING) {
                for (int i = 0; i < n; ++i)
                    p.sdefault.push_back (sym->get_string(i));
            }
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

    return true;
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



OSL_SHADEOP int
osl_range_check_err (int indexvalue, int length, const char *symname,
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
