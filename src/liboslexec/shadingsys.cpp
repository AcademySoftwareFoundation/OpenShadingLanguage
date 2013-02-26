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

#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>

#include "oslexec_pvt.h"
#include "genclosure.h"
#include "llvm_headers.h"

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"
#include "OpenImageIO/filesystem.h"
#include "OpenImageIO/optparser.h"

using namespace OSL;
using namespace OSL::pvt;

// avoid naming conflict with MSVC macro
#ifdef RGB
#undef RGB
#endif

OSL_NAMESPACE_ENTER



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
    : llvm_context(NULL), llvm_jitmm(NULL)
{
}



PerThreadInfo::~PerThreadInfo ()
{
    delete llvm_context;
    // N.B. Do NOT delete the jitmm -- another thread may need the code!
    // Don't worry, we stashed a pointer in the shadingsys.

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
ustring interp("interp"), closest("closest"), linear("linear");
ustring cubic("cubic"), smartcubic("smartcubic");
ustring perlin("perlin"), uperlin("uperlin");
ustring noise("noise"), snoise("snoise");
ustring cell("cell"), cellnoise("cellnoise"), pcellnoise("pcellnoise");
ustring pnoise("pnoise"), psnoise("psnoise");
ustring genericnoise("genericnoise"), genericpnoise("genericpnoise");
ustring gabor("gabor"), gabornoise("gabornoise"), gaborpnoise("gaborpnoise");
ustring anisotropic("anisotropic"), direction("direction");
ustring do_filter("do_filter"), bandwidth("bandwidth"), impulses("impulses");
ustring op_dowhile("dowhile"), op_for("for"), op_while("while");
ustring subimage("subimage"), subimagename("subimagename");

};



namespace pvt {   // OSL::pvt


ShadingSystemImpl::ShadingSystemImpl (RendererServices *renderer,
                                      TextureSystem *texturesystem,
                                      ErrorHandler *err)
    : m_renderer(renderer), m_texturesys(texturesystem), m_err(err),
      m_statslevel (0), m_lazylayers (true),
      m_lazyglobals (false),
      m_clearmemory (false), m_debugnan (false),
      m_lockgeom_default (false), m_strict_messages(true),
      m_range_checking(true), m_unknown_coordsys_error(true),
      m_greedyjit(false), m_countlayerexecs(false),
      m_max_warnings_per_thread(100),
      m_optimize(2),
      m_opt_constant_param(true), m_opt_constant_fold(true),
      m_opt_stale_assign(true), m_opt_elide_useless_ops(true),
      m_opt_elide_unconnected_outputs(true),
      m_opt_peephole(true), m_opt_coalesce_temps(true),
      m_opt_assign(true), m_opt_mix(true), m_opt_merge_instances(true),
      m_opt_fold_getattribute(true),
      m_optimize_nondebug(false),
      m_llvm_optimize(0),
      m_debug(false), m_llvm_debug(false),
      m_commonspace_synonym("world"),
      m_colorspace("Rec709"),
      m_max_local_mem_KB(1024),
      m_compile_report(false),
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
    m_stat_preopt_ops = 0;
    m_stat_postopt_ops = 0;
    m_stat_optimization_time = 0;
    m_stat_getattribute_time = 0;
    m_stat_getattribute_fail_time = 0;
    m_stat_getattribute_calls = 0;
    m_stat_pointcloud_searches = 0;
    m_stat_pointcloud_searches_total_results = 0;
    m_stat_pointcloud_max_results = 0;
    m_stat_pointcloud_failures = 0;
    m_stat_pointcloud_gets = 0;
    m_stat_pointcloud_writes = 0;
    m_stat_layers_executed = 0;

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

    attribute ("colorspace", TypeDesc::STRING, &m_colorspace);

    // Allow environment variable to override default options
    const char *options = getenv ("OSL_OPTIONS");
    if (options)
        ShadingSystem::attribute ("options", options);

    setup_op_descriptors ();
    SetupLLVM ();
}



static void
shading_system_setup_op_descriptors (ShadingSystemImpl::OpDescriptorMap& op_descriptor)
{
#define OP2(alias,name,ll,f,simp)                                        \
    extern bool llvm_gen_##ll (RuntimeOptimizer &rop, int opnum);        \
    extern int  constfold_##f (RuntimeOptimizer &rop, int opnum);        \
    op_descriptor[ustring(#alias)] = OpDescriptor(#name, llvm_gen_##ll,  \
                                                   constfold_##f, simp);
#define OP(name,ll,f,simp) OP2(name,name,ll,f,simp)

    // name          llvmgen              folder         simple
    OP (aassign,     aassign,             none,          false);
    OP (abs,         generic,             abs,           true);
    OP (acos,        generic,             none,          true);
    OP (add,         add,                 add,           true);
    OP (and,         andor,               and,           true);
    OP (area,        area,                none,          true);
    OP (aref,        aref,                aref,          true);
    OP (arraycopy,   arraycopy,           none,          false);
    OP (arraylength, arraylength,         arraylength,   true);
    OP (asin,        generic,             none,          true);
    OP (assign,      assign,              none,          true);
    OP (atan,        generic,             none,          true);
    OP (atan2,       generic,             none,          true);
    OP (backfacing,  get_simple_SG_field, none,          true);
    OP (bitand,      bitwise_binary_op,   none,          true);
    OP (bitor,       bitwise_binary_op,   none,          true);
    OP (blackbody,   blackbody,           none,          true);
    OP (break,       loopmod_op,          none,          false);
    OP (calculatenormal, calculatenormal, none,          true);
    OP (ceil,        generic,             ceil,          true);
    OP (cellnoise,   noise,               none,          true);
    OP (clamp,       clamp,               clamp,         true);
    OP (closure,     closure,             none,          true);
    OP (color,       construct_color,     triple,        true);
    OP (compassign,  compassign,          compassign,    false);
    OP (compl,       unary_op,            none,          true);
    OP (compref,     compref,             compref,       true);
    OP (concat,      generic,             concat,        true);
    OP (continue,    loopmod_op,          none,          false);
    OP (cos,         generic,             none,          true);
    OP (cosh,        generic,             none,          true);
    OP (cross,       generic,             none,          true);
    OP (degrees,     generic,             none,          true);
    OP (determinant, generic,             none,          true);
    OP (dict_find,   dict_find,           none,          false);
    OP (dict_next,   dict_next,           none,          false);
    OP (dict_value,  dict_value,          none,          false);
    OP (distance,    generic,             none,          true);
    OP (div,         div,                 div,           true);
    OP (dot,         generic,             dot,           true);
    OP (Dx,          DxDy,                none,          true);
    OP (Dy,          DxDy,                none,          true);
    OP (Dz,          Dz,                  none,          true);
    OP (dowhile,     loop_op,             none,          false);
    OP (end,         end,                 none,          false);
    OP (endswith,    generic,             endswith,      true);
    OP (environment, environment,         none,          true);
    OP (eq,          compare_op,          eq,            true);
    OP (erf,         generic,             none,          true);
    OP (erfc,        generic,             none,          true);
    OP (error,       printf,              none,          false);
    OP (exit,        return,              none,          false);
    OP (exp,         generic,             none,          true);
    OP (exp2,        generic,             none,          true);
    OP (expm1,       generic,             none,          true);
    OP (fabs,        generic,             none,          true);
    OP (filterwidth, filterwidth,         none,          true);
    OP (floor,       generic,             floor,          true);
    OP (fmod,        modulus,             none,          true);
    OP (for,         loop_op,             none,          false);
    OP (format,      printf,              format,        true);
    OP (functioncall, functioncall,       functioncall,  false);
    OP (ge,          compare_op,          ge,            true);
    OP (getattribute, getattribute,       getattribute,  false);
    OP (getmatrix,   getmatrix,           getmatrix,     false);
    OP (getmessage,  getmessage,          getmessage,    false);
    OP (gettextureinfo, gettextureinfo,   gettextureinfo,false);
    OP (gt,          compare_op,          gt,            true);
    OP (if,          if,                  if,            false);
    OP (inversesqrt, generic,             none,          true);
    OP (isconnected, generic,             none,          true);
    OP (isfinite,    generic,             none,          true);
    OP (isinf,       generic,             none,          true);
    OP (isnan,       generic,             none,          true);
    OP (le,          compare_op,          le,            true);
    OP (length,      generic,             none,          true);
    OP (log,         generic,             none,          true);
    OP (log10,       generic,             none,          true);
    OP (log2,        generic,             none,          true);
    OP (logb,        generic,             none,          true);
    OP (lt,          compare_op,          lt,            true);
    OP (luminance,   luminance,           none,          true);
    OP (matrix,      matrix,              matrix,        true);
    OP (max,         minmax,              max,           true);
    OP (mxcompassign, mxcompassign,       none,          false);
    OP (mxcompref,   mxcompref,           none,          true);
    OP (min,         minmax,              min,           true);
    OP (mix,         mix,                 mix,           true);
    OP (mod,         modulus,             none,          true);
    OP (mul,         mul,                 mul,           true);
    OP (neg,         neg,                 neg,           true);
    OP (neq,         compare_op,          neq,           true);
    OP (noise,       noise,               none,          true);
    OP (normal,      construct_triple,    triple,        true);
    OP (normalize,   generic,             none,          true);
    OP (or,          andor,               or,            true);
    OP (pnoise,      noise,               none,          true);
    OP (point,       construct_triple,    triple,        true);
    OP (pointcloud_search, pointcloud_search, pointcloud_search, false);
    OP (pointcloud_get, pointcloud_get,   pointcloud_get,false);
    OP (pointcloud_write, pointcloud_write, none,        false);
    OP (pow,         generic,             pow,           true);
    OP (printf,      printf,              none,          false);
    OP (psnoise,     noise,               none,          true);
    OP (radians,     generic,             none,          true);
    OP (raytype,     raytype,             none,          true);
    OP (regex_match, regex,               none,          false);
    OP (regex_search, regex,              regex_search,  false);
    OP (return,      return,              none,          false);
    OP (round,       generic,             none,          true);
    OP (setmessage,  setmessage,          setmessage,    false);
    OP (shl,         bitwise_binary_op,   none,          true);
    OP (shr,         bitwise_binary_op,   none,          true);
    OP (sign,        generic,             none,          true);
    OP (sin,         generic,             none,          true);
    OP (sincos,      sincos,              none,          false);
    OP (sinh,        generic,             none,          true);
    OP (smoothstep,  generic,             none,          true);
    OP (snoise,      noise,               none,          true);
    OP (spline,      spline,              none,          true);
    OP (splineinverse, spline,            none,          true);
    OP (split,       split,               split,         false);
    OP (sqrt,        generic,             sqrt,          true);
    OP (startswith,  generic,             none,          true);
    OP (step,        generic,             none,          true);
    OP (stof,        generic,             stof,          true);
    OP (stoi,        generic,             stoi,          true);
    OP (strlen,      generic,             strlen,        true);
    OP2(strtof,stof, generic,             stof,          true);
    OP2(strtoi,stoi, generic,             stoi,          true);
    OP (sub,         sub,                 sub,           true);
    OP (substr,      generic,             none,          true);
    OP (surfacearea, get_simple_SG_field, none,          true);
    OP (tan,         generic,             none,          true);
    OP (tanh,        generic,             none,          true);
    OP (texture,     texture,             texture,       true);
    OP (texture3d,   texture3d,           none,          true);
    OP (trace,       trace,               none,          false);
    OP (transform,   transform,           transform,     true);
    OP (transformn,  transform,           transform,     true);
    OP (transformv,  transform,           transform,     true);
    OP (transpose,   generic,             none,          true);
    OP (trunc,       generic,             none,          true);
    OP (useparam,    useparam,            useparam,      false);
    OP (vector,      construct_triple,    triple,        true);
    OP (warning,     printf,              warning,       false);
    OP (wavelength_color, blackbody,      none,          true);
    OP (while,       loop_op,             none,          false);
    OP (xor,         bitwise_binary_op,   none,          true);
#undef OP
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
ShadingSystemImpl::register_closure(const char *name, int id, const ClosureParam *params,
                                    PrepareClosureFunc prepare, SetupClosureFunc setup, CompareClosureFunc compare)
{
    for (int i = 0; params && params[i].type != TypeDesc(); ++i) {
        if (params[i].key == NULL && params[i].type.size() != (size_t)params[i].field_size) {
            error ("Parameter %d of '%s' closure is assigned to a field of incompatible size", i + 1, name);
            return;
        }
    }
    m_closure_registry.register_closure(name, id, params, prepare, setup, compare);
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
ShadingSystemImpl::attribute (const std::string &name, TypeDesc type,
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
        return OIIO::optparser (*(ShadingSystem *)this, *(const char **)val);
    }

    lock_guard guard (m_mutex);  // Thread safety
    ATTR_SET ("statistics:level", int, m_statslevel);
    ATTR_SET ("debug", int, m_debug);
    ATTR_SET ("lazylayers", int, m_lazylayers);
    ATTR_SET ("lazyglobals", int, m_lazyglobals);
    ATTR_SET ("clearmemory", int, m_clearmemory);
    ATTR_SET ("debugnan", int, m_debugnan);
    ATTR_SET ("lockgeom", int, m_lockgeom_default);
    ATTR_SET ("optimize", int, m_optimize);
    ATTR_SET ("opt_constant_param", int, m_opt_constant_param);
    ATTR_SET ("opt_constant_fold", int, m_opt_constant_fold);
    ATTR_SET ("opt_stale_assign", int, m_opt_stale_assign);
    ATTR_SET ("opt_elide_useless_ops", int, m_opt_elide_useless_ops);
    ATTR_SET ("opt_elide_unconnected_outputs", int, m_opt_elide_unconnected_outputs);
    ATTR_SET ("opt_peephole", int, m_opt_peephole);
    ATTR_SET ("opt_coalesce_temps", int, m_opt_coalesce_temps);
    ATTR_SET ("opt_assign", int, m_opt_assign);
    ATTR_SET ("opt_mix", int, m_opt_mix);
    ATTR_SET ("opt_merge_instances", int, m_opt_merge_instances);
    ATTR_SET ("opt_fold_getattribute", int, m_opt_fold_getattribute);
    ATTR_SET ("optimize_nondebug", int, m_optimize_nondebug);
    ATTR_SET ("llvm_optimize", int, m_llvm_optimize);
    ATTR_SET ("llvm_debug", int, m_llvm_debug);
    ATTR_SET ("strict_messages", int, m_strict_messages);
    ATTR_SET ("range_checking", int, m_range_checking);
    ATTR_SET ("unknown_coordsys_error", int, m_unknown_coordsys_error);
    ATTR_SET ("greedyjit", int, m_greedyjit);
    ATTR_SET ("countlayerexecs", int, m_countlayerexecs);
    ATTR_SET ("max_warnings_per_thread", int, m_max_warnings_per_thread);
    ATTR_SET ("max_local_mem_KB", int, m_max_local_mem_KB);
    ATTR_SET ("compile_report", int, m_compile_report);
    ATTR_SET_STRING ("commonspace", m_commonspace_synonym);
    ATTR_SET_STRING ("debug_groupname", m_debug_groupname);
    ATTR_SET_STRING ("debug_layername", m_debug_layername);
    ATTR_SET_STRING ("opt_layername", m_opt_layername);
    ATTR_SET_STRING ("only_groupname", m_only_groupname);

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
ShadingSystemImpl::getattribute (const std::string &name, TypeDesc type,
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
    ATTR_DECODE ("clearmemory", int, m_clearmemory);
    ATTR_DECODE ("debugnan", int, m_debugnan);
    ATTR_DECODE ("lockgeom", int, m_lockgeom_default);
    ATTR_DECODE ("optimize", int, m_optimize);
    ATTR_DECODE ("opt_constant_param", int, m_opt_constant_param);
    ATTR_DECODE ("opt_constant_fold", int, m_opt_constant_fold);
    ATTR_DECODE ("opt_stale_assign", int, m_opt_stale_assign);
    ATTR_DECODE ("opt_elide_useless_ops", int, m_opt_elide_useless_ops);
    ATTR_DECODE ("opt_elide_unconnected_outputs", int, m_opt_elide_unconnected_outputs);
    ATTR_DECODE ("opt_peephole", int, m_opt_peephole);
    ATTR_DECODE ("opt_coalesce_temps", int, m_opt_coalesce_temps);
    ATTR_DECODE ("opt_assign", int, m_opt_assign);
    ATTR_DECODE ("opt_mix", int, m_opt_mix);
    ATTR_DECODE ("opt_merge_instances", int, m_opt_merge_instances);
    ATTR_DECODE ("opt_fold_getattribute", int, m_opt_fold_getattribute);
    ATTR_DECODE ("optimize_nondebug", int, m_optimize_nondebug);
    ATTR_DECODE ("llvm_optimize", int, m_llvm_optimize);
    ATTR_DECODE ("debug", int, m_debug);
    ATTR_DECODE ("llvm_debug", int, m_llvm_debug);
    ATTR_DECODE ("strict_messages", int, m_strict_messages);
    ATTR_DECODE ("range_checking", int, m_range_checking);
    ATTR_DECODE ("unknown_coordsys_error", int, m_unknown_coordsys_error);
    ATTR_DECODE ("greedyjit", int, m_greedyjit);
    ATTR_DECODE ("countlayerexecs", int, m_countlayerexecs);
    ATTR_DECODE ("max_warnings_per_thread", int, m_max_warnings_per_thread);
    ATTR_DECODE_STRING ("commonspace", m_commonspace_synonym);
    ATTR_DECODE_STRING ("colorspace", m_colorspace);
    ATTR_DECODE_STRING ("debug_groupname", m_debug_groupname);
    ATTR_DECODE_STRING ("debug_layername", m_debug_layername);
    ATTR_DECODE_STRING ("opt_layername", m_opt_layername);
    ATTR_DECODE_STRING ("only_groupname", m_only_groupname);
    ATTR_DECODE ("max_local_mem_KB", int, m_max_local_mem_KB);
    ATTR_DECODE ("compile_report", int, m_compile_report);

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
    ATTR_DECODE ("stat:preopt_ops", int, m_stat_preopt_ops);
    ATTR_DECODE ("stat:postopt_ops", int, m_stat_postopt_ops);
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

    out << Strutil::format ("  Derivatives needed on %d / %d symbols (%.1f%%)\n",
                            (int)m_stat_syms_with_derivs, (int)m_stat_total_syms,
                            (100.0*(int)m_stat_syms_with_derivs)/std::max((int)m_stat_total_syms,1));
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
    out << "  Largest generated function local memory size: "
        << m_stat_max_llvm_local_mem/1024 << " KB\n";
    if (m_stat_getattribute_calls) {
        out << "  getattribute calls: " << m_stat_getattribute_calls << " ("
            << Strutil::timeintervalformat (m_stat_getattribute_time, 2) << ")\n";
        out << "     (fail time "
            << Strutil::timeintervalformat (m_stat_getattribute_fail_time, 2) << ")\n";
    }
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

    size_t jitmem = 0;
    for (size_t i = 0;  i < m_llvm_jitmm_hold.size();  ++i) {
        llvm::JITMemoryManager *mm = m_llvm_jitmm_hold[i].get();
        if (mm)
            jitmem += mm->GetDefaultCodeSlabSize() * mm->GetNumCodeSlabs()
                    + mm->GetDefaultDataSlabSize() * mm->GetNumDataSlabs()
                    + mm->GetDefaultStubSlabSize() * mm->GetNumStubSlabs();
    }
    out << "    LLVM JIT memory: " << Strutil::memformat(jitmem) << '\n';

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
ShadingSystemImpl::ShaderGroupBegin (const char *groupname)
{
    if (m_in_group) {
        error ("Nested ShaderGroupBegin() calls");
        return false;
    }
    m_in_group = true;
    m_group_use = ShadUseUnknown;
    m_group_name = ustring (groupname);
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
        sgroup.name (m_group_name);
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

        merge_instances (m_curattrib->shadergroup (m_group_use));
    }

    m_in_group = false;
    m_group_use = ShadUseUnknown;
    m_group_name.clear ();

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
        error ("ConnectShaders: destination layer must follow source layer (tried to connect %s.%s -> %s.%s)\n", srclayer, srcparam, dstlayer, dstparam);
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
    dstinst->instoverride(dstcon.param)->valuesource (Symbol::ConnectedVal);
    srcinst->instoverride(srccon.param)->connected_down (true);
    srcinst->outgoing_connections (true);

    if (debug())
        m_err->message ("ConnectShaders %s %s -> %s %s\n",
                        srclayer, srcparam, dstlayer, dstparam);

    return true;
}



ShadingAttribStateRef
ShadingSystemImpl::state ()
{
    {
        // Record the state for later greedy JITing
        spin_lock lock (m_groups_to_compile_mutex);
        m_groups_to_compile.push_back (m_curattrib);
        ++m_groups_to_compile_count;
    }
    return m_curattrib;
}



void
ShadingSystemImpl::clear_state ()
{
    m_curattrib.reset (new ShadingAttribState);
}




PerThreadInfo *
ShadingSystemImpl::create_thread_info()
{
    return new PerThreadInfo();
}




void
ShadingSystemImpl::destroy_thread_info (PerThreadInfo *threadinfo)
{
    delete threadinfo;
}



ShadingContext *
ShadingSystemImpl::get_context (PerThreadInfo *threadinfo)
{
    if (! threadinfo)
        threadinfo = get_perthread_info ();
    if (threadinfo->context_pool.empty()) {
        return new ShadingContext (*this, threadinfo);
    } else {
        return threadinfo->pop_context ();
    }
}



void
ShadingSystemImpl::release_context (ShadingContext *ctx)
{
    ctx->thread_info()->context_pool.push (ctx);
}



bool
ShadingSystemImpl::execute (ShadingContext &ctx, ShadingAttribState &sas,
                            ShaderGlobals &ssg, bool run)
{
    return ctx.execute (ShadUseSurface, sas, ssg, run);
}



const void *
ShadingSystemImpl::get_symbol (ShadingContext &ctx, ustring name,
                               TypeDesc &type)
{
    Symbol *sym = ctx.symbol (ShadUseSurface, name);
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




void ClosureRegistry::register_closure(const char *name, int id, const ClosureParam *params,
                                       PrepareClosureFunc prepare, SetupClosureFunc setup, CompareClosureFunc compare)
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
OSL_NAMESPACE_EXIT


#ifndef OSL_STATIC_LIBRARY
// Symbols needed to resolve some linkage issues because we pull some
// components in from liboslcomp.
int oslparse() { return 0; }
class oslFlexLexer {
public:
    oslFlexLexer (std::istream *in, std::ostream *out);
};
oslFlexLexer::oslFlexLexer (std::istream *in, std::ostream *out) { }
#endif
