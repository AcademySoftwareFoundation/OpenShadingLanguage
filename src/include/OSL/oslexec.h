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

#pragma once

#include <memory>

#include <OSL/oslconfig.h>
#include <OSL/shaderglobals.h>
#include <OSL/rendererservices.h>

#include <OpenImageIO/refcnt.h>
#include <OpenImageIO/ustring.h>
#include <OpenImageIO/array_view.h>
#if OPENIMAGEIO_VERSION <= 10902
#include <OpenImageIO/imagebufalgo_util.h>
#endif

OSL_NAMESPACE_ENTER

class RendererServices;
class ShaderGroup;
typedef std::shared_ptr<ShaderGroup> ShaderGroupRef;
struct ClosureParam;
struct PerThreadInfo;
class ShadingContext;
class ShaderSymbol;



/// Opaque pointer to whatever the renderer uses to represent a
/// (potentially motion-blurred) coordinate transformation.
typedef const void * TransformationPtr;


// Callbacks for closure creation
typedef void (*PrepareClosureFunc)(RendererServices *, int id, void *data);
typedef void (*SetupClosureFunc)(RendererServices *, int id, void *data);


namespace pvt {
    class ShadingSystemImpl;
}



class OSLEXECPUBLIC ShadingSystem
{
public:
    ShadingSystem (RendererServices *renderer=NULL,
                   TextureSystem *texturesystem=NULL,
                   ErrorHandler *err=NULL);
    ~ShadingSystem ();

    /// Set an attribute controlling the shading system.  Return true
    /// if the name and type were recognized and the attrib was set.
    /// Documented attributes are as follows:
    /// 1. Attributes that should be exposed to users:
    ///    int statistics:level   Automatically print OSL statistics (0).
    ///    string searchpath:shader  Colon-separated path to search for .oso
    ///    string colorspace      Name of RGB color space ("Rec709")
    ///    int range_checking     Generate extra code for component & array
    ///                              range checking (1)
    ///    int debug_nan          Add extra (expensive) code to pinpoint
    ///                              when NaN/Inf happens (0).
    ///    int debug_uninit       Add extra (expensive) code to pinpoint
    ///                              use of uninitialized variables (0).
    ///    int compile_report     Issue info messages to the renderer for
    ///                              every shader compiled (0).
    ///    int max_warnings_per_thread  Number of warning calls that should be
    ///                              processed per thread (100).
    ///    int buffer_printf      Buffer printf output from shaders and
    ///                              output atomically, to prevent threads
    ///                              from interleaving lines. (1)
    ///    int profile            Perform some rudimentary profiling (0)
    ///    int no_noise           Replace noise with constant value. (0)
    ///    int no_pointcloud      Skip pointcloud lookups. (0)
    ///    int exec_repeat        How many times to run each group (1).
    /// 2. Attributes that should be set by applications/renderers that
    /// incorporate OSL:
    ///    string commonspace     Name of "common" coord system ("world")
    ///    string[] raytypes      Array of ray type names
    ///    string[] renderer_outputs
    ///                           Array of names of renderer outputs (AOVs)
    ///                              that should not be optimized away.
    ///    int unknown_coordsys_error  Should errors be issued when unknown
    ///                              coord system names are used? (1)
    ///    int connection_error   Should errors be issued when ConnectShaders
    ///                              fails to find the layer or parameter? (1)
    ///    int strict_messages    Issue error if a message is set after
    ///                              being queried (1).
    ///    int error_repeats      If zero, suppress repeats of errors and
    ///                              warnings that are exact duplicates of
    ///                              earlier ones. (1)
    ///    int lazylayers         Evaluate shader layers only when their
    ///                              outputs are first needed (1)
    ///    int lazyglobals        Run layers lazily even if they write to
    ///                              globals (1)
    ///    int lazyunconnected    Run layers lazily even if they have no
    ///                              output connections (1). For debugging.
    ///    int lazy_userdata      Retrieve userdata lazily (0).
    ///    int userdata_isconnected  Should lockgeom=0 params (that may
    ///                              receive userdata) return true from
    ///                              isconnected()? (0)
    ///    int greedyjit          Optimize and compile all shaders up front,
    ///                              versus only as needed (0).
    ///    int lockgeom           Default 'lockgeom' value for shader params
    ///                              that don't specify it (1).  Lockgeom
    ///                              means a param CANNOT be overridden by
    ///                              interpolated geometric parameters.
    ///    int countlayerexecs    Add extra code to count total layers run.
    ///    int allow_shader_replacement Allow shader to be specified more than
    ///                              once, replacing former definition.
    ///    string archive_groupname  Name of a group to pickle and archive.
    ///    string archive_filename   Name of file to save the group archive.
    /// 3. Attributes that that are intended for developers debugging
    /// liboslexec itself:
    /// These attributes may be helpful for liboslexec developers or
    /// for debugging, but probably not for using OSL in production:
    ///    int debug              Set debug output level (0)
    ///    int clearmemory        Zero out working memory before each shade (0)
    ///    int optimize           Runtime optimization level (2)
    ///       And there are several int options that, if set to 0, will turn
    ///       off individual classes of runtime optimizations:
    ///         opt_simplify_param, opt_constant_fold, opt_stale_assign,
    ///         opt_elide_useless_ops, opt_elide_unconnected_outputs,
    ///         opt_peephole, opt_coalesce_temps, opt_assign, opt_mix
    ///         opt_merge_instances, opt_merge_instance_with_userdata,
    ///         opt_fold_getattribute, opt_middleman, opt_texture_handle
    ///         opt_seed_bblock_aliases
    ///    int opt_passes         Number of optimization passes per layer (10)
    ///    int llvm_optimize      Which of several LLVM optimize strategies (0)
    ///    int llvm_debug         Set LLVM extra debug level (0)
    ///    int llvm_debug_layers  Extra printfs upon entering and leaving
    ///                              layer functions.
    ///    int llvm_debug_ops     Extra printfs for each OSL op (helpful
    ///                              for devs to find crashes)
    ///    int llvm_output_bitcode  Output the full bitcode for each group,
    ///                              for debugging. (0)
    ///    int max_local_mem_KB   Error if shader group needs more than this
    ///                              much local storage to execute (1024K)
    ///    string debug_groupname Name of shader group -- debug only this one
    ///    string debug_layername Name of shader layer -- debug only this one
    ///    int optimize_nondebug  If 1, fully optimize shaders that are not
    ///                              designated as the debug shaders.
    ///    string opt_layername   If set, only optimize the named layer
    ///    string only_groupname  Compile only this one group (skip all others)
    ///    int force_derivs       Force all float-based variables to compute
    ///                              and store derivatives. (0)
    ///
    /// Note: the attributes referred to as "string" are actually on the app
    /// side as ustring or const char* (they have the same data layout), NOT
    /// std::string!
    bool attribute (string_view name, TypeDesc type, const void *val);

    // Shortcuts for common types
    bool attribute (string_view name, int val) {
        return attribute (name, TypeDesc::INT, &val);
    }
    bool attribute (string_view name, float val) {
        return attribute (name, TypeDesc::FLOAT, &val);
    }
    bool attribute (string_view name, double val) {
        float f = (float) val;
        return attribute (name, TypeDesc::FLOAT, &f);
    }
    bool attribute (string_view name, string_view val) {
        const char *s = val.c_str();
        return attribute (name, TypeDesc::STRING, &s);
    }


    /// Set an attribute for a specific shader group.  Return true if the
    /// name and type were recognized and the attrib was set. Documented
    /// attributes are as follows:
    ///    string[] renderer_outputs  Array of names of renderer outputs
    ///                                 (AOVs) specific to this shader group
    ///                                 that should not be optimized away.
    ///    string[] entry_layers      Array of names of layers that may be
    ///                                 callable entry points. They won't
    ///                                 be elided, but nor will they be
    ///                                 called unconditionally.
    ///    int exec_repeat            How many times to run the group (1).
    ///
    bool attribute (ShaderGroup *group, string_view name,
                    TypeDesc type, const void *val);
    bool attribute (ShaderGroup *group, string_view name, int val) {
        return attribute (group, name, TypeDesc::INT, &val);
    }
    bool attribute (ShaderGroup *group, string_view name, float val) {
        return attribute (group, name, TypeDesc::FLOAT, &val);
    }
    bool attribute (ShaderGroup *group, string_view name, double val) {
        float f = (float) val;
        return attribute (group, name, TypeDesc::FLOAT, &f);
    }
    bool attribute (ShaderGroup *group, string_view name, string_view val) {
        const char *s = val.c_str();
        return attribute (group, name, TypeDesc::STRING, &s);
    }

    /// Get the named attribute, store it in value.
    ///
    bool getattribute (string_view name, TypeDesc type, void *val);

    // Shortcuts for common types
    bool getattribute (string_view name, int &val) {
        return getattribute (name, TypeDesc::INT, &val);
    }
    bool getattribute (string_view name, float &val) {
        return getattribute (name, TypeDesc::FLOAT, &val);
    }
    bool getattribute (string_view name, double &val) {
        float f;
        bool ok = getattribute (name, TypeDesc::FLOAT, &f);
        if (ok)
            val = f;
        return ok;
    }
    bool getattribute (string_view name, char **val) {
        return getattribute (name, TypeDesc::STRING, val);
    }
    bool getattribute (string_view name, ustring &val) {
        return getattribute (name, TypeDesc::STRING, (char **)&val);
    }
    bool getattribute (string_view name, std::string &val) {
        const char *s = NULL;
        bool ok = getattribute (name, TypeDesc::STRING, &s);
        if (ok)
            val = s;
        return ok;
    }

    /// Get the named attribute about a particular shader group, store it
    /// in value.  Attributes that are currently documented include:
    ///   string groupname           The name of the shader group.
    ///   int num_layers             The number of layers in the group.
    ///   string[] layer_names       The names of the layers in the group.
    ///   int num_textures_needed    The number of texture names that are
    ///                                known to be potentially needed by the
    ///                                group (after optimization).
    ///   ptr textures_needed        Retrieves a pointer to the ustring array
    ///                                containing all textures known to be
    ///                                needed.
    ///   int unknown_textures_needed  Nonzero if additional textures may be
    ///                                needed, whose names can't be known
    ///                                without actually running the shader.
    ///   int num_closures_needed    The number of named closures needed.
    ///   ptr closures_needed        Retrieves a pointer to the ustring array
    ///                                containing all closures known to be
    ///                                needed.
    ///   int unknown_closures_needed  Nonzero if additional closures may be
    ///                                needed, whose names can't be known
    ///                                without actually running the shader.
    ///   int num_globals_needed     The number of named globals needed.
    ///   ptr globals_needed         Retrieves a pointer to the ustring array
    ///                                containing all globals needed.
    ///   int num_userdata           The number of "user data" variables
    ///                                retrieved by the shader.
    ///   ptr userdata_names         Retrieves a pointer to the array of
    ///                                ustring holding the userdata names.
    ///   ptr userdata_types         Retrieves a pointer to the array of
    ///                                 TypeDesc describing the userdata.
    ///   ptr userdata_offsets       Retrieves a pointer to the array of
    ///                                 int describing the userdata offsets
    ///                                 within the heap.
    ///   int num_attributes_needed  The number of attribute/scope pairs that
    ///                                are known to be queried by the group (the
    ///                                length of the attributes_needed and
    ///                                attribute_scopes arrays).
    ///   ptr attributes_needed      Retrieves a pointer to the ustring array
    ///                                containing the names of the needed attributes.
    ///	                               Note that if the same attribute
    ///                                is requested in multiple scopes, it will
    ///                                appear in the array multiple times - once for
    ///                                each scope in which is is queried.
    ///   ptr attribute_scopes       Retrieves a pointer to a ustring array containing
    ///                                the scopes associated with each attribute query
    ///                                in the attributes_needed array.
    ///   int unknown_attributes_needed  Nonzero if additonal attributes may be
    ///                                  needed, whose names will not be known
    ///                                  until the shader actually runs.
    ///   int num_renderer_outputs   Number of named renderer outputs.
    ///   string renderer_outputs[]  List of renderer outputs.
    ///   int raytype_queries        Bit field of all possible rayquery
    ///   int num_entry_layers       Number of named entry point layers.
    ///   string entry_layers[]      List of entry point layers.
    ///   string pickle              Retrieves a serialized representation
    ///                                 of the shader group declaration.
    /// Note: the attributes referred to as "string" are actually on the app
    /// side as ustring or const char* (they have the same data layout), NOT
    /// std::string!
    bool getattribute (ShaderGroup *group, string_view name,
                       TypeDesc type, void *val);
    // Shortcuts for common types
    bool getattribute (ShaderGroup *group, string_view name, int &val) {
        return getattribute (group, name, TypeDesc::INT, &val);
    }
    bool getattribute (ShaderGroup *group, string_view name, float &val) {
        return getattribute (group, name, TypeDesc::FLOAT, &val);
    }
    bool getattribute (ShaderGroup *group, string_view name, double &val) {
        float f;
        bool ok = getattribute (group, name, TypeDesc::FLOAT, &f);
        if (ok)
            val = f;
        return ok;
    }
    bool getattribute (ShaderGroup *group, string_view name, char **val) {
        return getattribute (group, name, TypeDesc::STRING, val);
    }
    bool getattribute (ShaderGroup *group, string_view name, ustring &val) {
        return getattribute (group, name, TypeDesc::STRING, (char **)&val);
    }
    bool getattribute (ShaderGroup *group, string_view name, std::string &val) {
        const char *s = NULL;
        bool ok = getattribute (group, name, TypeDesc::STRING, &s);
        if (ok)
            val = s;
        return ok;
    }


    /// Load compiled shader (oso) from a memory buffer, overriding
    /// shader lookups in the shader search path
    bool LoadMemoryCompiledShader (string_view shadername,
                                   string_view buffer);

    // The basic sequence for declaring a shader group looks like this:
    // ShadingSystem *ss = ...;
    // ShaderGroupRef group = ss->ShaderGroupBegin (groupname);
    //    /* First layer - texture lookup shader: */
    //       /* Specify instance parameter values */
    //       const char *mapname = "colormap.exr";
    //       ss->Parameter ("texturename", TypeDesc::TypeString, &mapname);
    //       float blur = 0.001;
    //       ss->Parameter ("blur", TypeDesc::TypeFloat, &blur);
    //    ss->Shader ("surface", "texmap", "texturelayer");
    //    /* Second layer - generate the BSDF closure: */
    //       float roughness = 0.05;
    //       ss->Parameter ("roughness", TypeDesc::TypeFloat, &roughness);
    //    ss->Shader ("surface", "plastic", "illumlayer");
    //    /* Make a connection between the layers */
    //    ss->ConnectShaders ("texturelayer", "Cout", "illumlayer", "Cs");
    // ss->ShaderGroupEnd ();

    /// Signal the start of a new shader group.  The return value is a
    /// reference-counted opaque handle to the ShaderGroup.
    ShaderGroupRef ShaderGroupBegin (string_view groupname = string_view());

    /// Alternate way to specify a shader group. The group specification
    /// syntax looks like this: (as a string, all whitespace is equivalent):
    ///     param <typename> <paramname> <value>... [[hints]] ;
    ///     shader <shadername> <layername> ;
    ///     connect <layername>.<paramname> <layername>.<paramname> ;
    /// For the sake of easy assembling on command lines, a comma ',' may
    /// substitute for the semicolon as a separator, and the last separator
    /// before the end of the string is optional.
    ShaderGroupRef ShaderGroupBegin (string_view groupname,
                                     string_view shaderusage,
                                     string_view groupspec = string_view());

    /// Signal the end of a new shader group.
    ///
    bool ShaderGroupEnd (void);

    /// Set a parameter of the next shader.
    ///
    bool Parameter (string_view name, TypeDesc t, const void *val);

    /// Set a parameter of the next shader, and override the 'lockgeom'
    /// metadata for that parameter (despite how it may have been set in
    /// the shader).  If lockgeom is false, it means that this parameter
    /// should NOT be considered locked against changes by the geometry,
    /// and therefore the shader should not optimize assuming that the
    /// instance value (the 'val' specified by this call) is a constant.
    bool Parameter (string_view name, TypeDesc t, const void *val,
                    bool lockgeom);

    /// Create a new shader instance, either replacing the one for the
    /// specified usage (if not within a group) or appending to the
    /// current group (if a group has been started).
    bool Shader (string_view shaderusage,
                 string_view shadername = string_view(),
                 string_view layername = string_view());

    /// Connect two shaders within the current group. The source layer must
    /// be *upstream* of down destination layer (i.e. source must be
    /// declared earlier within the shader group). The named parameters must
    /// be of compatible type -- float to float, color to color, array to
    /// array of the same length and element type, etc. In general, it is
    /// permissible to connect type A to type B if and only if it is allowed
    /// within OSL to assign an A to a B (i.e., if `A = B` is legal). So any
    /// "triple" may be connected to any other triple, and a float output
    /// may be connected to a triple input (but not the other way around).
    /// It is permitted to connect a single component of an aggregate to a
    /// float and vice versa, for example,
    ///   `ConnectShaders ("lay1", "mycolorout[2]", "lay2", "myfloatinput")`
    ///
    bool ConnectShaders (string_view srclayer, string_view srcparam,
                         string_view dstlayer, string_view dstparam);

    /// Replace a parameter value in a previously-declared shader group.
    /// This is meant to called after the ShaderGroupBegin/End, but will
    /// fail if the shader has already been irrevocably optimized/compiled,
    /// unless the paraticular parameter is marked as lockgeom=0 (which
    /// indicates that it's a parameter that may be overridden by the
    /// geometric primitive).  This call gives you a way of changing the
    /// instance value, even if it's not a geometric override.
    bool ReParameter (ShaderGroup &group,
                      string_view layername, string_view paramname,
                      TypeDesc type, const void *val);

    /// Optional: create the per-thread data needed for shader
    /// execution.  Doing this and passing it to get_context speeds is a
    /// bit faster than get_context having to do a thread-specific
    /// lookup on its own, but if you do it, it's important for the app
    /// to use one and only one PerThreadInfo per renderer thread, and
    /// destroy it with destroy_thread_info when the thread terminates.
    PerThreadInfo * create_thread_info();

    /// Destroy a PerThreadInfo that was allocated by
    /// create_thread_info().
    void destroy_thread_info (PerThreadInfo *threadinfo);

    /// Get a ShadingContext that we can use.  The context is specific
    /// to the renderer thread.  The 'threadinfo' parameter should be a
    /// thread-specific pointer created by create_thread_info, or NULL,
    /// in which case the ShadingSystem will do the thread-specific
    /// lookup automatically (and at some additional cost).  The context
    /// can be used to shade many points; a typical usage is to allocate
    /// just one context per thread and use it for the whole run.
    ShadingContext *get_context (PerThreadInfo *threadinfo=NULL,
                                 TextureSystem::Perthread *texture_threadinfo=NULL);

    /// Return a ShadingContext to the pool.
    ///
    void release_context (ShadingContext *ctx);

    /// Execute the shader group in this context. If ctx is NULL, then
    /// execute will request one (based on the running thread) on its own
    /// and then return it when it's done.  This is just a wrapper around
    /// execute_init, execute_layer of the last (presumably group entry)
    /// layer, and execute_cleanup. If run==false, just do the binding and
    /// setup, don't actually run the shader.
    bool execute (ShadingContext *ctx, ShaderGroup &group,
                  ShaderGlobals &globals, bool run=true);

    /// Bind a shader group and globals to the context, in preparation to
    /// execute, including optimization and JIT of the group (if it has not
    /// already been done).  If 'run' is true, also run any initialization
    /// necessary. If 'run' is false, we are not planning to actually
    /// execute any part of the shader, so do all the usual binding
    /// preparation, but don't actually run the shader.  Return true if the
    /// shader executed, false if it did not (including if the shader itself
    /// was empty).
    bool execute_init (ShadingContext &ctx, ShaderGroup &group,
                       ShaderGlobals &globals, bool run=true);

    /// Execute the layer whose index is specified, in this context. It is
    /// presumed that execute_init() has already been called, with
    /// run==true, and that the call to execute_init() returned true. (One
    /// reason why it might have returned false is if the shader group
    /// turned out, after optimization, to do nothing.)
    bool execute_layer (ShadingContext &ctx, ShaderGlobals &globals,
                        int layernumber);
    /// Execute the layer by name.
    bool execute_layer (ShadingContext &ctx, ShaderGlobals &globals,
                        ustring layername);
    /// Execute the layer that has the given ShaderSymbol as an output.
    /// (The symbol is one returned by find_symbol()).
    bool execute_layer (ShadingContext &ctx, ShaderGlobals &globals,
                        const ShaderSymbol *symbol);

    /// Signify that the context is done with the current execution of the
    /// group that was kicked off by execute_init and one or more calls to
    /// execute_layer.
    bool execute_cleanup (ShadingContext &ctx);

    /// Find the named layer within a group and return its index, or -1
    /// if no such named layer exists.
    int find_layer (const ShaderGroup &group, ustring layername) const;

    /// Get a raw pointer to a named symbol (such as you'd need to pull
    /// out the value of an output parameter).  ctx is the shading
    /// context (presumably already run), name is the name of the
    /// symbol.  If found, get_symbol will return the pointer to the
    /// symbol's data, and type will get the symbol's type.  If the
    /// symbol is not found, get_symbol will return NULL.
    /// If you give just a symbol name, it will search for the symbol in all
    /// layers, last-to-first. If a specific layer is named, it will search
    /// only that layer. You can specify a layer either by naming it
    /// separately, or by concatenating "layername.symbolname", but note
    /// that the latter will involve string manipulation inside get_symbol
    /// and is much more expensive than specifying them separately.
    ///
    /// These are considered somewhat deprecated, in favor of using
    /// find_symbol(), symbol_typedesc(), and symbol_address().
    const void* get_symbol (const ShadingContext &ctx, ustring layername,
                            ustring symbolname, TypeDesc &type) const;
    const void* get_symbol (const ShadingContext &ctx, ustring symbolname,
                            TypeDesc &type) const;

    /// Search for an output symbol by name (and optionally, layer) within
    /// the optimized shader group. If the symbol is found, return an opaque
    /// identifying pointer to it, otherwise return NULL. This is somewhat
    /// expensive because of the name-based search, but once done, you can
    /// reuse the pointer to the symbol for the lifetime of the group.
    ///
    /// If you give just a symbol name, it will search for the symbol in all
    /// layers, last-to-first. If a specific layer is named, it will search
    /// only that layer. You can specify a layer either by naming it
    /// separately, or by concatenating "layername.symbolname", but note
    /// that the latter will involve string manipulation inside find_symbol
    /// and is much more expensive than specifying them separately.
    const ShaderSymbol* find_symbol (const ShaderGroup &group,
                             ustring layername, ustring symbolname) const;
    const ShaderSymbol* find_symbol (const ShaderGroup &group,
                                     ustring symbolname) const;

    /// Given an opaque ShaderSymbol*, return the TypeDesc describing it.
    /// Note that a closure will end up with a TypeDesc::UNKNOWN value.
    TypeDesc symbol_typedesc (const ShaderSymbol *sym) const;

    /// Given a context (that has executed a shader) and an opaque
    /// ShserSymbol*, return the actual memory address where the value of
    /// the symbol resides within the heap memory of the context. This
    /// is only valid for the shader execution that had happened immediately
    /// prior for this context, but it is a very inexpensive operation.
    const void* symbol_address (const ShadingContext &ctx,
                                const ShaderSymbol *sym) const;

    /// Return the statistics output as a huge string.
    ///
    std::string getstats (int level=1) const;

    void register_closure (string_view name, int id, const ClosureParam *params,
                           PrepareClosureFunc prepare, SetupClosureFunc setup);

    /// Query either by name or id an existing closure. If name is non
    /// NULL it will use it for the search, otherwise id would be used
    /// and the name will be placed in name if successful. Also return
    /// pointer to the params array in the last argument. All args are
    /// optional but at least one of name or id must non NULL.
    bool query_closure (const char **name, int *id,
                        const ClosureParam **params);


    /// Register a device_string tag for the specified string. Return
    /// false if the string has been registered with a different tag.
    bool register_string_tag (string_view str, uint64_t tag);

    /// Lookup the tag registered for the given string.
    /// Return StringTags::UNKNOWNSTRING if the string is not registered.
    uint64_t lookup_string_tag (string_view str);

    /// For the proposed raytype name, return the bit pattern that
    /// describes it, or 0 for an unrecognized name.  (This retrieves
    /// data passed in via attribute("raytypes")).
    int raytype_bit (ustring name);

    /// Configure the default raytypes to assume to be on (or off) at optimization
    /// time for the given group. The raytypes_on gives a bitfield describing which
    /// ray flags are known to be 1, and raytypes_off describes which ray flags are
    /// known to be 0. Bits that are not set in either set of flags are not known
    /// to the optimizer, and will be determined strictly at execution time.
    void set_raytypes(ShaderGroup *group, int raytypes_on, int raytypes_off);

    /// Ensure that the group has been optimized and JITed.
    /// Ensure that the group has been optimized and JITed.
    void optimize_group (ShaderGroup *group);

    /// Ensure that the group has been optimized and JITed. This is a
    /// convenience function that simply calls set_raytypes followed by optimize_group.
    void optimize_group (ShaderGroup *group, int raytypes_on,
                         int raytypes_off);

    /// If option "greedyjit" was set, this call will trigger all
    /// shader groups that have not yet been compiled to do so with the
    /// specified number of threads (0 means use all available HW cores).
    void optimize_all_groups (int nthreads=0);

    /// Return a pointer to the TextureSystem being used.
    TextureSystem * texturesys () const;

    /// Return a pointer to the RendererServices being used.
    RendererServices * renderer () const;

    /// Archive the entire shader group so that it can be reconstituted
    /// later.
    bool archive_shadergroup (ShaderGroup *group, string_view filename);

    /// Helper function -- copy or convert a source value (described by
    /// srctype) to destination (described by dsttype).  The function
    /// returns true upon success, or false if the types differ in a way
    /// that cannot be converted.  As a special case, if dst==NULL or
    /// src==NULL, no copying is performed, and convert_value merely
    /// returns a bool indicating if the proposed type conversion is
    /// allowed.
    ///
    /// The following type conversions are supported:
    /// 1. Identical types copy without modification.
    /// 2. Conversions following the same rules as type casting and
    /// assignment in OSL itself:
    ///   int -> float             convert to float
    ///   int -> triple            convert to float and replicate x3
    ///   float -> triple          replicate x3
    ///   float -> int             truncate like a (int) type cast
    ///   triple -> triple         copy, regarless of differing vector types
    /// 3. Additional rules not allowed in OSL source code:
    ///   float -> float[2]        replicate x2
    ///   int -> float[2]          convert to float and replicate x2
    ///   float[2] -> triple       (f[0], f[1], 0)
    ///
    /// Observation: none of the supported conversions require more
    /// storage for src than for dst.
    static bool convert_value (void *dst, TypeDesc dsttype,
                               const void *src, TypeDesc srctype);

private:
    pvt::ShadingSystemImpl *m_impl;
};



#ifdef OPENIMAGEIO_IMAGEBUFALGO_H
// To keep from polluting all OSL clients with ImageBuf & ROI, only expose
// the following declarations if they have included OpenImageIO/imagebufalgo.h.

// enum describing where shades are located for shade_image().
enum ShadeImageLocations {
    ShadePixelCenters,   // locate shades at pixel centers: (i+0.5)/res
    ShadePixelGrid       // locate shades at grid nodes: i/(res-1)
};


/// Utility to execute a shader group on each pixel in a rectangular region
/// of an ImageBuf (which must already be allocated and which must have
/// FLOAT pixels).  The output parameters to save are specified by an array
/// of ustring values in 'outputs'. If there are multiple outputs, they will
/// simply be concatenated channel by channel in the image.
///
/// The roi specifies the region of the ImageBuf to shade (defaulting to the
/// whole thing), any pixels outside the roi will not be altered.
///
/// The 'defaultsg', if non-NULL, provides a template for the default
/// ShaderGlobals to use for each point. If not provided, reasonable
/// defaults will be chosen.
///
/// When shading, P will have the pixel lattice coordinates (i,j,k), and u
/// and v will vary from 0->1 across the full (aka "display") window.
/// Depending on the value of 'shadelocations', the shading locations
/// themselves will either be at "pixel centers" (position (i+0.5)/res), or
/// as if it were a grid that is shaded at exact endpoints (position
/// i/(res+1)). In either case, derivatives will be set appropriately.
OSLEXECPUBLIC
bool shade_image (ShadingSystem &shadingsys, ShaderGroup &group,
                  const ShaderGlobals *defaultsg,
                  OIIO::ImageBuf &buf, OIIO::array_view<ustring> outputs,
                  ShadeImageLocations shadelocations = ShadePixelCenters,
                  OIIO::ROI roi = OIIO::ROI(),
                  OIIO::ImageBufAlgo::parallel_image_options popt = 0);

#endif


OSL_NAMESPACE_EXIT
