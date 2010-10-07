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

#ifndef OSLEXEC_PVT_H
#define OSLEXEC_PVT_H

#include <string>
#include <vector>
#include <stack>
#include <map>
#include <list>
#include <set>

#include <boost/regex_fwd.hpp>

#include "OpenImageIO/hash.h"
#include "OpenImageIO/ustring.h"
#include "OpenImageIO/thread.h"
#include "OpenImageIO/paramlist.h"

#include "oslexec.h"
#include "oslclosure.h"
#include "osl_pvt.h"
#include "constantpool.h"


using namespace OSL;
using namespace OSL::pvt;

namespace llvm {
  class ExecutionEngine;
  class Function;
  class FunctionPassManager;
  class LLVMContext;
  class Linker;
  class Module;
  class PassManager;
}

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {


// forward definitions
class ShadingSystemImpl;
class ShadingContext;
class ShadingExecution;
class ShaderInstance;
typedef shared_ptr<ShaderInstance> ShaderInstanceRef;


/// Signature of the function that LLVM generates to run the shader
/// group.
typedef void (*RunLLVMGroupFunc)(void* /* shader globals */, void*); 



/// Like an int (of type T), but also internally keeps track of the 
/// maximum value is has held, and the total "requested" deltas.
template<typename T>
class PeakCounter
{
public:
    typedef T value_t;
    PeakCounter () : m_current(0), m_requested(0), m_peak(0) { }
    /// Reset all counts to zero.
    ///
    void clear () {
        m_current = 0;  m_requested = 0;  m_peak = 0;
    }
    /// Return the current value.
    ///
    value_t operator() () const { return m_current; }

    /// Return the current value.
    ///
    value_t current (void) const { return m_current; }
    /// Return the sum of all requests.
    ///
    value_t requested (void) const { return m_requested; }
    /// Return the peak value we saw.
    ///
    value_t peak (void) const { return m_peak; }

    /// Reassign the current value, adjust peak and requested as necessary.
    ///
    const value_t operator= (value_t newval) {
        value_t cur = m_current;
        if (newval > cur)
            m_requested += (cur-newval);
        m_current = newval;
        if (newval > m_peak)
            m_peak = newval;
        return m_current;
    }
    /// Add to current value, adjust peak and requested as necessary.
    ///
    const value_t operator+= (value_t sz) {
        m_requested += sz;
        m_current += sz;
        if (m_current > m_peak)
            m_peak = m_current;
        return m_current;
    }
    /// Add to current value, adjust peak and requested as necessary.
    ///
    const value_t operator-= (value_t sz) {
        m_current -= sz;
        return m_current;
    }
    const value_t operator++ ()    { *this += 1;  return m_current; }
    const value_t operator++ (int) { *this += 1;  return m_current-1; }
    const value_t operator-- ()    { *this -= 1;  return m_current; }
    const value_t operator-- (int) { *this -= 1;  return m_current+1; }

    friend std::ostream & operator<< (std::ostream &out, const PeakCounter &p)
    {
        out << p.requested() << " requested, " << p.peak() << " peak, "
            << p.current() << " current";
        return out;
    }
private:
    value_t m_current, m_requested, m_peak;
};



/// ShaderMaster is the full internal representation of a complete
/// shader that would be a .oso file on disk: symbols, instructions,
/// arguments, you name it.  A master copy is shared by all the 
/// individual instances of the shader.
class ShaderMaster : public RefCnt {
public:
    typedef intrusive_ptr<ShaderMaster> ref;
    ShaderMaster (ShadingSystemImpl &shadingsys) : m_shadingsys(shadingsys) { }
    ~ShaderMaster () { }

    std::string print ();  // Debugging

    /// Return a pointer to the shading system for this master.
    ///
    ShadingSystemImpl & shadingsys () const { return m_shadingsys; }

    /// Run through the symbols and set up various things we can know
    /// with just the master: the size (including padding), and their
    /// data pointers if they are constants or params (to the defaults).
    /// As a side effect, also set this->m_firstparam/m_lastparam.
    void resolve_syms ();

    /// Find the named symbol, return its index in the symbol array, or
    /// -1 if not found.
    int findsymbol (ustring name) const;

    /// Return a pointer to the symbol (specified by integer index),
    /// or NULL (if index was -1, as returned by 'findsymbol').
    Symbol *symbol (int index) { return index >= 0 ? &m_symbols[index] : NULL; }

    /// Return the name of the shader.
    ///
    const std::string &shadername () const { return m_shadername; }

private:
    ShadingSystemImpl &m_shadingsys;    ///< Back-ptr to the shading system
    ShaderType m_shadertype;            ///< Type of shader
    std::string m_shadername;           ///< Shader name
    std::string m_osofilename;          ///< Full path of oso file
    OpcodeVec m_ops;                    ///< Actual code instructions
    std::vector<int> m_args;            ///< Arguments for all the ops
    // Need the code offsets for each code block
    SymbolVec m_symbols;                ///< Symbols used by the shader
    std::vector<int> m_idefaults;       ///< int default param values
    std::vector<float> m_fdefaults;     ///< float default param values
    std::vector<ustring> m_sdefaults;   ///< string default param values
    std::vector<int> m_iconsts;         ///< int constant values
    std::vector<float> m_fconsts;       ///< float constant values
    std::vector<ustring> m_sconsts;     ///< string constant values
    int m_firstparam, m_lastparam;      ///< Subset of symbols that are params
    int m_maincodebegin, m_maincodeend; ///< Main shader code range

    friend class OSOReaderToMaster;
    friend class ShaderInstance;
    friend class ShadingExecution;
};



/// Describe one end of a parameter connetion: the parameter number, and
/// optinally an array index and/or channel number within that parameter.
struct ConnectedParam {
    int param;            ///< Parameter number (in the symbol table)
    int arrayindex;       ///< Array index (-1 for not an index)
    int channel;          ///< Channel number (-1 for no channel selection)
    int offset;           ///< Offset into the data of the element/channel
    TypeSpec type;        ///< Type of data being connected

    ConnectedParam () : param(-1), arrayindex(-1), channel(-1), offset(0) { }

    bool valid () const { return (param >= 0); }
};



/// Describe a parameter connection to an earlier layer.
///
struct Connection {
    int srclayer;          ///< Layer (within our group) of the source
    ConnectedParam src;    ///< Which source parameter (or part thereof)
    ConnectedParam dst;    ///< Which destination parameter (or part thereof)

    Connection (int srclay, const ConnectedParam &srccon,
                const ConnectedParam &dstcon)
        : srclayer (srclay), src (srccon), dst (dstcon)
    { }
};



typedef std::vector<Connection> ConnectionVec;



/// ShaderInstance is a particular instance of a shader, with its own
/// set of parameter values, coordinate transform, and connections to
/// other instances within the same shader group.
class ShaderInstance {
public:
    typedef ShaderInstanceRef ref;
    ShaderInstance (ShaderMaster::ref master, const char *layername="");
    ~ShaderInstance ();

    /// Return the layer name of this instance
    ///
    ustring layername () const { return m_layername; }

    /// Return the name of the shader used by this instance.
    ///
    const std::string &shadername () const { return m_master->shadername(); }

    /// Return a pointer to the master for this instance.
    ///
    ShaderMaster *master () const { return m_master.get(); }

    /// Return a reference to the shading system for this instance.
    ///
    ShadingSystemImpl & shadingsys () const { return m_master->shadingsys(); }

    /// Apply pending parameters
    /// 
    void parameters (const ParamValueList &params);

    /// Find the named symbol, return its index in the symbol array, or
    /// -1 if not found.
    int findsymbol (ustring name) const;

    /// Find the named parameter, return its index in the symbol array, or
    /// -1 if not found.
    int findparam (ustring name) const;

    /// Return a pointer to the symbol (specified by integer index),
    /// or NULL (if index was -1, as returned by 'findsymbol').
    Symbol *symbol (int index) { return index >= 0 ? &m_instsymbols[index] : NULL; }
    const Symbol *symbol (int index) const { return index >= 0 ? &m_instsymbols[index] : NULL; }

    /// Estimate how much to round the required heap size up if npoints
    /// is odd, to account for getting the desired alignment for each
    /// symbol.
    size_t heapround ();

    /// Add a connection
    ///
    void add_connection (int srclayer, const ConnectedParam &srccon,
                         const ConnectedParam &dstcon) {
        m_connections.push_back (Connection (srclayer, srccon, dstcon));
    }

    /// How many connections to earlier layers do we have?
    ///
    int nconnections () const { return (int) m_connections.size (); }

    /// Return a reference to the i-th connection to an earlier layer.
    ///
    const Connection & connection (int i) const { return m_connections[i]; }

    /// Reference to the connection list.
    ///
    ConnectionVec & connections () { return m_connections; }
    const ConnectionVec & connections () const { return m_connections; }

    /// Return the unique ID of this instance.
    ///
    int id () const { return m_id; }

    /// Does this instance potentially write to any global vars?
    ///
    bool writes_globals () const { return m_writes_globals; }

    /// Should this instance only be run lazily (i.e., not
    /// unconditionally)?
    bool run_lazily () const { return m_run_lazily; }
    void run_lazily (bool lazy) { m_run_lazily = lazy; }

    /// Does this instance have any outgoing connections?
    ///
    bool outgoing_connections () const { return m_outgoing_connections; }
    /// Set whether this instance has outgoing connections.
    ///
    void outgoing_connections (bool out) { m_outgoing_connections = out; }

    int maincodebegin () const { return m_maincodebegin; }
    int maincodeend () const { return m_maincodeend; }

    int firstparam () const { return m_firstparam; }
    int lastparam () const { return m_lastparam; }

    /// Return a begin/end Symbol* pair for the set of param symbols
    /// that is suitable to pass as a range for BOOST_FOREACH.
    friend std::pair<Symbol *,Symbol *> param_range (ShaderInstance *i) {
        return std::pair<Symbol*,Symbol*> (i->symbol(i->firstparam()),
                                           i->symbol(i->lastparam()));
    }

    friend std::pair<const Symbol *,const Symbol *> param_range (const ShaderInstance *i) {
        return std::pair<const Symbol*,const Symbol*> (i->symbol(i->firstparam()),
                                                       i->symbol(i->lastparam()));
    }

    int Psym () const { return m_Psym; }
    int Nsym () const { return m_Nsym; }


    const std::vector<int> & args () const { return m_instargs; }
    std::vector<int> & args () { return m_instargs; }
    int arg (int argnum) { return args()[argnum]; }
    Symbol *argsymbol (int argnum) { return symbol(arg(argnum)); }
    const OpcodeVec & ops () const { return m_instops; }
    OpcodeVec & ops () { return m_instops; }

    std::string print ();  // Debugging

    SymbolVec &symbols () { return m_instsymbols; }
    const SymbolVec &symbols () const { return m_instsymbols; }

    /// Make sure there's room for more symbols.
    ///
    void make_symbol_room (size_t moresyms=1);

    /// Does it appear that the layer is completely unused?
    ///
    bool unused () const { return run_lazily() && ! outgoing_connections(); }

private:
    ShaderMaster::ref m_master;         ///< Reference to the master
    SymbolVec m_instsymbols;            ///< Symbols used by the instance
    OpcodeVec m_instops;                ///< Actual code instructions
    std::vector<int> m_instargs;        ///< Arguments for all the ops
    ustring m_layername;                ///< Name of this layer
    std::vector<int> m_iparams;         ///< int param values
    std::vector<float> m_fparams;       ///< float param values
    std::vector<ustring> m_sparams;     ///< string param values
    int m_id;                           ///< Unique ID for the instance
    bool m_writes_globals;              ///< Do I have side effects?
    bool m_run_lazily;                  ///< OK to run this layer lazily?
    bool m_outgoing_connections;        ///< Any outgoing connections?
    std::vector<Connection> m_connections; ///< Connected input params
    int m_firstparam, m_lastparam;      ///< Subset of symbols that are params
    int m_maincodebegin, m_maincodeend; ///< Main shader code range
    int m_Psym, m_Nsym;                 ///< Quick lookups of common syms

    friend class ShadingExecution;
    friend class ShadingSystemImpl;
    friend class RuntimeOptimizer;
};



/// Macro to loop over just the params & output params of an instance,
/// with each iteration providing a Symbol& to symbolref.  Use like this:
///        FOREACH_PARAM (Symbol &s, inst) { ... stuff with s... }
///
#define FOREACH_PARAM(symboldecl,inst) \
    BOOST_FOREACH (symboldecl, param_range(inst))



/// A ShaderGroup consists of one or more layers (each of which is a
/// ShaderInstance), and the connections among them.
class ShaderGroup {
public:
    ShaderGroup ();
    ShaderGroup (const ShaderGroup &g);
    ~ShaderGroup ();

    /// Clear the layers
    ///
    void clear () { m_layers.clear ();  m_optimized = 0;  m_executions = 0; }

    /// Append a new shader instance on to the end of this group
    ///
    void append (ShaderInstanceRef newlayer) {
        ASSERT (! m_optimized && "should not append to optimized group");
        m_layers.push_back (newlayer);
    }

    /// How many layers are in this group?
    ///
    int nlayers () const { return (int) m_layers.size(); }

    /// Array indexing returns the i-th layer of the group
    ///
    ShaderInstance * operator[] (int i) const { return m_layers[i].get(); }

    int optimized () const { return m_optimized; }
    void optimized (int opt) { m_optimized = opt; }

    size_t llvm_groupdata_size () const { return m_llvm_groupdata_size; }
    void llvm_groupdata_size (size_t size) { m_llvm_groupdata_size = size; }

    RunLLVMGroupFunc llvm_compiled_version() const {
        return m_llvm_compiled_version;
    }
    void llvm_compiled_version (RunLLVMGroupFunc func) {
        m_llvm_compiled_version = func;
    }

    long long int executions () const { return m_executions; }

    void start_running () {
#ifdef DEBUG
       m_executions++;
#endif
    }

private:
    std::vector<ShaderInstanceRef> m_layers;
    RunLLVMGroupFunc m_llvm_compiled_version;
    size_t m_llvm_groupdata_size;
    volatile int m_optimized;        ///< Is it already optimized?
    atomic_ll m_executions;          ///< Number of times the group executed
    mutex m_mutex;                   ///< Thread-safe optimization
    friend class ShadingSystemImpl;
};



class ClosureRegistry {
public:

    struct ClosureEntry {
        // normally a closure is fully identified by its
        // name, but we might want to have an internal id
        // for fast dispatching
        int                       id;
        // The name again
        ustring                   name;
        // Number of formal arguments
        int                       nformal;
        // Number of keyword arguments
        int                       nkeyword;
        // The parameters
        std::vector<ClosureParam> params;
        // the needed size for the structure
        int                       struct_size;
        // Creation callbacks
        PrepareClosureFunc        prepare;
        SetupClosureFunc          setup;
        CompareClosureFunc        compare;
    };

    void register_closure(const char *name, int id, const ClosureParam *params, int size,
                          PrepareClosureFunc prepare, SetupClosureFunc setup, CompareClosureFunc compare);

    const ClosureEntry *get_entry(ustring name)const;
    const ClosureEntry *get_entry(int id)const {
        DASSERT((size_t)id < m_closure_table.size());
        return &m_closure_table[id];
    }

private:


    // A mapping from name to ID for the compiler
    std::map<ustring, int>    m_closure_name_to_id;
    // And the internal global table, indexed
    // by the internal ID for fast dispatching
    std::vector<ClosureEntry> m_closure_table;
};



class ShadingSystemImpl : public ShadingSystem
{
public:
    ShadingSystemImpl (RendererServices *renderer=NULL,
                       TextureSystem *texturesystem=NULL,
                       ErrorHandler *err=NULL);
    virtual ~ShadingSystemImpl ();

    virtual bool attribute (const std::string &name, TypeDesc type, const void *val);
    virtual bool getattribute (const std::string &name, TypeDesc type, void *val);

    virtual bool Parameter (const char *name, TypeDesc t, const void *val);
    virtual bool Shader (const char *shaderusage,
                         const char *shadername=NULL,
                         const char *layername=NULL);
    virtual bool ShaderGroupBegin (void);
    virtual bool ShaderGroupEnd (void);
    virtual bool ConnectShaders (const char *srclayer, const char *srcparam,
                                 const char *dstlayer, const char *dstparam);
    virtual ShadingAttribStateRef state () const;
    virtual void clear_state ();

//    virtual void RunShaders (ShadingAttribStateRef &attribstate,
//                             ShaderUse use);

    /// Internal error reporting routine, with printf-like arguments.
    ///
    void error (const char *message, ...);
    /// Internal warning reporting routine, with printf-like arguments.
    ///
    void warning (const char *message, ...);
    /// Internal info printing routine, with printf-like arguments.
    ///
    void info (const char *message, ...);
    /// Internal message printing routine, with printf-like arguments.
    ///
    void message (const char *message, ...);

    /// Error reporting routines that take a pre-formatted string only.
    ///
    void error (const std::string &message);
    void warning (const std::string &message);
    void info (const std::string &message);
    void message (const std::string &message);

    virtual std::string getstats (int level=1) const;

    ErrorHandler &errhandler () const { return *m_err; }

    ShaderMaster::ref loadshader (const char *name);

    void* create_thread_info();

    void destroy_thread_info(void* thread_info);

    /// Get a ShadingContext that we can use.
    ///
    ShadingContext *get_context (void* thread_info = NULL);

    /// Return a ShadingContext to the pool.
    ///
    void release_context (ShadingContext *sc, void* thread_info = NULL);

    void operator delete (void *todel) { ::delete ((char *)todel); }

    /// Return the precomputed heap offset of the named global, or -1 if
    /// it's not precomputed.
    int global_heap_offset (ustring name);

    /// Is the shading system in debug mode?
    ///
    bool debug () const { return m_debug; }

    /// Return a pointer to the renderer services object.
    ///
    RendererServices *renderer () const { return m_renderer; }

    /// Return a pointer to the texture system.
    ///
    TextureSystem *texturesys () const { return m_texturesys; }

    bool allow_rebind () const { return m_rebind; }

    bool debug_nan () const { return m_debugnan; }
    bool lockgeom_default () const { return m_lockgeom_default; }
    int optimize () const { return m_optimize; }
    int llvm_debug () const { return m_llvm_debug; }

    ustring commonspace_synonym () const { return m_commonspace_synonym; }

    /// The group is set and won't be changed again; take advantage of
    /// this by optimizing the code knowing all our instance parameters
    /// (at least the ones that can't be overridden by the geometry).
    void optimize_group (ShadingAttribState &attribstate, ShaderGroup &group);

    int *alloc_int_constants (size_t n) { return m_int_pool.alloc (n); }
    float *alloc_float_constants (size_t n) { return m_float_pool.alloc (n); }
    ustring *alloc_string_constants (size_t n) { return m_string_pool.alloc (n); }

    llvm::LLVMContext *llvm_context () { return m_llvm_context; }
    llvm::ExecutionEngine* ExecutionEngine () { return m_llvm_exec; }

    virtual void register_closure(const char *name, int id, const ClosureParam *params, int size,
                                  PrepareClosureFunc prepare, SetupClosureFunc setup, CompareClosureFunc compare);
    const ClosureRegistry::ClosureEntry *find_closure(ustring name) const {
        return m_closure_registry.get_entry(name);
    }
    const ClosureRegistry::ClosureEntry *find_closure(int id) const {
        return m_closure_registry.get_entry(id);
    }

    /// Convert a color in the named space to RGB.
    ///
    Color3 to_rgb (ustring fromspace, float a, float b, float c);

private:
    void printstats () const;
    void init_global_heap_offsets ();

    /// Find the index of the named layer in the current shader group.
    /// If found, return the index >= 0 and put a pointer to the instance
    /// in inst; if not found, return -1 and set inst to NULL.
    /// (This is a helper for ConnectShaders.)
    int find_named_layer_in_group (ustring layername, ShaderInstance * &inst);

    /// Turn a connectionname (such as "Kd" or "Cout[1]", etc.) into a
    /// ConnectedParam descriptor.  This routine is strictly a helper for
    /// ConnectShaders, and will issue error messages on its behalf.
    /// The return value will not be valid() if there is an error.
    ConnectedParam decode_connected_param (const char *connectionname,
                               const char *layername, ShaderInstance *inst);

    struct PerThreadInfo {
        std::stack<ShadingContext *> context_pool;

        ShadingContext *pop_context ();  ///< Get the pool top and then pop
        ~PerThreadInfo ();
    };

    /// Get the per-thread info, create it if necessary.
    ///
    PerThreadInfo *get_perthread_info () const {
        PerThreadInfo *p = m_perthread_info.get ();
        if (! p) {
            p = new PerThreadInfo;
            m_perthread_info.reset (p);
        }
        return p;
    }

    void SetupLLVM ();

    RendererServices *m_renderer;         ///< Renderer services
    TextureSystem *m_texturesys;          ///< Texture system

    ErrorHandler *m_err;                  ///< Error handler
    std::list<std::string> m_errseen, m_warnseen;
    static const int m_errseenmax = 32;
    mutable mutex m_errmutex;

    typedef std::map<ustring,ShaderMaster::ref> ShaderNameMap;
    ShaderNameMap m_shader_masters;       ///< name -> shader masters map

    ConstantPool<int> m_int_pool;
    ConstantPool<Float> m_float_pool;
    ConstantPool<ustring> m_string_pool;

    // Options
    int m_statslevel;                     ///< Statistics level
    bool m_debug;                         ///< Debugging output
    bool m_lazylayers;                    ///< Evaluate layers on demand?
    bool m_lazyglobals;                   ///< Run lazily even if globals write?
    bool m_clearmemory;                   ///< Zero mem before running shader?
    bool m_rebind;                        ///< Allow rebinding?
    bool m_debugnan;                      ///< Root out NaN's?
    bool m_lockgeom_default;              ///< Default value of lockgeom
    int m_optimize;                       ///< Runtime optimization level
    int m_llvm_debug;                     ///< More LLVM debugging output
    std::string m_searchpath;             ///< Shader search path
    std::vector<std::string> m_searchpath_dirs; ///< All searchpath dirs
    ustring m_commonspace_synonym;        ///< Synonym for "common" space

    bool m_in_group;                      ///< Are we specifying a group?
    ShaderUse m_group_use;                ///< Use of group
    ParamValueList m_pending_params;      ///< Pending Parameter() values
    ShadingAttribStateRef m_curattrib;    ///< Current shading attribute state
    std::map<ustring,int> m_global_heap_offsets; ///< Heap offsets of globals
    size_t m_global_heap_total;           ///< Heap size for globals
    mutable mutex m_mutex;                ///< Thread safety
    mutable thread_specific_ptr<PerThreadInfo> m_perthread_info;

    // Stats
    atomic_int m_stat_shaders_loaded;     ///< Stat: shaders loaded
    atomic_int m_stat_shaders_requested;  ///< Stat: shaders requested
    PeakCounter<int> m_stat_instances;    ///< Stat: instances
    PeakCounter<int> m_stat_contexts;     ///< Stat: shading contexts
    int m_stat_groups;                    ///< Stat: shading groups
    int m_stat_groupinstances;            ///< Stat: total inst in all groups
    atomic_int m_stat_regexes;            ///< Stat: how many regex's compiled
    atomic_ll m_layers_executed_uncond;   ///< Stat: Unconditional execs
    atomic_ll m_layers_executed_lazy;     ///< Stat: On-demand execs
    atomic_ll m_layers_executed_never;    ///< Stat: Layers never executed
    atomic_ll m_stat_binds;               ///< Stat: Number of binds;
    atomic_ll m_stat_rebinds;             ///< Stat: Number of rebinds;
    atomic_ll m_stat_paramstobind;        ///< Stat: All params in bound shaders
    atomic_ll m_stat_paramsbound;         ///< Stat: Number of params bound
    atomic_ll m_stat_instructions_run;    ///< Stat: total instructions run
    atomic_int m_stat_total_syms;         ///< Stat: total syms in all insts
    atomic_int m_stat_syms_with_derivs;   ///< Stat: syms with derivatives
    double m_stat_optimization_time;      ///< Stat: time spent optimizing
    double m_stat_opt_locking_time;       ///<   locking time
    double m_stat_specialization_time;    ///<   runtime specialization time
    double m_stat_total_llvm_time;        ///<   total time spent on LLVM
    double m_stat_llvm_setup_time;        ///<     llvm setup time
    double m_stat_llvm_irgen_time;        ///<     llvm IR generation time
    double m_stat_llvm_opt_time;          ///<     llvm IR optimization time
    double m_stat_llvm_jit_time;          ///<     llvm JIT time
    spin_mutex m_stat_mutex;              ///< Mutex for non-atomic stats
    ClosureRegistry m_closure_registry;

    // LLVM stuff
    llvm::LLVMContext *m_llvm_context;
    llvm::Module *m_llvm_module;
    llvm::ExecutionEngine *m_llvm_exec;

    friend class ShadingContext;
    friend class ShaderInstance;
    friend class RuntimeOptimizer;
};



class ShadingExecution;
typedef std::vector<ShadingExecution> ExecutionLayers;

template<int BlockSize>
class SimplePool {
public:
    SimplePool() {
        m_blocks.push_back(new char[BlockSize]);
        m_block_offset = BlockSize;
        m_current_block = 0;
    }

    ~SimplePool() {
        for (size_t i =0; i < m_blocks.size(); ++i)
            delete [] m_blocks[i];
    }

    char * alloc(size_t size) {
        ASSERT(size < BlockSize);
        if (size <= m_block_offset) {
            m_block_offset -= size;
        } else {
            m_current_block++;
            m_block_offset = BlockSize - size;
            if (m_blocks.size() == m_current_block)
                m_blocks.push_back(new char[BlockSize]);
        }
        return m_blocks[m_current_block] + m_block_offset;
    }

    void clear () { m_current_block = 0; m_block_offset = BlockSize; }

private:
    std::vector<char *> m_blocks;
    size_t              m_current_block;
    size_t              m_block_offset;
};

/// The full context for executing a shader group.  This contains
/// ShadingExecution states for each shader
///
class ShadingContext {
public:
    ShadingContext (ShadingSystemImpl &shadingsys);
    ~ShadingContext ();

    /// Return a reference to the shading system for this context.
    ///
    ShadingSystemImpl & shadingsys () const { return m_shadingsys; }

    /// Get a pointer to the RendererServices for this execution.
    ///
    RendererServices *renderer () const { return m_renderer; }

    /// Execute the shaders for the given use (for example,
    /// ShadUseSurface). If runflags are not supplied, they will be
    /// auto-generated with all points turned on.
    void execute (ShaderUse use, ShadingAttribState &sas,
                  ShaderGlobals &ssg);

    /// Return the current shader use being executed.
    ///
    ShaderUse use () const { return (ShaderUse) m_curuse; }

    ClosureComponent * closure_component_allot(int id, size_t prim_size, int nattrs) {
        size_t needed = sizeof(ClosureComponent) + (prim_size >= 4 ? prim_size - 4 : 0)
                                                 + sizeof(ClosureComponent::Attr) * nattrs;
        ClosureComponent *comp = (ClosureComponent *) m_closure_pool.alloc(needed);
        comp->type = ClosureColor::COMPONENT;
        comp->id = id;
        comp->size = prim_size;
        comp->nattrs = nattrs;
        return comp;
    }

    ClosureMul *closure_mul_allot (const Color3 &w, const ClosureColor *c) {
        ClosureMul *mul = (ClosureMul *) m_closure_pool.alloc(sizeof(ClosureMul));
        mul->type = ClosureColor::MUL;
        mul->weight = w;
        mul->closure = c;
        return mul;
    }

    ClosureMul *closure_mul_allot (float w, const ClosureColor *c) {
        ClosureMul *mul = (ClosureMul *) m_closure_pool.alloc(sizeof(ClosureMul));
        mul->type = ClosureColor::MUL;
        mul->weight.setValue (w,w,w);
        mul->closure = c;
        return mul;
    }

    ClosureAdd *closure_add_allot (const ClosureColor *a, const ClosureColor *b) {
        ClosureAdd *add = (ClosureAdd *) m_closure_pool.alloc(sizeof(ClosureAdd));
        add->type = ClosureColor::ADD;
        add->closureA = a;
        add->closureB = b;
        return add;
    }


    /// Find the named symbol in the (already-executed!) stack of
    /// ShadingExecution's of the given use, with priority given to
    /// later laters over earlier layers (if they name the same symbol).
    /// Return NULL if no such symbol is found.
    Symbol * symbol (ShaderUse use, ustring name);

    /// Return a pointer to where the symbol's data lives for the given
    /// grid point.
    void *symbol_data (Symbol &sym, int gridpoint);

    /// Return a reference to the ExecutionLayers
    ExecutionLayers &execlayer () { return m_exec; }

    /// Return a reference to a compiled regular expression for the
    /// given string, being careful to cache already-created ones so we
    /// aren't constantly compiling new ones.
    const boost::regex & find_regex (ustring r);

    /// Return a pointer to the shading attribs for this context.
    ///
    ShadingAttribState *attribs () { return m_attribs; }

    /// Return a reference to the ParamValueList containing messages.
    ///
    ParamValueList & messages () { return m_messages; }

private:
    /// Various setup of the context done by execute().  Return true if
    /// the function should be executed, otherwise false.
    bool prepare_execution (ShaderUse use, ShadingAttribState &sas);

    /// Execute the llvm-compiled shaders for the given use (for example,
    /// ShadUseSurface).  The context must already be bound.  If
    /// runflags are not supplied, they will be auto-generated with all
    /// points turned on.
    void execute_llvm (ShaderUse use, Runflag *rf=NULL,
                       int *ind=NULL, int nind=0);

    ShadingSystemImpl &m_shadingsys;    ///< Backpointer to shadingsys
    RendererServices *m_renderer;       ///< Ptr to renderer services
    ShadingAttribState *m_attribs;      ///< Ptr to shading attrib state
    std::vector<char> m_heap;           ///< Heap memory
    size_t m_closures_allotted;         ///< Closure memory allotted
    ExecutionLayers m_exec;             ///< Execution layers for the group
    int m_curuse;                       ///< Current use that we're running
#ifdef OIIO_HAVE_BOOST_UNORDERED_MAP
    typedef boost::unordered_map<ustring, boost::regex*, ustringHash> RegexMap;
#else
    typedef hash_map<ustring, boost::regex*, ustringHash> RegexMap;
#endif
    RegexMap m_regex_map;  ///< Compiled regex's
    ParamValueList m_messages;          ///< Message blackboard

    SimplePool<20 * 1024> m_closure_pool;

    friend class ShadingExecution;
};




/// The state and machinery necessary to execute a single shader (node).
///
class ShadingExecution {
public:
    ShadingExecution ();
    ~ShadingExecution ();

    /// Initialize a ShadingExecution to know what context, use, and layer
    /// it's part of.
    void init (ShadingContext *context, ShaderUse use, int layer) {
        m_use = use;
        m_layer = layer;
        m_context = context;
    }

    /// Get a reference to the symbol with the given index.
    /// Beware -- it had better be a valid index!
    Symbol &sym (int index) {
        DASSERT (index < (int)m_symbols.size() && index >= 0);
        return m_symbols[index];
    }

    /// Get a pointer to the symbol with the given index, or NULL if
    /// the index is < 0.
    Symbol *symptr (int index) {
        DASSERT (index < (int)m_symbols.size());
        return index >= 0 ? &m_symbols[index]: NULL;
    }

    bool debug () const { return m_debug; }

    /// Find the named symbol.  Return NULL if no such symbol is found.
    ///
    Symbol * symbol (ustring name) {
        return symptr (m_instance->findsymbol (name));
    }

    /// Format the value of sym using the printf-like format (taking a
    /// SINGLE value specifier), where 'whichpoint' gives the position
    /// in the set of shading points that we're concerned about.
    std::string format_symbol (const std::string &format, Symbol &sym,
                               int whichpoint);

    /// Turn the symbol into a string (for debugging).
    ///
    std::string printsymbolval (Symbol &sym);

    /// Get a pointer to the ShadingContext for this execution.
    ///
    ShadingContext *context () const { return m_context; }

    /// Get a pointer to the ShadingSystemImpl for this execution.
    ///
    ShadingSystemImpl *shadingsys () const { return m_shadingsys; }

    /// Get a pointer to the RendererServices for this execution.
    ///
    RendererServices *renderer () const { return m_renderer; }

    /// Get a pointer to the TextureSystem for this execution.
    ///
    TextureSystem *texturesys () const { return m_shadingsys->texturesys(); }

    /// Get the 4x4 matrix that transforms points from the named 'from'
    /// coordinate system to "common" space for the given shading point.
    void get_matrix (Matrix44 &result, ustring from, int whichpoint=0);

    /// Get the 4x4 matrix that transforms points from "common" space to
    /// the named 'from' coordinate system for the given shading point.
    void get_inverse_matrix (Matrix44 &result, ustring from, int whichpoint=0);

    /// Get the 4x4 matrix that transforms points from the named "from"
    /// coordinate system to the named 'to' coordinate system to at the
    /// given shading point.
    void get_matrix (Matrix44 &result, ustring from,
                     ustring to, int whichpoint=0);

    /// Return the ShaderUse of this execution.
    ///
    ShaderUse shaderuse () const { return m_use; }

    /// Which layer are we in the shader group?
    ///
    int layer () const { return m_layer; }

    /// Return the instance of this execution.
    ///
    ShaderInstance *instance () const { return m_instance; }

    /// Return the name of the shader used by this instance.
    ///
    const std::string &shadername () const { return m_master->shadername(); }

    /// Pass an error along to the ShadingSystem.
    ///
    void error (const char *message, ...);
    void warning (const char *message, ...);
    void info (const char *message, ...);
    void message (const char *message, ...);

    /// Quick link to the global P symbol, or NULL if there is none.
    ///
    Symbol *Psym () { return symptr (m_instance->m_Psym); }

    /// Quick link to the global N symbol, or NULL if there is none.
    ///
    Symbol *Nsym () { return symptr (m_instance->m_Nsym); }

    /// Get the named attribute from the renderer and if found then
    /// write it into 'val'.  Otherwise, return false.  If no object is
    /// specified (object == ustring()), then the renderer should search *first*
    /// for the attribute on the currently shaded object, and next, if
    /// unsuccessful, on the currently shaded "scene". 
    bool get_renderer_attribute(void *renderstate, bool derivatives, ustring object,
                                        TypeDesc type, ustring name, void *val);

    /// Similar to get_renderer_attribute();  this method will return the 'index'
    /// element of an attribute array.
    bool get_renderer_array_attribute (void *renderstate, bool derivatives, ustring object,
                                               TypeDesc type, ustring name,
                                               int index, void *val);

    /// Query the renderer for the named user-data on the current
    /// geometry.  Thi s function accepts an array of renderstate
    /// pointers and writes its value in the memory region pointed to by
    /// 'val'.
    bool get_renderer_userdata (Runflag *runflags, int npoints, bool derivatives, ustring name,
                                TypeDesc type, void *renderstate, 
                                int renderstate_stepsize, 
                                void *val, int val_stepsize);

    /// Determine whether the currently shaded object has the named
    /// user-data attached
    bool renderer_has_userdata (ustring name, TypeDesc type, void *renderstate);

private:

    ShaderUse m_use;              ///< Our shader use
    int m_layer;                  ///< Our layer number
    ShadingContext *m_context;    ///< Ptr to our shading context
    ShaderInstance *m_instance;   ///< Ptr to the shader instance
    ShaderMaster *m_master;       ///< Ptr to the instance's master
    ShadingSystemImpl *m_shadingsys; ///< Ptr to shading system
    RendererServices *m_renderer; ///< Ptr to renderer services
    bool m_debug;                 ///< Debug mode
    SymbolVec m_symbols;          ///< Our own copy of the syms
    int m_last_instance_id;       ///< ID of last instance bound
};



}; // namespace pvt





class ShadingAttribState
{
public:
    ShadingAttribState () { }

    ~ShadingAttribState () { }

    /// Return a reference to the shader group for a particular use
    ///
    ShaderGroup & shadergroup (ShaderUse use) {
        return m_shaders[(int)use];
    }

    /// Called when the shaders of the attrib state change (invalidate LLVM ?)
    void changed_shaders () { }

private:
    OSL::pvt::ShaderGroup m_shaders[OSL::pvt::ShadUseLast];
};



namespace Strings {
    extern ustring camera, common, object, shader;
    extern ustring rgb, RGB, hsv, hsl, YIQ, xyz;
    extern ustring null, default_;
    extern ustring label;
    extern ustring sidedness, front, back, both;
    extern ustring P, I, N, Ng, dPdu, dPdv, u, v, time, dtime, dPdtime, Ps;
    extern ustring Ci;
    extern ustring width, swidth, twidth, blur, sblur, tblur;
    extern ustring wrap, swrap, twrap, black, clamp, periodic, mirror;
    extern ustring firstchannel, fill, alpha;
}; // namespace Strings


}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif

#endif /* OSLEXEC_PVT_H */
