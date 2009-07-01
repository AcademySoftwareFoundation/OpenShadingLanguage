/*****************************************************************************
 *
 *             Copyright (c) 2009 Sony Pictures Imageworks, Inc.
 *                            All rights reserved.
 *
 *  This material contains the confidential and proprietary information
 *  of Sony Pictures Imageworks, Inc. and may not be disclosed, copied or
 *  duplicated in any form, electronic or hardcopy, in whole or in part,
 *  without the express prior written consent of Sony Pictures Imageworks,
 *  Inc. This copyright notice does not imply publication.
 *
 *****************************************************************************/

#ifndef OSLEXEC_PVT_H
#define OSLEXEC_PVT_H

#include "OpenImageIO/ustring.h"
#include "OpenImageIO/typedesc.h"
#include "OpenImageIO/thread.h"
#include "OpenImageIO/refcnt.h"

#include "oslexec.h"
#include "osl_pvt.h"


namespace OSL {
namespace pvt {


// forward definitions
class ShadingSystemImpl;
class ShadingContext;
class ShadingExecution;

/// Data type for flags that indicate on a point-by-point basis whether
/// we want computations to be performed.
typedef unsigned char Runflag;

/// Pre-defined values for Runflag's.
///
enum RunFlagVal { RunFlagOff = 0, RunFlagOn = 255 };




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



/// Reference to a shader Parameter from the user/app.  Use local
/// storage for simple scalar types (a single int, float, or string).
class ParamRef {
public:
    ParamRef (ustring name, TypeDesc type, const void *data)
        : m_name(name), m_type(type), m_data(data), m_is_local(false)
    {
        if (type == TypeDesc::TypeInt) {
            m_data = &m_local_data.i;
            m_local_data.i = *(const int *)data;
            m_is_local = true;
        } else if (type == TypeDesc::TypeFloat) {
            m_data = &m_local_data.f;
            m_local_data.f = *(const float *)data;
            m_is_local = true;
        } else if (type == TypeDesc::TypeString) {
            m_data = &m_local_data.s;
            m_local_data.s = ustring(*(const char **)data).c_str();
            m_is_local = true;
        }
    }
    ~ParamRef () { }
    ustring name () const { return m_name; }
    TypeDesc type () const { return m_type; }
    const void *data () const { return m_data; }
private:
    ustring m_name;         ///< Parameter name
    TypeDesc m_type;        ///< Parameter type
    const void *m_data;     ///< Pointer to data -- we are not the owner!
    bool m_is_local;        ///< Do we use local storage?
    union {
        int i;
        float f;
        const char *s;
    } m_local_data;         ///< Local storage for small simple types
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

    void print ();  // Debugging

    /// Return a pointer to the shading system for this master.
    ///
    ShadingSystemImpl & shadingsys () const { return m_shadingsys; }

    /// Run through the symbols and set their data pointers if they are
    /// constants or params (to the defaults).  As a side effect, also
    /// set m_firstparam/m_lastparam.
    void resolve_defaults ();

    /// Find the named symbol, return its index in the symbol array, or
    /// -1 if not found.
    int findsymbol (ustring name) const;

    /// Find the named parameter, return its index in the symbol array, or
    /// -1 if not found.
    int findparam (ustring name) const;

    /// Return a pointer to the symbol (specified by integer index),
    /// or NULL (if index was -1, as returned by 'findsymbol').
    Symbol *symbol (int index) { return index >= 0 ? &m_symbols[index] : NULL; }

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

    friend class OSOReaderToMaster;
    friend class ShaderInstance;
};



/// ShaderInstance is a particular instance of a shader, with its own
/// set of parameter values, coordinate transform, and connections to
/// other instances within the same shader group.
class ShaderInstance {
public:
    typedef ShaderInstanceRef ref;
    ShaderInstance (ShaderMaster::ref master, const char *layername="");
    ~ShaderInstance () { }

    /// Return a pointer to the master for this instance.
    ///
    ShaderMaster *master () const { return m_master.get(); }

    /// Return a reference to the shading system for this instance.
    ///
    ShadingSystemImpl & shadingsys () const { return m_master->shadingsys(); }

    /// Is this instance the first (head) of its group?
    ///
    bool is_first_in_group () const { return m_firstlayer; }

    void append (ShaderInstance::ref anotherlayer);

    /// Return a pointer to the next layer in the group.
    ///
    ShaderInstance *next_layer () const { return m_nextlayer.get(); }

    /// Apply pending parameters
    /// 
    void parameters (const std::vector<ParamRef> &params);

private:
    ShaderMaster::ref m_master;         ///< Reference to the master
    SymbolVec m_symbols;                ///< Symbols used by the instance
    ustring m_layername;                ///< Name of this layer
    ref m_nextlayer;                    ///< Next layer in the group
    bool m_firstlayer;                  ///< Is this the 1st layer of group?
    std::vector<int> m_iparams;         ///< int param values
    std::vector<float> m_fparams;       ///< float param values
    std::vector<ustring> m_sparams;     ///< string param values
};



class ShadingSystemImpl : public ShadingSystem
{
public:
    ShadingSystemImpl ();
    virtual ~ShadingSystemImpl ();

    virtual bool attribute (const std::string &name, TypeDesc type, const void *val);
    virtual bool getattribute (const std::string &name, TypeDesc type, void *val);

    virtual void Parameter (const char *name, TypeDesc t, const void *val);
    virtual void Shader (const char *shaderusage,
                         const char *shadername=NULL,
                         const char *layername=NULL);
    virtual void ShaderGroupBegin (void);
    virtual void ShaderGroupEnd (void);
    virtual void ConnectShaders (const char *srclayer, const char *srcparam,
                                 const char *dstlayer, const char *dstparam);
    virtual ShadingAttribStateRef state () const;
    virtual void clear_state ();

//    virtual void RunShaders (ShadingAttribStateRef &attribstate,
//                             ShaderUse use);

    /// Internal error reporting routine, with printf-like arguments.
    ///
    void error (const char *message, ...);

    virtual std::string geterror () const;
    virtual std::string getstats (int level=1) const;

    ShaderMaster::ref loadshader (const char *name);

    shared_ptr<ShadingContext> get_context ();

    void operator delete (void *todel) { ::delete ((char *)todel); }

private:
    void printstats () const;

    typedef std::map<ustring,ShaderMaster::ref> ShaderNameMap;
    ShaderNameMap m_shader_masters;       ///< name -> shader masters map
    int m_statslevel;                     ///< Statistics level
    std::string m_searchpath;             ///< Shader search path
    std::vector<std::string> m_searchpath_dirs; ///< All searchpath dirs
    bool m_in_group;                      ///< Are we specifying a group?
    ShaderInstanceRef m_group_head;       ///< Head of our group
    ShaderUse m_group_use;                ///< Use of group
    std::vector<ParamRef> m_pending_params; ///< Pending Parameter() values
    ShadingAttribStateRef m_curattrib;    ///< Current shading attribute state
    mutable mutex m_mutex;                ///< Thread safety
    mutable mutex m_errmutex;             ///< Safety for error messages
    mutable std::string m_errormessage;   ///< Saved error string.
    atomic_int m_stat_shaders_loaded;     ///< Stat: shaders loaded
    atomic_int m_stat_shaders_requested;  ///< Stat: shaders requested
    PeakCounter<int> m_stat_instances;    ///< Stat: instances
    PeakCounter<int> m_stat_contexts;     ///< Stat: shading contexts

    friend class ShadingContext;
};



class ShadingExecution;
typedef std::vector<ShadingExecution> ExecutionLayers;



/// The full context for executing a network of shaders.  This contains
/// ShadingExecution states for each shader
///
class ShadingContext {
public:
    ShadingContext (ShadingSystemImpl &shadingsys);
    ~ShadingContext ();

    /// Return a reference to the shading system for this context.
    ///
    ShadingSystemImpl & shadingsys () const { return m_shadingsys; }

    /// Set up this context for shading n points with the given shader
    /// attribute state and shader globals.  Resolve all the memory
    /// layout issues so that we're ready to execute().
    void bind (int n, ShadingAttribState &sas, ShaderGlobals &sg);

    /// Execute the shaders for the given use (for example,
    /// ShadUseSurface).  The context must already be bound.  If
    /// runflags are not supplied, they will be auto-generated with all
    /// points turned on.
    void execute (ShaderUse use, Runflag *rf=NULL);

private:
    ShadingSystemImpl &m_shadingsys;
    std::vector<float> *m_heap;                   ///< Heap memory
    ExecutionLayers m_surf, m_disp, m_volume;
};



/// The state and machinery necessary to execute a single shader (node).
///
class ShadingExecution {
public:
    ShadingExecution (ShadingContext *context=NULL) 
        : m_context(context), m_ourlayers(NULL)
    { }
    ~ShadingExecution () { }
private:
    ShadingContext *m_context;
    ShaderInstance::ref m_instance;
    ShaderMaster::ref m_master;
    ExecutionLayers *m_ourlayers;
};



}; // namespace pvt





class ShadingAttribState
{
public:
    ShadingAttribState () { }
    ~ShadingAttribState () { }

private:
    ShaderInstanceRef m_shaders[OSL::pvt::ShadUseLast];
    friend class OSL::pvt::ShadingSystemImpl;
};



}; // namespace OSL


#endif /* OSLEXEC_PVT_H */
