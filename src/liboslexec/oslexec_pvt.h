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


class ShadingSystemImpl;


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
        if (newval > m_current)
            m_requested += (m_current-newval);
        m_current = newval;
        if (m_current > m_peak)
            m_peak = m_current;
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

    void print ();  // Debugging

    /// Return a pointer to the shading system for this master.
    ///
    ShadingSystemImpl & shadingsys () const { return m_shadingsys; }

private:
    ShadingSystemImpl &m_shadingsys;    ///< Back-ptr to the shading system
    ShaderType m_shadertype;            ///< Type of shader
    std::string m_shadername;           ///< Shader name
    std::string m_osofilename;          ///< Full path of oso file
    OpcodeVec m_ops;                    ///< Actual code instructions
    std::vector<int> m_args;            ///< Arguments for all the ops
    // Need the code offsets for each code block
    SymbolVec m_symbols;                ///< Symbols used by the shader
    std::vector<int> m_idefaults;       ///< int default values
    std::vector<float> m_fdefaults;     ///< float default values
    std::vector<ustring> m_sdefaults;   ///< string default values

    friend class OSOReaderToMaster;
};



/// ShaderInstance is a particular instance of a shader, with its own
/// set of parameter values, coordinate transform, and connections to
/// other instances within the same shader group.
class ShaderInstance /*: public RefCnt*/ {
public:
//    typedef intrusive_ptr<ShaderInstance> ref;
    typedef ShaderInstanceRef ref;
    ShaderInstance (ShaderMaster::ref master, const char *layername="") 
        : m_master(master), m_layername(layername),
          m_firstlayer(true) { }
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

private:
    ShaderMaster::ref m_master;         ///< Reference to the master
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

    virtual void Parameter (const char *name, TypeDesc t, const void *val) { }
    virtual ShaderInstanceRef Shader (const char *shaderusage,
                                      const char *shadername=NULL,
                                      const char *layername=NULL) {
        return ShaderInstanceRef();
    }
    virtual void ShaderGroupBegin (void) { }
    virtual ShaderInstanceRef ShaderGroupEnd (void) { return ShaderInstanceRef(); }
    virtual void ConnectShaders (const char *srclayer, const char *srcparam,
                                 const char *dstlayer, const char *dstparam) {}


    /// Internal error reporting routine, with printf-like arguments.
    ///
    void error (const char *message, ...);

    virtual std::string geterror () const;
    virtual std::string getstats (int level=1) const;

    ShaderMaster::ref loadshader (const char *name);

    void operator delete (void *todel) { ::delete ((char *)todel); }

private:
    void printstats () const;

    typedef std::map<ustring,ShaderMaster::ref> ShaderNameMap;
    ShaderNameMap m_shader_masters;       ///< name -> shader masters map
    int m_statslevel;                     ///< Statistics level
    std::string m_searchpath;             ///< Shader search path
    std::vector<std::string> m_searchpath_dirs; ///< All searchpath dirs
    mutable mutex m_mutex;                ///< Thread safety
    mutable mutex m_errmutex;             ///< Safety for error messages
    mutable fast_mutex m_stats_mutex;     ///< Spin lock for non-atomic stats
    mutable std::string m_errormessage;   ///< Saved error string.
    atomic_int m_stat_shaders_loaded;     ///< Stat: shaders loaded
    atomic_int m_stat_shaders_requested;  ///< Stat: shaders requested
    PeakCounter<int> m_stat_instances;    ///< Stat: instances
};


}; // namespace pvt
}; // namespace OSL


#endif /* OSLEXEC_PVT_H */
