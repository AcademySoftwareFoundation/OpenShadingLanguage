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
#include <algorithm>

#include <boost/foreach.hpp>

#include <OpenImageIO/dassert.h>
#include <OpenImageIO/strutil.h>

#include "oslexec_pvt.h"


OSL_NAMESPACE_ENTER


namespace pvt {   // OSL::pvt

using OIIO::spin_lock;
using OIIO::ParamValue;
using OIIO::ParamValueList;

static int next_id = 0; // We can statically init an int, not an atomic



ShaderInstance::ShaderInstance (ShaderMaster::ref master,
                                string_view layername)
    : m_master(master),
      //DON'T COPY  m_instsymbols(m_master->m_symbols),
      //DON'T COPY  m_instops(m_master->m_ops), m_instargs(m_master->m_args),
      m_layername(layername),
      m_writes_globals(false),
      m_outgoing_connections(false),
      m_renderer_outputs(false), m_merged_unused(false),
      m_last_layer(false), m_entry_layer(false),
      m_firstparam(m_master->m_firstparam), m_lastparam(m_master->m_lastparam),
      m_maincodebegin(m_master->m_maincodebegin),
      m_maincodeend(m_master->m_maincodeend)
{
    m_id = ++(*(atomic_int *)&next_id);
    shadingsys().m_stat_instances += 1;

    // We don't copy the symbol table yet, it stays with the master, but
    // we'll keep track of local override information in m_instoverrides.

    // Make it easy for quick lookups of common symbols
    m_Psym = findsymbol (Strings::P);
    m_Nsym = findsymbol (Strings::N);

    // Adjust statistics
    ShadingSystemImpl &ss (shadingsys());
    off_t parammem = vectorbytes (m_iparams)
        + vectorbytes (m_fparams) + vectorbytes (m_sparams);
    off_t totalmem = (parammem + sizeof(ShaderInstance));
    {
        spin_lock lock (ss.m_stat_mutex);
        ss.m_stat_mem_inst_paramvals += parammem;
        ss.m_stat_mem_inst += totalmem;
        ss.m_stat_memory += totalmem;
    }
}



ShaderInstance::~ShaderInstance ()
{
    shadingsys().m_stat_instances -= 1;

    ASSERT (m_instops.size() == 0 && m_instargs.size() == 0);
    ShadingSystemImpl &ss (shadingsys());
    off_t symmem = vectorbytes (m_instsymbols) + vectorbytes(m_instoverrides);
    off_t parammem = vectorbytes (m_iparams)
        + vectorbytes (m_fparams) + vectorbytes (m_sparams);
    off_t connectionmem = vectorbytes (m_connections);
    off_t totalmem = (symmem + parammem + connectionmem +
                       sizeof(ShaderInstance));
    {
        spin_lock lock (ss.m_stat_mutex);
        ss.m_stat_mem_inst_syms -= symmem;
        ss.m_stat_mem_inst_paramvals -= parammem;
        ss.m_stat_mem_inst_connections -= connectionmem;
        ss.m_stat_mem_inst -= totalmem;
        ss.m_stat_memory -= totalmem;
    }
}



int
ShaderInstance::findsymbol (ustring name) const
{
    for (size_t i = 0, e = m_instsymbols.size();  i < e;  ++i)
        if (m_instsymbols[i].name() == name)
            return (int)i;

    // If we haven't yet copied the syms from the master, get it from there
    if (m_instsymbols.empty())
        return m_master->findsymbol (name);

    return -1;
}



int
ShaderInstance::findparam (ustring name) const
{
    if (m_instsymbols.size())
        for (int i = m_firstparam, e = m_lastparam;  i < e;  ++i)
            if (m_instsymbols[i].name() == name)
                return i;

    // Not found? Try the master.
    for (int i = m_firstparam, e = m_lastparam;  i < e;  ++i)
        if (master()->symbol(i)->name() == name)
            return i;

    return -1;
}



void *
ShaderInstance::param_storage (int index)
{
    const Symbol *sym = m_instsymbols.size() ? symbol(index) : mastersymbol(index);

    // Get the data offset. If there are instance overrides for symbols,
    // check whether we are overriding the array size, otherwise just read
    // the offset from the symbol.  Overrides for arraylength -- which occur
    // when an indefinite-sized array parameter gets a value (with a concrete
    // length) -- are special, because in that case the new storage is
    // allocated at the end of the previous parameter list, and thus is not
    // where the master may have thought it was.
    int offset;
    if (m_instoverrides.size() && m_instoverrides[index].arraylen())
        offset = m_instoverrides[index].dataoffset();
    else
        offset = sym->dataoffset();

    TypeDesc t = sym->typespec().simpletype();
    if (t.basetype == TypeDesc::INT) {
        return &m_iparams[offset];
    } else if (t.basetype == TypeDesc::FLOAT) {
        return &m_fparams[offset];
    } else if (t.basetype == TypeDesc::STRING) {
        return &m_sparams[offset];
    } else {
        return NULL;
    }
}



const void *
ShaderInstance::param_storage (int index) const
{
    // Rather than repeating code here, just use const_cast and call the
    // non-const version of this method.
    return (const_cast<ShaderInstance*>(this))->param_storage(index);
}



// Can a parameter with type 'a' be bound to a value of type b.
// This is true when they are identical types, but also when 'a' is an
// array of unspecified length, while b is an array of the same type, with
// definite length.
inline bool compatible (const TypeDesc& a, const TypeDesc& b)
{
    return a.basetype == b.basetype && a.aggregate == b.aggregate &&
           a.vecsemantics == b.vecsemantics &&
           (a.arraylen == b.arraylen || (a.arraylen == -1 && b.arraylen > 0));
}



void
ShaderInstance::parameters (const ParamValueList &params)
{
    // Seed the params with the master's defaults
    m_iparams = m_master->m_idefaults;
    m_fparams = m_master->m_fdefaults;
    m_sparams = m_master->m_sdefaults;

    m_instoverrides.resize (std::max (0, lastparam()));

    // Set the initial lockgeom and dataoffset on the instoverrides, based
    // on the master.
    for (int i = 0, e = (int)m_instoverrides.size(); i < e; ++i) {
        Symbol *sym = master()->symbol(i);
        m_instoverrides[i].lockgeom (sym->lockgeom());
        m_instoverrides[i].dataoffset (sym->dataoffset());
    }

    BOOST_FOREACH (const ParamValue &p, params) {
        if (p.name().size() == 0)
            continue;   // skip empty names
        int i = findparam (p.name());
        if (i >= 0) {
            // if (shadingsys().debug())
            //     shadingsys().info (" PARAMETER %s %s", p.name(), p.type());
            const Symbol *sm = master()->symbol(i);    // This sym in the master
            SymOverrideInfo *so = &m_instoverrides[i]; // Slot for sym's override info
            TypeSpec sm_typespec = sm->typespec(); // Type of the master's param
            if (sm_typespec.is_closure_based()) {
                // Can't assign a closure instance value.
                shadingsys().warning ("skipping assignment of closure: %s", sm->name());
                continue;
            }
            if (sm_typespec.is_structure())
                continue;    // structs are just placeholders; skip

            // Check type of parameter and matching symbol. Note that the
            // compatible accounts for indefinite-length arrays.
            TypeDesc paramtype = sm_typespec.simpletype();
            TypeDesc valuetype = p.type();
            if (!compatible(paramtype, valuetype)) {
                shadingsys().warning ("attempting to set parameter with wrong type: %s (expected '%s', received '%s')",
                                      sm->name(), paramtype, valuetype);
                continue;
            }

            // Mark that the override as an instance value
            so->valuesource (Symbol::InstanceVal);

            // Lock the param against geometric primitive overrides if the
            // master thinks it was so locked, AND the Parameter() call
            // didn't specify lockgeom=false (which would be indicated by
            // the parameter's interpolation being non-CONSTANT).
            bool lockgeom = (sm->lockgeom() &&
                             p.interp() == ParamValue::INTERP_CONSTANT);
            so->lockgeom (lockgeom);

            DASSERT (so->dataoffset() == sm->dataoffset());
            so->dataoffset (sm->dataoffset());

            if (paramtype.arraylen < 0) {
                // An array of definite size was supplied to a parameter
                // that was an array of indefinite size. Magic! The trick
                // here is that we need to allocate paramter space at the
                // END of the ordinary param storage, since when we assigned
                // data offsets to each parameter, we didn't know the length
                // needed to allocate this param in its proper spot.
                ASSERT (valuetype.arraylen > 0);
                // Store the actual length in the shader instance parameter
                // override info.
                so->arraylen (valuetype.arraylen);
                // Allocate space for the new param size at the end of its
                // usual parameter area, and set the new dataoffset to that
                // position.
                int nelements = valuetype.arraylen * valuetype.aggregate;
                if (paramtype.basetype == TypeDesc::FLOAT) {
                    so->dataoffset((int) m_fparams.size());
                    expand (m_fparams, nelements);
                } else if (paramtype.basetype == TypeDesc::INT) {
                    so->dataoffset((int) m_iparams.size());
                    expand (m_iparams, nelements);
                } else if (paramtype.basetype == TypeDesc::STRING) {
                    so->dataoffset((int) m_sparams.size());
                    expand (m_sparams, nelements);
                } else {
                    ASSERT (0 && "unexpected type");
                }
                // FIXME: There's a tricky case that we overlook here, where
                // an indefinite-length-array parameter is given DIFFERENT
                // definite length in subsequent rerenders. Don't do that.
            }
            else {
                // If the instance value is the same as the master's default,
                // just skip the parameter, let it "keep" the default.
                // Note that this can't/shouldn't happen for the indefinite-
                // sized array case, which is why we have it in the 'else'
                // clause of that test.
                void *defaultdata = m_master->param_default_storage(i);
                if (lockgeom &&
                      memcmp (defaultdata, p.data(), valuetype.size()) == 0) {
                    // Must reset valuesource to default, in case the parameter
                    // was set already, and now is being changed back to default.
                    so->valuesource (Symbol::DefaultVal);
                }
            }

            // Copy the supplied data into place.
            memcpy (param_storage(i), p.data(), valuetype.size());
        }
        else {
            shadingsys().warning ("attempting to set nonexistent parameter: %s", p.name());
        }
    }

    {
        // Adjust the stats
        ShadingSystemImpl &ss (shadingsys());
        size_t symmem = vectorbytes(m_instoverrides);
        size_t parammem = (vectorbytes(m_iparams) + vectorbytes(m_fparams) +
                           vectorbytes(m_sparams));
        spin_lock lock (ss.m_stat_mutex);
        ss.m_stat_mem_inst_syms += symmem;
        ss.m_stat_mem_inst_paramvals += parammem;
        ss.m_stat_mem_inst += (symmem+parammem);
        ss.m_stat_memory += (symmem+parammem);
    }
}



void
ShaderInstance::make_symbol_room (size_t moresyms)
{
    size_t oldsize = m_instsymbols.capacity();
    if (oldsize < m_instsymbols.size()+moresyms) {
        // Allocate a bit more than we need, so that most times we don't
        // need to reallocate.  But don't be wasteful by doubling or
        // anything like that, since we only expect a few to be added.
        const size_t extra_room = 10;
        size_t newsize = m_instsymbols.size() + moresyms + extra_room;
        m_instsymbols.reserve (newsize);

        // adjust stats
        spin_lock lock (shadingsys().m_stat_mutex);
        size_t mem = (newsize-oldsize) * sizeof(Symbol);
        shadingsys().m_stat_mem_inst_syms += mem;
        shadingsys().m_stat_mem_inst += mem;
        shadingsys().m_stat_memory += mem;
    }
}



void
ShaderInstance::add_connection (int srclayer, const ConnectedParam &srccon,
                                const ConnectedParam &dstcon)
{
    // specialize symbol in case of dstcon is an unsized array
    if (dstcon.type.is_unsized_array()) {
        SymOverrideInfo *so = &m_instoverrides[dstcon.param];
        so->arraylen(srccon.type.arraylength());

        const TypeDesc& type = srccon.type.simpletype();
        // Skip structs for now, they're just placeholders
        /*if      (t.is_structure()) {
        }
        else*/ if (type.basetype == TypeDesc::FLOAT) {
            so->dataoffset((int) m_fparams.size());
            expand (m_fparams,type.size());
        } else if (type.basetype == TypeDesc::INT) {
            so->dataoffset((int) m_iparams.size());
            expand (m_iparams, type.size());
        } else if (type.basetype == TypeDesc::STRING) {
            so->dataoffset((int) m_sparams.size());
            expand (m_sparams, type.size());
        }/* else if (t.is_closure()) {
            // Closures are pointers, so we allocate a string default taking
            // adventage of their default being NULL as well.
            so->dataoffset((int) m_sparams.size());
            expand (m_sparams, type.size());
        }*/ else {
            ASSERT (0 && "unexpected type");
        }
    }

    off_t oldmem = vectorbytes(m_connections);
    m_connections.push_back (Connection (srclayer, srccon, dstcon));

    // adjust stats
    off_t mem = vectorbytes(m_connections) - oldmem;
    {
        spin_lock lock (shadingsys().m_stat_mutex);
        shadingsys().m_stat_mem_inst_connections += mem;
        shadingsys().m_stat_mem_inst += mem;
        shadingsys().m_stat_memory += mem;
    }
}



void
ShaderInstance::evaluate_writes_globals_and_userdata_params ()
{
    writes_globals (false);
    userdata_params (false);
    BOOST_FOREACH (Symbol &s, symbols()) {
        if (s.symtype() == SymTypeGlobal && s.everwritten())
            writes_globals (true);
        if ((s.symtype() == SymTypeParam || s.symtype() == SymTypeOutputParam)
            && ! s.lockgeom() && ! s.connected())
            userdata_params (true);
        if (s.symtype() == SymTypeTemp) // Once we hit a temp, we'll never
            break;                      // see another global or param.
    }

    // In case this method is called before the Symbol vector is copied
    // (i.e. before copy_code_from_master is called), try to set
    // userdata_params as accurately as we can based on what we know from
    // the symbol overrides. This is very important to get instance merging
    // working correctly.
    int p = 0;
    BOOST_FOREACH (SymOverrideInfo &s, m_instoverrides) {
        if (! s.lockgeom())
            userdata_params (true);
        ++p;
    }
}



void
ShaderInstance::copy_code_from_master (ShaderGroup &group)
{
    ASSERT (m_instops.empty() && m_instargs.empty());
    // reserve with enough room for a few insertions
    m_instops.reserve (master()->m_ops.size()+10);
    m_instargs.reserve (master()->m_args.size()+10);
    m_instops = master()->m_ops;
    m_instargs = master()->m_args;

    // Copy the symbols from the master
    ASSERT (m_instsymbols.size() == 0 &&
            "should not have copied m_instsymbols yet");
    m_instsymbols = m_master->m_symbols;

    // Copy the instance override data
    // Also set the renderer_output flags where needed.
    ASSERT (m_instoverrides.size() == (size_t)std::max(0,lastparam()));
    ASSERT (m_instsymbols.size() >= (size_t)std::max(0,lastparam()));
    if (m_instoverrides.size()) {
        for (size_t i = 0, e = lastparam();  i < e;  ++i) {
            Symbol *si = &m_instsymbols[i];
            if (m_instoverrides[i].valuesource() == Symbol::DefaultVal) {
                // Fix the length of any default-value variable length array
                // parameters.
                if (si->typespec().is_unsized_array())
                    si->arraylen (si->initializers());
            } else {
                if (m_instoverrides[i].arraylen())
                    si->arraylen (m_instoverrides[i].arraylen());
                si->valuesource (m_instoverrides[i].valuesource());
                si->connected_down (m_instoverrides[i].connected_down());
                si->lockgeom (m_instoverrides[i].lockgeom());
                si->dataoffset (m_instoverrides[i].dataoffset());
                si->data (param_storage(i));
            }
            if (shadingsys().is_renderer_output (layername(), si->name(), &group)) {
                si->renderer_output (true);
                renderer_outputs (true);
            }
        }
    }
    evaluate_writes_globals_and_userdata_params ();
    off_t symmem = vectorbytes(m_instsymbols) - vectorbytes(m_instoverrides);
    SymOverrideInfoVec().swap (m_instoverrides);  // free it

    // adjust stats
    {
        spin_lock lock (shadingsys().m_stat_mutex);
        shadingsys().m_stat_mem_inst_syms += symmem;
        shadingsys().m_stat_mem_inst += symmem;
        shadingsys().m_stat_memory += symmem;
    }
}



// Are the two vectors equivalent(a[i],b[i]) in each of their members?
template<class T>
inline bool
equivalent (const std::vector<T> &a, const std::vector<T> &b)
{
    if (a.size() != b.size())
        return false;
    typename std::vector<T>::const_iterator ai, ae, bi;
    for (ai = a.begin(), ae = a.end(), bi = b.begin();  ai != ae;  ++ai, ++bi)
        if (! equivalent(*ai, *bi))
            return false;
    return true;
}



/// Are two symbols equivalent (from the point of view of merging
/// shader instances)?  Note that this is not a true ==, it ignores
/// the m_data, m_node, and m_alias pointers!
static bool
equivalent (const Symbol &a, const Symbol &b)
{
    // If they aren't used, don't consider them a mismatch
    if (! a.everused() && ! b.everused())
        return true;

    // Different symbol types or data types are a mismatch
    if (a.symtype() != b.symtype() || a.typespec() != b.typespec())
        return false;

    // Don't consider different names to be a mismatch if the symbol
    // is a temp or constant.
    if (a.symtype() != SymTypeTemp && a.symtype() != SymTypeConst &&
        a.name() != b.name())
        return false;
    // But constants had better match their values!
    if (a.symtype() == SymTypeConst &&
        memcmp (a.data(), b.data(), a.typespec().simpletype().size()))
        return false;

    return a.has_derivs() == b.has_derivs() &&
        a.lockgeom() == b.lockgeom() &&
        a.valuesource() == b.valuesource() &&
        a.fieldid() == b.fieldid() &&
        a.initbegin() == b.initbegin() &&
        a.initend() == b.initend()
        ;
}



bool
ShaderInstance::mergeable (const ShaderInstance &b, const ShaderGroup &g) const
{
    // Must both be instances of the same master -- very fast early-out
    // for most potential pair comparisons.
    if (master() != b.master())
        return false;

    // If the shaders haven't been optimized yet, they don't yet have
    // their own symbol tables and instructions (they just refer to
    // their unoptimized master), but they may have an "instance
    // override" vector that describes which parameters have
    // instance-specific values or connections.
    bool optimized = (m_instsymbols.size() != 0 || m_instops.size() != 0);

    // Same instance overrides
    if (m_instoverrides.size() || b.m_instoverrides.size()) {
        ASSERT (! optimized);  // should not be post-opt
        ASSERT (m_instoverrides.size() == b.m_instoverrides.size());
        for (size_t i = 0, e = m_instoverrides.size();  i < e;  ++i) {
            if ((m_instoverrides[i].valuesource() == Symbol::DefaultVal ||
                 m_instoverrides[i].valuesource() == Symbol::InstanceVal) &&
                (b.m_instoverrides[i].valuesource() == Symbol::DefaultVal ||
                 b.m_instoverrides[i].valuesource() == Symbol::InstanceVal)) {
                // If both params are defaults or instances, let the
                // instance parameter value checking below handle
                // things. No need to reject default-vs-instance
                // mismatches if the actual values turn out to be the
                // same later.
                continue;
            }

            if (! (equivalent(m_instoverrides[i], b.m_instoverrides[i]))) {
                const Symbol *sym = mastersymbol(i);  // remember, it's pre-opt
                const Symbol *bsym = b.mastersymbol(i);
                if (! sym->everused_in_group() && ! bsym->everused_in_group())
                    continue;
                return false;
            }
            // But still, if they differ in their lockgeom'edness, we can't
            // merge the instances.
            if (m_instoverrides[i].lockgeom() != b.m_instoverrides[i].lockgeom()) {
                return false;
            }
        }
    }

    // Make sure that the two nodes have the same parameter values.  If
    // the group has already been optimized, it's got an
    // instance-specific symbol table to check; but if it hasn't been
    // optimized, we check the symbol table in the master.
    for (int i = firstparam();  i < lastparam();  ++i) {
        const Symbol *sym = optimized ? symbol(i) : mastersymbol(i);
        if (! sym->everused_in_group())
            continue;
        if (sym->typespec().is_closure())
            continue;   // Closures can't have instance override values
        if ((sym->valuesource() == Symbol::InstanceVal || sym->valuesource() == Symbol::DefaultVal)
            && memcmp (param_storage(i), b.param_storage(i),
                       sym->typespec().simpletype().size())) {
            return false;
        }
    }

    if (run_lazily() != b.run_lazily()) {
        return false;
    }

    // The connection list need to be the same for the two shaders.
    if (m_connections.size() != b.m_connections.size()) {
        return false;
    }
    if (m_connections != b.m_connections) {
        return false;
    }

    // Make sure system didn't ask for instances that query userdata to be
    // immune from instance merging.
    if (! shadingsys().m_opt_merge_instances_with_userdata
        && (userdata_params() || b.userdata_params())) {
        return false;
    }

    // If there are no "local" ops or symbols, this instance hasn't been
    // optimized yet.  In that case, we've already done enough checking,
    // since the masters being the same and having the same instance
    // params and connections is all it takes.  The rest (below) only
    // comes into play after instances are more fully elaborated from
    // their masters in order to be optimized.
    if (!optimized) {
        return true;
    }

    // Same symbol table
    if (! equivalent (m_instsymbols, b.m_instsymbols)) {
        return false;
    }

    // Same opcodes to run
    if (! equivalent (m_instops, b.m_instops)) {
        return false;
    }
    // Same arguments to the ops
    if (m_instargs != b.m_instargs) {
        return false;
    }

    // Parameter and code ranges
    if (m_firstparam != b.m_firstparam ||
        m_lastparam != b.m_lastparam ||
        m_maincodebegin != b.m_maincodebegin ||
        m_maincodeend != b.m_maincodeend ||
        m_Psym != b.m_Psym || m_Nsym != b.m_Nsym) {
        return false;
    }

    // Nothing left to check, they must be identical!
    return true;
}


}; // namespace pvt



ShaderGroup::ShaderGroup (string_view name)
  : m_optimized(0), m_does_nothing(false),
    m_llvm_groupdata_size(0), m_num_entry_layers(0),
    m_llvm_compiled_version(NULL),
    m_name(name), m_exec_repeat(1), m_raytype_queries(-1)
{
    m_executions = 0;
    m_stat_total_shading_time_ticks = 0;
    m_id = ++(*(atomic_int *)&next_id);
}



ShaderGroup::ShaderGroup (const ShaderGroup &g, string_view name)
  : m_optimized(0), m_does_nothing(false),
    m_llvm_groupdata_size(0), m_num_entry_layers(g.m_num_entry_layers),
    m_llvm_compiled_version(NULL),
    m_layers(g.m_layers),
    m_name(name), m_exec_repeat(1), m_raytype_queries(-1)
{
    m_executions = 0;
    m_stat_total_shading_time_ticks = 0;
    m_id = ++(*(atomic_int *)&next_id);
}



ShaderGroup::~ShaderGroup ()
{
#if 0
    if (m_layers.size()) {
        ustring name = m_layers.back()->layername();
        std::cerr << "Shader group " << this 
                  << " id #" << m_layers.back()->id() << " (" 
                  << (name.c_str() ? name.c_str() : "<unnamed>")
                  << ") executed on " << executions() << " points\n";
    } else {
        std::cerr << "Shader group " << this << " (no layers?) " 
                  << "executed on " << executions() << " points\n";
    }
#endif
}



int
ShaderGroup::find_layer (ustring layername) const
{
    int i;
    for (i = nlayers()-1; i >= 0 && layer(i)->layername() != layername; --i)
        ;
    return i;  // will be -1 if we never found a match
}



const Symbol *
ShaderGroup::find_symbol (ustring layername, ustring symbolname) const
{
    for (int layer = nlayers()-1;  layer >= 0;  --layer) {
        const ShaderInstance *inst (m_layers[layer].get());
        if (layername.size() && layername != inst->layername())
            continue;  // They asked for a specific layer and this isn't it
        int symidx = inst->findsymbol (symbolname);
        if (symidx >= 0)
            return inst->symbol (symidx);
    }
    return NULL;
}



void
ShaderGroup::clear_entry_layers ()
{
    for (int i = 0;  i < nlayers();  ++i)
        m_layers[i]->entry_layer (false);
    m_num_entry_layers = 0;
}



void
ShaderGroup::mark_entry_layer (int layer)
{
    if (layer >= 0 && layer < nlayers() && ! m_layers[layer]->entry_layer()) {
        m_layers[layer]->entry_layer(true);
        ++m_num_entry_layers;
    }
}



std::string
ShaderGroup::serialize () const
{
    std::ostringstream out;
    out.precision (9);
    lock_guard lock (m_mutex);
    for (int i = 0, nl = nlayers(); i < nl; ++i) {
        const ShaderInstance *inst = m_layers[i].get();

        bool dstsyms_exist = inst->symbols().size();
        for (int p = 0;  p < inst->lastparam(); ++p) {
            const Symbol *s = dstsyms_exist ? inst->symbol(p) : inst->mastersymbol(p);
            ASSERT (s);
            if (s->symtype() != SymTypeParam && s->symtype() != SymTypeOutputParam)
                continue;
            Symbol::ValueSource vs = dstsyms_exist ? s->valuesource()
                                                   : inst->instoverride(p)->valuesource();
            if (vs == Symbol::InstanceVal) {
                TypeDesc type = s->typespec().simpletype();
                int offset = s->dataoffset();
                if (type.is_unsized_array() && ! dstsyms_exist) {
                    // If we're being asked to serialize a group that isn't
                    // yet optimized, any "unsized" arrays will have their
                    // concrete length and offset in the SymOverrideInfo,
                    // not in the Symbol belonging to the instance.
                    type.arraylen = inst->instoverride(p)->arraylen();
                    offset = inst->instoverride(p)->dataoffset();
                }
                out << "param " << type << ' ' << s->name();
                int nvals = type.numelements() * type.aggregate;
                if (type.basetype == TypeDesc::INT) {
                    const int *vals = &inst->m_iparams[offset];
                    for (int i = 0; i < nvals; ++i)
                        out << ' ' << vals[i];
                } else if (type.basetype == TypeDesc::FLOAT) {
                    const float *vals = &inst->m_fparams[offset];
                    for (int i = 0; i < nvals; ++i)
                        out << ' ' << vals[i];
                } else if (type.basetype == TypeDesc::STRING) {
                    const ustring *vals = &inst->m_sparams[offset];
                    for (int i = 0; i < nvals; ++i)
                        out << ' ' << '\"' << Strutil::escape_chars(vals[i]) << '\"';
                } else {
                    ASSERT_MSG (0, "unknown type for serialization: %s (%s)",
                                   type.c_str(), s->typespec().c_str());
                }
                bool lockgeom = dstsyms_exist ? s->lockgeom()
                                              : inst->instoverride(p)->lockgeom();
                if (! lockgeom)
                    out << Strutil::format (" [[int lockgeom=%d]]", lockgeom);
                out << " ;\n";
            }
        }
        out << "shader " << inst->shadername() << ' ' << inst->layername() << " ;\n";
        for (int c = 0, nc = inst->nconnections(); c < nc; ++c) {
            const Connection &con (inst->connection(c));
            ASSERT (con.srclayer >= 0);
            const ShaderInstance *srclayer = m_layers[con.srclayer].get();
            ASSERT (srclayer);
            ustring srclayername = srclayer->layername();
            ASSERT (con.src.param >= 0 && con.dst.param >= 0);
            bool srcsyms_exist = srclayer->symbols().size();
            ustring srcparam = srcsyms_exist ? srclayer->symbol(con.src.param)->name()
                                             : srclayer->mastersymbol(con.src.param)->name();
            ustring dstparam = dstsyms_exist ? inst->symbol(con.dst.param)->name()
                                             : inst->mastersymbol(con.dst.param)->name();
            // FIXME: Assertions to be sure we don't yet support individual
            // channel or array element connections. Fix eventually.
            ASSERT (con.src.arrayindex == -1 && con.src.channel == -1);
            ASSERT (con.dst.arrayindex == -1 && con.dst.channel == -1);
            out << "connect " <<  srclayername << '.' << srcparam << ' '
                << inst->layername() << '.' << dstparam << " ;\n";
        }
    }
    return out.str();
}


OSL_NAMESPACE_EXIT
