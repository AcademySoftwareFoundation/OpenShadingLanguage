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



ShaderInstance::ShaderInstance (ShaderMaster::ref master,
                                const char *layername) 
    : m_master(master),
      //DON'T COPY  m_instsymbols(m_master->m_symbols),
      //DON'T COPY  m_instops(m_master->m_ops), m_instargs(m_master->m_args),
      m_layername(layername),
      m_writes_globals(false), m_run_lazily(false),
      m_outgoing_connections(false),
      m_firstparam(m_master->m_firstparam), m_lastparam(m_master->m_lastparam),
      m_maincodebegin(m_master->m_maincodebegin),
      m_maincodeend(m_master->m_maincodeend)
{
    static int next_id = 0; // We can statically init an int, not an atomic
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
    TypeDesc t = sym->typespec().simpletype();
    if (t.basetype == TypeDesc::INT) {
        return &m_iparams[sym->dataoffset()];
    } else if (t.basetype == TypeDesc::FLOAT) {
        return &m_fparams[sym->dataoffset()];
    } else if (t.basetype == TypeDesc::STRING) {
        return &m_sparams[sym->dataoffset()];
    } else {
        return NULL;
    }
}



const void *
ShaderInstance::param_storage (int index) const
{
    const Symbol *sym = m_instsymbols.size() ? symbol(index) : mastersymbol(index);
    TypeDesc t = sym->typespec().simpletype();
    if (t.basetype == TypeDesc::INT) {
        return &m_iparams[sym->dataoffset()];
    } else if (t.basetype == TypeDesc::FLOAT) {
        return &m_fparams[sym->dataoffset()];
    } else if (t.basetype == TypeDesc::STRING) {
        return &m_sparams[sym->dataoffset()];
    } else {
        return NULL;
    }
}



void
ShaderInstance::parameters (const ParamValueList &params)
{
    // Seed the params with the master's defaults
    m_iparams = m_master->m_idefaults;
    m_fparams = m_master->m_fdefaults;
    m_sparams = m_master->m_sdefaults;

    m_instoverrides.resize (std::max (0, lastparam()));

    {
        // Adjust the stats
        ShadingSystemImpl &ss (shadingsys());
        spin_lock lock (ss.m_stat_mutex);
        size_t symmem = vectorbytes(m_instoverrides);
        size_t parammem = (vectorbytes(m_iparams) + vectorbytes(m_fparams) +
                           vectorbytes(m_sparams));
        ss.m_stat_mem_inst_syms += symmem;
        ss.m_stat_mem_inst_paramvals += parammem;
        ss.m_stat_mem_inst += (symmem+parammem);
        ss.m_stat_memory += (symmem+parammem);
    }

    BOOST_FOREACH (const ParamValue &p, params) {
        if (shadingsys().debug())
            shadingsys().info (" PARAMETER %s %s",
                               p.name().c_str(), p.type().c_str());
        int i = findparam (p.name());
        if (i >= 0) {
            const Symbol *sm = master()->symbol(i);
            SymOverrideInfo *so = &m_instoverrides[i];
            TypeSpec t = sm->typespec();
            // don't allow assignment of closures
            if (t.is_closure()) {
                shadingsys().warning ("skipping assignment of closure: %s", sm->name().c_str());
                continue;
            }
            if (t.is_structure())
                continue;
            // check type of parameter and matching symbol
            if (t.simpletype() != p.type()) {
                shadingsys().warning ("attempting to set parameter with wrong type: %s (exepected '%s', received '%s')", sm->name().c_str(), t.c_str(), p.type().c_str());
                continue;
            }

            // Lock the param against geometric primitive overrides if the
            // master thinks it was so locked, AND the Parameter() call
            // didn't specify lockgeom=false (which would be indicated by
            // the parameter's interpolation being non-CONSTANT).
            bool lockgeom = (sm->lockgeom() &&
                             p.interp() == ParamValue::INTERP_CONSTANT);
            so->lockgeom (lockgeom);

            // If the instance value is the same as the master's default,
            // just skip the parameter, let it "keep" the default.
            void *defaultdata = m_master->param_default_storage(i);
            if (lockgeom && 
                  memcmp (defaultdata, p.data(), t.simpletype().size()) == 0)
                continue;

            so->valuesource (Symbol::InstanceVal);
            void *data = param_storage(i);
            memcpy (data, p.data(), t.simpletype().size());
            if (shadingsys().debug())
                shadingsys().info ("    sym %s offset %llu address %p",
                        sm->name().c_str(),
                        (unsigned long long)sm->dataoffset(), data);
        }
        else {
            shadingsys().warning ("attempting to set nonexistent parameter: %s", p.name().c_str());
        }
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
ShaderInstance::copy_code_from_master ()
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
    ASSERT (m_instoverrides.size() == (size_t)std::max(0,lastparam()));
    ASSERT (m_instsymbols.size() >= (size_t)std::max(0,lastparam()));
    if (m_instoverrides.size()) {
        for (size_t i = 0, e = lastparam();  i < e;  ++i) {
            if (m_instoverrides[i].valuesource() != Symbol::DefaultVal) {
                Symbol *si = &m_instsymbols[i];
                si->data (param_storage(i));
                si->valuesource (m_instoverrides[i].valuesource());
                si->connected_down (m_instoverrides[i].connected_down());
                si->lockgeom (m_instoverrides[i].lockgeom());
            }
        }
    }
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



std::string
ShaderInstance::print ()
{
    std::stringstream out;
    out << "Shader " << shadername() << "\n";
    out << "  symbols:\n";
    for (size_t i = 0;  i < m_instsymbols.size();  ++i) {
        const Symbol &s (*symbol(i));
        s.print (out, 256);
    }
#if 0
    out << "  int consts:\n    ";
    for (size_t i = 0;  i < m_iconsts.size();  ++i)
        out << m_iconsts[i] << ' ';
    out << "\n";
    out << "  float consts:\n    ";
    for (size_t i = 0;  i < m_fconsts.size();  ++i)
        out << m_fconsts[i] << ' ';
    out << "\n";
    out << "  string consts:\n    ";
    for (size_t i = 0;  i < m_sconsts.size();  ++i)
        out << "\"" << Strutil::escape_chars(m_sconsts[i]) << "\" ";
    out << "\n";
#endif
    out << "  code:\n";
    for (size_t i = 0;  i < m_instops.size();  ++i) {
        const Opcode &op (m_instops[i]);
        if (i == (size_t)maincodebegin())
            out << "(main)\n";
        out << "    " << i << ": " << op.opname();
        bool allconst = true;
        for (int a = 0;  a < op.nargs();  ++a) {
            const Symbol *s (argsymbol(op.firstarg()+a));
            out << " " << s->name();
            if (s->symtype() == SymTypeConst) {
                out << " (";
                s->print_vals(out,16);
                out << ")";
            }
            if (op.argread(a))
                allconst &= s->is_constant();
        }
        for (size_t j = 0;  j < Opcode::max_jumps;  ++j)
            if (op.jump(j) >= 0)
                out << " " << op.jump(j);
//        out << "    rw " << Strutil::format("%x",op.argread_bits())
//            << ' ' << op.argwrite_bits();
        if (op.argtakesderivs_all())
            out << " %derivs(" << op.argtakesderivs_all() << ") ";
        if (allconst)
            out << "  CONST";
        std::string filename = op.sourcefile().string();
        size_t slash = filename.find_last_of ("/");
        if (slash != std::string::npos)
            filename.erase (0, slash+1);
        if (filename.length())
            out << "  (" << filename << ":" << op.sourceline() << ")";
        out << "\n";
    }
    return out.str ();
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
                if (! sym->everused())
                    continue;
                return false;
            }
        }
    }

    // Make sure that the two nodes have the same parameter values.  If
    // the group has already been optimized, it's got an
    // instance-specific symbol table to check; but if it hasn't been
    // optimized, we check the symbol table int he master.
    for (int i = firstparam();  i < lastparam();  ++i) {
        const Symbol *sym = optimized ? symbol(i) : mastersymbol(i);
        if (! sym->everused())
            continue;
        if (sym->typespec().is_closure())
            continue;   // Closures can't have instance override values
        if ((sym->valuesource() == Symbol::InstanceVal || sym->valuesource() == Symbol::DefaultVal)
            && memcmp (param_storage(i), b.param_storage(i),
                       sym->typespec().simpletype().size())) {
            return false;
        }
    }

    if (m_run_lazily != b.m_run_lazily) {
        return false;
    }

    // The connection list need to be the same for the two shaders.
    if (m_connections.size() != b.m_connections.size()) {
        return false;
    }
    if (m_connections != b.m_connections) {
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



ShaderGroup::ShaderGroup (const char *name)
  : m_name(name),
    m_llvm_compiled_version(NULL), m_llvm_groupdata_size(0),
    m_optimized(0), m_does_nothing(false)
{
    m_executions = 0;
}



ShaderGroup::ShaderGroup (const ShaderGroup &g, const char *name)
  : m_name(name), m_layers(g.m_layers),
    m_llvm_compiled_version(NULL), m_llvm_groupdata_size(0),
    m_optimized(0), m_does_nothing(false)
{
    m_executions = 0;
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


OSL_NAMESPACE_EXIT
