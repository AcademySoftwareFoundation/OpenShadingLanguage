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

#include <boost/foreach.hpp>

#include "OpenImageIO/dassert.h"
#include "OpenImageIO/strutil.h"

#include "oslexec_pvt.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {

namespace pvt {   // OSL::pvt


ShaderInstance::ShaderInstance (ShaderMaster::ref master,
                                const char *layername) 
    : m_master(master), m_instsymbols(m_master->m_symbols),
      m_instops(m_master->m_ops), m_instargs(m_master->m_args),
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

    // Make it easy for quick lookups of common symbols
    m_Psym = findsymbol (Strings::P);
    m_Nsym = findsymbol (Strings::N);
}



ShaderInstance::~ShaderInstance ()
{
    shadingsys().m_stat_instances -= 1;
}



int
ShaderInstance::findsymbol (ustring name) const
{
    for (size_t i = 0;  i < m_instsymbols.size();  ++i)
        if (m_instsymbols[i].name() == name)
            return (int)i;
    return -1;
}



int
ShaderInstance::findparam (ustring name) const
{
    for (int i = m_firstparam;  i <= m_lastparam;  ++i)
        if (m_instsymbols[i].name() == name)
            return i;
    return -1;
}



void
ShaderInstance::parameters (const ParamValueList &params)
{
    m_iparams = m_master->m_idefaults;
    m_fparams = m_master->m_fdefaults;
    m_sparams = m_master->m_sdefaults;
    BOOST_FOREACH (const ParamValue &p, params) {
        if (shadingsys().debug())
            shadingsys().info (" PARAMETER %s %s",
                               p.name().c_str(), p.type().c_str());
        int i = findparam (p.name());
        if (i >= 0) {
            Symbol *s = symbol(i);
            TypeSpec t = s->typespec();
            // don't allow assignment of closures
            if (t.is_closure()) {
                shadingsys().warning ("skipping assignment of closure: %s", s->name().c_str());
                continue;
            }
            if (t.is_structure())
                continue;
            // check type of parameter and matching symbol
            if (t.simpletype() != p.type()) {
                shadingsys().warning ("attempting to set parameter with wrong type: %s (exepected '%s', received '%s')", s->name().c_str(), t.c_str(), p.type().c_str());
                continue;
            }

            s->step (0);
            s->valuesource (Symbol::InstanceVal);
            if (t.simpletype().basetype == TypeDesc::INT) {
                s->data (&m_iparams[s->dataoffset()]);
            } else if (t.simpletype().basetype == TypeDesc::FLOAT) {
                s->data (&m_fparams[s->dataoffset()]);
            } else if (t.simpletype().basetype == TypeDesc::STRING) {
                s->data (&m_sparams[s->dataoffset()]);
            } else {
                ASSERT (0);
            }
            memcpy (s->data(), p.data(), t.simpletype().size());
            if (shadingsys().debug())
                shadingsys().info ("    sym %s offset %llu address %p",
                        s->name().c_str(),
                        (unsigned long long)s->dataoffset(), s->data());
        }
        else {
            shadingsys().warning ("attempting to set nonexistent parameter: %s", p.name().c_str());
        }
    }
}



void
ShaderInstance::make_symbol_room (size_t moresyms)
{
    if (m_instsymbols.capacity() < m_instsymbols.size()+moresyms)
        m_instsymbols.reserve (m_instsymbols.size() + moresyms + 10);
}



inline std::string
print_vals (const Symbol &s)
{
    std::stringstream out;
    TypeDesc t = s.typespec().simpletype();
    int n = t.aggregate * t.numelements();
    if (t.basetype == TypeDesc::FLOAT) {
        for (int j = 0;  j < n;  ++j)
            out << (j ? " " : "") << ((float *)s.data())[j];
    } else if (t.basetype == TypeDesc::INT) {
        for (int j = 0;  j < n;  ++j)
            out << (j ? " " : "") << ((int *)s.data())[j];
    } else if (t.basetype == TypeDesc::STRING) {
        for (int j = 0;  j < n;  ++j)
            out << (j ? " " : "") << "\"" << ((ustring *)s.data())[j] << "\"";
    }
    return out.str();
}



std::string
ShaderInstance::print ()
{
    std::stringstream out;
    out << "Shader " << shadername() << "\n";
    out << "  symbols:\n";
    for (size_t i = 0;  i < m_instsymbols.size();  ++i) {
        const Symbol &s (*symbol(i));
        out << "    " << i << ": " << Symbol::symtype_shortname(s.symtype())
            << " " << s.typespec().string() << " " << s.name();
        if (s.everused())
            out << " (used " << s.firstuse() << ' ' << s.lastuse() 
                << " read " << s.firstread() << ' ' << s.lastread() 
                << " write " << s.firstwrite() << ' ' << s.lastwrite();
        else
            out << " (unused";
        out << (s.has_derivs() ? " derivs" : "") << ")";
        if (s.symtype() == SymTypeParam || s.symtype() == SymTypeOutputParam) {
            if (s.has_init_ops())
                out << " init [" << s.initbegin() << ',' << s.initend() << ")";
            if (s.connected())
                out << " connected";
            if (s.connected_down())
                out << " down-connected";
            if (!s.connected() && !s.connected_down())
                out << " unconnected";
        }
        out << "\n";
        if (s.symtype() == SymTypeConst || 
            ((s.symtype() == SymTypeParam || s.symtype() == SymTypeOutputParam) &&
             s.valuesource() == Symbol::DefaultVal && !s.has_init_ops())) {
            if (s.symtype() == SymTypeConst)
                out << "\tconst: ";
            else
                out << "\tdefault: ";
            out << print_vals (s);
            out << "\n";
        }
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
        out << "\"" << m_sconsts[i] << "\" ";
    out << "\n";
#endif
    out << "  code:\n";
    for (size_t i = 0;  i < m_instops.size();  ++i) {
        const Opcode &op (m_instops[i]);
        out << "    " << i << ": " << op.opname();
        bool allconst = true;
        for (int a = 0;  a < op.nargs();  ++a) {
            const Symbol *s (argsymbol(op.firstarg()+a));
            out << " " << s->name();
            if (s->symtype() == SymTypeConst)
                out << " (" << print_vals(*s) << ")";
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
        out << "\n";
    }
    return out.str ();
}



ShaderGroup::ShaderGroup ()
    : m_llvm_compiled_version(NULL), m_optimized(0)
{
    m_executions = 0;
}



ShaderGroup::ShaderGroup (const ShaderGroup &g)
    : m_layers(g.m_layers), m_llvm_compiled_version(NULL), m_optimized(0)
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


}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
