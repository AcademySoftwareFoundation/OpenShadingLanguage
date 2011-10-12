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
#include <limits>
#include <sstream>

#include <boost/foreach.hpp>

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"

#include "oslexec_pvt.h"
#include "../liboslcomp/oslcomp_pvt.h"



#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {   // OSL::pvt


#ifdef OIIO_NAMESPACE
using OIIO::spin_lock;
#endif


ShaderMaster::~ShaderMaster ()
{
    // Adjust statistics
    size_t opmem = vectorbytes (m_ops);
    size_t argmem = vectorbytes (m_args);
    size_t symmem = vectorbytes (m_symbols);
    size_t defaultmem = vectorbytes (m_idefaults) 
        + vectorbytes (m_fdefaults) + vectorbytes (m_sdefaults);
    size_t constmem = vectorbytes (m_iconsts)
        + vectorbytes (m_fconsts) + vectorbytes (m_sconsts);
    size_t totalmem = (opmem + argmem + symmem + defaultmem +
                       constmem + sizeof(ShaderMaster));
    {
        ShadingSystemImpl &ss (shadingsys());
        spin_lock lock (ss.m_stat_mutex);
        ss.m_stat_mem_master_ops -= opmem;
        ss.m_stat_mem_master_args -= argmem;
        ss.m_stat_mem_master_syms -= symmem;
        ss.m_stat_mem_master_defaults -= defaultmem;
        ss.m_stat_mem_master_consts -= constmem;
        ss.m_stat_mem_master -= totalmem;
        ss.m_stat_memory -= totalmem;
    }
}



int
ShaderMaster::findsymbol (ustring name) const
{
    for (size_t i = 0;  i < m_symbols.size();  ++i)
        if (m_symbols[i].name() == name)
            return (int)i;
    return -1;
}



void
ShaderMaster::resolve_syms ()
{
    SymbolPtrVec allsymptrs;
    allsymptrs.reserve (m_symbols.size());
    m_firstparam = -1;
    m_lastparam = -1;
    int i = 0;
    BOOST_FOREACH (Symbol &s, m_symbols) {
        allsymptrs.push_back (&s);
        // Fix up the size of the symbol's data (for one point, not 
        // counting derivatives).
        if (s.typespec().is_closure()) {
            s.size (sizeof (ClosureColor *)); // heap stores ptrs to closures
        } else if (s.typespec().is_structure()) {
            // structs are just placeholders, their fields are separate
            // symbols that hold the real data.
            s.size (0);
        } else {
            s.size (s.typespec().simpletype().size());
            // FIXME -- some day we may want special padding here, like
            // if we REALLY want 3-vectors to take 16 bytes for HW SIMD
            // reasons.
        }

        if (s.symtype() == SymTypeParam || s.symtype() == SymTypeOutputParam) {
            if (m_firstparam < 0)
                m_firstparam = i;
            m_lastparam = i+1;
            if (s.dataoffset() >= 0) {
                if (s.typespec().simpletype().basetype == TypeDesc::INT)
                    s.data (&(m_idefaults[s.dataoffset()]));
                else if (s.typespec().simpletype().basetype == TypeDesc::FLOAT)
                    s.data (&(m_fdefaults[s.dataoffset()]));
                else if (s.typespec().simpletype().basetype == TypeDesc::STRING)
                    s.data (&(m_sdefaults[s.dataoffset()]));
            }
        }
        if (s.symtype() == SymTypeConst) {
            if (s.dataoffset() >= 0) {
                if (s.typespec().simpletype().basetype == TypeDesc::INT)
                    s.data (&(m_iconsts[s.dataoffset()]));
                else if (s.typespec().simpletype().basetype == TypeDesc::FLOAT)
                    s.data (&(m_fconsts[s.dataoffset()]));
                else if (s.typespec().simpletype().basetype == TypeDesc::STRING)
                    s.data (&(m_sconsts[s.dataoffset()]));
            }
        }
        ++i;
    }

    // Re-track variable lifetimes
    SymbolPtrVec oparg_ptrs;
    oparg_ptrs.reserve (m_args.size());
    BOOST_FOREACH (int a, m_args)
        oparg_ptrs.push_back (symbol (a));
    OSLCompilerImpl::track_variable_lifetimes (m_ops, oparg_ptrs, allsymptrs);

    // Adjust statistics
    size_t opmem = vectorbytes (m_ops);
    size_t argmem = vectorbytes (m_args);
    size_t symmem = vectorbytes (m_symbols);
    size_t defaultmem = vectorbytes (m_idefaults) 
        + vectorbytes (m_fdefaults) + vectorbytes (m_sdefaults);
    size_t constmem = vectorbytes (m_iconsts)
        + vectorbytes (m_fconsts) + vectorbytes (m_sconsts);
    size_t totalmem = (opmem + argmem + symmem + defaultmem +
                       constmem + sizeof(ShaderMaster));
    {
        ShadingSystemImpl &ss (shadingsys());
        spin_lock lock (ss.m_stat_mutex);
        ss.m_stat_mem_master_ops += opmem;
        ss.m_stat_mem_master_args += argmem;
        ss.m_stat_mem_master_syms += symmem;
        ss.m_stat_mem_master_defaults += defaultmem;
        ss.m_stat_mem_master_consts += constmem;
        ss.m_stat_mem_master += totalmem;
        ss.m_stat_memory += totalmem;
    }
}



std::string
ShaderMaster::print ()
{
    std::stringstream out;
    out << "Shader " << m_shadername << " type=" 
              << shadertypename(m_shadertype) << "\n";
    out << "  path = " << m_osofilename << "\n";
    out << "  symbols:\n";
    for (size_t i = 0;  i < m_symbols.size();  ++i) {
        const Symbol &s (m_symbols[i]);
        out << "    " << i << ": " << s.typespec().string() 
                  << " " << s.name() << "\n";
    }
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
    out << "  int defaults:\n    ";
    for (size_t i = 0;  i < m_idefaults.size();  ++i)
        out << m_idefaults[i] << ' ';
    out << "\n";
    out << "  float defaults:\n    ";
    for (size_t i = 0;  i < m_fdefaults.size();  ++i)
        out << m_fdefaults[i] << ' ';
    out << "\n";
    out << "  string defaults:\n    ";
    for (size_t i = 0;  i < m_sdefaults.size();  ++i)
        out << "\"" << m_sdefaults[i] << "\" ";
    out << "\n";
    out << "  code:\n";
    for (size_t i = 0;  i < m_ops.size();  ++i) {
        out << "    " << i << ": " << m_ops[i].opname();
        for (int a = 0;  a < m_ops[i].nargs();  ++a)
            out << " " << m_symbols[m_args[m_ops[i].firstarg()+a]].name();
        for (size_t j = 0;  j < Opcode::max_jumps;  ++j)
            if (m_ops[i].jump(j) >= 0)
                out << " " << m_ops[i].jump(j);
        if (m_ops[i].sourcefile())
            out << "\t(" << m_ops[i].sourcefile() << ":" 
                      << m_ops[i].sourceline() << ")";
        out << "\n";
    }
    return out.str ();
}

}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
