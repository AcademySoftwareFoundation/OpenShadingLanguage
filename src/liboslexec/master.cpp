/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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

#include <boost/foreach.hpp>

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"

#include "oslexec_pvt.h"
#include "oslops.h"



#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {   // OSL::pvt


int
ShaderMaster::findsymbol (ustring name) const
{
    for (size_t i = 0;  i < m_symbols.size();  ++i)
        if (m_symbols[i].name() == name)
            return (int)i;
    return -1;
}



int
ShaderMaster::findparam (ustring name) const
{
    for (int i = m_firstparam;  i <= m_lastparam;  ++i)
        if (m_symbols[i].name() == name)
            return i;
    return -1;
}



void
ShaderMaster::resolve_defaults ()
{
    m_firstparam = std::numeric_limits<int>::max();
    m_lastparam = -1;
    int i = 0;
    BOOST_FOREACH (Symbol &s, m_symbols) {
        if (s.symtype() == SymTypeParam || s.symtype() == SymTypeOutputParam) {
            if (m_firstparam > i)
                m_firstparam = i;
            m_lastparam = i;
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
}



void
ShaderMaster::print ()
{
    std::cout << "Shader " << m_shadername << " type=" 
              << shadertypename(m_shadertype) << "\n";
    std::cout << "  path = " << m_osofilename << "\n";
    std::cout << "  symbols:\n";
    for (size_t i = 0;  i < m_symbols.size();  ++i) {
        const Symbol &s (m_symbols[i]);
        std::cout << "    " << i << ": " << s.typespec().string() 
                  << " " << s.name() << "\n";
    }
    std::cout << "  int consts:\n    ";
    for (size_t i = 0;  i < m_iconsts.size();  ++i)
        std::cout << m_iconsts[i] << ' ';
    std::cout << "\n";
    std::cout << "  float consts:\n    ";
    for (size_t i = 0;  i < m_fconsts.size();  ++i)
        std::cout << m_fconsts[i] << ' ';
    std::cout << "\n";
    std::cout << "  string consts:\n    ";
    for (size_t i = 0;  i < m_sconsts.size();  ++i)
        std::cout << "\"" << m_sconsts[i] << "\" ";
    std::cout << "\n";
    std::cout << "  int defaults:\n    ";
    for (size_t i = 0;  i < m_idefaults.size();  ++i)
        std::cout << m_idefaults[i] << ' ';
    std::cout << "\n";
    std::cout << "  float defaults:\n    ";
    for (size_t i = 0;  i < m_fdefaults.size();  ++i)
        std::cout << m_fdefaults[i] << ' ';
    std::cout << "\n";
    std::cout << "  string defaults:\n    ";
    for (size_t i = 0;  i < m_sdefaults.size();  ++i)
        std::cout << "\"" << m_sdefaults[i] << "\" ";
    std::cout << "\n";
    std::cout << "  code:\n";
    for (size_t i = 0;  i < m_ops.size();  ++i) {
        std::cout << "    " << i << ": " << m_ops[i].opname();
        for (size_t a = 0;  a < m_ops[i].nargs();  ++a)
            std::cout << " " << m_symbols[m_args[m_ops[i].firstarg()+a]].name();
        for (size_t j = 0;  j < Opcode::max_jumps;  ++j)
            if (m_ops[i].jump(j) >= 0)
                std::cout << " " << m_ops[i].jump(j);
        if (m_ops[i].sourcefile())
            std::cout << "\t(" << m_ops[i].sourcefile() << ":" 
                      << m_ops[i].sourceline() << ")";
        std::cout << "\n";
    }
}



// Define a name/impl pair
struct OpNameEntry {
    const char *name;
    OpImpl impl;
};

// Static table of opcode names and implementations
static OpNameEntry op_name_entries[] = {
    { "acos", OP_acos },
    { "add", OP_add },
    { "asin", OP_asin },
    { "assign", OP_assign },
    { "atan", OP_atan },
    { "bitand", OP_bitand },
    { "bitor", OP_bitor },
    { "color", OP_color },
    { "compassign", OP_compassign },
    { "compl", OP_compl },
    { "compref", OP_compref },
    { "cos", OP_cos },
    { "cosh", OP_cosh },
    { "cross", OP_cross },
    { "degrees", OP_degrees },
    { "determinant", OP_determinant },
    { "distance", OP_distance },
    { "div", OP_div },
    { "dot", OP_dot },
    { "end", OP_end },
    { "eq", OP_eq },
    { "exp", OP_exp },
    { "exp2", OP_exp2 },
    { "expm1", OP_expm1 },
    { "fabs", OP_fabs },
    { "floor", OP_floor },
    { "ge", OP_ge },
    { "gt", OP_gt },
    { "if", OP_if },
    { "le", OP_le },
    { "length", OP_length },
    { "log", OP_log },
    { "log10", OP_log10 },
    { "log2", OP_log2 },
    { "logb", OP_logb },
    { "lt", OP_lt },
    { "luminance", OP_luminance },
    { "matrix", OP_matrix },
    { "mxcompassign", OP_mxcompassign },
    { "mxcompref", OP_mxcompref },
    { "mod", OP_mod },
    { "mul", OP_mul },
    { "neg", OP_neg },
    { "neq", OP_neq },
    { "normal", OP_normal },
    { "normalize", OP_normalize },
    { "point", OP_point },
    { "printf", OP_printf },
    { "radians", OP_radians },
    { "shl", OP_shl },
    { "shr", OP_shr },
    { "sin", OP_sin },
    { "sinh", OP_sinh },
    { "sub", OP_sub },
    { "tan", OP_tan },
    { "tanh", OP_tanh },
    { "transpose", OP_transpose },
    { "vector", OP_vector },
    { "xor", OP_xor },
    { NULL, NULL}
};

// Map for fast opname->implementation lookup
static std::map<ustring,OpImpl> ops_table;
// Mutex to guard the table
static mutex ops_table_mutex;


void
ShaderMaster::resolve_ops ()
{
    {
        // Make sure ops_table has been initialized
        lock_guard lock (ops_table_mutex);
        if (ops_table.empty()) {
            for (int i = 0;  op_name_entries[i].name;  ++i)
                ops_table[ustring(op_name_entries[i].name)] = 
                    op_name_entries[i].impl;
        }
    }

    BOOST_FOREACH (Opcode &op, m_ops) {
        if (shadingsys().debug())
            std::cout << "resolving " << op.opname() << "\n";
        std::map<ustring,OpImpl>::const_iterator found;
        found = ops_table.find (op.opname());
        if (found != ops_table.end())
            op.implementation (found->second);
        else
            op.implementation (OP_missing);
    }
}


}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
