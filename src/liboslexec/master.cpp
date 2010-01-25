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
ShaderMaster::resolve_syms ()
{
    m_firstparam = std::numeric_limits<int>::max();
    m_lastparam = -1;
    int i = 0;
    BOOST_FOREACH (Symbol &s, m_symbols) {
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
    // Make it easy for quick lookups of common symbols
    m_Psym = findsymbol (Strings::P);
    m_Nsym = findsymbol (Strings::N);
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



// Define a name/impl pair
struct OpNameEntry {
    const char *name;
    OpImpl impl;
};

// Static table of opcode names and implementations
static OpNameEntry op_name_entries[] = {
    { "aassign", OP_aassign },
    { "abs", OP_fabs },  // alias for fabs()
    { "acos", OP_acos },
    { "add", OP_add },
    { "and", OP_and },
    { "ashikhmin_velvet", OP_ashikhmin_velvet},
    { "area", OP_area },
    { "aref", OP_aref },
    { "arraylength", OP_arraylength },
    { "asin", OP_asin },
    { "assign", OP_assign },
    { "atan", OP_atan },
    { "atan2", OP_atan2 },
    { "background", OP_background },
    { "bitand", OP_bitand },
    { "bitor", OP_bitor },
    { "bssrdf_cubic", OP_bssrdf_cubic },
    { "calculatenormal", OP_calculatenormal },
    { "ceil", OP_ceil },
    { "cellnoise", OP_cellnoise },
    { "clamp", OP_clamp },
    { "cloth", OP_cloth },
    { "color", OP_color },
    { "compassign", OP_compassign },
    { "compl", OP_compl },
    { "compref", OP_compref },
    { "concat", OP_concat },
    { "cos", OP_cos },
    { "cosh", OP_cosh },
    { "cross", OP_cross },
    { "degrees", OP_degrees },
    { "determinant", OP_determinant },
    { "dielectric", OP_dielectric },
    { "diffuse", OP_diffuse },
    { "distance", OP_distance },
    { "div", OP_div },
    { "dot", OP_dot },
    { "Dx", OP_Dx },
    { "Dy", OP_Dy },
    { "dowhile", OP_dowhile },
    { "emission", OP_emission },
    { "end", OP_end },
    { "endswith", OP_endswith },
    { "eq", OP_eq },
    { "erf", OP_erf },
    { "erfc", OP_erfc },
    { "error", OP_error },
    { "exp", OP_exp },
    { "exp2", OP_exp2 },
    { "expm1", OP_expm1 },
    { "fabs", OP_fabs },
    { "filterwidth", OP_filterwidth },
    { "floor", OP_floor },
    { "fmod", OP_mod },  // alias for mod()
    { "for", OP_for },
    { "format", OP_format },
    { "fresnel", OP_fresnel },
    { "ge", OP_ge },
    { "getattribute", OP_getattribute },
    { "getmessage", OP_getmessage },
    { "gettextureinfo", OP_gettextureinfo },
    { "gt", OP_gt },
    { "hair_diffuse", OP_hair_diffuse },
    { "hair_specular", OP_hair_specular },
    { "hypot", OP_hypot },
    { "if", OP_if },
    { "inversesqrt", OP_inversesqrt },
    { "iscameraray", OP_iscameraray },
    { "isfinite", OP_isfinite },
    { "isinf", OP_isinf },
    { "isnan", OP_isnan },
    { "isshadowray", OP_isshadowray },
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
    { "max", OP_max },
    { "microfacet_beckmann", OP_microfacet_beckmann },
    { "microfacet_beckmann_refraction", OP_microfacet_beckmann_refraction },
    { "microfacet_ggx", OP_microfacet_ggx },
    { "microfacet_ggx_refraction", OP_microfacet_ggx_refraction },
    { "min", OP_min },
    { "mix", OP_mix },
    { "mod", OP_mod },
    { "mul", OP_mul },
    { "neg", OP_neg },
    { "neq", OP_neq },
    { "noise", OP_noise },
    { "nop", OP_nop },
    { "normal", OP_normal },
    { "normalize", OP_normalize },
    { "or", OP_or },
    { "phong", OP_phong },
    { "phong_ramp", OP_phong_ramp },
    { "pnoise", OP_pnoise },
    { "point", OP_point },
    { "pow", OP_pow },
    { "printf", OP_printf },
    { "psnoise", OP_psnoise },
    { "radians", OP_radians },
    { "reflect", OP_reflect },
    { "reflection", OP_reflection },
    { "refract", OP_refract },
    { "refraction", OP_refraction },
    { "regex_match", OP_regex_match },
    { "regex_search", OP_regex_search },
    { "round", OP_round },
    { "setmessage", OP_setmessage },
    { "shl", OP_shl },
    { "shr", OP_shr },
    { "sign", OP_sign },
    { "sin", OP_sin },
    { "sinh", OP_sinh },
    { "smoothstep", OP_smoothstep },
    { "snoise", OP_snoise },
    { "sqrt", OP_sqrt },
    { "startswith", OP_startswith },
    { "step", OP_step },
    { "strlen", OP_strlen },
    { "sub", OP_sub },
    { "substr", OP_substr },
    { "surfacearea", OP_surfacearea },
    { "tan", OP_tan },
    { "tanh", OP_tanh },
    { "texture", OP_texture },
    { "transform", OP_transform },
    { "transformn", OP_transformn },
    { "transformv", OP_transformv },
    { "translucent", OP_translucent },
    { "transparent", OP_transparent },
    { "transpose", OP_transpose },
    { "trunc", OP_trunc },
    { "useparam", OP_useparam },
    { "vector", OP_vector },
    { "ward", OP_ward },
    { "warning", OP_warning },
    { "westin_backscatter", OP_westin_backscatter},
    { "westin_sheen", OP_westin_sheen},
    { "while", OP_for },
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
