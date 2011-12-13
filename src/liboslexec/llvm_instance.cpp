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

#include <cmath>
#include <cstddef> // FIXME: OIIO's timer.h depends on NULL being defined and should include this itself

#include <OpenImageIO/timer.h>

#include "llvm_headers.h"

// More LLVM headers that we only need for setup and calling the JIT and
// optimizer
#include <llvm/Analysis/Verifier.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Utils/UnifyFunctionExitNodes.h>
#if OSL_LLVM_VERSION <= 29
# include <llvm/Support/StandardPasses.h>
# include <llvm/Target/TargetSelect.h>
#else
# include <llvm/Support/TargetSelect.h>
#endif

#include "oslexec_pvt.h"
#include "../liboslcomp/oslcomp_pvt.h"
#include "runtimeoptimize.h"

/*
This whole file is concerned with taking our post-optimized OSO
intermediate code and translating it into LLVM IR code so we can JIT it
and run it directly, for an expected huge speed gain over running our
interpreter.

Schematically, we want to create code that resembles the following:

    // Assume 2 layers. 
    struct GroupData_1 {
        // Array of ints telling if we have already run each layer
        int layer_run[nlayers];
        // For each layer in the group, we declare all shader params
        // whose values are not known -- they have init ops, or are
        // interpolated from the geom, or are connected to other layers.
        float param_0_foo;   // number is layer ID
        float param_1_bar;
    };

    // Name of layer entry is $layer_ID
    void $layer_0 (ShaderGlobals *sg, GroupData_1 *group)
    {
        // Only run if not already done.  Then mark as run.
        if (group->layer_run[0])
            return;
        group->layer_run[0] = 1;

        // Declare locals, temps, constants, params with known values.
        // Make them all look like stack memory locations:
        float *x = alloca (sizeof(float));
        // ...and so on for all the other locals & temps...

        // then run the shader body:
        *x = sg->u * group->param_2_bar;
        group->param_1_foo = *x;
    }

    void $layer_0 (ShaderGlobals *sg, GroupData_1 *group)
    {
        if (group->layer_run[0])
            return;
        group->layer_run[0] = 1;
        // ...
        $layer_0 (sg, group);    // because we need its outputs
        *y = sg->u * group->$param_2_bar;
    }

    void $group_1 (ShaderGlobals *sg, GroupData_1 *group)
    {
        group->layer_run[...] = 0;
        // Run just the unconditional layers
        $layer_1 (sg, group);
    }

*/

extern int osl_llvm_compiled_ops_size;
extern char osl_llvm_compiled_ops_block[];

using namespace OSL;
using namespace OSL::pvt;

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {

#ifdef OIIO_NAMESPACE
using OIIO::spin_lock;
using OIIO::Timer;
#endif

static ustring op_end("end");
static ustring op_nop("nop");



#define NOISE_IMPL(name)                        \
    "osl_" #name "_ff",  "ff",                  \
    "osl_" #name "_fff", "fff",                 \
    "osl_" #name "_fv",  "fv",                  \
    "osl_" #name "_fvf", "fvf",                 \
    "osl_" #name "_vf",  "xvf",                 \
    "osl_" #name "_vff", "xvff",                \
    "osl_" #name "_vv",  "xvv",                 \
    "osl_" #name "_vvf", "xvvf"

#define NOISE_DERIV_IMPL(name)                  \
    "osl_" #name "_dfdf",   "xXX",              \
    "osl_" #name "_dfdff",  "xXXf",             \
    "osl_" #name "_dffdf",  "xXfX",             \
    "osl_" #name "_dfdfdf", "xXXX",             \
    "osl_" #name "_dfdv",   "xXv",              \
    "osl_" #name "_dfdvf",  "xXvf",             \
    "osl_" #name "_dfvdf",  "xXvX",             \
    "osl_" #name "_dfdvdf", "xXvX",             \
    "osl_" #name "_dvdf",   "xvX",              \
    "osl_" #name "_dvdff",  "xvXf",             \
    "osl_" #name "_dvfdf",  "xvfX",             \
    "osl_" #name "_dvdfdf", "xvXX",             \
    "osl_" #name "_dvdv",   "xvv",              \
    "osl_" #name "_dvdvf",  "xvvf",             \
    "osl_" #name "_dvvdf",  "xvvX",             \
    "osl_" #name "_dvdvdf", "xvvX"

#define GENERIC_NOISE_DERIV_IMPL(name)          \
    "osl_" #name "_dfdf",   "xsXXXX",           \
    "osl_" #name "_dfdfdf", "xsXXXXX",          \
    "osl_" #name "_dfdv",   "xsXXXX",           \
    "osl_" #name "_dfdvdf", "xsXXXXX",          \
    "osl_" #name "_dvdf",   "xsXXXX",           \
    "osl_" #name "_dvdfdf", "xsXXXXX",          \
    "osl_" #name "_dvdv",   "xsXXXX",           \
    "osl_" #name "_dvdvdf", "xsXXXXX"

#define PNOISE_IMPL(name)                       \
    "osl_" #name "_fff",   "fff",               \
    "osl_" #name "_fffff", "fffff",             \
    "osl_" #name "_fvv",   "fvv",               \
    "osl_" #name "_fvfvf", "fvfvf",             \
    "osl_" #name "_vff",   "xvff",              \
    "osl_" #name "_vffff", "xvffff",            \
    "osl_" #name "_vvv",   "xvvv",              \
    "osl_" #name "_vvfvf", "xvvfvf"

#define PNOISE_DERIV_IMPL(name)                 \
    "osl_" #name "_dfdff",    "xXXf",           \
    "osl_" #name "_dfdffff",  "xXXfff",         \
    "osl_" #name "_dffdfff",  "xXfXff",         \
    "osl_" #name "_dfdfdfff", "xXXXff",         \
    "osl_" #name "_dfdvv",    "xXXv",           \
    "osl_" #name "_dfdvfvf",  "xXvfvf",         \
    "osl_" #name "_dfvdfvf",  "xXvXvf",         \
    "osl_" #name "_dfdvdfvf", "xXvXvf",         \
    "osl_" #name "_dvdff",    "xvXf",           \
    "osl_" #name "_dvdffff",  "xvXfff",         \
    "osl_" #name "_dvfdfff",  "xvfXff",         \
    "osl_" #name "_dvdfdfff", "xvXXff",         \
    "osl_" #name "_dvdvv",    "xvvv",           \
    "osl_" #name "_dvdvfvf",  "xvvfvf",         \
    "osl_" #name "_dvvdfvf",  "xvvXvf",         \
    "osl_" #name "_dvdvdfvf", "xvvXvf"

#define GENERIC_PNOISE_DERIV_IMPL(name)         \
    "osl_" #name "_dfdff",    "xsXXfXX",        \
    "osl_" #name "_dfdfdfff", "xsXXXffXX",      \
    "osl_" #name "_dfdvv",    "xsXXvXX",        \
    "osl_" #name "_dfdvdfvf", "xsXvXvfXX",      \
    "osl_" #name "_dvdff",    "xsvXfXX",        \
    "osl_" #name "_dvdfdfff", "xsvXXffXX",      \
    "osl_" #name "_dvdvv",    "xsvvvXX",        \
    "osl_" #name "_dvdvdfvf", "xsvvXvfXX"

#define UNARY_OP_IMPL(name)                     \
    "osl_" #name "_ff",   "ff",                 \
    "osl_" #name "_dfdf", "xXX",                \
    "osl_" #name "_vv",   "xXX",                \
    "osl_" #name "_dvdv", "xXX"

#define BINARY_OP_IMPL(name)                    \
    "osl_" #name "_fff",    "fff",              \
    "osl_" #name "_dfdfdf", "xXXX",             \
    "osl_" #name "_dffdf",  "xXfX",             \
    "osl_" #name "_dfdff",  "xXXf",             \
    "osl_" #name "_vvv",    "xXXX",             \
    "osl_" #name "_dvdvdv", "xXXX",             \
    "osl_" #name "_dvvdv",  "xXXX",             \
    "osl_" #name "_dvdvv",  "xXXX"

/// Table of all functions that we may call from the LLVM-compiled code.
/// Alternating name and argument list, much like we use in oslc's type
/// checking.  Note that nothing that's compiled into llvm_ops.cpp ought
/// to need a declaration here.
static const char *llvm_helper_function_table[] = {
    // TODO: remove these
    "osl_add_closure_closure", "CXCC",
    "osl_mul_closure_float", "CXCf",
    "osl_mul_closure_color", "CXCc",
    "osl_allocate_closure_component", "CXiii",
    "osl_closure_to_string", "sXC",
    "osl_format", "ss*",
    "osl_printf", "xXs*",
    "osl_error", "xXs*",
    "osl_warning", "xXs*",
#if 1
    NOISE_IMPL(cellnoise),
    NOISE_IMPL(noise),
    NOISE_DERIV_IMPL(noise),
    NOISE_IMPL(snoise),
    NOISE_DERIV_IMPL(snoise),
    GENERIC_NOISE_DERIV_IMPL(genericnoise),
    PNOISE_IMPL(pcellnoise),
    PNOISE_IMPL(pnoise),
    PNOISE_DERIV_IMPL(pnoise),
    PNOISE_IMPL(psnoise),
    PNOISE_DERIV_IMPL(psnoise),
    GENERIC_PNOISE_DERIV_IMPL(genericpnoise),
#endif
    "osl_spline_fff", "xXXXXi",
    "osl_spline_dfdfdf", "xXXXXi",
    "osl_spline_dfdff", "xXXXXi",
    "osl_spline_dffdf", "xXXXXi",
    "osl_spline_vfv", "xXXXXi",
    "osl_spline_dvdfdv", "xXXXXi",
    "osl_spline_dvdfv", "xXXXXi",
    "osl_spline_dvfdv", "xXXXXi",
    "osl_splineinverse_fff", "xXXXXi",
    "osl_splineinverse_dfdfdf", "xXXXXi",
    "osl_splineinverse_dfdff", "xXXXXi",
    "osl_splineinverse_dffdf", "xXXXXi",
    "osl_setmessage", "xXsLXisi",
    "osl_getmessage", "iXssLXiisi",
    "osl_pointcloud_search", "iXsXfiXXii*",
    "osl_pointcloud_get", "iXsXisLX",
    "osl_blackbody_vf", "xXXf",
    "osl_wavelength_color_vf", "xXXf",
    "osl_luminance_fv", "xXXX",
    "osl_luminance_dfdv", "xXXX",

#ifdef OSL_LLVM_NO_BITCODE
    "osl_assert_nonnull", "xXs",

    UNARY_OP_IMPL(sin),
    UNARY_OP_IMPL(cos),
    UNARY_OP_IMPL(tan),

    UNARY_OP_IMPL(asin),
    UNARY_OP_IMPL(acos),
    UNARY_OP_IMPL(atan),
    BINARY_OP_IMPL(atan2),
    UNARY_OP_IMPL(sinh),
    UNARY_OP_IMPL(cosh),
    UNARY_OP_IMPL(tanh),

    "osl_sincos_fff", "xfXX",
    "osl_sincos_dfdff", "xXXX",
    "osl_sincos_dffdf", "xXXX",
    "osl_sincos_dfdfdf", "xXXX",
    "osl_sincos_vvv", "xXXX",
    "osl_sincos_dvdvv", "xXXX",
    "osl_sincos_dvvdv", "xXXX",
    "osl_sincos_dvdvdv", "xXXX",

    UNARY_OP_IMPL(log),
    UNARY_OP_IMPL(log2),
    UNARY_OP_IMPL(log10),
    UNARY_OP_IMPL(logb),
    UNARY_OP_IMPL(exp),
    UNARY_OP_IMPL(exp2),
    UNARY_OP_IMPL(expm1),
    BINARY_OP_IMPL(pow),
    UNARY_OP_IMPL(erf),
    UNARY_OP_IMPL(erfc),

    "osl_pow_vvf", "xXXf",
    "osl_pow_dvdvdf", "xXXX",
    "osl_pow_dvvdf", "xXXX",
    "osl_pow_dvdvf", "xXXX",

    UNARY_OP_IMPL(sqrt),
    UNARY_OP_IMPL(inversesqrt),

    "osl_floor_ff", "ff",
    "osl_floor_vv", "xXX",
    "osl_ceil_ff", "ff",
    "osl_ceil_vv", "xXX",
    "osl_round_ff", "ff",
    "osl_round_vv", "xXX",
    "osl_trunc_ff", "ff",
    "osl_trunc_vv", "xXX",
    "osl_sign_ff", "ff",
    "osl_sign_vv", "XX",
    "osl_step_fff", "fff",
    "osl_step_vvv", "xXXX",

    "osl_isnan_if", "if",
    "osl_isinf_if", "if",
    "osl_isfinite_if", "if",
    "osl_abs_ii", "ii",
    "osl_fabs_ii", "ii",

    UNARY_OP_IMPL(abs),
    UNARY_OP_IMPL(fabs),

    "osl_smoothstep_ffff", "ffff",
    "osl_smoothstep_dfffdf", "xXffX",
    "osl_smoothstep_dffdff", "xXfXf",
    "osl_smoothstep_dffdfdf", "xXfXX",
    "osl_smoothstep_dfdfff", "xXXff",
    "osl_smoothstep_dfdffdf", "xXXfX",
    "osl_smoothstep_dfdfdff", "xXXXf",
    "osl_smoothstep_dfdfdfdf", "xXXXX",

    "osl_transform_vmv", "xXXX",
    "osl_transform_dvmdv", "xXXX",
    "osl_transformv_vmv", "xXXX",
    "osl_transformv_dvmdv", "xXXX",
    "osl_transformn_vmv", "xXXX",
    "osl_transformn_dvmdv", "xXXX",

    "osl_mul_mm", "xXXX",
    "osl_mul_mf", "xXXf",
    "osl_mul_m_ff", "xXff",
    "osl_div_mm", "xXXX",
    "osl_div_mf", "xXXf",
    "osl_div_fm", "xXfX",
    "osl_div_m_ff", "xXff",
    "osl_prepend_matrix_from", "iXXs",
    "osl_get_from_to_matrix", "iXXss",
    "osl_transpose_mm", "xXX",
    "osl_determinant_fm", "fX",

    "osl_dot_fvv", "fXX",
    "osl_dot_dfdvdv", "xXXX",
    "osl_dot_dfdvv", "xXXX",
    "osl_dot_dfvdv", "xXXX",
    "osl_cross_vvv", "xXXX",
    "osl_cross_dvdvdv", "xXXX",
    "osl_cross_dvdvv", "xXXX",
    "osl_cross_dvvdv", "xXXX",
    "osl_length_fv", "fX",
    "osl_length_dfdv", "xXX",
    "osl_distance_fvv", "fXX",
    "osl_distance_dfdvdv", "xXXX",
    "osl_distance_dfdvv", "xXXX",
    "osl_distance_dfvdv", "xXXX",
    "osl_normalize_vv", "xXX",
    "osl_normalize_dvdv", "xXX",
    "osl_prepend_color_from", "xXXs",

    "osl_concat_sss", "sss",
    "osl_strlen_is", "is",
    "osl_startswith_iss", "iss",
    "osl_endswith_iss", "iss",
    "osl_substr_ssii", "ssii",
    "osl_regex_impl", "iXsXisi",

    "osl_texture_clear", "xX",
    "osl_texture_set_firstchannel", "xXi",
    "osl_texture_set_swrap", "xXs",
    "osl_texture_set_twrap", "xXs",
    "osl_texture_set_rwrap", "xXs",
    "osl_texture_set_sblur", "xXf",
    "osl_texture_set_tblur", "xXf",
    "osl_texture_set_rblur", "xXf",
    "osl_texture_set_swidth", "xXf",
    "osl_texture_set_twidth", "xXf",
    "osl_texture_set_rwidth", "xXf",
    "osl_texture_set_fill", "xXf",
    "osl_texture_set_time", "xXf",
    "osl_texture", "iXsXffffffiXXX",
    "osl_texture_alpha", "iXsXffffffiXXXXXX",
    "osl_texture3d", "iXsXXXXXiXXXX",
    "osl_texture3d_alpha", "iXsXXXXXiXXXXXXXX",
    "osl_environment", "iXsXXXXiXXXXXX",
    "osl_get_textureinfo", "iXXXiiiX",

    "osl_trace_clear", "xX",
    "osl_trace_set_mindist", "xXf",
    "osl_trace_set_maxdist", "xXf",
    "osl_trace_set_shade", "xXi",
    "osl_trace", "iXXXXXXXX",

    "osl_get_attribute", "iXiXXiiXX",
    "osl_calculatenormal", "xXXX",
    "osl_area", "fX",
    "osl_filterwidth_fdf", "fX",
    "osl_filterwidth_vdv", "xXX",
    "osl_dict_find_iis", "iXiX",
    "osl_dict_find_iss", "iXXX",
    "osl_dict_next", "iXi",
    "osl_dict_value", "iXiXLX",
    "osl_raytype_name", "iXX",
    "osl_raytype_bit", "iXi",
    "osl_bind_interpolated_param", "iXXLiX",
#endif // OSL_LLVM_NO_BITCODE

    NULL
};



llvm::Type *
RuntimeOptimizer::llvm_type_sg ()
{
    // Create a type that defines the ShaderGlobals for LLVM IR.  This
    // absolutely MUST exactly match the ShaderGlobals struct in oslexec.h.
    if (m_llvm_type_sg)
        return m_llvm_type_sg;

    // Derivs look like arrays of 3 values
    llvm::Type *float_deriv = llvm_type (TypeDesc(TypeDesc::FLOAT, TypeDesc::SCALAR, 3));
    llvm::Type *triple_deriv = llvm_type (TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, 3));
    std::vector<llvm::Type*> sg_types;
    sg_types.push_back (triple_deriv);        // P, dPdx, dPdy
    sg_types.push_back (llvm_type_triple());  // dPdz
    sg_types.push_back (triple_deriv);        // I, dIdx, dIdy
    sg_types.push_back (llvm_type_triple());  // N
    sg_types.push_back (llvm_type_triple());  // Ng
    sg_types.push_back (float_deriv);         // u, dudx, dudy
    sg_types.push_back (float_deriv);         // v, dvdx, dvdy
    sg_types.push_back (llvm_type_triple());  // dPdu
    sg_types.push_back (llvm_type_triple());  // dPdv
    sg_types.push_back (llvm_type_float());   // time
    sg_types.push_back (llvm_type_float());   // dtime
    sg_types.push_back (llvm_type_triple());  // dPdtime
    sg_types.push_back (triple_deriv);        // Ps

    sg_types.push_back(llvm_type_void_ptr()); // opaque renderstate*
    sg_types.push_back(llvm_type_void_ptr()); // opaque tracedata*
    sg_types.push_back(llvm_type_void_ptr()); // opaque objdata*
    sg_types.push_back(llvm_type_void_ptr()); // ShadingContext*
    sg_types.push_back(llvm_type_void_ptr()); // object2common
    sg_types.push_back(llvm_type_void_ptr()); // shader2common
    sg_types.push_back(llvm_type_void_ptr()); // Ci

    sg_types.push_back (llvm_type_float());   // surfacearea
    sg_types.push_back (llvm_type_int());     // raytype
    sg_types.push_back (llvm_type_int());     // flipHandedness
    sg_types.push_back (llvm_type_int());     // backfacing

    return m_llvm_type_sg = llvm_type_struct (sg_types);
}



llvm::Type *
RuntimeOptimizer::llvm_type_sg_ptr ()
{
    return (llvm::Type *) llvm::PointerType::get (llvm_type_sg(), 0);
}



llvm::Type *
RuntimeOptimizer::llvm_type_groupdata ()
{
    // If already computed, return it
    if (m_llvm_type_groupdata)
        return m_llvm_type_groupdata;

    std::vector<llvm::Type*> fields;

    // First, add the array that tells if each layer has run.  But only make
    // slots for the layers that may be called/used.
    int sz = (m_num_used_layers + 3) & (~3);  // Round up to 32 bit boundary
    fields.push_back ((llvm::Type *)llvm::ArrayType::get(llvm_type_bool(), sz));
    size_t offset = sz * sizeof(bool);

    // For each layer in the group, add entries for all params that are
    // connected or interpolated, and output params.  Also mark those
    // symbols with their offset within the group struct.
    if (shadingsys().llvm_debug() >= 2)
        std::cout << "Group param struct:\n";
    m_param_order_map.clear ();
    int order = 1;
    for (int layer = 0;  layer < m_group.nlayers();  ++layer) {
        ShaderInstance *inst = m_group[layer];
        if (inst->unused())
            continue;
        FOREACH_PARAM (Symbol &sym, inst) {
            TypeSpec ts = sym.typespec();
            if (ts.is_structure())  // skip the struct symbol itself
                continue;
            int arraylen = std::max (1, sym.typespec().arraylength());
            int n = arraylen * (sym.has_derivs() ? 3 : 1);
            ts.make_array (n);
            fields.push_back (llvm_type (ts));

            // Alignment
            size_t align = sym.typespec().is_closure_based() ? sizeof(void*) :
                    sym.typespec().simpletype().basesize();
            if (offset & (align-1))
                offset += align - (offset & (align-1));
            if (shadingsys().llvm_debug() >= 2)
                std::cout << "  " << inst->layername() 
                          << " (" << inst->id() << ") " << sym.mangled()
                          << " " << ts.c_str() << ", field " << order 
                          << ", offset " << offset << std::endl;
            sym.dataoffset ((int)offset);
            offset += n * int(sym.size());

            m_param_order_map[&sym] = order;
            ++order;
        }
    }
    m_group.llvm_groupdata_size (offset);

    m_llvm_type_groupdata = llvm_type_struct (fields);

#ifdef DEBUG
//    llvm::outs() << "\nGroup struct = " << *m_llvm_type_groupdata << "\n";
//    llvm::outs() << "  size = " << offset << "\n";
#endif

    return m_llvm_type_groupdata;
}



llvm::Type *
RuntimeOptimizer::llvm_type_groupdata_ptr ()
{
    return llvm::PointerType::get (llvm_type_groupdata(), 0);
}



llvm::Type *
RuntimeOptimizer::llvm_type_closure_component ()
{
    if (m_llvm_type_closure_component)
        return m_llvm_type_closure_component;

    std::vector<llvm::Type*> comp_types;
    comp_types.push_back (llvm_type_int());     // parent.type
    comp_types.push_back (llvm_type_int());     // id
    comp_types.push_back (llvm_type_int());     // size
    comp_types.push_back (llvm_type_int());     // nattrs
    comp_types.push_back (llvm_type_int());     // fake field for char mem[4]

    return m_llvm_type_closure_component = llvm_type_struct (comp_types);
}



llvm::Type *
RuntimeOptimizer::llvm_type_closure_component_ptr ()
{
    return (llvm::Type *) llvm::PointerType::get (llvm_type_closure_component(), 0);
}


llvm::Type *
RuntimeOptimizer::llvm_type_closure_component_attr ()
{
    if (m_llvm_type_closure_component_attr)
        return m_llvm_type_closure_component_attr;

    std::vector<llvm::Type*> attr_types;
    attr_types.push_back (llvm_type_string());  // key

    std::vector<llvm::Type*> union_types;
    union_types.push_back (llvm_type_int());
    union_types.push_back (llvm_type_float());
    union_types.push_back (llvm_type_triple());
    union_types.push_back (llvm_type_void_ptr());

    attr_types.push_back (llvm_type_union (union_types)); // value union

    return m_llvm_type_closure_component_attr = llvm_type_struct (attr_types);
}



llvm::Type *
RuntimeOptimizer::llvm_type_closure_component_attr_ptr ()
{
    return (llvm::Type *) llvm::PointerType::get (llvm_type_closure_component_attr(), 0);
}



void
RuntimeOptimizer::llvm_assign_initial_value (const Symbol& sym)
{
    // Don't write over connections!  Connection values are written into
    // our layer when the earlier layer is run, as part of its code.  So
    // we just don't need to initialize it here at all.
    if (sym.valuesource() == Symbol::ConnectedVal &&
          !sym.typespec().is_closure_based())
        return;
    if (sym.typespec().is_closure_based() && sym.symtype() == SymTypeGlobal)
        return;

    int arraylen = std::max (1, sym.typespec().arraylength());

    // Closures need to get their storage before anything can be
    // assigned to them.  Unless they are params, in which case we took
    // care of it in the group entry point.
    if (sym.typespec().is_closure_based() &&
        sym.symtype() != SymTypeParam && sym.symtype() != SymTypeOutputParam) {
        llvm_assign_zero (sym);
    }

    if (sym.symtype() != SymTypeParam && sym.symtype() != SymTypeOutputParam &&
        sym.symtype() != SymTypeConst && sym.typespec().is_string_based()) {
        // Strings are pointers.  Can't take any chance on leaving local/tmp
        // syms uninitialized.
        DASSERT (sym.symtype() == SymTypeLocal || sym.symtype() == SymTypeTemp);
        llvm_assign_zero (sym);
        return;  // we're done, the parts below are just for params
    }

    if (sym.has_init_ops() && sym.valuesource() == Symbol::DefaultVal) {
        // Handle init ops.
        build_llvm_code (sym.initbegin(), sym.initend());
    } else {
        // Use default value
        int num_components = sym.typespec().simpletype().aggregate;
        for (int a = 0, c = 0; a < arraylen;  ++a) {
            llvm::Value *arrind = sym.typespec().is_array() ? llvm_constant(a) : NULL;
            if (sym.typespec().is_closure_based())
                continue;
            for (int i = 0; i < num_components; ++i, ++c) {
                // Fill in the constant val
                llvm::Value* init_val = 0;
                TypeSpec elemtype = sym.typespec().elementtype();
                if (elemtype.is_floatbased())
                    init_val = llvm_constant (((float*)sym.data())[c]);
                else if (elemtype.is_string())
                    init_val = llvm_constant (((ustring*)sym.data())[c]);
                else if (elemtype.is_int())
                    init_val = llvm_constant (((int*)sym.data())[c]);
                ASSERT (init_val);
                llvm_store_value (init_val, sym, 0, arrind, i);
            }
        }
        if (sym.has_derivs())
            llvm_zero_derivs (sym);
    }

    // Handle interpolated params.
    // FIXME -- really, we shouldn't assign defaults or run init ops if
    // the values are interpolated.  The perf hit is probably small, since
    // there are so few interpolated params, but we should come back and
    // fix this later.
    if ((sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam)
        && ! sym.lockgeom()) {
        std::vector<llvm::Value*> args;
        args.push_back (sg_void_ptr());
        args.push_back (llvm_constant (sym.name()));
        args.push_back (llvm_constant (sym.typespec().simpletype()));
        args.push_back (llvm_constant ((int) sym.has_derivs()));
        args.push_back (llvm_void_ptr (sym));
        llvm_call_function ("osl_bind_interpolated_param",
                            &args[0], args.size());                            
    }
}



llvm::Value *
RuntimeOptimizer::llvm_offset_ptr (llvm::Value *ptr, int offset,
                                   llvm::Type *ptrtype)
{
    llvm::Value *i = builder().CreatePtrToInt (ptr, llvm_type_addrint());
    i = builder().CreateAdd (i, llvm_constant ((size_t)offset));
    ptr = builder().CreateIntToPtr (i, llvm_type_void_ptr());
    if (ptrtype)
        ptr = llvm_ptr_cast (ptr, ptrtype);
    return ptr;
}



void
RuntimeOptimizer::llvm_generate_debugnan (const Opcode &op)
{
    for (int i = 0;  i < op.nargs();  ++i) {
        Symbol &sym (*opargsym (op, i));
        if (! op.argwrite(i))
            continue;
        TypeDesc t = sym.typespec().simpletype();
        if (t.basetype != TypeDesc::FLOAT)
            continue;  // just check float-based types
        int ncomps = t.numelements() * t.aggregate;
        llvm::Value *args[] = { llvm_constant(ncomps),
                                llvm_void_ptr(sym),
                                llvm_constant((int)sym.has_derivs()),
                                sg_void_ptr(), 
                                llvm_constant(op.sourcefile()),
                                llvm_constant(op.sourceline()),
                                llvm_constant(sym.name()) };
        llvm_call_function ("osl_naninf_check", args, 7);
    }
}



bool
RuntimeOptimizer::build_llvm_code (int beginop, int endop, llvm::BasicBlock *bb)
{
    if (bb)
        builder().SetInsertPoint (bb);

    for (int opnum = beginop;  opnum < endop;  ++opnum) {
        const Opcode& op = inst()->ops()[opnum];
        const OpDescriptor *opd = m_shadingsys.op_descriptor (op.opname());
        if (opd && opd->llvmgen) {
            bool ok = (*opd->llvmgen) (*this, opnum);
            if (! ok)
                return false;
            if (m_shadingsys.debug_nan() /* debug NaN/Inf */
                && op.farthest_jump() < 0 /* Jumping ops don't need it */) {
                llvm_generate_debugnan (op);
            }
        } else if (op.opname() == op_nop ||
                   op.opname() == op_end) {
            // Skip this op, it does nothing...
        } else {
            m_shadingsys.error ("LLVMOSL: Unsupported op %s in layer %s\n", op.opname().c_str(), inst()->layername().c_str());
            return false;
        }

        // If the op we coded jumps around, skip past its recursive block
        // executions.
        int next = op.farthest_jump ();
        if (next >= 0)
            opnum = next-1;
    }
    return true;
}



llvm::Function*
RuntimeOptimizer::build_llvm_instance (bool groupentry)
{
    // Make a layer function: void layer_func(ShaderGlobals*, GroupData*)
    // Note that the GroupData* is passed as a void*.
    std::string unique_layer_name = Strutil::format ("%s_%d", inst()->layername().c_str(), inst()->id());

    m_layer_func = llvm::cast<llvm::Function>(m_llvm_module->getOrInsertFunction(unique_layer_name,
                    llvm_type_void(), llvm_type_sg_ptr(),
                    llvm_type_groupdata_ptr(), NULL));
    // Use fastcall for non-entry layer functions to encourage register calling
    if (!groupentry) m_layer_func->setCallingConv(llvm::CallingConv::Fast);
    llvm::Function::arg_iterator arg_it = m_layer_func->arg_begin();
    // Get shader globals pointer
    m_llvm_shaderglobals_ptr = arg_it++;
    m_llvm_groupdata_ptr = arg_it++;

    llvm::BasicBlock *entry_bb = llvm_new_basic_block (unique_layer_name);

    // Set up a new IR builder
    delete m_builder;
    m_builder = new llvm::IRBuilder<> (entry_bb);
    // llvm_gen_debug_printf (std::string("enter layer ")+inst()->shadername());

    if (groupentry) {
        if (m_num_used_layers > 1) {
            // If this is the group entry point, clear all the "layer
            // executed" bits.  If it's not the group entry (but rather is
            // an upstream node), then set its bit!
            int sz = (m_num_used_layers + 3) & (~3);  // round up to 32 bits
            llvm_memset (llvm_void_ptr(layer_run_ptr(0)), 0, sz, 4 /*align*/);
        }
        // Group entries also need to allot space for ALL layers' params
        // that are closures (to avoid weird order of layer eval problems).
        for (int i = 0;  i < group().nlayers();  ++i) {
            ShaderInstance *gi = group()[i];
            if (gi->unused())
                continue;
            FOREACH_PARAM (Symbol &sym, gi) {
               if (sym.typespec().is_closure_based()) {
                    int arraylen = std::max (1, sym.typespec().arraylength());
                    llvm::Value *val = llvm_constant_ptr(NULL, llvm_type_void_ptr());
                    for (int a = 0; a < arraylen;  ++a) {
                        llvm::Value *arrind = sym.typespec().is_array() ? llvm_constant(a) : NULL;
                        llvm_store_value (val, sym, 0, arrind, 0);
                    }
                }
            }
            // Unconditionally execute earlier layers that are not lazy
            if (! gi->run_lazily() && i < group().nlayers()-1)
                llvm_call_layer (i, true /* unconditionally run */);
        }
    }

    // Setup the symbols
    m_named_values.clear ();
    BOOST_FOREACH (Symbol &s, inst()->symbols()) {
        // Skip non-array constants -- we always inline them
        if (s.symtype() == SymTypeConst && !s.typespec().is_array())
            continue;
        // Skip structure placeholders
        if (s.typespec().is_structure())
            continue;
        // Allocate space for locals, temps, aggregate constants
        if (s.symtype() == SymTypeLocal || s.symtype() == SymTypeTemp ||
                s.symtype() == SymTypeConst)
            getOrAllocateLLVMSymbol (s);
        // Set initial value for constants, closures, and strings that are
        // not parameters.
        if (s.symtype() != SymTypeParam && s.symtype() != SymTypeOutputParam &&
            (s.is_constant() || s.typespec().is_closure_based() ||
             s.typespec().is_string_based()))
            llvm_assign_initial_value (s);
        // If debugnan is turned on, globals check that their values are ok
        if (s.symtype() == SymTypeGlobal && m_shadingsys.debug_nan()) {
            TypeDesc t = s.typespec().simpletype();
            if (t.basetype == TypeDesc::FLOAT) { // just check float-based types
                int ncomps = t.numelements() * t.aggregate;
                llvm::Value *args[] = { llvm_constant(ncomps), llvm_void_ptr(s),
                     llvm_constant((int)s.has_derivs()), sg_void_ptr(), 
                     llvm_constant(ustring(inst()->shadername())),
                     llvm_constant(0), llvm_constant(s.name()) };
                llvm_call_function ("osl_naninf_check", args, 7);
            }
        }
    }
    // make a second pass for the parameters (which may make use of
    // locals and constants from the first pass)
    FOREACH_PARAM (Symbol &s, inst()) {
        // Skip structure placeholders
        if (s.typespec().is_structure())
            continue;
        // Skip if it's never read and isn't connected
        if (! s.everread() && ! s.connected_down() && ! s.connected())
            continue;
        // Set initial value for params (may contain init ops)
        llvm_assign_initial_value (s);
    }

    // All the symbols are stack allocated now.

    // Mark all the basic blocks, including allocating llvm::BasicBlock
    // records for each.
    find_basic_blocks (true);

    build_llvm_code (inst()->maincodebegin(), inst()->maincodeend());

    // Transfer all of this layer's outputs into the downstream shader's
    // inputs.
    for (int layer = m_layer+1;  layer < group().nlayers();  ++layer) {
        ShaderInstance *child = m_group[layer];
        for (int c = 0;  c < child->nconnections();  ++c) {
            const Connection &con (child->connection (c));
            if (con.srclayer == m_layer) {
                ASSERT (con.src.arrayindex == -1 && con.src.channel == -1 &&
                        con.dst.arrayindex == -1 && con.dst.channel == -1 &&
                        "no support for individual element/channel connection");
                Symbol *srcsym (inst()->symbol (con.src.param));
                Symbol *dstsym (child->symbol (con.dst.param));
                llvm_run_connected_layers (*srcsym, con.src.param, NULL);
                // FIXME -- I'm not sure I understand this.  Isn't this
                // unnecessary if we wrote to the parameter ourself?
                llvm_assign_impl (*dstsym, *srcsym);
            }
        }
    }
    // llvm_gen_debug_printf ("done copying connections");

    // All done
    // llvm_gen_debug_printf (std::string("exit layer ")+inst()->shadername());
    builder().CreateRetVoid();

    if (shadingsys().llvm_debug())
        llvm::outs() << "layer_func (" << unique_layer_name << ") after llvm  = " << *m_layer_func << "\n";

    delete m_builder;
    m_builder = NULL;

    return m_layer_func;
}



/// OSL_Dummy_JITMemoryManager - Create a shell that passes on requests
/// to a real JITMemoryManager underneath, but can be retained after the
/// dummy is destroyed.  Also, we don't pass along any deallocations.
class OSL_Dummy_JITMemoryManager : public llvm::JITMemoryManager {
protected:
    llvm::JITMemoryManager *mm;
public:
    OSL_Dummy_JITMemoryManager(llvm::JITMemoryManager *realmm) : mm(realmm) { HasGOT = realmm->isManagingGOT(); }
    virtual ~OSL_Dummy_JITMemoryManager() { }
    virtual void setMemoryWritable() { mm->setMemoryWritable(); }
    virtual void setMemoryExecutable() { mm->setMemoryExecutable(); }
    virtual void setPoisonMemory(bool poison) { mm->setPoisonMemory(poison); }
    virtual void AllocateGOT() { ASSERT(HasGOT == false); ASSERT(HasGOT == mm->isManagingGOT()); mm->AllocateGOT(); HasGOT = true; ASSERT(HasGOT == mm->isManagingGOT()); }
    virtual uint8_t *getGOTBase() const { return mm->getGOTBase(); }
    virtual uint8_t *startFunctionBody(const llvm::Function *F,
                                       uintptr_t &ActualSize) {
        return mm->startFunctionBody (F, ActualSize);
    }
    virtual uint8_t *allocateStub(const llvm::GlobalValue* F, unsigned StubSize,
                                  unsigned Alignment) {
        return mm->allocateStub (F, StubSize, Alignment);
    }
    virtual void endFunctionBody(const llvm::Function *F,
                                 uint8_t *FunctionStart, uint8_t *FunctionEnd) {
        mm->endFunctionBody (F, FunctionStart, FunctionEnd);
    }
    virtual uint8_t *allocateSpace(intptr_t Size, unsigned Alignment) {
        return mm->allocateSpace (Size, Alignment);
    }
    virtual uint8_t *allocateGlobal(uintptr_t Size, unsigned Alignment) {
        return mm->allocateGlobal (Size, Alignment);
    }
    virtual void deallocateFunctionBody(void *Body) {
        // DON'T DEALLOCATE mm->deallocateFunctionBody (Body);
    }
    virtual uint8_t* startExceptionTable(const llvm::Function* F,
                                         uintptr_t &ActualSize) {
        return mm->startExceptionTable (F, ActualSize);
    }
    virtual void endExceptionTable(const llvm::Function *F, uint8_t *TableStart,
                                   uint8_t *TableEnd, uint8_t* FrameRegister) {
        mm->endExceptionTable (F, TableStart, TableEnd, FrameRegister);
    }
    virtual void deallocateExceptionTable(void *ET) {
        // DON'T DEALLOCATE mm->deallocateExceptionTable(ET);
    }
    virtual bool CheckInvariants(std::string &s) {
        return mm->CheckInvariants(s);
    }
    virtual size_t GetDefaultCodeSlabSize() {
        return mm->GetDefaultCodeSlabSize();
    }
    virtual size_t GetDefaultDataSlabSize() {
        return mm->GetDefaultDataSlabSize();
    }
    virtual size_t GetDefaultStubSlabSize() {
        return mm->GetDefaultStubSlabSize();
    }
    virtual unsigned GetNumCodeSlabs() { return mm->GetNumCodeSlabs(); }
    virtual unsigned GetNumDataSlabs() { return mm->GetNumDataSlabs(); }
    virtual unsigned GetNumStubSlabs() { return mm->GetNumStubSlabs(); }
};



void
RuntimeOptimizer::build_llvm_group ()
{
    // At this point, we already hold the lock for this group, by virtue
    // of ShadingSystemImpl::optimize_group.
    Timer timer;

    if (! m_thread->llvm_context)
        m_thread->llvm_context = new llvm::LLVMContext();

    if (! m_thread->llvm_jitmm) {
        m_thread->llvm_jitmm = llvm::JITMemoryManager::CreateDefaultMemManager();
        spin_lock lock (m_shadingsys.m_llvm_mutex);  // lock m_llvm_jitmm_hold
        m_shadingsys.m_llvm_jitmm_hold.push_back (shared_ptr<llvm::JITMemoryManager>(m_thread->llvm_jitmm));
    }

    ASSERT (! m_llvm_module);
    // Load the LLVM bitcode and parse it into a Module
    const char *data = osl_llvm_compiled_ops_block;
    llvm::MemoryBuffer* buf = llvm::MemoryBuffer::getMemBuffer (llvm::StringRef(data, osl_llvm_compiled_ops_size));
    std::string err;
#ifdef OSL_LLVM_NO_BITCODE
    m_llvm_module = new llvm::Module("llvm_ops", *llvm_context());
#else
    // Load the LLVM bitcode and parse it into a Module
    m_llvm_module = llvm::ParseBitcodeFile (buf, *m_thread->llvm_context, &err);
    if (err.length())
        m_shadingsys.error ("ParseBitcodeFile returned '%s'\n", err.c_str());
    delete buf;
#endif

    // Create the ExecutionEngine
    ASSERT (! m_llvm_exec);
    err.clear ();
    llvm::JITMemoryManager *mm = new OSL_Dummy_JITMemoryManager(m_thread->llvm_jitmm);
    m_llvm_exec = llvm::ExecutionEngine::createJIT (m_llvm_module, &err, mm, llvm::CodeGenOpt::Default, /*AllocateGVsWithCode*/ false);
    if (! m_llvm_exec) {
        m_shadingsys.error ("Failed to create engine: %s\n", err.c_str());
        ASSERT (0);
        return;
    }
    // Force it to JIT as soon as we ask it for the code pointer,
    // don't take any chances that it might JIT lazily, since we
    // will be stealing the JIT code memory from under its nose and
    // destroying the Module & ExecutionEngine.
    m_llvm_exec->DisableLazyCompilation ();

    m_stat_llvm_setup_time += timer.lap();

    // Set up m_num_used_layers to be the number of layers that are
    // actually used, and m_layer_remap[] to map original layer numbers
    // to the shorter list of actually-called layers.
    int nlayers = m_group.nlayers();
    m_layer_remap.resize (nlayers);
    m_num_used_layers = 0;
    for (int layer = 0;  layer < m_group.nlayers();  ++layer) {
        bool lastlayer = (layer == (nlayers-1));
        if (! m_group[layer]->unused() || lastlayer)
            m_layer_remap[layer] = m_num_used_layers++;
        else
            m_layer_remap[layer] = -1;
    }
    m_shadingsys.m_stat_empty_instances += m_group.nlayers()-m_num_used_layers;

    initialize_llvm_group ();

    // Generate the LLVM IR for each layer
    //
    llvm::Function** funcs = (llvm::Function**)alloca(m_num_used_layers * sizeof(llvm::Function*));
    for (int layer = 0; layer < nlayers; ++layer) {
        set_inst (layer);
        bool lastlayer = (layer == (nlayers-1));
        int index = m_layer_remap[layer];
        if (index != -1) funcs[index] = build_llvm_instance (lastlayer);
    }
    llvm::Function* entry_func = funcs[m_num_used_layers-1];
    m_stat_llvm_irgen_time += timer.lap();

    // Optimize the LLVM IR unless it's just a ret void group (1 layer, 1 BB, 1 inst == retvoid)
    bool skip_optimization = m_num_used_layers == 1 && entry_func->size() == 1 && entry_func->front().size() == 1;
    // Label the group as being retvoid or not.
    m_group.does_nothing(skip_optimization);
    if (skip_optimization) {
        m_shadingsys.m_stat_empty_groups += 1;
    } else {
#if 0
      // First do the simple function passes
      m_llvm_func_passes->doInitialization();
      for (llvm::Module::iterator i = llvm_module()->begin();
           i != llvm_module()->end(); ++i) {
        m_llvm_func_passes->run (*i);
      }
      m_llvm_func_passes->doFinalization();
#endif

      // Next do the module passes
      m_llvm_passes->run (*llvm_module());

#if 0
      // Now do additional highly optimized function passes on just the
      // new functions we added for the shader layers
      m_llvm_func_passes_optimized->doInitialization();
      for (int i = 0; i < m_num_used_layers; ++i) {
          m_llvm_func_passes_optimized->run (funcs[i]);
      }
      m_llvm_func_passes_optimized->doFinalization();
#endif
    }

    if (shadingsys().llvm_debug())
        llvm::outs() << "func after opt  = " << *entry_func << "\n";

    // Debug code to dump the resulting bitcode to a file
    if (shadingsys().llvm_debug() >= 2) {
        std::string err_info;
        std::string name = Strutil::format ("%s_%d.bc",
                                            inst()->layername().c_str(),
                                            inst()->id());
        llvm::raw_fd_ostream out (name.c_str(), err_info);
        llvm::WriteBitcodeToFile (llvm_module(), out);
    }

    m_stat_llvm_opt_time += timer.lap();

    // Force the JIT to happen now
    {
#if OSL_LLVM_VERSION <= 29
        // Lock this! -- there seems to be at least one bug in LLVM 2.9
        // where the JIT isn't really thread-safe.
        // Doesn't seem to be necessary for LLVM 3.0.
        static mutex jit_mutex;
        lock_guard lock (jit_mutex);
#endif
        RunLLVMGroupFunc f = (RunLLVMGroupFunc) m_llvm_exec->getPointerToFunction(entry_func);
        m_group.llvm_compiled_version (f);
    }

    // Remove the IR for the group layer functions, we've already JITed it
    // and will never need the IR again.  This saves memory, and also saves
    // a huge amount of time since we won't re-optimize it again and again
    // if we keep adding new shader groups to the same Module.
    for (int i = 0; i < m_num_used_layers; ++i) {
        funcs[i]->deleteBody();
    }

    // Free the exec and module to reclaim all the memory.  This definitely
    // saves memory, and has almost no effect on runtime.
    delete m_llvm_exec;
    m_llvm_exec = NULL;

    // N.B. Destroying the EE should have destroyed the module as well.
    m_llvm_module = NULL;

    m_stat_llvm_jit_time += timer.lap();
}



void
RuntimeOptimizer::initialize_llvm_group ()
{
    // I don't think we actually need to lock here (lg)
    // static spin_mutex mutex;
    // spin_lock lock (mutex);

    m_llvm_context = m_thread->llvm_context;
    ASSERT (m_llvm_context && m_llvm_module);

    llvm_setup_optimization_passes ();

    // Clear the shaderglobals and groupdata types -- they will be
    // created on demand.
    m_llvm_type_sg = NULL;
    m_llvm_type_groupdata = NULL;
    m_llvm_type_closure_component = NULL;
    m_llvm_type_closure_component_attr = NULL;

    // Set up aliases for types we use over and over
    m_llvm_type_float = (llvm::Type *) llvm::Type::getFloatTy (*m_llvm_context);
    m_llvm_type_int = (llvm::Type *) llvm::Type::getInt32Ty (*m_llvm_context);
    if (sizeof(char *) == 4)
        m_llvm_type_addrint = (llvm::Type *) llvm::Type::getInt32Ty (*m_llvm_context);
    else
        m_llvm_type_addrint = (llvm::Type *) llvm::Type::getInt64Ty (*m_llvm_context);
    m_llvm_type_int_ptr = (llvm::PointerType *) llvm::Type::getInt32PtrTy (*m_llvm_context);
    m_llvm_type_bool = (llvm::Type *) llvm::Type::getInt1Ty (*m_llvm_context);
    m_llvm_type_longlong = (llvm::Type *) llvm::Type::getInt64Ty (*m_llvm_context);
    m_llvm_type_void = (llvm::Type *) llvm::Type::getVoidTy (*m_llvm_context);
    m_llvm_type_char_ptr = (llvm::PointerType *) llvm::Type::getInt8PtrTy (*m_llvm_context);
    m_llvm_type_float_ptr = (llvm::PointerType *) llvm::Type::getFloatPtrTy (*m_llvm_context);
    m_llvm_type_ustring_ptr = (llvm::PointerType *) llvm::PointerType::get (m_llvm_type_char_ptr, 0);

    // A triple is a struct composed of 3 floats
    std::vector<llvm::Type*> triplefields(3, m_llvm_type_float);
    m_llvm_type_triple = llvm_type_struct (triplefields);
    m_llvm_type_triple_ptr = (llvm::PointerType *) llvm::PointerType::get (m_llvm_type_triple, 0);

    // A matrix is a struct composed 16 floats
    std::vector<llvm::Type*> matrixfields(16, m_llvm_type_float);
    m_llvm_type_matrix = llvm_type_struct (matrixfields);
    m_llvm_type_matrix_ptr = (llvm::PointerType *) llvm::PointerType::get (m_llvm_type_matrix, 0);

    for (int i = 0;  llvm_helper_function_table[i];  i += 2) {
        const char *funcname = llvm_helper_function_table[i];
        bool varargs = false;
        const char *types = llvm_helper_function_table[i+1];
        int advance;
        TypeSpec rettype = OSLCompilerImpl::type_from_code (types, &advance);
        types += advance;
        std::vector<llvm::Type*> params;
        while (*types) {
            TypeSpec t = OSLCompilerImpl::type_from_code (types, &advance);
            if (t.simpletype().basetype == TypeDesc::UNKNOWN) {
                if (*types == '*')
                    varargs = true;
                else
                    ASSERT (0);
            } else {
                params.push_back (llvm_pass_type (t));
            }
            types += advance;
        }
#if OSL_LLVM_VERSION <= 29
        char *pp = (char *)&params;
        llvm::FunctionType *func = llvm::FunctionType::get (llvm_type(rettype), *(std::vector<const llvm::Type*>*)pp, varargs);
#else
        llvm::FunctionType *func = llvm::FunctionType::get (llvm_type(rettype), params, varargs);
#endif
        m_llvm_module->getOrInsertFunction (funcname, func);
    }

    // Needed for closure setup
    std::vector<llvm::Type*> params(3);
    params[0] = m_llvm_type_char_ptr;
    params[1] = m_llvm_type_int;
    params[2] = m_llvm_type_char_ptr;
#if OSL_LLVM_VERSION <= 29
    char *pp = (char *)&params;
    m_llvm_type_prepare_closure_func = (llvm::PointerType *)llvm::PointerType::getUnqual (llvm::FunctionType::get (m_llvm_type_void, *(std::vector<const llvm::Type*>*)pp, false));
#else
    m_llvm_type_prepare_closure_func = llvm::PointerType::getUnqual (llvm::FunctionType::get (m_llvm_type_void, params, false));
#endif
    m_llvm_type_setup_closure_func = m_llvm_type_prepare_closure_func;
}



void
ShadingSystemImpl::SetupLLVM ()
{
    static mutex setup_mutex;
    static bool done = false;
    lock_guard lock (setup_mutex);
    if (done)
        return;
    // Some global LLVM initialization for the first thread that
    // gets here.
    info ("Setting up LLVM");
    llvm::DisablePrettyStackTrace = true;
    llvm::llvm_start_multithreaded ();  // enable it to be thread-safe
    llvm::InitializeNativeTarget();
    done = true;
}



void
RuntimeOptimizer::llvm_setup_optimization_passes ()
{
    ASSERT (m_llvm_passes == NULL && m_llvm_func_passes == NULL);

    // Specify per-function passes
    //
    m_llvm_func_passes = new llvm::FunctionPassManager(llvm_module());
    llvm::FunctionPassManager &fpm (*m_llvm_func_passes);
    fpm.add (new llvm::TargetData(llvm_module()));

    // Specify module-wide (interprocedural optimization) passes
    //
    m_llvm_passes = new llvm::PassManager;
    llvm::PassManager &passes (*m_llvm_passes);
    passes.add (new llvm::TargetData(llvm_module()));

    // More highly optimized function passes we use just for the group
    // entry point.
    m_llvm_func_passes_optimized = new llvm::FunctionPassManager(llvm_module());
    llvm::FunctionPassManager &fpmo (*m_llvm_func_passes_optimized);
    fpmo.add (new llvm::TargetData(llvm_module()));


#if 1
    // Specify everything as a module pass

    // Always add verifier?
    passes.add (llvm::createVerifierPass());
    // Simplify the call graph if possible (deleting unreachable blocks, etc.)
    passes.add (llvm::createCFGSimplificationPass());
    // Change memory references to registers
//    passes.add (llvm::createPromoteMemoryToRegisterPass());
    passes.add (llvm::createScalarReplAggregatesPass());
    // Combine instructions where possible -- peephole opts & bit-twiddling
    passes.add (llvm::createInstructionCombiningPass());
    // Inline small functions
    passes.add (llvm::createFunctionInliningPass());  // 250?
    // Eliminate early returns
    passes.add (llvm::createUnifyFunctionExitNodesPass());
    // resassociate exprssions (a = x + (3 + y) -> a = x + y + 3)
    passes.add (llvm::createReassociatePass());
    // Eliminate common sub-expressions
    passes.add (llvm::createGVNPass());
    passes.add (llvm::createSCCPPass());          // Constant prop with SCCP
    // More dead code elimination
    passes.add (llvm::createAggressiveDCEPass());
    // Combine instructions where possible -- peephole opts & bit-twiddling
    passes.add (llvm::createInstructionCombiningPass());
    // Simplify the call graph if possible (deleting unreachable blocks, etc.)
    passes.add (llvm::createCFGSimplificationPass());
    // Try to make stuff into registers one last time.
    passes.add (llvm::createPromoteMemoryToRegisterPass());

#elif 0
    // This code would apply the standard optimizations used by
    // llvm-gcc.  We have found that for our purposes, they spend too
    // much time optimizing.  But useful to have for reference.  See
    // llvm's include/llvm/Support/StandardPasses.h to see what they do.
    int optlevel = 3;
    llvm::createStandardFunctionPasses (&fpm, optlevel /*opt level*/);
    passes.add (llvm::createFunctionInliningPass(250));
    llvm::Pass *inlining_pass = llvm::createFunctionInliningPass (250 /*threshold*/);
    llvm::createStandardModulePasses (&passes, optlevel /*opt level*/,
                                      false /* optimize size */,
                                      true /* unit at a time */,
                                      true /* unroll loops */,
                                      true /* simplify lib calls */,
                                      false /* have exceptions */,
                                      inlining_pass);
#else
    // These are our custom set of optimizations.


    // Simplify the call graph if possible (deleting unreachable blocks, etc.)
    fpm.add (llvm::createCFGSimplificationPass());

    // Change memory references to registers
    fpm.add (llvm::createPromoteMemoryToRegisterPass());
    fpm.add (llvm::createScalarReplAggregatesPass());
    // Combine instructions where possible -- peephole opts & bit-twiddling
    fpm.add (llvm::createInstructionCombiningPass());
    // Eliminate early returns
    fpm.add (llvm::createUnifyFunctionExitNodesPass());

    // resassociate exprssions (a = x + (3 + y) -> a = x + y + 3)
//    fpm.add (llvm::createReassociatePass());
    // Eliminate common sub-expressions
//    fpm.add (llvm::createGVNPass());
    // Simplify the call graph if possible (deleting unreachable blocks, etc.)
//    fpm.add (llvm::createCFGSimplificationPass());
    // More dead code elimination
//    fpm.add (llvm::createAggressiveDCEPass());
    // Try to make stuff into registers one last time.
//    fpm.add (llvm::createPromoteMemoryToRegisterPass());
    // Always add verifier?
//    fpm.add (llvm::createVerifierPass());


    // passes.add (llvm::createGlobalOptimizerPass()); // Optimize out global vars
    //  ? createIPSCCPPass()
    //  ? createDeadArgEliminationPass
    // Combine instructions where possible -- peephole opts & bit-twiddling
    passes.add (llvm::createInstructionCombiningPass());
    // Simplify the call graph if possible (deleting unreachable blocks, etc.)
    passes.add (llvm::createCFGSimplificationPass());

    // Inline small functions
    passes.add (llvm::createFunctionInliningPass());  // 250?
//    passes.add (llvm::createFunctionInliningPass());  // 250?

//    passes.add (llvm::createFunctionAttrsPass());       // Set readonly/readnone attrs
//    passes.add (llvm::createArgumentPromotionPass());   // Scalarize uninlined fn args
#if 0
//    passes.add (llvm::createScalarReplAggregatesPass());  // Break up aggregate allocas
    // if (SimplifyLibCalls)
    //  passes.add (createSimplifyLibCallsPass());    // Library Call Optimizations
    passes.add (llvm::createInstructionCombiningPass());  // Cleanup for scalarrepl.
    passes.add (llvm::createJumpThreadingPass());         // Thread jumps.
//    passes.add (llvm::createCFGSimplificationPass());  // Merge & remove BBs
//    passes.add (llvm::createInstructionCombiningPass());  // Combine silly seq's
    
    //passes.add (createTailCallEliminationPass());   // Eliminate tail calls
    //passes.add (createCFGSimplificationPass());     // Merge & remove BBs
    passes.add (llvm::createReassociatePass());   // Reassociate expressions
    //passes.add (createLoopRotatePass());            // Rotate Loop
    //passes.add (createLICMPass());                  // Hoist loop invariants
    //passes.add (createLoopUnswitchPass(OptimizeSize || OptimizationLevel < 3));
//    passes.add (llvm::createInstructionCombiningPass());  
    //passes.add (createIndVarSimplifyPass());        // Canonicalize indvars
    //passes.add (createLoopDeletionPass());          // Delete dead loops
    //if (UnrollLoops)
#endif
////    passes.add (llvm::createLoopUnrollPass());          // Unroll small loops
#if 0
//    passes.add (llvm::createInstructionCombiningPass());  // Clean up after the unroller
    //if (OptimizationLevel > 1)
    passes.add (llvm::createGVNPass());           // Remove redundancies
    //passes.add (createMemCpyOptPass());             // Remove memcpy / form memset
    passes.add (llvm::createSCCPPass());          // Constant prop with SCCP
      // Run instcombine after redundancy elimination to exploit opportunities
    // opened up by them.
    passes.add (llvm::createInstructionCombiningPass());
    //passes.add (createJumpThreadingPass());         // Thread jumps
    //passes.add (createDeadStoreEliminationPass());  // Delete dead stores
    passes.add (llvm::createAggressiveDCEPass()); // Delete dead instructions
    passes.add (llvm::createCFGSimplificationPass());     // Merge & remove BBs
#endif

    //if (UnitAtATime) {
    //  passes.add (createStripDeadPrototypesPass()); // Get rid of dead prototypes
    //  passes.add (createDeadTypeEliminationPass()); // Eliminate dead types
      // GlobalOpt already deletes dead functions and globals, at -O3 try a
      // late pass of GlobalDCE.  It is capable of deleting dead cycles.
    //  if (OptimizationLevel > 2)
    //    passes.add (createGlobalDCEPass());         // Remove dead fns and globals.    
    //  if (OptimizationLevel > 1)
    //    passes.add (createConstantMergePass());       // Merge dup global constants
    //}

    passes.add (llvm::createVerifierPass());

    fpmo.add (llvm::createScalarReplAggregatesPass());
    fpmo.add (llvm::createInstructionCombiningPass());
    fpmo.add (llvm::createUnifyFunctionExitNodesPass());
    fpmo.add (llvm::createCFGSimplificationPass());
//    fpmo.add (llvm::createFunctionAttrsPass());       // Set readonly/readnone attrs
//    fpmo.add (llvm::createArgumentPromotionPass());   // Scalarize uninlined fn args
    fpmo.add (llvm::createInstructionCombiningPass());  // Cleanup for scalarrepl.
    fpmo.add (llvm::createJumpThreadingPass());         // Thread jumps.
    fpmo.add (llvm::createReassociatePass());   // Reassociate expressions
//    fpmo.add (llvm::createLoopUnrollPass());          // Unroll small loops
    fpmo.add (llvm::createGVNPass());           // Remove redundancies
    fpmo.add (llvm::createSCCPPass());          // Constant prop with SCCP
    fpmo.add (llvm::createInstructionCombiningPass());
    fpmo.add (llvm::createAggressiveDCEPass()); // Delete dead instructions
    fpmo.add (llvm::createCFGSimplificationPass());     // Merge & remove BBs
#endif
}



void
RuntimeOptimizer::llvm_do_optimization (llvm::Function *func,
                                        bool interproc)
{
    ASSERT (m_llvm_passes != NULL && m_llvm_func_passes != NULL);

#if 1
    m_llvm_func_passes->doInitialization();
    m_llvm_func_passes->run (*func);
    m_llvm_func_passes->doFinalization();
#else
    for (llvm::Module::iterator i = llvm_module()->begin();
         i != llvm_module()->end(); ++i) {
        m_llvm_func_passes->doInitialization();
        m_llvm_func_passes->run (*i);
        m_llvm_func_passes->doFinalization();
    }
#endif

    if (interproc) {
        // Run module-wide (interprocedural optimization) passes
        m_llvm_passes->run (*llvm_module());

        // Since the passes above inlined function calls, among other
        // things, we should rerun our whole optimization set on the master
        // function now.
#if 0
        ASSERT (func);
        m_llvm_func_passes_optimized->doInitialization ();
        m_llvm_func_passes_optimized->run (*func);
        m_llvm_func_passes_optimized->doFinalization ();
#endif
    }
}



}; // namespace pvt
}; // namespace osl

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
