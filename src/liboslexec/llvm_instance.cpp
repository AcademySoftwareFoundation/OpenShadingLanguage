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
#include <llvm/Analysis/Verifier.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/Utils/UnifyFunctionExitNodes.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/ExecutionEngine/JITMemoryManager.h>

#include "oslexec_pvt.h"
#include "oslops.h"
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

static ustring op_abs("abs");
static ustring op_and("and");
static ustring op_bitand("bitand");
static ustring op_bitor("bitor");
static ustring op_ceil("ceil");
static ustring op_cellnoise("cellnoise");
static ustring op_color("color");
static ustring op_compl("compl");
static ustring op_dowhile("dowhile");
static ustring op_end("end");
static ustring op_eq("eq");
static ustring op_error("error");
static ustring op_fabs("fabs");
static ustring op_floor("floor");
static ustring op_for("for");
static ustring op_format("format");
static ustring op_ge("ge");
static ustring op_gt("gt");
static ustring op_if("if");
static ustring op_le("le");
static ustring op_lt("lt");
static ustring op_min("min");
static ustring op_neq("neq");
static ustring op_nop("nop");
static ustring op_normal("normal");
static ustring op_or("or");
static ustring op_point("point");
static ustring op_printf("printf");
static ustring op_round("round");
static ustring op_shl("shl");
static ustring op_shr("shr");
static ustring op_step("step");
static ustring op_trunc("trunc");
static ustring op_vector("vector");
static ustring op_warning("warning");
static ustring op_xor("xor");



/// Macro that defines the arguments to LLVM IR generating routines
///
#define LLVMGEN_ARGS     RuntimeOptimizer &rop, int opnum

/// Macro that defines the full declaration of an LLVM generator.
/// 
#define LLVMGEN(name)  bool name (LLVMGEN_ARGS)

/// Function pointer to an LLVM IR-generating routine
///
typedef bool (*OpLLVMGen) (LLVMGEN_ARGS);


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
    "osl_" #name "_dfdvdv",   "xXvv",           \
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
    PNOISE_IMPL(pnoise),
    PNOISE_DERIV_IMPL(pnoise),
    PNOISE_IMPL(psnoise),
    PNOISE_DERIV_IMPL(psnoise),
#endif
    "osl_spline_fff", "xXXXXi",
    "osl_spline_dfdfdf", "xXXXXi",
    "osl_spline_dfdff", "xXXXXi",
    "osl_spline_dffdf", "xXXXXi",
    "osl_spline_vfv", "xXXXXi",
    "osl_spline_dvdfdv", "xXXXXi",
    "osl_spline_dvdfv", "xXXXXi",
    "osl_spline_dvfdv", "xXXXXi",
    "osl_setmessage", "xXsLX",
    "osl_getmessage", "iXssLX",
    "osl_pointcloud", "iXsvfiXi*",

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
    "osl_prepend_matrix_from", "xXXs",
    "osl_get_from_to_matrix", "xXXss",
    "osl_transpose_mm", "xXX",
    "osl_determinant_fm", "fX",

    "osl_prepend_point_from", "xXXs",
    "osl_prepend_vector_from", "xXXs",
    "osl_prepend_normal_from", "xXXs",

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
    "osl_area_fv", "fX",
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



const llvm::Type *
RuntimeOptimizer::llvm_type_union(const std::vector<const llvm::Type *> &types)
{
    llvm::TargetData target(llvm_module());
    size_t max_size = 0;
    size_t max_align = 1;
    for (size_t i = 0; i < types.size(); ++i) {
        size_t size = target.getTypeStoreSize(types[i]);
        size_t align = target.getABITypeAlignment(types[i]);
        max_size  = size  > max_size  ? size  : max_size;
        max_align = align > max_align ? align : max_align;
    }
    size_t padding = (max_size % max_align) ? max_align - (max_size % max_align) : 0;
    size_t union_size = max_size + padding;

    const llvm::Type * base_type = NULL;
    // to ensure the alignment when included in a struct use
    // an appropiate type for the array
    if (max_align == sizeof(void*))
        base_type = llvm_type_void_ptr();
    else if (max_align == 4)
        base_type = llvm::Type::getInt32Ty (llvm_context());
    else if (max_align == 2)
        base_type = llvm::Type::getInt16Ty (llvm_context());
    else
        base_type = llvm::Type::getInt8Ty (llvm_context());

    size_t array_len = union_size / target.getTypeStoreSize(base_type);
    return llvm::ArrayType::get (base_type, array_len);
}



const llvm::Type *
RuntimeOptimizer::llvm_type_sg ()
{
    // Create a type that defines the ShaderGlobals for LLVM IR.  This
    // absolutely MUST exactly match the ShaderGlobals struct in oslexec.h.
    if (m_llvm_type_sg)
        return m_llvm_type_sg;

    // Derivs look like arrays of 3 values
    const llvm::Type *float_deriv = llvm_type (TypeDesc(TypeDesc::FLOAT, TypeDesc::SCALAR, 3));
    const llvm::Type *triple_deriv = llvm_type (TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, 3));
    std::vector<const llvm::Type*> sg_types;
    sg_types.push_back (triple_deriv);        // P, dPdx, dPdy
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

    sg_types.push_back(llvm_type_void_ptr()); // opaque render state*
    sg_types.push_back(llvm_type_void_ptr()); // opaque trace data*
    sg_types.push_back(llvm_type_void_ptr()); // ShadingContext*
    sg_types.push_back(llvm_type_void_ptr()); // object2common
    sg_types.push_back(llvm_type_void_ptr()); // shader2common
    sg_types.push_back(llvm_type_void_ptr()); // Ci

    sg_types.push_back (llvm_type_float());   // surfacearea
    sg_types.push_back (llvm_type_int());     // raytype
    sg_types.push_back (llvm_type_int());     // flipHandedness
    sg_types.push_back (llvm_type_int());     // backfacing

    return m_llvm_type_sg = llvm::StructType::get (llvm_context(), sg_types);
}



const llvm::Type *
RuntimeOptimizer::llvm_type_sg_ptr ()
{
    return llvm::PointerType::get (llvm_type_sg(), 0);
}



const llvm::Type *
RuntimeOptimizer::llvm_type_groupdata ()
{
    // If already computed, return it
    if (m_llvm_type_groupdata)
        return m_llvm_type_groupdata;

    std::vector<const llvm::Type*> fields;

    // First, add the array that tells if each layer has run.  But only make
    // slots for the layers that may be called/used.
    int sz = (m_num_used_layers + 3) & (~3);  // Round up to 32 bit boundary
    fields.push_back (llvm::ArrayType::get(llvm_type_bool(), sz));
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
            size_t align = sym.typespec().is_closure() ? sizeof(void*) :
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

    m_llvm_type_groupdata = llvm::StructType::get (llvm_context(), fields);

#ifdef DEBUG
//    llvm::outs() << "\nGroup struct = " << *m_llvm_type_groupdata << "\n";
//    llvm::outs() << "  size = " << offset << "\n";
#endif

    return m_llvm_type_groupdata;
}



const llvm::Type *
RuntimeOptimizer::llvm_type_groupdata_ptr ()
{
    return llvm::PointerType::get (llvm_type_groupdata(), 0);
}



const llvm::Type *
RuntimeOptimizer::llvm_type_closure_component ()
{
    if (m_llvm_type_closure_component)
        return m_llvm_type_closure_component;

    std::vector<const llvm::Type*> comp_types;
    comp_types.push_back (llvm_type_int());     // parent.type
    comp_types.push_back (llvm_type_int());     // id
    comp_types.push_back (llvm_type_int());     // size
    comp_types.push_back (llvm_type_int());     // nattrs
    comp_types.push_back (llvm_type_int());     // fake field for char mem[4]

    return m_llvm_type_closure_component = llvm::StructType::get (llvm_context(), comp_types);
}



const llvm::Type *
RuntimeOptimizer::llvm_type_closure_component_ptr ()
{
    return llvm::PointerType::get (llvm_type_closure_component(), 0);
}


const llvm::Type *
RuntimeOptimizer::llvm_type_closure_component_attr ()
{
    if (m_llvm_type_closure_component_attr)
        return m_llvm_type_closure_component_attr;

    std::vector<const llvm::Type*> attr_types;
    attr_types.push_back (llvm_type_string());  // key

    std::vector<const llvm::Type*> union_types;
    union_types.push_back (llvm_type_int());
    union_types.push_back (llvm_type_float());
    union_types.push_back (llvm_type_triple());
    union_types.push_back (llvm_type_void_ptr());

    attr_types.push_back (llvm_type_union (union_types)); // value union

    return m_llvm_type_closure_component_attr = llvm::StructType::get (llvm_context(), attr_types);
}



const llvm::Type *
RuntimeOptimizer::llvm_type_closure_component_attr_ptr ()
{
    return llvm::PointerType::get (llvm_type_closure_component_attr(), 0);
}



/// Convert the name of a global (and its derivative index) into the
/// field number of the ShaderGlobals struct.
static int
ShaderGlobalNameToIndex (ustring name)
{
    // N.B. The order of names in this table MUST exactly match the
    // ShaderGlobals struct in oslexec.h, as well as the llvm 'sg' type
    // defined in llvm_type_sg().
    static ustring fields[] = {
        Strings::P, Strings::I, Strings::N, Strings::Ng,
        Strings::u, Strings::v, Strings::dPdu, Strings::dPdv,
        Strings::time, Strings::dtime, Strings::dPdtime, Strings::Ps,
        ustring("renderstate"), ustring("tracedata"),
        ustring("shadingcontext"),
        ustring("object2common"), ustring("shader2common"),
        Strings::Ci,
        ustring("surfacearea"), ustring("raytype"),
        ustring("flipHandedness"), ustring("backfacing")
    };

    for (int i = 0;  i < int(sizeof(fields)/sizeof(fields[0]));  ++i)
        if (name == fields[i])
            return i;
    return -1;
}



llvm::Value *
RuntimeOptimizer::llvm_constant (float f)
{
    return llvm::ConstantFP::get (llvm_context(), llvm::APFloat(f));
}



llvm::Value *
RuntimeOptimizer::llvm_constant (int i)
{
    return llvm::ConstantInt::get (llvm_context(), llvm::APInt(32,i));
}



llvm::Value *
RuntimeOptimizer::llvm_constant (size_t i)
{
    int bits = sizeof(size_t)*8;
    return llvm::ConstantInt::get (llvm_context(), llvm::APInt(bits,i));
}



llvm::Value *
RuntimeOptimizer::llvm_constant_bool (bool i)
{
    return llvm::ConstantInt::get (llvm_context(), llvm::APInt(1,i));
}



llvm::Value *
RuntimeOptimizer::llvm_constant (ustring s)
{
    // Create a const size_t with the ustring contents
    size_t bits = sizeof(size_t)*8;
    llvm::Value *str = llvm::ConstantInt::get (llvm_context(),
                               llvm::APInt(bits,size_t(s.c_str()), true));
    // Then cast the int to a char*.
    return builder().CreateIntToPtr (str, llvm_type_string(), "ustring constant");
}



llvm::Value *
RuntimeOptimizer::llvm_constant_ptr (void *p)
{
    // Create a const size_t with the address
    size_t bits = sizeof(size_t)*8;
    llvm::Value *str = llvm::ConstantInt::get (llvm_context(),
                               llvm::APInt(bits,size_t(p), true));
    // Then cast the size_t to a char*.
    return builder().CreateIntToPtr (str, llvm_type_void_ptr());
}



llvm::Value *
RuntimeOptimizer::llvm_constant (const TypeDesc &type)
{
    long long *i = (long long *)&type;
    return llvm::ConstantInt::get (llvm_context(), llvm::APInt(64,*i));
}



const llvm::Type *
RuntimeOptimizer::llvm_type (const TypeSpec &typespec)
{
    if (typespec.is_closure())
        return llvm_type_void_ptr();
    TypeDesc t = typespec.simpletype().elementtype();
    const llvm::Type *lt = NULL;
    if (t == TypeDesc::FLOAT)
        lt = llvm_type_float();
    else if (t == TypeDesc::INT)
        lt = llvm_type_int();
    else if (t == TypeDesc::STRING)
        lt = llvm_type_string();
    else if (t.aggregate == TypeDesc::VEC3)
        lt = llvm_type_triple();
    else if (t.aggregate == TypeDesc::MATRIX44)
        lt = llvm_type_matrix();
    else if (t == TypeDesc::NONE)
        lt = llvm_type_void();
    else if (t == TypeDesc::PTR)
        lt = llvm_type_void_ptr();
    else {
        std::cerr << "Bad llvm_type(" << typespec.c_str() << ")\n";
        ASSERT (0 && "not handling this type yet");
    }
    if (typespec.is_array())
        lt = llvm::ArrayType::get (lt, typespec.simpletype().numelements());
    return lt;
}



const llvm::Type *
RuntimeOptimizer::llvm_pass_type (const TypeSpec &typespec)
{
    if (typespec.is_closure())
        return llvm_type_void_ptr();
    TypeDesc t = typespec.simpletype().elementtype();
    const llvm::Type *lt = NULL;
    if (t == TypeDesc::FLOAT)
        lt = llvm_type_float();
    else if (t == TypeDesc::INT)
        lt = llvm_type_int();
    else if (t == TypeDesc::STRING)
        lt = llvm_type_string();
    else if (t.aggregate == TypeDesc::VEC3)
        lt = llvm_type_void_ptr(); //llvm_type_triple_ptr();
    else if (t.aggregate == TypeDesc::MATRIX44)
        lt = llvm_type_void_ptr(); //llvm_type_matrix_ptr();
    else if (t == TypeDesc::NONE)
        lt = llvm_type_void();
    else if (t == TypeDesc::PTR)
        lt = llvm_type_void_ptr();
    else if (t == TypeDesc::LONGLONG)
        lt = llvm_type_longlong();
    else {
        std::cerr << "Bad llvm_pass_type(" << typespec.c_str() << ")\n";
        ASSERT (0 && "not handling this type yet");
    }
    if (t.arraylen) {
        ASSERT (0 && "should never pass an array directly as a parameter");
    }
    return lt;
}



void
RuntimeOptimizer::llvm_assign_zero (const Symbol &sym)
{
    // Just memset the whole thing to zero, let LLVM sort it out.
    // This even works for closures.
    int len = sym.typespec().is_closure() ? sizeof(void *) : sym.derivsize();
    // N.B. derivsize() includes derivs, if there are any
    size_t align = sym.typespec().is_closure() ? sizeof(void*) :
                         sym.typespec().simpletype().basesize();
    llvm_memset (llvm_void_ptr(sym), 0, len, (int)align);
}



void
RuntimeOptimizer::llvm_zero_derivs (const Symbol &sym)
{
    // Just memset the derivs to zero, let LLVM sort it out.
    TypeSpec elemtype = sym.typespec().elementtype();
    if (sym.has_derivs() && elemtype.is_floatbased()) {
        int len = sym.typespec().is_closure() ? sizeof(void *) : sym.size();
        size_t align = sym.typespec().is_closure() ? sizeof(void*) :
                             sym.typespec().simpletype().basesize();
        llvm_memset (llvm_void_ptr(sym,1), /* point to start of x deriv */
                     0, 2*len /* size of both derivs */, (int)align);
    }
}



/// Given the OSL symbol, return the llvm::Value* corresponding to the
/// start of that symbol (first element, first component, and just the
/// plain value if it has derivatives).
llvm::Value *
RuntimeOptimizer::getLLVMSymbolBase (const Symbol &sym)
{
    Symbol* dealiased = sym.dealias();

    if (sym.symtype() == SymTypeGlobal) {
        // Special case for globals -- they live in the shader globals struct
        int sg_index = ShaderGlobalNameToIndex (sym.name());
        ASSERT (sg_index >= 0);
        llvm::Value *result = builder().CreateConstGEP2_32 (sg_ptr(), 0, sg_index);
        // No derivs?  We're one indirection too few?
        result = builder().CreatePointerCast (result, llvm::PointerType::get(llvm_type(sym.typespec().elementtype()), 0));
        return result;
    }

    if (sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam) {
        // Special case for params -- they live in the group data
        int fieldnum = m_param_order_map[&sym];
        llvm::Value *result = builder().CreateConstGEP2_32 (groupdata_ptr(), 0,
                                                            fieldnum);
        // No derivs?  We're one indirection too few?
        result = builder().CreatePointerCast (result, llvm::PointerType::get(llvm_type(sym.typespec().elementtype()), 0));
        return result;
    }

    std::string mangled_name = dealiased->mangled();
    AllocationMap::iterator map_iter = named_values().find (mangled_name);
    if (map_iter == named_values().end()) {
        shadingsys().error ("Couldn't find symbol '%s' (unmangled = '%s'). Did you forget to allocate it?",
                            mangled_name.c_str(), dealiased->name().c_str());
        return 0;
    }
    return map_iter->second;
}



llvm::Value *
RuntimeOptimizer::getOrAllocateLLVMSymbol (const Symbol& sym)
{
    Symbol* dealiased = sym.dealias();
    std::string mangled_name = dealiased->mangled();
    AllocationMap::iterator map_iter = named_values().find(mangled_name);

    if (map_iter == named_values().end()) {
        const llvm::Type *alloctype = sym.typespec().is_array()
                                  ? llvm_type (sym.typespec().elementtype())
                                  : llvm_type (sym.typespec());
        int arraylen = std::max (1, sym.typespec().arraylength());
        int n = arraylen * (sym.has_derivs() ? 3 : 1);
        llvm::ConstantInt* numalloc = (llvm::ConstantInt*)llvm_constant(n);
        llvm::AllocaInst* allocation = builder().CreateAlloca(alloctype, numalloc, mangled_name);

        // llvm::outs() << "Allocation = " << *allocation << "\n";
        named_values()[mangled_name] = allocation;
        return allocation;
    }
    return map_iter->second;
}



llvm::Value *
RuntimeOptimizer::llvm_get_pointer (const Symbol& sym, int deriv,
                                    llvm::Value *arrayindex)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Return NULL for request for pointer to derivs that don't exist
        return llvm_ptr_cast (llvm_void_ptr_null(),
                              llvm::PointerType::get (llvm_type(sym.typespec()), 0));
    }

    if (sym.symtype() == SymTypeConst) {
        // For constants, just return *OUR* pointer to the constant values.
        return llvm_ptr_cast (llvm_constant_ptr (sym.data()),
                              llvm::PointerType::get (llvm_type(sym.typespec()), 0));
    }

    // Start with the initial pointer to the variable's memory location
    llvm::Value* result = getLLVMSymbolBase (sym);
    if (!result)
        return NULL;  // Error

    // If it's an array or we're dealing with derivatives, step to the
    // right element.
    TypeDesc t = sym.typespec().simpletype();
    if (t.arraylen || has_derivs) {
        int d = deriv * std::max(1,t.arraylen);
        if (arrayindex)
            arrayindex = builder().CreateAdd (arrayindex, llvm_constant(d));
        else
            arrayindex = llvm_constant(d);
        result = builder().CreateGEP (result, arrayindex);
    }

    return result;
}



llvm::Value *
RuntimeOptimizer::llvm_load_value (const Symbol& sym, int deriv,
                                   llvm::Value *arrayindex, int component,
                                   TypeDesc cast)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Regardless of what object this is, if it doesn't have derivs but
        // we're asking for them, return 0.  Integers don't have derivs
        // so we don't need to worry about that case.
        return llvm_constant (0.0f);
    }

    if (sym.is_constant()) {
        // Shortcut for simple float & int constants
        ASSERT (!arrayindex);
        if (sym.typespec().is_float()) {
            if (cast == TypeDesc::TypeInt)
                return llvm_constant ((int)*(float *)sym.data());
            else
                return llvm_constant (*(float *)sym.data());
        }
        if (sym.typespec().is_int()) {
            if (cast == TypeDesc::TypeFloat)
                return llvm_constant ((float)*(int *)sym.data());
            else
                return llvm_constant (*(int *)sym.data());
        }
        if (sym.typespec().is_triple() || sym.typespec().is_matrix()) {
            return llvm_constant (((float *)sym.data())[component]);
        }
        if (sym.typespec().is_string()) {
            return llvm_constant (*(ustring *)sym.data());
        }
        ASSERT (0 && "unhandled constant type");
    }

    // Start with the initial pointer to the value's memory location
    llvm::Value* result = llvm_get_pointer (sym, deriv, arrayindex);
    if (!result)
        return NULL;  // Error

    // If it's multi-component (triple or matrix), step to the right field
    TypeDesc t = sym.typespec().simpletype();
    if (! sym.typespec().is_closure() && t.aggregate > 1)
        result = builder().CreateConstGEP2_32 (result, 0, component);

    // Now grab the value
    result = builder().CreateLoad (result);

    if (sym.typespec().is_closure())
        return result;

    // Handle int<->float type casting
    if (sym.typespec().is_floatbased() && cast == TypeDesc::TypeInt)
        result = llvm_float_to_int (result);
    else if (sym.typespec().is_int() && cast == TypeDesc::TypeFloat)
        result = llvm_int_to_float (result);

    return result;
}



llvm::Value *
RuntimeOptimizer::llvm_load_component_value (const Symbol& sym, int deriv,
                                             llvm::Value *component)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Regardless of what object this is, if it doesn't have derivs but
        // we're asking for them, return 0.  Integers don't have derivs
        // so we don't need to worry about that case.
        ASSERT (sym.typespec().is_floatbased() && 
                "can't ask for derivs of an int");
        return llvm_constant (0.0f);
    }

    // Start with the initial pointer to the value's memory location
    llvm::Value* result = llvm_get_pointer (sym, deriv);
    if (!result)
        return NULL;  // Error

    TypeDesc t = sym.typespec().simpletype();
    ASSERT (t.aggregate != TypeDesc::SCALAR);
    std::vector<llvm::Value *> indexes;
    indexes.push_back(llvm_constant(0));
    indexes.push_back(component);
    result = builder().CreateGEP (result, indexes.begin(), indexes.end(), "compaccess");  // Find the component

    // Now grab the value
    return builder().CreateLoad (result);
}



bool
RuntimeOptimizer::llvm_store_value (llvm::Value* new_val, const Symbol& sym,
                                    int deriv, llvm::Value* arrayindex,
                                    int component)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Attempt to store deriv in symbol that doesn't have it is just a nop
        return true;
    }

    // Let llvm_get_pointer do most of the heavy lifting to get us a
    // pointer to where our data lives.
    llvm::Value *result = llvm_get_pointer (sym, deriv, arrayindex);
    if (!result)
        return false;  // Error

    // If it's multi-component (triple or matrix), step to the right field
    TypeDesc t = sym.typespec().simpletype();
    if (! sym.typespec().is_closure() && t.aggregate > 1)
        result = builder().CreateConstGEP2_32 (result, 0, component);

    // Finally, store the value.
    builder().CreateStore (new_val, result);
    return true;
}



bool
RuntimeOptimizer::llvm_store_component_value (llvm::Value* new_val,
                                              const Symbol& sym, int deriv,
                                              llvm::Value* component)
{
    bool has_derivs = sym.has_derivs();
    if (!has_derivs && deriv != 0) {
        // Attempt to store deriv in symbol that doesn't have it is just a nop
        return true;
    }

    // Let llvm_get_pointer do most of the heavy lifting to get us a
    // pointer to where our data lives.
    llvm::Value *result = llvm_get_pointer (sym, deriv);
    if (!result)
        return false;  // Error

    TypeDesc t = sym.typespec().simpletype();
    ASSERT (t.aggregate != TypeDesc::SCALAR);
    // Find the component
    llvm::Value *indexes[2] = { llvm_constant(0), component };
    result = builder().CreateGEP (result, indexes, indexes+2, "compaccess");

    // Finally, store the value.
    builder().CreateStore (new_val, result);
    return true;
}



llvm::Value *
RuntimeOptimizer::layer_run_ptr (int layer)
{
    llvm::Value *layer_run = builder().CreateConstGEP2_32 (groupdata_ptr(), 0, 0);
    return builder().CreateConstGEP2_32 (layer_run, 0, layer);
}



llvm::Value *
RuntimeOptimizer::llvm_call_function (const char *name,
                                      llvm::Value **args, int nargs)
{
    llvm::Function *func = llvm_module()->getFunction (name);
    if (! func)
        std::cerr << "Couldn't find function " << name << "\n";
    ASSERT (func);
#if 0
    llvm::outs() << "llvm_call_function " << *func << "\n";
    llvm::outs() << "\nargs:\n";
    for (int i = 0;  i < nargs;  ++i)
        llvm::outs() << "\t" << *(args[i]) << "\n";
#endif
    //llvm_gen_debug_printf (std::string("start ") + std::string(name));
    llvm::Value *r = builder().CreateCall (func, args, args+nargs);
    //llvm_gen_debug_printf (std::string(" end  ") + std::string(name));
    return r;
}



llvm::Value *
RuntimeOptimizer::llvm_call_function (const char *name, 
                                      const Symbol **symargs, int nargs,
                                      bool deriv_ptrs)
{
    std::vector<llvm::Value *> valargs;
    valargs.resize ((size_t)nargs);
    for (int i = 0;  i < nargs;  ++i) {
        const Symbol &s = *(symargs[i]);
        if (s.typespec().is_closure())
            valargs[i] = llvm_load_value (s);
        else if (s.typespec().simpletype().aggregate > 1 ||
                 (deriv_ptrs && s.has_derivs()))
            valargs[i] = llvm_void_ptr (s);
        else
            valargs[i] = llvm_load_value (s);
    }
    return llvm_call_function (name, &valargs[0], (int)valargs.size());
}



llvm::Value *
RuntimeOptimizer::llvm_call_function (const char *name, const Symbol &A,
                                      bool deriv_ptrs)
{
    const Symbol *args[1];
    args[0] = &A;
    return llvm_call_function (name, args, 1, deriv_ptrs);
}



llvm::Value *
RuntimeOptimizer::llvm_call_function (const char *name, const Symbol &A,
                                      const Symbol &B, bool deriv_ptrs)
{
    const Symbol *args[2];
    args[0] = &A;
    args[1] = &B;
    return llvm_call_function (name, args, 2, deriv_ptrs);
}



llvm::Value *
RuntimeOptimizer::llvm_call_function (const char *name, const Symbol &A,
                                      const Symbol &B, const Symbol &C,
                                      bool deriv_ptrs)
{
    const Symbol *args[3];
    args[0] = &A;
    args[1] = &B;
    args[2] = &C;
    return llvm_call_function (name, args, 3, deriv_ptrs);
}



void
RuntimeOptimizer::llvm_memset (llvm::Value *ptr, int val,
                               int len, int align)
{
    // memset with i32 len
#if OSL_LLVM_28
    // and with an i8 pointer (dst) for LLVM-2.8
    const llvm::Type* types[] = { llvm::PointerType::get(llvm::Type::getInt8Ty(llvm_context()), 0)  , llvm::Type::getInt32Ty(llvm_context()) };
#else
    const llvm::Type* types[] = { llvm::Type::getInt32Ty(llvm_context()) };
#endif

    llvm::Function* func = llvm::Intrinsic::getDeclaration(
      llvm_module(),
      llvm::Intrinsic::memset,
      types,
      sizeof(types)/sizeof(llvm::Type*));

    // NOTE(boulos): rop.llvm_constant(0) would return an i32
    // version of 0, but we need the i8 version. If we make an
    // ::llvm_constant(char val) though then we'll get ambiguity
    // everywhere.
    llvm::Value* fill_val = llvm::ConstantInt::get (llvm_context(),
                                                    llvm::APInt(8, val));
#if OSL_LLVM_28
    // Non-volatile (allow optimizer to move it around as it wishes
    // and even remove it if it can prove it's useless)
    builder().CreateCall5 (func, ptr, fill_val,
                           llvm_constant(len), llvm_constant(align), llvm_constant_bool(false));
#else
    builder().CreateCall4 (func, ptr, fill_val,
                           llvm_constant(len), llvm_constant(align));
#endif
}



void
RuntimeOptimizer::llvm_memcpy (llvm::Value *dst, llvm::Value *src,
                               int len, int align)
{
    // i32 len
#if OSL_LLVM_28
    // and with i8 pointers (dst and src) for LLVM-2.8
  const llvm::Type* types[] = { llvm::PointerType::get(llvm::Type::getInt8Ty(llvm_context()), 0), llvm::PointerType::get(llvm::Type::getInt8Ty(llvm_context()), 0)  , llvm::Type::getInt32Ty(llvm_context()) };
#else
    const llvm::Type* types[] = { llvm::Type::getInt32Ty(llvm_context()) };
#endif

    llvm::Function* func = llvm::Intrinsic::getDeclaration(llvm_module(),
                                                           llvm::Intrinsic::memcpy, types, sizeof(types) / sizeof(llvm::Type*));
#if OSL_LLVM_28
    // Non-volatile (allow optimizer to move it around as it wishes
    // and even remove it if it can prove it's useless)
    builder().CreateCall5 (func, dst, src,
                           llvm_constant(len), llvm_constant(align), llvm_constant_bool(false));
#else
    builder().CreateCall4 (func, dst, src,
                           llvm_constant(len), llvm_constant(align));
#endif
}



void
RuntimeOptimizer::llvm_gen_debug_printf (const std::string &message)
{
    ustring s = ustring::format ("(%s %s) %s", inst()->shadername().c_str(),
                                 inst()->layername().c_str(), message.c_str());
    llvm::Value *args[3] = { sg_void_ptr(), llvm_constant("%s\n"),
                             llvm_constant(s) };
    llvm::Function *func = llvm_module()->getFunction ("osl_printf");
    builder().CreateCall (func, args, args+3);
    // N.B. Above we need to do the "getFunction/CreateCall" by hand,
    // rather than call our own RuntimeOptimizer::llvm_call_function, in
    // order to avoid infinite recursion that would result if somebody
    // uncomments the debugging printfs in llvm_call_function itself.
}



/// Execute the upstream connection (if any, and if not yet run) that
/// establishes the value of symbol sym, which has index 'symindex'
/// within the current layer rop.inst().  If already_run is not NULL,
/// it points to a vector of layer indices that are known to have been 
/// run -- those can be skipped without dynamically checking their
/// execution status.
static void
llvm_run_connected_layer (RuntimeOptimizer &rop, Symbol &sym, int symindex,
                          std::vector<int> *already_run = NULL)
{
    if (sym.valuesource() != Symbol::ConnectedVal)
        return;  // Nothing to do

    // Prep the args that will be used for all earlier-layer invocations
    llvm::Value *args[2];
    args[0] = rop.sg_ptr ();
    args[1] = rop.groupdata_ptr ();

    for (int c = 0;  c < rop.inst()->nconnections();  ++c) {
        const Connection &con (rop.inst()->connection (c));
        // If the connection gives a value to this param
        if (con.dst.param == symindex) {
            if (already_run) {
                if (std::find (already_run->begin(), already_run->end(), con.srclayer) != already_run->end())
                    continue;  // already ran that one
                else
                    already_run->push_back (con.srclayer);  // mark it
            }

            // If the earlier layer it comes from has not yet
            // been executed, do so now.
            // Make code that looks like:
            //   if (! groupdata->run[parentlayer]) {
            //       groupdata->run[parentlayer] = 1;
            //       parent_layer (sg, groupdata);
            //   }
            llvm::Value *layerfield = rop.layer_run_ptr(rop.layer_remap(con.srclayer));
            llvm::Value *trueval = rop.llvm_constant_bool(true);
            ShaderInstance *parent = rop.group()[con.srclayer];
            llvm::Value *executed = rop.builder().CreateLoad (layerfield);
            executed = rop.builder().CreateICmpNE (executed, trueval);
            llvm::BasicBlock *then_block = rop.llvm_new_basic_block ("");
            llvm::BasicBlock *after_block = rop.llvm_new_basic_block ("");
            rop.builder().CreateCondBr (executed, then_block, after_block);
            rop.builder().SetInsertPoint (then_block);
            rop.builder().CreateStore (trueval, layerfield);
            std::string name = Strutil::format ("%s_%d", parent->layername().c_str(), parent->id());
            // Mark the call as a fast call
            llvm::CallInst* call_inst = llvm::cast<llvm::CallInst>(rop.llvm_call_function (name.c_str(), args, 2));
            call_inst->setCallingConv(llvm::CallingConv::Fast);
            rop.builder().CreateBr (after_block);
            rop.builder().SetInsertPoint (after_block);
        }
    }
}



LLVMGEN (llvm_gen_useparam)
{
    ASSERT (! rop.inst()->unused() &&
            "oops, thought this layer was unused, why do we call it?");

    // If we have multiple params needed on this statement, don't waste
    // time checking the same upstream layer more than once.
    std::vector<int> already_run;

    Opcode &op (rop.inst()->ops()[opnum]);
    for (int i = 0;  i < op.nargs();  ++i) {
        Symbol& sym = *rop.opargsym (op, i);
        int symindex = rop.inst()->arg (op.firstarg()+i);
        llvm_run_connected_layer (rop, sym, symindex, &already_run);
    }
    return true;
}



// Used for printf, error, warning, format
LLVMGEN (llvm_gen_printf)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    // Prepare the args for the call
    
    // Which argument is the format string?  Usually 0, but for op
    // format(), the formatting string is argument #1.
    int format_arg = (op.opname() == "format" ? 1 : 0);
    Symbol& format_sym = *rop.opargsym (op, format_arg);

    std::vector<llvm::Value*> call_args;
    if (!format_sym.is_constant()) {
        rop.shadingsys().warning ("%s must currently have constant format\n",
                                  op.opname().c_str());
        return false;
    }

    // For some ops, we push the shader globals pointer
    if (op.opname() == op_printf || op.opname() == op_error ||
            op.opname() == op_warning)
        call_args.push_back (rop.sg_void_ptr());

    // We're going to need to adjust the format string as we go, but I'd
    // like to reserve a spot for the char*.
    size_t new_format_slot = call_args.size();
    call_args.push_back(NULL);

    ustring format_ustring = *((ustring*)format_sym.data());
    const char* format = format_ustring.c_str();
    std::string s;
    int arg = format_arg + 1;
    while (*format != '\0') {
        if (*format == '%') {
            if (format[1] == '%') {
                // '%%' is a literal '%'
                s += "%%";
                format += 2;  // skip both percentages
                continue;
            }
            const char *oldfmt = format;  // mark beginning of format
            while (*format &&
                   *format != 'c' && *format != 'd' && *format != 'e' &&
                   *format != 'f' && *format != 'g' && *format != 'i' &&
                   *format != 'm' && *format != 'n' && *format != 'o' &&
                   *format != 'p' && *format != 's' && *format != 'u' &&
                   *format != 'v' && *format != 'x' && *format != 'X')
                ++format;
            ++format; // Also eat the format char
            if (arg >= op.nargs()) {
                rop.shadingsys().error ("Mismatch between format string and arguments");
                return false;
            }

            std::string ourformat (oldfmt, format);  // straddle the format
            // Doctor it to fix mismatches between format and data
            Symbol& sym (*rop.opargsym (op, arg));
            TypeDesc simpletype (sym.typespec().simpletype());
            int num_elements = simpletype.numelements();
            int num_components = simpletype.aggregate;
            // NOTE(boulos): Only for debug mode do the derivatives get printed...
            for (int a = 0;  a < num_elements;  ++a) {
                llvm::Value *arrind = simpletype.arraylen ? rop.llvm_constant(a) : NULL;
                if (sym.typespec().is_closure()) {
                    s += ourformat;
                    llvm::Value *v = rop.llvm_load_value (sym, 0, arrind, 0);
                    v = rop.llvm_call_function ("osl_closure_to_string", rop.sg_void_ptr(), v);
                    call_args.push_back (v);
                    continue;
                }

                for (int c = 0; c < num_components; c++) {
                    if (c != 0 || a != 0)
                        s += " ";
                    s += ourformat;

                    llvm::Value* loaded = rop.llvm_load_value (sym, 0, arrind, c);
                    if (sym.typespec().is_floatbased()) {
                        // C varargs convention upconverts float->double.
                        loaded = rop.builder().CreateFPExt(loaded, llvm::Type::getDoubleTy(rop.llvm_context()));
                    }

                    call_args.push_back (loaded);
                }
            }
            ++arg;
        } else if (*format == '\\') {
            // Escape sequence
            ++format;  // skip the backslash
            switch (*format) {
            case 'n' : s += '\n';     break;
            case 'r' : s += '\r';     break;
            case 't' : s += '\t';     break;
            default:   s += *format;  break;  // Catches '\\' also!
            }
            ++format;
        } else {
            // Everything else -- just copy the character and advance
            s += *format++;
        }
    }

    // Some ops prepend things
    if (op.opname() == op_error || op.opname() == op_warning) {
        std::string prefix = Strutil::format ("Shader %s [%s]: ",
                                              op.opname().c_str(),
                                              rop.inst()->shadername().c_str());
        s = prefix + s;
    }

    // Now go back and put the new format string in its place
    call_args[new_format_slot] = rop.llvm_constant (s.c_str());

    // Construct the function name and call it.
    std::string opname = std::string("osl_") + op.opname().string();
    llvm::Value *ret = rop.llvm_call_function (opname.c_str(), &call_args[0],
                                               (int)call_args.size());

    // The format op returns a string value, put in in the right spot
    if (op.opname() == op_format)
        rop.llvm_store_value (ret, *rop.opargsym (op, 0));
    return true;
}



/// Convert a float llvm value to an integer.
///
llvm::Value *
RuntimeOptimizer::llvm_float_to_int (llvm::Value* fval)
{
    return builder().CreateFPToSI(fval, llvm_type_int());
}



/// Convert an integer llvm value to a float.
///
llvm::Value *
RuntimeOptimizer::llvm_int_to_float (llvm::Value* ival)
{
    return builder().CreateSIToFP(ival, llvm_type_float());
}



/// Generate IR code for simple a/b, but considering OSL's semantics
/// that x/0 = 0, not inf.
static llvm::Value *
llvm_make_safe_div (RuntimeOptimizer &rop, TypeDesc type,
                    llvm::Value *a, llvm::Value *b)
{
    if (type.basetype == TypeDesc::FLOAT) {
        llvm::Value *div = rop.builder().CreateFDiv (a, b);
        llvm::Value *zero = rop.llvm_constant (0.0f);
        llvm::Value *iszero = rop.builder().CreateFCmpOEQ (b, zero);
        return rop.builder().CreateSelect (iszero, zero, div);
    } else {
        llvm::Value *div = rop.builder().CreateSDiv (a, b);
        llvm::Value *zero = rop.llvm_constant (0);
        llvm::Value *iszero = rop.builder().CreateICmpEQ (b, zero);
        return rop.builder().CreateSelect (iszero, zero, div);
    }
}



LLVMGEN (llvm_gen_add)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);

    if (Result.typespec().is_closure()) {
        ASSERT (A.typespec().is_closure() && B.typespec().is_closure());
        llvm::Value *valargs[3];
        valargs[0] = rop.sg_void_ptr();
        valargs[1] = rop.llvm_load_value (A);
        valargs[2] = rop.llvm_load_value (B);
        llvm::Value *res = rop.llvm_call_function ("osl_add_closure_closure", valargs, 3);
        rop.llvm_store_value (res, Result, 0, NULL, 0);
        return true;
    }

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = Result.typespec().is_floatbased();
    int num_components = type.aggregate;

    // The following should handle f+f, v+v, v+f, f+v, i+i
    // That's all that should be allowed by oslc.
    for (int i = 0; i < num_components; i++) {
        llvm::Value *a = rop.loadLLVMValue (A, i, 0, type);
        llvm::Value *b = rop.loadLLVMValue (B, i, 0, type);
        if (!a || !b)
            return false;
        llvm::Value *r = is_float ? rop.builder().CreateFAdd(a, b)
                                  : rop.builder().CreateAdd(a, b);
        rop.storeLLVMValue (r, Result, i, 0);
    }

    if (Result.has_derivs()) {
        ASSERT (is_float);
        if (A.has_derivs() || B.has_derivs()) {
            for (int d = 1;  d <= 2;  ++d) {  // dx, dy
                for (int i = 0; i < num_components; i++) {
                    llvm::Value *a = rop.loadLLVMValue (A, i, d, type);
                    llvm::Value *b = rop.loadLLVMValue (B, i, d, type);
                    llvm::Value *r = rop.builder().CreateFAdd(a, b);
                    rop.storeLLVMValue (r, Result, i, d);
                }
            }
        } else {
            // Result has derivs, operands do not
            rop.llvm_zero_derivs (Result);
        }
    }
    return true;
}



LLVMGEN (llvm_gen_sub)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = Result.typespec().is_floatbased();
    int num_components = type.aggregate;

    ASSERT (! Result.typespec().is_closure() &&
            "subtraction of closures not supported");

    // The following should handle f-f, v-v, v-f, f-v, i-i
    // That's all that should be allowed by oslc.
    for (int i = 0; i < num_components; i++) {
        llvm::Value *a = rop.loadLLVMValue (A, i, 0, type);
        llvm::Value *b = rop.loadLLVMValue (B, i, 0, type);
        if (!a || !b)
            return false;
        llvm::Value *r = is_float ? rop.builder().CreateFSub(a, b)
                                  : rop.builder().CreateSub(a, b);
        rop.storeLLVMValue (r, Result, i, 0);
    }

    if (Result.has_derivs()) {
        ASSERT (is_float);
        if (A.has_derivs() || B.has_derivs()) {
            for (int d = 1;  d <= 2;  ++d) {  // dx, dy
                for (int i = 0; i < num_components; i++) {
                    llvm::Value *a = rop.loadLLVMValue (A, i, d, type);
                    llvm::Value *b = rop.loadLLVMValue (B, i, d, type);
                    llvm::Value *r = rop.builder().CreateFSub(a, b);
                    rop.storeLLVMValue (r, Result, i, d);
                }
            }
        } else {
            // Result has derivs, operands do not
            rop.llvm_zero_derivs (Result);
        }
    }
    return true;
}



LLVMGEN (llvm_gen_mul)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = !Result.typespec().is_closure() && Result.typespec().is_floatbased();
    int num_components = type.aggregate;

    // multiplication involving closures
    if (Result.typespec().is_closure()) {
        llvm::Value *valargs[3];
        valargs[0] = rop.sg_void_ptr();
        bool tfloat;
        if (A.typespec().is_closure()) {
            tfloat = B.typespec().is_float();
            valargs[1] = rop.llvm_load_value (A);
            valargs[2] = tfloat ? rop.llvm_load_value (B) : rop.llvm_void_ptr(B);
        } else {
            tfloat = A.typespec().is_float();
            valargs[1] = rop.llvm_load_value (B);
            valargs[2] = tfloat ? rop.llvm_load_value (A) : rop.llvm_void_ptr(A);
        }
        llvm::Value *res = tfloat ? rop.llvm_call_function ("osl_mul_closure_float", valargs, 3)
                                  : rop.llvm_call_function ("osl_mul_closure_color", valargs, 3);
        rop.llvm_store_value (res, Result, 0, NULL, 0);
        return true;
    }

    // multiplication involving matrices
    if (Result.typespec().is_matrix()) {
        if (A.typespec().is_float()) {
            if (B.typespec().is_float())
                rop.llvm_call_function ("osl_mul_m_ff", Result, A, B);
            else if (B.typespec().is_matrix())
                rop.llvm_call_function ("osl_mul_mf", Result, B, A);
            else ASSERT(0);
        } else if (A.typespec().is_matrix()) {
            if (B.typespec().is_float())
                rop.llvm_call_function ("osl_mul_mf", Result, A, B);
            else if (B.typespec().is_matrix())
                rop.llvm_call_function ("osl_mul_mm", Result, A, B);
            else ASSERT(0);
        } else ASSERT (0);
        if (Result.has_derivs())
            rop.llvm_zero_derivs (Result);
        return true;
    }

    // The following should handle f*f, v*v, v*f, f*v, i*i
    // That's all that should be allowed by oslc.
    for (int i = 0; i < num_components; i++) {
        llvm::Value *a = rop.loadLLVMValue (A, i, 0, type);
        llvm::Value *b = rop.loadLLVMValue (B, i, 0, type);
        if (!a || !b)
            return false;
        llvm::Value *r = is_float ? rop.builder().CreateFMul(a, b)
                                  : rop.builder().CreateMul(a, b);
        rop.storeLLVMValue (r, Result, i, 0);
    }

    if (Result.has_derivs()) {
        ASSERT (is_float);
        if (A.has_derivs() || B.has_derivs()) {
            // Multiplication of duals: (a*b, a*b.dx + a.dx*b, a*b.dy + a.dy*b)
            for (int i = 0; i < num_components; i++) {
                llvm::Value *a = rop.loadLLVMValue (A, i, 0, type);
                llvm::Value *b = rop.loadLLVMValue (B, i, 0, type);
                llvm::Value *ax = rop.loadLLVMValue (A, i, 1, type);
                llvm::Value *bx = rop.loadLLVMValue (B, i, 1, type);
                llvm::Value *abx = rop.builder().CreateFMul(a, bx);
                llvm::Value *axb = rop.builder().CreateFMul(ax, b);
                llvm::Value *r = rop.builder().CreateFAdd(abx, axb);
                rop.storeLLVMValue (r, Result, i, 1);
            }

            for (int i = 0; i < num_components; i++) {
                llvm::Value *a = rop.loadLLVMValue (A, i, 0, type);
                llvm::Value *b = rop.loadLLVMValue (B, i, 0, type);
                llvm::Value *ay = rop.loadLLVMValue (A, i, 2, type);
                llvm::Value *by = rop.loadLLVMValue (B, i, 2, type);
                llvm::Value *aby = rop.builder().CreateFMul(a, by);
                llvm::Value *ayb = rop.builder().CreateFMul(ay, b);
                llvm::Value *r = rop.builder().CreateFAdd(aby, ayb);
                rop.storeLLVMValue (r, Result, i, 2);
            }
        } else {
            // Result has derivs, operands do not
            rop.llvm_zero_derivs (Result);
        }
    }
    return true;
}



LLVMGEN (llvm_gen_div)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = Result.typespec().is_floatbased();
    int num_components = type.aggregate;

    ASSERT (! Result.typespec().is_closure());

    // division involving matrices
    if (Result.typespec().is_matrix()) {
        if (A.typespec().is_float()) {
            if (B.typespec().is_float())
                rop.llvm_call_function ("osl_div_m_ff", Result, A, B);
            else if (B.typespec().is_matrix())
                rop.llvm_call_function ("osl_div_fm", Result, A, B);
            else ASSERT (0);
        } else if (A.typespec().is_matrix()) {
            if (B.typespec().is_float())
                rop.llvm_call_function ("osl_div_mf", Result, A, B);
            else if (B.typespec().is_matrix())
                rop.llvm_call_function ("osl_div_mm", Result, A, B);
            else ASSERT (0);
        } else ASSERT (0);
        if (Result.has_derivs())
            rop.llvm_zero_derivs (Result);
        return true;
    }

    // The following should handle f/f, v/v, v/f, f/v, i/i
    // That's all that should be allowed by oslc.
    for (int i = 0; i < num_components; i++) {
        llvm::Value *a = rop.loadLLVMValue (A, i, 0, type);
        llvm::Value *b = rop.loadLLVMValue (B, i, 0, type);
        if (!a || !b)
            return false;
        llvm::Value *r = llvm_make_safe_div (rop, type, a, b);
        rop.storeLLVMValue (r, Result, i, 0);
    }

    if (Result.has_derivs()) {
        ASSERT (is_float);
        if (A.has_derivs() || B.has_derivs()) {
            // Division of duals: (a/b, 1/b*(ax-a/b*bx), 1/b*(ay-a/b*by))
            for (int i = 0; i < num_components; i++) {
                llvm::Value *a = rop.loadLLVMValue (A, i, 0, type);
                llvm::Value *b = rop.loadLLVMValue (B, i, 0, type);
                llvm::Value *binv = llvm_make_safe_div (rop, type,
                                                   rop.llvm_constant(1.0f), b);
                llvm::Value *ax = rop.loadLLVMValue (A, i, 1, type);
                llvm::Value *bx = rop.loadLLVMValue (B, i, 1, type);
                llvm::Value *a_div_b = rop.builder().CreateFMul (a, binv);
                llvm::Value *a_div_b_mul_bx = rop.builder().CreateFMul (a_div_b, bx);
                llvm::Value *ax_minus_a_div_b_mul_bx = rop.builder().CreateFSub (ax, a_div_b_mul_bx);
                llvm::Value *r = rop.builder().CreateFMul (binv, ax_minus_a_div_b_mul_bx);
                rop.storeLLVMValue (r, Result, i, 1);
            }

            for (int i = 0; i < num_components; i++) {
                llvm::Value *a = rop.loadLLVMValue (A, i, 0, type);
                llvm::Value *b = rop.loadLLVMValue (B, i, 0, type);
                llvm::Value *binv = llvm_make_safe_div (rop, type,
                                                   rop.llvm_constant(1.0f), b);
                llvm::Value *ay = rop.loadLLVMValue (A, i, 2, type);
                llvm::Value *by = rop.loadLLVMValue (B, i, 2, type);
                llvm::Value *a_div_b = rop.builder().CreateFMul (a, binv);
                llvm::Value *a_div_b_mul_by = rop.builder().CreateFMul (a_div_b, by);
                llvm::Value *ay_minus_a_div_b_mul_by = rop.builder().CreateFSub (ay, a_div_b_mul_by);
                llvm::Value *r = rop.builder().CreateFMul (binv, ay_minus_a_div_b_mul_by);
                rop.storeLLVMValue (r, Result, i, 2);
            }
        } else {
            // Result has derivs, operands do not
            rop.llvm_zero_derivs (Result);
        }
    }
    return true;
}



/// Generate IR code for simple a mod b, but considering OSL's semantics
/// that x mod 0 = 0, not inf.
static llvm::Value *
llvm_make_safe_mod (RuntimeOptimizer &rop, TypeDesc type,
                    llvm::Value *a, llvm::Value *b)
{
    if (type.basetype == TypeDesc::FLOAT) {
        llvm::Value *mod = rop.builder().CreateFRem (a, b);
        llvm::Value *zero = rop.llvm_constant (0.0f);
        llvm::Value *iszero = rop.builder().CreateFCmpOEQ (b, zero);
        return rop.builder().CreateSelect (iszero, zero, mod);
    } else {
        llvm::Value *mod = rop.builder().CreateSRem (a, b);
        llvm::Value *zero = rop.llvm_constant (0);
        llvm::Value *iszero = rop.builder().CreateICmpEQ (b, zero);
        return rop.builder().CreateSelect (iszero, zero, mod);
    }
}



LLVMGEN (llvm_gen_mod)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = Result.typespec().is_floatbased();
    int num_components = type.aggregate;

    // The following should handle f%f, v%v, v%f, i%i
    // That's all that should be allowed by oslc.
    for (int i = 0; i < num_components; i++) {
        llvm::Value *a = rop.loadLLVMValue (A, i, 0, type);
        llvm::Value *b = rop.loadLLVMValue (B, i, 0, type);
        if (!a || !b)
            return false;
        llvm::Value *r = llvm_make_safe_mod (rop, type, a, b);
        rop.storeLLVMValue (r, Result, i, 0);
    }

    if (Result.has_derivs()) {
        ASSERT (is_float);
        if (A.has_derivs()) {
            // Modulus of duals: (a mod b, ax, ay)
            for (int d = 1;  d <= 2;  ++d) {
                for (int i = 0; i < num_components; i++) {
                    llvm::Value *deriv = rop.loadLLVMValue (A, i, d, type);
                    rop.storeLLVMValue (deriv, Result, i, d);
                }
            }
        } else {
            // Result has derivs, operands do not
            rop.llvm_zero_derivs (Result);
        }
    }
    return true;
}



LLVMGEN (llvm_gen_neg)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = Result.typespec().is_floatbased();
    int num_components = type.aggregate;
    for (int d = 0;  d < 3;  ++d) {  // dx, dy
        for (int i = 0; i < num_components; i++) {
            llvm::Value *a = rop.llvm_load_value (A, d, i, type);
            llvm::Value *r = is_float ? rop.builder().CreateFNeg(a)
                                      : rop.builder().CreateNeg(a);
            rop.llvm_store_value (r, Result, d, i);
        }
        if (! Result.has_derivs())
            break;
    }
    return true;
}



// Implementation for clamp
LLVMGEN (llvm_gen_clamp)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& X = *rop.opargsym (op, 1);
    Symbol& Min = *rop.opargsym (op, 2);
    Symbol& Max = *rop.opargsym (op, 3);

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = Result.typespec().is_floatbased();
    int num_components = type.aggregate;
    for (int i = 0; i < num_components; i++) {
        // First do the lower bound
        llvm::Value *val = rop.llvm_load_value (X, 0, i, type);
        llvm::Value *min = rop.llvm_load_value (Min, 0, i, type);
        llvm::Value *cond = is_float ? rop.builder().CreateFCmpULT(val, min)
                                     : rop.builder().CreateICmpSLT(val, min);
        val = rop.builder().CreateSelect (cond, min, val);
        llvm::Value *valdx=NULL, *valdy=NULL;
        if (Result.has_derivs()) {
            valdx = rop.llvm_load_value (X, 1, i, type);
            valdy = rop.llvm_load_value (X, 2, i, type);
            llvm::Value *mindx = rop.llvm_load_value (Min, 1, i, type);
            llvm::Value *mindy = rop.llvm_load_value (Min, 2, i, type);
            valdx = rop.builder().CreateSelect (cond, mindx, valdx);
            valdy = rop.builder().CreateSelect (cond, mindy, valdy);
        }
        // Now do the upper bound
        llvm::Value *max = rop.llvm_load_value (Max, 0, i, type);
        cond = is_float ? rop.builder().CreateFCmpUGT(val, max)
                        : rop.builder().CreateICmpSGT(val, max);
        val = rop.builder().CreateSelect (cond, max, val);
        if (Result.has_derivs()) {
            llvm::Value *maxdx = rop.llvm_load_value (Max, 1, i, type);
            llvm::Value *maxdy = rop.llvm_load_value (Max, 2, i, type);
            valdx = rop.builder().CreateSelect (cond, maxdx, valdx);
            valdy = rop.builder().CreateSelect (cond, maxdy, valdy);
        }
        rop.llvm_store_value (val, Result, 0, i);
        rop.llvm_store_value (valdx, Result, 1, i);
        rop.llvm_store_value (valdy, Result, 2, i);
    }
    return true;
}

// Implementation for min/max
LLVMGEN (llvm_gen_minmax)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& x = *rop.opargsym (op, 1);
    Symbol& y = *rop.opargsym (op, 2);

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = Result.typespec().is_floatbased();
    int num_components = type.aggregate;
    for (int i = 0; i < num_components; i++) {
        // First do the lower bound
        llvm::Value *x_val = rop.llvm_load_value (x, 0, i, type);
        llvm::Value *y_val = rop.llvm_load_value (y, 0, i, type);

        llvm::Value* cond = NULL;
        // NOTE(boulos): Using <= instead of < to match old behavior
        // (only matters for derivs)
        if (op.opname() == op_min) {
          cond = (is_float) ? rop.builder().CreateFCmpULE(x_val, y_val) :
            rop.builder().CreateICmpSLE(x_val, y_val);
        } else {
          cond = (is_float) ? rop.builder().CreateFCmpUGT(x_val, y_val) :
            rop.builder().CreateICmpSGT(x_val, y_val);
        }

        llvm::Value* res_val = rop.builder().CreateSelect (cond, x_val, y_val);
        rop.llvm_store_value (res_val, Result, 0, i);
        if (Result.has_derivs()) {
          llvm::Value* x_dx = rop.llvm_load_value (x, 1, i, type);
          llvm::Value* x_dy = rop.llvm_load_value (x, 2, i, type);
          llvm::Value* y_dx = rop.llvm_load_value (y, 1, i, type);
          llvm::Value* y_dy = rop.llvm_load_value (y, 2, i, type);
          rop.llvm_store_value (rop.builder().CreateSelect(cond, x_dx, y_dx), Result, 1, i);
          rop.llvm_store_value (rop.builder().CreateSelect(cond, x_dy, y_dy), Result, 2, i);
        }
    }
    return true;
}



LLVMGEN (llvm_gen_bitwise_binary_op)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);
    ASSERT (Result.typespec().is_int() && A.typespec().is_int() && 
            B.typespec().is_int());

    llvm::Value *a = rop.loadLLVMValue (A);
    llvm::Value *b = rop.loadLLVMValue (B);
    if (!a || !b)
        return false;
    llvm::Value *r = NULL;
    if (op.opname() == op_bitand)
        r = rop.builder().CreateAnd (a, b);
    else if (op.opname() == op_bitor)
        r = rop.builder().CreateOr (a, b);
    else if (op.opname() == op_xor)
        r = rop.builder().CreateXor (a, b);
    else if (op.opname() == op_shl)
        r = rop.builder().CreateShl (a, b);
    else if (op.opname() == op_shr)
        r = rop.builder().CreateAShr (a, b);  // signed int -> arithmetic shift
    else
        return false;
    rop.storeLLVMValue (r, Result);
    return true;
}



// Simple (pointwise) unary ops (Abs, ..., 
LLVMGEN (llvm_gen_unary_op)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& dst  = *rop.opargsym (op, 0);
    Symbol& src = *rop.opargsym (op, 1);
    bool dst_derivs = dst.has_derivs();
    int num_components = dst.typespec().simpletype().aggregate;

    bool dst_float = dst.typespec().is_floatbased();
    bool src_float = src.typespec().is_floatbased();

    for (int i = 0; i < num_components; i++) {
        // Get src1/2 component i
        llvm::Value* src_load = rop.loadLLVMValue (src, i, 0);
        if (!src_load) return false;

        llvm::Value* src_val = src_load;

        // Perform the op
        llvm::Value* result = 0;
        ustring opname = op.opname();

        if (opname == op_compl) {
            ASSERT (dst.typespec().is_int());
            result = rop.builder().CreateNot(src_val);
        } else {
            // Don't know how to handle this.
            rop.shadingsys().error ("Don't know how to handle op '%s', eliding the store\n", opname.c_str());
        }

        // Store the result
        if (result) {
            // if our op type doesn't match result, convert
            if (dst_float && !src_float) {
                // Op was int, but we need to store float
                result = rop.llvm_int_to_float (result);
            } else if (!dst_float && src_float) {
                // Op was float, but we need to store int
                result = rop.llvm_float_to_int (result);
            } // otherwise just fine
            rop.storeLLVMValue (result, dst, i, 0);
        }

        if (dst_derivs) {
            // mul results in <a * b, a * b_dx + b * a_dx, a * b_dy + b * a_dy>
            rop.shadingsys().info ("punting on derivatives for now\n");
            // FIXME!!
        }
    }
    return true;
}



// Implementaiton of Simple assignment.  If arrayindex >= 0, in designates
// a particular array index to assign.
static bool
llvm_assign_impl (RuntimeOptimizer &rop, Symbol &Result, Symbol &Src,
                  int arrayindex = -1)
{
    ASSERT (! Result.typespec().is_structure());
    ASSERT (! Src.typespec().is_structure());

    llvm::Value *arrind = arrayindex >= 0 ? rop.llvm_constant (arrayindex) : NULL;

    if (Result.typespec().is_closure() || Src.typespec().is_closure()) {
        if (Src.typespec().is_closure()) {
            llvm::Value *srcval = rop.llvm_load_value (Src, 0, arrind, 0);
            rop.llvm_store_value (srcval, Result, 0, arrind, 0);
        } else {
            llvm::Value *null = rop.llvm_constant_ptr(NULL, rop.llvm_type_void_ptr());
            rop.llvm_store_value (null, Result, 0, arrind, 0);
        }
        return true;
    }

    if (Result.typespec().is_matrix() && Src.typespec().is_int_or_float()) {
        // Handle m=f, m=i separately
        llvm::Value *src = rop.llvm_load_value (Src, 0, arrind, 0, TypeDesc::FLOAT /*cast*/);
        // m=f sets the diagonal components to f, the others to zero
        llvm::Value *zero = rop.llvm_constant (0.0f);
        for (int i = 0;  i < 4;  ++i)
            for (int j = 0;  j < 4;  ++j)
                rop.llvm_store_value (i==j ? src : zero, Result, 0, arrind, i*4+j);
        rop.llvm_zero_derivs (Result);  // matrices don't have derivs currently
        return true;
    }

    // The following code handles f=f, f=i, v=v, v=f, v=i, m=m, s=s.
    // Remember that llvm_load_value will automatically convert scalar->triple.
    TypeDesc rt = Result.typespec().simpletype();
    TypeDesc basetype = TypeDesc::BASETYPE(rt.basetype);
    int num_components = rt.aggregate;
    for (int i = 0; i < num_components; ++i) {
        llvm::Value* src_val = rop.llvm_load_value (Src, 0, arrind, i, basetype);
        if (!src_val)
            return false;
        rop.llvm_store_value (src_val, Result, 0, arrind, i);
    }

    // Handle derivatives
    if (Result.has_derivs()) {
        if (Src.has_derivs()) {
            // src and result both have derivs -- copy them
            for (int d = 1;  d <= 2;  ++d) {
                for (int i = 0; i < num_components; ++i) {
                    llvm::Value* val = rop.llvm_load_value (Src, d, arrind, i);
                    rop.llvm_store_value (val, Result, d, arrind, i);
                }
            }
        } else {
            // Result wants derivs but src didn't have them -- zero them
            rop.llvm_zero_derivs (Result);
        }
    }
    return true;
}



// Simple assignment
LLVMGEN (llvm_gen_assign)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result (*rop.opargsym (op, 0));
    Symbol& Src (*rop.opargsym (op, 1));

    return llvm_assign_impl (rop, Result, Src);
}



// Vector component reference
LLVMGEN (llvm_gen_compref)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Val = *rop.opargsym (op, 1);
    Symbol& Index = *rop.opargsym (op, 2);

    for (int d = 0;  d < 3;  ++d) {  // deriv
        llvm::Value *val = NULL; 
        // FIXME -- should we test for out-of-range component?
        if (Index.is_constant()) {
            val = rop.llvm_load_value (Val, d, *((int*)Index.data()));
        } else {
            llvm::Value *c = rop.llvm_load_value(Index, d);
            val = rop.llvm_load_component_value (Val, d, c);
        }
        rop.llvm_store_value (val, Result, d);
        if (! Result.has_derivs())  // skip the derivs if we don't need them
            break;
    }
    return true;
}



// Vector component assignment
LLVMGEN (llvm_gen_compassign)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Index = *rop.opargsym (op, 1);
    Symbol& Val = *rop.opargsym (op, 2);

    for (int d = 0;  d < 3;  ++d) {  // deriv
        llvm::Value *val = rop.llvm_load_value (Val, d, 0, TypeDesc::TypeFloat);
        // FIXME -- should we test for out-of-range component?
        if (Index.is_constant()) {
            rop.llvm_store_value (val, Result, d, *((int*)Index.data()));
        } else {
            llvm::Value *c = rop.llvm_load_value(Index, d);
            rop.llvm_store_component_value (val, Result, d, c);
        }
        if (! Result.has_derivs())  // skip the derivs if we don't need them
            break;
    }
    return true;
}



// Matrix component reference
LLVMGEN (llvm_gen_mxcompref)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& M = *rop.opargsym (op, 1);
    Symbol& Row = *rop.opargsym (op, 2);
    Symbol& Col = *rop.opargsym (op, 3);

    llvm::Value *val = NULL; 
    // FIXME -- should we test for out-of-range component?
    if (Row.is_constant() && Col.is_constant()) {
        int comp = 4 * ((int*)Row.data())[0] + ((int*)Col.data())[0];
        val = rop.llvm_load_value (M, 0, comp);
    } else {
        llvm::Value *row = rop.llvm_load_value (Row);
        llvm::Value *col = rop.llvm_load_value (Col);
        llvm::Value *comp = rop.builder().CreateMul (row, rop.llvm_constant(4));
        comp = rop.builder().CreateAdd (comp, col);
        val = rop.llvm_load_component_value (M, 0, comp);
    }
    rop.llvm_store_value (val, Result);
    rop.llvm_zero_derivs (Result);

    return true;
}



// Matrix component assignment
LLVMGEN (llvm_gen_mxcompassign)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Row = *rop.opargsym (op, 1);
    Symbol& Col = *rop.opargsym (op, 2);
    Symbol& Val = *rop.opargsym (op, 3);

    llvm::Value *val = rop.llvm_load_value (Val, 0, 0, TypeDesc::TypeFloat);

    // FIXME -- should we test for out-of-range component?
    if (Row.is_constant() && Col.is_constant()) {
        int comp = 4 * ((int*)Row.data())[0] + ((int*)Col.data())[0];
        rop.llvm_store_value (val, Result, 0, comp);
    } else {
        llvm::Value *row = rop.llvm_load_value(Row);
        llvm::Value *col = rop.llvm_load_value(Col);
        llvm::Value *comp = rop.builder().CreateMul (row, rop.llvm_constant(4));
        comp = rop.builder().CreateAdd (comp, col);
        rop.llvm_store_component_value (val, Result, 0, comp);
    }
    return true;
}



// Array length
LLVMGEN (llvm_gen_arraylength)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    DASSERT (Result.typespec().is_int() && A.typespec().is_array());

    int len = A.typespec().arraylength();
    rop.llvm_store_value (rop.llvm_constant(len), Result);
    return true;
}



// Array reference
LLVMGEN (llvm_gen_aref)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Src = *rop.opargsym (op, 1);
    Symbol& Index = *rop.opargsym (op, 2);

    // Get array index we're interested in
    llvm::Value *index = rop.loadLLVMValue (Index);
    if (! index)
        return false;
    // FIXME -- do range detection here for constant indices
    // Or should we always do range detection for safety?

    int num_components = Src.typespec().simpletype().aggregate;
    for (int d = 0;  d <= 2;  ++d) {
        for (int c = 0;  c < num_components;  ++c) {
            llvm::Value *val = rop.llvm_load_value (Src, d, index, c);
            rop.storeLLVMValue (val, Result, c, d);
        }
        if (! Result.has_derivs())
            break;
    }

    return true;
}



// Array assignment
LLVMGEN (llvm_gen_aassign)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Index = *rop.opargsym (op, 1);
    Symbol& Src = *rop.opargsym (op, 2);

    // Get array index we're interested in
    llvm::Value *index = rop.loadLLVMValue (Index);
    if (! index)
        return false;
    // FIXME -- do range detection here for constant indices
    // Or should we always do range detection for safety?

    int num_components = Result.typespec().simpletype().aggregate;
    for (int d = 0;  d <= 2;  ++d) {
        for (int c = 0;  c < num_components;  ++c) {
            llvm::Value *val = rop.loadLLVMValue (Src, c, d);
            rop.llvm_store_value (val, Result, d, index, c);
        }
        if (! Result.has_derivs())
            break;
    }

    return true;
}



// Construct triple (point, vector, normal, color)
LLVMGEN (llvm_gen_construct_triple)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    bool using_space = (op.nargs() == 5);
    Symbol& Space = *rop.opargsym (op, 1);
    Symbol& X = *rop.opargsym (op, 1+using_space);
    Symbol& Y = *rop.opargsym (op, 2+using_space);
    Symbol& Z = *rop.opargsym (op, 3+using_space);
    ASSERT (Result.typespec().is_triple() && X.typespec().is_float() &&
            Y.typespec().is_float() && Z.typespec().is_float() &&
            (using_space == false || Space.typespec().is_string()));

    for (int d = 0;  d < 3;  ++d) {  // loop over derivs
        // First, copy the floats into the vector
        for (int c = 0;  c < 3;  ++c) {  // loop over components
            const Symbol& comp = *rop.opargsym (op, c+1+using_space);
            llvm::Value* val = rop.llvm_load_value (comp, d, NULL, 0, TypeDesc::TypeFloat);
            rop.llvm_store_value (val, Result, d, NULL, c);
        }
        // Do the transformation in-place, if called for
        if (using_space) {
            llvm::Value *args[3];
            args[0] = rop.sg_void_ptr ();  // shader globals
            args[1] = rop.llvm_void_ptr (Result, d);  // vector
            args[2] = rop.llvm_load_value (Space); // from
            if (op.opname() == op_color) {
                if (d == 0)
                    rop.llvm_call_function ("osl_prepend_color_from", args, 3);
            } else if (op.opname() == op_vector || d > 0) {
                // NB. treat derivs of points and normals as vecs
                rop.llvm_call_function ("osl_prepend_vector_from", args, 3);
            }
            else if (op.opname() == op_point)
                rop.llvm_call_function ("osl_prepend_point_from", args, 3);
            else if (op.opname() == op_normal)
                rop.llvm_call_function ("osl_prepend_normal_from", args, 3);
            else
                ASSERT(0 && "unsupported color ctr with color space name");
        }
        if (! Result.has_derivs())
            break;    // don't bother if Result doesn't have derivs
    }

    // FIXME: Punt on derivs for color ctrs with space names.  We should 
    // try to do this right, but we never had it right for the interpreter,
    // to it's probably not an emergency.
    if (using_space && op.opname() == op_color && Result.has_derivs())
        rop.llvm_zero_derivs (Result);

    return true;
}



/// matrix constructor.  Comes in several varieties:
///    matrix (float)
///    matrix (space, float)
///    matrix (...16 floats...)
///    matrix (space, ...16 floats...)
///    matrix (fromspace, tospace)
LLVMGEN (llvm_gen_matrix)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    int nargs = op.nargs();
    bool using_space = (nargs == 3 || nargs == 18);
    bool using_two_spaces = (nargs == 3 && rop.opargsym(op,2)->typespec().is_string());
    int nfloats = nargs - 1 - (int)using_space;
    ASSERT (nargs == 2 || nargs == 3 || nargs == 17 || nargs == 18);

    if (using_two_spaces) {
        llvm::Value *args[4];
        args[0] = rop.sg_void_ptr();  // shader globals
        args[1] = rop.llvm_void_ptr(Result);  // result
        args[2] = rop.llvm_load_value(*rop.opargsym (op, 1));  // from
        args[3] = rop.llvm_load_value(*rop.opargsym (op, 2));  // to
        rop.llvm_call_function ("osl_get_from_to_matrix", args, 4);
    } else {
        if (nfloats == 1) {
            for (int i = 0; i < 16; i++) {
                llvm::Value* src_val = ((i%4) == (i/4)) 
                    ? rop.llvm_load_value (*rop.opargsym(op,1+using_space))
                    : rop.llvm_constant(0.0f);
                rop.llvm_store_value (src_val, Result, 0, i);
            }
        } else if (nfloats == 16) {
            for (int i = 0; i < 16; i++) {
                llvm::Value* src_val = rop.llvm_load_value (*rop.opargsym(op,i+1+using_space));
                rop.llvm_store_value (src_val, Result, 0, i);
            }
        } else {
            ASSERT (0);
        }
        if (using_space) {
            llvm::Value *args[3];
            args[0] = rop.sg_void_ptr();  // shader globals
            args[1] = rop.llvm_void_ptr(Result);  // result
            args[2] = rop.llvm_load_value(*rop.opargsym (op, 1));  // from
            rop.llvm_call_function ("osl_prepend_matrix_from", args, 3);
        }
    }
    if (Result.has_derivs())
        rop.llvm_zero_derivs (Result);
    return true;
}



// Derivs
LLVMGEN (llvm_gen_DxDy)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result (*rop.opargsym (op, 0));
    Symbol& Src (*rop.opargsym (op, 1));
    int deriv = (op.opname() == "Dx") ? 1 : 2;

    for (int i = 0; i < Result.typespec().aggregate(); ++i) {
        llvm::Value* src_val = rop.llvm_load_value (Src, deriv, i);
        rop.storeLLVMValue (src_val, Result, i, 0);
    }

    // Don't have 2nd order derivs
    rop.llvm_zero_derivs (Result);
    return true;
}



// Derivs
LLVMGEN (llvm_gen_filterwidth)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result (*rop.opargsym (op, 0));
    Symbol& Src (*rop.opargsym (op, 1));

    ASSERT (Src.typespec().is_float() || Src.typespec().is_triple());
    if (Src.has_derivs()) {
        if (Src.typespec().is_float()) {
            llvm::Value *r = rop.llvm_call_function ("osl_filterwidth_fdf",
                                                     rop.llvm_void_ptr (Src));
            rop.llvm_store_value (r, Result);
        } else {
            rop.llvm_call_function ("osl_filterwidth_vdv",
                                    rop.llvm_void_ptr (Result),
                                    rop.llvm_void_ptr (Src));
        }
        // Don't have 2nd order derivs
        rop.llvm_zero_derivs (Result);
    } else {
        // No derivs to be had
        rop.llvm_assign_zero (Src);
    }

    return true;
}



// Comparison ops
LLVMGEN (llvm_gen_compare_op)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &Result (*rop.opargsym (op, 0));
    Symbol &A (*rop.opargsym (op, 1));
    Symbol &B (*rop.opargsym (op, 2));
    ASSERT (Result.typespec().is_int() && ! Result.has_derivs());

    int num_components = std::max (A.typespec().aggregate(), B.typespec().aggregate());
    bool float_based = A.typespec().is_floatbased() || B.typespec().is_floatbased();
    TypeDesc cast (float_based ? TypeDesc::FLOAT : TypeDesc::UNKNOWN);

    llvm::Value* final_result = 0;
    ustring opname = op.opname();

    for (int i = 0; i < num_components; i++) {
        // Get A&B component i -- note that these correctly handle mixed
        // scalar/triple comparisons as well as int->float casts as needed.
        llvm::Value* a = rop.loadLLVMValue (A, i, 0, cast);
        llvm::Value* b = rop.loadLLVMValue (B, i, 0, cast);

        // Trickery for mixed matrix/scalar comparisons -- compare
        // on-diagonal to the scalar, off-diagonal to zero
        if (A.typespec().is_matrix() && !B.typespec().is_matrix()) {
            if ((i/4) != (i%4))
                b = rop.llvm_constant (0.0f);
        }
        if (! A.typespec().is_matrix() && B.typespec().is_matrix()) {
            if ((i/4) != (i%4))
                a = rop.llvm_constant (0.0f);
        }

        // Perform the op
        llvm::Value* result = 0;
        if (opname == op_lt) {
            result = float_based ? rop.builder().CreateFCmpULT(a, b) : rop.builder().CreateICmpSLT(a, b);
        } else if (opname == op_le) {
            result = float_based ? rop.builder().CreateFCmpULE(a, b) : rop.builder().CreateICmpSLE(a, b);
        } else if (opname == op_eq) {
            result = float_based ? rop.builder().CreateFCmpUEQ(a, b) : rop.builder().CreateICmpEQ(a, b);
        } else if (opname == op_ge) {
            result = float_based ? rop.builder().CreateFCmpUGE(a, b) : rop.builder().CreateICmpSGE(a, b);
        } else if (opname == op_gt) {
            result = float_based ? rop.builder().CreateFCmpUGT(a, b) : rop.builder().CreateICmpSGT(a, b);
        } else if (opname == op_neq) {
            result = float_based ? rop.builder().CreateFCmpUNE(a, b) : rop.builder().CreateICmpNE(a, b);
        } else {
            // Don't know how to handle this.
            ASSERT (0 && "Comparison error");
        }
        ASSERT (result);

        if (final_result) {
            // Combine the component bool based on the op
            if (opname != op_neq)        // final_result &= result
                final_result = rop.builder().CreateAnd(final_result, result);
            else                         // final_result |= result
                final_result = rop.builder().CreateOr(final_result, result);
        } else {
            final_result = result;
        }
    }
    ASSERT (final_result);

    // Convert the single bit bool into an int for now.
    final_result = rop.builder().CreateZExt (final_result, rop.llvm_type_int());
    rop.storeLLVMValue (final_result, Result, 0, 0);
    return true;
}



// int regex_search (string subject, string pattern)
// int regex_search (string subject, int results[], string pattern)
// int regex_match (string subject, string pattern)
// int regex_match (string subject, int results[], string pattern)
LLVMGEN (llvm_gen_regex)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    int nargs = op.nargs();
    ASSERT (nargs == 3 || nargs == 4);
    Symbol &Result (*rop.opargsym (op, 0));
    Symbol &Subject (*rop.opargsym (op, 1));
    bool do_match_results = (nargs == 4);
    bool fullmatch = (op.opname() == "regex_match");
    Symbol &Match (*rop.opargsym (op, 2));
    Symbol &Pattern (*rop.opargsym (op, 2+do_match_results));
    ASSERT (Result.typespec().is_int() && Subject.typespec().is_string() &&
            Pattern.typespec().is_string());
    ASSERT (!do_match_results || 
            (Match.typespec().is_array() &&
             Match.typespec().elementtype().is_int()));

    std::vector<llvm::Value*> call_args;
    // First arg is ShaderGlobals ptr
    call_args.push_back (rop.sg_void_ptr());
    // Next arg is subject string
    call_args.push_back (rop.llvm_load_value (Subject));
    // Pass the results array and length (just pass 0 if no results wanted).
    call_args.push_back (rop.llvm_void_ptr(Match));
    if (do_match_results)
        call_args.push_back (rop.llvm_constant(Match.typespec().arraylength()));
    else
        call_args.push_back (rop.llvm_constant(0));
    // Pass the regex match pattern
    call_args.push_back (rop.llvm_load_value (Pattern));
    // Pass whether or not to do the full match
    call_args.push_back (rop.llvm_constant(fullmatch));

    llvm::Value *ret = rop.llvm_call_function ("osl_regex_impl", &call_args[0],
                                               (int)call_args.size());
    rop.llvm_store_value (ret, Result);
    return true;
}



// Generic llvm code generation.  See the comments in llvm_ops.cpp for
// the full list of assumptions and conventions.  But in short:
//   1. All polymorphic and derivative cases implemented as functions in
//      llvm_ops.cpp -- no custom IR is needed.
//   2. Naming conention is: osl_NAME_{args}, where args is the
//      concatenation of type codes for all args including return value --
//      f/i/v/m/s for float/int/triple/matrix/string, and df/dv/dm for
//      duals.
//   3. The function returns scalars as an actual return value (that
//      must be stored), but "returns" aggregates or duals in the first
//      argument.
//   4. Duals and aggregates are passed as void*'s, float/int/string 
//      passed by value.
//   5. Note that this only works if triples are all treated identically,
//      this routine can't be used if it must be polymorphic based on
//      color, point, vector, normal differences.
//
LLVMGEN (llvm_gen_generic)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result  = *rop.opargsym (op, 0);
    std::vector<const Symbol *> args;
    bool any_deriv_args = false;
    for (int i = 0;  i < op.nargs();  ++i) {
        Symbol *s (rop.opargsym (op, i));
        args.push_back (s);
        any_deriv_args |= (i > 0 && s->has_derivs() && !s->typespec().is_matrix());
    }

    // Special cases: functions that have no derivs -- suppress them
    if (any_deriv_args)
        if (op.opname() == op_floor || op.opname() == op_ceil ||
            op.opname() == op_round || op.opname() == op_step ||
            op.opname() == op_trunc || op.opname() == op_cellnoise)
            any_deriv_args = false;

    std::string name = std::string("osl_") + op.opname().string() + "_";
    for (int i = 0;  i < op.nargs();  ++i) {
        Symbol *s (rop.opargsym (op, i));
        if (any_deriv_args && Result.has_derivs() && s->has_derivs() && !s->typespec().is_matrix())
            name += "d";
        if (s->typespec().is_float())
            name += "f";
        else if (s->typespec().is_triple())
            name += "v";
        else if (s->typespec().is_matrix())
            name += "m";
        else if (s->typespec().is_string())
            name += "s";
        else if (s->typespec().is_int())
            name += "i";
        else ASSERT (0);
    }

    if (! Result.has_derivs() || ! any_deriv_args) {
        // Don't compute derivs -- either not needed or not provided in args
        if (Result.typespec().aggregate() == TypeDesc::SCALAR) {
            llvm::Value *r = rop.llvm_call_function (name.c_str(),
                                                     &(args[1]), op.nargs()-1);
            rop.llvm_store_value (r, Result);
        } else {
            rop.llvm_call_function (name.c_str(), &(args[0]), op.nargs());
        }
        rop.llvm_zero_derivs (Result);
    } else {
        // Cases with derivs
        ASSERT (Result.has_derivs() && any_deriv_args);
        rop.llvm_call_function (name.c_str(), &(args[0]), op.nargs(), true);
    }
    return true;
}



LLVMGEN (llvm_gen_sincos)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Theta   = *rop.opargsym (op, 0);
    Symbol& Sin_out = *rop.opargsym (op, 1);
    Symbol& Cos_out = *rop.opargsym (op, 2);
    std::vector<llvm::Value *> valargs;
    bool theta_deriv   = Theta.has_derivs();
    bool result_derivs = (Sin_out.has_derivs() || Cos_out.has_derivs());

    std::string name = std::string("osl_sincos_");
    for (int i = 0;  i < op.nargs();  ++i) {
        Symbol *s (rop.opargsym (op, i));
        if (s->has_derivs() && result_derivs  && theta_deriv)
            name += "d";
        if (s->typespec().is_float())
            name += "f";
        else if (s->typespec().is_triple())
            name += "v";
        else ASSERT (0);
    }
    // push back llvm arguments
    valargs.push_back ( (theta_deriv && result_derivs) || Theta.typespec().is_triple() ? 
          rop.llvm_void_ptr (Theta) : rop.llvm_load_value (Theta));
    valargs.push_back (rop.llvm_void_ptr (Sin_out));
    valargs.push_back (rop.llvm_void_ptr (Cos_out));

    rop.llvm_call_function (name.c_str(), &valargs[0], 3);

    return true;
}

LLVMGEN (llvm_gen_andor)
{
    Opcode& op (rop.inst()->ops()[opnum]);
    Symbol& result = *rop.opargsym (op, 0);
    Symbol& a = *rop.opargsym (op, 1);
    Symbol& b = *rop.opargsym (op, 2);

    llvm::Value* i1_res = NULL;
    llvm::Value* a_val = rop.llvm_load_value (a, 0, 0, TypeDesc::TypeInt);
    llvm::Value* b_val = rop.llvm_load_value (b, 0, 0, TypeDesc::TypeInt);
    if (op.opname() == op_and) {
        // From the old bitcode generated
        // define i32 @osl_and_iii(i32 %a, i32 %b) nounwind readnone ssp {
        //     %1 = icmp ne i32 %b, 0
        //  %not. = icmp ne i32 %a, 0
        //     %2 = and i1 %1, %not.
        //     %3 = zext i1 %2 to i32
        //   ret i32 %3
        llvm::Value* b_ne_0 = rop.builder().CreateICmpNE (b_val, rop.llvm_constant(0));
        llvm::Value* a_ne_0 = rop.builder().CreateICmpNE (a_val, rop.llvm_constant(0));
        llvm::Value* both_ne_0 = rop.builder().CreateAnd (b_ne_0, a_ne_0);
        i1_res = both_ne_0;
    } else {
        // Also from the bitcode
        // %1 = or i32 %b, %a
        // %2 = icmp ne i32 %1, 0
        // %3 = zext i1 %2 to i32
        llvm::Value* or_ab = rop.builder().CreateOr(a_val, b_val);
        llvm::Value* or_ab_ne_0 = rop.builder().CreateICmpNE (or_ab, rop.llvm_constant(0));
        i1_res = or_ab_ne_0;
    }
    llvm::Value* i32_res = rop.builder().CreateZExt(i1_res, rop.llvm_type_int());
    rop.llvm_store_value(i32_res, result, 0, 0);
    return true;
}


LLVMGEN (llvm_gen_if)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& cond = *rop.opargsym (op, 0);

    // Load the condition variable and figure out if it's nonzero
    llvm::Value* cond_val = rop.llvm_load_value (cond, 0, 0, TypeDesc::TypeInt);
    cond_val = rop.builder().CreateICmpNE (cond_val, rop.llvm_constant(0));
    
    // Branch on the condition, to our blocks
    llvm::BasicBlock* then_block = rop.llvm_new_basic_block ("then");
    llvm::BasicBlock* else_block = rop.llvm_new_basic_block ("else");
    llvm::BasicBlock* after_block = rop.llvm_new_basic_block ("");
    rop.builder().CreateCondBr (cond_val, then_block, else_block);

    // Then block
    rop.build_llvm_code (opnum+1, op.jump(0), then_block);
    rop.builder().CreateBr (after_block);

    // Else block
    rop.build_llvm_code (op.jump(0), op.jump(1), else_block);
    rop.builder().CreateBr (after_block);

    // Continue on with the previous flow
    rop.builder().SetInsertPoint (after_block);
    return true;
}



LLVMGEN (llvm_gen_loop_op)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& cond = *rop.opargsym (op, 0);

    // Branch on the condition, to our blocks
    llvm::BasicBlock* cond_block = rop.llvm_new_basic_block ("cond");
    llvm::BasicBlock* body_block = rop.llvm_new_basic_block ("body");
    llvm::BasicBlock* step_block = rop.llvm_new_basic_block ("step");
    llvm::BasicBlock* after_block = rop.llvm_new_basic_block ("");

    // Initialization (will be empty except for "for" loops)
    rop.build_llvm_code (opnum+1, op.jump(0));

    // For "do-while", we go straight to the body of the loop, but for
    // "for" or "while", we test the condition next.
    rop.builder().CreateBr (op.opname() == op_dowhile ? body_block : cond_block);

    // Load the condition variable and figure out if it's nonzero
    rop.build_llvm_code (op.jump(0), op.jump(1), cond_block);
    llvm::Value* cond_val = rop.llvm_load_value (cond, 0, 0, TypeDesc::TypeInt);
    cond_val = rop.builder().CreateICmpNE (cond_val, rop.llvm_constant(0));
    // Jump to either LoopBody or AfterLoop
    rop.builder().CreateCondBr (cond_val, body_block, after_block);

    // Body of loop
    rop.build_llvm_code (op.jump(1), op.jump(2), body_block);
    rop.builder().CreateBr (step_block);

    // Step
    rop.build_llvm_code (op.jump(2), op.jump(3), step_block);
    rop.builder().CreateBr (cond_block);

    // Continue on with the previous flow
    rop.builder().SetInsertPoint (after_block);

    return true;
}




static llvm::Value *
llvm_gen_texture_options (RuntimeOptimizer &rop, int opnum,
                          int first_optional_arg, bool tex3d,
                          llvm::Value* &alpha, llvm::Value* &dalphadx,
                          llvm::Value* &dalphady)
{
    // Reserve space for the TextureOpt, with alignment
    size_t tosize = (sizeof(TextureOpt)+sizeof(char*)-1) / sizeof(char*);
    llvm::Value* opt = rop.builder().CreateAlloca(rop.llvm_type_void_ptr(),
                                                  rop.llvm_constant((int)tosize));
    opt = rop.llvm_void_ptr (opt);
    rop.llvm_call_function ("osl_texture_clear", opt);

    Opcode &op (rop.inst()->ops()[opnum]);
    for (int a = first_optional_arg;  a < op.nargs();  ++a) {
        Symbol &Name (*rop.opargsym(op,a));
        ASSERT (Name.typespec().is_string() &&
                "optional texture token must be a string");
        ASSERT (a+1 < op.nargs() && "malformed argument list for texture");
        ustring name = *(ustring *)Name.data();

        ++a;  // advance to next argument
        Symbol &Val (*rop.opargsym(op,a));
        TypeDesc valtype = Val.typespec().simpletype ();
        
        llvm::Value *val = rop.llvm_load_value (Val);
        if (name == Strings::width && valtype == TypeDesc::FLOAT) {
            rop.llvm_call_function ("osl_texture_set_swidth", opt, val);
            rop.llvm_call_function ("osl_texture_set_twidth", opt, val);
            if (tex3d)
                rop.llvm_call_function ("osl_texture_set_rwidth", opt, val);
        } else if (name == Strings::swidth && valtype == TypeDesc::FLOAT) {
            rop.llvm_call_function ("osl_texture_set_swidth", opt, val);
        } else if (name == Strings::twidth && valtype == TypeDesc::FLOAT) {
            rop.llvm_call_function ("osl_texture_set_twidth", opt, val);
        } else if (name == Strings::rwidth && valtype == TypeDesc::FLOAT) {
            rop.llvm_call_function ("osl_texture_set_rwidth", opt, val);

        } else if (name == Strings::blur && valtype == TypeDesc::FLOAT) {
            rop.llvm_call_function ("osl_texture_set_sblur", opt, val);
            rop.llvm_call_function ("osl_texture_set_tblur", opt, val);
            if (tex3d)
                rop.llvm_call_function ("osl_texture_set_rblur",opt, val);
        } else if (name == Strings::sblur && valtype == TypeDesc::FLOAT) {
            rop.llvm_call_function ("osl_texture_set_sblur", opt, val);
        } else if (name == Strings::tblur && valtype == TypeDesc::FLOAT) {
            rop.llvm_call_function ("osl_texture_set_tblur", opt, val);
        } else if (name == Strings::rblur && valtype == TypeDesc::FLOAT) {
            rop.llvm_call_function ("osl_texture_set_rblur", opt, val);

        } else if (name == Strings::wrap && valtype == TypeDesc::STRING) {
            rop.llvm_call_function ("osl_texture_set_swrap", opt, val);
            rop.llvm_call_function ("osl_texture_set_twrap", opt, val);
            if (tex3d)
                rop.llvm_call_function ("osl_texture_set_rwrap", opt, val);
        } else if (name == Strings::swrap && valtype == TypeDesc::STRING) {
            rop.llvm_call_function ("osl_texture_set_swrap", opt, val);
        } else if (name == Strings::twrap && valtype == TypeDesc::STRING) {
            rop.llvm_call_function ("osl_texture_set_twrap", opt, val);
        } else if (name == Strings::rwrap && valtype == TypeDesc::STRING) {
            rop.llvm_call_function ("osl_texture_set_rwrap", opt, val);

        } else if (name == Strings::firstchannel && valtype == TypeDesc::INT) {
            rop.llvm_call_function ("osl_texture_set_firstchannel", opt, val);
        } else if (name == Strings::fill && valtype == TypeDesc::FLOAT) {
            rop.llvm_call_function ("osl_texture_set_fill", opt, val);
        } else if (name == Strings::time && valtype == TypeDesc::FLOAT) {
            rop.llvm_call_function ("osl_texture_set_time", opt, val);

        } else if (name == Strings::interp && valtype == TypeDesc::STRING) {
            rop.llvm_call_function ("osl_texture_set_interp_name", opt, val);

        } else if (name == Strings::alpha && valtype == TypeDesc::FLOAT) {
            alpha = rop.llvm_get_pointer (Val);
            if (Val.has_derivs()) {
                dalphadx = rop.llvm_get_pointer (Val, 1);
                dalphady = rop.llvm_get_pointer (Val, 2);
                // NO z derivs!  dalphadz = rop.llvm_get_pointer (Val, 3);
            }
        } else {
            rop.shadingsys().error ("Unknown texture%s optional argument: \"%s\", <%s> (%s:%d)",
                                    tex3d ? "3d" : "",
                                    name.c_str(), valtype.c_str(),
                                    op.sourcefile().c_str(), op.sourceline());
        }
    }

    return opt;
}



LLVMGEN (llvm_gen_texture)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &Result = *rop.opargsym (op, 0);
    Symbol &Filename = *rop.opargsym (op, 1);
    Symbol &S = *rop.opargsym (op, 2);
    Symbol &T = *rop.opargsym (op, 3);

    bool user_derivs = false;
    int first_optional_arg = 4;
    if (op.nargs() > 4 && rop.opargsym(op,4)->typespec().is_float()) {
        user_derivs = true;
        first_optional_arg = 8;
        DASSERT (rop.opargsym(op,5)->typespec().is_float());
        DASSERT (rop.opargsym(op,6)->typespec().is_float());
        DASSERT (rop.opargsym(op,7)->typespec().is_float());
    }

    llvm::Value* opt;   // TextureOpt
    llvm::Value *alpha = NULL, *dalphadx = NULL, *dalphady = NULL;
    opt = llvm_gen_texture_options (rop, opnum, first_optional_arg,
                                    false /*3d*/, alpha, dalphadx, dalphady);

    // Now call the osl_texture function, passing the options and all the
    // explicit args like texture coordinates.
    std::vector<llvm::Value *> args;
    args.push_back (rop.sg_void_ptr());
    args.push_back (rop.llvm_load_value (Filename));
    args.push_back (opt);
    args.push_back (rop.llvm_load_value (S));
    args.push_back (rop.llvm_load_value (T));
    if (user_derivs) {
        args.push_back (rop.llvm_load_value (*rop.opargsym (op, 4)));
        args.push_back (rop.llvm_load_value (*rop.opargsym (op, 5)));
        args.push_back (rop.llvm_load_value (*rop.opargsym (op, 6)));
        args.push_back (rop.llvm_load_value (*rop.opargsym (op, 7)));
    } else {
        // Auto derivs of S and T
        args.push_back (rop.llvm_load_value (S, 1));
        args.push_back (rop.llvm_load_value (T, 1));
        args.push_back (rop.llvm_load_value (S, 2));
        args.push_back (rop.llvm_load_value (T, 2));
    }
    args.push_back (rop.llvm_constant ((int)Result.typespec().aggregate()));
    args.push_back (rop.llvm_void_ptr (rop.llvm_get_pointer (Result, 0)));
    args.push_back (rop.llvm_void_ptr (rop.llvm_get_pointer (Result, 1)));
    args.push_back (rop.llvm_void_ptr (rop.llvm_get_pointer (Result, 2)));
    if (alpha) {
        args.push_back (rop.llvm_void_ptr (alpha));
        args.push_back (rop.llvm_void_ptr (dalphadx ? dalphadx : rop.llvm_void_ptr_null()));
        args.push_back (rop.llvm_void_ptr (dalphady ? dalphady : rop.llvm_void_ptr_null()));
        rop.llvm_call_function ("osl_texture_alpha", &args[0], (int)args.size());
    } else {
        rop.llvm_call_function ("osl_texture", &args[0], (int)args.size());
    }
    return true;
}



LLVMGEN (llvm_gen_texture3d)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &Result = *rop.opargsym (op, 0);
    Symbol &Filename = *rop.opargsym (op, 1);
    Symbol &P = *rop.opargsym (op, 2);

    bool user_derivs = false;
    int first_optional_arg = 4;
    if (op.nargs() > 3 && rop.opargsym(op,3)->typespec().is_triple()) {
        user_derivs = true;
        first_optional_arg = 6;
        DASSERT (rop.opargsym(op,4)->typespec().is_triple());
        DASSERT (rop.opargsym(op,5)->typespec().is_triple());
    }

    llvm::Value* opt;   // TextureOpt
    llvm::Value *alpha = NULL, *dalphadx = NULL, *dalphady = NULL;
    opt = llvm_gen_texture_options (rop, opnum, first_optional_arg,
                                    true /*3d*/, alpha, dalphadx, dalphady);

    // Now call the osl_texture3d function, passing the options and all the
    // explicit args like texture coordinates.
    std::vector<llvm::Value *> args;
    args.push_back (rop.sg_void_ptr());
    args.push_back (rop.llvm_load_value (Filename));
    args.push_back (opt);
    args.push_back (rop.llvm_void_ptr (P));
    if (user_derivs) {
        args.push_back (rop.llvm_void_ptr (*rop.opargsym (op, 3)));
        args.push_back (rop.llvm_void_ptr (*rop.opargsym (op, 4)));
        args.push_back (rop.llvm_void_ptr (*rop.opargsym (op, 5)));
    } else {
        // Auto derivs of P
        args.push_back (rop.llvm_void_ptr (P, 1));
        args.push_back (rop.llvm_void_ptr (P, 2));
        // zero for dPdz, for now
        llvm::Value *fzero = rop.llvm_constant (0.0f);
        llvm::Value *vzero = rop.builder().CreateAlloca (rop.llvm_type_triple(),
                                                     rop.llvm_constant((int)1));
        for (int i = 0;  i < 3;  ++i)
            rop.builder().CreateStore (fzero, rop.builder().CreateConstGEP2_32 (vzero, 0, i));
        args.push_back (rop.llvm_void_ptr(vzero));
    }
    args.push_back (rop.llvm_constant ((int)Result.typespec().aggregate()));
    args.push_back (rop.llvm_void_ptr (rop.llvm_void_ptr (Result, 0)));
    args.push_back (rop.llvm_void_ptr (rop.llvm_void_ptr (Result, 1)));
    args.push_back (rop.llvm_void_ptr (rop.llvm_void_ptr (Result, 2)));
    args.push_back (rop.llvm_void_ptr_null());  // no dresultdz for now
    if (alpha) {
        args.push_back (rop.llvm_void_ptr (alpha));
        args.push_back (dalphadx ? rop.llvm_void_ptr (dalphadx) : rop.llvm_void_ptr_null());
        args.push_back (dalphady ? rop.llvm_void_ptr (dalphady) : rop.llvm_void_ptr_null());
        args.push_back (rop.llvm_void_ptr_null());  // No dalphadz for now
        rop.llvm_call_function ("osl_texture3d_alpha", &args[0], (int)args.size());
    } else {
        rop.llvm_call_function ("osl_texture3d", &args[0], (int)args.size());
    }
    return true;
}



LLVMGEN (llvm_gen_environment)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &Result = *rop.opargsym (op, 0);
    Symbol &Filename = *rop.opargsym (op, 1);
    Symbol &R = *rop.opargsym (op, 2);

    bool user_derivs = false;
    int first_optional_arg = 3;
    if (op.nargs() > 3 && rop.opargsym(op,3)->typespec().is_triple()) {
        user_derivs = true;
        first_optional_arg = 5;
        DASSERT (rop.opargsym(op,4)->typespec().is_triple());
    }

    llvm::Value* opt;   // TextureOpt
    llvm::Value *alpha = NULL, *dalphadx = NULL, *dalphady = NULL;
    opt = llvm_gen_texture_options (rop, opnum, first_optional_arg,
                                    false /*3d*/, alpha, dalphadx, dalphady);

    // Now call the osl_environment function, passing the options and all the
    // explicit args like texture coordinates.
    std::vector<llvm::Value *> args;
    args.push_back (rop.sg_void_ptr());
    args.push_back (rop.llvm_load_value (Filename));
    args.push_back (opt);
    args.push_back (rop.llvm_void_ptr (R));
    if (user_derivs) {
        args.push_back (rop.llvm_void_ptr (*rop.opargsym (op, 3)));
        args.push_back (rop.llvm_void_ptr (*rop.opargsym (op, 4)));
    } else {
        // Auto derivs of R
        args.push_back (rop.llvm_void_ptr (R, 1));
        args.push_back (rop.llvm_void_ptr (R, 2));
    }
    args.push_back (rop.llvm_constant ((int)Result.typespec().aggregate()));
    args.push_back (rop.llvm_void_ptr (Result, 0));
    args.push_back (rop.llvm_void_ptr (Result, 1));
    args.push_back (rop.llvm_void_ptr (Result, 2));
    if (alpha) {
        args.push_back (rop.llvm_void_ptr (alpha));
        args.push_back (dalphadx ? rop.llvm_void_ptr (dalphadx) : rop.llvm_void_ptr_null());
        args.push_back (dalphady ? rop.llvm_void_ptr (dalphady) : rop.llvm_void_ptr_null());
    } else {
        args.push_back (rop.llvm_void_ptr_null());
        args.push_back (rop.llvm_void_ptr_null());
        args.push_back (rop.llvm_void_ptr_null());
    }
    rop.llvm_call_function ("osl_environment", &args[0], (int)args.size());
    return true;
}



static llvm::Value *
llvm_gen_trace_options (RuntimeOptimizer &rop, int opnum,
                        int first_optional_arg)
{
    // Reserve space for the TraceOpt, with alignment
    size_t tosize = (sizeof(RendererServices::TraceOpt)+sizeof(char*)-1) / sizeof(char*);
    llvm::Value* opt = rop.builder().CreateAlloca(rop.llvm_type_void_ptr(),
                                                  rop.llvm_constant((int)tosize));
    opt = rop.llvm_void_ptr (opt);
    rop.llvm_call_function ("osl_trace_clear", opt);

    Opcode &op (rop.inst()->ops()[opnum]);
    for (int a = first_optional_arg;  a < op.nargs();  ++a) {
        Symbol &Name (*rop.opargsym(op,a));
        ASSERT (Name.typespec().is_string() &&
                "optional trace token must be a string");
        ASSERT (a+1 < op.nargs() && "malformed argument list for trace");
        ustring name = *(ustring *)Name.data();

        ++a;  // advance to next argument
        Symbol &Val (*rop.opargsym(op,a));
        TypeDesc valtype = Val.typespec().simpletype ();
        
        llvm::Value *val = rop.llvm_load_value (Val);
        static ustring kmindist("mindist"), kmaxdist("maxdist");
        static ustring kshade("shade");
        if (name == kmindist && valtype == TypeDesc::FLOAT) {
            rop.llvm_call_function ("osl_trace_set_mindist", opt, val);
        } else if (name == kmaxdist && valtype == TypeDesc::FLOAT) {
            rop.llvm_call_function ("osl_trace_set_maxdist", opt, val);
        } else if (name == kshade && valtype == TypeDesc::INT) {
            rop.llvm_call_function ("osl_trace_set_shade", opt, val);
        } else {
            rop.shadingsys().error ("Unknown trace() optional argument: \"%s\", <%s> (%s:%d)",
                                    name.c_str(), valtype.c_str(),
                                    op.sourcefile().c_str(), op.sourceline());
        }
    }

    return opt;
}



LLVMGEN (llvm_gen_trace)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &Result = *rop.opargsym (op, 0);
    Symbol &Pos = *rop.opargsym (op, 1);
    Symbol &Dir = *rop.opargsym (op, 2);
    int first_optional_arg = 3;

    llvm::Value* opt;   // TraceOpt
    opt = llvm_gen_trace_options (rop, opnum, first_optional_arg);

    // Now call the osl_trace function, passing the options and all the
    // explicit args like trace coordinates.
    std::vector<llvm::Value *> args;
    args.push_back (rop.sg_void_ptr());
    args.push_back (opt);
    args.push_back (rop.llvm_void_ptr (Pos, 0));
    args.push_back (rop.llvm_void_ptr (Pos, 1));
    args.push_back (rop.llvm_void_ptr (Pos, 2));
    args.push_back (rop.llvm_void_ptr (Dir, 0));
    args.push_back (rop.llvm_void_ptr (Dir, 1));
    args.push_back (rop.llvm_void_ptr (Dir, 2));
    llvm::Value *r = rop.llvm_call_function ("osl_trace", &args[0],
                                             (int)args.size());
    rop.llvm_store_value (r, Result);
    return true;
}



// pnoise and psnoise -- we can't use llvm_gen_generic because of the
// special case that the periods should never pass derivatives.
LLVMGEN (llvm_gen_pnoise)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    // N.B. we don't use the derivatives of periods.  There are as many
    // period arguments as position arguments, and argument 0 is the
    // result.  So f=pnoise(f,f) => firstperiod = 2; f=pnoise(v,f,v,f)
    // => firstperiod = 3.
    int firstperiod = (op.nargs() - 1) / 2 + 1;

    Symbol& Result  = *rop.opargsym (op, 0);
    bool any_deriv_args = false;
    for (int i = 0;  i < op.nargs();  ++i) {
        Symbol *s (rop.opargsym (op, i));
        any_deriv_args |= (i > 0 && i < firstperiod &&
                           s->has_derivs() && !s->typespec().is_matrix());
    }

    std::string name = std::string("osl_") + op.opname().string() + "_";
    std::vector<llvm::Value *> valargs;
    valargs.resize (op.nargs());
    for (int i = 0;  i < op.nargs();  ++i) {
        Symbol *s (rop.opargsym (op, i));
        bool use_derivs = any_deriv_args && i < firstperiod && Result.has_derivs() && s->has_derivs() && !s->typespec().is_matrix();
        if (use_derivs)
            name += "d";
        if (s->typespec().is_float())
            name += "f";
        else if (s->typespec().is_triple())
            name += "v";
        else ASSERT (0);


        if (s->typespec().simpletype().aggregate > 1 || use_derivs)
            valargs[i] = rop.llvm_void_ptr (*s);
        else
            valargs[i] = rop.llvm_load_value (*s);
    }

    if (! Result.has_derivs() || ! any_deriv_args) {
        // Don't compute derivs -- either not needed or not provided in args
        if (Result.typespec().aggregate() == TypeDesc::SCALAR) {
            llvm::Value *r = rop.llvm_call_function (name.c_str(), &valargs[1], op.nargs()-1);
            rop.llvm_store_value (r, Result);
        } else {
            rop.llvm_call_function (name.c_str(), &valargs[0], op.nargs());
        }
        rop.llvm_zero_derivs (Result);
    } else {
        // Cases with derivs
        ASSERT (Result.has_derivs() && any_deriv_args);
        rop.llvm_call_function (name.c_str(), &valargs[0], op.nargs());
    }
    return true;
}



LLVMGEN (llvm_gen_getattribute)
{
    // getattribute() has eight "flavors":
    //   * getattribute (attribute_name, value)
    //   * getattribute (attribute_name, value[])
    //   * getattribute (attribute_name, index, value)
    //   * getattribute (attribute_name, index, value[])
    //   * getattribute (object, attribute_name, value)
    //   * getattribute (object, attribute_name, value[])
    //   * getattribute (object, attribute_name, index, value)
    //   * getattribute (object, attribute_name, index, value[])
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() >= 3 && op.nargs() <= 5);

    bool object_lookup = false;
    bool array_lookup  = false;

    // slot indices when (nargs==3)
    int result_slot = 0; // never changes
    int attrib_slot = 1;
    int object_slot = 0; // initially not used
    int index_slot  = 0; // initially not used
    int dest_slot   = 2;

    // figure out which "flavor" of getattribute() to use
    if (op.nargs() == 5) {
        object_slot = 1;
        attrib_slot = 2;
        index_slot  = 3;
        dest_slot   = 4;
        array_lookup  = true;
        object_lookup = true;
    }
    else if (op.nargs() == 4) {
        if (rop.opargsym (op, 2)->typespec().is_int()) {
            attrib_slot = 1;
            index_slot  = 2;
            dest_slot   = 3;
            array_lookup = true;
        }
        else {
            object_slot = 1;
            attrib_slot = 2;
            dest_slot   = 3;
            object_lookup = true;
        }
    }

    Symbol& Result      = *rop.opargsym (op, result_slot);
    Symbol& ObjectName  = *rop.opargsym (op, object_slot); // might be aliased to Result
    Symbol& Index       = *rop.opargsym (op, index_slot);  // might be aliased to Result
    Symbol& Attribute   = *rop.opargsym (op, attrib_slot);
    Symbol& Destination = *rop.opargsym (op, dest_slot);

    TypeDesc attribute_type = Destination.typespec().simpletype();
    bool     dest_derivs    = Destination.has_derivs();

    DASSERT (!Result.typespec().is_closure()    && !ObjectName.typespec().is_closure() && 
             !Attribute.typespec().is_closure() && !Index.typespec().is_closure()      && 
             !Destination.typespec().is_closure());

    // We'll pass the destination's attribute type directly to the 
    // RenderServices callback so that the renderer can perform any
    // necessary conversions from its internal format to OSL's.
    const TypeDesc* dest_type = &Destination.typespec().simpletype();

    std::vector<llvm::Value *> args;
    args.push_back (rop.sg_void_ptr());
    args.push_back (rop.llvm_constant ((int)dest_derivs));
    args.push_back (object_lookup ? rop.llvm_load_value (ObjectName) :
                                    rop.llvm_constant (ustring()));
    args.push_back (rop.llvm_load_value (Attribute));
    args.push_back (rop.llvm_constant ((int)array_lookup));
    args.push_back (rop.llvm_load_value (Index));
    args.push_back (rop.llvm_constant_ptr ((void *) dest_type));
    args.push_back (rop.llvm_void_ptr (Destination));

    llvm::Value *r = rop.llvm_call_function ("osl_get_attribute", &args[0], args.size());
    rop.llvm_store_value (r, Result);

    return true;
}



void
RuntimeOptimizer::llvm_assign_initial_value (const Symbol& sym)
{
    // Don't write over connections!  Connection values are written into
    // our layer when the earlier layer is run, as part of its code.  So
    // we just don't need to initialize it here at all.
    if (sym.valuesource() == Symbol::ConnectedVal &&
          !sym.typespec().is_closure())
        return;
    if (sym.typespec().is_closure() && sym.symtype() == SymTypeGlobal)
        return;

    int arraylen = std::max (1, sym.typespec().arraylength());

    // Closures need to get their storage before anything can be
    // assigned to them.  Unless they are params, in which case we took
    // care of it in the group entry point.
    if (sym.typespec().is_closure() &&
        sym.symtype() != SymTypeParam && sym.symtype() != SymTypeOutputParam) {
        llvm::Value *init_val = llvm_constant_ptr(NULL, llvm_type_void_ptr());
        for (int a = 0; a < arraylen;  ++a) {
            llvm::Value *arrind = sym.typespec().is_array() ? llvm_constant(a) : NULL;
            llvm_store_value (init_val, sym, 0, arrind, 0);
        }
    }

    if (sym.has_init_ops() && sym.valuesource() == Symbol::DefaultVal) {
        // Handle init ops.
        build_llvm_code (sym.initbegin(), sym.initend());
    } else {
        // Use default value
        int num_components = sym.typespec().simpletype().aggregate;
        for (int a = 0, c = 0; a < arraylen;  ++a) {
            llvm::Value *arrind = sym.typespec().is_array() ? llvm_constant(a) : NULL;
            if (sym.typespec().is_closure())
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
        args.push_back (llvm_void_ptr (llvm_get_pointer (sym)));
        llvm_call_function ("osl_bind_interpolated_param",
                            &args[0], args.size());                            
    }
}



llvm::Value *
RuntimeOptimizer::llvm_offset_ptr (llvm::Value *ptr, int offset,
                                   const llvm::Type *ptrtype)
{
    llvm::Value *i = builder().CreatePtrToInt (ptr, llvm_type_addrint());
    i = builder().CreateAdd (i, llvm_constant ((size_t)offset));
    ptr = builder().CreateIntToPtr (i, llvm_type_void_ptr());
    if (ptrtype)
        ptr = llvm_ptr_cast (ptr, ptrtype);
    return ptr;
}



LLVMGEN (llvm_gen_gettextureinfo)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() == 4);

    Symbol& Result   = *rop.opargsym (op, 0);
    Symbol& Filename = *rop.opargsym (op, 1);
    Symbol& Dataname = *rop.opargsym (op, 2);
    Symbol& Data     = *rop.opargsym (op, 3);

    DASSERT (!Result.typespec().is_closure() && Filename.typespec().is_string() && 
             Dataname.typespec().is_string() && !Data.typespec().is_closure()   && 
             Result.typespec().is_int());

    std::vector<llvm::Value *> args;

    args.push_back (rop.sg_void_ptr());
    args.push_back (rop.llvm_load_value (Filename));
    args.push_back (rop.llvm_load_value (Dataname));
    // this is passes a TypeDesc to an LLVM op-code
    args.push_back (rop.llvm_constant((int) Data.typespec().simpletype().basetype));
    args.push_back (rop.llvm_constant((int) Data.typespec().simpletype().arraylen));
    args.push_back (rop.llvm_constant((int) Data.typespec().simpletype().aggregate));
    // destination
    args.push_back (rop.llvm_void_ptr (Data));

    llvm::Value *r = rop.llvm_call_function ("osl_get_textureinfo", &args[0], args.size());
    rop.llvm_store_value (r, Result);

    return true;
}



LLVMGEN (llvm_gen_getmessage)
{
    // getmessage() has four "flavors":
    //   * getmessage (attribute_name, value)
    //   * getmessage (attribute_name, value[])
    //   * getmessage (source, attribute_name, value)
    //   * getmessage (source, attribute_name, value[])
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() == 3 || op.nargs() == 4);
    int has_source = (op.nargs() == 4);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Source = *rop.opargsym (op, 1);
    Symbol& Name   = *rop.opargsym (op, 1+has_source);
    Symbol& Data   = *rop.opargsym (op, 2+has_source);
    DASSERT (Result.typespec().is_int() && Name.typespec().is_string());
    DASSERT (has_source == 0 || Source.typespec().is_string());

    llvm::Value *args[5];
    args[0] = rop.sg_void_ptr();
    args[1] = has_source ? rop.llvm_load_value(Source) 
                         : rop.llvm_constant(ustring());
    args[2] = rop.llvm_load_value (Name);
    args[3] = rop.llvm_constant (Data.typespec().simpletype());
    if (Data.typespec().is_closure())
        // We need a void ** here so the function can modify the closure
        args[4] = rop.llvm_ptr_cast(rop.llvm_get_pointer(Data), rop.llvm_type_void_ptr());
    else
        args[4] = rop.llvm_void_ptr (Data);

    llvm::Value *r = rop.llvm_call_function ("osl_getmessage", args, 5);
    rop.llvm_store_value (r, Result);
    return true;
}



LLVMGEN (llvm_gen_setmessage)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() == 2);
    Symbol& Name   = *rop.opargsym (op, 0);
    Symbol& Data   = *rop.opargsym (op, 1);
    DASSERT (Name.typespec().is_string());

    llvm::Value *args[4];
    args[0] = rop.sg_void_ptr();
    args[1] = rop.llvm_load_value (Name);
    args[2] = rop.llvm_constant (Data.typespec().simpletype());
    args[3] = rop.llvm_void_ptr (Data);

    rop.llvm_call_function ("osl_setmessage", args, 4);
    return true;
}



LLVMGEN (llvm_gen_get_simple_SG_field)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() == 1);

    Symbol& Result = *rop.opargsym (op, 0);
    int sg_index = ShaderGlobalNameToIndex (op.opname());
    ASSERT (sg_index >= 0);
    llvm::Value *sg_field = rop.builder().CreateConstGEP2_32 (rop.sg_ptr(), 0, sg_index);
    llvm::Value* r = rop.builder().CreateLoad(sg_field);
    rop.llvm_store_value (r, Result);

    return true;
}



LLVMGEN (llvm_gen_calculatenormal)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() == 2);

    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& P      = *rop.opargsym (op, 1);

    DASSERT (Result.typespec().is_triple() && P.typespec().is_triple());
    if (! P.has_derivs()) {
        rop.llvm_assign_zero (Result);
        return true;
    }
    
    std::vector<llvm::Value *> args;
    args.push_back (rop.llvm_void_ptr (Result));
    args.push_back (rop.sg_void_ptr());
    args.push_back (rop.llvm_void_ptr (P));
    rop.llvm_call_function ("osl_calculatenormal", &args[0], args.size());
    if (Result.has_derivs())
        rop.llvm_zero_derivs (Result);
    return true;
}



LLVMGEN (llvm_gen_spline)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() >= 4 && op.nargs() <= 5);

    bool has_knot_count = (op.nargs() == 5);
    Symbol& Result   = *rop.opargsym (op, 0);
    Symbol& Spline   = *rop.opargsym (op, 1);
    Symbol& Value    = *rop.opargsym (op, 2);
    Symbol& Knot_count = *rop.opargsym (op, 3); // might alias Knots
    Symbol& Knots    = has_knot_count ? *rop.opargsym (op, 4) :
                                        *rop.opargsym (op, 3);

    DASSERT (!Result.typespec().is_closure() && Spline.typespec().is_string()  && 
             Value.typespec().is_float()     && !Knots.typespec().is_closure() &&
             Knots.typespec().is_array()     &&  
             (!has_knot_count || (has_knot_count && Knot_count.typespec().is_int())));

    std::string name = "osl_spline_";
    std::vector<llvm::Value *> args;
    // only use derivatives for result if:
    //   result has derivs and (value || knots) have derivs
    bool result_derivs = Result.has_derivs() && (Value.has_derivs() || Knots.has_derivs());

    if (result_derivs)
        name += "d";
    if (Result.typespec().is_float())
        name += "f";
    else if (Result.typespec().is_triple())
        name += "v";

    if (result_derivs && Value.has_derivs())
        name += "d";
    if (Value.typespec().is_float())
        name += "f";
    else if (Value.typespec().is_triple())
        name += "v";

    if (result_derivs && Knots.has_derivs())
        name += "d";
    if (Knots.typespec().simpletype().elementtype() == TypeDesc::FLOAT)
        name += "f";
    else if (Knots.typespec().simpletype().elementtype().aggregate == TypeDesc::VEC3)
        name += "v";

    args.push_back (rop.llvm_void_ptr (Result));
    args.push_back (rop.llvm_load_value (Spline));
    args.push_back (rop.llvm_void_ptr (Value)); // make things easy
    args.push_back (rop.llvm_void_ptr (Knots));
    if (has_knot_count)
        args.push_back (rop.llvm_load_value (Knot_count));
    else
        args.push_back (rop.llvm_constant ((int)Knots.typespec().arraylength()));
    rop.llvm_call_function (name.c_str(), &args[0], args.size());

    if (Result.has_derivs() && !result_derivs)
        rop.llvm_zero_derivs (Result);

    return true;
}



static void
llvm_gen_keyword_fill(RuntimeOptimizer &rop, Opcode &op, const ClosureRegistry::ClosureEntry *clentry, ustring clname, llvm::Value *attr_p, int argsoffset)
{
    DASSERT(((op.nargs() - argsoffset) % 2) == 0);

    int Nattrs = (op.nargs() - argsoffset) / 2;

    for (int attr_i = 0; attr_i < Nattrs; ++attr_i) {
        int argno = attr_i * 2 + argsoffset;;
        Symbol &Key     = *rop.opargsym (op, argno);
        Symbol &Value   = *rop.opargsym (op, argno + 1);
        ASSERT(Key.typespec().is_string());
        ASSERT(Key.is_constant());
        ustring *key = (ustring *)Key.data();
        TypeDesc ValueType = Value.typespec().simpletype();

        bool legal = false;
        // Make sure there is some keyword arg that has the name and the type
        for (int t = 0; t < clentry->nkeyword; ++t) {
            const ClosureParam &param = clentry->params[clentry->nformal + t];
            // strcmp might be too much, we could precompute the ustring for the param,
            // but in this part of the code is not a big deal
            if (param.type == ValueType && !strcmp(key->c_str(), param.key))
                legal = true;
        }
        if (!legal) {
            rop.shadingsys().warning("Unsupported closure keyword arg \"%s\" for %s (%s:%d)", key->c_str(), clname.c_str(), op.sourcefile().c_str(), op.sourceline());
            continue;
        }

        llvm::Value *key_to     = rop.builder().CreateConstGEP2_32 (attr_p, attr_i, 0);
        llvm::Value *key_const  = rop.llvm_constant_ptr(*((void **)key), rop.llvm_type_string());
        llvm::Value *value_to   = rop.builder().CreateConstGEP2_32 (attr_p, attr_i, 1);
        llvm::Value *value_from = rop.llvm_void_ptr (Value);
        value_to = rop.llvm_ptr_cast (value_to, rop.llvm_type_void_ptr());

        rop.builder().CreateStore (key_const, key_to);
        rop.llvm_memcpy (value_to, value_from, (int)ValueType.size(), 4);
    }
}



LLVMGEN (llvm_gen_closure)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    ASSERT (op.nargs() >= 2); // at least the result and the ID

    Symbol &Result   = *rop.opargsym (op, 0);
    Symbol &Id       = *rop.opargsym (op, 1);
    DASSERT(Result.typespec().is_closure());
    DASSERT(Id.typespec().is_string());
    ustring closure_name = *((ustring *)Id.data());

    const ClosureRegistry::ClosureEntry * clentry = rop.shadingsys().find_closure(closure_name);
    if (!clentry) {
        rop.shadingsys().error ("Closure '%s' is not supported by the current renderer, called from (%s:%d)",
                                closure_name.c_str(), op.sourcefile().c_str(), op.sourceline());
        return false;
    }

    ASSERT (op.nargs() >= (2 + clentry->nformal));
    int nattrs = (op.nargs() - (2 + clentry->nformal)) / 2;

    // Call osl_allocate_closure_component(closure, id, size).  It returns
    // the memory for the closure parameter data.
    llvm::Value *render_ptr = rop.llvm_constant_ptr(rop.shadingsys().renderer(), rop.llvm_type_void_ptr());
    llvm::Value *sg_ptr = rop.sg_void_ptr();
    llvm::Value *id_int = rop.llvm_constant(clentry->id);
    llvm::Value *size_int = rop.llvm_constant(clentry->struct_size);
    llvm::Value *nattrs_int = rop.llvm_constant(nattrs);
    llvm::Value *alloc_args[4] = {sg_ptr, id_int, size_int, nattrs_int};
    llvm::Value *comp_void_ptr = rop.llvm_call_function ("osl_allocate_closure_component", alloc_args, 4);
    rop.llvm_store_value (comp_void_ptr, Result, 0, NULL, 0);
    llvm::Value *comp_ptr = rop.llvm_ptr_cast(comp_void_ptr, rop.llvm_type_closure_component_ptr());
    // Get the address of the primitive buffer, which is the 5th field
    llvm::Value *mem_void_ptr = rop.builder().CreateConstGEP2_32 (comp_ptr, 0, 4);
    mem_void_ptr = rop.llvm_ptr_cast(mem_void_ptr, rop.llvm_type_void_ptr());

    // If the closure has a "prepare" method, call
    // prepare(renderer, id, memptr).  If there is no prepare method, just
    // zero out the closure parameter memory.
    if (clentry->prepare) {
        // Call clentry->prepare(renderservices *, int id, void *mem)
        llvm::Value *funct_ptr = rop.llvm_constant_ptr((void *)clentry->prepare, rop.llvm_type_prepare_closure_func());
        llvm::Value *args[3] = {render_ptr, id_int, mem_void_ptr};
        rop.builder().CreateCall (funct_ptr, args, args+3);
    } else {
        rop.llvm_memset (mem_void_ptr, 0, clentry->struct_size, 4 /*align*/);
    }

    // Here is where we fill the struct using the params
    for (int carg = 0; carg < clentry->nformal; ++carg) {
        const ClosureParam &p = clentry->params[carg];
        if (p.key != NULL) break;
        ASSERT(p.offset < clentry->struct_size);
        Symbol &sym = *rop.opargsym (op, carg + 2);
        TypeDesc t = sym.typespec().simpletype();
        if (t.vecsemantics == TypeDesc::NORMAL || t.vecsemantics == TypeDesc::POINT)
            t.vecsemantics = TypeDesc::VECTOR;
        if (!sym.typespec().is_closure() && !sym.typespec().is_structure() && t == p.type) {
            llvm::Value* dst = rop.llvm_offset_ptr (mem_void_ptr, p.offset);
            llvm::Value* src = rop.llvm_void_ptr (sym);
            rop.llvm_memcpy (dst, src, (int)p.type.size(),
                             4 /* use 4 byte alignment for now */);
        } else {
            rop.shadingsys().error ("Incompatible formal argument %d to '%s' closure. Prototypes don't match renderer registry.",
                                    carg + 1, closure_name.c_str());
        }
    }

    // If the closure has a "setup" method, call
    // setup(render_services, id, mem_ptr).
    if (clentry->setup) {
        // Call clentry->setup(renderservices *, int id, void *mem)
        llvm::Value *funct_ptr = rop.llvm_constant_ptr((void *)clentry->setup, rop.llvm_type_setup_closure_func());
        llvm::Value *args[3] = {render_ptr, id_int, mem_void_ptr};
        rop.builder().CreateCall (funct_ptr, args, args+3);
    }

    llvm::Value *attrs_void_ptr = rop.llvm_offset_ptr (mem_void_ptr, clentry->struct_size);
    llvm::Value *attrs_ptr = rop.llvm_ptr_cast(attrs_void_ptr, rop.llvm_type_closure_component_attr_ptr());
    llvm_gen_keyword_fill(rop, op, clentry, closure_name, attrs_ptr, clentry->nformal + 2);

    return true;
}



LLVMGEN (llvm_gen_pointcloud_search)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    DASSERT (op.nargs() >= 5);
    // Does the compiler check this? Can we turn it
    // into a DASSERT?
    ASSERT (((op.nargs() - 5) % 2) == 0);

    Symbol& Result     = *rop.opargsym (op, 0);
    Symbol& Filename   = *rop.opargsym (op, 1);
    Symbol& Center     = *rop.opargsym (op, 2);
    Symbol& Radius     = *rop.opargsym (op, 3);
    Symbol& Max_points = *rop.opargsym (op, 4);

    DASSERT (Result.typespec().is_int() && Filename.typespec().is_string() &&
             Center.typespec().is_triple() && Radius.typespec().is_float() &&
             Max_points.typespec().is_int());

    std::vector<llvm::Value *> args;
    args.push_back (rop.sg_void_ptr());
    args.push_back (rop.llvm_load_value (Filename));
    args.push_back (rop.llvm_void_ptr   (Center));
    args.push_back (rop.llvm_load_value (Radius));
    args.push_back (rop.llvm_load_value (Max_points));
    // We will create a query later and it needs to be passed
    // here, so make space for another argument and save the pos
    int query_pos = args.size();
    args.push_back (NULL); // attr_query place holder

    // Noew parse the "string", value pairs that the caller uses
    // to give us the output arrays where we have to put the attributes
    // for the found points
    std::vector<ustring>  attr_names;
    std::vector<TypeDesc> attr_types;

    int attr_arg_offset = 5; // where the ot attrs begin
    int nattrs = (op.nargs() - 5) / 2;
    // pass the number of attributes before the
    // var arg list
    args.push_back (rop.llvm_constant(nattrs));

    for (int i = 0; i < nattrs; ++i)
    {
        Symbol& Name  = *rop.opargsym (op, attr_arg_offset + i*2);
        Symbol& Value = *rop.opargsym (op, attr_arg_offset + i*2 + 1);
        // The names of the attribute has to be a string and a constant.
        // We don't allow runtine generated attributes because the
        // queries have to be pre-baked
        ASSERT (Name.typespec().is_string());
        ASSERT (Name.is_constant());
        ustring *name = (ustring *)Name.data();
        // We save this to generate the query object later, both name
        // and type will never change during the render
        attr_names.push_back (*name);
        attr_types.push_back (Value.typespec().simpletype());
        // And now pass the actual pointer to the data
        args.push_back (rop.llvm_void_ptr (Value));
    }

    // Try to build a query and get the handle from the renderer
    void *attr_query = rop.shadingsys().renderer()->get_pointcloud_attr_query (&attr_names[0], &attr_types[0], attr_names.size());
    if (!attr_query)
    {
        rop.shadingsys().error ("Failed to create pointcloud query at (%s:%d)",
                                 op.sourcefile().c_str(), op.sourceline());
        return false;
    }
    // Every pointcloud call that appears in the code gets its own query object.
    // Not a big waste, and it can be used until the end of the render. It is a
    // constant handle that we put in the arguments for the renderer.
    args[query_pos] = rop.llvm_constant_ptr(attr_query, rop.llvm_type_void_ptr());

    llvm::Value *ret = rop.llvm_call_function ("osl_pointcloud", &args[0], args.size());
    // Return the number of results
    rop.llvm_store_value (ret, Result);
    return true;
}



LLVMGEN (llvm_gen_dict_find)
{
    // OSL has two variants of this function:
    //     dict_find (string dict, string query)
    //     dict_find (int nodeID, string query)
    Opcode &op (rop.inst()->ops()[opnum]);
    DASSERT (op.nargs() == 3);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Source = *rop.opargsym (op, 1);
    Symbol& Query  = *rop.opargsym (op, 2);
    DASSERT (Result.typespec().is_int() && Query.typespec().is_string() &&
             (Source.typespec().is_int() || Source.typespec().is_string()));
    bool sourceint = Source.typespec().is_int();  // is it an int?
    llvm::Value *args[3];
    args[0] = rop.sg_void_ptr();
    args[1] = rop.llvm_load_value(Source);
    args[2] = rop.llvm_load_value (Query);
    const char *func = sourceint ? "osl_dict_find_iis" : "osl_dict_find_iss";
    llvm::Value *ret = rop.llvm_call_function (func, &args[0], 3);
    rop.llvm_store_value (ret, Result);
    return true;
}



LLVMGEN (llvm_gen_dict_next)
{
    // dict_net is very straightforward -- just insert sg ptr as first arg
    Opcode &op (rop.inst()->ops()[opnum]);
    DASSERT (op.nargs() == 3);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& NodeID = *rop.opargsym (op, 1);
    DASSERT (Result.typespec().is_int() && NodeID.typespec().is_int());
    llvm::Value *ret = rop.llvm_call_function ("osl_dict_next",
                                               rop.sg_void_ptr(),
                                               rop.llvm_load_value(NodeID));
    rop.llvm_store_value (ret, Result);
    return true;
}



LLVMGEN (llvm_gen_dict_value)
{
    // int dict_value (int nodeID, string attribname, output TYPE value)
    Opcode &op (rop.inst()->ops()[opnum]);
    DASSERT (op.nargs() == 3);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& NodeID = *rop.opargsym (op, 1);
    Symbol& Name   = *rop.opargsym (op, 2);
    Symbol& Value  = *rop.opargsym (op, 3);
    DASSERT (Result.typespec().is_int() && NodeID.typespec().is_int() &&
             Name.typespec().is_string());
    llvm::Value *args[5];
    // arg 0: shaderglobals ptr
    args[0] = rop.sg_void_ptr();
    // arg 1: nodeID
    args[1] = rop.llvm_load_value(NodeID);
    // arg 2: attribute name
    args[2] = rop.llvm_load_value(Name);
    // arg 3: encoded type of Value
    args[3] = rop.llvm_constant(Value.typespec().simpletype());
    // arg 4: pointer to Value
    args[4] = rop.llvm_void_ptr (rop.llvm_get_pointer (Value));
    llvm::Value *ret = rop.llvm_call_function ("osl_dict_value", &args[0], 5);
    rop.llvm_store_value (ret, Result);
    return true;
}



LLVMGEN (llvm_gen_raytype)
{
    // int raytype (string name)
    Opcode &op (rop.inst()->ops()[opnum]);
    DASSERT (op.nargs() == 2);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Name = *rop.opargsym (op, 1);
    llvm::Value *args[2] = { rop.sg_void_ptr(), NULL };
    const char *func = NULL;
    if (Name.is_constant()) {
        // We can statically determine the bit pattern
        ustring name = ((ustring *)Name.data())[0];
        args[1] = rop.llvm_constant (rop.shadingsys().raytype_bit (name));
        func = "osl_raytype_bit";
    } else {
        // No way to know which name is being asked for
        args[1] = rop.llvm_get_pointer (Name);
        func = "osl_raytype_name";
    }
    llvm::Value *ret = rop.llvm_call_function (func, args, 2);
    rop.llvm_store_value (ret, Result);
    return true;
}




#ifdef OIIO_HAVE_BOOST_UNORDERED_MAP
typedef boost::unordered_map<ustring, OpLLVMGen, ustringHash> GeneratorTable;
#else
typedef hash_map<ustring, OpLLVMGen, ustringHash> GeneratorTable;
#endif



static GeneratorTable llvm_generator_table;



static void
initialize_llvm_generator_table ()
{
    static spin_mutex table_mutex;
    static bool table_initialized = false;
    spin_lock lock (table_mutex);
    if (table_initialized)
        return;   // already initialized
#define INIT2(name,func) llvm_generator_table[ustring(#name)] = func
#define INIT(name) llvm_generator_table[ustring(#name)] = llvm_gen_##name;

    INIT (aassign);
    INIT2 (abs, llvm_gen_generic);
    INIT2 (acos, llvm_gen_generic);
    INIT (add);
    INIT2 (and, llvm_gen_andor);
    INIT2 (area, llvm_gen_generic);
    INIT (aref);
    INIT (arraylength);
    INIT2 (asin, llvm_gen_generic);
    INIT (assign);
    INIT2 (atan, llvm_gen_generic);
    INIT2 (atan2, llvm_gen_generic);
    INIT2 (backfacing, llvm_gen_get_simple_SG_field);
    INIT2 (bitand, llvm_gen_bitwise_binary_op);
    INIT2 (bitor, llvm_gen_bitwise_binary_op);
    INIT (calculatenormal);
    INIT2 (ceil, llvm_gen_generic);
    INIT2 (cellnoise, llvm_gen_generic);
    INIT (clamp);
    INIT (closure);
    INIT2 (color, llvm_gen_construct_triple);
    INIT (compassign);
    INIT2 (compl, llvm_gen_unary_op);
    INIT (compref);
    INIT2 (concat, llvm_gen_generic);
    INIT2 (cos, llvm_gen_generic);
    INIT2 (cosh, llvm_gen_generic);
    INIT2 (cross, llvm_gen_generic);
    INIT2 (degrees, llvm_gen_generic);
    INIT2 (determinant, llvm_gen_generic);
    INIT (dict_find);
    INIT (dict_next);
    INIT (dict_value);
    INIT2 (distance, llvm_gen_generic);
    INIT (div);
    INIT2 (dot, llvm_gen_generic);
    INIT2 (Dx, llvm_gen_DxDy);
    INIT2 (Dy, llvm_gen_DxDy);
    INIT2 (dowhile, llvm_gen_loop_op);
    // INIT (end);
    INIT2 (endswith, llvm_gen_generic);
    INIT (environment);
    INIT2 (eq, llvm_gen_compare_op);
    INIT2 (erf, llvm_gen_generic);
    INIT2 (erfc, llvm_gen_generic);
    INIT2 (error, llvm_gen_printf);
    INIT2 (exp, llvm_gen_generic);
    INIT2 (exp2, llvm_gen_generic);
    INIT2 (expm1, llvm_gen_generic);
    INIT2 (fabs, llvm_gen_generic);
    INIT (filterwidth);
    INIT2 (floor, llvm_gen_generic);
    INIT2 (fmod, llvm_gen_mod);
    INIT2 (for, llvm_gen_loop_op);
    INIT2 (format, llvm_gen_printf);
    //stdosl.h INIT (fresnel);
    INIT2 (ge, llvm_gen_compare_op);
    INIT (getattribute);
    INIT (getmessage);
    INIT (gettextureinfo);
    INIT2 (gt, llvm_gen_compare_op);
    //stdosl.h  INIT (hypot);
    INIT (if);
    INIT2 (inversesqrt, llvm_gen_generic);
    INIT2 (isfinite, llvm_gen_generic);
    INIT2 (isinf, llvm_gen_generic);
    INIT2 (isnan, llvm_gen_generic);
    INIT2 (le, llvm_gen_compare_op);
    INIT2 (length, llvm_gen_generic);
    INIT2 (log, llvm_gen_generic);
    INIT2 (log10, llvm_gen_generic);
    INIT2 (log2, llvm_gen_generic);
    INIT2 (logb, llvm_gen_generic);
    INIT2 (lt, llvm_gen_compare_op);
    //stdosl.h   INIT (luminance);
    INIT (matrix);
    INIT (mxcompassign);
    INIT (mxcompref);
    INIT2 (min, llvm_gen_minmax);
    INIT2 (max, llvm_gen_minmax);
    //stdosl.h   INIT (mix);
    INIT (mod);
    INIT (mul);
    INIT (neg);
    INIT2 (neq, llvm_gen_compare_op);
    INIT2 (noise, llvm_gen_generic);
    // INIT (nop);
    INIT2 (normal, llvm_gen_construct_triple);
    INIT2 (normalize, llvm_gen_generic);
    INIT2 (or, llvm_gen_andor);
    INIT2 (pnoise, llvm_gen_pnoise);
    INIT2 (point, llvm_gen_construct_triple);
    INIT  (pointcloud_search);
    INIT2 (pow, llvm_gen_generic);
    INIT (printf);
    INIT2 (psnoise, llvm_gen_pnoise);
    INIT2 (radians, llvm_gen_generic);
    INIT (raytype);
    //stdosl.h INIT (reflect);
    //stdosl.h INIT (refract);
    INIT2 (regex_match, llvm_gen_regex);
    INIT2 (regex_search, llvm_gen_regex);
    INIT2 (round, llvm_gen_generic);
    INIT (setmessage);
    INIT2 (shl, llvm_gen_bitwise_binary_op);
    INIT2 (shr, llvm_gen_bitwise_binary_op);
    INIT2 (sign, llvm_gen_generic);
    INIT2 (sin, llvm_gen_generic);
    INIT (sincos);
    INIT2 (sinh, llvm_gen_generic);
    INIT2 (smoothstep, llvm_gen_generic);
    INIT2 (snoise, llvm_gen_generic);
    INIT (spline);
    INIT2 (sqrt, llvm_gen_generic);
    INIT2 (startswith, llvm_gen_generic);
    INIT2 (step, llvm_gen_generic);
    INIT2 (strlen, llvm_gen_generic);
    INIT (sub);
    INIT2 (substr, llvm_gen_generic);
    INIT2 (surfacearea, llvm_gen_get_simple_SG_field);
    INIT2 (tan, llvm_gen_generic);
    INIT2 (tanh, llvm_gen_generic);
    INIT (texture);
    INIT (texture3d);
    INIT (trace);
    INIT2 (transform,  llvm_gen_generic);
    INIT2 (transformn, llvm_gen_generic);
    INIT2 (transformv, llvm_gen_generic);
    INIT2 (transpose, llvm_gen_generic);
    INIT2 (trunc, llvm_gen_generic);
    INIT (useparam);
    INIT2 (vector, llvm_gen_construct_triple);
    INIT2 (warning, llvm_gen_printf);
    INIT2 (while, llvm_gen_loop_op);
    INIT2 (xor, llvm_gen_bitwise_binary_op);

#undef INIT
#undef INIT2

    table_initialized = true;
}



bool
RuntimeOptimizer::build_llvm_code (int beginop, int endop, llvm::BasicBlock *bb)
{
    if (bb)
        builder().SetInsertPoint (bb);

    for (int opnum = beginop;  opnum < endop;  ++opnum) {
        const Opcode& op = inst()->ops()[opnum];

        GeneratorTable::const_iterator found = llvm_generator_table.find (op.opname());
        if (found != llvm_generator_table.end()) {
            bool ok = (*found->second) (*this, opnum);
            if (! ok)
                return false;
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
               if (sym.typespec().is_closure()) {
                    int arraylen = std::max (1, sym.typespec().arraylength());
                    llvm::Value *val = llvm_constant_ptr(NULL, llvm_type_void_ptr());
                    for (int a = 0; a < arraylen;  ++a) {
                        llvm::Value *arrind = sym.typespec().is_array() ? llvm_constant(a) : NULL;
                        llvm_store_value (val, sym, 0, arrind, 0);
                    }
                }
            }
        }
    }

    // Setup the symbols
    m_named_values.clear ();
    BOOST_FOREACH (Symbol &s, inst()->symbols()) {
        // Skip constants -- we always inline them
        if (s.symtype() == SymTypeConst)
            continue;
        // Skip structure placeholders
        if (s.typespec().is_structure())
            continue;
        // Allocate space for locals, temps, aggregate constants
        if (s.symtype() == SymTypeLocal || s.symtype() == SymTypeTemp ||
                s.symtype() == SymTypeConst)
            getOrAllocateLLVMSymbol (s);
        // Set initial value for constants, and closures
        if (s.symtype() != SymTypeParam && s.symtype() != SymTypeOutputParam &&
            (s.is_constant() || s.typespec().is_closure()))
            llvm_assign_initial_value (s);
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
                llvm_run_connected_layer (*this, *srcsym, con.src.param, NULL);
                if (srcsym->typespec().is_array()) {
                    for (int i = 0;  i < srcsym->typespec().arraylength();  ++i)
                        llvm_assign_impl (*this, *dstsym, *srcsym, i);
                } else {
                    // Not an array case
                    llvm_assign_impl (*this, *dstsym, *srcsym);
                }
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



void
RuntimeOptimizer::build_llvm_group ()
{
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

    Timer timer;
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
    m_stat_llvm_irgen_time = timer();  timer.reset();  timer.start();

    // Optimize the LLVM IR unless it's just a ret void group (1 layer, 1 BB, 1 inst == retvoid)
    bool skip_optimization = m_num_used_layers == 1 && entry_func->size() == 1 && entry_func->front().size() == 1;
    // Label the group as being retvoid or not.
    m_group.does_nothing(skip_optimization);
    if (!skip_optimization) {
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

    m_stat_llvm_opt_time = timer();  timer.reset();  timer.start();

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

    // Force the JIT to happen now, while we have the lock
    llvm::ExecutionEngine* ee = shadingsys().ExecutionEngine();
    RunLLVMGroupFunc f = (RunLLVMGroupFunc) ee->getPointerToFunction(entry_func);
    m_group.llvm_compiled_version (f);

    // Remove the IR for the group layer functions, we've already JITed it
    // and will never need the IR again.  This saves memory, and also saves
    // a huge amount of time since we won't re-optimize it again and again
    // if we keep adding new shader groups to the same Module.
    for (int i = 0; i < m_num_used_layers; ++i) {
        funcs[i]->deleteBody();
    }

#if 1
    // Free the exec and module to reclaim all the memory.  This definitely
    // saves memory, and has almost no effect on runtime.
    delete shadingsys().m_llvm_exec;
    // N.B. Destroying the EE should have destroyed the module as well.
    shadingsys().m_llvm_exec = NULL;
    shadingsys().m_llvm_module = NULL;
#endif

#if 0
    // Enable this code to delete the whole context after processing the
    // group.  We did this experiment to see how much memory was being
    // held in the context.  Answer: nothing appreciable, not worth the
    // extra work of constant creation and tear-down.
    delete m_llvm_passes;  m_llvm_passes = NULL;
    delete m_llvm_func_passes;  m_llvm_func_passes = NULL;
    delete m_llvm_func_passes_optimized;  m_llvm_func_passes_optimized = NULL;
    delete shadingsys().m_llvm_context;
    shadingsys().m_llvm_context = NULL;
#endif

    m_stat_llvm_jit_time = timer();
}



void
RuntimeOptimizer::initialize_llvm_group ()
{
    // I don't think we actually need to lock here (lg)
    // static spin_mutex mutex;
    // spin_lock lock (mutex);

    m_llvm_context = m_shadingsys.llvm_context ();
    m_llvm_module = m_shadingsys.m_llvm_module;
    ASSERT (m_llvm_context && m_llvm_module);

    llvm_setup_optimization_passes ();

    // Clear the shaderglobals and groupdata types -- they will be
    // created on demand.
    m_llvm_type_sg = NULL;
    m_llvm_type_groupdata = NULL;
    m_llvm_type_closure_component = NULL;
    m_llvm_type_closure_component_attr = NULL;

    // Set up aliases for types we use over and over
    m_llvm_type_float = llvm::Type::getFloatTy (*m_llvm_context);
    m_llvm_type_int = llvm::Type::getInt32Ty (*m_llvm_context);
    if (sizeof(char *) == 4)
        m_llvm_type_addrint = llvm::Type::getInt32Ty (*m_llvm_context);
    else
        m_llvm_type_addrint = llvm::Type::getInt64Ty (*m_llvm_context);
    m_llvm_type_int_ptr = llvm::Type::getInt32PtrTy (*m_llvm_context);
    m_llvm_type_bool = llvm::Type::getInt1Ty (*m_llvm_context);
    m_llvm_type_longlong = llvm::Type::getInt64Ty (*m_llvm_context);
    m_llvm_type_void = llvm::Type::getVoidTy (*m_llvm_context);
    m_llvm_type_char_ptr = llvm::Type::getInt8PtrTy (*m_llvm_context);
    m_llvm_type_int_ptr = llvm::Type::getInt32PtrTy (*m_llvm_context);
    m_llvm_type_float_ptr = llvm::Type::getFloatPtrTy (*m_llvm_context);
    m_llvm_type_ustring_ptr = llvm::PointerType::get (m_llvm_type_char_ptr, 0);

    // A triple is a struct composed of 3 floats
    std::vector<const llvm::Type*> triplefields(3, m_llvm_type_float);
    m_llvm_type_triple = llvm::StructType::get(llvm_context(), triplefields);
    m_llvm_type_triple_ptr = llvm::PointerType::get (m_llvm_type_triple, 0);

    // A matrix is a struct composed 16 floats
    std::vector<const llvm::Type*> matrixfields(16, m_llvm_type_float);
    m_llvm_type_matrix = llvm::StructType::get(llvm_context(), matrixfields);
    m_llvm_type_matrix_ptr = llvm::PointerType::get (m_llvm_type_matrix, 0);

    for (int i = 0;  llvm_helper_function_table[i];  i += 2) {
        const char *funcname = llvm_helper_function_table[i];
        bool varargs = false;
        const char *types = llvm_helper_function_table[i+1];
        int advance;
        TypeSpec rettype = OSLCompilerImpl::type_from_code (types, &advance);
        types += advance;
        std::vector<const llvm::Type*> params;
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
        llvm::FunctionType *func = llvm::FunctionType::get (llvm_type(rettype), params, varargs);
        m_llvm_module->getOrInsertFunction (funcname, func);
    }

    // Needed for closure setup
    std::vector<const llvm::Type*> params(3);
    params[0] = m_llvm_type_char_ptr;
    params[1] = m_llvm_type_int;
    params[2] = m_llvm_type_char_ptr;
    m_llvm_type_prepare_closure_func = llvm::PointerType::getUnqual (llvm::FunctionType::get (m_llvm_type_void, params, false));
    m_llvm_type_setup_closure_func = m_llvm_type_prepare_closure_func;
}



/// OSL_Dummy_JITMemoryManager - Create a shell that passes on requests
/// to a real JITMemoryManager underneath, but can be retained after the
/// dummy is destroyed.  Also, we don't pass along any deallocations.
class OSL_Dummy_JITMemoryManager : public llvm::JITMemoryManager {
protected:
    llvm::JITMemoryManager *mm;
public:
    OSL_Dummy_JITMemoryManager(llvm::JITMemoryManager *realmm) : mm(realmm) { }
    virtual ~OSL_Dummy_JITMemoryManager() { }
    virtual void setMemoryWritable() { mm->setMemoryWritable(); }
    virtual void setMemoryExecutable() { mm->setMemoryExecutable(); }
    virtual void setPoisonMemory(bool poison) { mm->setPoisonMemory(poison); }
    virtual void AllocateGOT() { mm->AllocateGOT(); }
    bool isManagingGOT() const { return mm->isManagingGOT(); }
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
ShadingSystemImpl::SetupLLVM ()
{
    if (! m_llvm_context) {
        // First time through -- basic LLVM context setup
        info ("Setting up LLVM");
        llvm::DisablePrettyStackTrace = true;
        llvm::llvm_start_multithreaded ();  // enable it to be thread-safe
        m_llvm_context = new llvm::LLVMContext();
        llvm::InitializeNativeTarget();

        initialize_llvm_generator_table ();
    }

    if (! m_llvm_jitmm)
        m_llvm_jitmm = llvm::JITMemoryManager::CreateDefaultMemManager();

    if (! m_llvm_module) {
#ifdef OSL_LLVM_NO_BITCODE
        m_llvm_module = new llvm::Module("llvm_ops", *llvm_context());
#else
        // Load the LLVM bitcode and parse it into a Module
        const char *data = osl_llvm_compiled_ops_block;
#if OSL_LLVM_28
        llvm::MemoryBuffer* buf = llvm::MemoryBuffer::getMemBuffer (llvm::StringRef(data, osl_llvm_compiled_ops_size));
#else
        llvm::MemoryBuffer *buf =
            llvm::MemoryBuffer::getMemBuffer (data, data + osl_llvm_compiled_ops_size);
#endif
        std::string err;
        m_llvm_module = llvm::ParseBitcodeFile (buf, *llvm_context(), &err);
        if (err.length())
            error ("ParseBitcodeFile returned '%s'\n", err.c_str());
        delete buf;
#endif
    }

    // Create the ExecutionEngine
    if (m_llvm_exec
        && false /* FIXME -- leak the EE for now */) {
        m_llvm_exec->addModule (m_llvm_module);
    } else {
        std::string error_msg;
        llvm::JITMemoryManager *mm = new OSL_Dummy_JITMemoryManager(m_llvm_jitmm);
        m_llvm_exec = llvm::ExecutionEngine::createJIT (m_llvm_module,
                                                        &error_msg, mm);
        // Force it to JIT as soon as we ask it for the code pointer,
        // don't take any chances that it might JIT lazily, since we
        // will be stealing the JIT code memory from under its nose and
        // destroying the Module & ExecutionEngine.
        m_llvm_exec->DisableLazyCompilation ();
        if (! m_llvm_exec) {
            error ("Failed to create engine: %s\n", error_msg.c_str());
            DASSERT (0);
            return;
        }
    }
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
