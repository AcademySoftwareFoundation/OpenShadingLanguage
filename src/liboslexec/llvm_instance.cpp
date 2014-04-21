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

#include <OpenImageIO/timer.h>
#include <OpenImageIO/sysutil.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/strutil.h>

#include "oslexec_pvt.h"
#include "../liboslcomp/oslcomp_pvt.h"
#include "backendllvm.h"

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
        *x = sg->u * group->param_0_bar;
        group->param_1_foo = *x;
    }

    void $layer_1 (ShaderGlobals *sg, GroupData_1 *group)
    {
        if (group->layer_run[1])
            return;
        group->layer_run[1] = 1;
        // ...
        $layer_0 (sg, group);    // because we need its outputs
        *y = sg->u * group->$param_1_bar;
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

using namespace OSL::pvt;

OSL_NAMESPACE_ENTER

namespace pvt {

static spin_mutex llvm_mutex;

static ustring op_end("end");
static ustring op_nop("nop");
static ustring op_aassign("aassign");
static ustring op_compassign("compassign");
static ustring op_aref("aref");
static ustring op_compref("compref");

// Trickery to force linkage of files when building static libraries.
extern int opclosure_cpp_dummy, opcolor_cpp_dummy;
extern int opmessage_cpp_dummy, opnoise_cpp_dummy;
extern int opspline_cpp_dummy, opstring_cpp_dummy;
#ifdef OSL_LLVM_NO_BITCODE
extern int llvm_ops_cpp_dummy;
#endif
int *force_osl_op_linkage[] = {
    &opclosure_cpp_dummy, &opcolor_cpp_dummy, &opmessage_cpp_dummy,
    &opnoise_cpp_dummy, &opspline_cpp_dummy,  &opstring_cpp_dummy,
#ifdef OSL_LLVM_NO_BITCODE
    &llvm_ops_cpp_dummy
#endif
};


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
    "osl_allocate_weighted_closure_component", "CXiiiX",
    "osl_closure_to_string", "sXC",
    "osl_format", "ss*",
    "osl_printf", "xXs*",
    "osl_error", "xXs*",
    "osl_warning", "xXs*",
    "osl_incr_layers_executed", "xX",

    NOISE_IMPL(cellnoise),
    NOISE_DERIV_IMPL(cellnoise),
    NOISE_IMPL(noise),
    NOISE_DERIV_IMPL(noise),
    NOISE_IMPL(snoise),
    NOISE_DERIV_IMPL(snoise),
    NOISE_IMPL(simplexnoise),
    NOISE_DERIV_IMPL(simplexnoise),
    NOISE_IMPL(usimplexnoise),
    NOISE_DERIV_IMPL(usimplexnoise),
    GENERIC_NOISE_DERIV_IMPL(gabornoise),
    GENERIC_NOISE_DERIV_IMPL(genericnoise),
    PNOISE_IMPL(pcellnoise),
    PNOISE_DERIV_IMPL(pcellnoise),
    PNOISE_IMPL(pnoise),
    PNOISE_DERIV_IMPL(pnoise),
    PNOISE_IMPL(psnoise),
    PNOISE_DERIV_IMPL(psnoise),
    GENERIC_PNOISE_DERIV_IMPL(gaborpnoise),
    GENERIC_PNOISE_DERIV_IMPL(genericpnoise),
    "osl_noiseparams_clear", "xX",
    "osl_noiseparams_set_anisotropic", "xXi",
    "osl_noiseparams_set_do_filter", "xXi",
    "osl_noiseparams_set_direction", "xXv",
    "osl_noiseparams_set_bandwidth", "xXf",
    "osl_noiseparams_set_impulses", "xXf",

    "osl_spline_fff", "xXXXXii",
    "osl_spline_dfdfdf", "xXXXXii",
    "osl_spline_dfdff", "xXXXXii",
    "osl_spline_dffdf", "xXXXXii",
    "osl_spline_vfv", "xXXXXii",
    "osl_spline_dvdfdv", "xXXXXii",
    "osl_spline_dvdfv", "xXXXXii",
    "osl_spline_dvfdv", "xXXXXii",
    "osl_splineinverse_fff", "xXXXXii",
    "osl_splineinverse_dfdfdf", "xXXXXii",
    "osl_splineinverse_dfdff", "xXXXXii",
    "osl_splineinverse_dffdf", "xXXXXii",
    "osl_setmessage", "xXsLXisi",
    "osl_getmessage", "iXssLXiisi",
    "osl_pointcloud_search", "iXsXfiiXXii*",
    "osl_pointcloud_get", "iXsXisLX",
    "osl_pointcloud_write", "iXsXiXXX",
    "osl_pointcloud_write_helper", "xXXXisLX",
    "osl_blackbody_vf", "xXXf",
    "osl_wavelength_color_vf", "xXXf",
    "osl_luminance_fv", "xXXX",
    "osl_luminance_dfdv", "xXXX",
    "osl_split", "isXsii",

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
    "osl_pow_dvdvf", "xXXf",

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
    "osl_sign_vv", "xXX",
    "osl_step_fff", "fff",
    "osl_step_vvv", "xXXX",

    "osl_isnan_if", "if",
    "osl_isinf_if", "if",
    "osl_isfinite_if", "if",
    "osl_abs_ii", "ii",
    "osl_fabs_ii", "ii",

    UNARY_OP_IMPL(abs),
    UNARY_OP_IMPL(fabs),

    BINARY_OP_IMPL(fmod),

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

    "osl_transform_triple", "iXXiXiXXi",
    "osl_transform_triple_nonlinear", "iXXiXiXXi",

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
    "osl_stoi_is", "is",
    "osl_stof_fs", "fs",
    "osl_substr_ssii", "ssii",
    "osl_regex_impl", "iXsXisi",

    "osl_texture_clear", "xX",
    "osl_texture_set_firstchannel", "xXi",
    "osl_texture_set_swrap", "xXs",
    "osl_texture_set_twrap", "xXs",
    "osl_texture_set_rwrap", "xXs",
    "osl_texture_set_stwrap", "xXs",
    "osl_texture_set_swrap_code", "xXi",
    "osl_texture_set_twrap_code", "xXi",
    "osl_texture_set_rwrap_code", "xXi",
    "osl_texture_set_stwrap_code", "xXi",
    "osl_texture_set_sblur", "xXf",
    "osl_texture_set_tblur", "xXf",
    "osl_texture_set_rblur", "xXf",
    "osl_texture_set_stblur", "xXf",
    "osl_texture_set_swidth", "xXf",
    "osl_texture_set_twidth", "xXf",
    "osl_texture_set_rwidth", "xXf",
    "osl_texture_set_stwidth", "xXf",
    "osl_texture_set_fill", "xXf",
    "osl_texture_set_time", "xXf",
    "osl_texture_set_interp", "xXs",
    "osl_texture_set_interp_code", "xXi",
    "osl_texture_set_subimage", "xXi",
    "osl_texture_set_subimagename", "xXs",
    "osl_texture_set_missingcolor_arena", "xXX",
    "osl_texture_set_missingcolor_alpha", "xXif",
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
    "osl_trace_set_traceset", "xXs",
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
    "osl_range_check", "iiiXXi",
    "osl_naninf_check", "xiXiXXiXiiX",
    "osl_uninit_check", "xLXXXiXii",

    NULL
};



llvm::Type *
BackendLLVM::llvm_type_sg ()
{
    // Create a type that defines the ShaderGlobals for LLVM IR.  This
    // absolutely MUST exactly match the ShaderGlobals struct in oslexec.h.
    if (m_llvm_type_sg)
        return m_llvm_type_sg;

    // Derivs look like arrays of 3 values
    llvm::Type *float_deriv = llvm_type (TypeDesc(TypeDesc::FLOAT, TypeDesc::SCALAR, 3));
    llvm::Type *triple_deriv = llvm_type (TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, 3));
    std::vector<llvm::Type*> sg_types;
    sg_types.push_back (triple_deriv);      // P, dPdx, dPdy
    sg_types.push_back (ll.type_triple());  // dPdz
    sg_types.push_back (triple_deriv);      // I, dIdx, dIdy
    sg_types.push_back (ll.type_triple());  // N
    sg_types.push_back (ll.type_triple());  // Ng
    sg_types.push_back (float_deriv);       // u, dudx, dudy
    sg_types.push_back (float_deriv);       // v, dvdx, dvdy
    sg_types.push_back (ll.type_triple());  // dPdu
    sg_types.push_back (ll.type_triple());  // dPdv
    sg_types.push_back (ll.type_float());   // time
    sg_types.push_back (ll.type_float());   // dtime
    sg_types.push_back (ll.type_triple());  // dPdtime
    sg_types.push_back (triple_deriv);      // Ps

    llvm::Type *vp = (llvm::Type *)ll.type_void_ptr();
    sg_types.push_back(vp);                 // opaque renderstate*
    sg_types.push_back(vp);                 // opaque tracedata*
    sg_types.push_back(vp);                 // opaque objdata*
    sg_types.push_back(vp);                 // ShadingContext*
    sg_types.push_back(vp);                 // RendererServices*
    sg_types.push_back(vp);                 // object2common
    sg_types.push_back(vp);                 // shader2common
    sg_types.push_back(vp);                 // Ci

    sg_types.push_back (ll.type_float());   // surfacearea
    sg_types.push_back (ll.type_int());     // raytype
    sg_types.push_back (ll.type_int());     // flipHandedness
    sg_types.push_back (ll.type_int());     // backfacing

    return m_llvm_type_sg = ll.type_struct (sg_types, "ShaderGlobals");
}



llvm::Type *
BackendLLVM::llvm_type_sg_ptr ()
{
    return ll.type_ptr (llvm_type_sg());
}



llvm::Type *
BackendLLVM::llvm_type_groupdata ()
{
    // If already computed, return it
    if (m_llvm_type_groupdata)
        return m_llvm_type_groupdata;

    std::vector<llvm::Type*> fields;

    // First, add the array that tells if each layer has run.  But only make
    // slots for the layers that may be called/used.
    int sz = (m_num_used_layers + 3) & (~3);  // Round up to 32 bit boundary
    fields.push_back (ll.type_array (ll.type_bool(), sz));
    size_t offset = sz * sizeof(bool);

    // For each layer in the group, add entries for all params that are
    // connected or interpolated, and output params.  Also mark those
    // symbols with their offset within the group struct.
    if (llvm_debug() >= 2)
        std::cout << "Group param struct:\n";
    m_param_order_map.clear ();
    int order = 1;
    for (int layer = 0;  layer < group().nlayers();  ++layer) {
        ShaderInstance *inst = group()[layer];
        if (inst->unused())
            continue;
        FOREACH_PARAM (Symbol &sym, inst) {
            TypeSpec ts = sym.typespec();
            if (ts.is_structure())  // skip the struct symbol itself
                continue;
            int arraylen = std::max (1, sym.typespec().arraylength());
            int deriv_mult = sym.has_derivs() ? 3 : 1;
            int n = arraylen * deriv_mult;
            ts.make_array (n);
            fields.push_back (llvm_type (ts));

            // Alignment
            size_t align = sym.typespec().is_closure_based() ? sizeof(void*) :
                    sym.typespec().simpletype().basesize();
            if (offset & (align-1))
                offset += align - (offset & (align-1));
            if (llvm_debug() >= 2)
                std::cout << "  " << inst->layername() 
                          << " (" << inst->id() << ") " << sym.mangled()
                          << " " << ts.c_str() << ", field " << order 
                          << ", offset " << offset << std::endl;
            sym.dataoffset ((int)offset);
            offset += int(sym.size()) * deriv_mult;

            m_param_order_map[&sym] = order;
            ++order;
        }
    }
    group().llvm_groupdata_size (offset);

    std::string groupdataname = Strutil::format("Groupdata_%llu",
                                                (long long unsigned int)group().name().hash());
    m_llvm_type_groupdata = ll.type_struct (fields, groupdataname);

    return m_llvm_type_groupdata;
}



llvm::Type *
BackendLLVM::llvm_type_groupdata_ptr ()
{
    return ll.type_ptr (llvm_type_groupdata());
}



llvm::Type *
BackendLLVM::llvm_type_closure_component ()
{
    if (m_llvm_type_closure_component)
        return m_llvm_type_closure_component;

    std::vector<llvm::Type*> comp_types;
    comp_types.push_back (ll.type_int());     // parent.type
    comp_types.push_back (ll.type_int());     // id
    comp_types.push_back (ll.type_int());     // size
    comp_types.push_back (ll.type_int());     // nattrs
    comp_types.push_back (ll.type_triple());  // w
    comp_types.push_back (ll.type_int());     // fake field for char mem[4]

    return m_llvm_type_closure_component = ll.type_struct (comp_types, "ClosureComponent");
}



llvm::Type *
BackendLLVM::llvm_type_closure_component_ptr ()
{
    return ll.type_ptr (llvm_type_closure_component());
}


llvm::Type *
BackendLLVM::llvm_type_closure_component_attr ()
{
    if (m_llvm_type_closure_component_attr)
        return m_llvm_type_closure_component_attr;

    std::vector<llvm::Type*> attr_types;
    attr_types.push_back ((llvm::Type *) ll.type_string());  // key

    std::vector<llvm::Type*> union_types;
    union_types.push_back (ll.type_int());
    union_types.push_back (ll.type_float());
    union_types.push_back (ll.type_triple());
    union_types.push_back ((llvm::Type *) ll.type_void_ptr());

    attr_types.push_back (ll.type_union (union_types)); // value union

    return m_llvm_type_closure_component_attr = ll.type_struct (attr_types, "ClosureComponentAttr");
}



llvm::Type *
BackendLLVM::llvm_type_closure_component_attr_ptr ()
{
    return ll.type_ptr (llvm_type_closure_component_attr());
}



void
BackendLLVM::llvm_assign_initial_value (const Symbol& sym)
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
        return;
    }

    if ((sym.symtype() == SymTypeLocal || sym.symtype() == SymTypeTemp)
          && shadingsys().debug_uninit()) {
        // Handle the "debug uninitialized values" case
        bool isarray = sym.typespec().is_array();
        int alen = isarray ? sym.typespec().arraylength() : 1;
        llvm::Value *u = NULL;
        if (sym.typespec().is_closure_based()) {
            // skip closures
        }
        else if (sym.typespec().is_floatbased())
            u = ll.constant (std::numeric_limits<float>::quiet_NaN());
        else if (sym.typespec().is_int_based())
            u = ll.constant (std::numeric_limits<int>::min());
        else if (sym.typespec().is_string_based())
            u = ll.constant (Strings::uninitialized_string);
        if (u) {
            for (int a = 0;  a < alen;  ++a) {
                llvm::Value *aval = isarray ? ll.constant(a) : NULL;
                for (int c = 0;  c < (int)sym.typespec().aggregate(); ++c)
                    llvm_store_value (u, sym, 0, aval, c);
            }
        }
        return;
    }

    if ((sym.symtype() == SymTypeLocal || sym.symtype() == SymTypeTemp) &&
        sym.typespec().is_string_based()) {
        // Strings are pointers.  Can't take any chance on leaving
        // local/tmp syms uninitialized.
        llvm_assign_zero (sym);
        return;  // we're done, the parts below are just for params
    }
    ASSERT_MSG (sym.symtype() == SymTypeParam || sym.symtype() == SymTypeOutputParam,
                "symtype was %d, data type was %s", (int)sym.symtype(), sym.typespec().c_str());

    if (sym.has_init_ops() && sym.valuesource() == Symbol::DefaultVal) {
        // Handle init ops.
        build_llvm_code (sym.initbegin(), sym.initend());
    } else if (! sym.lockgeom() && ! sym.typespec().is_closure()) {
        // geometrically-varying param; memcpy its default value
        TypeDesc t = sym.typespec().simpletype();
        ll.op_memcpy (llvm_void_ptr (sym), ll.constant_ptr (sym.data()),
                      t.size(), t.basesize() /*align*/);
        if (sym.has_derivs())
            llvm_zero_derivs (sym);
    } else {
        // Use default value
        int num_components = sym.typespec().simpletype().aggregate;
        TypeSpec elemtype = sym.typespec().elementtype();
        for (int a = 0, c = 0; a < arraylen;  ++a) {
            llvm::Value *arrind = sym.typespec().is_array() ? ll.constant(a) : NULL;
            if (sym.typespec().is_closure_based())
                continue;
            for (int i = 0; i < num_components; ++i, ++c) {
                // Fill in the constant val
                llvm::Value* init_val = 0;
                if (elemtype.is_floatbased())
                    init_val = ll.constant (((float*)sym.data())[c]);
                else if (elemtype.is_string())
                    init_val = ll.constant (((ustring*)sym.data())[c]);
                else if (elemtype.is_int())
                    init_val = ll.constant (((int*)sym.data())[c]);
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
        args.push_back (ll.constant (sym.name()));
        args.push_back (ll.constant (sym.typespec().simpletype()));
        args.push_back (ll.constant ((int) sym.has_derivs()));
        args.push_back (llvm_void_ptr (sym));
        ll.call_function ("osl_bind_interpolated_param",
                          &args[0], args.size());                            
    }
}



void
BackendLLVM::llvm_generate_debugnan (const Opcode &op)
{
    for (int i = 0;  i < op.nargs();  ++i) {
        Symbol &sym (*opargsym (op, i));
        if (! op.argwrite(i))
            continue;
        TypeDesc t = sym.typespec().simpletype();
        if (t.basetype != TypeDesc::FLOAT)
            continue;  // just check float-based types
        llvm::Value *ncomps = ll.constant (int(t.numelements() * t.aggregate));
        llvm::Value *offset = ll.constant(0);
        llvm::Value *ncheck = ncomps;
        if (op.opname() == op_aassign) {
            // Special case -- array assignment -- only check one element
            ASSERT (i == 0 && "only arg 0 is written for aassign");
            llvm::Value *ind = llvm_load_value (*opargsym (op, 1));
            llvm::Value *agg = ll.constant(t.aggregate);
            offset = t.aggregate == 1 ? ind : ll.op_mul (ind, agg);
            ncheck = agg;
        } else if (op.opname() == op_compassign) {
            // Special case -- component assignment -- only check one channel
            ASSERT (i == 0 && "only arg 0 is written for compassign");
            llvm::Value *ind = llvm_load_value (*opargsym (op, 1));
            offset = ind;
            ncheck = ll.constant(1);
        }

        llvm::Value *args[] = { ncomps,
                                llvm_void_ptr(sym),
                                ll.constant((int)sym.has_derivs()),
                                sg_void_ptr(), 
                                ll.constant(op.sourcefile()),
                                ll.constant(op.sourceline()),
                                ll.constant(sym.name()),
                                offset,
                                ncheck,
                                ll.constant(op.opname())
                              };
        ll.call_function ("osl_naninf_check", args, 10);
    }
}



void
BackendLLVM::llvm_generate_debug_uninit (const Opcode &op)
{
    for (int i = 0;  i < op.nargs();  ++i) {
        Symbol &sym (*opargsym (op, i));
        if (! op.argread(i))
            continue;
        if (sym.typespec().is_closure_based())
            continue;
        TypeDesc t = sym.typespec().simpletype();
        if (t.basetype != TypeDesc::FLOAT && t.basetype != TypeDesc::INT &&
            t.basetype != TypeDesc::STRING)
            continue;  // just check float, int, string based types
        llvm::Value *ncheck = ll.constant (int(t.numelements() * t.aggregate));
        llvm::Value *offset = ll.constant(0);
        // Some special cases...
        if (op.opname() == Strings::op_for && i == 0) {
            // The first argument of 'for' is the condition temp, but
            // note that it may not have had its initializer run yet, so
            // don't generate uninit test code for it.
            continue;
        }
        if (op.opname() == op_aref && i == 1) {
            // Special case -- array assignment -- only check one element
            llvm::Value *ind = llvm_load_value (*opargsym (op, 2));
            llvm::Value *agg = ll.constant(t.aggregate);
            offset = t.aggregate == 1 ? ind : ll.op_mul (ind, agg);
            ncheck = agg;
        } else if (op.opname() == op_compref && i == 1) {
            // Special case -- component assignment -- only check one channel
            llvm::Value *ind = llvm_load_value (*opargsym (op, 2));
            offset = ind;
            ncheck = ll.constant(1);
        }

        llvm::Value *args[] = { ll.constant(t),
                                llvm_void_ptr(sym),
                                sg_void_ptr(), 
                                ll.constant(op.sourcefile()),
                                ll.constant(op.sourceline()),
                                ll.constant(sym.name()),
                                offset,
                                ncheck
                              };
        ll.call_function ("osl_uninit_check", args, 8);
    }
}



bool
BackendLLVM::build_llvm_code (int beginop, int endop, llvm::BasicBlock *bb)
{
    if (bb)
        ll.set_insert_point (bb);

    for (int opnum = beginop;  opnum < endop;  ++opnum) {
        const Opcode& op = inst()->ops()[opnum];
        const OpDescriptor *opd = shadingsys().op_descriptor (op.opname());
        if (opd && opd->llvmgen) {
            if (shadingsys().debug_uninit() /* debug uninitialized vals */)
                llvm_generate_debug_uninit (op);
            bool ok = (*opd->llvmgen) (*this, opnum);
            if (! ok)
                return false;
            if (shadingsys().debug_nan() /* debug NaN/Inf */
                && op.farthest_jump() < 0 /* Jumping ops don't need it */) {
                llvm_generate_debugnan (op);
            }
        } else if (op.opname() == op_nop ||
                   op.opname() == op_end) {
            // Skip this op, it does nothing...
        } else {
            shadingcontext()->error ("LLVMOSL: Unsupported op %s in layer %s\n",
                                     op.opname(), inst()->layername());
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
BackendLLVM::build_llvm_instance (bool groupentry)
{
    // Make a layer function: void layer_func(ShaderGlobals*, GroupData*)
    // Note that the GroupData* is passed as a void*.
    std::string unique_layer_name = Strutil::format ("%s_%d", inst()->layername(), inst()->id());

    ll.current_function (
           ll.make_function (unique_layer_name,
                             !groupentry, // fastcall for non-entry layer functions
                             ll.type_void(), // return type
                             llvm_type_sg_ptr(), llvm_type_groupdata_ptr()));

    // Get shader globals and groupdata pointers
    m_llvm_shaderglobals_ptr = ll.current_function_arg(0); //arg_it++;
    m_llvm_groupdata_ptr = ll.current_function_arg(1); //arg_it++;

    llvm::BasicBlock *entry_bb = ll.new_basic_block (unique_layer_name);
    m_exit_instance_block = NULL;

    // Set up a new IR builder
    ll.new_builder (entry_bb);
#if 0 /* helpful for debuggin */
    if (llvm_debug() && groupentry)
        llvm_gen_debug_printf (Strutil::format("\n\n\n\nGROUP! %s",group().name()));
    if (llvm_debug())
        llvm_gen_debug_printf (Strutil::format("enter layer %s %s",
                                  inst()->layername(), inst()->shadername()));
#endif
    if (shadingsys().countlayerexecs())
        ll.call_function ("osl_incr_layers_executed", sg_void_ptr());

    if (groupentry) {
        if (m_num_used_layers > 1) {
            // If this is the group entry point, clear all the "layer
            // executed" bits.  If it's not the group entry (but rather is
            // an upstream node), then set its bit!
            int sz = (m_num_used_layers + 3) & (~3);  // round up to 32 bits
            ll.op_memset (ll.void_ptr(layer_run_ptr(0)), 0, sz, 4 /*align*/);
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
                    llvm::Value *val = ll.constant_ptr(NULL, ll.type_void_ptr());
                    for (int a = 0; a < arraylen;  ++a) {
                        llvm::Value *arrind = sym.typespec().is_array() ? ll.constant(a) : NULL;
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
        // Skip constants -- we always inline scalar constants, and for
        // array constants we will just use the pointers to the copy of
        // the constant that belongs to the instance.
        if (s.symtype() == SymTypeConst)
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
            s.symtype() != SymTypeGlobal &&
            (s.is_constant() || s.typespec().is_closure_based() ||
             s.typespec().is_string_based() || 
             ((s.symtype() == SymTypeLocal || s.symtype() == SymTypeTemp)
              && shadingsys().debug_uninit())))
            llvm_assign_initial_value (s);
        // If debugnan is turned on, globals check that their values are ok
        if (s.symtype() == SymTypeGlobal && shadingsys().debug_nan()) {
            TypeDesc t = s.typespec().simpletype();
            if (t.basetype == TypeDesc::FLOAT) { // just check float-based types
                int ncomps = t.numelements() * t.aggregate;
                llvm::Value *args[] = { ll.constant(ncomps), llvm_void_ptr(s),
                     ll.constant((int)s.has_derivs()), sg_void_ptr(), 
                     ll.constant(ustring(inst()->shadername())),
                     ll.constant(0), ll.constant(s.name()),
                     ll.constant(0), ll.constant(ncomps),
                     ll.constant("<none>")
                };
                ll.call_function ("osl_naninf_check", args, 10);
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
        if (! s.everread() && ! s.connected_down() && ! s.connected()
              && ! shadingsys().is_renderer_output(s.name()))
            continue;
        // Set initial value for params (may contain init ops)
        llvm_assign_initial_value (s);
    }

    // All the symbols are stack allocated now.

    // Mark all the basic blocks, including allocating llvm::BasicBlock
    // records for each.
    find_basic_blocks ();
    find_conditionals ();
    m_layers_already_run.clear ();

    build_llvm_code (inst()->maincodebegin(), inst()->maincodeend());

    if (llvm_has_exit_instance_block())
        ll.op_branch (m_exit_instance_block); // also sets insert point

    // Transfer all of this layer's outputs into the downstream shader's
    // inputs.
    for (int layer = this->layer()+1;  layer < group().nlayers();  ++layer) {
        ShaderInstance *child = group()[layer];
        for (int c = 0;  c < child->nconnections();  ++c) {
            const Connection &con (child->connection (c));
            if (con.srclayer == this->layer()) {
                ASSERT (con.src.arrayindex == -1 && con.src.channel == -1 &&
                        con.dst.arrayindex == -1 && con.dst.channel == -1 &&
                        "no support for individual element/channel connection");
                Symbol *srcsym (inst()->symbol (con.src.param));
                Symbol *dstsym (child->symbol (con.dst.param));
                llvm_run_connected_layers (*srcsym, con.src.param);
                // FIXME -- I'm not sure I understand this.  Isn't this
                // unnecessary if we wrote to the parameter ourself?
                llvm_assign_impl (*dstsym, *srcsym);
            }
        }
    }
    // llvm_gen_debug_printf ("done copying connections");

    // All done
#if 0 /* helpful for debugging */
    if (llvm_debug())
        llvm_gen_debug_printf (Strutil::format("exit layer %s %s",
                                   inst()->layername(), inst()->shadername()));
#endif
    ll.op_return();

    if (llvm_debug())
        std::cout << "layer_func (" << unique_layer_name << ") "<< this->layer() 
                  << "/" << group().nlayers() << " after llvm  = " 
                  << ll.bitcode_string(ll.current_function()) << "\n";

    ll.end_builder();  // clear the builder

    return ll.current_function();
}



void
BackendLLVM::initialize_llvm_group ()
{
    ll.setup_optimization_passes (shadingsys().llvm_optimize());

    // Clear the shaderglobals and groupdata types -- they will be
    // created on demand.
    m_llvm_type_sg = NULL;
    m_llvm_type_groupdata = NULL;
    m_llvm_type_closure_component = NULL;
    m_llvm_type_closure_component_attr = NULL;

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
        ll.make_function (funcname, false, llvm_type(rettype), params, varargs);
    }

    // Needed for closure setup
    std::vector<llvm::Type*> params(3);
    params[0] = (llvm::Type *) ll.type_char_ptr();
    params[1] = ll.type_int();
    params[2] = (llvm::Type *) ll.type_char_ptr();
    m_llvm_type_prepare_closure_func = ll.type_function_ptr (ll.type_void(), params);
    m_llvm_type_setup_closure_func = m_llvm_type_prepare_closure_func;
}



void
BackendLLVM::run ()
{
    // At this point, we already hold the lock for this group, by virtue
    // of ShadingSystemImpl::optimize_group.
    OIIO::Timer timer;
    std::string err;

    {
#ifdef OSL_LLVM_NO_BITCODE
    // I don't know which exact part has thread safety issues, but it
    // crashes on windows when we don't lock.
    // FIXME -- try subsequent LLVM releases on Windows to see if this
    // is a problem that is eventually fixed on the LLVM side.
    static spin_mutex mutex;
    OIIO::spin_lock lock (mutex);
#endif

#ifdef OSL_LLVM_NO_BITCODE
    ll.module (ll.new_module ("llvm_ops"));
#else
    ll.module (ll.module_from_bitcode (osl_llvm_compiled_ops_block,
                                       osl_llvm_compiled_ops_size,
                                       "llvm_ops", &err));
    if (err.length())
        shadingcontext()->error ("ParseBitcodeFile returned '%s'\n", err.c_str());
    ASSERT (ll.module());
#endif

    // Create the ExecutionEngine
    if (! ll.make_jit_execengine (&err)) {
        shadingcontext()->error ("Failed to create engine: %s\n", err.c_str());
        ASSERT (0);
        return;
    }

    // End of mutex lock, for the OSL_LLVM_NO_BITCODE case
    }

    m_stat_llvm_setup_time += timer.lap();

    // Set up m_num_used_layers to be the number of layers that are
    // actually used, and m_layer_remap[] to map original layer numbers
    // to the shorter list of actually-called layers.
    int nlayers = group().nlayers();
    m_layer_remap.resize (nlayers);
    m_num_used_layers = 0;
    for (int layer = 0;  layer < group().nlayers();  ++layer) {
        bool lastlayer = (layer == (nlayers-1));
        if (! group()[layer]->unused() || lastlayer)
            m_layer_remap[layer] = m_num_used_layers++;
        else
            m_layer_remap[layer] = -1;
    }
    shadingsys().m_stat_empty_instances += group().nlayers()-m_num_used_layers;

    initialize_llvm_group ();

    // Generate the LLVM IR for each layer.  Skip unused layers.
    m_llvm_local_mem = 0;
    llvm::Function** funcs = (llvm::Function**)alloca(m_num_used_layers * sizeof(llvm::Function*));
    for (int layer = 0; layer < nlayers; ++layer) {
        set_inst (layer);
        bool lastlayer = (layer == (nlayers-1));
        int index = m_layer_remap[layer];
        if (index != -1)
            funcs[index] = build_llvm_instance (lastlayer);
    }
    llvm::Function* entry_func = funcs[m_num_used_layers-1];
    m_stat_llvm_irgen_time += timer.lap();

    if (shadingsys().m_max_local_mem_KB &&
        m_llvm_local_mem/1024 > shadingsys().m_max_local_mem_KB) {
        shadingcontext()->error ("Shader group \"%s\" needs too much local storage: %d KB",
                                 group().name(), m_llvm_local_mem/1024);
    }

    // Optimize the LLVM IR unless it's just a ret void group (1 layer,
    // 1 BB, 1 inst == retvoid)
    bool skip_optimization = m_num_used_layers == 1 && ll.func_is_empty(entry_func);
    // Label the group as being retvoid or not.
    group().does_nothing(skip_optimization);
    if (skip_optimization) {
        shadingsys().m_stat_empty_groups += 1;
        shadingsys().m_stat_empty_instances += 1;  // the one layer is empty
    } else {
        ll.do_optimize();
    }

    m_stat_llvm_opt_time += timer.lap();

    if (llvm_debug()) {
        std::cout << "func after opt  = " << ll.bitcode_string (entry_func) << "\n";
        std::cout.flush();
    }

    // Debug code to dump the resulting bitcode to a file
    if (llvm_debug() >= 2) {
        std::string name = Strutil::format ("%s_%d.bc", inst()->layername(),
                                            inst()->id());
        ll.write_bitcode_file (name.c_str());
    }

    // Force the JIT to happen now and retrieve the JITed function
    group().llvm_compiled_version ((RunLLVMGroupFunc) ll.getPointerToFunction(entry_func));

    // Remove the IR for the group layer functions, we've already JITed it
    // and will never need the IR again.  This saves memory, and also saves
    // a huge amount of time since we won't re-optimize it again and again
    // if we keep adding new shader groups to the same Module.
    for (int i = 0; i < m_num_used_layers; ++i) {
        ll.delete_func_body (funcs[i]);
    }

    // Free the exec and module to reclaim all the memory.  This definitely
    // saves memory, and has almost no effect on runtime.
    ll.execengine (NULL);

    // N.B. Destroying the EE should have destroyed the module as well.
    ll.module (NULL);

    m_stat_llvm_jit_time += timer.lap();

    m_stat_total_llvm_time = timer();

    if (shadingsys().m_compile_report) {
        shadingcontext()->info ("JITed shader group %s:", group().name());
        shadingcontext()->info ("    (%1.2fs = %1.2f setup, %1.2f ir, %1.2f opt, %1.2f jit; local mem %dKB)",
                           m_stat_total_llvm_time, 
                           m_stat_llvm_setup_time,
                           m_stat_llvm_irgen_time, m_stat_llvm_opt_time,
                           m_stat_llvm_jit_time,
                           m_llvm_local_mem/1024);
    }
}



}; // namespace pvt
OSL_NAMESPACE_EXIT
