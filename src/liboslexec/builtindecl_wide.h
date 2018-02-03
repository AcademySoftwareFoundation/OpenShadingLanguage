/*
Copyright (c) 2009-2014 Sony Pictures Imageworks Inc., et al.
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


// This file contains "declarations" for all the functions that might get
// called from JITed shader code. But the declaration itself is dependent on
// the DECL macro, which should be declared by the outer file prior to
// including this file. Thus, this list may be repurposed and included
// multiple times, with different DECL definitions.


#ifndef DECL
#error Do not include this file unless DECL is defined
#endif

#define WIDE_NOISE_IMPL_INDIRECT(name, LANE_COUNT)                           \
    DECL (osl_ ## name ## _w ##LANE_COUNT## fw ##LANE_COUNT## f,  "wfwf")      \
    DECL (osl_ ## name ## _w ##LANE_COUNT## fw ##LANE_COUNT## v,  "wfwv")		\
    DECL (osl_ ## name ## _w ##LANE_COUNT## vw ##LANE_COUNT## v,  "wvwv")      \
    DECL (osl_ ## name ## _w ##LANE_COUNT## vw ##LANE_COUNT## f,  "wvwf")      \
    DECL (osl_ ## name ## _w ##LANE_COUNT## vw ##LANE_COUNT## vw ##LANE_COUNT## f, "wvwvwf") \

#if 0 // incomplete of NOISE_IMPL
    DECL (osl_ ## name ## _fff, "fff")
    DECL (osl_ ## name ## _fvf, "fvf")
    DECL (osl_ ## name ## _vff, "xvff")
#endif

#define WIDE_NOISE_IMPL(name, LANE_COUNT)                           \
	WIDE_NOISE_IMPL_INDIRECT(name, LANE_COUNT)


#define WIDE_NOISE_DERIV_IMPL_INDIRECT(name, LANE_COUNT)                     \
    DECL (osl_ ## name ## _w ##LANE_COUNT## dfw ##LANE_COUNT## dv,   "xXX")          \

#if 0 // incomplete of NOISE_DERIV_IMPL
    DECL (osl_ ## name ## _dfdf,   "xXX")          \
    DECL (osl_ ## name ## _dfdff,  "xXXf")         \
    DECL (osl_ ## name ## _dffdf,  "xXfX")         \
    DECL (osl_ ## name ## _dfdfdf, "xXXX")         \
    DECL (osl_ ## name ## _dfdvf,  "xXvf")         \
    DECL (osl_ ## name ## _dfvdf,  "xXvX")         \
    DECL (osl_ ## name ## _dfdvdf, "xXvX")         \
    DECL (osl_ ## name ## _dvdf,   "xvX")          \
    DECL (osl_ ## name ## _dvdff,  "xvXf")         \
    DECL (osl_ ## name ## _dvfdf,  "xvfX")         \
    DECL (osl_ ## name ## _dvdfdf, "xvXX")         \
    DECL (osl_ ## name ## _dvdv,   "xvv")          \
    DECL (osl_ ## name ## _dvdvf,  "xvvf")         \
    DECL (osl_ ## name ## _dvvdf,  "xvvX")         \
    DECL (osl_ ## name ## _dvdvdf, "xvvX")
#endif

#define WIDE_NOISE_DERIV_IMPL(name, LANE_COUNT)                           \
	WIDE_NOISE_DERIV_IMPL_INDIRECT(name, LANE_COUNT)


#define WIDE_GENERIC_NOISE_DERIV_IMPL_INDIRECT(name, LANE_COUNT)             \
    DECL (osl_ ## name ## _w ##LANE_COUNT## dfw ##LANE_COUNT## dv,   "xsXXXX")   \
    DECL (osl_ ## name ## _w ##LANE_COUNT## dvw ##LANE_COUNT## dvw ##LANE_COUNT## df,   "xsXXXXX")

#if 0 // incomplete of NOISE_DERIV_IMPL
    DECL (osl_ ## name ## _dfdf,   "xsXXXX")       \
    DECL (osl_ ## name ## _dfdfdf, "xsXXXXX")      \
    DECL (osl_ ## name ## _dfdvdf, "xsXXXXX")      \
    DECL (osl_ ## name ## _dvdf,   "xsXXXX")       \
    DECL (osl_ ## name ## _dvdfdf, "xsXXXXX")      \
    DECL (osl_ ## name ## _dvdv,   "xsXXXX")
#endif

#define WIDE_GENERIC_NOISE_DERIV_IMPL(name, LANE_COUNT)             \
	WIDE_GENERIC_NOISE_DERIV_IMPL_INDIRECT(name, LANE_COUNT)             \

#if 0 // incomplete
#define PNOISE_IMPL(name)                          \
    DECL (osl_ ## name ## _fff,   "fff")           \
    DECL (osl_ ## name ## _fffff, "fffff")         \
    DECL (osl_ ## name ## _fvv,   "fvv")           \
    DECL (osl_ ## name ## _fvfvf, "fvfvf")         \
    DECL (osl_ ## name ## _vff,   "xvff")          \
    DECL (osl_ ## name ## _vffff, "xvffff")        \
    DECL (osl_ ## name ## _vvv,   "xvvv")          \
    DECL (osl_ ## name ## _vvfvf, "xvvfvf")

#define PNOISE_DERIV_IMPL(name)                    \
    DECL (osl_ ## name ## _dfdff,    "xXXf")       \
    DECL (osl_ ## name ## _dfdffff,  "xXXfff")     \
    DECL (osl_ ## name ## _dffdfff,  "xXfXff")     \
    DECL (osl_ ## name ## _dfdfdfff, "xXXXff")     \
    DECL (osl_ ## name ## _dfdvv,    "xXXv")       \
    DECL (osl_ ## name ## _dfdvfvf,  "xXvfvf")     \
    DECL (osl_ ## name ## _dfvdfvf,  "xXvXvf")     \
    DECL (osl_ ## name ## _dfdvdfvf, "xXvXvf")     \
    DECL (osl_ ## name ## _dvdff,    "xvXf")       \
    DECL (osl_ ## name ## _dvdffff,  "xvXfff")     \
    DECL (osl_ ## name ## _dvfdfff,  "xvfXff")     \
    DECL (osl_ ## name ## _dvdfdfff, "xvXXff")     \
    DECL (osl_ ## name ## _dvdvv,    "xvvv")       \
    DECL (osl_ ## name ## _dvdvfvf,  "xvvfvf")     \
    DECL (osl_ ## name ## _dvvdfvf,  "xvvXvf")     \
    DECL (osl_ ## name ## _dvdvdfvf, "xvvXvf")

#define GENERIC_PNOISE_DERIV_IMPL(name)            \
    DECL (osl_ ## name ## _dfdff,    "xsXXfXX")    \
    DECL (osl_ ## name ## _dfdfdfff, "xsXXXffXX")  \
    DECL (osl_ ## name ## _dfdvv,    "xsXXvXX")    \
    DECL (osl_ ## name ## _dfdvdfvf, "xsXvXvfXX")  \
    DECL (osl_ ## name ## _dvdff,    "xsvXfXX")    \
    DECL (osl_ ## name ## _dvdfdfff, "xsvXXffXX")  \
    DECL (osl_ ## name ## _dvdvv,    "xsvvvXX")    \
    DECL (osl_ ## name ## _dvdvdfvf, "xsvvXvfXX")
#endif

#define WIDE_UNARY_OP_IMPL(name)                   \
    DECL (osl_ ## name ## _w16fw16f,  "xXX")    \
    DECL (osl_ ## name ## _w16fw16f_masked,  "xXXi")    \
    DECL (osl_ ## name ## _w16dfw16df, "xXX")            \
    DECL (osl_ ## name ## _w16dfw16df_masked, "xXXi")            \
    DECL (osl_ ## name ## _w16vw16v,  "xXX")             \
    DECL (osl_ ## name ## _w16vw16v_masked,  "xXXi")             \
    DECL (osl_ ## name ## _w16dvw16dv, "xXX")
#if 0 // incomplete WIDE_UNARY_OP_IMPL
    DECL (osl_ ## name ## _w16dvw16dv_masked, "xXXi")
#endif

#define WIDE_UNARY_I_OP_IMPL(name)              \
    DECL (osl_ ## name ## _w16iw16i,  "xXX")
#if 0 // incomplete WIDE_UNARY_I_OP_IMPL
    DECL (osl_ ## name ## _w16iw16i_masked,  "xXXi")
#endif

#define WIDE_TEST_F_OP_IMPL(name)              \
    DECL (osl_ ## name ## _w16iw16f,  "xXX")
#if 0 // incomplete WIDE_TEST_F_OP_IMPL
    DECL (osl_ ## name ## _w16iw16f_masked,  "xXXi")
#endif

#define WIDE_UNARY_F_OR_V_OP_IMPL(name)               \
    DECL (osl_ ## name ## _w16fw16f,  "xXX")          \
    DECL (osl_ ## name ## _w16fw16f_masked,  "xXXi")          \
    DECL (osl_ ## name ## _w16vw16v,  "xXX")
#if 0 // incomplete WIDE_TEST_F_OP_IMPL
    DECL (osl_ ## name ## _w16vw16v_masked,  "xXXi")
#endif

// TODO: add _masked variations and unit test to exercise them first
#define WIDE_BINARY_OP_IMPL(name)                       \
    DECL (osl_ ## name ## _w16fw16fw16f,    "xXXX")          \
    DECL (osl_ ## name ## _w16fw16fw16f_masked,    "xXXXi")          \
    DECL (osl_ ## name ## _w16dfw16dfw16df, "xXXX")         \
    DECL (osl_ ## name ## _w16dfw16dfw16df_masked, "xXXXi")         \
    DECL (osl_ ## name ## _w16dfw16fw16df,  "xXXX")         \
    DECL (osl_ ## name ## _w16dfw16dfw16f,  "xXXX")         \
    DECL (osl_ ## name ## _w16dfw16dfw16f_masked,  "xXXXi")         \
    DECL (osl_ ## name ## _w16vw16vw16v,    "xXXX")         \
    DECL (osl_ ## name ## _w16vw16vw16v_masked,    "xXXXi")         \
    DECL (osl_ ## name ## _w16dvw16dvw16dv, "xXXX")         \
    DECL (osl_ ## name ## _w16dvw16vw16dv,  "xXXX")         \
    DECL (osl_ ## name ## _w16dvw16dvw16v,  "xXXX")
#if 0 // incomplete WIDE_BINARY_OP_IMPL
    DECL (osl_ ## name ## _w16dfw16fw16df_masked,  "xXXXi")         \
    DECL (osl_ ## name ## _w16dvw16dvw16dv_masked, "xXXXi")         \
    DECL (osl_ ## name ## _w16dvw16vw16dv_masked,  "xXXXi")         \
    DECL (osl_ ## name ## _w16dvw16dvw16v_masked,  "xXXXi")
#endif

#define WIDE_BINARY_FI_OP_IMPL(name)                       \
    DECL (osl_ ## name ## _w16fw16fw16f,    "xXXX")        \
    DECL (osl_ ## name ## _w16iw16iw16i,    "xXXX")
#if 0 // incomplete WIDE_BINARY_FI_OP_IMPL
    DECL (osl_ ## name ## _w16fw16fw16f_masked,    "xXXXi")        \
    DECL (osl_ ## name ## _w16iw16iw16i_masked,    "xXXXi")
#endif

// Investigate if we need any variations of uniform and varying
// for Vec3 parameters
#define WIDE_BINARY_VF_OP_IMPL(name)                       \
    DECL (osl_ ## name ## _w16vw16vw16f,    "xXXX")        \
    DECL (osl_ ## name ## _w16dvw16dvw16df, "xXXX")        \
    DECL (osl_ ## name ## _w16dvw16dvw16f,  "xXXX")        \
    DECL (osl_ ## name ## _w16dvw16vw16df,  "xXXX")
    #if 0 // incomplete WIDE_BINARY_VF_OP_IMPL
    DECL (osl_ ## name ## _w16vw16vw16f_masked,    "xXXXi")        \
    DECL (osl_ ## name ## _w16dvw16dvw16df_masked, "xXXXi")        \
    DECL (osl_ ## name ## _w16dvw16dvw16f_masked,  "xXXXi")        \
    DECL (osl_ ## name ## _w16dvw16vw16df_masked,  "xXXXi")
    #endif

#define WIDE_BINARY_F_OR_V_OP_IMPL(name)                   \
    DECL (osl_ ## name ## _w16fw16fw16f,    "xXXX")        \
    DECL (osl_ ## name ## _w16fw16fw16f_masked,    "xXXXi")\
    DECL (osl_ ## name ## _w16vw16vw16v,    "xXXX")        \
    DECL (osl_ ## name ## _w16vw16vw16v_masked,    "xXXXi")

#if 0 // incomplete closure support
    DECL (osl_add_closure_closure, "CXCC")
    DECL (osl_mul_closure_float, "CXCf")
    DECL (osl_mul_closure_color, "CXCc")
    DECL (osl_allocate_closure_component, "CXii")
    DECL (osl_allocate_weighted_closure_component, "CXiiX")
    DECL (osl_closure_to_string, "sXC")
#endif

// DECL (osl_format, "ss*") // not called by anyone?
DECL (osl_printf_batched, "xXis*")
DECL (osl_error_batched, "xXis*")
DECL (osl_warning_batched, "xXis*")
#if 0 // incomplete
DECL (osl_split, "isXsii")
#endif
// DECL (osl_incr_layers_executed, "xX") // original used by wide currently



WIDE_NOISE_IMPL(cellnoise, __OSL_SIMD_LANE_COUNT)
//WIDE_NOISE_DERIV_IMPL(cellnoise, __OSL_SIMD_LANE_COUNT) // commented out in non-wide

WIDE_NOISE_IMPL(noise, __OSL_SIMD_LANE_COUNT)
WIDE_NOISE_DERIV_IMPL(noise, __OSL_SIMD_LANE_COUNT)

WIDE_NOISE_IMPL(snoise, __OSL_SIMD_LANE_COUNT)
WIDE_NOISE_DERIV_IMPL(snoise, __OSL_SIMD_LANE_COUNT)

WIDE_NOISE_IMPL(simplexnoise, __OSL_SIMD_LANE_COUNT)
#if 0 // incomplete
WIDE_NOISE_DERIV_IMPL(simplexnoise, __OSL_SIMD_LANE_COUNT)
#endif

WIDE_NOISE_IMPL(usimplexnoise, __OSL_SIMD_LANE_COUNT)
#if 0 // incomplete
WIDE_NOISE_DERIV_IMPL(usimplexnoise, __OSL_SIMD_LANE_COUNT)
#endif

WIDE_GENERIC_NOISE_DERIV_IMPL(gabornoise, __OSL_SIMD_LANE_COUNT)
#if 0 // incomplete
WIDE_GENERIC_NOISE_DERIV_IMPL(genericnoise, __OSL_SIMD_LANE_COUNT)

WIDE_NOISE_IMPL(nullnoise, __OSL_SIMD_LANE_COUNT)
WIDE_NOISE_DERIV_IMPL(nullnoise, __OSL_SIMD_LANE_COUNT)

WIDE_NOISE_IMPL(unullnoise, __OSL_SIMD_LANE_COUNT)
WIDE_NOISE_DERIV_IMPL(unullnoise, __OSL_SIMD_LANE_COUNT)
WIDE_PNOISE_IMPL(pcellnoise, __OSL_SIMD_LANE_COUNT)

//WIDE_PNOISE_DERIV_IMPL(pcellnoise, __OSL_SIMD_LANE_COUNT) // commented out in non-wide
WIDE_PNOISE_IMPL(pnoise, __OSL_SIMD_LANE_COUNT)
WIDE_PNOISE_DERIV_IMPL(pnoise, __OSL_SIMD_LANE_COUNT)

WIDE_PNOISE_IMPL(psnoise, __OSL_SIMD_LANE_COUNT)
WIDE_PNOISE_DERIV_IMPL(psnoise, __OSL_SIMD_LANE_COUNT)

WIDE_GENERIC_PNOISE_DERIV_IMPL(gaborpnoise, __OSL_SIMD_LANE_COUNT)
WIDE_GENERIC_PNOISE_DERIV_IMPL(genericpnoise, __OSL_SIMD_LANE_COUNT)
#endif

// Need test garbor noise options for batched
//DECL (osl_noiseparams_set_anisotropic, "xXi") // share non-wide impl
//DECL (osl_noiseparams_set_do_filter, "xXi") // share non-wide impl
//DECL (osl_noiseparams_set_direction, "xXv") // share non-wide impl
//DECL (osl_noiseparams_set_bandwidth, "xXf") // share non-wide impl
//DECL (osl_noiseparams_set_impulses, "xXf")  // share non-wide impl

#if 0 // incomplete
DECL (osl_count_noise, "xX")
#endif


DECL (osl_spline_w16fff, "xXXXXii")
DECL (osl_spline_w16fw16ff, "xXXXXii")
DECL (osl_spline_w16fw16fw16f, "xXXXXii")
DECL (osl_spline_w16dfw16dfw16df, "xXXXXii")
#if 0 // incomplete spline
// Need w16 for combinations of the 3 parameters allowed to be uniform
// caveat, some combos are unreachable/uneeded
DECL (osl_spline_dfdfdf, "xXXXXii")
DECL (osl_spline_dfdff, "xXXXXii")
DECL (osl_spline_dffdf, "xXXXXii")
DECL (osl_spline_vfv, "xXXXXii")
DECL (osl_spline_dvdfdv, "xXXXXii")
DECL (osl_spline_dvdfv, "xXXXXii")
DECL (osl_spline_dvfdv, "xXXXXii")
DECL (osl_splineinverse_fff, "xXXXXii")
DECL (osl_splineinverse_dfdfdf, "xXXXXii")
DECL (osl_splineinverse_dfdff, "xXXXXii")
DECL (osl_splineinverse_dffdf, "xXXXXii")
#endif

#if 0 // incomplete
DECL (osl_setmessage, "xXsLXisi")
DECL (osl_getmessage, "iXssLXiisi")
DECL (osl_pointcloud_search, "iXsXfiiXXii*")
DECL (osl_pointcloud_get, "iXsXisLX")
DECL (osl_pointcloud_write, "iXsXiXXX")
DECL (osl_pointcloud_write_helper, "xXXXisLX")
DECL (osl_blackbody_vf, "xXXf")
DECL (osl_wavelength_color_vf, "xXXf")
#endif

DECL (osl_luminance_fv_batched, "xXXX")
DECL (osl_luminance_w16fv_batched, "xXXX")
DECL (osl_luminance_w16fw16v_batched, "xXXX")
DECL (osl_luminance_dfdv_batched, "xXXX")
DECL (osl_luminance_w16dfw16dv_batched, "xXXX")

DECL (osl_prepend_color_from_v_batched, "xXXs")
DECL (osl_prepend_color_from_w16v_batched, "xXXs")
#if 0 // incomplete
DECL (osl_prepend_color_from_v_masked, "xXXsi")
DECL (osl_prepend_color_from_w16v_masked, "xXXi")
#endif

DECL (osl_prepend_matrix_from_w16ms_batched, "xXXs")
DECL (osl_prepend_matrix_from_w16ms_masked, "xXXsi")
DECL (osl_prepend_matrix_from_w16mw16s_batched, "xXXX")
DECL (osl_prepend_matrix_from_w16mw16s_masked, "xXXXi")

// Batched code gen uses a combination of osl_build_transform_matrix
// with osl_transform_[point|vector|normal] to do the follow functions
// DECL (osl_get_matrix, "iXXs")  // unneeded
// DECL (osl_get_inverse_matrix, "iXXs") // unneeded
// DECL (osl_transform_triple, "iXXiXiXXi") // unneeded
// DECL (osl_transform_triple_nonlinear, "iXXiXiXXi") // unneeded

DECL (osl_build_transform_matrix_ss_masked, "iXXXXi")
DECL (osl_build_transform_matrix_w16ss_masked, "iXXXXi")
DECL (osl_build_transform_matrix_sw16s_masked, "iXXXXi")
DECL (osl_build_transform_matrix_w16sw16s_masked, "iXXXXi")


#if 0 // incomplete
DECL (osl_dict_find_iis, "iXiX")
DECL (osl_dict_find_iss, "iXXX")
DECL (osl_dict_next, "iXi")
DECL (osl_dict_value, "iXiXLX")
#endif
DECL (osl_raytype_name_batched, "iXX")
#if 0 // incomplete
DECL (osl_range_check, "iiiXXXiXiXX")
DECL (osl_naninf_check, "xiXiXXiXiiX")
DECL (osl_uninit_check, "xLXXXiXiXXiXiXii")
#endif

DECL (osl_get_attribute_batched, "iXiXXiiXXi")
DECL (osl_get_attribute_w16attr_name_batched, "iXiXXiiXXi")
DECL (osl_get_attribute_batched_uniform, "iXiXXiiXX")

#ifdef OSL_EXPERIMENTAL_BIND_USER_DATA_WITH_LAYERNAME
DECL (osl_bind_interpolated_param, "iXXXLiXiXiXi")
DECL (osl_bind_interpolated_param_wide, "iXXXLiXiXiXii")
#else
DECL (osl_bind_interpolated_param, "iXXLiXiXiXi")
DECL (osl_bind_interpolated_param_wide, "iXXLiXiXiXii")
#endif
//DECL (osl_get_texture_options, "XX"); // uneeded
DECL (osl_wide_get_noise_options, "XX");
#if 0 // incomplete
DECL (osl_get_noise_options, "XX");
#endif

// The following are defined inside llvm_ops.cpp. Only include these
// declarations in the OSL_LLVM_NO_BITCODE case.
#ifdef OSL_LLVM_NO_BITCODE
// Because we have to provide the no bitcode versions
// of these functions anyway, we are not providing
// bit code versions initially.
// Can be revisited as an optimization.
#endif


WIDE_UNARY_OP_IMPL(sin)
WIDE_UNARY_OP_IMPL(cos)
WIDE_UNARY_OP_IMPL(tan)
WIDE_UNARY_OP_IMPL(asin)
WIDE_UNARY_OP_IMPL(acos)
WIDE_UNARY_OP_IMPL(atan)
WIDE_BINARY_OP_IMPL(atan2)
WIDE_UNARY_OP_IMPL(sinh)
WIDE_UNARY_OP_IMPL(cosh)
WIDE_UNARY_OP_IMPL(tanh)

// DECL (osl_safe_div_iii, "iii") // impl by code generator
// DECL (osl_safe_div_fff, "fff") // impl by code generator
// DECL (osl_safe_mod_iii, "iii") // unneeded stdosl.h should have handled int mod(int, int)

// incomplete masked versions of sincos(if needed)
DECL (osl_sincos_w16fw16fw16f, "xXXX")
DECL (osl_sincos_w16dfw16dfw16f, "xXXX")
DECL (osl_sincos_w16dfw16fw16df, "xXXX")
DECL (osl_sincos_w16dfw16dfw16df, "xXXX")

DECL (osl_sincos_w16vw16vw16v, "xXXX")
DECL (osl_sincos_w16dvw16dvw16v, "xXXX")
DECL (osl_sincos_w16dvw16vw16dv, "xXXX")
DECL (osl_sincos_w16dvw16dvw16dv, "xXXX")

WIDE_UNARY_OP_IMPL(log)
WIDE_UNARY_OP_IMPL(log2)
WIDE_UNARY_OP_IMPL(log10)
WIDE_UNARY_OP_IMPL(exp)
WIDE_UNARY_OP_IMPL(exp2)
WIDE_UNARY_OP_IMPL(expm1)
WIDE_BINARY_OP_IMPL(pow)
WIDE_UNARY_OP_IMPL(erf)
WIDE_UNARY_OP_IMPL(erfc)

WIDE_BINARY_VF_OP_IMPL(pow)

WIDE_UNARY_OP_IMPL(sqrt)
WIDE_UNARY_OP_IMPL(inversesqrt)

WIDE_UNARY_F_OR_V_OP_IMPL(logb)
WIDE_UNARY_F_OR_V_OP_IMPL(floor)
WIDE_UNARY_F_OR_V_OP_IMPL(ceil)
WIDE_UNARY_F_OR_V_OP_IMPL(round)
WIDE_UNARY_F_OR_V_OP_IMPL(trunc)
WIDE_UNARY_F_OR_V_OP_IMPL(sign)
WIDE_BINARY_F_OR_V_OP_IMPL(step)

WIDE_TEST_F_OP_IMPL(isnan)
WIDE_TEST_F_OP_IMPL(isinf)
WIDE_TEST_F_OP_IMPL(isfinite)
WIDE_UNARY_OP_IMPL(abs)
WIDE_UNARY_I_OP_IMPL(abs)
WIDE_UNARY_OP_IMPL(fabs)
WIDE_UNARY_I_OP_IMPL(fabs)
WIDE_BINARY_OP_IMPL(fmod)

DECL (osl_smoothstep_w16fw16fw16fw16f, "xXXXX")
DECL (osl_smoothstep_w16fw16fw16fw16f_masked, "xXXXXi")
DECL (osl_smoothstep_w16dfw16dfw16dfw16df, "xXXXX")
DECL (osl_smoothstep_w16dfw16dfw16dfw16df_masked, "xXXXXi")
DECL (osl_smoothstep_w16dfw16fw16dfw16df, "xXXXX")
DECL (osl_smoothstep_w16dfw16fw16dfw16df_masked, "xXXXXi")
DECL (osl_smoothstep_w16dfw16dfw16fw16df, "xXXXX")
DECL (osl_smoothstep_w16dfw16dfw16fw16df_masked, "xXXXXi")
DECL (osl_smoothstep_w16dfw16dfw16dfw16f, "xXXXX")
DECL (osl_smoothstep_w16dfw16dfw16dfw16f_masked, "xXXXXi")
DECL (osl_smoothstep_w16dfw16fw16fw16df, "xXXXX")
DECL (osl_smoothstep_w16dfw16fw16fw16df_masked, "xXXXXi")
DECL (osl_smoothstep_w16dfw16dfw16fw16f, "xXXXX")
DECL (osl_smoothstep_w16dfw16dfw16fw16f_masked, "xXXXXi")
DECL (osl_smoothstep_w16dfw16fw16dfw16f, "xXXXX")
DECL (osl_smoothstep_w16dfw16fw16dfw16f_masked, "xXXXXi")


// Replaced by osl_transform_[point|vector|normal]
// DECL (osl_transform_vmv, "xXXX")
// DECL (osl_transform_dvmdv, "xXXX")
// DECL (osl_transformv_vmv, "xXXX")
// DECL (osl_transformv_dvmdv, "xXXX")
// DECL (osl_transformn_vmv, "xXXX")
// DECL (osl_transformn_dvmdv, "xXXX")
// DECL (osl_transform_point_vw16vw16m_masked, "xXXXii")
DECL (osl_transform_point_w16vw16vw16m_masked, "xXXXii")
DECL (osl_transform_point_w16dvw16dvw16m_masked, "xXXXii")

DECL (osl_transform_point_w16vw16vm_masked, "xXXXii")
DECL (osl_transform_point_w16dvw16dvm_masked, "xXXXii")

DECL (osl_transform_vector_vw16vw16m_masked, "xXXXii")
DECL (osl_transform_vector_w16vw16vw16m_masked, "xXXXii")
DECL (osl_transform_vector_w16dvw16dvw16m_masked, "xXXXii")

DECL (osl_transform_vector_w16vw16vm_masked, "xXXXii")
DECL (osl_transform_vector_w16dvw16dvm_masked, "xXXXii")

DECL (osl_transform_normal_vw16vw16m_masked, "xXXXii")
DECL (osl_transform_normal_w16vw16vw16m_masked, "xXXXii")
DECL (osl_transform_normal_w16dvw16dvw16m_masked, "xXXXii")

DECL (osl_transform_normal_w16vw16vm_masked, "xXXXii")
DECL (osl_transform_normal_w16dvw16dvm_masked, "xXXXii")


DECL (osl_dot_w16fw16vw16v, "xXXX")
DECL (osl_dot_w16fw16vw16v_masked, "xXXXi")
DECL (osl_dot_w16dfw16dvw16dv, "xXXX")
DECL (osl_dot_w16dfw16dvw16dv_masked, "xXXXi")
#if 0 // incomplete
DECL (osl_dot_w16dfw16dvw16v, "xXXX")
DECL (osl_dot_w16dfw16dvw16v_masked, "xXXXi")
DECL (osl_dot_w16dfw16vw16dv, "xXXX")
DECL (osl_dot_w16dfw16vw16dv_masked, "xXXXi")
#endif

DECL (osl_cross_w16vw16vw16v, "xXXX")
DECL (osl_cross_w16vw16vw16v_masked, "xXXXi")
DECL (osl_cross_w16dvw16dvw16dv, "xXXX")
DECL (osl_cross_w16dvw16dvw16dv_masked, "xXXXi")
#if 0 // incomplete
DECL (osl_cross_w16dvw16dvw16v, "xXXX")
DECL (osl_cross_w16dvw16dvw16v_masked, "xXXXi")
DECL (osl_cross_w16dvw16vw16dv, "xXXX")
DECL (osl_cross_w16dvw16vw16dv_masked, "xXXXi")
#endif

DECL (osl_length_w16fw16v, "xXX")
DECL (osl_length_w16fw16v_masked, "xXXi")
DECL (osl_length_w16dfw16dv, "xXX")
DECL (osl_length_w16dfw16dv_masked, "xXXi")


DECL (osl_distance_w16fw16vw16v, "xXXX")
DECL (osl_distance_w16fw16vw16v_masked, "xXXXi")
DECL (osl_distance_w16dfw16dvw16dv, "xXXX")
DECL (osl_distance_w16dfw16dvw16dv_masked, "xXXXi")
#if 0 // incomplete
DECL (osl_distance_w16dfw16dvw16v, "xXXX")
DECL (osl_distance_w16dfw16dvw16v_masked, "xXXXi")
DECL (osl_distance_w16dfw16vw16dv, "xXXX")
DECL (osl_distance_w16dfw16vw16dv_masked, "xXXXi")
#endif

DECL (osl_normalize_w16vw16v, "xXX")
DECL (osl_normalize_w16vw16v_masked, "xXXi")
DECL (osl_normalize_w16dvw16dv, "xXX")
DECL (osl_normalize_w16dvw16dv_masked, "xXXi")

DECL (osl_mul_w16mw16mw16m, "xXXX")
DECL (osl_mul_w16mw16fw16m, "xXXX")
DECL (osl_mul_w16mw16mw16f, "xXXX")
DECL (osl_mul_w16mw16fw16f, "xXXX")
#if 0 // incomplete
DECL (osl_mul_w16mw16mw16m_masked, "xXXXi")
DECL (osl_mul_w16mw16fw16m_masked, "xXXXi")
DECL (osl_mul_w16mw16mw16f_masked, "xXXXi")
DECL (osl_mul_w16mw16fw16f_masked, "xXXXi")
#endif

DECL (osl_div_w16mw16mw16m, "xXXX")
DECL (osl_div_w16mw16mw16f, "xXXX")
DECL (osl_div_w16mw16fw16m, "xXXX")
DECL (osl_div_w16mw16fw16f, "xXXf")
#if 0 // incomplete
DECL (osl_div_w16mw16mw16m_masked, "xXXX")
DECL (osl_div_w16mw16mw16f_masked, "xXXX")
DECL (osl_div_w16mw16fw16m_masked, "xXXX")
DECL (osl_div_w16mw16fw16f_masked, "xXXf")
#endif

DECL (osl_get_from_to_matrix_w16mss_batched, "iXXss")
DECL (osl_get_from_to_matrix_w16mss_masked, "iXXssi")
DECL (osl_get_from_to_matrix_w16msw16s_batched, "iXXsX")
DECL (osl_get_from_to_matrix_w16msw16s_masked, "iXXsXi")
DECL (osl_get_from_to_matrix_w16mw16ss_batched, "iXXXs")
DECL (osl_get_from_to_matrix_w16mw16ss_masked, "iXXXsi")
DECL (osl_get_from_to_matrix_w16mw16sw16s_batched, "iXXXX")
DECL (osl_get_from_to_matrix_w16mw16sw16s_masked, "iXXXXi")

DECL (osl_transpose_w16mw16m, "xXX")
#if 0 // incomplete ?
DECL (osl_transpose_w16mw16m_masked, "xXXi")
#endif
DECL (osl_determinant_w16fw16m, "xXX")
#if 0 // incomplete ?
DECL (osl_determinant_w16fw16m_masked, "xXXi")
#endif

#if 0 // incomplete
DECL (osl_concat_sss, "sss")
DECL (osl_strlen_is, "is")
DECL (osl_hash_is, "is")
DECL (osl_getchar_isi, "isi");
DECL (osl_startswith_iss, "iss")
DECL (osl_endswith_iss, "iss")
DECL (osl_stoi_is, "is")
DECL (osl_stof_fs, "fs")
DECL (osl_substr_ssii, "ssii")
DECL (osl_regex_impl, "iXsXisi")
#endif

// BATCH texturing manages the BatchedTextureOptions
// directly in LLVM ir, and has no need for wide versions
// of osl_texture_set_XXX functions
#ifdef OSL_LLVM_NO_BITCODE
#endif
DECL (osl_texture_decode_wrapmode, "iX");
DECL (osl_texture_decode_interpmode, "iX");
DECL (osl_texture_batched, "iXXXXXXXXXXiXiXiXi")
#if 0 // incomplete
DECL (osl_texture3d, "iXXXXXXXXiXXXXXXXXX")
DECL (osl_environment, "iXXXXXXXiXXXXXXX")
#endif
DECL (osl_get_textureinfo_batched, "iXXXXXi")
DECL (osl_get_textureinfo_batched_uniform, "iXXXXXX")

#if 0
DECL (osl_trace_set_mindist, "xXf")
DECL (osl_trace_set_maxdist, "xXf")
DECL (osl_trace_set_shade, "xXi")
DECL (osl_trace_set_traceset, "xXs")
DECL (osl_trace, "iXXXXXXXX")
#endif

DECL (osl_calculatenormal_batched, "xXXX")
DECL (osl_area_w16, "xXX")
DECL (osl_area_w16_masked, "xXXi")
DECL (osl_filterwidth_w16fw16df, "xXX")
DECL (osl_filterwidth_w16vw16dv, "xXX")
DECL (osl_raytype_bit_batched, "iXi")


// Clean up local definitions
#undef WIDE_NOISE_IMPL_INDIRECT
#undef WIDE_NOISE_IMPL
#undef WIDE_NOISE_DERIV_IMPL_INDIRECT
#undef WIDE_NOISE_DERIV_IMPL
#undef WIDE_GENERIC_NOISE_DERIV_IMPL_INDIRECT
#undef WIDE_GENERIC_NOISE_DERIV_IMPL
#undef WIDE_PNOISE_IMPL_INDIRECT
#undef WIDE_PNOISE_IMPL
#undef WIDE_PNOISE_DERIV_IMPL_INDIRECT
#undef WIDE_PNOISE_DERIV_IMPL

#undef WIDE_UNARY_OP_IMPL
#undef WIDE_UNARY_I_OP_IMPL
#undef WIDE_UNARY_F_OR_V_OP_IMPL
#undef WIDE_TEST_F_OP_IMPL
#undef WIDE_BINARY_OP_IMPL
#undef WIDE_BINARY_F_OR_V_OP_IMPL
#undef WIDE_BINARY_FI_OP_IMPL
#undef WIDE_BINARY_VF_OP_IMPL
