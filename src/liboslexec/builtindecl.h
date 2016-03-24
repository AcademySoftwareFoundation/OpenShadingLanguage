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



#define NOISE_IMPL(name)                           \
    DECL (ei_osl_ ## name ## _ff,  "ff")              \
    DECL (ei_osl_ ## name ## _fff, "fff")             \
    DECL (ei_osl_ ## name ## _fv,  "fv")              \
    DECL (ei_osl_ ## name ## _fvf, "fvf")             \
    DECL (ei_osl_ ## name ## _vf,  "xvf")             \
    DECL (ei_osl_ ## name ## _vff, "xvff")            \
    DECL (ei_osl_ ## name ## _vv,  "xvv")             \
    DECL (ei_osl_ ## name ## _vvf, "xvvf")

#define NOISE_DERIV_IMPL(name)                     \
    DECL (ei_osl_ ## name ## _dfdf,   "xXX")          \
    DECL (ei_osl_ ## name ## _dfdff,  "xXXf")         \
    DECL (ei_osl_ ## name ## _dffdf,  "xXfX")         \
    DECL (ei_osl_ ## name ## _dfdfdf, "xXXX")         \
    DECL (ei_osl_ ## name ## _dfdv,   "xXv")          \
    DECL (ei_osl_ ## name ## _dfdvf,  "xXvf")         \
    DECL (ei_osl_ ## name ## _dfvdf,  "xXvX")         \
    DECL (ei_osl_ ## name ## _dfdvdf, "xXvX")         \
    DECL (ei_osl_ ## name ## _dvdf,   "xvX")          \
    DECL (ei_osl_ ## name ## _dvdff,  "xvXf")         \
    DECL (ei_osl_ ## name ## _dvfdf,  "xvfX")         \
    DECL (ei_osl_ ## name ## _dvdfdf, "xvXX")         \
    DECL (ei_osl_ ## name ## _dvdv,   "xvv")          \
    DECL (ei_osl_ ## name ## _dvdvf,  "xvvf")         \
    DECL (ei_osl_ ## name ## _dvvdf,  "xvvX")         \
    DECL (ei_osl_ ## name ## _dvdvdf, "xvvX")

#define GENERIC_NOISE_DERIV_IMPL(name)             \
    DECL (ei_osl_ ## name ## _dfdf,   "xsXXXX")       \
    DECL (ei_osl_ ## name ## _dfdfdf, "xsXXXXX")      \
    DECL (ei_osl_ ## name ## _dfdv,   "xsXXXX")       \
    DECL (ei_osl_ ## name ## _dfdvdf, "xsXXXXX")      \
    DECL (ei_osl_ ## name ## _dvdf,   "xsXXXX")       \
    DECL (ei_osl_ ## name ## _dvdfdf, "xsXXXXX")      \
    DECL (ei_osl_ ## name ## _dvdv,   "xsXXXX")       \
    DECL (ei_osl_ ## name ## _dvdvdf, "xsXXXXX")

#define PNOISE_IMPL(name)                          \
    DECL (ei_osl_ ## name ## _fff,   "fff")           \
    DECL (ei_osl_ ## name ## _fffff, "fffff")         \
    DECL (ei_osl_ ## name ## _fvv,   "fvv")           \
    DECL (ei_osl_ ## name ## _fvfvf, "fvfvf")         \
    DECL (ei_osl_ ## name ## _vff,   "xvff")          \
    DECL (ei_osl_ ## name ## _vffff, "xvffff")        \
    DECL (ei_osl_ ## name ## _vvv,   "xvvv")          \
    DECL (ei_osl_ ## name ## _vvfvf, "xvvfvf")

#define PNOISE_DERIV_IMPL(name)                    \
    DECL (ei_osl_ ## name ## _dfdff,    "xXXf")       \
    DECL (ei_osl_ ## name ## _dfdffff,  "xXXfff")     \
    DECL (ei_osl_ ## name ## _dffdfff,  "xXfXff")     \
    DECL (ei_osl_ ## name ## _dfdfdfff, "xXXXff")     \
    DECL (ei_osl_ ## name ## _dfdvv,    "xXXv")       \
    DECL (ei_osl_ ## name ## _dfdvfvf,  "xXvfvf")     \
    DECL (ei_osl_ ## name ## _dfvdfvf,  "xXvXvf")     \
    DECL (ei_osl_ ## name ## _dfdvdfvf, "xXvXvf")     \
    DECL (ei_osl_ ## name ## _dvdff,    "xvXf")       \
    DECL (ei_osl_ ## name ## _dvdffff,  "xvXfff")     \
    DECL (ei_osl_ ## name ## _dvfdfff,  "xvfXff")     \
    DECL (ei_osl_ ## name ## _dvdfdfff, "xvXXff")     \
    DECL (ei_osl_ ## name ## _dvdvv,    "xvvv")       \
    DECL (ei_osl_ ## name ## _dvdvfvf,  "xvvfvf")     \
    DECL (ei_osl_ ## name ## _dvvdfvf,  "xvvXvf")     \
    DECL (ei_osl_ ## name ## _dvdvdfvf, "xvvXvf")

#define GENERIC_PNOISE_DERIV_IMPL(name)            \
    DECL (ei_osl_ ## name ## _dfdff,    "xsXXfXX")    \
    DECL (ei_osl_ ## name ## _dfdfdfff, "xsXXXffXX")  \
    DECL (ei_osl_ ## name ## _dfdvv,    "xsXXvXX")    \
    DECL (ei_osl_ ## name ## _dfdvdfvf, "xsXvXvfXX")  \
    DECL (ei_osl_ ## name ## _dvdff,    "xsvXfXX")    \
    DECL (ei_osl_ ## name ## _dvdfdfff, "xsvXXffXX")  \
    DECL (ei_osl_ ## name ## _dvdvv,    "xsvvvXX")    \
    DECL (ei_osl_ ## name ## _dvdvdfvf, "xsvvXvfXX")

#define UNARY_OP_IMPL(name)                        \
    DECL (ei_osl_ ## name ## _ff,  "ff")              \
    DECL (ei_osl_ ## name ## _dfdf, "xXX")            \
    DECL (ei_osl_ ## name ## _vv,  "xXX")             \
    DECL (ei_osl_ ## name ## _dvdv, "xXX")

#define BINARY_OP_IMPL(name)                       \
    DECL (ei_osl_ ## name ## _fff,    "fff")          \
    DECL (ei_osl_ ## name ## _dfdfdf, "xXXX")         \
    DECL (ei_osl_ ## name ## _dffdf,  "xXfX")         \
    DECL (ei_osl_ ## name ## _dfdff,  "xXXf")         \
    DECL (ei_osl_ ## name ## _vvv,    "xXXX")         \
    DECL (ei_osl_ ## name ## _dvdvdv, "xXXX")         \
    DECL (ei_osl_ ## name ## _dvvdv,  "xXXX")         \
    DECL (ei_osl_ ## name ## _dvdvv,  "xXXX")




DECL (ei_osl_add_closure_closure, "CXCC")
DECL (ei_osl_mul_closure_float, "CXCf")
DECL (ei_osl_mul_closure_color, "CXCc")
DECL (ei_osl_allocate_closure_component, "CXii")
DECL (ei_osl_allocate_weighted_closure_component, "CXiiX")
DECL (ei_osl_closure_to_string, "sXC")
DECL (ei_osl_format, "ss*")
DECL (ei_osl_printf, "xXs*")
DECL (ei_osl_error, "xXs*")
DECL (ei_osl_warning, "xXs*")
DECL (ei_osl_split, "isXsii")
DECL (ei_osl_incr_layers_executed, "xX")

NOISE_IMPL(cellnoise)
//NOISE_DERIV_IMPL(cellnoise)
NOISE_IMPL(noise)
NOISE_DERIV_IMPL(noise)
NOISE_IMPL(snoise)
NOISE_DERIV_IMPL(snoise)
NOISE_IMPL(simplexnoise)
NOISE_DERIV_IMPL(simplexnoise)
NOISE_IMPL(usimplexnoise)
NOISE_DERIV_IMPL(usimplexnoise)
GENERIC_NOISE_DERIV_IMPL(gabornoise)
GENERIC_NOISE_DERIV_IMPL(genericnoise)
PNOISE_IMPL(pcellnoise)
//PNOISE_DERIV_IMPL(pcellnoise)
PNOISE_IMPL(pnoise)
PNOISE_DERIV_IMPL(pnoise)
PNOISE_IMPL(psnoise)
PNOISE_DERIV_IMPL(psnoise)
GENERIC_PNOISE_DERIV_IMPL(gaborpnoise)
GENERIC_PNOISE_DERIV_IMPL(genericpnoise)
DECL (ei_osl_noiseparams_set_anisotropic, "xXi")
DECL (ei_osl_noiseparams_set_do_filter, "xXi")
DECL (ei_osl_noiseparams_set_direction, "xXv")
DECL (ei_osl_noiseparams_set_bandwidth, "xXf")
DECL (ei_osl_noiseparams_set_impulses, "xXf")

DECL (ei_osl_spline_fff, "xXXXXii")
DECL (ei_osl_spline_dfdfdf, "xXXXXii")
DECL (ei_osl_spline_dfdff, "xXXXXii")
DECL (ei_osl_spline_dffdf, "xXXXXii")
DECL (ei_osl_spline_vfv, "xXXXXii")
DECL (ei_osl_spline_dvdfdv, "xXXXXii")
DECL (ei_osl_spline_dvdfv, "xXXXXii")
DECL (ei_osl_spline_dvfdv, "xXXXXii")
DECL (ei_osl_splineinverse_fff, "xXXXXii")
DECL (ei_osl_splineinverse_dfdfdf, "xXXXXii")
DECL (ei_osl_splineinverse_dfdff, "xXXXXii")
DECL (ei_osl_splineinverse_dffdf, "xXXXXii")
DECL (ei_osl_setmessage, "xXsLXisi")
DECL (ei_osl_getmessage, "iXssLXiisi")
DECL (ei_osl_pointcloud_search, "iXsXfiiXXii*")
DECL (ei_osl_pointcloud_get, "iXsXisLX")
DECL (ei_osl_pointcloud_write, "iXsXiXXX")
DECL (ei_osl_pointcloud_write_helper, "xXXXisLX")
DECL (ei_osl_blackbody_vf, "xXXf")
DECL (ei_osl_wavelength_color_vf, "xXXf")
DECL (ei_osl_luminance_fv, "xXXX")
DECL (ei_osl_luminance_dfdv, "xXXX")
DECL (ei_osl_prepend_color_from, "xXXs")
DECL (ei_osl_prepend_matrix_from, "iXXs")
DECL (ei_osl_get_matrix, "iXXs")
DECL (ei_osl_get_inverse_matrix, "iXXs")
DECL (ei_osl_transform_triple, "iXXiXiXXi")
DECL (ei_osl_transform_triple_nonlinear, "iXXiXiXXi")

DECL (ei_osl_dict_find_iis, "iXiX")
DECL (ei_osl_dict_find_iss, "iXXX")
DECL (ei_osl_dict_next, "iXi")
DECL (ei_osl_dict_value, "iXiXLX")
DECL (ei_osl_raytype_name, "iXX")
DECL (ei_osl_range_check, "iiiXXXiXiXX")
DECL (ei_osl_naninf_check, "xiXiXXiXiiX")
DECL (ei_osl_uninit_check, "xLXXXiXiXXiXiXii")
DECL (ei_osl_get_attribute, "iXiXXiiXX")
DECL (ei_osl_bind_interpolated_param, "iXXLiXiXiXi")
DECL (ei_osl_get_texture_options, "XX");
DECL (ei_osl_get_noise_options, "XX");
DECL (ei_osl_get_trace_options, "XX");


// The following are defined inside llvm_ops.cpp. Only include these
// declarations in the OSL_LLVM_NO_BITCODE case.
#ifdef OSL_LLVM_NO_BITCODE
UNARY_OP_IMPL(sin)
UNARY_OP_IMPL(cos)
UNARY_OP_IMPL(tan)
UNARY_OP_IMPL(asin)
UNARY_OP_IMPL(acos)
UNARY_OP_IMPL(atan)
BINARY_OP_IMPL(atan2)
UNARY_OP_IMPL(sinh)
UNARY_OP_IMPL(cosh)
UNARY_OP_IMPL(tanh)

DECL (ei_osl_safe_div_iii, "iii")
DECL (ei_osl_safe_div_fff, "fff")
DECL (ei_osl_safe_mod_iii, "iii")
DECL (ei_osl_sincos_fff, "xfXX")
DECL (ei_osl_sincos_dfdff, "xXXX")
DECL (ei_osl_sincos_dffdf, "xXXX")
DECL (ei_osl_sincos_dfdfdf, "xXXX")
DECL (ei_osl_sincos_vvv, "xXXX")
DECL (ei_osl_sincos_dvdvv, "xXXX")
DECL (ei_osl_sincos_dvvdv, "xXXX")
DECL (ei_osl_sincos_dvdvdv, "xXXX")

UNARY_OP_IMPL(log)
UNARY_OP_IMPL(log2)
UNARY_OP_IMPL(log10)
UNARY_OP_IMPL(exp)
UNARY_OP_IMPL(exp2)
UNARY_OP_IMPL(expm1)
BINARY_OP_IMPL(pow)
UNARY_OP_IMPL(erf)
UNARY_OP_IMPL(erfc)

DECL (ei_osl_pow_vvf, "xXXf")
DECL (ei_osl_pow_dvdvdf, "xXXX")
DECL (ei_osl_pow_dvvdf, "xXXX")
DECL (ei_osl_pow_dvdvf, "xXXf")

UNARY_OP_IMPL(sqrt)
UNARY_OP_IMPL(inversesqrt)

DECL (ei_osl_logb_ff, "ff")
DECL (ei_osl_logb_vv, "xXX")

DECL (ei_osl_floor_ff, "ff")
DECL (ei_osl_floor_vv, "xXX")
DECL (ei_osl_ceil_ff, "ff")
DECL (ei_osl_ceil_vv, "xXX")
DECL (ei_osl_round_ff, "ff")
DECL (ei_osl_round_vv, "xXX")
DECL (ei_osl_trunc_ff, "ff")
DECL (ei_osl_trunc_vv, "xXX")
DECL (ei_osl_sign_ff, "ff")
DECL (ei_osl_sign_vv, "xXX")
DECL (ei_osl_step_fff, "fff")
DECL (ei_osl_step_vvv, "xXXX")

DECL (ei_osl_isnan_if, "if")
DECL (ei_osl_isinf_if, "if")
DECL (ei_osl_isfinite_if, "if")
DECL (ei_osl_abs_ii, "ii")
DECL (ei_osl_fabs_ii, "ii")

UNARY_OP_IMPL(abs)
UNARY_OP_IMPL(fabs)
BINARY_OP_IMPL(fmod)

DECL (ei_osl_smoothstep_ffff, "ffff")
DECL (ei_osl_smoothstep_dfffdf, "xXffX")
DECL (ei_osl_smoothstep_dffdff, "xXfXf")
DECL (ei_osl_smoothstep_dffdfdf, "xXfXX")
DECL (ei_osl_smoothstep_dfdfff, "xXXff")
DECL (ei_osl_smoothstep_dfdffdf, "xXXfX")
DECL (ei_osl_smoothstep_dfdfdff, "xXXXf")
DECL (ei_osl_smoothstep_dfdfdfdf, "xXXXX")

DECL (ei_osl_transform_vmv, "xXXX")
DECL (ei_osl_transform_dvmdv, "xXXX")
DECL (ei_osl_transformv_vmv, "xXXX")
DECL (ei_osl_transformv_dvmdv, "xXXX")
DECL (ei_osl_transformn_vmv, "xXXX")
DECL (ei_osl_transformn_dvmdv, "xXXX")

DECL (ei_osl_dot_fvv, "fXX")
DECL (ei_osl_dot_dfdvdv, "xXXX")
DECL (ei_osl_dot_dfdvv, "xXXX")
DECL (ei_osl_dot_dfvdv, "xXXX")
DECL (ei_osl_cross_vvv, "xXXX")
DECL (ei_osl_cross_dvdvdv, "xXXX")
DECL (ei_osl_cross_dvdvv, "xXXX")
DECL (ei_osl_cross_dvvdv, "xXXX")
DECL (ei_osl_length_fv, "fX")
DECL (ei_osl_length_dfdv, "xXX")
DECL (ei_osl_distance_fvv, "fXX")
DECL (ei_osl_distance_dfdvdv, "xXXX")
DECL (ei_osl_distance_dfdvv, "xXXX")
DECL (ei_osl_distance_dfvdv, "xXXX")
DECL (ei_osl_normalize_vv, "xXX")
DECL (ei_osl_normalize_dvdv, "xXX")
#endif

DECL (ei_osl_mul_mm, "xXXX")
DECL (ei_osl_mul_mf, "xXXf")
DECL (ei_osl_mul_m_ff, "xXff")
DECL (ei_osl_div_mm, "xXXX")
DECL (ei_osl_div_mf, "xXXf")
DECL (ei_osl_div_fm, "xXfX")
DECL (ei_osl_div_m_ff, "xXff")
DECL (ei_osl_get_from_to_matrix, "iXXss")
DECL (ei_osl_transpose_mm, "xXX")
DECL (ei_osl_determinant_fm, "fX")

DECL (ei_osl_concat_sss, "sss")
DECL (ei_osl_strlen_is, "is")
DECL (ei_osl_hash_is, "is")
DECL (ei_osl_getchar_isi, "isi");
DECL (ei_osl_startswith_iss, "iss")
DECL (ei_osl_endswith_iss, "iss")
DECL (ei_osl_stoi_is, "is")
DECL (ei_osl_stof_fs, "fs")
DECL (ei_osl_substr_ssii, "ssii")
DECL (ei_osl_regex_impl, "iXsXisi")

DECL (ei_osl_texture_set_firstchannel, "xXi")
DECL (ei_osl_texture_set_swrap, "xXs")
DECL (ei_osl_texture_set_twrap, "xXs")
DECL (ei_osl_texture_set_rwrap, "xXs")
DECL (ei_osl_texture_set_stwrap, "xXs")
DECL (ei_osl_texture_set_swrap_code, "xXi")
DECL (ei_osl_texture_set_twrap_code, "xXi")
DECL (ei_osl_texture_set_rwrap_code, "xXi")
DECL (ei_osl_texture_set_stwrap_code, "xXi")
DECL (ei_osl_texture_set_sblur, "xXf")
DECL (ei_osl_texture_set_tblur, "xXf")
DECL (ei_osl_texture_set_rblur, "xXf")
DECL (ei_osl_texture_set_stblur, "xXf")
DECL (ei_osl_texture_set_swidth, "xXf")
DECL (ei_osl_texture_set_twidth, "xXf")
DECL (ei_osl_texture_set_rwidth, "xXf")
DECL (ei_osl_texture_set_stwidth, "xXf")
DECL (ei_osl_texture_set_fill, "xXf")
DECL (ei_osl_texture_set_time, "xXf")
DECL (ei_osl_texture_set_interp, "xXs")
DECL (ei_osl_texture_set_interp_code, "xXi")
DECL (ei_osl_texture_set_subimage, "xXi")
DECL (ei_osl_texture_set_subimagename, "xXs")
DECL (ei_osl_texture_set_missingcolor_arena, "xXX")
DECL (ei_osl_texture_set_missingcolor_alpha, "xXif")
DECL (ei_osl_texture, "iXXXXffffffiXXXXXX")
DECL (ei_osl_texture3d, "iXXXXXXXXiXXXXXXXX")
DECL (ei_osl_environment, "iXXXXXXXiXXXXXX")
DECL (ei_osl_get_textureinfo, "iXXXXiiiX")

DECL (ei_osl_trace_set_mindist, "xXf")
DECL (ei_osl_trace_set_maxdist, "xXf")
DECL (ei_osl_trace_set_shade, "xXi")
DECL (ei_osl_trace_set_traceset, "xXs")
DECL (ei_osl_trace, "iXXXXXXXX")

#ifdef OSL_LLVM_NO_BITCODE
DECL (ei_osl_calculatenormal, "xXXX")
DECL (ei_osl_area, "fX")
DECL (ei_osl_filterwidth_fdf, "fX")
DECL (ei_osl_filterwidth_vdv, "xXX")
DECL (ei_osl_raytype_bit, "iXi")
#endif


// Clean up local definitions
#undef NOISE_IMPL
#undef NOISE_DERIV_IMPL
#undef GENERIC_NOISE_DERIV_IMPL
#undef PNOISE_IMPL
#undef PNOISE_DERIV_IMPL
#undef GENERIC_PNOISE_DERIV_IMPL
#undef UNARY_OP_IMPL
#undef BINARY_OP_IMPL
