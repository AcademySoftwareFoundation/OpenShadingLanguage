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
    DECL (name ## _ff,  "ff")              \
    DECL (name ## _fff, "fff")             \
    DECL (name ## _fv,  "fv")              \
    DECL (name ## _fvf, "fvf")             \
    DECL (name ## _vf,  "xvf")             \
    DECL (name ## _vff, "xvff")            \
    DECL (name ## _vv,  "xvv")             \
    DECL (name ## _vvf, "xvvf")

#define NOISE_DERIV_IMPL(name)                     \
    DECL (name ## _dfdf,   "xXX")          \
    DECL (name ## _dfdff,  "xXXf")         \
    DECL (name ## _dffdf,  "xXfX")         \
    DECL (name ## _dfdfdf, "xXXX")         \
    DECL (name ## _dfdv,   "xXv")          \
    DECL (name ## _dfdvf,  "xXvf")         \
    DECL (name ## _dfvdf,  "xXvX")         \
    DECL (name ## _dfdvdf, "xXvX")         \
    DECL (name ## _dvdf,   "xvX")          \
    DECL (name ## _dvdff,  "xvXf")         \
    DECL (name ## _dvfdf,  "xvfX")         \
    DECL (name ## _dvdfdf, "xvXX")         \
    DECL (name ## _dvdv,   "xvv")          \
    DECL (name ## _dvdvf,  "xvvf")         \
    DECL (name ## _dvvdf,  "xvvX")         \
    DECL (name ## _dvdvdf, "xvvX")

#define GENERIC_NOISE_DERIV_IMPL(name)             \
    DECL (name ## _dfdf,   "xsXXXX")       \
    DECL (name ## _dfdfdf, "xsXXXXX")      \
    DECL (name ## _dfdv,   "xsXXXX")       \
    DECL (name ## _dfdvdf, "xsXXXXX")      \
    DECL (name ## _dvdf,   "xsXXXX")       \
    DECL (name ## _dvdfdf, "xsXXXXX")      \
    DECL (name ## _dvdv,   "xsXXXX")       \
    DECL (name ## _dvdvdf, "xsXXXXX")

#define PNOISE_IMPL(name)                          \
    DECL (name ## _fff,   "fff")           \
    DECL (name ## _fffff, "fffff")         \
    DECL (name ## _fvv,   "fvv")           \
    DECL (name ## _fvfvf, "fvfvf")         \
    DECL (name ## _vff,   "xvff")          \
    DECL (name ## _vffff, "xvffff")        \
    DECL (name ## _vvv,   "xvvv")          \
    DECL (name ## _vvfvf, "xvvfvf")

#define PNOISE_DERIV_IMPL(name)                    \
    DECL (name ## _dfdff,    "xXXf")       \
    DECL (name ## _dfdffff,  "xXXfff")     \
    DECL (name ## _dffdfff,  "xXfXff")     \
    DECL (name ## _dfdfdfff, "xXXXff")     \
    DECL (name ## _dfdvv,    "xXXv")       \
    DECL (name ## _dfdvfvf,  "xXvfvf")     \
    DECL (name ## _dfvdfvf,  "xXvXvf")     \
    DECL (name ## _dfdvdfvf, "xXvXvf")     \
    DECL (name ## _dvdff,    "xvXf")       \
    DECL (name ## _dvdffff,  "xvXfff")     \
    DECL (name ## _dvfdfff,  "xvfXff")     \
    DECL (name ## _dvdfdfff, "xvXXff")     \
    DECL (name ## _dvdvv,    "xvvv")       \
    DECL (name ## _dvdvfvf,  "xvvfvf")     \
    DECL (name ## _dvvdfvf,  "xvvXvf")     \
    DECL (name ## _dvdvdfvf, "xvvXvf")

#define GENERIC_PNOISE_DERIV_IMPL(name)            \
    DECL (name ## _dfdff,    "xsXXfXX")    \
    DECL (name ## _dfdfdfff, "xsXXXffXX")  \
    DECL (name ## _dfdvv,    "xsXXvXX")    \
    DECL (name ## _dfdvdfvf, "xsXvXvfXX")  \
    DECL (name ## _dvdff,    "xsvXfXX")    \
    DECL (name ## _dvdfdfff, "xsvXXffXX")  \
    DECL (name ## _dvdvv,    "xsvvvXX")    \
    DECL (name ## _dvdvdfvf, "xsvvXvfXX")

#define UNARY_OP_IMPL(name)                        \
    DECL (name ## _ff,  "ff")              \
    DECL (name ## _dfdf, "xXX")            \
    DECL (name ## _vv,  "xXX")             \
    DECL (name ## _dvdv, "xXX")

#define BINARY_OP_IMPL(name)                       \
    DECL (name ## _fff,    "fff")          \
    DECL (name ## _dfdfdf, "xXXX")         \
    DECL (name ## _dffdf,  "xXfX")         \
    DECL (name ## _dfdff,  "xXXf")         \
    DECL (name ## _vvv,    "xXXX")         \
    DECL (name ## _dvdvdv, "xXXX")         \
    DECL (name ## _dvvdv,  "xXXX")         \
    DECL (name ## _dvdvv,  "xXXX")




DECL (add_closure_closure, "CXCC")
DECL (mul_closure_float, "CXCf")
DECL (mul_closure_color, "CXCc")
DECL (allocate_closure_component, "CXii")
DECL (allocate_weighted_closure_component, "CXiiX")
DECL (closure_to_string, "sXC")
DECL (format, "ss*")
DECL (printf, "xXs*")
DECL (error, "xXs*")
DECL (warning, "xXs*")
DECL (split, "isXsii")
DECL (incr_layers_executed, "xX")

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
NOISE_IMPL(nullnoise)
NOISE_DERIV_IMPL(nullnoise)
NOISE_IMPL(unullnoise)
NOISE_DERIV_IMPL(unullnoise)
PNOISE_IMPL(pcellnoise)
//PNOISE_DERIV_IMPL(pcellnoise)
PNOISE_IMPL(pnoise)
PNOISE_DERIV_IMPL(pnoise)
PNOISE_IMPL(psnoise)
PNOISE_DERIV_IMPL(psnoise)
GENERIC_PNOISE_DERIV_IMPL(gaborpnoise)
GENERIC_PNOISE_DERIV_IMPL(genericpnoise)
DECL (noiseparams_set_anisotropic, "xXi")
DECL (noiseparams_set_do_filter, "xXi")
DECL (noiseparams_set_direction, "xXv")
DECL (noiseparams_set_bandwidth, "xXf")
DECL (noiseparams_set_impulses, "xXf")
DECL (count_noise, "xX")

DECL (spline_fff, "xXXXXii")
DECL (spline_dfdfdf, "xXXXXii")
DECL (spline_dfdff, "xXXXXii")
DECL (spline_dffdf, "xXXXXii")
DECL (spline_vfv, "xXXXXii")
DECL (spline_dvdfdv, "xXXXXii")
DECL (spline_dvdfv, "xXXXXii")
DECL (spline_dvfdv, "xXXXXii")
DECL (splineinverse_fff, "xXXXXii")
DECL (splineinverse_dfdfdf, "xXXXXii")
DECL (splineinverse_dfdff, "xXXXXii")
DECL (splineinverse_dffdf, "xXXXXii")
DECL (setmessage, "xXsLXisi")
DECL (getmessage, "iXssLXiisi")
DECL (pointcloud_search, "iXsXfiiXXii*")
DECL (pointcloud_get, "iXsXisLX")
DECL (pointcloud_write, "iXsXiXXX")
DECL (pointcloud_write_helper, "xXXXisLX")
DECL (blackbody_vf, "xXXf")
DECL (wavelength_color_vf, "xXXf")
DECL (luminance_fv, "xXXX")
DECL (luminance_dfdv, "xXXX")
DECL (prepend_color_from, "xXXs")
DECL (prepend_matrix_from, "iXXs")
DECL (get_matrix, "iXXs")
DECL (get_inverse_matrix, "iXXs")
DECL (transform_triple, "iXXiXiXXi")
DECL (transform_triple_nonlinear, "iXXiXiXXi")

DECL (dict_find_iis, "iXiX")
DECL (dict_find_iss, "iXXX")
DECL (dict_next, "iXi")
DECL (dict_value, "iXiXLX")
DECL (raytype_name, "iXX")
DECL (range_check, "iiiXXXiXiXX")
DECL (naninf_check, "xiXiXXiXiiX")
DECL (uninit_check, "xLXXXiXiXXiXiXii")
DECL (get_attribute, "iXiXXiiXX")
DECL (bind_interpolated_param, "iXXLiXiXiXi")
DECL (get_texture_options, "XX");
DECL (get_noise_options, "XX");
DECL (get_trace_options, "XX");


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

DECL (safe_div_iii, "iii")
DECL (safe_div_fff, "fff")
DECL (safe_mod_iii, "iii")
DECL (sincos_fff, "xfXX")
DECL (sincos_dfdff, "xXXX")
DECL (sincos_dffdf, "xXXX")
DECL (sincos_dfdfdf, "xXXX")
DECL (sincos_vvv, "xXXX")
DECL (sincos_dvdvv, "xXXX")
DECL (sincos_dvvdv, "xXXX")
DECL (sincos_dvdvdv, "xXXX")

UNARY_OP_IMPL(log)
UNARY_OP_IMPL(log2)
UNARY_OP_IMPL(log10)
UNARY_OP_IMPL(exp)
UNARY_OP_IMPL(exp2)
UNARY_OP_IMPL(expm1)
BINARY_OP_IMPL(pow)
UNARY_OP_IMPL(erf)
UNARY_OP_IMPL(erfc)

DECL (pow_vvf, "xXXf")
DECL (pow_dvdvdf, "xXXX")
DECL (pow_dvvdf, "xXXX")
DECL (pow_dvdvf, "xXXf")

UNARY_OP_IMPL(sqrt)
UNARY_OP_IMPL(inversesqrt)

DECL (logb_ff, "ff")
DECL (logb_vv, "xXX")

DECL (floor_ff, "ff")
DECL (floor_vv, "xXX")
DECL (ceil_ff, "ff")
DECL (ceil_vv, "xXX")
DECL (round_ff, "ff")
DECL (round_vv, "xXX")
DECL (trunc_ff, "ff")
DECL (trunc_vv, "xXX")
DECL (sign_ff, "ff")
DECL (sign_vv, "xXX")
DECL (step_fff, "fff")
DECL (step_vvv, "xXXX")

DECL (isnan_if, "if")
DECL (isinf_if, "if")
DECL (isfinite_if, "if")
DECL (abs_ii, "ii")
DECL (fabs_ii, "ii")

UNARY_OP_IMPL(abs)
UNARY_OP_IMPL(fabs)
BINARY_OP_IMPL(fmod)

DECL (smoothstep_ffff, "ffff")
DECL (smoothstep_dfffdf, "xXffX")
DECL (smoothstep_dffdff, "xXfXf")
DECL (smoothstep_dffdfdf, "xXfXX")
DECL (smoothstep_dfdfff, "xXXff")
DECL (smoothstep_dfdffdf, "xXXfX")
DECL (smoothstep_dfdfdff, "xXXXf")
DECL (smoothstep_dfdfdfdf, "xXXXX")

DECL (transform_vmv, "xXXX")
DECL (transform_dvmdv, "xXXX")
DECL (transformv_vmv, "xXXX")
DECL (transformv_dvmdv, "xXXX")
DECL (transformn_vmv, "xXXX")
DECL (transformn_dvmdv, "xXXX")

DECL (dot_fvv, "fXX")
DECL (dot_dfdvdv, "xXXX")
DECL (dot_dfdvv, "xXXX")
DECL (dot_dfvdv, "xXXX")
DECL (cross_vvv, "xXXX")
DECL (cross_dvdvdv, "xXXX")
DECL (cross_dvdvv, "xXXX")
DECL (cross_dvvdv, "xXXX")
DECL (length_fv, "fX")
DECL (length_dfdv, "xXX")
DECL (distance_fvv, "fXX")
DECL (distance_dfdvdv, "xXXX")
DECL (distance_dfdvv, "xXXX")
DECL (distance_dfvdv, "xXXX")
DECL (normalize_vv, "xXX")
DECL (normalize_dvdv, "xXX")
#endif

DECL (mul_mm, "xXXX")
DECL (mul_mf, "xXXf")
DECL (mul_m_ff, "xXff")
DECL (div_mm, "xXXX")
DECL (div_mf, "xXXf")
DECL (div_fm, "xXfX")
DECL (div_m_ff, "xXff")
DECL (get_from_to_matrix, "iXXss")
DECL (transpose_mm, "xXX")
DECL (determinant_fm, "fX")

DECL (concat_sss, "sss")
DECL (strlen_is, "is")
DECL (hash_is, "is")
DECL (getchar_isi, "isi");
DECL (startswith_iss, "iss")
DECL (endswith_iss, "iss")
DECL (stoi_is, "is")
DECL (stof_fs, "fs")
DECL (substr_ssii, "ssii")
DECL (regex_impl, "iXsXisi")

DECL (texture_set_firstchannel, "xXi")
DECL (texture_set_swrap, "xXs")
DECL (texture_set_twrap, "xXs")
DECL (texture_set_rwrap, "xXs")
DECL (texture_set_stwrap, "xXs")
DECL (texture_set_swrap_code, "xXi")
DECL (texture_set_twrap_code, "xXi")
DECL (texture_set_rwrap_code, "xXi")
DECL (texture_set_stwrap_code, "xXi")
DECL (texture_set_sblur, "xXf")
DECL (texture_set_tblur, "xXf")
DECL (texture_set_rblur, "xXf")
DECL (texture_set_stblur, "xXf")
DECL (texture_set_swidth, "xXf")
DECL (texture_set_twidth, "xXf")
DECL (texture_set_rwidth, "xXf")
DECL (texture_set_stwidth, "xXf")
DECL (texture_set_fill, "xXf")
DECL (texture_set_time, "xXf")
DECL (texture_set_interp, "xXs")
DECL (texture_set_interp_code, "xXi")
DECL (texture_set_subimage, "xXi")
DECL (texture_set_subimagename, "xXs")
DECL (texture_set_missingcolor_arena, "xXX")
DECL (texture_set_missingcolor_alpha, "xXif")
DECL (texture, "iXXXXffffffiXXXXXXX")
DECL (texture3d, "iXXXXXXXXiXXXXXXXXX")
DECL (environment, "iXXXXXXXiXXXXXXX")
DECL (get_textureinfo, "iXXXXiiiX")

DECL (trace_set_mindist, "xXf")
DECL (trace_set_maxdist, "xXf")
DECL (trace_set_shade, "xXi")
DECL (trace_set_traceset, "xXs")
DECL (trace, "iXXXXXXXX")

#ifdef OSL_LLVM_NO_BITCODE
DECL (calculatenormal, "xXXX")
DECL (area, "fX")
DECL (filterwidth_fdf, "fX")
DECL (filterwidth_vdv, "xXX")
DECL (raytype_bit, "iXi")
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
