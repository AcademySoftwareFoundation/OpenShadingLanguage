// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


// This file contains "declarations" for all the functions that might get
// called from JITed shader code. But the declaration itself is dependent on
// the DECL macro, which should be declared by the outer file prior to
// including this file. Thus, this list may be repurposed and included
// multiple times, with different DECL definitions.


#ifndef DECL
#    error Do not include this file unless DECL is defined
#endif

// NOTE: the trailing 'i' at the end of the signature
// is an 32 bit integer representing the mask (on bits are active lanes)

#define WIDE_NOISE_IMPL_INDIRECT(name)                  \
    DECL(__OSL_MASKED_OP2(name, Wf, Wf), "WfWfi")       \
    DECL(__OSL_MASKED_OP2(name, Wf, Wv), "WfWvi")       \
    DECL(__OSL_MASKED_OP2(name, Wv, Wv), "WvWvi")       \
    DECL(__OSL_MASKED_OP2(name, Wv, Wf), "WvWfi")       \
    DECL(__OSL_MASKED_OP3(name, Wv, Wv, Wf), "WvWvWfi") \
    DECL(__OSL_MASKED_OP3(name, Wf, Wf, Wf), "WfWfWfi") \
    DECL(__OSL_MASKED_OP3(name, Wf, Wv, Wf), "WfWvWfi") \
    DECL(__OSL_MASKED_OP3(name, Wv, Wf, Wf), "WvWfWfi")


#define WIDE_NOISE_IMPL(name) WIDE_NOISE_IMPL_INDIRECT(name)


#define WIDE_NOISE_DERIV_IMPL_INDIRECT(name)             \
    DECL(__OSL_MASKED_OP2(name, Wdf, Wdf), "xXXi")       \
    DECL(__OSL_MASKED_OP3(name, Wdf, Wdf, Wdf), "xXXXi") \
    DECL(__OSL_MASKED_OP2(name, Wdf, Wdv), "xXXi")       \
    DECL(__OSL_MASKED_OP3(name, Wdf, Wdv, Wdf), "xXXXi") \
    DECL(__OSL_MASKED_OP2(name, Wdv, Wdf), "xXXi")       \
    DECL(__OSL_MASKED_OP3(name, Wdv, Wdf, Wdf), "xXXXi") \
    DECL(__OSL_MASKED_OP2(name, Wdv, Wdv), "xXXi")       \
    DECL(__OSL_MASKED_OP3(name, Wdv, Wdv, Wdf), "xXXXi")

#define WIDE_NOISE_DERIV_IMPL(name) WIDE_NOISE_DERIV_IMPL_INDIRECT(name)


#define WIDE_GENERIC_NOISE_DERIV_IMPL_INDIRECT(name)         \
    DECL(__OSL_MASKED_OP2(name, Wdf, Wdf), "xsXXXXXi")       \
    DECL(__OSL_MASKED_OP3(name, Wdf, Wdf, Wdf), "xsXXXXXXi") \
    DECL(__OSL_MASKED_OP2(name, Wdf, Wdv), "xsXXXXXi")       \
    DECL(__OSL_MASKED_OP3(name, Wdf, Wdv, Wdf), "xsXXXXXXi") \
    DECL(__OSL_MASKED_OP2(name, Wdv, Wdf), "xsXXXXXi")       \
    DECL(__OSL_MASKED_OP3(name, Wdv, Wdf, Wdf), "xsXXXXXXi") \
    DECL(__OSL_MASKED_OP2(name, Wdv, Wdv), "xsXXXXXi")       \
    DECL(__OSL_MASKED_OP3(name, Wdv, Wdv, Wdf), "xsXXXXXXi")

#define WIDE_GENERIC_NOISE_DERIV_IMPL(name) \
    WIDE_GENERIC_NOISE_DERIV_IMPL_INDIRECT(name)


#define WIDE_PNOISE_IMPL_INDIRECT(name)                             \
    DECL(__OSL_MASKED_OP3(name, Wf, Wf, Wf), "WfWfWfi")             \
    DECL(__OSL_MASKED_OP5(name, Wf, Wf, Wf, Wf, Wf), "WfWfWfWfWfi") \
    DECL(__OSL_MASKED_OP3(name, Wf, Wv, Wv), "WfWvWvi")             \
    DECL(__OSL_MASKED_OP5(name, Wf, Wv, Wf, Wv, Wf), "WfWvWfWvWfi") \
    DECL(__OSL_MASKED_OP3(name, Wv, Wf, Wf), "WvWfWfi")             \
    DECL(__OSL_MASKED_OP5(name, Wv, Wf, Wf, Wf, Wf), "WvWfWfWfWfi") \
    DECL(__OSL_MASKED_OP3(name, Wv, Wv, Wv), "WvWvWvi")             \
    DECL(__OSL_MASKED_OP5(name, Wv, Wv, Wf, Wv, Wf), "WvWvWfWvWfi")


#define WIDE_PNOISE_IMPL(name) WIDE_PNOISE_IMPL_INDIRECT(name)


// NOTE:  any constants passed along with other derivative parameters,
// are promoted to wide derivatives on the stack by the code generator
// the upshot of this is that one parameter is a derivative/wide then
// there will be no combinations with any non-wide parameters
#define WIDE_PNOISE_DERIV_IMPL_INDIRECT(name)                      \
    DECL(__OSL_MASKED_OP3(name, Wdf, Wdf, Wf), "xXXXi")            \
    DECL(__OSL_MASKED_OP5(name, Wdf, Wdf, Wdf, Wf, Wf), "xXXXXXi") \
    DECL(__OSL_MASKED_OP3(name, Wdf, Wdv, Wv), "xXXXi")            \
    DECL(__OSL_MASKED_OP5(name, Wdf, Wdv, Wdf, Wv, Wf), "xXXXXXi") \
    DECL(__OSL_MASKED_OP3(name, Wdv, Wdf, Wf), "xXXXi")            \
    DECL(__OSL_MASKED_OP5(name, Wdv, Wdf, Wdf, Wf, Wf), "xXXXXXi") \
    DECL(__OSL_MASKED_OP3(name, Wdv, Wdv, Wv), "xXXXi")            \
    DECL(__OSL_MASKED_OP5(name, Wdv, Wdv, Wdf, Wv, Wf), "xXXXXXi")

#define WIDE_PNOISE_DERIV_IMPL(name) WIDE_PNOISE_DERIV_IMPL_INDIRECT(name)


#define WIDE_GENERIC_PNOISE_DERIV_IMPL_INDIRECT(name)                  \
    DECL(__OSL_MASKED_OP3(name, Wdf, Wdf, Wf), "xsXXXXXXi")            \
    DECL(__OSL_MASKED_OP5(name, Wdf, Wdf, Wdf, Wf, Wf), "xsXXXXXXXXi") \
    DECL(__OSL_MASKED_OP3(name, Wdf, Wdv, Wv), "xsXXXXXXi")            \
    DECL(__OSL_MASKED_OP5(name, Wdf, Wdv, Wdf, Wv, Wf), "xsXXXXXXXXi") \
    DECL(__OSL_MASKED_OP3(name, Wdv, Wdf, Wf), "xsXXXXXXi")            \
    DECL(__OSL_MASKED_OP5(name, Wdv, Wdf, Wdf, Wf, Wf), "xsXXXXXXXXi") \
    DECL(__OSL_MASKED_OP3(name, Wdv, Wdv, Wv), "xsXXXXXXi")            \
    DECL(__OSL_MASKED_OP5(name, Wdv, Wdv, Wdf, Wv, Wf), "xsXXXXXXXXi")

#define WIDE_GENERIC_PNOISE_DERIV_IMPL(name) \
    WIDE_GENERIC_PNOISE_DERIV_IMPL_INDIRECT(name)


#define WIDE_UNARY_F_OP_IMPL(name)               \
    DECL(__OSL_OP2(name, Wf, Wf), "xXX")         \
    DECL(__OSL_MASKED_OP2(name, Wf, Wf), "xXXi") \
    DECL(__OSL_OP2(name, Wdf, Wdf), "xXX")       \
    DECL(__OSL_MASKED_OP2(name, Wdf, Wdf), "xXXi")

#define WIDE_UNARY_OP_IMPL(name)                 \
    WIDE_UNARY_F_OP_IMPL(name)                   \
    DECL(__OSL_OP2(name, Wv, Wv), "xXX")         \
    DECL(__OSL_MASKED_OP2(name, Wv, Wv), "xXXi") \
    DECL(__OSL_OP2(name, Wdv, Wdv), "xXX")       \
    DECL(__OSL_MASKED_OP2(name, Wdv, Wdv), "xXXi")


#define WIDE_UNARY_I_OP_IMPL(name)       \
    DECL(__OSL_OP2(name, Wi, Wi), "xXX") \
    DECL(__OSL_MASKED_OP2(name, Wi, Wi), "xXXi")


#define WIDE_TEST_F_OP_IMPL(name)        \
    DECL(__OSL_OP2(name, Wi, Wf), "xXX") \
    DECL(__OSL_MASKED_OP2(name, Wi, Wf), "xXXi")


#define WIDE_UNARY_F_OR_V_OP_IMPL(name)          \
    DECL(__OSL_OP2(name, Wf, Wf), "xXX")         \
    DECL(__OSL_MASKED_OP2(name, Wf, Wf), "xXXi") \
    DECL(__OSL_OP2(name, Wv, Wv), "xXX")         \
    DECL(__OSL_MASKED_OP2(name, Wv, Wv), "xXXi")

#define WIDE_BINARY_OP_MASKED_IMPL(name)                 \
    DECL(__OSL_MASKED_OP3(name, Wf, Wf, Wf), "xXXXi")    \
    DECL(__OSL_MASKED_OP3(name, Wdf, Wdf, Wdf), "xXXXi") \
    DECL(__OSL_MASKED_OP3(name, Wdf, Wf, Wdf), "xXXXi")  \
    DECL(__OSL_MASKED_OP3(name, Wdf, Wdf, Wf), "xXXXi")  \
    DECL(__OSL_MASKED_OP3(name, Wv, Wv, Wv), "xXXXi")    \
    DECL(__OSL_MASKED_OP3(name, Wdv, Wdv, Wdv), "xXXXi") \
    DECL(__OSL_MASKED_OP3(name, Wdv, Wv, Wdv), "xXXXi")  \
    DECL(__OSL_MASKED_OP3(name, Wdv, Wdv, Wv), "xXXXi")

#define WIDE_BINARY_OP_IMPL(name)                \
    DECL(__OSL_OP3(name, Wf, Wf, Wf), "xXXX")    \
    DECL(__OSL_OP3(name, Wdf, Wdf, Wdf), "xXXX") \
    DECL(__OSL_OP3(name, Wdf, Wf, Wdf), "xXXX")  \
    DECL(__OSL_OP3(name, Wdf, Wdf, Wf), "xXXX")  \
    DECL(__OSL_OP3(name, Wv, Wv, Wv), "xXXX")    \
    DECL(__OSL_OP3(name, Wdv, Wdv, Wdv), "xXXX") \
    DECL(__OSL_OP3(name, Wdv, Wv, Wdv), "xXXX")  \
    DECL(__OSL_OP3(name, Wdv, Wdv, Wv), "xXXX")  \
    WIDE_BINARY_OP_MASKED_IMPL(name)


#define WIDE_BINARY_VF_OP_MASKED_IMPL(name)              \
    DECL(__OSL_MASKED_OP3(name, Wv, Wv, Wf), "xXXXi")    \
    DECL(__OSL_MASKED_OP3(name, Wdv, Wdv, Wdf), "xXXXi") \
    DECL(__OSL_MASKED_OP3(name, Wdv, Wdv, Wf), "xXXXi")  \
    DECL(__OSL_MASKED_OP3(name, Wdv, Wv, Wdf), "xXXXi")


#define WIDE_BINARY_F_OR_V_OP_IMPL(name)              \
    DECL(__OSL_OP3(name, Wf, Wf, Wf), "xXXX")         \
    DECL(__OSL_MASKED_OP3(name, Wf, Wf, Wf), "xXXXi") \
    DECL(__OSL_OP3(name, Wv, Wv, Wv), "xXXX")         \
    DECL(__OSL_MASKED_OP3(name, Wv, Wv, Wv), "xXXXi")

DECL(__OSL_MASKED_OP(allocate_closure_component), "xXXiii")
DECL(__OSL_MASKED_OP(allocate_weighted_closure_component), "xXXiiXi")
DECL(__OSL_MASKED_OP(add_closure_closure), "xXXXXi")
DECL(__OSL_MASKED_OP(mul_closure_float), "xXXXXi")
DECL(__OSL_MASKED_OP(mul_closure_color), "xXXXXi")
DECL(__OSL_OP(closure_to_string), "sXX")

DECL(__OSL_OP(format), "xXis*")
DECL(__OSL_OP(printf), "xXis*")
DECL(__OSL_OP(error), "xXis*")
DECL(__OSL_OP(warning), "xXis*")
DECL(__OSL_OP(fprintf), "xXiss*")

DECL(__OSL_MASKED_OP(split), "xXXXXXii")

// DECL (osl_incr_layers_executed, "xX") // original used by wide currently

WIDE_NOISE_IMPL(cellnoise)
// commented out in non-wide, there is no derivative version of cellnoise
//WIDE_NOISE_DERIV_IMPL(cellnoise)

WIDE_NOISE_IMPL(hashnoise)
// There is no derivative version of hashnoise
//WIDE_NOISE_DERIV_IMPL(hashnoise)

WIDE_NOISE_IMPL(noise)
WIDE_NOISE_DERIV_IMPL(noise)

WIDE_NOISE_IMPL(snoise)
WIDE_NOISE_DERIV_IMPL(snoise)

WIDE_NOISE_IMPL(simplexnoise)
WIDE_NOISE_DERIV_IMPL(simplexnoise)

WIDE_NOISE_IMPL(usimplexnoise)
WIDE_NOISE_DERIV_IMPL(usimplexnoise)


WIDE_PNOISE_IMPL(pnoise)
WIDE_PNOISE_DERIV_IMPL(pnoise)

WIDE_PNOISE_IMPL(psnoise)
WIDE_PNOISE_DERIV_IMPL(psnoise)

WIDE_PNOISE_IMPL(pcellnoise)
// commented out in non-wide, there is no derivative version of pcellnoise
//WIDE_PNOISE_DERIV_IMPL(pcellnoise)

WIDE_PNOISE_IMPL(phashnoise)
// commented out in non-wide, there is no derivative version of phashnoise
//WIDE_PNOISE_DERIV_IMPL(phashnoise)


WIDE_GENERIC_NOISE_DERIV_IMPL(gabornoise)
WIDE_GENERIC_PNOISE_DERIV_IMPL(gaborpnoise)

WIDE_GENERIC_NOISE_DERIV_IMPL(genericnoise)
WIDE_GENERIC_PNOISE_DERIV_IMPL(genericpnoise)

WIDE_NOISE_IMPL(nullnoise)
WIDE_NOISE_DERIV_IMPL(nullnoise)

WIDE_NOISE_IMPL(unullnoise)
WIDE_NOISE_DERIV_IMPL(unullnoise)

//DECL (osl_noiseparams_set_anisotropic, "xXi") // share non-wide impl
//DECL (osl_noiseparams_set_do_filter, "xXi") // share non-wide impl
//DECL (osl_noiseparams_set_direction, "xXv") // share non-wide impl
//DECL (osl_noiseparams_set_bandwidth, "xXf") // share non-wide impl
//DECL (osl_noiseparams_set_impulses, "xXf")  // share non-wide impl

DECL(__OSL_MASKED_OP(count_noise), "xXi")

DECL(__OSL_OP2(hash, Wi, Wi), "xXX")
DECL(__OSL_OP2(hash, Wi, Wf), "xXX")
DECL(__OSL_OP3(hash, Wi, Wf, Wf), "xXXX")
DECL(__OSL_OP2(hash, Wi, Wv), "xXX")
DECL(__OSL_OP3(hash, Wi, Wv, Wf), "xXXX")
DECL(__OSL_MASKED_OP2(hash, Wi, Wi), "xXXi")
DECL(__OSL_MASKED_OP2(hash, Wi, Wf), "xXXi")
DECL(__OSL_MASKED_OP3(hash, Wi, Wf, Wf), "xXXXi")
DECL(__OSL_MASKED_OP2(hash, Wi, Wv), "xXXi")
DECL(__OSL_MASKED_OP3(hash, Wi, Wv, Wf), "xXXXi")

// Need wide for combinations of the 3 parameters allowed to be uniform
// caveat, some combos are unreachable/unneeded
// When result has a derivative, there is
// no "easy" have a input parameter be non-derivative based on code
// analysis promoting all inputs to be derivative base.
// Some exceptions are possible, such a directly passing a shader global
// Those cases can be worked around by creating a variable on the stack
// first vs. directly passing the shader global.  We don't expect this
// to be encountered, but is possible

DECL(__OSL_MASKED_OP3(spline, Wf, Wf, Wf), "xXXXXiii")
DECL(__OSL_MASKED_OP3(spline, Wf, Wf, f), "xXXXXiii")
DECL(__OSL_MASKED_OP3(spline, Wf, f, Wf), "xXXXXiii")


DECL(__OSL_MASKED_OP3(spline, Wdf, Wdf, Wdf), "xXXXXiii")
DECL(__OSL_MASKED_OP3(spline, Wdf, Wdf, df), "xXXXXiii")
DECL(__OSL_MASKED_OP3(spline, Wdf, Wf, df), "xXXXXiii")
DECL(__OSL_MASKED_OP3(spline, Wdf, df, Wdf), "xXXXXiii")

DECL(__OSL_MASKED_OP3(spline, Wdf, Wdf, f), "xXXXXiii")

DECL(__OSL_MASKED_OP3(spline, Wv, Wf, Wv), "xXXXXiii")
DECL(__OSL_MASKED_OP3(spline, Wv, Wf, v), "xXXXXiii")
DECL(__OSL_MASKED_OP3(spline, Wv, f, Wv), "xXXXXiii")

DECL(__OSL_MASKED_OP3(spline, Wdv, Wdf, Wdv), "xXXXXiii")
DECL(__OSL_MASKED_OP3(spline, Wdv, Wdf, dv), "xXXXXiii")
DECL(__OSL_MASKED_OP3(spline, Wdv, df, Wdv), "xXXXXiii")

DECL(__OSL_MASKED_OP3(spline, Wdv, Wdf, v), "xXXXXiii")
DECL(__OSL_MASKED_OP3(spline, Wdv, Wdf, Wv), "xXXXXiii")
DECL(__OSL_MASKED_OP3(spline, Wdv, df, Wv), "xXXXXiii")

DECL(__OSL_MASKED_OP3(spline, Wdf, f, Wdf), "xXXXXiii")
DECL(__OSL_MASKED_OP3(spline, Wdf, Wf, Wdf), "xXXXXiii")

DECL(__OSL_MASKED_OP3(spline, Wdv, f, Wdv), "xXXXXiii")
DECL(__OSL_MASKED_OP3(spline, Wdv, Wf, Wdv), "xXXXXiii")
DECL(__OSL_MASKED_OP3(spline, Wdv, Wf, dv), "xXXXXiii")

//---------------------------------------------------------------
DECL(__OSL_MASKED_OP3(splineinverse, Wf, Wf, Wf), "xXXXXiii")
DECL(__OSL_MASKED_OP3(splineinverse, Wf, Wf, f), "xXXXXiii")
DECL(__OSL_MASKED_OP3(splineinverse, Wf, f, Wf), "xXXXXiii")

//dfdfdf is treated as dfdff
DECL(__OSL_MASKED_OP3(splineinverse, Wdf, Wdf, Wdf), "xXXXXiii")  //redone
DECL(__OSL_MASKED_OP3(splineinverse, Wdf, Wdf, df), "xXXXXiii")
DECL(__OSL_MASKED_OP3(splineinverse, Wdf, df, Wdf), "xXXXXiii")
//======
DECL(__OSL_MASKED_OP3(splineinverse, Wdf, Wdf, f), "xXXXXiii")

//dffdf is treated as fff
DECL(__OSL_MASKED_OP3(splineinverse, Wdf, f, Wdf), "xXXXXiii")
// // unreachable, can't find .osl to produce this combination
//DECL(__OSL_MASKED_OP3(splineinverse, Wdf, Wf, Wdf), "xXXXXiii")

DECL(__OSL_MASKED_OP(pointcloud_search), "xXXsXXiiXiXiiii*")
DECL(__OSL_MASKED_OP(pointcloud_get), "iXsXiXsLXi")
DECL(__OSL_MASKED_OP(pointcloud_write), "iXsXiXXXi")

DECL(__OSL_MASKED_OP(getmessage), "xXXssLXiisii")
DECL(__OSL_MASKED_OP2(setmessage, s, WX), "xXsLXisii")
DECL(__OSL_MASKED_OP2(setmessage, Ws, WX), "xXXLXisii")

DECL(__OSL_OP(blackbody_vf), "xXXf")
DECL(__OSL_MASKED_OP2(blackbody, Wv, Wf), "xXXXi")

DECL(__OSL_OP(wavelength_color_vf), "xXXf")
DECL(__OSL_MASKED_OP2(wavelength_color, Wv, Wf), "xXXXi")

// Code generator handles luminance by itself
//DECL (osl_luminance_fv, "xXXX")
//DECL (osl_luminance_dfdv, "xXXX")

DECL(__OSL_OP(prepend_color_from_vs), "xXXs")
DECL(__OSL_MASKED_OP2(prepend_color_from, Wv, s), "xXXsi")
DECL(__OSL_MASKED_OP2(prepend_color_from, Wv, Ws), "xXXXi")


// forced masked version only
DECL(__OSL_MASKED_OP2(prepend_matrix_from, Wm, s), "xXXsi")
DECL(__OSL_MASKED_OP2(prepend_matrix_from, Wm, Ws), "xXXXi")

// Batched code gen uses a combination of osl_build_transform_matrix
// with osl_transform_[point|vector|normal] to do the follow functions
// DECL (osl_get_matrix, "iXXs")  // unneeded
// DECL (osl_get_inverse_matrix, "iXXs") // unneeded
// DECL (osl_transform_triple, "iXXiXiXXi") // unneeded
// DECL (osl_transform_triple_nonlinear, "iXXiXiXXi") // unneeded

DECL(__OSL_MASKED_OP3(build_transform_matrix, Wm, s, s), "iXXXXi")
DECL(__OSL_MASKED_OP3(build_transform_matrix, Wm, Ws, s), "iXXXXi")
DECL(__OSL_MASKED_OP3(build_transform_matrix, Wm, s, Ws), "iXXXXi")
DECL(__OSL_MASKED_OP3(build_transform_matrix, Wm, Ws, Ws), "iXXXXi")

DECL(__OSL_OP(dict_find_iis), "iXiX")
DECL(__OSL_MASKED_OP3(dict_find, Wi, Wi, Ws), "xXXXXi")

DECL(__OSL_OP(dict_find_iss), "iXXX")
DECL(__OSL_MASKED_OP3(dict_find, Wi, Ws, Ws), "xXXXXi")

DECL(__OSL_OP(dict_next), "iXi")
DECL(__OSL_MASKED_OP(dict_next), "xXXXi")

DECL(__OSL_OP(dict_value), "iXiXLX")
DECL(__OSL_MASKED_OP(dict_value), "xXXXXLXi")


DECL(__OSL_OP(raytype_name), "iXs")
DECL(__OSL_MASKED_OP(raytype_name), "xXXXi")
DECL(__OSL_OP(naninf_check), "xiXiXsisiis")
DECL(__OSL_MASKED_OP1(naninf_check_offset, i), "xiiXiXsisiis")
DECL(__OSL_MASKED_OP1(naninf_check_offset, Wi), "xiiXiXsisXis")
DECL(__OSL_OP(range_check), "iiisXsisiss")
DECL(__OSL_MASKED_OP(range_check), "xXiisXsisiss")
DECL(__OSL_OP2(uninit_check_values_offset, X, i), "xLXXsisissisisii")
DECL(__OSL_MASKED_OP2(uninit_check_values_offset, X, Wi), "xiLXXsisissisisXi")
DECL(__OSL_MASKED_OP2(uninit_check_values_offset, WX, i), "xiLXXsisissisisii")
DECL(__OSL_MASKED_OP2(uninit_check_values_offset, WX, Wi), "xiLXXsisissisisXi")

DECL(__OSL_OP1(get_attribute, s), "iXissiiXXi")
DECL(__OSL_MASKED_OP1(get_attribute, Ws), "iXisXiiXXi")
DECL(__OSL_OP(get_attribute_uniform), "iXissiiXX")

// TODO:  shouldn't bind_interpolated_param be MASKED?  change name to reflect
DECL(__OSL_OP(bind_interpolated_param), "iXsLiXiXiXii")

//DECL (osl_get_texture_options, "XX") // unneeded
DECL(__OSL_OP(get_noise_options), "XX")

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

DECL(__OSL_OP3(sincos, Wf, Wf, Wf), "xXXX")
DECL(__OSL_OP3(sincos, Wdf, Wdf, Wf), "xXXX")
DECL(__OSL_OP3(sincos, Wdf, Wf, Wdf), "xXXX")
DECL(__OSL_OP3(sincos, Wdf, Wdf, Wdf), "xXXX")

DECL(__OSL_OP3(sincos, Wv, Wv, Wv), "xXXX")
DECL(__OSL_OP3(sincos, Wdv, Wdv, Wv), "xXXX")
DECL(__OSL_OP3(sincos, Wdv, Wv, Wdv), "xXXX")
DECL(__OSL_OP3(sincos, Wdv, Wdv, Wdv), "xXXX")

DECL(__OSL_MASKED_OP3(sincos, Wf, Wf, Wf), "xXXXi")
DECL(__OSL_MASKED_OP3(sincos, Wdf, Wdf, Wf), "xXXXi")
DECL(__OSL_MASKED_OP3(sincos, Wdf, Wf, Wdf), "xXXXi")
DECL(__OSL_MASKED_OP3(sincos, Wdf, Wdf, Wdf), "xXXXi")

DECL(__OSL_MASKED_OP3(sincos, Wv, Wv, Wv), "xXXXi")
DECL(__OSL_MASKED_OP3(sincos, Wdv, Wdv, Wv), "xXXXi")
DECL(__OSL_MASKED_OP3(sincos, Wdv, Wv, Wdv), "xXXXi")
DECL(__OSL_MASKED_OP3(sincos, Wdv, Wdv, Wdv), "xXXXi")

WIDE_UNARY_OP_IMPL(log)
WIDE_UNARY_OP_IMPL(log2)
WIDE_UNARY_OP_IMPL(log10)
WIDE_UNARY_OP_IMPL(exp)
WIDE_UNARY_OP_IMPL(exp2)
WIDE_UNARY_OP_IMPL(expm1)
// pow is only masked implementation for performance reasons
WIDE_BINARY_OP_MASKED_IMPL(pow)
WIDE_UNARY_F_OP_IMPL(erf)
WIDE_UNARY_F_OP_IMPL(erfc)

// pow is only masked implementation for performance reasons
WIDE_BINARY_VF_OP_MASKED_IMPL(pow)

WIDE_UNARY_OP_IMPL(sqrt)
WIDE_UNARY_OP_IMPL(inversesqrt)
WIDE_UNARY_OP_IMPL(cbrt)

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

WIDE_BINARY_F_OR_V_OP_IMPL(fmod)
// mod for integers is handled by the code generator

DECL(__OSL_OP4(smoothstep, Wf, Wf, Wf, Wf), "xXXXX")
DECL(__OSL_MASKED_OP4(smoothstep, Wf, Wf, Wf, Wf), "xXXXXi")
DECL(__OSL_OP4(smoothstep, Wdf, Wdf, Wdf, Wdf), "xXXXX")
DECL(__OSL_MASKED_OP4(smoothstep, Wdf, Wdf, Wdf, Wdf), "xXXXXi")
DECL(__OSL_OP4(smoothstep, Wdf, Wf, Wdf, Wdf), "xXXXX")
DECL(__OSL_MASKED_OP4(smoothstep, Wdf, Wf, Wdf, Wdf), "xXXXXi")
DECL(__OSL_OP4(smoothstep, Wdf, Wdf, Wf, Wdf), "xXXXX")
DECL(__OSL_MASKED_OP4(smoothstep, Wdf, Wdf, Wf, Wdf), "xXXXXi")
DECL(__OSL_OP4(smoothstep, Wdf, Wdf, Wdf, Wf), "xXXXX")
DECL(__OSL_MASKED_OP4(smoothstep, Wdf, Wdf, Wdf, Wf), "xXXXXi")
DECL(__OSL_OP4(smoothstep, Wdf, Wf, Wf, Wdf), "xXXXX")
DECL(__OSL_MASKED_OP4(smoothstep, Wdf, Wf, Wf, Wdf), "xXXXXi")
DECL(__OSL_OP4(smoothstep, Wdf, Wdf, Wf, Wf), "xXXXX")
DECL(__OSL_MASKED_OP4(smoothstep, Wdf, Wdf, Wf, Wf), "xXXXXi")
DECL(__OSL_OP4(smoothstep, Wdf, Wf, Wdf, Wf), "xXXXX")
DECL(__OSL_MASKED_OP4(smoothstep, Wdf, Wf, Wdf, Wf), "xXXXXi")

// Replaced by osl_transform_[point|vector|normal]
// DECL (osl_transform_vmv, "xXXX")
// DECL (osl_transform_dvmdv, "xXXX")
// DECL (osl_transformv_vmv, "xXXX")
// DECL (osl_transformv_dvmdv, "xXXX")
// DECL (osl_transformn_vmv, "xXXX")
// DECL (osl_transformn_dvmdv, "xXXX")

DECL(__OSL_MASKED_OP3(transform_point, v, Wv, m), "xXXXii")
DECL(__OSL_MASKED_OP3(transform_point, v, Wv, Wm), "xXXXii")
DECL(__OSL_MASKED_OP3(transform_point, Wv, Wv, Wm), "xXXXii")
DECL(__OSL_MASKED_OP3(transform_point, Wdv, Wdv, Wm), "xXXXii")

DECL(__OSL_MASKED_OP3(transform_point, Wv, Wv, m), "xXXXii")
DECL(__OSL_MASKED_OP3(transform_point, Wdv, Wdv, m), "xXXXii")

DECL(__OSL_MASKED_OP3(transform_vector, v, Wv, m), "xXXXii")
DECL(__OSL_MASKED_OP3(transform_vector, v, Wv, Wm), "xXXXii")
DECL(__OSL_MASKED_OP3(transform_vector, Wv, Wv, Wm), "xXXXii")
DECL(__OSL_MASKED_OP3(transform_vector, Wdv, Wdv, Wm), "xXXXii")

DECL(__OSL_MASKED_OP3(transform_vector, Wv, Wv, m), "xXXXii")
DECL(__OSL_MASKED_OP3(transform_vector, Wdv, Wdv, m), "xXXXii")


DECL(__OSL_MASKED_OP3(transform_normal, v, Wv, m), "xXXXii")
DECL(__OSL_MASKED_OP3(transform_normal, v, Wv, Wm), "xXXXii")
DECL(__OSL_MASKED_OP3(transform_normal, Wv, Wv, Wm), "xXXXii")
DECL(__OSL_MASKED_OP3(transform_normal, Wdv, Wdv, Wm), "xXXXii")

DECL(__OSL_MASKED_OP3(transform_normal, Wv, Wv, m), "xXXXii")
DECL(__OSL_MASKED_OP3(transform_normal, Wdv, Wdv, m), "xXXXii")

DECL(__OSL_MASKED_OP3(transform_color, Wv, s, s), "xXXiXiXXi")
DECL(__OSL_OP3(transform_color, v, s, s), "xXXiXiXX")

DECL(__OSL_OP3(dot, Wf, Wv, Wv), "xXXX")
DECL(__OSL_MASKED_OP3(dot, Wf, Wv, Wv), "xXXXi")

DECL(__OSL_OP3(dot, Wdf, Wdv, Wdv), "xXXX")
DECL(__OSL_MASKED_OP3(dot, Wdf, Wdv, Wdv), "xXXXi")

DECL(__OSL_OP3(dot, Wdf, Wdv, Wv), "xXXX")
DECL(__OSL_MASKED_OP3(dot, Wdf, Wdv, Wv), "xXXXi")

DECL(__OSL_OP3(dot, Wdf, Wv, Wdv), "xXXX")
DECL(__OSL_MASKED_OP3(dot, Wdf, Wv, Wdv), "xXXXi")


DECL(__OSL_OP3(cross, Wv, Wv, Wv), "xXXX")
DECL(__OSL_MASKED_OP3(cross, Wv, Wv, Wv), "xXXXi")


DECL(__OSL_OP3(cross, Wdv, Wdv, Wdv), "xXXX")
DECL(__OSL_MASKED_OP3(cross, Wdv, Wdv, Wdv), "xXXXi")

DECL(__OSL_OP3(cross, Wdv, Wdv, Wv), "xXXX")
DECL(__OSL_MASKED_OP3(cross, Wdv, Wdv, Wv), "xXXXi")


DECL(__OSL_OP3(cross, Wdv, Wv, Wdv), "xXXX")
DECL(__OSL_MASKED_OP3(cross, Wdv, Wv, Wdv), "xXXXi")


DECL(__OSL_OP2(length, Wf, Wv), "xXX")
DECL(__OSL_MASKED_OP2(length, Wf, Wv), "xXXi")
DECL(__OSL_OP2(length, Wdf, Wdv), "xXX")
DECL(__OSL_MASKED_OP2(length, Wdf, Wdv), "xXXi")


DECL(__OSL_OP3(distance, Wf, Wv, Wv), "xXXX")
DECL(__OSL_MASKED_OP3(distance, Wf, Wv, Wv), "xXXXi")

DECL(__OSL_OP3(distance, Wdf, Wdv, Wdv), "xXXX")
DECL(__OSL_MASKED_OP3(distance, Wdf, Wdv, Wdv), "xXXXi")

DECL(__OSL_OP3(distance, Wdf, Wdv, Wv), "xXXX")
DECL(__OSL_MASKED_OP3(distance, Wdf, Wdv, Wv), "xXXXi")

DECL(__OSL_OP3(distance, Wdf, Wv, Wdv), "xXXX")
DECL(__OSL_MASKED_OP3(distance, Wdf, Wv, Wdv), "xXXXi")


DECL(__OSL_OP2(normalize, Wv, Wv), "xXX")
DECL(__OSL_MASKED_OP2(normalize, Wv, Wv), "xXXi")
DECL(__OSL_OP2(normalize, Wdv, Wdv), "xXX")
DECL(__OSL_MASKED_OP2(normalize, Wdv, Wdv), "xXXi")


DECL(__OSL_OP3(mul, Wm, Wm, Wm), "xXXX")
DECL(__OSL_MASKED_OP3(mul, Wm, Wm, Wm), "xXXXi")
DECL(__OSL_OP3(mul, Wm, Wm, Wf), "xXXX")
DECL(__OSL_MASKED_OP3(mul, Wm, Wm, Wf), "xXXXi")

// forced masked version only
DECL(__OSL_MASKED_OP3(div, Wm, Wm, Wm), "xXXXi")
DECL(__OSL_OP3(div, Wm, Wm, Wf), "xXXX")
DECL(__OSL_MASKED_OP3(div, Wm, Wm, Wf), "xXXXi")
DECL(__OSL_MASKED_OP3(div, Wm, Wf, Wm), "xXXXi")

// forced masked version only
DECL(__OSL_MASKED_OP3(get_from_to_matrix, Wm, s, s), "iXXssi")
DECL(__OSL_MASKED_OP3(get_from_to_matrix, Wm, s, Ws), "iXXsXi")
DECL(__OSL_MASKED_OP3(get_from_to_matrix, Wm, Ws, s), "iXXXsi")
DECL(__OSL_MASKED_OP3(get_from_to_matrix, Wm, Ws, Ws), "iXXXXi")

//varying vs non varying
DECL(__OSL_OP2(transpose, Wm, Wm), "xXX")
DECL(__OSL_MASKED_OP2(transpose, Wm, Wm), "xXXi")

DECL(__OSL_OP2(determinant, Wf, Wm), "xXX")
DECL(__OSL_MASKED_OP2(determinant, Wf, Wm), "xXXi")

// forced masked version only
DECL(__OSL_MASKED_OP3(concat, Ws, Ws, Ws), "xXXXi")
DECL(__OSL_MASKED_OP2(strlen, Wi, Ws), "xXXi")
DECL(__OSL_MASKED_OP2(hash, Wi, Ws), "xXXi")
DECL(__OSL_MASKED_OP3(getchar, Wi, Ws, Wi), "xXXXi")
DECL(__OSL_MASKED_OP3(startswith, Wi, Ws, Ws), "xXXXi")
DECL(__OSL_MASKED_OP3(endswith, Wi, Ws, Ws), "xXXXi")
DECL(__OSL_MASKED_OP2(stoi, Wi, Ws), "xXXi")
DECL(__OSL_MASKED_OP2(stof, Wf, Ws), "xXXi")
DECL(__OSL_MASKED_OP4(substr, Ws, Ws, Wi, Wi), "xXXXXi")

DECL(__OSL_MASKED_OP(regex_impl), "xXXXXiXii")
DECL(__OSL_OP(regex_impl), "iXsXisi")

// BATCH texturing manages the BatchedTextureOptions
// directly in LLVM ir, and has no need for wide versions
// of osl_texture_set_XXX functions
DECL(__OSL_MASKED_OP(texture), "iXXXXXXXXXXiXiXiXi")
DECL(__OSL_MASKED_OP(texture3d), "iXXXXXXXXiXiXiXi")
DECL(__OSL_MASKED_OP(environment), "iXXXXXXXiXiXiXi")
DECL(__OSL_OP(resolve_udim_uniform), "XXXXff")
DECL(__OSL_MASKED_OP(resolve_udim), "xXXXXXXi")
DECL(__OSL_OP(get_textureinfo_uniform), "iXXXXXX")

// Wide Code generator will set trace options directly in LLVM IR
// without calling helper functions
//DECL (osl_trace_set_mindist, "xXf") // unneeded
//DECL (osl_trace_set_maxdist, "xXf") // unneeded
//DECL (osl_trace_set_shade, "xXi") // unneeded
//DECL (osl_trace_set_traceset, "xXs") // unneeded
DECL(__OSL_MASKED_OP(trace), "xXXXXXXXXXi")

DECL(__OSL_OP(calculatenormal), "xXXX")
DECL(__OSL_MASKED_OP(calculatenormal), "xXXXi")
DECL(__OSL_OP2(area, Wf, Wdv), "xXX")
DECL(__OSL_MASKED_OP2(area, Wf, Wdv), "xXXi")
DECL(__OSL_OP2(filterwidth, Wf, Wdf), "xXX")
DECL(__OSL_OP2(filterwidth, Wv, Wdv), "xXX")

DECL(__OSL_MASKED_OP2(filterwidth, Wf, Wdf), "xXXi")
DECL(__OSL_MASKED_OP2(filterwidth, Wv, Wdv), "xXXi")


DECL(__OSL_OP(raytype_bit), "iXi")


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
#undef WIDE_BINARY_OP_MASKED_IMPL
#undef WIDE_BINARY_F_OR_V_OP_IMPL
#undef WIDE_BINARY_FI_OP_IMPL
#undef WIDE_BINARY_VF_OP_MASKED_IMPL
