// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// Internal header included by every generated .cpp shader file.
// Not installed; not part of the public API.

#pragma once

#include <cstring>

#include <OSL/dual_vec.h>
#include <OSL/encodedtypes.h>
#include <OSL/oslclosure.h>
#include <OSL/oslconfig.h>
#include <OSL/shaderglobals.h>

// A closure value in generated code is just a pointer to a ClosureColor (the
// closure runtime functions all take/return `const void*`).  Spelled at global
// scope because generated declarations use it unqualified.
using closure_color_t = const void*;

OSL_NAMESPACE_BEGIN

/// ABI version for generated C++ shader code.  Folding in the OSL major/minor
/// version *guarantees* incompatibility across minor releases (no one has to
/// remember to bump anything); the trailing manually-incremented digit covers
/// an incompatible change to the generated-code interface (GroupData layout,
/// entry function signature, or the osl_* runtime function set) made *within*
/// a single minor release cycle.  The patch version is excluded: patch
/// releases must stay ABI-stable.  This must match the identical definition in
/// oslexec_pvt.h — a mismatch is caught loudly (every generated DSO fails the
/// ABI check at load).
constexpr int OSL_CPP_ABI_VERSION = 10000 * OSL_VERSION_MAJOR
                                    + 100 * OSL_VERSION_MINOR + 1;

namespace pvt {
/// Layout-compatible mirror of OSL::pvt::NoiseParams (oslexec_pvt.h).  Generated
/// code allocates one of these and passes it to the osl_*noise options API
/// (osl_init_noise_options / osl_noiseparams_set_*).  The field layout MUST stay
/// in sync with the authoritative definition in oslexec_pvt.h.
struct NoiseParams {
    int anisotropic;
    int do_filter;
    OSL::Vec3 direction;
    float bandwidth;
    float impulses;
};
}  // namespace pvt

OSL_NAMESPACE_END


// Forward declarations for OSL runtime functions called from generated code.
// Declarations are added incrementally as op generators are implemented.

extern "C" {
// clang-format off

// printf-family ops: exported wrappers that route through RendererServices
// virtual methods (rs_printfmt and friends have hidden visibility and cannot
// be bound by generated DSOs at dlopen time).
void osl_cpp_printfmt(void* sg, uint64_t fmt_hash, int32_t arg_count,
                      const uint8_t* etypes, uint32_t values_size,
                      const uint8_t* values);
void osl_cpp_errorfmt(void* sg, uint64_t fmt_hash, int32_t arg_count,
                      const uint8_t* etypes, uint32_t values_size,
                      const uint8_t* values);
void osl_cpp_warningfmt(void* sg, uint64_t fmt_hash, int32_t arg_count,
                        const uint8_t* etypes, uint32_t values_size,
                        const uint8_t* values);
void osl_cpp_filefmt(void* sg, uint64_t filename_hash, uint64_t fmt_hash,
                     int32_t arg_count, const uint8_t* etypes,
                     uint32_t values_size, const uint8_t* values);
// format() op: decode_message + ustring, exported from liboslexec.
OSL::ustringhash_pod osl_cpp_formatfmt(uint64_t fmt_hash, int32_t arg_count,
                                       const uint8_t* etypes,
                                       uint32_t values_size,
                                       const uint8_t* values);

// sincos
void osl_sincos_fff(float x, void* s, void* c);
void osl_sincos_dfdff(void* x, void* s, void* c);
void osl_sincos_dffdf(void* x, void* s, void* c);
void osl_sincos_dfdfdf(void* x, void* s, void* c);
void osl_sincos_vvv(void* x, void* s, void* c);
void osl_sincos_dvdvv(void* x, void* s, void* c);
void osl_sincos_dvvdv(void* x, void* s, void* c);
void osl_sincos_dvdvdv(void* x, void* s, void* c);

// rounding / sign
float osl_floor_ff(float x);
void  osl_floor_vv(void* r, void* x);
float osl_ceil_ff(float x);
void  osl_ceil_vv(void* r, void* x);
float osl_round_ff(float x);
void  osl_round_vv(void* r, void* x);
float osl_trunc_ff(float x);
void  osl_trunc_vv(void* r, void* x);
float osl_sign_ff(float x);
void  osl_sign_vv(void* r, void* x);
float osl_logb_ff(float x);
void  osl_logb_vv(void* r, void* x);

// integer abs/fabs
int osl_abs_ii(int x);
int osl_fabs_ii(int x);

// safe arithmetic
float osl_safe_div_fff(float a, float b);
int   osl_safe_div_iii(int a, int b);
int   osl_safe_mod_iii(int a, int b);

// numeric predicates
int osl_isnan_if(float f);
int osl_isinf_if(float f);
int osl_isfinite_if(float f);

// step / smoothstep
float osl_step_fff(float edge, float x);
void  osl_step_vvv(void* result, void* edge, void* x);
float osl_smoothstep_ffff(float e0, float e1, float x);
void  osl_smoothstep_dfffdf(void* result, float e0, float e1, void* x);
void  osl_smoothstep_dffdff(void* result, float e0, void* e1, float x);
void  osl_smoothstep_dffdfdf(void* result, float e0, void* e1, void* x);
void  osl_smoothstep_dfdfff(void* result, void* e0, float e1, float x);
void  osl_smoothstep_dfdffdf(void* result, void* e0, float e1, void* x);
void  osl_smoothstep_dfdfdff(void* result, void* e0, void* e1, float x);
void  osl_smoothstep_dfdfdfdf(void* result, void* e0, void* e1, void* x);

// trig
float osl_sin_ff(float x);
void  osl_sin_vv(void* r, void* x);
float osl_cos_ff(float x);
void  osl_cos_vv(void* r, void* x);
float osl_tan_ff(float x);
void  osl_tan_vv(void* r, void* x);
float osl_asin_ff(float x);
void  osl_asin_vv(void* r, void* x);
float osl_acos_ff(float x);
void  osl_acos_vv(void* r, void* x);
float osl_atan_ff(float x);
void  osl_atan_vv(void* r, void* x);
float osl_atan2_fff(float y, float x);
void  osl_atan2_vvv(void* r, void* y, void* x);
float osl_sinh_ff(float x);
void  osl_sinh_vv(void* r, void* x);
float osl_cosh_ff(float x);
void  osl_cosh_vv(void* r, void* x);
float osl_tanh_ff(float x);
void  osl_tanh_vv(void* r, void* x);

// exp / log
float osl_exp_ff(float x);
void  osl_exp_vv(void* r, void* x);
float osl_exp2_ff(float x);
void  osl_exp2_vv(void* r, void* x);
float osl_expm1_ff(float x);
void  osl_expm1_vv(void* r, void* x);
float osl_log_ff(float x);
void  osl_log_vv(void* r, void* x);
float osl_log2_ff(float x);
void  osl_log2_vv(void* r, void* x);
float osl_log10_ff(float x);
void  osl_log10_vv(void* r, void* x);

// power / root
float osl_sqrt_ff(float x);
void  osl_sqrt_vv(void* r, void* x);
float osl_cbrt_ff(float x);
void  osl_cbrt_vv(void* r, void* x);
float osl_inversesqrt_ff(float x);
void  osl_inversesqrt_vv(void* r, void* x);
float osl_pow_fff(float base, float exp);
void  osl_pow_vvf(void* r, void* base, float exp);
void  osl_pow_vvv(void* r, void* base, void* exp);

// special functions
float osl_erf_ff(float x);
void  osl_erf_vv(void* r, void* x);
float osl_erfc_ff(float x);
void  osl_erfc_vv(void* r, void* x);

// abs / fabs (float+triple; int versions already declared above)
float osl_abs_ff(float x);
void  osl_abs_vv(void* r, void* x);
float osl_fabs_ff(float x);
void  osl_fabs_vv(void* r, void* x);

// Scalar (Dual2<float>) derivative variants of the per-component math ops.
// These have native implementations in liboslexec (generated by the
// MAKE_UNARY/BINARY_PERCOMPONENT_OP macros in llvm_ops.cpp); declared here so
// the deriv-aware generated code links against them.  Triple (Dual2<Vec3>)
// variants of these are not declared until a test needs one.
void osl_sin_dfdf(void* r, void* x);
void osl_cos_dfdf(void* r, void* x);
void osl_tan_dfdf(void* r, void* x);
void osl_asin_dfdf(void* r, void* x);
void osl_acos_dfdf(void* r, void* x);
void osl_atan_dfdf(void* r, void* x);
void osl_atan2_dfdfdf(void* r, void* y, void* x);
void osl_sinh_dfdf(void* r, void* x);
void osl_cosh_dfdf(void* r, void* x);
void osl_tanh_dfdf(void* r, void* x);
void osl_exp_dfdf(void* r, void* x);
void osl_exp2_dfdf(void* r, void* x);
void osl_expm1_dfdf(void* r, void* x);
void osl_log_dfdf(void* r, void* x);
void osl_log2_dfdf(void* r, void* x);
void osl_log10_dfdf(void* r, void* x);
void osl_sqrt_dfdf(void* r, void* x);
void osl_cbrt_dfdf(void* r, void* x);
void osl_inversesqrt_dfdf(void* r, void* x);
void osl_pow_dfdfdf(void* r, void* base, void* exp);
void osl_pow_dfdff(void* r, void* base, float exp);
void osl_erf_dfdf(void* r, void* x);
void osl_erfc_dfdf(void* r, void* x);
void osl_abs_dfdf(void* r, void* x);
void osl_fabs_dfdf(void* r, void* x);

// fmod (incl. derivative variants — fmod's deriv is the numerator's deriv)
float osl_fmod_fff(float a, float b);
void  osl_fmod_vvf(void* r, void* a, float b);
void  osl_fmod_vvv(void* r, void* a, void* b);
void  osl_fmod_dfdfdf(void* r, void* a, void* b);
void  osl_fmod_dfdff(void* r, void* a, float b);
void  osl_fmod_dffdf(void* r, float a, void* b);
void  osl_fmod_dvdvdv(void* r, void* a, void* b);
void  osl_fmod_dvdvv(void* r, void* a, void* b);
void  osl_fmod_dvvdv(void* r, void* a, void* b);
void  osl_fmod_dvdvdf(void* r, void* a, void* b);
void  osl_fmod_dvvdf(void* r, void* a, void* b);
void  osl_fmod_dvdvf(void* r, void* a, float b);

// matrix
void  osl_transpose_mm(void* r, void* m);
float osl_determinant_fm(void* m);
void  osl_div_mmm(void* r, void* a, void* b);
void  osl_div_mmf(void* r, void* a, float b);
void  osl_div_mfm(void* r, float a, void* b);

// geometry
float osl_dot_fvv(void* a, void* b);
void  osl_dot_dfdvdv(void* result, void* a, void* b);
void  osl_dot_dfdvv(void* result, void* a, void* b);
void  osl_dot_dfvdv(void* result, void* a, void* b);
void  osl_cross_vvv(void* result, void* a, void* b);
void  osl_cross_dvdvdv(void* result, void* a, void* b);
void  osl_cross_dvdvv(void* result, void* a, void* b);
void  osl_cross_dvvdv(void* result, void* a, void* b);
float osl_length_fv(void* a);
void  osl_length_dfdv(void* result, void* a);
float osl_distance_fvv(void* a, void* b);
void  osl_distance_dfdvdv(void* result, void* a, void* b);
void  osl_distance_dfdvv(void* result, void* a, void* b);
void  osl_distance_dfvdv(void* result, void* a, void* b);
void  osl_normalize_vv(void* result, void* a);
void  osl_normalize_dvdv(void* result, void* a);
float osl_area(void* P);
void  osl_calculatenormal(void* result, void* ec, void* p);

// pointcloud. The names/types/values arrays are built at the call site (the
// helper fills one slot per attribute); out_indices/out_distances/out_data
// receive a contiguous value layout (distances additionally carry derivs in a
// [val][dx][dy] SoA region keyed by derivs_offset).
int osl_pointcloud_search(void* sg, OSL::ustringhash_pod filename, void* center,
                          float radius, int max_points, int sort,
                          void* out_indices, void* out_distances,
                          int derivs_offset, int nattrs, const void* names,
                          const void* types, const void* values);
int osl_pointcloud_get(void* sg, OSL::ustringhash_pod filename,
                       void* in_indices, int count,
                       OSL::ustringhash_pod attr_name, long long attr_type,
                       void* out_data);
void osl_pointcloud_write_helper(void* names, void* types, void* values,
                                 int index, OSL::ustringhash_pod name,
                                 long long type, void* val);
int osl_pointcloud_write(void* sg, OSL::ustringhash_pod filename,
                         const void* pos, int nattribs, const void* names,
                         const void* types, const void* values);

// Interpolated (userdata) parameter binding. Retrieves the renderer's userdata
// for a lockgeom=0 param into the GroupData userdata slot and copies it into the
// symbol; returns nonzero if userdata was available, 0 otherwise (caller then
// uses the param's default).
int osl_bind_interpolated_param(void* sg, OSL::ustringhash_pod name,
                                long long type, int userdata_has_derivs,
                                void* userdata_data, int symbol_has_derivs,
                                void* symbol_data, int symbol_data_size,
                                char* userdata_initialized, int userdata_index);

// string ops
OSL::ustringhash_pod osl_concat_sss(OSL::ustringhash_pod s, OSL::ustringhash_pod t);
int   osl_strlen_is(OSL::ustringhash_pod s);
int   osl_startswith_iss(OSL::ustringhash_pod s, OSL::ustringhash_pod sub);
int   osl_endswith_iss(OSL::ustringhash_pod s, OSL::ustringhash_pod sub);
int   osl_getchar_isi(OSL::ustringhash_pod s, int i);
int   osl_stoi_is(OSL::ustringhash_pod s);
float osl_stof_fs(OSL::ustringhash_pod s);
OSL::ustringhash_pod osl_substr_ssii(OSL::ustringhash_pod s, int start, int len);
int   osl_split(OSL::ustringhash_pod str, void* results,
                OSL::ustringhash_pod sep, int maxsplit, int resultslen);

// hash
int osl_hash_ii(int x);
int osl_hash_if(float x);
int osl_hash_iv(void* x);
int osl_hash_is(OSL::ustringhash_pod x);
int osl_hash_iff(float x, float y);
int osl_hash_ivf(void* x, float y);

// Noise families.  These macros mirror the osl_* noise entry points defined in
// opnoise.cpp (NOISE_IMPL / NOISE_DERIV_IMPL / PNOISE_IMPL / ... there).
// Pointer parameters are spelled void* (the definitions use char*; extern "C"
// linkage makes the pointee type irrelevant for binding, and void* accepts the
// void* the generated code passes).  The float-vs-pointer argument *pattern* of
// each variant matches the runtime typecode logic shared by the JIT and the
// cpp backend, so a generated call binds to the matching declaration.

#define OSL_CPP_NOISE_IMPL(name)                                               \
    float name##_ff(float);                                                    \
    float name##_fff(float, float);                                            \
    float name##_fv(void*);                                                    \
    float name##_fvf(void*, float);                                            \
    void  name##_vf(void*, float);                                             \
    void  name##_vff(void*, float, float);                                     \
    void  name##_vv(void*, void*);                                             \
    void  name##_vvf(void*, void*, float);

#define OSL_CPP_NOISE_DERIV_IMPL(name)                                         \
    void name##_dfdf(void*, void*);                                            \
    void name##_dfdff(void*, void*, float);                                    \
    void name##_dffdf(void*, float, void*);                                    \
    void name##_dfdfdf(void*, void*, void*);                                   \
    void name##_dfdv(void*, void*);                                            \
    void name##_dfdvf(void*, void*, float);                                    \
    void name##_dfvdf(void*, void*, void*);                                    \
    void name##_dfdvdf(void*, void*, void*);                                   \
    void name##_dvdf(void*, void*);                                            \
    void name##_dvdff(void*, void*, float);                                    \
    void name##_dvfdf(void*, float, void*);                                    \
    void name##_dvdfdf(void*, void*, void*);                                   \
    void name##_dvdv(void*, void*);                                            \
    void name##_dvdvf(void*, void*, float);                                    \
    void name##_dvvdf(void*, void*, void*);                                    \
    void name##_dvdvdf(void*, void*, void*);

#define OSL_CPP_GENERIC_NOISE_DERIV_IMPL(name)                                 \
    void name##_dfdf(OSL::ustringhash_pod, void*, void*, void*, void*);        \
    void name##_dfdfdf(OSL::ustringhash_pod, void*, void*, void*, void*,       \
                       void*);                                                 \
    void name##_dfdv(OSL::ustringhash_pod, void*, void*, void*, void*);        \
    void name##_dfdvdf(OSL::ustringhash_pod, void*, void*, void*, void*,       \
                       void*);                                                 \
    void name##_dvdf(OSL::ustringhash_pod, void*, void*, void*, void*);        \
    void name##_dvdfdf(OSL::ustringhash_pod, void*, void*, void*, void*,       \
                       void*);                                                 \
    void name##_dvdv(OSL::ustringhash_pod, void*, void*, void*, void*);        \
    void name##_dvdvdf(OSL::ustringhash_pod, void*, void*, void*, void*,       \
                       void*);

#define OSL_CPP_PNOISE_IMPL(name)                                              \
    float name##_fff(float, float);                                           \
    float name##_fffff(float, float, float, float);                           \
    float name##_fvv(void*, void*);                                           \
    float name##_fvfvf(void*, float, void*, float);                           \
    void  name##_vff(void*, float, float);                                    \
    void  name##_vffff(void*, float, float, float, float);                    \
    void  name##_vvv(void*, void*, void*);                                    \
    void  name##_vvfvf(void*, void*, float, void*, float);

#define OSL_CPP_PNOISE_DERIV_IMPL(name)                                        \
    void name##_dfdff(void*, void*, float);                                    \
    void name##_dfdffff(void*, void*, float, float, float);                    \
    void name##_dffdfff(void*, float, void*, float, float);                    \
    void name##_dfdfdfff(void*, void*, void*, float, float);                   \
    void name##_dfdvv(void*, void*, void*);                                    \
    void name##_dfdvfvf(void*, void*, float, void*, float);                    \
    void name##_dfvdfvf(void*, void*, void*, void*, float);                    \
    void name##_dfdvdfvf(void*, void*, void*, void*, float);                   \
    void name##_dvdff(void*, void*, float);                                    \
    void name##_dvdffff(void*, void*, float, float, float);                    \
    void name##_dvfdfff(void*, float, void*, float, float);                    \
    void name##_dvdfdfff(void*, void*, void*, float, float);                   \
    void name##_dvdvv(void*, void*, void*);                                    \
    void name##_dvdvfvf(void*, void*, float, void*, float);                    \
    void name##_dvvdfvf(void*, void*, void*, void*, float);                    \
    void name##_dvdvdfvf(void*, void*, void*, void*, float);

#define OSL_CPP_GENERIC_PNOISE_DERIV_IMPL(name)                                \
    void name##_dfdff(OSL::ustringhash_pod, void*, void*, float, void*,        \
                      void*);                                                  \
    void name##_dfdfdfff(OSL::ustringhash_pod, void*, void*, void*, float,     \
                         float, void*, void*);                                 \
    void name##_dfdvv(OSL::ustringhash_pod, void*, void*, void*, void*,        \
                      void*);                                                  \
    void name##_dfdvdfvf(OSL::ustringhash_pod, void*, void*, void*, void*,     \
                         float, void*, void*);                                 \
    void name##_dvdff(OSL::ustringhash_pod, void*, void*, float, void*,        \
                      void*);                                                  \
    void name##_dvdfdfff(OSL::ustringhash_pod, void*, void*, void*, float,     \
                         float, void*, void*);                                 \
    void name##_dvdvv(OSL::ustringhash_pod, void*, void*, void*, void*,        \
                      void*);                                                  \
    void name##_dvdvdfvf(OSL::ustringhash_pod, void*, void*, void*, void*,     \
                         float, void*, void*);

OSL_CPP_NOISE_IMPL(osl_cellnoise)
OSL_CPP_NOISE_IMPL(osl_hashnoise)
OSL_CPP_NOISE_IMPL(osl_noise)
OSL_CPP_NOISE_DERIV_IMPL(osl_noise)
OSL_CPP_NOISE_IMPL(osl_snoise)
OSL_CPP_NOISE_DERIV_IMPL(osl_snoise)
OSL_CPP_NOISE_IMPL(osl_simplexnoise)
OSL_CPP_NOISE_DERIV_IMPL(osl_simplexnoise)
OSL_CPP_NOISE_IMPL(osl_usimplexnoise)
OSL_CPP_NOISE_DERIV_IMPL(osl_usimplexnoise)
OSL_CPP_GENERIC_NOISE_DERIV_IMPL(osl_gabornoise)
OSL_CPP_GENERIC_NOISE_DERIV_IMPL(osl_genericnoise)
OSL_CPP_PNOISE_IMPL(osl_pcellnoise)
OSL_CPP_PNOISE_IMPL(osl_phashnoise)
OSL_CPP_PNOISE_IMPL(osl_pnoise)
OSL_CPP_PNOISE_DERIV_IMPL(osl_pnoise)
OSL_CPP_PNOISE_IMPL(osl_psnoise)
OSL_CPP_PNOISE_DERIV_IMPL(osl_psnoise)
OSL_CPP_GENERIC_PNOISE_DERIV_IMPL(osl_gaborpnoise)
OSL_CPP_GENERIC_PNOISE_DERIV_IMPL(osl_genericpnoise)

#undef OSL_CPP_NOISE_IMPL
#undef OSL_CPP_NOISE_DERIV_IMPL
#undef OSL_CPP_GENERIC_NOISE_DERIV_IMPL
#undef OSL_CPP_PNOISE_IMPL
#undef OSL_CPP_PNOISE_DERIV_IMPL
#undef OSL_CPP_GENERIC_PNOISE_DERIV_IMPL

// Noise options API (used by the generic / gabor noise paths).
void osl_init_noise_options(void* sg, void* opt);
void osl_noiseparams_set_anisotropic(void* opt, int a);
void osl_noiseparams_set_do_filter(void* opt, int a);
void osl_noiseparams_set_direction(void* opt, void* dir);
void osl_noiseparams_set_bandwidth(void* opt, float b);
void osl_noiseparams_set_impulses(void* opt, float i);
void osl_count_noise(void* sg);

// dict
int osl_dict_find_iis(void* ec, int nodeptr, OSL::ustringhash_pod query);
int osl_dict_find_iss(void* ec, OSL::ustringhash_pod dict, OSL::ustringhash_pod query);
int osl_dict_next(void* ec, int nodeptr);
int osl_dict_value(void* ec, int nodeptr, OSL::ustringhash_pod attrib,
                   long long type, void* data);

// filterwidth (deriv-carrying input; result has no derivs).  The float form
// returns the width directly; the triple form writes through an out-pointer.
float osl_filterwidth_fdf(void* x);
void osl_filterwidth_vdv(void* result, void* x);

// closures: construction (allocate + fill), arithmetic, and to-string.
// All take/return a ClosureColor* spelled as void*.
void* osl_allocate_closure_component(void* ec, int id, int size);
void* osl_allocate_weighted_closure_component(void* ec, int id, int size,
                                              const void* w);
const void* osl_add_closure_closure(void* ec, const void* a, const void* b);
const void* osl_mul_closure_color(void* ec, const void* a, const void* w);
const void* osl_mul_closure_float(void* ec, const void* a, float w);
OSL::ustringhash_pod osl_closure_to_ustringhash(void* ec, const void* c);

// range check (array/component bounds). On out-of-range it reports an error and
// returns a clamped in-range index; otherwise returns the index unchanged.
int osl_range_check(int index, int len, OSL::ustringhash_pod symname, void* ec,
                    OSL::ustringhash_pod sourcefile, int sourceline,
                    OSL::ustringhash_pod groupname, int layer,
                    OSL::ustringhash_pod layername,
                    OSL::ustringhash_pod shadername);

// debug_nan: after an op writes a float-based value, check it for NaN/Inf and
// report (with the op name and source location) if found. firstcheck/nchecks
// restrict the check to the components actually written (for partial writes like
// aassign/compassign/mxcompassign).
void osl_naninf_check(int ncomps, const void* vals, int has_derivs, void* sg,
                      OSL::ustringhash_pod sourcefile, int sourceline,
                      OSL::ustringhash_pod symbolname, int firstcheck,
                      int nchecks, OSL::ustringhash_pod opname);

// debug_uninit: before an op reads a value, check the read components for the
// "uninitialized" marker (NaN float / INT_MIN int / the uninitialized string)
// and report. firstcheck/nchecks restrict the check to the components read.
void osl_uninit_check(long long typedesc, void* vals, void* sg,
                      OSL::ustringhash_pod sourcefile, int sourceline,
                      OSL::ustringhash_pod groupname, int layer,
                      OSL::ustringhash_pod layername,
                      OSL::ustringhash_pod shadername, int opnum,
                      OSL::ustringhash_pod opname, int argnum,
                      OSL::ustringhash_pod symbolname, int firstcheck,
                      int nchecks);

// getattribute (common path; type packed as a long long bit-cast to TypeDesc)
int osl_get_attribute(void* sg, int dest_derivs, OSL::ustringhash_pod obj_name,
                      OSL::ustringhash_pod attr_name, int array_lookup,
                      int index, long long attr_type, void* attr_dest);

// message passing (type packed as a long long bit-cast to TypeDesc)
void osl_setmessage(OSL::ShaderGlobals* sg, OSL::ustringhash_pod name,
                    long long type, void* val, int layeridx,
                    OSL::ustringhash_pod sourcefile, int sourceline);
int osl_getmessage(OSL::ShaderGlobals* sg, OSL::ustringhash_pod source,
                   OSL::ustringhash_pod name, long long type, void* val,
                   int derivs, int layeridx, OSL::ustringhash_pod sourcefile,
                   int sourceline);
int osl_trace_get(void* ec, OSL::ustringhash_pod name, long long type,
                  void* val, int derivatives);

// regex_match / regex_search (results is an int array, or null)
int osl_regex_impl(void* sg, OSL::ustringhash_pod subject, void* results,
                   int nresults, OSL::ustringhash_pod pattern, int fullmatch);

// transform a triple by an explicit matrix (point/vector/normal forms)
void osl_transform_vmv(void* result, void* M, void* v);
void osl_transformv_vmv(void* result, void* M, void* v);
void osl_transformn_vmv(void* result, void* M, void* v);

// blackbody / wavelength_color: write a color through the out-pointer.
void osl_blackbody_vf(void* ec, void* out, float temp);
void osl_wavelength_color_vf(void* ec, void* out, float lambda);

// trace: options struct (OSL::TraceOpt) then the ray cast.  Pos/Dir are passed
// as their value/dx/dy block pointers.
void osl_init_trace_options(void* ec, void* opt);
void osl_trace_set_mindist(void* opt, float x);
void osl_trace_set_maxdist(void* opt, float x);
void osl_trace_set_shade(void* opt, int x);
void osl_trace_set_traceset(void* opt, OSL::ustringhash_pod x);
int osl_trace(void* ec, void* opt, void* Pos, void* dPosdx, void* dPosdy,
              void* Dir, void* dDirdx, void* dDirdy);

// raytype
int osl_raytype_bit(void* ec, int bit);
int osl_raytype_name(void* ec, OSL::ustringhash_pod name);

// spline / splineinverse: out-ptr, spline-type string, value-ptr, knots-ptr,
// knot count, knot array length.  One variant per deriv/type-code combination.
#define OSL_CPP_SPLINE(suffix)                                              \
    void osl_spline_##suffix(void* out, OSL::ustringhash_pod spline,        \
                             void* x, void* knots, int knot_count,          \
                             int knot_arraylen);
OSL_CPP_SPLINE(fff) OSL_CPP_SPLINE(dfdfdf) OSL_CPP_SPLINE(dffdf)
OSL_CPP_SPLINE(dfdff) OSL_CPP_SPLINE(vfv) OSL_CPP_SPLINE(dvdfv)
OSL_CPP_SPLINE(dvfdv) OSL_CPP_SPLINE(dvdfdv)
#undef OSL_CPP_SPLINE
#define OSL_CPP_SPLINEINV(suffix)                                           \
    void osl_splineinverse_##suffix(void* out, OSL::ustringhash_pod spline, \
                                    void* x, void* knots, int knot_count,   \
                                    int knot_arraylen);
OSL_CPP_SPLINEINV(fff) OSL_CPP_SPLINEINV(dfdfdf) OSL_CPP_SPLINEINV(dffdf)
OSL_CPP_SPLINEINV(dfdff)
#undef OSL_CPP_SPLINEINV

// coordinate-system / colorspace construction (matrix/point/vector/normal/color
// with a named space)
void osl_prepend_color_from(void* sg, void* c, OSL::ustringhash_pod from);
int  osl_prepend_matrix_from(void* sg, void* r, OSL::ustringhash_pod from);
int  osl_get_from_to_matrix(void* sg, void* r, OSL::ustringhash_pod from,
                            OSL::ustringhash_pod to);
int  osl_transform_triple(void* sg, void* Pin, int Pin_derivs, void* Pout,
                          int Pout_derivs, OSL::ustringhash_pod from,
                          OSL::ustringhash_pod to, int vectype);
int  osl_transform_triple_nonlinear(void* sg, void* Pin, int Pin_derivs,
                                    void* Pout, int Pout_derivs,
                                    OSL::ustringhash_pod from,
                                    OSL::ustringhash_pod to, int vectype);

// luminance (needs the colorsystem from the exec context)
void osl_luminance_fv(void* sg, void* out, void* c);
void osl_luminance_dfdv(void* sg, void* out, void* c);

// transformc (colorspace conversion)
int osl_transformc(void* sg, void* Cin, int Cin_derivs, void* Cout,
                   int Cout_derivs, OSL::ustringhash_pod from,
                   OSL::ustringhash_pod to);

// Texture options API
void osl_init_texture_options(void* sg, void* opt);
void osl_texture_set_firstchannel(void* opt, int x);
void osl_texture_set_subimage(void* opt, int x);
void osl_texture_set_subimagename(void* opt, OSL::ustringhash_pod x);
void osl_texture_set_swrap(void* opt, OSL::ustringhash_pod x);
void osl_texture_set_twrap(void* opt, OSL::ustringhash_pod x);
void osl_texture_set_rwrap(void* opt, OSL::ustringhash_pod x);
void osl_texture_set_stwrap(void* opt, OSL::ustringhash_pod x);
void osl_texture_set_swrap_code(void* opt, int mode);
void osl_texture_set_twrap_code(void* opt, int mode);
void osl_texture_set_rwrap_code(void* opt, int mode);
void osl_texture_set_stwrap_code(void* opt, int mode);
void osl_texture_set_sblur(void* opt, float x);
void osl_texture_set_tblur(void* opt, float x);
void osl_texture_set_rblur(void* opt, float x);
void osl_texture_set_stblur(void* opt, float x);
void osl_texture_set_swidth(void* opt, float x);
void osl_texture_set_twidth(void* opt, float x);
void osl_texture_set_rwidth(void* opt, float x);
void osl_texture_set_stwidth(void* opt, float x);
void osl_texture_set_fill(void* opt, float x);
void osl_texture_set_time(void* opt, float x);
void osl_texture_set_interp(void* opt, OSL::ustringhash_pod x);
void osl_texture_set_interp_code(void* opt, int mode);
void osl_texture_set_missingcolor_arena(void* opt, const void* missing);
void osl_texture_set_missingcolor_alpha(void* opt, int alphaindex, float missingalpha);

// Texture lookup functions
int osl_texture(void* sg, OSL::ustringhash_pod name, void* handle,
                void* opt, float s, float t,
                float dsdx, float dtdx, float dsdy, float dtdy,
                int chans, void* result, void* dresultdx, void* dresultdy,
                void* alpha, void* dalphadx, void* dalphady, void* errormsg);
int osl_texture3d(void* sg, OSL::ustringhash_pod name, void* handle,
                  void* opt, void* P, void* dPdx, void* dPdy, void* dPdz,
                  int chans, void* result, void* dresultdx, void* dresultdy,
                  void* alpha, void* dalphadx, void* dalphady, void* errormsg);
int osl_environment(void* sg, OSL::ustringhash_pod name, void* handle,
                    void* opt, void* R, void* dRdx, void* dRdy,
                    int chans, void* result, void* dresultdx, void* dresultdy,
                    void* alpha, void* dalphadx, void* dalphady, void* errormsg);
int osl_get_textureinfo(void* sg, OSL::ustringhash_pod name, void* handle,
                        OSL::ustringhash_pod dataname, int type, int arraylen,
                        int aggregate, void* data, void* errormsg);
int osl_get_textureinfo_st(void* sg, OSL::ustringhash_pod name, void* handle,
                           float s, float t, OSL::ustringhash_pod dataname,
                           int type, int arraylen, int aggregate,
                           void* data, void* errormsg);

// clang-format on
}  // extern "C"


// Inline helpers for ops the JIT emits as inline IR (no exported osl_* symbol).
// All triple args/results are passed as void* matching the osl_* ABI convention.

// div — use safe_div to avoid UB on divide-by-zero
static inline float
osl_div_fff(float a, float b)
{
    return osl_safe_div_fff(a, b);
}



// Dual2 safe divide, matching llvm_gen_div exactly: value and 1/b both go
// through osl_safe_div_fff (returns 0 when the quotient is non-finite), and the
// derivatives use binv*(ax - (a/b)*bx).  Operands are wrapped to Dual2<float> at
// the call site so a plain float promotes (zero derivs).
static inline OSL::Dual2<float>
osl_div_dual(OSL::Dual2<float> a, OSL::Dual2<float> b)
{
    float a_div_b = osl_safe_div_fff(a.val(), b.val());
    float binv    = osl_safe_div_fff(1.0f, b.val());
    return OSL::Dual2<float>(a_div_b, binv * (a.dx() - a_div_b * b.dx()),
                             binv * (a.dy() - a_div_b * b.dy()));
}



static inline int
osl_div_iii(int a, int b)
{
    return osl_safe_div_iii(a, b);
}



static inline void
osl_div_vvf(void* r_, void* a_, float b)
{
    const float* a = (const float*)a_;
    float* r       = (float*)r_;
    r[0]           = osl_safe_div_fff(a[0], b);
    r[1]           = osl_safe_div_fff(a[1], b);
    r[2]           = osl_safe_div_fff(a[2], b);
}



static inline void
osl_div_vvv(void* r_, void* a_, void* b_)
{
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* r       = (float*)r_;
    r[0]           = osl_safe_div_fff(a[0], b[0]);
    r[1]           = osl_safe_div_fff(a[1], b[1]);
    r[2]           = osl_safe_div_fff(a[2], b[2]);
}



static inline void
osl_div_vfv(void* r_, float a, void* b_)
{
    const float* b = (const float*)b_;
    float* r       = (float*)r_;
    r[0]           = osl_safe_div_fff(a, b[0]);
    r[1]           = osl_safe_div_fff(a, b[1]);
    r[2]           = osl_safe_div_fff(a, b[2]);
}



// mod — float mod matches fmod; int mod uses safe_mod
static inline float
osl_mod_fff(float a, float b)
{
    return osl_fmod_fff(a, b);
}



static inline int
osl_mod_iii(int a, int b)
{
    return osl_safe_mod_iii(a, b);
}



static inline void
osl_mod_vvf(void* r_, void* a_, float b)
{
    const float* a = (const float*)a_;
    float* r       = (float*)r_;
    r[0]           = osl_fmod_fff(a[0], b);
    r[1]           = osl_fmod_fff(a[1], b);
    r[2]           = osl_fmod_fff(a[2], b);
}



static inline void
osl_mod_vvv(void* r_, void* a_, void* b_)
{
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* r       = (float*)r_;
    r[0]           = osl_fmod_fff(a[0], b[0]);
    r[1]           = osl_fmod_fff(a[1], b[1]);
    r[2]           = osl_fmod_fff(a[2], b[2]);
}



// min / max
static inline float
osl_min_fff(float a, float b)
{
    return a < b ? a : b;
}



static inline int
osl_min_iii(int a, int b)
{
    return a < b ? a : b;
}



static inline void
osl_min_vvv(void* r_, void* a_, void* b_)
{
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* r       = (float*)r_;
    r[0]           = a[0] < b[0] ? a[0] : b[0];
    r[1]           = a[1] < b[1] ? a[1] : b[1];
    r[2]           = a[2] < b[2] ? a[2] : b[2];
}



static inline float
osl_max_fff(float a, float b)
{
    return a > b ? a : b;
}



static inline int
osl_max_iii(int a, int b)
{
    return a > b ? a : b;
}



static inline void
osl_max_vvv(void* r_, void* a_, void* b_)
{
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* r       = (float*)r_;
    r[0]           = a[0] > b[0] ? a[0] : b[0];
    r[1]           = a[1] > b[1] ? a[1] : b[1];
    r[2]           = a[2] > b[2] ? a[2] : b[2];
}



// min/max derivative variants: select the chosen operand's full Dual2 (carrying
// its derivatives), matching the value-path comparison.  min/max have no native
// osl_* function (inline helpers here), so deriv variants live here too.
static inline void
osl_min_dfdfdf(void* r_, void* a_, void* b_)
{
    OSL::Dual2<float>& a = *(OSL::Dual2<float>*)a_;
    OSL::Dual2<float>& b = *(OSL::Dual2<float>*)b_;
    // <= (not <) to match llvm_gen_minmax — only the deriv tie-break differs.
    *(OSL::Dual2<float>*)r_ = (a.val() <= b.val()) ? a : b;
}



static inline void
osl_max_dfdfdf(void* r_, void* a_, void* b_)
{
    OSL::Dual2<float>& a    = *(OSL::Dual2<float>*)a_;
    OSL::Dual2<float>& b    = *(OSL::Dual2<float>*)b_;
    *(OSL::Dual2<float>*)r_ = (a.val() > b.val()) ? a : b;
}



// clamp
static inline float
osl_clamp_ffff(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}



static inline int
osl_clamp_iiii(int x, int lo, int hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}



static inline void
osl_clamp_vvvv(void* r_, void* x_, void* lo_, void* hi_)
{
    const float* x  = (const float*)x_;
    const float* lo = (const float*)lo_;
    const float* hi = (const float*)hi_;
    float* r        = (float*)r_;
    r[0]            = x[0] < lo[0] ? lo[0] : (x[0] > hi[0] ? hi[0] : x[0]);
    r[1]            = x[1] < lo[1] ? lo[1] : (x[1] > hi[1] ? hi[1] : x[1]);
    r[2]            = x[2] < lo[2] ? lo[2] : (x[2] > hi[2] ? hi[2] : x[2]);
}



// mix: a*(1-x) + b*x
static inline float
osl_mix_ffff(float a, float b, float x)
{
    return a + (b - a) * x;
}



static inline void
osl_mix_vvvf(void* r_, void* a_, void* b_, float x)
{
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* r       = (float*)r_;
    r[0]           = a[0] + (b[0] - a[0]) * x;
    r[1]           = a[1] + (b[1] - a[1]) * x;
    r[2]           = a[2] + (b[2] - a[2]) * x;
}



static inline void
osl_mix_vvvv(void* r_, void* a_, void* b_, void* x_)
{
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    const float* x = (const float*)x_;
    float* r       = (float*)r_;
    r[0]           = a[0] + (b[0] - a[0]) * x[0];
    r[1]           = a[1] + (b[1] - a[1]) * x[1];
    r[2]           = a[2] + (b[2] - a[2]) * x[2];
}



// mix derivative variants: a + (b-a)*x evaluated with OSL::Dual2 arithmetic,
// which propagates derivatives.  mix has no native osl_* function (it is an
// inline helper here), so the deriv variants live here too.
static inline void
osl_mix_dfdfdfdf(void* r_, void* a_, void* b_, void* x_)
{
    OSL::Dual2<float>& a    = *(OSL::Dual2<float>*)a_;
    OSL::Dual2<float>& b    = *(OSL::Dual2<float>*)b_;
    OSL::Dual2<float>& x    = *(OSL::Dual2<float>*)x_;
    *(OSL::Dual2<float>*)r_ = a + (b - a) * x;
}



static inline void
osl_mix_dvdvdvdv(void* r_, void* a_, void* b_, void* x_)
{
    OSL::Dual2<OSL::Vec3>& a    = *(OSL::Dual2<OSL::Vec3>*)a_;
    OSL::Dual2<OSL::Vec3>& b    = *(OSL::Dual2<OSL::Vec3>*)b_;
    OSL::Dual2<OSL::Vec3>& x    = *(OSL::Dual2<OSL::Vec3>*)x_;
    *(OSL::Dual2<OSL::Vec3>*)r_ = a + (b - a) * x;
}



static inline void
osl_mix_dvdvdvdf(void* r_, void* a_, void* b_, void* x_)
{
    OSL::Dual2<OSL::Vec3>& a    = *(OSL::Dual2<OSL::Vec3>*)a_;
    OSL::Dual2<OSL::Vec3>& b    = *(OSL::Dual2<OSL::Vec3>*)b_;
    OSL::Dual2<float>& x        = *(OSL::Dual2<float>*)x_;
    *(OSL::Dual2<OSL::Vec3>*)r_ = a + (b - a) * x;
}



// select: cond!=0 ? b : a  (matches llvm_gen_select semantics)
static inline float
osl_select_fffi(float a, float b, int cond)
{
    return cond ? b : a;
}



static inline float
osl_select_ffff(float a, float b, float cond)
{
    return cond ? b : a;
}



static inline int
osl_select_iiii(int a, int b, int cond)
{
    return cond ? b : a;
}



static inline void
osl_select_vvvi(void* r_, void* a_, void* b_, int cond)
{
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* r       = (float*)r_;
    r[0]           = cond ? b[0] : a[0];
    r[1]           = cond ? b[1] : a[1];
    r[2]           = cond ? b[2] : a[2];
}



static inline void
osl_select_vvvf(void* r_, void* a_, void* b_, float cond)
{
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* r       = (float*)r_;
    r[0]           = cond ? b[0] : a[0];
    r[1]           = cond ? b[1] : a[1];
    r[2]           = cond ? b[2] : a[2];
}



// Derivatives — C++ backend carries no derivative state; return zero.
static inline float
osl_Dx_ff(float)
{
    return 0.0f;
}



static inline float
osl_Dy_ff(float)
{
    return 0.0f;
}



static inline float
osl_Dz_ff(float)
{
    return 0.0f;
}



static inline void
osl_Dx_vv(void* r_, void*)
{
    float* r = (float*)r_;
    r[0] = r[1] = r[2] = 0.0f;
}



static inline void
osl_Dy_vv(void* r_, void*)
{
    float* r = (float*)r_;
    r[0] = r[1] = r[2] = 0.0f;
}



static inline void
osl_Dz_vv(void* r_, void*)
{
    float* r = (float*)r_;
    r[0] = r[1] = r[2] = 0.0f;
}



// filterwidth — without derivative info, return a nominal value of 1
static inline float
osl_filterwidth_ff(float)
{
    return 1.0f;
}



static inline void
osl_filterwidth_vv(void* r_, void*)
{
    float* r = (float*)r_;
    r[0] = r[1] = r[2] = 1.0f;
}



// area and calculatenormal each have a dedicated cpp generator (cpp_gen_area /
// cpp_gen_calculatenormal), so no generic mangled-name alias is needed.

// strtof/strtoi are OP2 aliases for stof/stoi
static inline float
osl_strtof_fs(OSL::ustringhash_pod s)
{
    return osl_stof_fs(s);
}



static inline int
osl_strtoi_is(OSL::ustringhash_pod s)
{
    return osl_stoi_is(s);
}
