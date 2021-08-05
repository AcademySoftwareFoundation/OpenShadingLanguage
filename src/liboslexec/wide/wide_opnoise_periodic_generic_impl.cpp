// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/oslconfig.h>

#include <OSL/batched_shaderglobals.h>
#include <OSL/wide.h>

#include "oslexec_pvt.h"

using namespace OSL;

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

/***********************************************************************
 * batched generic routines callable by the LLVM-generated code.
 */

#include "define_opname_macros.h"


#define __OSL_GENERIC_DISPATCH3(A, B, C, NONDERIV_A, NONDERIV_B, DUALTYPE)           \
    OSL_BATCHOP void __OSL_MASKED_OP3(gaborpnoise, A, B, C)(                         \
        char* name_ptr, char* r_ptr, char* x_ptr, char* px_ptr, char* bsg,           \
        char* opt, char* varying_direction_ptr, unsigned int mask_value);            \
    OSL_BATCHOP void __OSL_MASKED_OP3(pnoise, A, B,                                  \
                                      C)(char* r_ptr, char* x_ptr,                   \
                                         char* px_ptr,                               \
                                         unsigned int mask_value);                   \
    OSL_BATCHOP void __OSL_MASKED_OP3(psnoise, A, B,                                 \
                                      C)(char* r_ptr, char* x_ptr,                   \
                                         char* px_ptr,                               \
                                         unsigned int mask_value);                   \
    OSL_BATCHOP void __OSL_MASKED_OP3(pcellnoise, NONDERIV_A, NONDERIV_B,            \
                                      C)(char* r_ptr, char* x_ptr,                   \
                                         char* px_ptr,                               \
                                         unsigned int mask_value);                   \
    OSL_BATCHOP void __OSL_MASKED_OP3(phashnoise, NONDERIV_A, NONDERIV_B,            \
                                      C)(char* r_ptr, char* x_ptr,                   \
                                         char* px_ptr,                               \
                                         unsigned int mask_value);                   \
    OSL_BATCHOP void __OSL_MASKED_OP3(genericpnoise, A, B, C)(                       \
        char* name_ptr, char* r_ptr, char* x_ptr, char* px_ptr, char* bsg,           \
        char* opt, char* varying_direction_ptr, unsigned int mask_value)             \
    {                                                                                \
        ustring name = USTR(name_ptr);                                               \
        if (name == Strings::uperlin || name == Strings::noise) {                    \
            __OSL_MASKED_OP3(pnoise, A, B, C)                                        \
            (r_ptr, x_ptr, px_ptr, mask_value);                                      \
        } else if (name == Strings::perlin || name == Strings::snoise) {             \
            __OSL_MASKED_OP3(psnoise, A, B, C)                                       \
            (r_ptr, x_ptr, px_ptr, mask_value);                                      \
        } else if (name == Strings::cell) {                                          \
            /* NOTE: calling non derivative version */                               \
            __OSL_MASKED_OP3(pcellnoise, NONDERIV_A, NONDERIV_B, C)                  \
            (r_ptr, x_ptr, px_ptr, mask_value);                                      \
            Masked<DUALTYPE> wr(r_ptr, Mask(mask_value));                            \
            /* Need to handle clearing derivatives here */                           \
            OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))                            \
            for (int lane = 0; lane < __OSL_WIDTH; ++lane) {                         \
                DUALTYPE result = wr[lane];                                          \
                result.clear_d();                                                    \
                /* TODO: add helper that operates on Block<Dual2<T>> SOA directly */ \
                wr[lane] = result;                                                   \
            }                                                                        \
        } else if (name == Strings::gabor) {                                         \
            __OSL_MASKED_OP3(gaborpnoise, A, B, C)                                   \
            (name_ptr, r_ptr, x_ptr, px_ptr, bsg, opt, varying_direction_ptr,        \
             mask_value);                                                            \
        } else if (name == Strings::hash) {                                          \
            /* NOTE: calling non derivative version */                               \
            __OSL_MASKED_OP3(phashnoise, NONDERIV_A, NONDERIV_B, C)                  \
            (r_ptr, x_ptr, px_ptr, mask_value);                                      \
            Masked<DUALTYPE> wr(r_ptr, Mask(mask_value));                            \
            /* Need to handle clearing derivatives here */                           \
            OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))                            \
            for (int lane = 0; lane < __OSL_WIDTH; ++lane) {                         \
                DUALTYPE result = wr[lane];                                          \
                result.clear_d();                                                    \
                /* TODO: add helper that operates on Block<Dual2<T>> SOA directly */ \
                wr[lane] = result;                                                   \
            }                                                                        \
        } else {                                                                     \
            ((BatchedShaderGlobals*)bsg)                                             \
                ->uniform.context->errorf("Unknown noise type \"%s\"",               \
                                          name.c_str());                             \
        }                                                                            \
    }


#define __OSL_GENERIC_DISPATCH5(A, B, C, D, E, NONDERIV_A, NONDERIV_B,               \
                                NONDERIV_C, DUALTYPE)                                \
    OSL_BATCHOP void __OSL_MASKED_OP5(gaborpnoise, A, B, C, D, E)(                   \
        char* name_ptr, char* r_ptr, char* x_ptr, char* y_ptr, char* px_ptr,         \
        char* py_ptr, char* bsg, char* opt, char* varying_direction_ptr,             \
        unsigned int mask_value);                                                    \
    OSL_BATCHOP void __OSL_MASKED_OP5(pnoise, A, B, C, D,                            \
                                      E)(char* r_ptr, char* x_ptr,                   \
                                         char* y_ptr, char* px_ptr,                  \
                                         char* py_ptr,                               \
                                         unsigned int mask_value);                   \
    OSL_BATCHOP void __OSL_MASKED_OP5(psnoise, A, B, C, D,                           \
                                      E)(char* r_ptr, char* x_ptr,                   \
                                         char* y_ptr, char* px_ptr,                  \
                                         char* py_ptr,                               \
                                         unsigned int mask_value);                   \
    OSL_BATCHOP void __OSL_MASKED_OP5(pcellnoise, NONDERIV_A, NONDERIV_B,            \
                                      NONDERIV_C, D,                                 \
                                      E)(char* r_ptr, char* x_ptr,                   \
                                         char* y_ptr, char* px_ptr,                  \
                                         char* py_ptr,                               \
                                         unsigned int mask_value);                   \
    OSL_BATCHOP void __OSL_MASKED_OP5(phashnoise, NONDERIV_A, NONDERIV_B,            \
                                      NONDERIV_C, D,                                 \
                                      E)(char* r_ptr, char* x_ptr,                   \
                                         char* y_ptr, char* px_ptr,                  \
                                         char* py_ptr,                               \
                                         unsigned int mask_value);                   \
    OSL_BATCHOP void __OSL_MASKED_OP5(genericpnoise, A, B, C, D, E)(                 \
        char* name_ptr, char* r_ptr, char* x_ptr, char* y_ptr, char* px_ptr,         \
        char* py_ptr, char* bsg, char* opt, char* varying_direction_ptr,             \
        unsigned int mask_value)                                                     \
    {                                                                                \
        ustring name = USTR(name_ptr);                                               \
        if (name == Strings::uperlin || name == Strings::noise) {                    \
            __OSL_MASKED_OP5(pnoise, A, B, C, D, E)                                  \
            (r_ptr, x_ptr, y_ptr, px_ptr, py_ptr, mask_value);                       \
        } else if (name == Strings::perlin || name == Strings::snoise) {             \
            __OSL_MASKED_OP5(psnoise, A, B, C, D, E)                                 \
            (r_ptr, x_ptr, y_ptr, px_ptr, py_ptr, mask_value);                       \
        } else if (name == Strings::cell) {                                          \
            /* NOTE: calling non derivative version */                               \
            __OSL_MASKED_OP5(pcellnoise, NONDERIV_A, NONDERIV_B, NONDERIV_C,         \
                             D, E)                                                   \
            (r_ptr, x_ptr, y_ptr, px_ptr, py_ptr, mask_value);                       \
            Masked<DUALTYPE> wr(r_ptr, Mask(mask_value));                            \
            /* Need to handle clearing derivatives here */                           \
            OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))                            \
            for (int lane = 0; lane < __OSL_WIDTH; ++lane) {                         \
                DUALTYPE result = wr[lane];                                          \
                result.clear_d();                                                    \
                /* TODO: add helper that operates on Block<Dual2<T>> SOA directly */ \
                wr[lane] = result;                                                   \
            }                                                                        \
        } else if (name == Strings::gabor) {                                         \
            __OSL_MASKED_OP5(gaborpnoise, A, B, C, D, E)                             \
            (name_ptr, r_ptr, x_ptr, y_ptr, px_ptr, py_ptr, bsg, opt,                \
             varying_direction_ptr, mask_value);                                     \
        } else if (name == Strings::hash) {                                          \
            /* NOTE: calling non derivative version */                               \
            __OSL_MASKED_OP5(phashnoise, NONDERIV_A, NONDERIV_B, NONDERIV_C,         \
                             D, E)                                                   \
            (r_ptr, x_ptr, y_ptr, px_ptr, py_ptr, mask_value);                       \
            Masked<DUALTYPE> wr(r_ptr, Mask(mask_value));                            \
            /* Need to handle clearing derivatives here */                           \
            OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))                            \
            for (int lane = 0; lane < __OSL_WIDTH; ++lane) {                         \
                DUALTYPE result = wr[lane];                                          \
                result.clear_d();                                                    \
                /* TODO: add helper that operates on Block<Dual2<T>> SOA directly */ \
                wr[lane] = result;                                                   \
            }                                                                        \
        } else {                                                                     \
            ((BatchedShaderGlobals*)bsg)                                             \
                ->uniform.context->errorf("Unknown noise type \"%s\"",               \
                                          name.c_str());                             \
        }                                                                            \
    }

__OSL_GENERIC_DISPATCH3(Wdf, Wdf, Wf, Wf, Wf, Dual2<float>)
__OSL_GENERIC_DISPATCH5(Wdf, Wdf, Wdf, Wf, Wf, Wf, Wf, Wf, Dual2<float>)
__OSL_GENERIC_DISPATCH3(Wdf, Wdv, Wv, Wf, Wv, Dual2<float>)
__OSL_GENERIC_DISPATCH5(Wdf, Wdv, Wdf, Wv, Wf, Wf, Wv, Wf, Dual2<float>)

__OSL_GENERIC_DISPATCH3(Wdv, Wdf, Wf, Wv, Wf, Dual2<Vec3>)
__OSL_GENERIC_DISPATCH5(Wdv, Wdf, Wdf, Wf, Wf, Wv, Wf, Wf, Dual2<Vec3>)
__OSL_GENERIC_DISPATCH3(Wdv, Wdv, Wv, Wv, Wv, Dual2<Vec3>)
__OSL_GENERIC_DISPATCH5(Wdv, Wdv, Wdf, Wv, Wf, Wv, Wv, Wf, Dual2<Vec3>)

}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#include "undef_opname_macros.h"
