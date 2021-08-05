// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/oslconfig.h>

#include <OSL/batched_shaderglobals.h>
#include <OSL/wide.h>

#include "oslexec_pvt.h"

#include "null_noise.h"


using namespace OSL;

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)


// NOTE: working at the object level means we are reading the Dual2::val()
// and rewriting it back unnecessarily, as we only modified dx() and dy()
// template<typename DataT>
// OSL_FORCEINLINE void
// zero_derivs(Masked<Dual2<DataT>> wr) {
//      OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
//      for(int lane=0; lane< __OSL_WIDTH; ++lane) {
//          DUALTYPE result = wr[lane];
//          result.clear_d();
//          wr[lane] = result;
//      }
// }
// Instead we overload for specific types that zero out dx, dy of the
// the underlying data block

OSL_FORCEINLINE void
zero_derivs(Masked<Dual2<Vec3>> wr)
{
    Block<Dual2<Vec3>>& block = wr.data();
    Mask mask                 = wr.mask();

    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
        if (mask[lane]) {
            block.dx_x[lane] = 0.0f;
            block.dx_y[lane] = 0.0f;
            block.dx_z[lane] = 0.0f;
            block.dy_x[lane] = 0.0f;
            block.dy_y[lane] = 0.0f;
            block.dy_z[lane] = 0.0f;
        }
    }
}

OSL_FORCEINLINE void
zero_derivs(Masked<Dual2<float>> wr)
{
    Block<Dual2<float>>& block = wr.data();
    Mask mask                  = wr.mask();

    OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
    for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
        if (mask[lane]) {
            block.dx[lane] = 0.0f;
            block.dy[lane] = 0.0f;
        }
    }
}


/***********************************************************************
 * batched generic routines callable by the LLVM-generated code.
 */

#include "define_opname_macros.h"


#define __OSL_GENERIC_DISPATCH2(A, B, NONDERIV_A, NONDERIV_B, DUALTYPE)       \
    OSL_BATCHOP void __OSL_MASKED_OP2(gabornoise, A,                          \
                                      B)(char* name_ptr, char* r_ptr,         \
                                         char* x_ptr, char* bsg, char* opt,   \
                                         char* varying_direction_ptr,         \
                                         unsigned int mask_value);            \
    OSL_BATCHOP void __OSL_MASKED_OP2(noise, A, B)(char* r_ptr, char* x_ptr,  \
                                                   unsigned int mask_value);  \
    OSL_BATCHOP void __OSL_MASKED_OP2(simplexnoise, A,                        \
                                      B)(char* r_ptr, char* x_ptr,            \
                                         unsigned int mask_value);            \
    OSL_BATCHOP void __OSL_MASKED_OP2(snoise, A, B)(char* r_ptr, char* x_ptr, \
                                                    unsigned int mask_value); \
    OSL_BATCHOP void __OSL_MASKED_OP2(usimplexnoise, A,                       \
                                      B)(char* r_ptr, char* x_ptr,            \
                                         unsigned int mask_value);            \
    OSL_BATCHOP void __OSL_MASKED_OP2(cellnoise, NONDERIV_A,                  \
                                      NONDERIV_B)(char* r_ptr, char* x_ptr,   \
                                                  unsigned int mask_value);   \
    OSL_BATCHOP void __OSL_MASKED_OP2(nullnoise, A,                           \
                                      B)(char* r_ptr, char* x_ptr,            \
                                         unsigned int mask_value);            \
    OSL_BATCHOP void __OSL_MASKED_OP2(unullnoise, A,                          \
                                      B)(char* r_ptr, char* x_ptr,            \
                                         unsigned int mask_value);            \
    OSL_BATCHOP void __OSL_MASKED_OP2(hashnoise, NONDERIV_A,                  \
                                      NONDERIV_B)(char* r_ptr, char* x_ptr,   \
                                                  unsigned int mask_value);   \
    OSL_BATCHOP void __OSL_MASKED_OP2(genericnoise, A,                        \
                                      B)(char* name_ptr, char* r_ptr,         \
                                         char* x_ptr, char* bsg, char* opt,   \
                                         char* varying_direction_ptr,         \
                                         unsigned int mask_value)             \
    {                                                                         \
        ustring name = USTR(name_ptr);                                        \
        if (name == Strings::uperlin || name == Strings::noise) {             \
            __OSL_MASKED_OP2(noise, A, B)(r_ptr, x_ptr, mask_value);          \
        } else if (name == Strings::perlin || name == Strings::snoise) {      \
            __OSL_MASKED_OP2(snoise, A, B)(r_ptr, x_ptr, mask_value);         \
        } else if (name == Strings::simplexnoise                              \
                   || name == Strings::simplex) {                             \
            __OSL_MASKED_OP2(simplexnoise, A, B)(r_ptr, x_ptr, mask_value);   \
        } else if (name == Strings::usimplexnoise                             \
                   || name == Strings::usimplex) {                            \
            __OSL_MASKED_OP2(usimplexnoise, A, B)(r_ptr, x_ptr, mask_value);  \
        } else if (name == Strings::cell) {                                   \
            /* NOTE: calling non derivative version */                        \
            __OSL_MASKED_OP2(cellnoise, NONDERIV_A, NONDERIV_B)               \
            (r_ptr, x_ptr, mask_value);                                       \
            Masked<DUALTYPE> wr(r_ptr, Mask(mask_value));                     \
            zero_derivs(wr);                                                  \
        } else if (name == Strings::gabor) {                                  \
            __OSL_MASKED_OP2(gabornoise, A, B)                                \
            (name_ptr, r_ptr, x_ptr, bsg, opt, varying_direction_ptr,         \
             mask_value);                                                     \
        } else if (name == Strings::null) {                                   \
            __OSL_MASKED_OP2(nullnoise, A, B)(r_ptr, x_ptr, mask_value);      \
        } else if (name == Strings::unull) {                                  \
            __OSL_MASKED_OP2(unullnoise, A, B)(r_ptr, x_ptr, mask_value);     \
        } else if (name == Strings::hash) {                                   \
            /* NOTE: calling non derivative version */                        \
            __OSL_MASKED_OP2(hashnoise, NONDERIV_A, NONDERIV_B)               \
            (r_ptr, x_ptr, mask_value);                                       \
            Masked<DUALTYPE> wr(r_ptr, Mask(mask_value));                     \
            zero_derivs(wr);                                                  \
        } else {                                                              \
            ((BatchedShaderGlobals*)bsg)                                      \
                ->uniform.context->errorf("Unknown noise type \"%s\"",        \
                                          name.c_str());                      \
        }                                                                     \
    }


__OSL_GENERIC_DISPATCH2(Wdf, Wdf, Wf, Wf, Dual2<float>)


#define __OSL_GENERIC_DISPATCH3(A, B, C, NONDERIV_A, NONDERIV_B, NONDERIV_C,   \
                                DUALTYPE)                                      \
    OSL_BATCHOP void __OSL_MASKED_OP3(gabornoise, A, B, C)(                    \
        char* name_ptr, char* r_ptr, char* x_ptr, char* y_ptr, char* bsg,      \
        char* opt, char* varying_direction_ptr, unsigned int mask_value);      \
    OSL_BATCHOP void __OSL_MASKED_OP3(noise, A, B,                             \
                                      C)(char* r_ptr, char* x_ptr,             \
                                         char* y_ptr,                          \
                                         unsigned int mask_value);             \
    OSL_BATCHOP void __OSL_MASKED_OP3(simplexnoise, A, B,                      \
                                      C)(char* r_ptr, char* x_ptr,             \
                                         char* y_ptr,                          \
                                         unsigned int mask_value);             \
    OSL_BATCHOP void __OSL_MASKED_OP3(snoise, A, B,                            \
                                      C)(char* r_ptr, char* x_ptr,             \
                                         char* y_ptr,                          \
                                         unsigned int mask_value);             \
    OSL_BATCHOP void __OSL_MASKED_OP3(usimplexnoise, A, B,                     \
                                      C)(char* r_ptr, char* x_ptr,             \
                                         char* y_ptr,                          \
                                         unsigned int mask_value);             \
    OSL_BATCHOP void __OSL_MASKED_OP3(cellnoise, NONDERIV_A, NONDERIV_B,       \
                                      NONDERIV_C)(char* r_ptr, char* x_ptr,    \
                                                  char* y_ptr,                 \
                                                  unsigned int mask_value);    \
    OSL_BATCHOP void __OSL_MASKED_OP3(nullnoise, A, B,                         \
                                      C)(char* r_ptr, char* x_ptr,             \
                                         char* y_ptr,                          \
                                         unsigned int mask_value);             \
    OSL_BATCHOP void __OSL_MASKED_OP3(unullnoise, A, B,                        \
                                      C)(char* r_ptr, char* x_ptr,             \
                                         char* y_ptr,                          \
                                         unsigned int mask_value);             \
    OSL_BATCHOP void __OSL_MASKED_OP3(hashnoise, NONDERIV_A, NONDERIV_B,       \
                                      NONDERIV_C)(char* r_ptr, char* x_ptr,    \
                                                  char* y_ptr,                 \
                                                  unsigned int mask_value);    \
    OSL_BATCHOP void __OSL_MASKED_OP3(genericnoise, A, B, C)(                  \
        char* name_ptr, char* r_ptr, char* x_ptr, char* y_ptr, char* bsg,      \
        char* opt, char* varying_direction_ptr, unsigned int mask_value)       \
    {                                                                          \
        ustring name = USTR(name_ptr);                                         \
        if (name == Strings::uperlin || name == Strings::noise) {              \
            __OSL_MASKED_OP3(noise, A, B, C)(r_ptr, x_ptr, y_ptr, mask_value); \
        } else if (name == Strings::perlin || name == Strings::snoise) {       \
            __OSL_MASKED_OP3(snoise, A, B, C)                                  \
            (r_ptr, x_ptr, y_ptr, mask_value);                                 \
        } else if (name == Strings::simplexnoise                               \
                   || name == Strings::simplex) {                              \
            __OSL_MASKED_OP3(simplexnoise, A, B, C)                            \
            (r_ptr, x_ptr, y_ptr, mask_value);                                 \
        } else if (name == Strings::usimplexnoise                              \
                   || name == Strings::usimplex) {                             \
            __OSL_MASKED_OP3(usimplexnoise, A, B, C)                           \
            (r_ptr, x_ptr, y_ptr, mask_value);                                 \
        } else if (name == Strings::cell) {                                    \
            /* NOTE: calling non derivative version */                         \
            __OSL_MASKED_OP3(cellnoise, NONDERIV_A, NONDERIV_B, NONDERIV_C)    \
            (r_ptr, x_ptr, y_ptr, mask_value);                                 \
            Masked<DUALTYPE> wr(r_ptr, Mask(mask_value));                      \
            zero_derivs(wr);                                                   \
        } else if (name == Strings::gabor) {                                   \
            __OSL_MASKED_OP3(gabornoise, A, B, C)                              \
            (name_ptr, r_ptr, x_ptr, y_ptr, bsg, opt, varying_direction_ptr,   \
             mask_value);                                                      \
        } else if (name == Strings::null) {                                    \
            __OSL_MASKED_OP3(nullnoise, A, B, C)                               \
            (r_ptr, x_ptr, y_ptr, mask_value);                                 \
        } else if (name == Strings::unull) {                                   \
            __OSL_MASKED_OP3(unullnoise, A, B, C)                              \
            (r_ptr, x_ptr, y_ptr, mask_value);                                 \
        } else if (name == Strings::hash) {                                    \
            /* NOTE: calling non derivative version */                         \
            __OSL_MASKED_OP3(hashnoise, NONDERIV_A, NONDERIV_B, NONDERIV_C)    \
            (r_ptr, x_ptr, y_ptr, mask_value);                                 \
            Masked<DUALTYPE> wr(r_ptr, Mask(mask_value));                      \
            zero_derivs(wr);                                                   \
        } else {                                                               \
            ((BatchedShaderGlobals*)bsg)                                       \
                ->uniform.context->errorf("Unknown noise type \"%s\"",         \
                                          name.c_str());                       \
        }                                                                      \
    }

__OSL_GENERIC_DISPATCH3(Wdf, Wdf, Wdf, Wf, Wf, Wf, Dual2<float>)
__OSL_GENERIC_DISPATCH2(Wdf, Wdv, Wf, Wv, Dual2<float>)
__OSL_GENERIC_DISPATCH3(Wdf, Wdv, Wdf, Wf, Wv, Wf, Dual2<float>)

__OSL_GENERIC_DISPATCH3(Wdv, Wdv, Wdf, Wv, Wv, Wf, Dual2<Vec3>)
__OSL_GENERIC_DISPATCH2(Wdv, Wdf, Wv, Wf, Dual2<Vec3>)
__OSL_GENERIC_DISPATCH3(Wdv, Wdf, Wdf, Wv, Wf, Wf, Dual2<Vec3>)
__OSL_GENERIC_DISPATCH2(Wdv, Wdv, Wv, Wv, Dual2<Vec3>)


}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#include "undef_opname_macros.h"
