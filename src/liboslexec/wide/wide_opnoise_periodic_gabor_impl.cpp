// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <limits>

#include <OSL/oslconfig.h>

#include "oslexec_pvt.h"

#include <OSL/Imathx/Imathx.h>
#include <OSL/dual_vec.h>
#include <OSL/oslnoise.h>
#include <OSL/wide/wide_gabornoise_fwd.h>

#include <OpenImageIO/fmath.h>

using namespace OSL;

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

/***********************************************************************
 * batched periodic gabor routines callable by the LLVM-generated code.
 */

#include "define_opname_macros.h"
#define __OSL_PNOISE_OP3(A, B, C) __OSL_MASKED_OP3(gaborpnoise, A, B, C)
#define __OSL_PNOISE_OP5(A, B, C, D, E) \
    __OSL_MASKED_OP5(gaborpnoise, A, B, C, D, E)


#define LOOKUP_WIDE_PGABOR_IMPL_BY_OPT(lookup_name, func_name)           \
    template<typename FuncPtrT>                                          \
    static OSL_FORCEINLINE FuncPtrT lookup_name(const NoiseParams* opt)  \
    {                                                                    \
        static constexpr FuncPtrT impl_by_filter_and_ansiotropic[2][3]   \
            = { {                                                        \
                    /*disabled filter*/                                  \
                    &func_name<0 /*isotropic*/, DisabledFilterPolicy>,   \
                    &func_name<1 /*ansiotropic*/, DisabledFilterPolicy>, \
                    &func_name<2 /*hybrid*/, DisabledFilterPolicy>,      \
                },                                                       \
                {                                                        \
                    /*enabled filter*/                                   \
                    &func_name<0 /*isotropic*/, EnabledFilterPolicy>,    \
                    &func_name<1 /*ansiotropic*/, EnabledFilterPolicy>,  \
                    &func_name<2 /*hybrid*/, EnabledFilterPolicy>,       \
                } };                                                     \
        int clampedAnisotropic = opt->anisotropic;                       \
        if (clampedAnisotropic != 0 && clampedAnisotropic != 1) {        \
            clampedAnisotropic = 2;                                      \
        }                                                                \
        return impl_by_filter_and_ansiotropic[opt->do_filter]            \
                                             [clampedAnisotropic];       \
    }

namespace  // anonymous
{

LOOKUP_WIDE_PGABOR_IMPL_BY_OPT(lookup_wide_pgabor_float_impl, wide_pgabor)
LOOKUP_WIDE_PGABOR_IMPL_BY_OPT(lookup_wide_pgabor_Vec3_impl, wide_pgabor3)

template<typename... ArgsT>
void
dispatch_pgabor_float_result(const NoiseParams* opt,
                             Block<Vec3>* opt_varying_direction, ArgsT... args)
{
    typedef void (*FuncPtr)(ArgsT..., const NoiseParams* opt, Block<Vec3>*);

    lookup_wide_pgabor_float_impl<FuncPtr>(opt)(args..., opt,
                                                opt_varying_direction);
}

template<typename... ArgsT>
void
dispatch_pgabor_Vec3_result(const NoiseParams* opt,
                            Block<Vec3>* opt_varying_direction, ArgsT... args)
{
    typedef void (*FuncPtr)(ArgsT..., const NoiseParams* opt, Block<Vec3>*);

    lookup_wide_pgabor_Vec3_impl<FuncPtr>(opt)(args..., opt,
                                               opt_varying_direction);
}

}  // namespace

OSL_BATCHOP void __OSL_PNOISE_OP3(Wdf, Wdf, Wf)(char* name, char* r_ptr,
                                                char* x_ptr, char* px_ptr,
                                                char* sgb, char* opt,
                                                char* varying_direction_ptr,
                                                unsigned int mask_value)
{
    dispatch_pgabor_float_result(reinterpret_cast<const NoiseParams*>(opt),
                                 reinterpret_cast<Block<Vec3>*>(
                                     varying_direction_ptr),
                                 Masked<Dual2<Float>>(r_ptr, Mask(mask_value)),
                                 Wide<const Dual2<float>>(x_ptr),
                                 Wide<const float>(px_ptr));
}

OSL_BATCHOP void __OSL_PNOISE_OP5(Wdf, Wdf, Wdf, Wf,
                                  Wf)(char* name, char* r_ptr, char* x_ptr,
                                      char* y_ptr, char* px_ptr, char* py_ptr,
                                      char* sgb, char* opt,
                                      char* varying_direction_ptr,
                                      unsigned int mask_value)
{
    dispatch_pgabor_float_result(
        reinterpret_cast<const NoiseParams*>(opt),
        reinterpret_cast<Block<Vec3>*>(varying_direction_ptr),
        Masked<Dual2<Float>>(r_ptr, Mask(mask_value)),
        Wide<const Dual2<Float>>(x_ptr), Wide<const Dual2<Float>>(y_ptr),
        Wide<const float>(px_ptr), Wide<const float>(py_ptr));
}

OSL_BATCHOP void __OSL_PNOISE_OP3(Wdf, Wdv, Wv)(char* name, char* r_ptr,
                                                char* p_ptr, char* pp_ptr,
                                                char* sgb, char* opt,
                                                char* varying_direction_ptr,
                                                unsigned int mask_value)
{
    dispatch_pgabor_float_result(reinterpret_cast<const NoiseParams*>(opt),
                                 reinterpret_cast<Block<Vec3>*>(
                                     varying_direction_ptr),
                                 Masked<Dual2<Float>>(r_ptr, Mask(mask_value)),
                                 Wide<const Dual2<Vec3>>(p_ptr),
                                 Wide<const Vec3>(pp_ptr));
}

OSL_BATCHOP void __OSL_PNOISE_OP5(Wdf, Wdv, Wdf, Wv,
                                  Wf)(char* name, char* r_ptr, char* p_ptr,
                                      char* t_ptr, char* pp_ptr, char* pt_ptr,
                                      char* sgb, char* opt,
                                      char* varying_direction_ptr,
                                      unsigned int mask_value)
{
    /* FIXME -- This is very broken, we are ignoring 4D! */
    dispatch_pgabor_float_result(reinterpret_cast<const NoiseParams*>(opt),
                                 reinterpret_cast<Block<Vec3>*>(
                                     varying_direction_ptr),
                                 Masked<Dual2<float>>(r_ptr, Mask(mask_value)),
                                 Wide<const Dual2<Vec3>>(p_ptr),
                                 Wide<const Vec3>(pp_ptr));
}

OSL_BATCHOP void __OSL_PNOISE_OP3(Wdv, Wdf, Wf)(char* name, char* r_ptr,
                                                char* x_ptr, char* px_ptr,
                                                char* sgb, char* opt,
                                                char* varying_direction_ptr,
                                                unsigned int mask_value)
{
    dispatch_pgabor_Vec3_result(reinterpret_cast<const NoiseParams*>(opt),
                                reinterpret_cast<Block<Vec3>*>(
                                    varying_direction_ptr),
                                Masked<Dual2<Vec3>>(r_ptr, Mask(mask_value)),
                                Wide<const Dual2<float>>(x_ptr),
                                Wide<const float>(px_ptr));
}

OSL_BATCHOP void __OSL_PNOISE_OP5(Wdv, Wdf, Wdf, Wf,
                                  Wf)(char* name, char* r_ptr, char* x_ptr,
                                      char* y_ptr, char* px_ptr, char* py_ptr,
                                      char* sgb, char* opt,
                                      char* varying_direction_ptr,
                                      unsigned int mask_value)
{
    dispatch_pgabor_Vec3_result(
        reinterpret_cast<const NoiseParams*>(opt),
        reinterpret_cast<Block<Vec3>*>(varying_direction_ptr),
        Masked<Dual2<Vec3>>(r_ptr, Mask(mask_value)),
        Wide<const Dual2<float>>(x_ptr), Wide<const Dual2<float>>(y_ptr),
        Wide<const float>(px_ptr), Wide<const float>(py_ptr));
}

OSL_BATCHOP void __OSL_PNOISE_OP3(Wdv, Wdv, Wv)(char* name, char* r_ptr,
                                                char* p_ptr, char* pp_ptr,
                                                char* sgb, char* opt,
                                                char* varying_direction_ptr,
                                                unsigned int mask_value)
{
    dispatch_pgabor_Vec3_result(reinterpret_cast<const NoiseParams*>(opt),
                                reinterpret_cast<Block<Vec3>*>(
                                    varying_direction_ptr),
                                Masked<Dual2<Vec3>>(r_ptr, Mask(mask_value)),
                                Wide<const Dual2<Vec3>>(p_ptr),
                                Wide<const Vec3>(pp_ptr));
}

OSL_BATCHOP void __OSL_PNOISE_OP5(Wdv, Wdv, Wdf, Wv,
                                  Wf)(char* name, char* r_ptr, char* p_ptr,
                                      char* t_ptr, char* pp_ptr, char* pt_ptr,
                                      char* sgb, char* opt,
                                      char* varying_direction_ptr,
                                      unsigned int mask_value)
{
    /* FIXME -- This is very broken, we are ignoring 4D! */
    dispatch_pgabor_Vec3_result(reinterpret_cast<const NoiseParams*>(opt),
                                reinterpret_cast<Block<Vec3>*>(
                                    varying_direction_ptr),
                                Masked<Dual2<Vec3>>(r_ptr, Mask(mask_value)),
                                Wide<const Dual2<Vec3>>(p_ptr),
                                Wide<const Vec3>(pp_ptr));
}


}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#undef __OSL_PNOISE_OP3
#undef __OSL_PNOISE_OP5
#undef LOOKUP_WIDE_PGABOR_IMPL_BY_OPT

#include "undef_opname_macros.h"
