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

OSL_NAMESPACE_BEGIN
namespace __OSL_WIDE_PVT {

#include "define_opname_macros.h"
#define __OSL_NOISE_OP2(A, B)    __OSL_MASKED_OP2(gabornoise, A, B)
#define __OSL_NOISE_OP3(A, B, C) __OSL_MASKED_OP3(gabornoise, A, B, C)


#define LOOKUP_WIDE_GABOR_IMPL_BY_OPT(lookup_name, func_name)            \
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

LOOKUP_WIDE_GABOR_IMPL_BY_OPT(lookup_wide_float_impl, wide_gabor)
LOOKUP_WIDE_GABOR_IMPL_BY_OPT(lookup_wide_Vec3_impl, wide_gabor3)

template<typename... ArgsT>
void
dispatch_float_result(const NoiseParams* opt,
                      Block<Vec3>* opt_varying_direction, ArgsT... args)
{
    typedef void (*FuncPtr)(ArgsT..., const NoiseParams* opt, Block<Vec3>*);

    lookup_wide_float_impl<FuncPtr>(opt)(args..., opt, opt_varying_direction);
}

template<typename... ArgsT>
void
dispatch_Vec3_result(const NoiseParams* opt, Block<Vec3>* opt_varying_direction,
                     ArgsT... args)
{
    typedef void (*FuncPtr)(ArgsT..., const NoiseParams* opt, Block<Vec3>*);

    lookup_wide_Vec3_impl<FuncPtr>(opt)(args..., opt, opt_varying_direction);
}

}  // namespace

OSL_BATCHOP void
__OSL_NOISE_OP2(Wdf, Wdf)(char* name, char* r_ptr, char* x_ptr, char* bsg,
                          char* opt, char* varying_direction_ptr,
                          unsigned int mask_value)
{
    dispatch_float_result(reinterpret_cast<const NoiseParams*>(opt),
                          reinterpret_cast<Block<Vec3>*>(varying_direction_ptr),
                          Masked<Dual2<Float>>(r_ptr, Mask(mask_value)),
                          Wide<const Dual2<float>>(x_ptr));
}



OSL_BATCHOP void
__OSL_NOISE_OP3(Wdf, Wdf, Wdf)(char* name, char* r_ptr, char* x_ptr,
                               char* y_ptr, char* bsg, char* opt,
                               char* varying_direction_ptr,
                               unsigned int mask_value)
{
    dispatch_float_result(reinterpret_cast<const NoiseParams*>(opt),
                          reinterpret_cast<Block<Vec3>*>(varying_direction_ptr),
                          Masked<Dual2<Float>>(r_ptr, Mask(mask_value)),
                          Wide<const Dual2<Float>>(x_ptr),
                          Wide<const Dual2<Float>>(y_ptr));
}



OSL_BATCHOP void
__OSL_NOISE_OP2(Wdf, Wdv)(char* name, char* r_ptr, char* p_ptr, char* bsg,
                          char* opt, char* varying_direction_ptr,
                          unsigned int mask_value)
{
    dispatch_float_result(reinterpret_cast<const NoiseParams*>(opt),
                          reinterpret_cast<Block<Vec3>*>(varying_direction_ptr),
                          Masked<Dual2<Float>>(r_ptr, Mask(mask_value)),
                          Wide<const Dual2<Vec3>>(p_ptr));
}



OSL_BATCHOP void
__OSL_NOISE_OP3(Wdf, Wdv, Wdf)(char* name, char* r_ptr, char* p_ptr,
                               char* t_ptr, char* bsg, char* opt,
                               char* varying_direction_ptr,
                               unsigned int mask_value)
{
    /* FIXME -- This is very broken, we are ignoring 4D! */
    dispatch_float_result(reinterpret_cast<const NoiseParams*>(opt),
                          reinterpret_cast<Block<Vec3>*>(varying_direction_ptr),
                          Masked<Dual2<float>>(r_ptr, Mask(mask_value)),
                          Wide<const Dual2<Vec3>>(p_ptr));
}



OSL_BATCHOP void
__OSL_NOISE_OP3(Wdv, Wdv, Wdf)(char* name, char* r_ptr, char* p_ptr,
                               char* t_ptr, char* bsg, char* opt,
                               char* varying_direction_ptr,
                               unsigned int mask_value)
{
    /* FIXME -- This is very broken, we are ignoring 4D! */
    dispatch_Vec3_result(reinterpret_cast<const NoiseParams*>(opt),
                         reinterpret_cast<Block<Vec3>*>(varying_direction_ptr),
                         Masked<Dual2<Vec3>>(r_ptr, Mask(mask_value)),
                         Wide<const Dual2<Vec3>>(p_ptr));
}



OSL_BATCHOP void
__OSL_NOISE_OP2(Wdv, Wdf)(char* name, char* r_ptr, char* x_ptr, char* bsg,
                          char* opt, char* varying_direction_ptr,
                          unsigned int mask_value)
{
    dispatch_Vec3_result(reinterpret_cast<const NoiseParams*>(opt),
                         reinterpret_cast<Block<Vec3>*>(varying_direction_ptr),
                         Masked<Dual2<Vec3>>(r_ptr, Mask(mask_value)),
                         Wide<const Dual2<float>>(x_ptr));
}



OSL_BATCHOP void
__OSL_NOISE_OP3(Wdv, Wdf, Wdf)(char* name, char* r_ptr, char* x_ptr,
                               char* y_ptr, char* bsg, char* opt,
                               char* varying_direction_ptr,
                               unsigned int mask_value)
{
    dispatch_Vec3_result(reinterpret_cast<const NoiseParams*>(opt),
                         reinterpret_cast<Block<Vec3>*>(varying_direction_ptr),
                         Masked<Dual2<Vec3>>(r_ptr, Mask(mask_value)),
                         Wide<const Dual2<float>>(x_ptr),
                         Wide<const Dual2<float>>(y_ptr));
}



OSL_BATCHOP void
__OSL_NOISE_OP2(Wdv, Wdv)(char* name, char* r_ptr, char* p_ptr, char* bsg,
                          char* opt, char* varying_direction_ptr,
                          unsigned int mask_value)
{
    dispatch_Vec3_result(reinterpret_cast<const NoiseParams*>(opt),
                         reinterpret_cast<Block<Vec3>*>(varying_direction_ptr),
                         Masked<Dual2<Vec3>>(r_ptr, Mask(mask_value)),
                         Wide<const Dual2<Vec3>>(p_ptr));
}



}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_END

#undef LOOKUP_WIDE_GABOR_IMPL_BY_OPT

#undef __OSL_NOISE_OP2
#undef __OSL_NOISE_OP3

#include "undef_opname_macros.h"
