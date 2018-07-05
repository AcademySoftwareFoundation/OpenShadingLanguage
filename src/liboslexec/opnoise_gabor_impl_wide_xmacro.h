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
#ifdef __OSL_XMACRO_ARGS

#define __OSL_XMACRO_LANE_COUNT __OSL_EXPAND(__OSL_XMACRO_ARG1 __OSL_XMACRO_ARGS)

#endif

#ifndef __OSL_XMACRO_LANE_COUNT
#error must define __OSL_XMACRO_LANE_COUNT to number of SIMD lanes before including this header
#endif


#define __WDV __OSL_CONCAT3(w,__OSL_XMACRO_LANE_COUNT,dv)
#define __WDF __OSL_CONCAT3(w,__OSL_XMACRO_LANE_COUNT,df)

#define __OSL_NOISE_OP2(A, B) __OSL_CONCAT6(osl_,gabornoise,_,A,B,_masked)
#define __OSL_NOISE_OP3(A, B, C) __OSL_CONCAT7(osl_,gabornoise,_,A,B,C,_masked)

#define __OSL_FUNCNAME3 __OSL_CONCAT(__OSL_XMACRO_FUNCNAMEBASE,3)

#define LOOKUP_WIDE_GABOR_IMPL_BY_OPT(lookup_name, func_name) \
template<typename FuncPtrT> \
static OSL_INLINE FuncPtrT lookup_name(const NoiseParams *opt) { \
    static constexpr FuncPtrT impl_by_filter_and_ansiotropic[2][4] = { \
        { /*disabled filter*/ \
            &func_name<0 /*isotropic*/, DisabledFilterPolicy, __OSL_SIMD_LANE_COUNT>, \
            &func_name<1 /*ansiotropic*/, DisabledFilterPolicy, __OSL_SIMD_LANE_COUNT>, \
            nullptr, \
            &func_name<3 /*hybrid*/, DisabledFilterPolicy, __OSL_SIMD_LANE_COUNT>, \
        }, \
        { /*enabled filter*/ \
            &func_name<0 /*isotropic*/, EnabledFilterPolicy, __OSL_SIMD_LANE_COUNT>, \
            &func_name<1 /*ansiotropic*/, EnabledFilterPolicy, __OSL_SIMD_LANE_COUNT>, \
            nullptr, \
            &func_name<3 /*hybrid*/, EnabledFilterPolicy, __OSL_SIMD_LANE_COUNT>, \
        } \
    }; \
    return impl_by_filter_and_ansiotropic[opt->do_filter][opt->anisotropic]; \
} \

namespace // anonymous
{

    LOOKUP_WIDE_GABOR_IMPL_BY_OPT(lookup_wide_float_impl, wide_gabor)
    LOOKUP_WIDE_GABOR_IMPL_BY_OPT(lookup_wide_Vec3_impl, wide_gabor3)

    template<typename ... ArgsT>
    void dispatch_float_result(const NoiseParams *opt, ArgsT ... args) {
        typedef void (*FuncPtr)(ArgsT..., const NoiseParams *opt);

        lookup_wide_float_impl<FuncPtr>(opt)(args..., opt);
    }

    template<typename ... ArgsT>
    void dispatch_Vec3_result(const NoiseParams *opt, ArgsT ... args) {
        typedef void (*FuncPtr)(ArgsT..., const NoiseParams *opt);

        lookup_wide_Vec3_impl<FuncPtr>(opt)(args..., opt);
    }

} // anonymous

OSL_SHADEOP void __OSL_NOISE_OP2(__WDF,__WDF) (
        char *name, char *r_ptr, char *x_ptr, char *sgb, char *opt, unsigned int mask_value) {
    dispatch_float_result ((const NoiseParams *)opt,
                MaskedAccessor<Dual2<Float>,__OSL_XMACRO_LANE_COUNT>(r_ptr, Mask(mask_value)),
                ConstWideAccessor<Dual2<float>,__OSL_XMACRO_LANE_COUNT>(x_ptr));
}

OSL_SHADEOP void __OSL_NOISE_OP3(__WDF,__WDF,__WDF) (
        char *name, char *r_ptr, char *x_ptr, char *y_ptr, char *sgb, char *opt, unsigned int mask_value) {
    dispatch_float_result ((const NoiseParams *)opt,
                  MaskedAccessor<Dual2<Float>,__OSL_XMACRO_LANE_COUNT>(r_ptr, Mask(mask_value)),
                  ConstWideAccessor<Dual2<Float>,__OSL_XMACRO_LANE_COUNT>(x_ptr),
                  ConstWideAccessor<Dual2<Float>,__OSL_XMACRO_LANE_COUNT>(y_ptr));
}

OSL_SHADEOP void __OSL_NOISE_OP2(__WDF,__WDV) (
        char *name, char *r_ptr, char *p_ptr, char *sgb, char *opt, unsigned int mask_value) {
    dispatch_float_result ((const NoiseParams *)opt,
                  MaskedAccessor<Dual2<Float>,__OSL_XMACRO_LANE_COUNT>(r_ptr, Mask(mask_value)),
                  ConstWideAccessor<Dual2<Vec3>,__OSL_XMACRO_LANE_COUNT>(p_ptr));
}

OSL_SHADEOP void __OSL_NOISE_OP3(__WDF,__WDV,__WDF) (
        char *name, char *r_ptr, char *p_ptr, char *t_ptr, char *sgb, char *opt, unsigned int mask_value) {
    /* FIXME -- This is very broken, we are ignoring 4D! */
    dispatch_float_result (  (const NoiseParams *)opt,
                MaskedAccessor<Dual2<float>,__OSL_XMACRO_LANE_COUNT>(r_ptr, Mask(mask_value)),
                ConstWideAccessor<Dual2<Vec3>,__OSL_XMACRO_LANE_COUNT>(p_ptr));
}



OSL_SHADEOP void __OSL_NOISE_OP3(__WDV,__WDV,__WDF) (
        char *name, char *r_ptr, char *p_ptr, char *t_ptr, char *sgb, char *opt, unsigned int mask_value) {
    /* FIXME -- This is very broken, we are ignoring 4D! */
    dispatch_Vec3_result (  (const NoiseParams *)opt,
        MaskedAccessor<Dual2<Vec3>,__OSL_XMACRO_LANE_COUNT>(r_ptr, Mask(mask_value)),
        ConstWideAccessor<Dual2<Vec3>,__OSL_XMACRO_LANE_COUNT>(p_ptr));
}

OSL_SHADEOP void __OSL_NOISE_OP2(__WDV,__WDF) (
        char *name, char *r_ptr, char *x_ptr, char *sgb, char *opt, unsigned int mask_value) {
    dispatch_Vec3_result (  (const NoiseParams *)opt,
        MaskedAccessor<Dual2<Vec3>,__OSL_XMACRO_LANE_COUNT>(r_ptr, Mask(mask_value)),
        ConstWideAccessor<Dual2<float>,__OSL_XMACRO_LANE_COUNT>(x_ptr));
}

OSL_SHADEOP void __OSL_NOISE_OP3(__WDV,__WDF,__WDF) (
        char *name, char *r_ptr, char *x_ptr, char *y_ptr, char *sgb, char *opt, unsigned int mask_value) {
    dispatch_Vec3_result (  (const NoiseParams *)opt,
        MaskedAccessor<Dual2<Vec3>,__OSL_XMACRO_LANE_COUNT>(r_ptr, Mask(mask_value)),
        ConstWideAccessor<Dual2<float>,__OSL_XMACRO_LANE_COUNT>(x_ptr),
        ConstWideAccessor<Dual2<float>,__OSL_XMACRO_LANE_COUNT>(y_ptr));
}

OSL_SHADEOP void __OSL_NOISE_OP2(__WDV,__WDV) (
        char *name, char *r_ptr, char *p_ptr, char *sgb, char *opt, unsigned int mask_value) {
    dispatch_Vec3_result (  (const NoiseParams *)opt,
        MaskedAccessor<Dual2<Vec3>,__OSL_XMACRO_LANE_COUNT>(r_ptr, Mask(mask_value)),
        ConstWideAccessor<Dual2<Vec3>,__OSL_XMACRO_LANE_COUNT>(p_ptr));
}


#undef LOOKUP_WIDE_GABOR_IMPL_BY_OPT
#undef __OSL_XMACRO_ARGS
#undef __OSL_XMACRO_LANE_COUNT


#undef __OSL_FUNCNAME3

#undef __WDV
#undef __WDF
#undef __OSL_NOISE_OP2
#undef __OSL_NOISE_OP3

