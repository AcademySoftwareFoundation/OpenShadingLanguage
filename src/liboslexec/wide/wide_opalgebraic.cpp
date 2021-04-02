// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader implementation of Algebraic operations
/// NOTE: Execute from the library (vs. LLVM-IR) to take advantage
/// of compiler's small vector math library.
///
/////////////////////////////////////////////////////////////////////////

#include <cmath>

#include <OSL/batched_shaderglobals.h>
#include <OSL/dual.h>
#include <OSL/dual_vec.h>
#include <OSL/oslconfig.h>
#include <OSL/sfmath.h>
#include <OSL/wide.h>

#include <OpenImageIO/fmath.h>

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

#include "define_opname_macros.h"

#define __OSL_XMACRO_ARGS (sqrt, OIIO::safe_sqrt, sqrt)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS (inversesqrt, OIIO::safe_inversesqrt, inversesqrt)
#include "wide_opunary_per_component_xmacro.h"

// emitted directly by llvm_gen_wide.cpp
//MAKE_BINARY_FI_OP(safe_div, sfm::safe_div, sfm::safe_div)

#define __OSL_XMACRO_ARGS (floor, floorf)
#include "wide_opunary_per_component_float_or_vector_xmacro.h"

#define __OSL_XMACRO_ARGS (ceil, ceilf)
#include "wide_opunary_per_component_float_or_vector_xmacro.h"

#define __OSL_XMACRO_ARGS (trunc, truncf)
#include "wide_opunary_per_component_float_or_vector_xmacro.h"

#define __OSL_XMACRO_ARGS (round, roundf)
#include "wide_opunary_per_component_float_or_vector_xmacro.h"


static OSL_FORCEINLINE float
impl_sign(float x)
{
    // Avoid nested conditional logic as per language
    // rules the right operand may only be evaluated
    // if the 1st conditional is false.
    // Thus complex control flow vs. just 2 compares
    // and masked assignments
    //return x < 0.0f ? -1.0f : (x==0.0f ? 0.0f : 1.0f);
    float sign = 0.0f;
    if (x < 0.0f)
        sign = -1.0f;
    if (x > 0.0f)
        sign = 1.0f;
    return sign;
}
#define __OSL_XMACRO_ARGS (sign, impl_sign)
#include "wide_opunary_per_component_float_or_vector_xmacro.h"

// TODO: move to dual.h
OSL_FORCEINLINE OSL_HOSTDEVICE Dual2<float>
abs(const Dual2<float>& x)
{
    return x.val() >= 0 ? x : sfm::negate(x);
}

#define __OSL_XMACRO_ARGS (abs, std::abs, abs)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS (fabs, std::abs, abs)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS (abs, std::abs)
#include "wide_opunary_int_xmacro.h"

#define __OSL_XMACRO_ARGS (fabs, std::abs)
#include "wide_opunary_int_xmacro.h"

#define __OSL_XMACRO_ARGS (fmod, OIIO::safe_fmod)
#include "wide_opbinary_per_component_float_or_vector_xmacro.h"


static OSL_FORCEINLINE float
impl_step(float edge, float x)
{
    // Avoid ternary, as only constants are in the
    // conditional branches, this may be unnecessary.
    // return x < edge ? 0.0f : 1.0f;
    float result = 0.0f;
    if (x >= edge) {
        result = 1.0f;
    }
    return result;
}

// TODO: consider moving step to batched_llvm_gen.cpp
#define __OSL_XMACRO_ARGS (step, impl_step)
#include "wide_opbinary_per_component_float_or_vector_xmacro.h"


}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#include "undef_opname_macros.h"
