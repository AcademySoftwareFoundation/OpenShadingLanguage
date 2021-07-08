// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of Transcendental operations
/// NOTE: many functions are left as LLVM IR, but some are better to
/// execute from the library to take advantage of compiler's small vector
/// math library versions.
///
/////////////////////////////////////////////////////////////////////////

#include <cmath>

#include <OSL/oslconfig.h>

#include <OSL/batched_shaderglobals.h>
#include <OSL/dual.h>
#include <OSL/dual_vec.h>
#include <OSL/sfmath.h>
#include <OSL/wide.h>

#include <OpenImageIO/fmath.h>

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

#include "define_opname_macros.h"

#if OSL_FAST_MATH
#define __OSL_XMACRO_ARGS     (log        , OIIO::fast_log       , fast_log)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS     (log2       , OIIO::fast_log2      , fast_log2)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS     (log10      , OIIO::fast_log10     , fast_log10)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS     (exp        , OIIO::fast_exp       , fast_exp)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS     (exp2       , OIIO::fast_exp2      , fast_exp2)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS     (expm1       , OIIO::fast_expm1    , fast_expm1)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS (pow        , OIIO::fast_safe_pow  , fast_safe_pow)
#include "wide_opbinary_per_component_xmacro.h"
#define __OSL_XMACRO_ARGS (pow        , OIIO::fast_safe_pow  , fast_safe_pow)
#include "wide_opbinary_per_component_mixed_vector_float_xmacro.h"

#define __OSL_XMACRO_ARGS (erf, OIIO::fast_erf, fast_erf)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS (erfc, OIIO::fast_erfc, fast_erfc)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS (cbrt, OIIO::fast_cbrt, fast_cbrt)
#include "wide_opunary_per_component_xmacro.h"

#else
#define __OSL_XMACRO_ARGS     (log        , OIIO::safe_log       , safe_log)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS     (log2       , OIIO::safe_log2      , safe_log2)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS     (log10      , OIIO::safe_log10     , safe_log10)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS     (exp        , expf                 , exp)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS     (exp2       , exp2f                , exp2)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS     (expm1      , expm1f               , expm1)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS     (pow        , OIIO::safe_pow       , safe_pow)
#define __OSL_XMACRO_MASKED_ONLY
#include "wide_opbinary_per_component_xmacro.h"
#define __OSL_XMACRO_ARGS     (pow        , OIIO::safe_pow       , safe_pow)
#define __OSL_XMACRO_MASKED_ONLY
#include "wide_opbinary_per_component_mixed_vector_float_xmacro.h"

#define __OSL_XMACRO_ARGS     (erf        , erff                 , erf)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS     (erfc       , erfcf                , erfc)
#include "wide_opunary_per_component_xmacro.h"

#define __OSL_XMACRO_ARGS     (cbrt       , cbrtf                , cbrt)
#include "wide_opunary_per_component_xmacro.h"

#endif

#define __OSL_XMACRO_ARGS (logb, OIIO::fast_logb)
#include "wide_opunary_per_component_float_or_vector_xmacro.h"

}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#include "undef_opname_macros.h"
