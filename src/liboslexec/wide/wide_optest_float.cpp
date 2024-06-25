// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

/////////////////////////////////////////////////////////////////////////
/// \file
///
/// Shader interpreter implementation of Floating Point Test operations
/// NOTE: many functions are left as LLVM IR, but some are better to
/// execute from the library to take advantage of compiler's small vector
/// math library versions.
///
/////////////////////////////////////////////////////////////////////////

#include <OSL/oslconfig.h>

#include <OSL/sfmath.h>
#include <OSL/wide.h>
#include <OpenImageIO/fmath.h>

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

#include "define_opname_macros.h"

#define __OSL_XMACRO_ARGS (isnan, std::isnan)
#include "wide_optest_float_xmacro.h"

#define __OSL_XMACRO_ARGS (isinf, std::isinf)
//#define __OSL_XMACRO_ARGS (isinf, sfm::isinf)
#include "wide_optest_float_xmacro.h"

#define __OSL_XMACRO_ARGS (isfinite, std::isfinite)
#include "wide_optest_float_xmacro.h"


}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#include "undef_opname_macros.h"
