// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// To improve parallel compile times, split noise with float results and
// Vec3 results into different cpp files
#define __OSL_XMACRO_FLOAT_RESULTS_ONLY
#define __OSL_XMACRO_ARGS (simplexnoise, SimplexNoiseScalar, SimplexNoise)
#include "wide_opnoise_impl_deriv_xmacro.h"
