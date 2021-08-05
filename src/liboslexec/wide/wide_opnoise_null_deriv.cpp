// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// Force SIMD execution with threshold of 1
#define __OSL_BATCHED_CG_SIMD_THRESHOLD 1
#include "null_noise.h"

#define __OSL_XMACRO_ARGS (nullnoise, NullNoise, NullNoise)
#include "wide_opnoise_impl_deriv_xmacro.h"
