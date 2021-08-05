// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#ifndef __OSL_USE_REFERENCE_INT_HASH
#    define __OSL_USE_REFERENCE_INT_HASH 0
#endif
#if __OSL_USE_REFERENCE_INT_HASH
// incorrect results when vectorizing with reference hash
#    undef OSL_OPENMP_SIMD
#endif

#define __OSL_XMACRO_ARGS (pcellnoise, PeriodicCellNoise, PeriodicCellNoise)
#include "wide_opnoise_periodic_impl_xmacro.h"
