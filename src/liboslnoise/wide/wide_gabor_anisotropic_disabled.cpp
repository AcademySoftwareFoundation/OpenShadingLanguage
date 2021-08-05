// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// To improve parallel compile times, split explicit template instantiation
// based on an anisotropic and filter policies into different cpp files
#define __OSL_XMACRO_ARGS (1 /*anisotropic*/, DisabledFilterPolicy)
#include "wide_gabor_xmacro.h"
