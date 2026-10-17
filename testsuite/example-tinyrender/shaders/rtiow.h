// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

// Closures
closure color rtiow_metal(normal N, float fuzz)[[int builtin = 1]];
closure color rtiow_dielectric(normal N, float eta)[[int builtin = 1]];
