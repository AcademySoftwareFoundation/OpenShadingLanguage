// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include "rtweekend.h"

#include <OSL/genclosure.h>
#include <OSL/oslclosure.h>
#include <OSL/oslexec.h>

enum ClosureIDs : int {
    DIFFUSE_ID = 1,
    RTIOW_METAL_ID,
    RTIOW_DIELECTRIC_ID,
};

// OSL copies closure arguments into these structs, so registered parameters
// must match the shader declarations in order and type.
struct DiffuseParams {
    OSL::Vec3 N;
};

struct RtiowMetalParams {
    OSL::Vec3 N;
    float fuzz;
};

struct RtiowDielectricParams {
    OSL::Vec3 N;
    float eta;
};

void
register_closures(OSL::ShadingSystem& shadingsys);

// Samples one lobe of the shader's closure tree, chosen in proportion to its
// weight. `I` points toward the surface. On success, `wi` is the sampled
// direction and `weight` its throughput.
[[nodiscard]] bool
sample_closure(const OSL::ClosureColor* Ci, const vec3& I, vec3& wi,
               color& weight);
