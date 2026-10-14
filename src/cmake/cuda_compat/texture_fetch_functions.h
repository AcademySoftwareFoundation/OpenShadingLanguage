// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause

// CUDA 13 removed this header, but older Clang CUDA wrappers still include it.
// OSL does not use the legacy texture-reference API, so an empty stub is
// sufficient for CUDA bitcode generation.
#pragma once
