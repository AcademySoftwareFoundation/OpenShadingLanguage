// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// Preload these headers before compiling OSL device code. CUDA's __noinline__
// macro conflicts with attribute names in libstdc++; hide it during the includes
// and restore it for device code. The headers' include guards prevent reprocessing.
// Related upstream fix: https://github.com/llvm/llvm-project/pull/66138
#pragma push_macro("__noinline__")
#undef __noinline__
#include <memory>
#include <string>
#pragma pop_macro("__noinline__")
