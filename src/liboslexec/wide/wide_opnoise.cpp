// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/oslconfig.h>

#include <OSL/batched_shaderglobals.h>

#include "oslexec_pvt.h"

#include "define_opname_macros.h"

using namespace OSL;

OSL_NAMESPACE_ENTER
namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)

OSL_BATCHOP void
__OSL_OP(osl_init_noise_options)(void* bsg_, void* opt)
{
    new (opt) NoiseParams;
}



OSL_BATCHOP void
__OSL_MASKED_OP(count_noise)(void* bsg_, unsigned int mask_value)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    Mask mask(mask_value);
    bsg->uniform.context->shadingsys().count_noise(mask.count());
}


}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT

#include "undef_opname_macros.h"
