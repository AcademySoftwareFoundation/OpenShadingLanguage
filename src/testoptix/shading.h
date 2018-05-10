#pragma once

#include <OSL/dual_vec.h>
#include <OSL/oslexec.h>
#include <OSL/oslclosure.h>
#include <OSL/oslconfig.h>


OSL_NAMESPACE_ENTER

void register_closures(ShadingSystem* shadingsys);
void register_string_tags(ShadingSystem* shadingsys);

OSL_NAMESPACE_EXIT
