// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once
#include <OSL/shaderglobals.h>
#include "opcolor.h"

OSL_NAMESPACE_ENTER

namespace pvt {

// Intent is for a device to export or serialize the ShadingStateUniform from the
// ShadingSystem into a device side buffer that can be passed into
// kernel launches.

// Requires data to be self-contained and flat in memory to be device friendly.


struct ShadingStateUniform {
    ColorSystem m_colorsystem;      ///< Data for current colorspace
    ustring m_commonspace_synonym;  ///< Synonym for "common" space
    bool m_unknown_coordsys_error;  ///< Error to use unknown xform name?
    int m_max_warnings_per_thread;  ///< How many warnings to display per thread before giving up?
};

inline bool
get_unknown_coordsys_error(const OpaqueExecContextPtr oec)
{
    auto ec                  = pvt::get_ec(oec);
    ShadingStateUniform* ssu = (ShadingStateUniform*)(ec->shadingStateUniform);
    return ssu->m_unknown_coordsys_error;
}

inline ustring
get_commonspace_synonym(const OpaqueExecContextPtr oec)
{
    auto ec                  = pvt::get_ec(oec);
    ShadingStateUniform* ssu = (ShadingStateUniform*)(ec->shadingStateUniform);
    return ssu->m_commonspace_synonym;
}

}  // namespace pvt

OSL_NAMESPACE_EXIT