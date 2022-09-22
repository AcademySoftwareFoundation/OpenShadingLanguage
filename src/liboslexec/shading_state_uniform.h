// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "opcolor.h"

OSL_NAMESPACE_ENTER

namespace pvt{

// Intent is for a device to export or serialize the ShadingStateUniform from the 
// ShadingSystem into a device side buffer that can be passed into 
// kernel launches.

// Requires data to be self-contained and flat in memory to be device friendly.


struct ShadingStateUniform {
    ColorSystem m_colorsystem;        ///< Data for current colorspace
    ustring m_commonspace_synonym;    ///< Synonym for "common" space
    bool m_unknown_coordsys_error;    ///< Error to use unknown xform name?
};

} // namespace pvt

OSL_NAMESPACE_EXIT