// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>

#include <OSL/device_string.h>  // for StringParam

// All the the state free functions in rs_simplerend.cpp will need to do their job
// NOTE:  Additional data is here that will be used by rs_simplerend.cpp in future PR's
//        procedurally generating ShaderGlobals.
struct RenderState {
    void *journal_buffer; 
    
};



