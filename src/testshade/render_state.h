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
    int xres;
    int yres;
    OSL::Matrix44 world_to_camera;
    OSL::StringParam projection;
    float pixelaspect;
    float screen_window[4];
    float shutter[2];
    float fov;
    float hither;
    float yon;
    void* journal_buffer;
};
