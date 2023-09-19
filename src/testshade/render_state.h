// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/hashes.h>
#include <OSL/oslconfig.h>

// All the the state free functions in rs_simplerend.cpp will need to do their job
// NOTE:  Additional data is here that will be used by rs_simplerend.cpp in future PR's
//        procedurally generating ShaderGlobals.
struct RenderState {
    int xres;
    int yres;
    OSL::Matrix44 world_to_camera;
    OSL::ustringhash projection;
    float pixelaspect;
    float screen_window[4];
    float shutter[2];
    float fov;
    float hither;
    float yon;
    void* journal_buffer;
};


// Create constexpr hashes for all strings used by the free function renderer services.
// NOTE:  Actually ustring's should also be instantiated in host code someplace as well
// to allow the reverse mapping of hash->string to work when processing messages
namespace RS {
namespace {
namespace Hashes {
#define RS_STRDECL(str, var_name) \
    constexpr OSL::ustringhash var_name(OSL::strhash(str));
#include "rs_strdecls.h"
#undef RS_STRDECL
};  //namespace Hashes
}  // unnamed namespace
};  //namespace RS
