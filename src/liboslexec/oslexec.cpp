// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <vector>
#include <string>
#include <cstdio>

#include <OpenImageIO/strutil.h>
#include <OpenImageIO/thread.h>

#include "oslexec_pvt.h"
#include "osoreader.h"



OSL_NAMESPACE_ENTER

namespace pvt {   // OSL::pvt


string_view
shadertypename (ShaderType s)
{
    switch (s) {
    case ShaderType::Generic :      return ("shader");
    case ShaderType::Surface :      return ("surface");
    case ShaderType::Displacement : return ("displacement");
    case ShaderType::Volume :       return ("volume");
    case ShaderType::Light :        return ("light");
    default:
        OSL_DASSERT (0 && "Invalid shader type");
        return "unknown";
    }
}



ShaderType
shadertype_from_name (string_view name)
{
    if (name == "shader" || name == "generic")
        return ShaderType::Generic;
    if (name == "surface")
        return ShaderType::Surface;
    if (name == "displacement")
        return ShaderType::Displacement;
    if (name == "volume")
        return ShaderType::Volume;
    if (name == "light")
        return ShaderType::Light;
    return ShaderType::Unknown;
}


}; // namespace pvt
OSL_NAMESPACE_EXIT
