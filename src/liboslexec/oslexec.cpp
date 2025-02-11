// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <cstdio>
#include <string>
#include <vector>

#include <OpenImageIO/strutil.h>
#include <OpenImageIO/thread.h>

#include "oslexec_pvt.h"
#include "osoreader.h"



OSL_NAMESPACE_ENTER

namespace pvt {  // OSL::pvt


string_view
shadertypename(ShaderType s)
{
    switch (s) {
    case ShaderType::Generic: return ("shader");
    case ShaderType::Surface: return ("surface");
    case ShaderType::Displacement: return ("displacement");
    case ShaderType::Volume: return ("volume");
    case ShaderType::Light: return ("light");
    default: OSL_DASSERT(0 && "Invalid shader type"); return "unknown";
    }
}



ShaderType
shadertype_from_name(string_view name)
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



std::string
optix_cache_wrap(string_view ptx, size_t groupdata_size)
{
    // Cache string is the ptx file with groupdata size on top as a comment.
    // This way the cache string is a valid ptx program, which can be useful
    // for debugging.
    return fmtformat("// {}\n{}", groupdata_size, ptx);
}



void
optix_cache_unwrap(string_view cache_value, std::string& ptx,
                   size_t& groupdata_size)
{
    size_t groupdata_end_index = cache_value.find('\n');
    if (groupdata_end_index != std::string::npos) {
        constexpr int offset = 3;  // Account for the "// " prefix
        std::string groupdata_string
            = cache_value.substr(offset, groupdata_end_index - offset);
        groupdata_size = std::stoll(groupdata_string);

        ptx = cache_value.substr(groupdata_end_index + 1);
    }
}

};  // namespace pvt
OSL_NAMESPACE_EXIT
