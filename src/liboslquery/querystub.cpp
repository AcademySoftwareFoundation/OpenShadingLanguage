// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

/// The purpose of this file is to provide a stub for the OSLQuery::init()
/// method for an app that do not have a full ShadingSystem (oslinfo for
/// instance)
/// This enables such apps to link on the Windows platform while not
/// introducing dependencies to liboslexec (where the actual method
/// is implemented)

#include <OSL/oslquery.h>
using namespace OSL;



bool
OSLQuery::init (const ShaderGroup* /*group*/, int /*layernum*/)
{
    return false;
}

