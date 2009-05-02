/*****************************************************************************
 *
 *             Copyright (c) 2009 Sony Pictures Imageworks, Inc.
 *                            All rights reserved.
 *
 *  This material contains the confidential and proprietary information
 *  of Sony Pictures Imageworks, Inc. and may not be disclosed, copied or
 *  duplicated in any form, electronic or hardcopy, in whole or in part,
 *  without the express prior written consent of Sony Pictures Imageworks,
 *  Inc. This copyright notice does not imply publication.
 *
 *****************************************************************************/


#include <vector>
#include <string>
#include <cstdio>

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"

#include "oslexec_pvt.h"
#include "osoreader.h"




namespace OSL {

namespace pvt {   // OSL::pvt


const char *
shadertypename (ShaderType s)
{
    switch (s) {
    case ShadTypeGeneric:      return ("shader");
    case ShadTypeSurface:      return ("surface");
    case ShadTypeDisplacement: return ("displacement");
    case ShadTypeVolume:       return ("volume");
    case ShadTypeLight:        return ("light");
    default:
        ASSERT (0 && "Invalid shader type");
    }
}



ShaderType
shadertype_from_name (const char *name)
{
    if (! strcmp (name, "shader") || ! strcmp (name, "generic"))
        return ShadTypeGeneric;
    if (! strcmp (name, "surface"))
        return ShadTypeSurface;
    if (! strcmp (name, "displacement"))
        return ShadTypeDisplacement;
    if (! strcmp (name, "volume"))
        return ShadTypeVolume;
    if (! strcmp (name, "light"))
        return ShadTypeLight;
    return ShadTypeUnknown;
}


}; // namespace pvt
}; // namespace OSL
