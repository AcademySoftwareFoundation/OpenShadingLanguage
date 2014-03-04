/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <vector>
#include <string>
#include <cstdio>

#include <OpenImageIO/strutil.h>
#include <OpenImageIO/dassert.h>
#include <OpenImageIO/thread.h>

#include "oslexec_pvt.h"
#include "osoreader.h"



OSL_NAMESPACE_ENTER

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


const char *
shaderusename (ShaderUse s)
{
    switch (s) {
    case ShadUseSurface:      return ("surface");
    case ShadUseDisplacement: return ("displacement");
    default:
        ASSERT (0 && "Invalid shader use");
    }
}



ShaderUse
shaderuse_from_name (const char *name)
{
    if (! strcmp (name, "surface"))
        return ShadUseSurface;
    if (! strcmp (name, "displacement"))
        return ShadUseDisplacement;
    return ShadUseLast;
}


}; // namespace pvt
OSL_NAMESPACE_EXIT
