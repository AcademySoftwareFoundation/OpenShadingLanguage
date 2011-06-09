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

#include "oslops.h"
#include "oslexec_pvt.h"

#define USTR(cstr) (*((ustring *)&cstr))
#define TYPEDESC(x) (*(TypeDesc *)&x)

OSL_SHADEOP int
osl_pointcloud_search (ShaderGlobals *sg, const char *_filename, void *_center, float radius,
                       int max_points, void *out_indices, void *out_distances, int derivs_offset)
{
    const ustring &filename (USTR(_filename));
    Vec3 *center = (Vec3 *)_center;

    return sg->context->renderer()->pointcloud_search (filename, *center, radius, max_points, (size_t *)out_indices,
                                                       (float *)out_distances, derivs_offset);
}

OSL_SHADEOP int
osl_pointcloud_get (ShaderGlobals *sg, const char *_filename, void *indices, int count,
                    const char *_attr_name, long long _attr_type, void *out_data)
{
    const ustring &filename  (USTR(_filename));
    const ustring &attr_name (USTR(_attr_name));
    TypeDesc      &attr_type (TYPEDESC(_attr_type));

    return sg->context->renderer()->pointcloud_get (filename, (size_t *)indices, count, attr_name, attr_type, out_data);
}
