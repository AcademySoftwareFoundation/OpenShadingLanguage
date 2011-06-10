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

inline ustring USTR(const char *cstr) { return (*((const ustring *)&cstr)); }
inline TypeDesc TYPEDESC(long long x) { return (*(const TypeDesc *)&x); }

OSL_SHADEOP int
osl_pointcloud_search (ShaderGlobals *sg, const char *filename, void *center, float radius,
                       int max_points, void *out_indices, void *out_distances, int derivs_offset,
                       int nattrs, ...)
{
    size_t *indices = (size_t *)alloca (sizeof(size_t) * max_points);

    int count = sg->context->renderer()->pointcloud_search (USTR(filename), *((Vec3 *)center), radius, max_points,
                                                            indices, (float *)out_distances, derivs_offset);
    va_list args;
    va_start (args, nattrs);
    for (int i = 0; i < nattrs; i += 3)
    {  
        ustring  attr_name = USTR (va_arg (args, const char *));
        TypeDesc attr_type = TYPEDESC (va_arg (args, long long));
        void     *out_data = va_arg (args, void*);
        sg->context->renderer()->pointcloud_get (USTR(filename), indices, count, attr_name, attr_type, out_data);
    }
    va_end (args);

    if (out_indices)
        for(int i = 0; i < count; ++i)
            ((int *)out_indices)[i] = indices[i];
    return count;
}

OSL_SHADEOP int
osl_pointcloud_get (ShaderGlobals *sg, const char *filename, void *in_indices, int count,
                    const char *attr_name, long long attr_type, void *out_data)
{
    size_t *indices = (size_t *)alloca (sizeof(size_t) * count);

    for(int i = 0; i < count; ++i)
        indices[i] = ((int *)in_indices)[i];

    return sg->context->renderer()->pointcloud_get (USTR(filename), (size_t *)indices, count, USTR(attr_name),
                                                    TYPEDESC(attr_type), out_data);
}

