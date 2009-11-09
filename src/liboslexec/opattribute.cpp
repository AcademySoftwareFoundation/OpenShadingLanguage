/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif
namespace OSL {
namespace pvt {

DECLOP (OP_getattribute)
{
    DASSERT (nargs == 3 || nargs == 4);
    const bool object_specified = (nargs == 4);
    Symbol &Result      (exec->sym (args[0]));
    Symbol &ObjectName  (exec->sym (args[1]));
    Symbol &Attribute   (exec->sym (args[1+object_specified]));
    Symbol &Destination (exec->sym (args[2+object_specified]));
    DASSERT (Attribute.typespec().is_string() && ObjectName.typespec().is_string());
    DASSERT (!Result.typespec().is_closure() && !ObjectName.typespec().is_closure() && !Attribute.typespec().is_closure() && !Destination.typespec().is_closure());

    ShaderGlobals *globals = exec->context()->globals();

    // default to true -- we don't know what the renderer will
    // return
    exec->adjust_varying (Result,      true);
    exec->adjust_varying (Destination, true);

    TypeDesc attribute_type;
    VaryingRef<int>     result         ((int *)Result.data(),         Result.step());
    VaryingRef<ustring> object_name    ((ustring *)ObjectName.data(), ObjectName.step());
    VaryingRef<ustring> attribute_name ((ustring *)Attribute.data(),  Attribute.step());
    VaryingRef<void *>  destination    ((void *)Destination.data(),   Destination.step());

    attribute_type = Destination.typespec().simpletype();

    // FIXME:  what about arrays?
   
    if (result.is_uniform()) {
        // Uniform case
        void *d = &destination[0];
        *result = exec->get_renderer_attribute( *globals->renderstate, 
                                                object_specified ? object_name[0] : ustring(),
                                                *attribute_name, attribute_type, d); //destination;
    } else {
        // Fully varying case
        for (int i = beginpoint;  i < endpoint;  ++i) {
            if (runflags[i]) {
                void *d = &destination[i];
                result[i] = exec->get_renderer_attribute(globals->renderstate[i], 
                                                         object_specified ? object_name[i] : ustring(),
                                                         attribute_name[i], attribute_type, d); //destination[i)
            }
        }
    }
    // FIXME: Disable derivatives (for now)
    if (Destination.has_derivs())
        exec->zero_derivs (Destination);
}


}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
