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
    // getattribute() has four "flavors":
    //   * getattribute (attribute_name, value)
    //   * getattribute (attribute_name, index, value)
    //   * getattribute (object, attribute_name, value)
    //   * getattribute (object, attribute_name, index, value)

    DASSERT (nargs >= 3 && nargs <= 5);

    bool object_lookup = false;
    bool array_lookup  = false;

    // slot indices when (nargs==3)
    int result_slot = 0; // never changes
    int attrib_slot = 1;
    int object_slot = 0; // initially not used
    int index_slot  = 0; // initially not used
    int dest_slot   = 2;

    // figure out which "flavor" of getattribute() to use
    if (nargs == 5) {
        object_slot = 1;
        attrib_slot = 2;
        index_slot  = 3;
        dest_slot   = 4;
        array_lookup  = true;
        object_lookup = true;
    }
    else if (nargs == 4) {
        if (exec->sym (args[2]).typespec().is_int()) {
            attrib_slot = 1;
            index_slot  = 2;
            dest_slot   = 3;
            array_lookup = true;
        }
        else {
            object_slot = 1;
            attrib_slot = 2;
            dest_slot   = 3;
            object_lookup = true;
        }
    }
    Symbol &Result      (exec->sym (args[result_slot]));
    Symbol &ObjectName  (exec->sym (args[object_slot])); // might be aliased to Result
    Symbol &Index       (exec->sym (args[index_slot ])); // might be aliased to Result
    Symbol &Attribute   (exec->sym (args[attrib_slot]));
    Symbol &Destination (exec->sym (args[dest_slot  ]));

    DASSERT (!Result.typespec().is_closure()    && !ObjectName.typespec().is_closure() && 
             !Attribute.typespec().is_closure() && !Index.typespec().is_closure()      && 
             !Destination.typespec().is_closure());

    ShaderGlobals *globals = exec->context()->globals();

    // default to true -- we don't know what the renderer will
    // return
    exec->adjust_varying (Result,      true);
    exec->adjust_varying (Destination, true);

    TypeDesc attribute_type;
    VaryingRef<int>     result         ((int *)Result.data(),         Result.step()     );
    VaryingRef<ustring> object_name    ((ustring *)ObjectName.data(), ObjectName.step() ); // might be aliased to Result
    VaryingRef<ustring> attribute_name ((ustring *)Attribute.data(),  Attribute.step()  );
    VaryingRef<void *>  destination    ((void *)Destination.data(),   Destination.step());
    VaryingRef<int>     index          ((int *)Index.data(),          Index.step()      ); // might be aliased to Result

    attribute_type = Destination.typespec().simpletype();

    // always fully varying case
    for (int i = beginpoint;  i < endpoint;  ++i) {
        if (runflags[i]) {
            //void *d = &destination[i];
            result[i] = array_lookup ? 
               exec->get_renderer_array_attribute(globals->renderstate[i], 
                                                  object_lookup ? object_name[i] : ustring(),
                                                  attribute_type, attribute_name[i],
                                                  index[i], &destination[i]) :
               exec->get_renderer_attribute(globals->renderstate[i], 
                                            object_lookup ? object_name[i] : ustring(),
                                            attribute_type, attribute_name[i],
                                            &destination[i]);
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
