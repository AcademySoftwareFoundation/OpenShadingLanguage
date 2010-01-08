/*
Copyright (c) 2010 Sony Pictures Imageworks, et al.
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



// void setmessage (string name, ANY value)
DECLOP (OP_setmessage)
{
    ASSERT (nargs == 2);
    Symbol &Name (exec->sym (args[0]));
    Symbol &Val (exec->sym (args[1]));
    ASSERT (Name.typespec().is_string() && !Val.typespec().is_closure());

    VaryingRef<ustring> name ((ustring *)Name.data(), Name.step());
    ParamValueList &messages (exec->context()->messages());

    bool varying = (Name.is_varying() || Val.is_varying() ||
                    ! exec->all_points_on());
    TypeDesc type = Val.typespec().simpletype();
    size_t datasize = type.size();

    ustring lastname;       // Last message name that we matched
    ParamValue *p = NULL;   // Pointer to the PV for the message

    for (int i = beginpoint;  i < endpoint;  ++i) {
        if (runflags[i]) {
            if (i == beginpoint || name[i] != lastname) {
                // Different message than last time -- search anew
                p = NULL;
                for (size_t m = 0;  m < messages.size() && !p;  ++m)
                    if (messages[m].name() == name[i] &&
                          messages[m].type() == Val.typespec().simpletype())
                        p = &messages[m];
                // If the message doesn't already exist, create it
                if (! p) {
                    p = & messages.grow ();
                    p->init (name[i], Val.typespec().simpletype(),
                             varying ? exec->npoints() : 1, NULL);
                }
                lastname = name[i];
            }

            // Copy the data
            DASSERT (p != NULL);
            memcpy ((char *)p->data() + varying*datasize*i,
                    (char *)Val.data() + Val.step()*i, datasize);
        }
        if (! varying)
            break;      // Non-uniform case can take early out
    }

    // FIXME -- this scheme only stores the values, and DOES NOT
    // preserve the derivatives of the message!  We should come back and
    // fix this later.
}



// int getmessage (string name, ANY value)
DECLOP (OP_getmessage)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Name (exec->sym (args[1]));
    Symbol &Val (exec->sym (args[2]));
    ASSERT (Result.typespec().is_int() && 
            Name.typespec().is_string() && !Val.typespec().is_closure());

    bool varying = (Name.is_varying());
    exec->adjust_varying (Result, varying);
    exec->adjust_varying (Val, varying);

    VaryingRef<int> result ((int *)Result.data(), Result.step());
    VaryingRef<ustring> name ((ustring *)Name.data(), Name.step());
    ParamValueList &messages (exec->context()->messages());

    TypeDesc type = Val.typespec().simpletype();
    size_t datasize = type.size();

    ustring lastname;       // Last message name that we matched
    ParamValue *p = NULL;   // Pointer to the PV for the message

    for (int i = beginpoint;  i < endpoint;  ++i) {
        if (runflags[i]) {
            if (i == beginpoint || name[i] != lastname) {
                // Different message than last time -- search anew
                p = NULL;
                for (size_t m = 0;  m < messages.size() && !p;  ++m)
                    if (messages[m].name() == name[i] &&
                          messages[m].type() == Val.typespec().simpletype())
                        p = &messages[m];
                if (p && Val.is_uniform() && p->nvalues() > 1) {
                    // all the parameters to the function were uniform,
                    // but the message itself is varying, so adjust Val.
                    exec->adjust_varying (Val, true);
                    varying = true;
                }
                lastname = name[i];
            }

            if (p) {
                result[i] = 1;   // found
                memcpy ((char *)Val.data() + Val.step()*i,
                        (char *)p->data() + varying*datasize*i, datasize);
            } else {
                result[i] = 0;   // not found
            }
        }
        if (! varying)
            break;      // Non-uniform case can take early out
    }

    exec->zero_derivs (Val);
    // FIXME -- setmessage only stores message values, not derivs, so
    // getmessage only retrieves the values and has zero derivs.
    // We should come back and fix this later.
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
