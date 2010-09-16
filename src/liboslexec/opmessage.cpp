/*
Copyright (c) 2010 Sony Pictures Imageworks Inc., et al.
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


/////////////////////////////////////////////////////////////////////////
// Notes on how messages work:
//
// The messages are stored in a ParamValueList in the ShadingContext.
// For simple types, just slurp them up into the PVL.
//
// FIXME -- setmessage only stores message values, not derivs, so
// getmessage only retrieves the values and has zero derivs.
// We should come back and fix this later.
//
// FIXME -- I believe that if you try to set a message that is an array
// of closures, it will only store the first element.  Also something to
// come back to, not an emergency at the moment.
//


// void setmessage (string name, ANY value).
DECLOP (OP_setmessage)
{
    ASSERT (nargs == 2);
    Symbol &Name (exec->sym (args[0]));
    Symbol &Val (exec->sym (args[1]));
    ASSERT (Name.typespec().is_string());

    VaryingRef<ustring> name ((ustring *)Name.data(), Name.step());
    ParamValueList &messages (exec->context()->messages());

    // We are forcing messages to be varying here to avoid a bug
    // we still haven't identify. This was the old line:
    //
    // FIXME: locate the actual bug and restore the adaptive behaviour
    //
    // bool varying = (Name.is_varying() || Val.is_varying() ||
    //                 ! exec->all_points_on());
    bool varying = true;
    TypeDesc type = Val.typespec().simpletype();
    if (Val.typespec().is_closure ())
        type = TypeDesc::PTR;     // It is going to be a pointer
    size_t datasize = type.size();

    ustring lastname;       // Last message name that we matched
    ParamValue *p = NULL;   // Pointer to the PV for the message

    SHADE_LOOP_BEGIN
        if (i == exec->beginpoint() || name[i] != lastname) {
            // Different message than last time -- search anew
            p = NULL;
            for (size_t m = 0;  m < messages.size() && !p;  ++m)
                if (messages[m].name() == name[i] &&
                    messages[m].type() == type)
                    p = &messages[m];
            // If the message doesn't already exist, create it
            if (! p) {
                p = & messages.grow ();
                p->init (name[i], type,
                         varying ? exec->npoints() : 1, NULL);
            }
            lastname = name[i];
        }

        // Copy the data
        DASSERT (p != NULL);
        char *msgdata = (char *)p->data() + varying*datasize*i;
        memcpy (msgdata, Val.data(i), datasize);
        if (! varying)
            SHADE_LOOP_EXIT      // Non-uniform case can take early out
    SHADE_LOOP_END
}



// int getmessage (string name, ANY value)
DECLOP (OP_getmessage)
{
    ASSERT (nargs == 3);
    Symbol &Result (exec->sym (args[0]));
    Symbol &Name (exec->sym (args[1]));
    Symbol &Val (exec->sym (args[2]));
    ASSERT (Result.typespec().is_int() && Name.typespec().is_string());

    // Now we force varying to work around an unidentified bug.
    // This was the previous line:
    //
    // FIXME: locate the actual bug and restore the adaptive behaviour
    //
    // bool varying = (Name.is_varying());
    bool varying = true;
    exec->adjust_varying (Result, varying);
    exec->adjust_varying (Val, varying);
    // And we also removed this (now) unecessary or:
    // varying |= Result.is_varying();  // adjust in case we're in a conditional

    VaryingRef<int> result ((int *)Result.data(), Result.step());
    VaryingRef<ustring> name ((ustring *)Name.data(), Name.step());
    ParamValueList &messages (exec->context()->messages());

    TypeDesc type = Val.typespec().simpletype();
    if (Val.typespec().is_closure ())
        type = TypeDesc::PTR;
    size_t datasize = type.size();

    ustring lastname;       // Last message name that we matched
    ParamValue *p = NULL;   // Pointer to the PV for the message

    SHADE_LOOP_BEGIN
        if (i == exec->beginpoint() || name[i] != lastname) {
            // Different message than last time -- search anew
            p = NULL;
            for (size_t m = 0;  m < messages.size() && !p;  ++m)
                if (messages[m].name() == name[i] &&
                    messages[m].type() == type)
                    p = &messages[m];
            if (p && (! varying || Val.is_uniform()) && p->nvalues() > 1) {
                // all the parameters to the function were uniform,
                // but the message itself is varying, so adjust Val.
                exec->adjust_varying (Val, true);
                varying = true;
            }
            lastname = name[i];
        }

        if (p) {
            result[i] = 1;   // found
            char *msgdata = (char *)p->data() + varying*datasize*i;
            memcpy (Val.data(i), msgdata, datasize);
        } else {
            result[i] = 0;   // not found
        }
        if (! varying)
            SHADE_LOOP_EXIT      // Non-uniform case can take early out
    SHADE_LOOP_END

    if (Val.has_derivs ())
        exec->zero_derivs (Val);
}



}; // namespace pvt
}; // namespace OSL
#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif


#define USTR(cstr) (*((ustring *)&cstr))

extern "C" void
osl_setmessage (SingleShaderGlobal *sg, const char *name_, long long type_, void *val)
{
    const ustring &name (USTR(name_));
    // recreate TypeDesc -- we just crammed it into an int!
    TypeDesc type (*(TypeDesc *)&type_);
    bool is_closure = (type == TypeDesc::UNKNOWN); // secret code for closure
    if (is_closure)
        type = TypeDesc::PTR;  // for closures, we store a pointer

    ParamValueList &messages (sg->context->messages());
    ParamValue *p = NULL;
    for (size_t m = 0;  m < messages.size() && !p;  ++m)
        if (messages[m].name() == name && messages[m].type() == type)
            p = &messages[m];
    // If the message doesn't already exist, create it
    if (! p) {
        p = & messages.grow ();
        ASSERT (p == &(messages.back()));
        ASSERT (p == &(messages[messages.size()-1]));
        p->init (name, type, 1, NULL);
    }
    
    memcpy ((void *)p->data(), val, type.size());
}



extern "C" int
osl_getmessage (SingleShaderGlobal *sg, const char *name_, long long type_, void *val)
{
    const ustring &name (USTR(name_));
    // recreate TypeDesc -- we just crammed it into an int!
    TypeDesc type (*(TypeDesc *)&type_);
    bool is_closure = (type == TypeDesc::UNKNOWN); // secret code for closure
    if (is_closure)
        type = TypeDesc::PTR;  // for closures, we store a pointer

    ParamValueList &messages (sg->context->messages());
    ParamValue *p = NULL;
    for (size_t m = 0;  m < messages.size() && !p;  ++m)
        if (messages[m].name() == name && messages[m].type() == type)
            p = &messages[m];

    if (p) {
        // Message found
        memcpy (val, p->data(), type.size());
        return 1;
    }

    // Message not found
    return 0;
}
