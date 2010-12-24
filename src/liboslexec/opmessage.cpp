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


#ifdef OIIO_NAMESPACE
using OIIO::ParamValue;
#endif

#define USTR(cstr) (*((ustring *)&cstr))

extern "C" void
osl_setmessage (ShaderGlobals *sg, const char *name_, long long type_, void *val)
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
osl_getmessage (ShaderGlobals *sg, const char *name_, long long type_, void *val)
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
