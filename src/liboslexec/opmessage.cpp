// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include "oslexec_pvt.h"
#include <OSL/shaderglobals.h>


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


OSL_NAMESPACE_ENTER
namespace pvt {


OSL_SHADEOP void
osl_setmessage (ShaderGlobals *sg, const char *name_, long long type_, void *val, int layeridx, const char* sourcefile_, int sourceline)
{
    const ustring &name (USTR(name_));
    const ustring &sourcefile (USTR(sourcefile_));
    // recreate TypeDesc -- we just crammed it into an int!
    TypeDesc type = TYPEDESC(type_);
    bool is_closure = (type.basetype == TypeDesc::UNKNOWN); // secret code for closure
    if (is_closure)
        type.basetype = TypeDesc::PTR;  // for closures, we store a pointer

    MessageList &messages (sg->context->messages());
    const Message* m = messages.find(name);
    if (m != NULL) {
        if (m->name == name) {
            // message already exists?
            if (m->has_data())
                sg->context->errorf(
                   "message \"%s\" already exists (created here: %s:%d)"
                   " cannot set again from %s:%d",
                   name, m->sourcefile, m->sourceline, sourcefile, sourceline);
            else // NOTE: this cannot be triggered when strict_messages=false because we won't record "failed" getmessage calls
               sg->context->errorf(
                   "message \"%s\" was queried before being set (queried here: %s:%d)"
                   " setting it now (%s:%d) would lead to inconsistent results",
                   name, m->sourcefile, m->sourceline, sourcefile, sourceline);
            return;
        }
    }
    // The message didn't exist - create it
    messages.add(name, val, type, layeridx, sourcefile, sourceline);
}



OSL_SHADEOP int
osl_getmessage (ShaderGlobals *sg, const char *source_, const char *name_,
                long long type_, void *val, int derivs,
                int layeridx, const char* sourcefile_, int sourceline)
{
    const ustring &source (USTR(source_));
    const ustring &name (USTR(name_));
    const ustring &sourcefile (USTR(sourcefile_));

    // recreate TypeDesc -- we just crammed it into an int!
    TypeDesc type = TYPEDESC(type_);
    bool is_closure = (type.basetype == TypeDesc::UNKNOWN); // secret code for closure
    if (is_closure)
        type.basetype = TypeDesc::PTR;  // for closures, we store a pointer

    static ustring ktrace ("trace");
    if (source == ktrace) {
        // Source types where we need to ask the renderer
        return sg->renderer->getmessage (sg, source, name, type, val, derivs);
    }

    MessageList &messages (sg->context->messages());
    const Message* m = messages.find(name);
    if (m != NULL) {
        if (m->name == name) {
            if (m->type != type) {
                // found message, but types don't match
                sg->context->errorf(
                    "type mismatch for message \"%s\" (%s as %s here: %s:%d)"
                    " cannot fetch as %s from %s:%d",
                    name, m->has_data() ? "created" : "queried",
                    m->type == TypeDesc::PTR ? "closure color" : m->type.c_str(),
                    m->sourcefile, m->sourceline,
                    is_closure ? "closure color" : type.c_str(),
                    sourcefile, sourceline);
                return 0;
            }
            if (!m->has_data()) {
                // getmessage ran before and found nothing - just return 0
                return 0;
            }
            if (m->layeridx > layeridx) {
                // found message, but was set by a layer deeper than the one querying the message
                sg->context->errorf(
                    "message \"%s\" was set by layer #%d (%s:%d)"
                    " but is being queried by layer #%d (%s:%d)"
                    " - messages may only be transfered from nodes "
                    "that appear earlier in the shading network",
                    name, m->layeridx, m->sourcefile, m->sourceline,
                    layeridx, sourcefile, sourceline);
                return 0;
            }
            // Message found!
            size_t size = type.size();
            memcpy (val, m->data, size);
            if (derivs) // TODO: move this to llvm code gen?
                memset (((char *)val)+size, 0, 2*size);
            return 1;
        }
    }
    // Message not found -- we must record this event in case another layer tries to set the message again later on
    if (sg->context->shadingsys().strict_messages())
        messages.add(name, NULL, type, layeridx, sourcefile, sourceline);
    return 0;
}


} // namespace pvt
OSL_NAMESPACE_EXIT
