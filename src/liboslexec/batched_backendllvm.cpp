// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include "oslexec_pvt.h"
#include <type_traits>

using namespace OSL;
using namespace OSL::pvt;

OSL_NAMESPACE_ENTER

namespace Strings {

// TODO: What qualifies these to move to strdecls.h?
//       Being used in more than one .cpp?

// Shader global strings
static ustring backfacing("backfacing");
static ustring surfacearea("surfacearea");
static ustring object2common("object2common");
static ustring shader2common("shader2common");
static ustring flipHandedness("flipHandedness");
}  // namespace Strings

namespace pvt {

namespace  // Unnamed
{
// The order of names in this table MUST exactly match the
// BatchedShaderGlobals struct in batched_shaderglobals.h,
// as well as the llvm 'sg' type
// defined in BatchedBackendLLVM::llvm_type_sg().
static ustring fields[] = {
    // Uniform
    ustring("renderstate"),     //
    ustring("tracedata"),       //
    ustring("objdata"),         //
    ustring("shadingcontext"),  //
    ustring("renderer"),        //
    Strings::Ci,                //
    Strings::raytype,           //
    ustring("pad0"),            //
    ustring("pad1"),            //
    ustring("pad2"),            //
    // Varying
    Strings::P,               //
    ustring("dPdz"),          //
    Strings::I,               //
    Strings::N,               //
    Strings::Ng,              //
    Strings::u,               //
    Strings::v,               //
    Strings::dPdu,            //
    Strings::dPdv,            //
    Strings::time,            //
    Strings::dtime,           //
    Strings::dPdtime,         //
    Strings::Ps,              //
    Strings::object2common,   //
    Strings::shader2common,   //
    Strings::surfacearea,     //
    Strings::flipHandedness,  //
    Strings::backfacing
};

static bool field_is_uniform[] = {
    // Uniform
    true,  // renderstate
    true,  // tracedata
    true,  // objdata
    true,  // shadingcontext
    true,  // renderer
    true,  // Ci
    true,  // raytype
    true,  // pad0
    true,  // pad1
    true,  // pad2
    // Varying
    false,  // P
    false,  // dPdz
    false,  // I
    false,  // N
    false,  // Ng
    false,  // u
    false,  // v
    false,  // dPdu
    false,  // dPdv
    false,  // time
    false,  // dtime
    false,  // dPdtime
    false,  // Ps
    false,  // object2common
    false,  // shader2common
    false,  // surfacearea
    false,  // flipHandedness
    false,  // backfacing
};

}  // namespace

extern bool
is_shader_global_uniform_by_name(ustring name)
{
    for (int i = 0; i < int(std::extent<decltype(fields)>::value); ++i) {
        if (name == fields[i]) {
            return field_is_uniform[i];
        }
    }
    return false;
}

// Implementation for BatchedBackendLLVM will be added here in future PR.
// That implementation will make use of IsShaderGlobalUniformByName

};  // namespace pvt
OSL_NAMESPACE_EXIT
