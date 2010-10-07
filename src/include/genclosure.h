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

#ifndef GENCLOSURE_H
#define GENCLOSURE_H

#include <OpenImageIO/ustring.h>
#include "oslconfig.h"

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {


struct ClosureParam {
    TypeDesc    type;
    int         offset;
    const char *key;
};

#define reckless_offsetof(st, fld) (((char *)&(((st *)1)->fld)) - (char *)1)

#define CLOSURE_INT_PARAM(st, fld) \
    { TypeDesc::TypeInt, reckless_offsetof(st, fld), NULL }
#define CLOSURE_FLOAT_PARAM(st, fld) \
    { TypeDesc::TypeFloat, reckless_offsetof(st, fld), NULL }
#define CLOSURE_COLOR_PARAM(st, fld) \
    { TypeDesc::TypeColor, reckless_offsetof(st, fld), NULL }
#define CLOSURE_VECTOR_PARAM(st, fld) \
    { TypeDesc::TypeVector, reckless_offsetof(st, fld), NULL }
#define CLOSURE_STRING_PARAM(st, fld) \
    { TypeDesc::TypeString, reckless_offsetof(st, fld), NULL }

#define CLOSURE_INT_ARRAY_PARAM(st, fld, n) \
    { TypeDesc(TypeDesc::INT,   TypeDesc::SCALAR, TypeDesc::NOXFORM, n),reckless_offsetof(st, fld), NULL }
#define CLOSURE_VECTOR_ARRAY_PARAM(st,fld,n) \
    { TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3,   TypeDesc::VECTOR,  n),reckless_offsetof(st, fld), NULL }
#define CLOSURE_COLOR_ARRAY_PARAM(st,fld,n) \
    { TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3,   TypeDesc::COLOR,   n),reckless_offsetof(st, fld), NULL }
#define CLOSURE_FLOAT_ARRAY_PARAM(st,fld,n) \
    { TypeDesc(TypeDesc::FLOAT, TypeDesc::SCALAR, TypeDesc::NOXFORM, n),reckless_offsetof(st, fld), NULL }
#define CLOSURE_STRING_ARRAY_PARAM(st,fld,n) \
    { TypeDesc(TypeDesc::STRING, TypeDesc::SCALAR, TypeDesc::NOXFORM, n),reckless_offsetof(st, fld), NULL }

// NOTE: this keyword args have to be always at the end of the list
#define CLOSURE_INT_KEYPARAM(key) \
    { TypeDesc::TypeInt, 0, key }
#define CLOSURE_FLOAT_KEYPARAM(key) \
    { TypeDesc::TypeFloat, 0, key }
#define CLOSURE_COLOR_KEYPARAM(key) \
    { TypeDesc::TypeColor, 0, key }
#define CLOSURE_VECTOR_KEYPARAM(key) \
    { TypeDesc::TypeVector, 0, key }
#define CLOSURE_STRING_KEYPARAM(key) \
    { TypeDesc::TypeString, 0, key }

#define CLOSURE_FINISH_PARAM(st) { TypeDesc(), sizeof(st), NULL }

}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif


#endif /* OSLCLOSURE_H */
