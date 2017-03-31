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

#pragma once

#include <OpenImageIO/ustring.h>
#include <OSL/oslconfig.h>

OSL_NAMESPACE_ENTER


struct ClosureParam {
    TypeDesc    type;
    int         offset;
    const char *key;
    // This is only for sanity checks
    int         field_size;
};

#define reckless_offsetof(st, fld) (((char *)&(((st *)16)->fld)) - (char *)16)
#define fieldsize(st, fld) sizeof(((st *)0)->fld)

#define CLOSURE_INT_PARAM(st, fld) \
    { TypeDesc::TypeInt, (int)reckless_offsetof(st, fld), NULL, fieldsize(st, fld) }
#define CLOSURE_FLOAT_PARAM(st, fld) \
    { TypeDesc::TypeFloat, (int)reckless_offsetof(st, fld), NULL, fieldsize(st, fld) }
#define CLOSURE_COLOR_PARAM(st, fld) \
    { TypeDesc::TypeColor, (int)reckless_offsetof(st, fld), NULL, fieldsize(st, fld) }
#define CLOSURE_VECTOR_PARAM(st, fld) \
    { TypeDesc::TypeVector, (int)reckless_offsetof(st, fld), NULL, fieldsize(st, fld) }
#define CLOSURE_STRING_PARAM(st, fld) \
    { TypeDesc::TypeString, (int)reckless_offsetof(st, fld), NULL, fieldsize(st, fld) }
#define CLOSURE_CLOSURE_PARAM(st, fld) \
    { TypeDesc::PTR, (int)reckless_offsetof(st, fld), NULL, fieldsize(st, fld) }

#define CLOSURE_INT_ARRAY_PARAM(st, fld, n) \
    { TypeDesc(TypeDesc::INT,   TypeDesc::SCALAR, TypeDesc::NOXFORM, n),(int)reckless_offsetof(st, fld), NULL, fieldsize(st, fld) }
#define CLOSURE_VECTOR_ARRAY_PARAM(st,fld,n) \
    { TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3,   TypeDesc::VECTOR,  n),(int)reckless_offsetof(st, fld), NULL, fieldsize(st, fld) }
#define CLOSURE_COLOR_ARRAY_PARAM(st,fld,n) \
    { TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3,   TypeDesc::COLOR,   n),(int)reckless_offsetof(st, fld), NULL, fieldsize(st, fld) }
#define CLOSURE_FLOAT_ARRAY_PARAM(st,fld,n) \
    { TypeDesc(TypeDesc::FLOAT, TypeDesc::SCALAR, TypeDesc::NOXFORM, n),(int)reckless_offsetof(st, fld), NULL, fieldsize(st, fld) }
#define CLOSURE_STRING_ARRAY_PARAM(st,fld,n) \
    { TypeDesc(TypeDesc::STRING, TypeDesc::SCALAR, TypeDesc::NOXFORM, n),(int)reckless_offsetof(st, fld), NULL, fieldsize(st, fld) }

// NOTE: this keyword args have to be always at the end of the list
#define CLOSURE_INT_KEYPARAM(st, fld, key) \
    { TypeDesc::TypeInt, (int)reckless_offsetof(st, fld), key, fieldsize(st, fld) }
#define CLOSURE_FLOAT_KEYPARAM(st, fld, key) \
    { TypeDesc::TypeFloat, (int)reckless_offsetof(st, fld), key, fieldsize(st, fld) }
#define CLOSURE_COLOR_KEYPARAM(st, fld, key) \
    { TypeDesc::TypeColor, (int)reckless_offsetof(st, fld), key, fieldsize(st, fld) }
#define CLOSURE_VECTOR_KEYPARAM(st, fld, key) \
    { TypeDesc::TypeVector, (int)reckless_offsetof(st, fld), key, fieldsize(st, fld) }
#define CLOSURE_STRING_KEYPARAM(st, fld, key) \
    { TypeDesc::TypeString, (int)reckless_offsetof(st, fld), key, fieldsize(st, fld) }

#define CLOSURE_FINISH_PARAM(st) { TypeDesc(), sizeof(st), NULL, 0 }

OSL_NAMESPACE_EXIT
