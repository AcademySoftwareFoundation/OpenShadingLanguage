// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// clang-format off

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

#define CLOSURE_FINISH_PARAM(st) { TypeDesc(), sizeof(st), nullptr, alignof(st) }

OSL_NAMESPACE_EXIT
