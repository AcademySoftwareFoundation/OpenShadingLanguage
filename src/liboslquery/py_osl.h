// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

// Python.h uses the 'register' keyword, don't warn about it being
// deprecated in C++17.
#if (__cplusplus >= 201703L && defined(__GNUC__))
#    pragma GCC diagnostic ignored "-Wregister"
#endif

// clang-format off
// Must include Python.h first to avoid certain warnings
#ifdef _POSIX_C_SOURCE
#  error "You must include Python.h (and therefore py_osl.h) BEFORE anything that defines _POSIX_C_SOURCE"
#endif
#include <Python.h>
// clang-format on

// Avoid a compiler warning from a duplication in tiffconf.h/pyconfig.h
#undef SIZEOF_LONG

#include <OSL/oslquery.h>

#include <OSL/oslconfig.h>

#include <Imath/half.h>

#include "py_backend.h"


namespace PyOSL {

using namespace OSL;

// clang-format off

void declare_oslquery (py_module& m);


template<typename T> struct PyTypeForCType { };
template<> struct PyTypeForCType<int> { typedef py::int_ type; };
template<> struct PyTypeForCType<unsigned int> { typedef py::int_ type; };
template<> struct PyTypeForCType<short> { typedef py::int_ type; };
template<> struct PyTypeForCType<unsigned short> { typedef py::int_ type; };
template<> struct PyTypeForCType<int64_t> { typedef py::int_ type; };
template<> struct PyTypeForCType<float> { typedef py::float_ type; };
template<> struct PyTypeForCType<half> { typedef py::float_ type; };
template<> struct PyTypeForCType<double> { typedef py::float_ type; };
template<> struct PyTypeForCType<const char*> { typedef py::str type; };
template<> struct PyTypeForCType<std::string> { typedef py::str type; };
template<> struct PyTypeForCType<ustring> { typedef py::str type; };

// clang-format on



template<typename T>
inline py::tuple
C_to_tuple(cspan<T> vals)
{
    return osl_py::make_tuple(vals.size(), [&](size_t i) {
        return typename PyTypeForCType<T>::type(vals[i]);
    });
}


// Special case for ustring
template<>
inline py::tuple
C_to_tuple<ustring>(cspan<ustring> vals)
{
    return osl_py::make_tuple(vals.size(), [&](size_t i) {
        return osl_py::str(vals[i].string());
    });
}



// Convert an array of T values (described by type) into either a simple
// Python object (if it's an int, float, or string and a SCALAR) or a
// Python tuple.
template<typename T>
inline py::object
C_to_val_or_tuple(cspan<T> vals, TypeDesc type)
{
    if (vals.size() == 1 && !type.arraylen)
        return typename PyTypeForCType<T>::type(vals[0]);
    else
        return C_to_tuple(vals);
}


template<>
inline py::object
C_to_val_or_tuple(cspan<ustring> vals, TypeDesc type)
{
    size_t n = type.numelements() * type.aggregate * vals.size();
    if (n == 1 && !type.arraylen)
        return osl_py::str(vals[0].string());
    else
        return C_to_tuple(vals);
}


}  // namespace PyOSL
