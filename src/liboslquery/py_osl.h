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

#include <memory>

// Avoid a compiler warning from a duplication in tiffconf.h/pyconfig.h
#undef SIZEOF_LONG

#include <OSL/oslquery.h>

#if OSL_USING_IMATH >= 3
#    include <Imath/half.h>
#else
#    include <OpenEXR/half.h>
#endif

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;


#if PY_MAJOR_VERSION == 2
// Preferred Python string caster for Python2 is py::bytes, so it's a byte
// string (not unicode).
#    define PY_STR py::bytes
#else
// Python3 is always unicode, so return a true str
#    define PY_STR py::str
#endif


namespace PyOSL {

using namespace OSL;

// clang-format off

void declare_oslquery (py::module& m);


// bool PyProgressCallback(void*, float);
// object C_array_to_Python_array (const char *data, TypeDesc type, size_t size);
const char * python_array_code (TypeDesc format);
TypeDesc typedesc_from_python_array_code (char code);


inline std::string
object_classname(const py::object& obj)
{
    return obj.attr("__class__").attr("__name__").cast<py::str>();
}



template<typename T> struct PyTypeForCType { };
template<> struct PyTypeForCType<int> { typedef py::int_ type; };
template<> struct PyTypeForCType<unsigned int> { typedef py::int_ type; };
template<> struct PyTypeForCType<short> { typedef py::int_ type; };
template<> struct PyTypeForCType<unsigned short> { typedef py::int_ type; };
template<> struct PyTypeForCType<int64_t> { typedef py::int_ type; };
template<> struct PyTypeForCType<float> { typedef py::float_ type; };
template<> struct PyTypeForCType<half> { typedef py::float_ type; };
template<> struct PyTypeForCType<double> { typedef py::float_ type; };
template<> struct PyTypeForCType<const char*> { typedef PY_STR type; };
template<> struct PyTypeForCType<std::string> { typedef PY_STR type; };
template<> struct PyTypeForCType<ustring> { typedef PY_STR type; };

// clang-format on



template<typename T>
inline py::tuple
C_to_tuple(cspan<T> vals)
{
    size_t size = vals.size();
    py::tuple result(size);
    for (size_t i = 0; i < size; ++i)
        result[i] = typename PyTypeForCType<T>::type(vals[i]);
    return result;
}


template<typename T>
inline py::tuple
C_to_tuple(const T* vals, size_t size)
{
    py::tuple result(size);
    for (size_t i = 0; i < size; ++i)
        result[i] = typename PyTypeForCType<T>::type(vals[i]);
    return result;
}


// Special case for TypeDesc
template<>
inline py::tuple
C_to_tuple<TypeDesc>(cspan<TypeDesc> vals)
{
    size_t size = vals.size();
    py::tuple result(size);
    for (size_t i = 0; i < size; ++i)
        result[i] = py::cast(vals[i]);
    return result;
}

// Special case for ustring
template<>
inline py::tuple
C_to_tuple<ustring>(cspan<ustring> vals)
{
    size_t size = vals.size();
    py::tuple result(size);
    for (size_t i = 0; i < size; ++i)
        result[i] = PY_STR(vals[i].string());
    return result;
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
        return PY_STR(vals[0].string());
    else
        return C_to_tuple(vals);
}


}  // namespace PyOSL
