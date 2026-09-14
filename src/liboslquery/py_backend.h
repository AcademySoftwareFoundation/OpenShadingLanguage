// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// Compatibility shim that lets the OSLQuery Python bindings be compiled
// against either pybind11 or nanobind from a single set of sources. Which
// one you get is selected by the build (OSL_PYTHON_BINDINGS_BACKEND), which
// defines OSL_PY_BACKEND_NANOBIND for the nanobind variant.
//
// Everything the two frameworks spell differently is absorbed here, so that
// py_osl.h and py_osl.cpp stay backend-neutral. Conditional compilation
// elsewhere should be a last resort -- see MIGRATION_STATUS.md.
//
// Do not include this directly; include py_osl.h, which arranges for
// Python.h to come first.

#pragma once

#include <string>
#include <utility>

#if defined(OSL_PY_BACKEND_NANOBIND)

#    include <nanobind/make_iterator.h>
#    include <nanobind/nanobind.h>
// nanobind's STL type casters are opt-in, one header per type -- unlike
// pybind11, where <pybind11/stl.h> brings them all in at once. Omitting one
// is not a compile error at the point of use; it fails at runtime when the
// conversion is attempted. stl/vector.h is what turns the
// std::vector<OSLQuery::Parameter> returned by `.parameters` and `.metadata`
// into a Python list, and stl/string.h covers std::string arguments and
// returns (e.g. geterror()).
#    include <nanobind/stl/string.h>
#    include <nanobind/stl/vector.h>

namespace py    = nanobind;
using py_module = nanobind::module_;
using namespace py::literals;

#    define OSL_PY_RW      def_rw
#    define OSL_PY_PROP_RO def_prop_ro
#    define OSL_PY_PROP_RW def_prop_rw

#else  // pybind11

#    include <pybind11/pybind11.h>
#    include <pybind11/stl.h>

namespace py    = pybind11;
using py_module = pybind11::module;
using namespace py::literals;

#    define OSL_PY_RW      def_readwrite
#    define OSL_PY_PROP_RO def_property_readonly
#    define OSL_PY_PROP_RW def_property

#endif


namespace PyOSL {
namespace osl_py {

// Make a Python str. Python 3 strings are always unicode, so py::str is the
// real thing in both backends.
//
// Use these rather than py::str directly: pybind11's str has a std::string
// constructor and nanobind's does not, and the obvious workaround of passing
// ustring::c_str() is a trap -- that returns nullptr for an empty or
// default-constructed ustring, which crashes inside CPython. ustring::string()
// is null-safe, and so is the const char* overload below.
inline py::str
str(const std::string& s)
{
#if defined(OSL_PY_BACKEND_NANOBIND)
    return py::str(s.c_str(), s.size());
#else
    return py::str(s);
#endif
}


inline py::str
str(const char* s)
{
    return py::str(s ? s : "");
}


// Build a tuple of `size` elements, where element i is fill(i).
//
// pybind11 lets you size a tuple up front and then assign into it; nanobind's
// tuple is immutable from C++, so there we accumulate into a list and convert.
template<typename F>
inline py::tuple
make_tuple(size_t size, F&& fill)
{
#if defined(OSL_PY_BACKEND_NANOBIND)
    py::list list;
    for (size_t i = 0; i < size; ++i)
        list.append(fill(i));
    return py::steal<py::tuple>(PyList_AsTuple(list.ptr()));
#else
    py::tuple result(size);
    for (size_t i = 0; i < size; ++i)
        result[i] = fill(i);
    return result;
#endif
}


// Make a Python iterator over [first, last).
//
// nanobind additionally wants the type object that the iterator type should
// be scoped to, plus a name for it. Scope must be a *bound* class: passing
// something never registered (std::vector<Parameter>, say) yields a null
// handle. pybind11 derives all of that itself and ignores Scope.
//
// The std::move calls are load-bearing, not an optimization. Both frameworks
// offer two make_iterator overloads: a (first, last) pair, and a whole
// container spelled `make_iterator(Type& value, Extra&&... extra)`. Hand the
// latter an lvalue iterator and it becomes viable too, swallowing `last` into
// `Extra`. Whether that is ambiguous depends on how the pair overload takes
// its arguments: by value (every pybind11 except 2.10, and nanobind) is fine,
// but pybind11 2.10 alone used forwarding references, which deduce to `It&`
// for an lvalue and so tie exactly with `Type&`. Callers passing .begin() and
// .end() directly never see this, because a prvalue cannot bind `Type&`;
// naming them as parameters here is what makes them lvalues, and moving
// restores the value category those overloads were written to expect.
template<typename Scope, typename It>
inline auto
make_iterator(It first, It last)
{
#if defined(OSL_PY_BACKEND_NANOBIND)
    return py::make_iterator(py::type<Scope>(), "Iterator", std::move(first),
                             std::move(last));
#else
    return py::make_iterator(std::move(first), std::move(last));
#endif
}


// nanobind's key_error takes a const char*, pybind11's takes a std::string.
[[noreturn]] inline void
throw_key_error(const std::string& msg)
{
#if defined(OSL_PY_BACKEND_NANOBIND)
    throw py::key_error(msg.c_str());
#else
    throw py::key_error(msg);
#endif
}

}  // namespace osl_py
}  // namespace PyOSL
