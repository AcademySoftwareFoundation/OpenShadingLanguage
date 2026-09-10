<!--
  Copyright Contributors to the Open Shading Language project.
  SPDX-License-Identifier: BSD-3-Clause
  https://github.com/AcademySoftwareFoundation/OpenShadingLanguage
-->

Python bindings: pybind11 / nanobind dual-backend status
========================================================

OSL's Python bindings can be compiled against either
[pybind11](https://github.com/pybind/pybind11) or
[nanobind](https://github.com/wjakob/nanobind), selected at build time by
`OSL_PYTHON_BINDINGS_BACKEND` (`pybind11` | `nanobind` | `both`, default
`pybind11`). Both are built from **one** set of sources; the only difference
is the `OSL_PY_BACKEND_NANOBIND` define.

This mirrors what OpenImageIO did, and for the same reason: the two projects'
Python modules share `TypeDesc` through the binding framework's process-wide
type registry, so they need to be able to agree on which framework that is.
See `INSTALL.md`, section "Python binding backends", for the user-facing view.

The end state is nanobind only. pybind11 support exists so that the switch
can happen without a flag day, and so that any behavioral difference shows up
as a test failure rather than as a bug report.

Layout
------

| File | Role |
|---|---|
| `py_backend.h` | The compatibility shim. Everything the two frameworks spell differently lives here. |
| `py_osl.h` | Conversion helpers. Backend-neutral; no `#if` at all. |
| `py_osl.cpp` | The bindings. Backend-neutral except for the module entry point. |
| `__init__.py` | Package init for the single-backend case. |
| `nanobind/__init__.py` | Package init for the second module in `both` mode. |

Conventions for maintainers
---------------------------

- **Use the shim macros**, not the framework's own spellings:
  `OSL_PY_RW`, `OSL_PY_PROP_RO`, `OSL_PY_PROP_RW`.
- **Use the `osl_py::` helpers**: `str()`, `make_tuple()`, `make_iterator()`,
  `throw_key_error()`. Each exists because a direct call differs between the
  frameworks; the reason is commented at each definition.
- **`declare_*()` functions take `py_module&`**, never `py::module&`.
- **Adding a `#if defined(OSL_PY_BACKEND_NANOBIND)` outside `py_backend.h` is
  a last resort.** There is currently exactly one, for the module entry point
  (`NB_MODULE` must be at global scope; `PYBIND11_MODULE` need not be), and
  none at all in `py_osl.h`. If you need another, first try to absorb the
  difference into the shim, and if you can't, comment it with the specific
  framework difference that forces it.
- **Consumer-visible differences between the backends: none intended.** If
  you find one, that is a bug, not a documented quirk -- fix it and add a
  regression test. The one exception, which cannot be helped, is that
  pybind11 3.x injects a `_pybind11_conduit_v1_` member into every class it
  binds; that is a framework interop hook, not part of OSL's API.

Testing
-------

`testsuite/python-oslquery` runs once per selected backend, against a single
copy of the test script and a single `ref/out.txt`. The script contains no
backend awareness whatsoever -- it just does `import oslquery`, and
`PYTHONPATH` (set by `src/cmake/testing.cmake`) decides which module that
resolves to. In `both` mode you get `python-oslquery` and
`python-oslquery.nanobind`; with a single backend selected, whichever one it
is takes the plain name.

Byte-identical output from that one reference file is the equivalence
guarantee. If you add a binding, add coverage for it, and remember that the
interesting cases are often the empty ones -- a `ustring` that is empty or
default-constructed has a null `c_str()`, and that exact hazard once produced
a segfault that the entire testsuite failed to notice.

Known framework differences encountered so far
----------------------------------------------

All of these are already handled; the list is here so the next person doesn't
have to rediscover them.

| Difference | Handled by |
|---|---|
| `NB_MODULE` must be at global scope; `PYBIND11_MODULE` can be inside a namespace | The one `#if` in `py_osl.cpp` |
| nanobind has no value-returning `py::init(lambda)` | Use the templated `py::init<Args...>()`, which both accept |
| nanobind's `py::str` has no `std::string` constructor | `osl_py::str()` |
| nanobind's `py::tuple` can't be sized up front and assigned into | `osl_py::make_tuple()` |
| nanobind's `make_iterator` wants a scope type object and a name, and the scope must be a *bound* type | `osl_py::make_iterator<Scope>()` |
| Both frameworks overload `make_iterator` on (first, last) *and* on a whole container; passing lvalue iterators can make both viable. pybind11 2.10 alone took the pair by forwarding reference, making it genuinely ambiguous | `osl_py::make_iterator()` moves its arguments; see the comment there |
| nanobind's `key_error` takes `const char*`, pybind11's takes `std::string` | `osl_py::throw_key_error()` |
| nanobind's STL casters are opt-in, one header per type | Explicit includes in `py_backend.h`; note that a missing one fails at *runtime*, not compile time |
| nanobind has no `.cast<T>()` member function, only free `py::cast<T>()` | No longer relevant -- the only user was dead code |
| Python 3.9 provokes spurious nanobind leak warnings at shutdown | `py::set_leak_warnings(false)`, version-guarded (wjakob/nanobind#1405) |
| nanobind's CMake wants unversioned `Python::` targets, not `Python3::` | Second `find_package(Python ...)` in `find_python()` |
| nanobind's CMake package config installs somewhere `find_package` won't look | `discover_nanobind_cmake_dir()` and `build_nanobind.cmake` |

Remaining work
--------------

- Flip the default to `nanobind` once it has soaked in CI.
- Remove pybind11 support, and with it `py_backend.h`'s second arm, most of
  the `osl_py::` helpers, and the `both` mode plumbing.
