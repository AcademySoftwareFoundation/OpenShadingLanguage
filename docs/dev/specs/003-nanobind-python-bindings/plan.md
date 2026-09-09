# Implementation Plan: nanobind Python bindings (dual-backend)

**Branch**: `lg-nanobind` | **Date**: 2026-08-02 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `docs/dev/specs/003-nanobind-python-bindings/spec.md`

## Summary

Add `OSL_PYTHON_BINDINGS_BACKEND` (`pybind11` | `nanobind` | `both`; when unset,
auto-selected as of 2026-09-09 - nanobind when OIIO >= 3.2 and Python >= 3.10,
else pybind11; originally shipped with a fixed `pybind11` default) and make
`src/liboslquery/py_osl.{h,cpp}` compile unchanged under either
framework by routing every framework-specific spelling through a new
`src/liboslquery/py_backend.h`. The same sources are compiled once per selected backend
into separate CMake targets. The single Python test is registered once per selected
backend, differentiated only by `PYTHONPATH`, sharing one test script and one reference
output. Structure, naming, and known pitfalls follow OpenImageIO's implementation
(see [research.md](./research.md)).

## Technical Context

**Language/Version**: C++17 (OSL's `DOWNSTREAM_CXX_STANDARD`), Python >= 3.9, CMake

**Primary Dependencies**: pybind11 >= 2.7 (existing, now conditional); nanobind >= 2.8.0
(new, conditional); OpenImageIO (existing, unchanged)

**Storage**: N/A

**Testing**: CTest via `testsuite/runtest.py`; the single test `python-oslquery`

**Target Platform**: Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x64)

**Project Type**: C++ library with a Python extension module

**Performance Goals**: N/A - `OSLQuery` is a metadata-inspection API, not a hot path.
No shader execution is involved.

**Constraints**: The default configuration must be behaviorally identical to the
pre-change build. The two backends must expose an identical public Python surface.
Exactly one copy of the binding source.

**Scale/Scope**: 2 binding source files, ~350 lines (~120 live). 1 class family
(`OSLQuery` + `OSLQuery::Parameter`), 8 module attributes, 27 bound members. 1 test.

## Constitution Check

| Gate | Status | Notes |
|------|--------|-------|
| **I. Backward Compatibility** | PASS | No public C++ header changes. No Python API changes: same module name, same classes, same attributes, same semantics. `Parameter.type` is retained with its existing (pre-existing) OIIO-Python coupling. When the option is unset the default backend is auto-selected (as of 2026-09-09) from the OpenImageIO and Python versions; the Python API is identical whichever backend is chosen, only the binding framework differs. CHANGES.md entry required for the new option and conditional dependency (Principle IV also requires calling out the new dependency minimum). |
| **II. Physical Accuracy** | N/A | No shader execution, no closures, no numerics. `OSLQuery` reads compiled-shader metadata. |
| **III. Test-Driven Quality** | PASS | `python-oslquery` runs against every selected backend against one shared `ref/out.txt` - byte-identical output is the equivalence proof (SC-002, SC-006). Additionally re-enables the long-disabled `for p in q:` iteration coverage (FR-024). No reference images change. |
| **IV. Cross-Platform Portability** | PASS | nanobind is documented with a minimum version (>= 2.8.0) in INSTALL.md and CHANGES.md per the dependency-version rule. The CI jobs default `python_bindings_backend` to `both`, so every job on all three platforms builds and tests both the nanobind default and pybind11 (SC-011). Windows multi-config handling for the second module is explicitly addressed (T019). |
| **V. Performance & Scalability** | PASS | Nothing on any execution path changes. The only build-time cost is compiling ~350 lines a second time, and only when `both` is selected. |

No violations; Complexity Tracking is omitted.

## Project Structure

### Documentation (this feature)

```text
docs/dev/specs/003-nanobind-python-bindings/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output
├── tasks.md             # Phase 2 output (/speckit-tasks)
└── checklists/
    └── requirements.md  # Spec quality checklist
```

### Source Code (repository root)

```text
src/liboslquery/
├── py_backend.h              # NEW - the compatibility shim
├── py_osl.h                  # MODIFIED - includes py_backend.h; helpers de-pybind11-ed
├── py_osl.cpp                # MODIFIED - macro substitution; dual module entry point
├── __init__.py               # unchanged (pybind11 / single-backend package init)
├── nanobind/
│   └── __init__.py           # NEW - `both`-mode package init for the second module
├── CMakeLists.txt            # MODIFIED - two conditional module blocks
└── MIGRATION_STATUS.md       # NEW - maintainer conventions

src/cmake/
├── pythonutils.cmake         # MODIFIED - backend option, nanobind discovery, second setup macro
├── externalpackages.cmake    # MODIFIED - conditional pybind11 / nanobind finds
├── build_nanobind.cmake      # NEW - local build recipe (copied from OIIO)
└── testing.cmake             # MODIFIED - PYTHONPATH helper, per-backend test registration

testsuite/python-oslquery/    # test script, run.py, ref/ all shared unchanged

.github/workflows/            # MODIFIED - build-steps.yml input, ci.yml matrix entries
src/build-scripts/            # MODIFIED - ci-build.bash flag, dependency installs
INSTALL.md, CHANGES.md        # MODIFIED - dependency + option documentation
```

**Structure Decision**: OSL's bindings live inside `src/liboslquery/` alongside the C++
library, not in a top-level `src/python/`. This plan keeps them there. Unlike OIIO -
which needed a whole second directory (`src/python-nanobind/`) to hold a differing
package `__init__.py` - OSL needs only a one-file `nanobind/` subdirectory for the same
purpose, because the sources can be listed once and compiled twice from a single
`CMakeLists.txt`.

---

## Phase 0 - Prep under pybind11 (no behavior change)

Lands first; every step is verifiable with the existing test before nanobind enters the
picture. Justification for each item is in [research.md §1](./research.md).

**`src/liboslquery/py_osl.cpp`**

1. Delete the five `py::return_value_policy::reference_internal` arguments
   (`Parameter.metadata`, `OSLQuery.parameters`, `OSLQuery.metadata`, and both
   `__getitem__` overloads). Every decorated lambda returns *by value*, so the policy
   never governs a reference into internal state; it is already a no-op. Removing it
   deletes a shim entry and the single riskiest nanobind interaction.
2. Replace the value-returning factory constructor with the templated form:
   ```cpp
   .def(py::init<const std::string&, const std::string&>(),
        "shadername"_a, "searchpath"_a = "")
   ```
   `OSLQuery(string_view, string_view = {})` exists at `src/include/OSL/oslquery.h:118`
   and `std::string` converts implicitly. This matters: nanobind has no equivalent of
   pybind11's value-returning `py::init(lambda)`, so doing this now avoids needing a
   placement-new `__init__` shim later.
3. Delete the unused `#include <pybind11/embed.h>`.

**`src/liboslquery/py_osl.h`**

4. Delete `python_array_code()` / `typedesc_from_python_array_code()` (declared, never
   defined, never called) and `object_classname()` (unused, and uses the `.cast<>()`
   member syntax nanobind lacks).
5. Delete the unused `<pybind11/numpy.h>` and `<pybind11/operators.h>` includes.
6. Delete the `C_to_tuple<TypeDesc>` specialization (never instantiated).

**`testsuite/python-oslquery/src/test_oslquery.py`**

7. Fix the stale comment describing the printed type as "an OpenImageIO::TypeDesc but it
   can print like a string" - since 642ab36f the test prints `type_name`, a `str`.
8. Re-enable the `for p in q:` loop disabled by a 2020-era `FIXME(pybind11)` about a
   macOS crash with pybind11 2.6, and drop the `for i in range(len(q))` workaround. This
   is the only coverage of `__iter__` / `make_iterator` / `keep_alive<0,1>` - the
   construct that differs most between the backends (FR-024). `ref/out.txt` should not
   change; if it does, the change is a bug, not a reference update. If the macOS crash
   recurs, leave the loop disabled and record it - it is pre-existing.

**Gate**: `python-oslquery` passes against an unmodified `ref/out.txt`.

---

## Phase 1 - The compatibility shim

### New: `src/liboslquery/py_backend.h`

Modeled on `~/code/oiio/oiio.lg/src/python/py_backend.h`, trimmed to what OSL uses.
Selects on `OSL_PY_BACKEND_NANOBIND`. Provides:

| Shim name | pybind11 | nanobind |
|---|---|---|
| `namespace py` | `pybind11` | `nanobind` |
| `py_module` | `pybind11::module` | `nanobind::module_` |
| `OSL_PY_RW` | `def_readwrite` | `def_rw` |
| `OSL_PY_PROP_RO` | `def_property_readonly` | `def_prop_ro` |
| `OSL_PY_PROP_RW` | `def_property` | `def_prop_rw` |
| `osl_py::str(s)` | `py::str(s)` | `py::str(s.c_str(), s.size())` - nanobind's `str` has no `std::string` ctor |
| `osl_py::make_tuple(n, fn)` | `py::tuple t(n); t[i] = fn(i);` | `py::list` + `PyList_AsTuple` + `py::steal<py::tuple>` |
| `osl_py::make_iterator<Scope>(b, e)` | `py::make_iterator(b, e)` | `py::make_iterator(py::type<Scope>(), "Iterator", b, e)` |
| `osl_py::throw_key_error(s)` | `throw py::key_error(s)` | `throw py::key_error(s.c_str())` |

nanobind include set (opt-in, unlike `pybind11/stl.h`):
`<nanobind/nanobind.h>`, `<nanobind/make_iterator.h>`, `<nanobind/stl/string.h>`,
`<nanobind/stl/vector.h>`. **`stl/vector.h` is load-bearing** - it is what converts
`std::vector<Parameter>` for `.parameters` and `.metadata`; omitting it fails at
runtime, not at compile time.

Needing no shim, because both frameworks spell them the same: `py::int_`, `py::float_`,
`py::str`, `py::none`, `py::object`, `py::tuple`, `py::index_error`, `py::class_`,
`py::init<...>`, `py::keep_alive<0,1>`, and the `_a` literal.

`osl_py::make_iterator`'s scope argument must be a **bound** type -
`py::type<OSLQuery>()`. `py::type<std::vector<Parameter>>()` would be null since that
type is never registered.

### Modified: `src/liboslquery/py_osl.h`

Replace the pybind11 include block, the `namespace py` alias, and the `PY_STR` define
with `#include "py_backend.h"`. (`PY_STR` disappears entirely: it was doing double duty
as a type in `PyTypeForCType` and as a constructor at call sites. The type uses become
plain `py::str`, valid in both backends; the call sites become `osl_py::str()`.) `PyTypeForCType<>` and `C_to_val_or_tuple` are unchanged.
`C_to_tuple` bodies switch to `osl_py::make_tuple`. `declare_oslquery()` takes
`py_module&`, not `py::module&`.

### Modified: `src/liboslquery/py_osl.cpp`

Mechanical substitution per the table above, plus:

- Factor the eight `m.attr(...)` assignments into
  `void declare_module_attributes(py_module& m)` so each arm of the module-entry `#if`
  is three lines.
- The module entry point:

```cpp
#if defined(OSL_PY_BACKEND_NANOBIND)
}  // namespace PyOSL  -- NB_MODULE must be at global scope, unlike PYBIND11_MODULE

#  if defined(OSL_PY_NANOBIND_ISOLATED_PACKAGE)
NB_MODULE(_oslquery, m)
#  else
NB_MODULE(oslquery, m)
#  endif
{
#  if PY_VERSION_HEX < 0x030a0000
    // Python 3.9's shutdown/refcounting order produces bogus nanobind leak
    // warnings that do not occur on 3.10+. wjakob/nanobind#1405
    py::set_leak_warnings(false);
#  endif
    PyOSL::declare_module_attributes(m);
    PyOSL::declare_oslqueryparam(m);
    PyOSL::declare_oslquery(m);
}
#else
PYBIND11_MODULE(oslquery, m)
{
    declare_module_attributes(m);
    declare_oslqueryparam(m);
    declare_oslquery(m);
}
}  // namespace PyOSL
#endif
```

**Invariant (FR-018, SC-005)**: `#if defined(OSL_PY_BACKEND_NANOBIND)` appears at
**one** site in `py_osl.cpp` (the module macro; the Python-version guard nested inside
it is not backend-conditional) and **zero** sites in `py_osl.h`. Every other difference
is absorbed by `py_backend.h`, which carries six. Any new conditional site outside the
shim needs a comment naming the framework difference that forces it.

---

## Phase 2 - Build system

### `src/cmake/pythonutils.cmake`

**The option** - mirroring OIIO exactly. `set_cache` routes through
`set_utils.cmake:80 super_set` -> `set_from_env`, which is what makes an environment
variable of the same name work; CI depends on that (FR-003).

```cmake
set_cache (OSL_PYTHON_BINDINGS_BACKEND "pybind11"
     "Which Python binding backend(s) to build: pybind11, nanobind, or both" VERBOSE)
set_property (CACHE OSL_PYTHON_BINDINGS_BACKEND PROPERTY STRINGS pybind11 nanobind both)
string (TOLOWER "${OSL_PYTHON_BINDINGS_BACKEND}" OSL_PYTHON_BINDINGS_BACKEND)
if (NOT OSL_PYTHON_BINDINGS_BACKEND MATCHES "^(pybind11|nanobind|both)$")
    message (FATAL_ERROR
             "OSL_PYTHON_BINDINGS_BACKEND must be one of: pybind11, nanobind, both")
endif ()
```

plus derived `OSL_BUILD_PYTHON_PYBIND11` / `OSL_BUILD_PYTHON_NANOBIND` booleans, which
are what the rest of the build tests. `CMakeLists.txt:209 include (pythonutils)` already
precedes `include (externalpackages)`, so no reordering is needed.

**`find_python()`** - when `OSL_BUILD_PYTHON_NANOBIND`, add a second
`find_package (Python ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR} EXACT REQUIRED COMPONENTS ...)`.
nanobind's CMake package wants the unversioned `Python::Module` target, not OSL's
`Python3::*`.

**`discover_nanobind_cmake_dir()`** - copy OIIO's `pythonutils.cmake:112-145` verbatim.
It runs `python -m nanobind --cmake_dir` to find nanobind installed as a pip/brew Python
package (FR-006). Keep it a `function`, not a `macro`: a macro's early `return()` would
abort whatever file included pythonutils.cmake and called it at file scope. OIIO
documents this in a comment worth carrying over.

**`setup_python_module_nanobind()`** - beside the existing `setup_python_module()`. Same
body except:
- `nanobind_add_module(${target} ${sources})` takes no `${PYLIB_LIB_TYPE}` argument.
- `target_compile_options (nanobind-static PUBLIC -Wno-format-nonliteral)` for
  Clang/AppleClang/IntelLLVM - nanobind's own sources warn otherwise.
- Output and install directories per the layout table below.

### `src/cmake/externalpackages.cmake:89-93`

```cmake
find_python ()
if (USE_PYTHON AND OSL_BUILD_PYTHON_PYBIND11)
    checked_find_package (pybind11 REQUIRED VERSION_MIN 2.7)
endif ()
if (USE_PYTHON AND OSL_BUILD_PYTHON_NANOBIND)
    discover_nanobind_cmake_dir()
    checked_find_package (nanobind CONFIG REQUIRED VERSION_MIN 2.8.0 BUILD_LOCAL missing)
endif ()
```

### New: `src/cmake/build_nanobind.cmake`

Copy OIIO's verbatim. OSL already has the full
`build_dependency_with_cmake()` / `BUILD_LOCAL` / `<pkg>_REFIND` machinery in
`src/cmake/dependency_utils.cmake` (lines 278-473, 605-800), and the same
`${PROJECT_NAME}_LOCAL_DEPS_ROOT` variable name, so it drops in unchanged. Preserve both
of OIIO's workarounds and their explanatory comments:

1. Pre-clone the source and run `git submodule update --init --depth 1 -- ext/robin_map`;
   nanobind's `CMakeLists.txt` hard-errors without that submodule, and
   `build_dependency_with_cmake()`'s plain `git clone` does not init submodules.
2. `set (nanobind_DIR "${nanobind_LOCAL_INSTALL_DIR}/nanobind/cmake" CACHE PATH ... FORCE)`;
   nanobind installs its package config to a subdirectory that generic prefix search
   does not check.

This is OSL's first use of `BUILD_LOCAL`. If it misbehaves, the fallback is a
`src/build-scripts/build_nanobind.bash` in the style of the existing `build_pybind11.bash`.

### Layouts

| Backend | Targets | Init name | Build tree | Install |
|---|---|---|---|---|
| `pybind11` | `pyoslquery` | `oslquery` | `lib/python/site-packages/oslquery.so` | `${PYTHON_SITE_DIR}/oslquery/` |
| `nanobind` | `pyoslquery` | `oslquery` | `lib/python/site-packages/oslquery.so` | `${PYTHON_SITE_DIR}/oslquery/` |
| `both` | `pyoslquery` + `pyoslquery_nanobind` | `oslquery` + `_oslquery` | pybind as above; nanobind at `lib/python/nanobind/oslquery/{__init__.py,_oslquery.so}` | pybind as above; nanobind at `${PYTHON_SITE_DIR}/nanobind/oslquery/` |

Single-backend layouts are byte-identical to today's (FR-008, SC-001). The `both`-mode
install deliberately deviates from OIIO, which installs its nanobind `__init__.py` into
the same site directory as pybind11's, where the two collide; a `nanobind/`
subdirectory avoids that (FR-009, SC-010). Build-tree layout matches OIIO either way.

### `src/liboslquery/CMakeLists.txt:42-50`

Split into two conditional blocks over the *same* `file(GLOB py_*.cpp)` source list. The
nanobind block adds `-DOSL_PY_BACKEND_NANOBIND`, plus
`-DOSL_PY_NANOBIND_ISOLATED_PACKAGE` when the backend is `both`, and in `both` mode
`configure_file(nanobind/__init__.py ... COPYONLY)` into
`${CMAKE_BINARY_DIR}/lib/python/nanobind/oslquery/`.

### New: `src/liboslquery/nanobind/__init__.py`

`from ._oslquery import *`, plus the same Windows `add_dll_directory` block as the
existing `src/liboslquery/__init__.py`, plus OIIO's trick of appending
`Release`/`Debug`/`RelWithDebInfo`/`MinSizeRel` to `__path__` so the per-configuration
`.pyd` resolves on MSVC multi-config builds - handled in Python rather than by fighting
CMake's output-directory layout.

---

## Phase 3 - Tests

**The blocker**: CMake never sets `PYTHONPATH` for tests today. It is set only in
`Makefile:293`, for the `make test` target - which is why a bare `ctest` in the build
directory cannot run `python-oslquery` at all. Per-backend selection is entirely
`PYTHONPATH`-driven, so this must move into CMake (FR-023).

### `src/cmake/testing.cmake`

Add `osl_tests_pythonpath_env_entry(out_var prefix_dir)`, mirroring OIIO's
`testing.cmake:28`: emits one `PYTHONPATH=<dir>:$ENV{PYTHONPATH}` string, and on Windows
uses `<dir>` alone, because semicolon-separated values get split by CMake list
processing when used as a CTest `ENVIRONMENT` entry.

Replace the registration at `testing.cmake:474`:

```cmake
if (USE_PYTHON AND Python3_Development_FOUND AND NOT SANITIZE)
    set (_py_testsrc "${CMAKE_SOURCE_DIR}/testsuite/python-oslquery")
    osl_tests_pythonpath_env_entry (_pybind_pypath
                                    "${CMAKE_BINARY_DIR}/lib/python/site-packages")
    if (OSL_PYTHON_BINDINGS_BACKEND STREQUAL "both")
        osl_tests_pythonpath_env_entry (_nb_pypath
                                        "${CMAKE_BINARY_DIR}/lib/python/nanobind")
    else ()
        set (_nb_pypath "${_pybind_pypath}")
    endif ()
    set (_nb_suffix ".nanobind")
    if (OSL_BUILD_PYTHON_PYBIND11)
        add_one_testsuite ("python-oslquery" "${_py_testsrc}"
                           ENV TESTSHADE_OPT=0 "${_pybind_pypath}")
    else ()
        set (_nb_suffix "")   # nanobind-only: the test takes the plain name
    endif ()
    if (OSL_BUILD_PYTHON_NANOBIND)
        add_one_testsuite ("python-oslquery${_nb_suffix}" "${_py_testsrc}"
                           ENV TESTSHADE_OPT=0 "${_nb_pypath}")
    endif ()
endif ()
```

The empty-suffix trick is OIIO's and satisfies FR-022: with a single backend the test
keeps its historical name regardless of which backend that is; with `both`, CTest shows
`python-oslquery` and `python-oslquery.nanobind`. Distinct names give distinct sandbox
directories automatically - `add_one_testsuite` derives `testdir` from `testname`
(`testing.cmake:37`) - while `OSL_TESTSUITE_SRC` still points at the one source
directory, so both variants share one `run.py` and one `ref/` (FR-019, FR-021).

Calling `add_one_testsuite()` directly rather than going through `TESTSUITE()` is
deliberate: `TESTSUITE()` has no `ENV` pass-through, and it currently also generates a
`python-oslquery.rs_bitcode` variant of a test that executes no shader. Dropping that
halves the variant count before `both` doubles it. The `NOOPTIMIZE` and `NOOPTIX` marker
files in `testsuite/python-oslquery/` become vestigial; remove them and say so in the
commit.

### Everything else in the testsuite is untouched

`run.py`, `src/test_oslquery.py`, and `ref/out.txt` are shared verbatim. The test does
`import oslquery` with no backend awareness whatsoever (FR-020) - that is the whole
point of the design, and it is why `runtest.py` needs no changes either.

`Makefile:293` keeps its `PYTHONPATH` prefix for interactive use, but the CTest
`ENVIRONMENT` property now takes precedence for these tests. Note that in the commit.

---

## Phase 4 - CI and documentation

**`.github/workflows/build-steps.yml`**: new `python_bindings_backend` input
(default `''`), exported as env `OSL_PYTHON_BINDINGS_BACKEND` alongside the existing
`PYBIND11_VERSION` / `PYTHON_VERSION` at lines 135-136.

**`src/build-scripts/ci-build.bash`**: after the `USE_SIMD` block,

```bash
if [[ -n "${OSL_PYTHON_BINDINGS_BACKEND:-}" ]] ; then
    OSL_CMAKE_FLAGS="$OSL_CMAKE_FLAGS -DOSL_PYTHON_BINDINGS_BACKEND=${OSL_PYTHON_BINDINGS_BACKEND}"
fi
```

**`.github/workflows/ci.yml`**: forward the input at the ~5 job-level sites that already
forward `pybind11_ver` (lines 66, 150, 435, 515, 585), and set
`python_bindings_backend: both` on exactly three matrix entries - one recent Linux, one
macOS-arm, one Windows (SC-011). Leave the other ~18 `pybind11_ver` entries alone; they
continue to exercise the default. `analysis.yml` is untouched.

**Dependency installation**: `install_homebrew_deps.bash` gains a conditional
`brew install nanobind`; `gh-installdeps.bash` and `gh-win-installdeps.bash` gain a
pinned `pip install`, following OIIO's `ci-requirements-nanobind.txt` sha256-pinning
pattern. The `BUILD_LOCAL missing` path covers anything these miss.

**Stale CI exclusion**: `ci.yml:199-201, 212-214` exclude `python-oslquery` on two
ASWF-container jobs "until the ASWF container properly includes OIIO's python bindings".
Since 642ab36f the test no longer imports OpenImageIO. Verify and remove - as a separate
commit, since it is an independent fix that stands on its own.

**Documentation**:
- `INSTALL.md:69-73` - nanobind as a conditional dependency with its minimum version
  (Principle IV), and an `OSL_PYTHON_BINDINGS_BACKEND=pybind11|nanobind|both` entry in
  the build-options section (FR-025).
- The `Parameter.type` interoperability note (FR-026): it returns an OIIO `TypeDesc` and
  therefore needs OIIO's Python module, built with the same binding backend and a
  compatible internals/ABI version, to have been imported; `type_name` is the
  coupling-free alternative. Also as a comment at the binding site.
- `CHANGES.md` - **deferred to release preparation** at the maintainer's direction;
  CHANGES.md is written as part of the release process rather than per-PR.
- `src/liboslquery/MIGRATION_STATUS.md` (FR-028) - a short version of OIIO's, carrying
  the maintainer conventions: which macros to use, `py_module&` rather than
  `py::module&`, keep `#if` sites minimal and commented, and "consumer-visible
  differences: none intended - if you find one, it is a bug; add a regression test."

---

## Risks and mitigations

| # | Risk | Mitigation | Fallback |
|---|---|---|---|
| 1 | nanobind's list caster rejects or mishandles `std::vector<Parameter>` | Phase 0 removes the `reference_internal` policies that would conflict | Convert to a list explicitly in the lambda, or bind the vector as an opaque type |
| 2 | `make_iterator` scope handle resolves to null | Shim passes `py::type<OSLQuery>()`, a bound type | Implement `__iter__` as iteration over the already-converted list |
| 3 | First use of `BUILD_LOCAL` in OSL | Machinery is present and identical to OIIO's | `src/build-scripts/build_nanobind.bash`, matching `build_pybind11.bash` |
| 4 | Re-enabled `for p in q:` still crashes on macOS | Pre-existing condition, not a regression | Leave disabled, record it; the nanobind variant still exercises iteration if it is only the pybind11 path that fails |
| 5 | Windows `both` mode - two modules, per-config output dirs | Handled Python-side in `nanobind/__init__.py`, as OIIO does | Restrict `both` to non-Windows in CI and document it |

## Verification

Three full configure/build/test cycles from the repository root, with
`OpenImageIO_ROOT=~/code/oiio/oiio.lg/dist`:

```bash
make OSL_PYTHON_BINDINGS_BACKEND=pybind11 && make test TEST=python-oslquery
make OSL_PYTHON_BINDINGS_BACKEND=nanobind && make test TEST=python-oslquery
make OSL_PYTHON_BINDINGS_BACKEND=both     && make test TEST=python-oslquery
```

Expected: one test in the first two cases (named `python-oslquery` in both), two in the
third (`python-oslquery` and `python-oslquery.nanobind`), all diffing clean against the
single unmodified `testsuite/python-oslquery/ref/out.txt`.

Then, by hand:

- `oslquery.__file__` resolves to the expected module for each `PYTHONPATH`.
- `sorted(dir(oslquery.OSLQuery))` and `sorted(dir(oslquery.Parameter))` and the sorted
  module attribute list compare equal across backends, with zero differences (SC-003).
- `p.type` raises the same error class under both backends when `OpenImageIO` has not
  been imported (FR-015).
- `cmake --build build --target install` for each backend; the installed layout matches
  the table above, and the `both` install leaves both `__init__.py` files intact
  (SC-010).
- Configuring with `-DOSL_PYTHON_BINDINGS_BACKEND=garbage` fails immediately with a
  message naming the three accepted values (SC-007).
- A pybind11 build succeeds with nanobind absent, and a nanobind build succeeds with
  pybind11 absent (SC-008, SC-009).
