# Research: nanobind Python bindings for OSL

**Feature**: 003-nanobind-python-bindings | **Date**: 2026-08-02

This is the Phase 0 output. It records (a) what OSL's bindings actually consist of,
(b) how OpenImageIO implemented the same dual-backend arrangement, and (c) which of the
pybind11/nanobind differences OIIO hit are relevant to OSL.

---

## 1. OSL's current binding surface (complete inventory)

The bindings live in `src/liboslquery/`, not in a `src/python/` directory.

| File | Lines | Role |
|---|---|---|
| `src/liboslquery/py_osl.h` | 158 | Python.h/pybind11 includes, `PY_STR`, C-to-Python conversion helpers |
| `src/liboslquery/py_osl.cpp` | 193 | The whole binding: `Parameter`, `OSLQuery`, `PYBIND11_MODULE(oslquery, m)` |
| `src/liboslquery/__init__.py` | 18 | Windows DLL-path workaround, `from .oslquery import *` |
| `src/liboslquery/CMakeLists.txt` | 51 | Lines 42-50 build the module via `setup_python_module()` |

**Module name**: `oslquery`, hardcoded in `PYBIND11_MODULE`. Note that
`setup_python_module()` defines `PYMODULE_NAME` (`pythonutils.cmake:89-90`) but nothing
in the source ever references it.

**Everything bound** (`py_osl.cpp`):

- Module attributes: `osl_version`, `VERSION`, `VERSION_STRING`, `VERSION_MAJOR`,
  `VERSION_MINOR`, `VERSION_PATCH`, `INTRO_STRING`, `__version__`.
- `Parameter`: `__init__()`, `__init__(Parameter)`, `name`, `type`, `type_name`,
  `isoutput`, `varlenarray`, `isstruct`, `isclosure`, `value`, `spacename`, `fields`,
  `structname`, `metadata`.
- `OSLQuery`: `__init__()`, `__init__(shadername, searchpath="")`, `open()`,
  `open_bytecode()`, `shadertype()`, `shadername()`, `nparams`, `parameters`,
  `metadata`, `__len__`, `__getitem__(int)`, `__getitem__(str)`, `__iter__`,
  `geterror(clear_error=True)`.

**pybind11 features actually used**: `py::class_`, `py::init<...>` and value-returning
`py::init(lambda)`, `def_readwrite`, `def_property`, `def_property_readonly`,
`return_value_policy::reference_internal`, `keep_alive<0,1>`, `make_iterator`,
`py::tuple`/`str`/`int_`/`float_`/`none`/`object`, `py::index_error`, `py::key_error`,
`pybind11::literals` (`_a`), and `pybind11/stl.h` auto-conversion of
`std::vector<Parameter>` and `std::string`.

**Not used at all**: buffer protocol, numpy, custom type casters, `py::enum_`,
`py::implicitly_convertible`, operators, GIL scope objects, submodules,
`py::module::import`. This is why OSL's port is far smaller than OIIO's - OIIO hit
almost all of those, OSL hits none of them.

**Dead code found while inventorying** (delete during the port):

- `py_osl.h:54-55` - `python_array_code()` and `typedesc_from_python_array_code()` are
  declared, defined nowhere in the repo, called nowhere.
- `py_osl.h:58-62` - `object_classname()` unused, and uses the `.cast<py::str>()`
  member-function syntax nanobind does not have.
- `py_osl.h:107-116` - the `C_to_tuple<TypeDesc>` specialization is never instantiated;
  it is also the only `py::cast` of a `TypeDesc` in a conversion path.
- `py_osl.h:32-33` - `<pybind11/numpy.h>` and `<pybind11/operators.h>` unused.
- `py_osl.cpp:7` - `<pybind11/embed.h>` unused.

**Finding: all five `reference_internal` policies are already no-ops.** Every lambda
they decorate returns *by value*: `return p.metadata;` deduces `std::vector<Parameter>`,
`return self.parameters();` likewise (the function returns a const ref, the lambda's
deduced return type strips it), and `return *p;` deduces `Parameter`. The policy
therefore never governs a reference to internal state. Removing them is a no-op under
pybind11 and removes the single riskiest nanobind interaction (see §4.1).

**Finding: the factory-lambda constructor is avoidable.**
`src/include/OSL/oslquery.h:118` declares
`OSLQuery(string_view shadername, string_view searchpath = string_view())`, and
`std::string` converts implicitly to `string_view`. So
`py::init<const std::string&, const std::string&>()` works and is portable to nanobind,
whereas the current value-returning `py::init(lambda)` is a pybind11-only form.

---

## 2. Prior art in OSL: commit 642ab36f

`642ab36f "python: migration of oslquery python bindings away from OpenImageIO types"`
touched four files and did the essential prep:

- `src/include/OSL/oslquery.h:97-101` - added `std::string type_name() const` and
  `void type_name(const std::string&)` to `OSLQuery::Parameter`.
- `src/liboslquery/oslquery.cpp` - trivial implementations (`return type.c_str();` /
  `type = TypeDesc(typestring);`).
- `src/liboslquery/py_osl.cpp:29-32` - added the `type_name` property; **removed** the
  forced `py::module oiio = py::module::import("OpenImageIO");` that used to sit at the
  top of `PYBIND11_MODULE`.
- `testsuite/python-oslquery/src/test_oslquery.py` - switched `p.type` to `p.type_name`.

Its commit message states that `Parameter.type` is "the one and only spot in which OSL's
OSLQuery python API *requires* use of anything provided by the OIIO python bindings",
and that mixing a nanobind OIIO with a pybind11 OSL "DOES NOT WORK" on main before that
PR. That is the problem this feature closes.

**Side effect worth acting on**: `.github/workflows/ci.yml:199-201, 212-214` exclude
`python-oslquery` on two ASWF-container jobs with the comment "until the ASWF container
properly includes OIIO's python bindings". After 642ab36f the test no longer imports
OpenImageIO at all, so those exclusions are probably stale. Verify and remove separately.

---

## 3. OpenImageIO's implementation (the reference)

Source: `~/code/oiio/oiio.lg`. Relevant commits, in order:

```
409621b0f feat: Nanobind for python bindings (first steps -- pybind11 still working) (#5084)
6ea45b11c ci: Run both tests pybind11 and nanobind in CI (#5176)
667365f45 Unify pybind11 and nanobind into single-source bindings (#5254)
1cb1da122 feat(python): Expand dual-backend support in Python bindings (#5310)
62d139599 python: nanobind tidying, auto-build, CI testing
```

**Important trajectory lesson**: #5084 created `src/python-nanobind/` as a *full
duplicate* of every binding source file. #5254 then deleted all of those duplicates and
made both backends compile from the single `src/python/` tree via a compatibility
header. OSL should skip straight to the end state; there is no reason to repeat the
duplicate-then-unify detour on a 350-line binding.

Today `src/python-nanobind/` contains only `CMakeLists.txt` and `__init__.py` - it
exists purely to build a second, isolated module when the backend is `both`.

### 3.1 The option

`src/cmake/pythonutils.cmake:6-35`. Name `OIIO_PYTHON_BINDINGS_BACKEND`, default
`"pybind11"`, values `pybind11|nanobind|both`, lowercased then validated with
`FATAL_ERROR`. Declared with `set_cache(...)`, which routes through
`set_utils.cmake:80 super_set` -> `set_from_env`, meaning **an environment variable of
the same name sets it** - that is exactly how CI drives it
(`ci-startup.bash:56-57 export OIIO_PYTHON_BINDINGS_BACKEND=both`). Two derived
booleans, `OIIO_BUILD_PYTHON_PYBIND11` and `OIIO_BUILD_PYTHON_NANOBIND`, are what the
rest of the build tests.

Include order matters: `include (pythonutils)` precedes `include (externalpackages)` so
the derived booleans exist when dependency resolution runs. OSL already has this order
(`CMakeLists.txt:209`).

### 3.2 Dependency discovery

`src/cmake/externalpackages.cmake:116-130` - both finds are guarded by the derived
booleans; nanobind is `checked_find_package (nanobind CONFIG REQUIRED VERSION_MIN 2.8.0 BUILD_LOCAL missing)`.

`discover_nanobind_cmake_dir()` (`pythonutils.cmake:112-145`) handles nanobind installed
as a pip/brew *Python package* by running `python -m nanobind --cmake_dir` and setting
`nanobind_DIR` from the result. It is deliberately a `function`, not a `macro`, with
this comment: a macro's early `return()` would abort whatever file included
pythonutils.cmake and called it at file scope.

`find_python()` (`pythonutils.cmake:78-86`) does a *second*
`find_package (Python <maj>.<min> EXACT REQUIRED COMPONENTS ...)` when nanobind is
enabled, because nanobind's CMake package expects the unversioned `Python::Module`
targets, not the versioned `Python3::*` ones.

### 3.3 Local build of nanobind

`src/cmake/build_nanobind.cmake`. Two non-obvious workarounds, both of which OSL will
need verbatim:

1. nanobind vendors `tsl::robin_map` as a git submodule and its `CMakeLists.txt`
   hard-errors if it is not checked out; `build_dependency_with_cmake()`'s plain
   `git clone` does not init submodules. So the file clones the source itself and runs
   `git submodule update --init --depth 1 -- ext/robin_map` before delegating.
2. nanobind installs its CMake package config to `<prefix>/nanobind/cmake`, a layout
   generic prefix search does not check, so `nanobind_DIR` is set explicitly with
   `FORCE`.

Pinned at `nanobind_BUILD_VERSION 2.13.0`, with a `nanobind_GIT_COMMIT` hash verified
against the tag. Configured with `-D NB_TEST=OFF`; there is effectively nothing to
compile, the "build" just copies headers/sources/CMake helpers to a prefix.

Related: OIIO switched robin-map to `checked_find_package (Robinmap CONFIG ... NAMES tsl-robin-map ...)`
specifically so nanobind's own CMake reuses that `tsl::robin_map` target instead of its
private vendored copy.

### 3.4 Module targets and layout

`src/python/CMakeLists.txt` builds the pybind11 module from `file(GLOB *.cpp)`. In
nanobind-**only** mode it builds the same files again through
`setup_python_module_nanobind()` with `-DOIIO_PY_BACKEND_NANOBIND`. In **both** mode it
skips that and `src/python-nanobind/CMakeLists.txt` builds them instead, referencing the
sources by explicit `../python/py_*.cpp` paths, adding `-DOIIO_PY_NANOBIND_ISOLATED_PACKAGE`.

So: **the same .cpp files, two CMake targets, two sets of object files, different
`-D` flags.**

| Backend | Targets | Init name | Build tree | Install |
|---|---|---|---|---|
| pybind11 | `PyOpenImageIO` | `OpenImageIO` | `lib/python/site-packages` | `${PYTHON_SITE_DIR}` |
| nanobind | `PyOpenImageIO` | `OpenImageIO` | `lib/python/site-packages` | `${PYTHON_SITE_DIR}` (drop-in) |
| both | `PyOpenImageIO` + `PyOpenImageIONanobind` | `OpenImageIO` + `_OpenImageIO` | pybind in `lib/python/site-packages`; nanobind in `lib/python/nanobind/OpenImageIO/` | both under `${PYTHON_SITE_DIR}` |

`setup_python_module_nanobind()` (`pythonutils.cmake:230+`) mirrors the pybind11 macro
except that `nanobind_add_module()` takes no `MODULE`/`SHARED` type argument, and it
applies `target_compile_options (nanobind-static PUBLIC -Wno-format-nonliteral)` for
Clang - nanobind's own sources warn otherwise.

**Deviation OSL should make**: in `both` mode OIIO installs its nanobind `__init__.py`
to the same `${PYTHON_SITE_DIR}` as pybind11's. The extension modules do not collide
(`OpenImageIO.so` vs `_OpenImageIO.so`) but the two `__init__.py` files do. OSL should
install the nanobind package under a `nanobind/` subdirectory of the site dir instead.

### 3.5 The compatibility header

`src/python/py_backend.h` (4.6K). Selects on `OIIO_PY_BACKEND_NANOBIND`. Provides a
`namespace py` alias, a `py_module` typedef, a set of `OIIO_PY_*` macros for the binding
verbs, and a `namespace oiio_py` of small inline helpers.

Full mapping table (OIIO names; OSL will use `OSL_PY_*` / `osl_py`):

| Shim name | pybind11 | nanobind |
|---|---|---|
| `namespace py` | `pybind11` | `nanobind` |
| `py_module` | `pybind11::module` | `nanobind::module_` |
| `OIIO_PY_RW` | `def_readwrite` | `def_rw` |
| `OIIO_PY_RO` | `def_readonly` | `def_ro` |
| `OIIO_PY_PROP_RO` | `def_property_readonly` | `def_prop_ro` |
| `OIIO_PY_PROP_RW` | `def_property` | `def_prop_rw` |
| `OIIO_PY_PROP_RW_NONE` | `def_property` | `def_prop_rw(..., py::for_setter(py::arg().none()))` |
| `OIIO_PY_RO_STATIC` | `def_property_readonly_static` | `def_prop_ro_static` |
| `oiio_py::ref` | `return_value_policy::reference` | `rv_policy::reference` |
| `oiio_py::ref_internal` | `return_value_policy::reference_internal` | `rv_policy::reference_internal` |
| `oiio_py::str(x)` | `py::str(x)` | `std::string(x)` |
| `oiio_py::str_to_stdstring(h)` | `std::string(cast<py::str>(h))` | `std::string(cast<py::str>(h).c_str())` |
| `oiio_py::bytes_to_stdstring(b)` | `std::string(b)` | `std::string(b.c_str(), b.size())` |
| `oiio_py::throw_key_error(s)` | `py::key_error(std::string)` | `py::key_error(const char*)` |
| `oiio_py::make_tuple(n, fn)` | `py::tuple(n)` + indexed assign | `py::list` + `PyList_AsTuple` + `py::steal` |
| `oiio_py::make_iterator(c)` | `py::make_iterator(b, e)` | `py::make_iterator(py::type<C>(), "iterator", b, e)` |
| `oiio_py::return_object(o)` | identity | `py::borrow(o)` |
| `oiio_py::make_numpy_array(p,n)` | `py::array_t<T>` | `py::ndarray<py::numpy, T, py::shape<-1>>` + `rv_policy::move` |
| `PY_STR(x)` | `py::str` | `oiio_py::str(x)` |

nanobind's stl casters are **opt-in headers**, unlike `pybind11/stl.h`. OIIO includes
`<nanobind/stl/{array,optional,string,string_view,unique_ptr,vector}.h>` plus
`<nanobind/{nanobind,ndarray,operators,make_iterator}.h>`.

`declare_*()` free functions take `py_module&`, never `py::module&`.

### 3.6 How much `#if` survived in OIIO

Across 13 binding `.cpp` files:

```
py_oiio.cpp          6
py_imagebufalgo.cpp  2
py_imagebuf.cpp      1
py_paramvalue.cpp    1
py_typedesc.cpp      1
everything else      0
```

That is the bar: nine conditional sites across ~15k lines of bindings. OSL's target is
at most three across 350 lines, and realistically two.

### 3.7 Test registration

The test scripts contain **zero** backend awareness - they just
`import OpenImageIO as oiio`. Selection is entirely `PYTHONPATH`.

`src/cmake/testing.cmake:28` defines `oiio_tests_pythonpath_env_entry()`, which builds
one `PYTHONPATH=<dir>:$ENV{PYTHONPATH}` string, and on Windows uses `<dir>` alone
because semicolon-separated values get split by CMake list processing when used as a
CTest `ENVIRONMENT` entry.

The registration block (`testing.cmake:246-335`) uses this trick:

```cmake
set (nanobind_python_test_suffix ".nanobind")
if (OIIO_BUILD_PYTHON_PYBIND11)
    oiio_add_tests (<list> ENVIRONMENT "${_pybind_tests_pythonpath}")
else ()
    set (nanobind_python_test_suffix "")   # nanobind-only: take the plain names
endif ()
if (OIIO_BUILD_PYTHON_NANOBIND)
    oiio_add_tests (<list> SUFFIX ${nanobind_python_test_suffix}
                    ENVIRONMENT "${_nanobind_tests_pythonpath}")
endif ()
```

The suffix also names a distinct build-tree run directory while `OIIO_TESTSUITE_SRC`
still points at the single source directory - so both variants share one `run.py` and
one `ref/` and run in isolated scratch dirs. `testsuite/runtest.py` has no backend
awareness whatsoever.

The `both`-mode `__init__.py` (`src/python-nanobind/__init__.py`) does
`from ._OpenImageIO import *`, and on Windows appends `Release`/`Debug`/`RelWithDebInfo`/`MinSizeRel`
to `__path__` so the per-configuration `.pyd` location resolves - handled in Python
rather than by fighting CMake's multi-config output layout.

### 3.8 Docs and CI in OIIO

- `INSTALL.md:47-50` conditional nanobind dependency; `:166-170` the option; `:262` a
  make-wrapper row.
- `CHANGES.md` three entries (one per PR).
- `src/python/MIGRATION_STATUS.md` - the living maintainer doc. Its "Conventions" section
  is the checklist worth copying, and it states: "**Consumer-visible differences: None
  intended.** ... If you find a behavioral difference, treat it as a bug and add a
  regression test."
- CI: a `oiio_python_bindings_backend` workflow input (`build-steps.yml:96-98`) exported
  as an env var (`:159`), forwarded from `ci.yml` at three job sites, set to `both` on
  one linux, one macos14-arm, and one windows-2025 matrix entry.
  `ci-build.bash:28-30` turns the env var into a `-D` flag.
  `install_homebrew_deps.bash:60-61` brew-installs nanobind conditionally;
  `ci-requirements-nanobind.txt` pip-pins it with a sha256 hash for Linux/Windows.
- `pyproject.toml` still requires only pybind11 - wheels remain pybind11-only. Stub
  generation is likewise still pybind11-driven, with the same `.pyi` installed for both.
  Both are out of scope for OSL (it has neither).

---

## 4. pybind11/nanobind differences: which ones OSL actually hits

OIIO's tree documents about twenty. Sorted by relevance to OSL:

### 4.1 Hit by OSL

| # | Difference | OSL's exposure | Resolution |
|---|---|---|---|
| 1 | `PYBIND11_MODULE` can live inside a namespace; `NB_MODULE` must be at global scope | `py_osl.cpp:9,176,193` wraps the macro in `namespace PyOSL` | Close the namespace before the nanobind arm. One of the two permitted `#if` sites. |
| 2 | No value-returning `py::init(lambda)` in nanobind; it needs a placement-new `__init__` | `py_osl.cpp:98-102` | **Avoid entirely** - use `py::init<const std::string&, const std::string&>()`, valid in both. See §1. |
| 3 | `make_iterator` signature: nanobind needs a type handle and a name | `py_osl.cpp:158-164` | Shim helper. The scope handle must be a *bound* type - `py::type<OSLQuery>()`, not `py::type<std::vector<Parameter>>()` which would be null. |
| 4 | `py::tuple` cannot be built by indexed assignment in nanobind | `py_osl.h:88-103, 121-128` | `osl_py::make_tuple(n, fn)` shim building a `py::list` then `PyList_AsTuple` + `py::steal`. |
| 5 | `py::key_error` takes `const char*` in nanobind, `std::string` in pybind11 | `py_osl.cpp:153` | `osl_py::throw_key_error(std::string)` shim. |
| 6 | stl casters are opt-in headers in nanobind | `pybind11/stl.h` at `py_osl.h:35` is what converts `std::vector<Parameter>` and `std::string` | Include `<nanobind/stl/vector.h>` and `<nanobind/stl/string.h>` in the shim. **Load-bearing** - omit and `.parameters` silently fails to convert. |
| 7 | nanobind's `py::str` has no `std::string` constructor (pybind11's does) | `PY_STR(p.name.string())` and friends | `osl_py::str()` shim: `py::str(s)` vs `py::str(s.c_str(), s.size())`. **Do not "fix" this by passing `ustring::c_str()`** - see #7b. |
| 7b | `ustring::c_str()` returns `nullptr` for an empty or default-constructed ustring; `ustring::string()` is null-safe | `Parameter::structname` on any non-struct param, and any empty `ustring` in `sdefault`/`spacename`/`fields` | Found the hard way: routing `PY_STR` through `.c_str()` segfaults inside `PyUnicode_FromString`. The existing testsuite does **not** catch it - `test_oslquery.py` only prints `structname` inside the `isstruct` branch. The shim's `str()` takes `const std::string&` and its `const char*` overload maps null to `""`. |
| 8 | Python 3.9 emits spurious nanobind leak warnings at interpreter shutdown | any 3.9 build | `py::set_leak_warnings(false)` guarded on `PY_VERSION_HEX < 0x030a0000`. The second permitted `#if` site. (wjakob/nanobind#1405) |
| 9 | `.cast<T>()` member function does not exist in nanobind; only free `py::cast<T>(obj)` | `py_osl.h:61` `object_classname()` | Dead code - delete it. |
| 10 | `reference_internal` interacting with stl container casters | five sites in `py_osl.cpp` | All five are already no-ops (§1). Delete them in Phase 0, before nanobind exists. |
| 11 | nanobind's CMake wants unversioned `Python::` targets | `find_python()` uses `Python3::*` | Second `find_package(Python ... EXACT REQUIRED)`. |
| 12 | Clang warns inside nanobind's own sources | any Clang/AppleClang build | `target_compile_options (nanobind-static PUBLIC -Wno-format-nonliteral)`. |
| 13 | nanobind's git submodule + non-standard CMake config install dir | local builds | Both handled in the copied `build_nanobind.cmake`. |
| 14 | MSVC multi-config puts the extension in a per-config subdir | Windows `both` mode | Handled Python-side in the `both`-mode `__init__.py` by appending config names to `__path__`. |
| 15 | pybind11 3.x adds a `_pybind11_conduit_v1_` member to every bound class (its cross-extension interop hook); nanobind has no equivalent | every bound class | **The only public-surface difference between the two modules.** It is a framework-internal artifact, not part of OSL's API, and nothing can or should be done about it. SC-003 excludes it explicitly. |
| 15b | `make_iterator` is overloaded on (first, last) *and* on a container (`Type& value, Extra&&...`) in both frameworks. pybind11 **2.10 only** took the pair overload by forwarding reference, so lvalue iterators deduce to `It&` and tie exactly with `Type&` -> ambiguous | Only shows up when the call is wrapped in a helper that names its parameters, as `osl_py::make_iterator` does; direct `.begin()`/`.end()` calls pass prvalues, which can't bind `Type&`. Fixed by `std::move`-ing into the call. Verified by compiling `py_osl.cpp` against pybind11 2.7.0, 2.9.0, 2.10.0, 2.11.1, 3.0.1 and master. |
| 16 | `from .X import *` in a package `__init__.py` skips underscore-prefixed names | `__version__` | Not a backend difference at all - a **pre-existing OSL bug** the equivalence harness surfaced. `oslquery.__version__` worked when importing the extension module straight out of the build tree, but was missing from every *installed* OSL, because the installed package wraps it in an `__init__.py` doing `import *`. Both `__init__.py` files now re-export it explicitly. |

### 4.2 Not applicable to OSL

`export_values()` on enums (OSL binds no enums); buffer protocol and the absence of
`nb::buffer` (no buffers); numpy array construction (no numpy); `half` having no
nanobind dtype (no `half` in any bound signature - `PyTypeForCType<half>` exists but is
never instantiated); assigning `None` to a property (no nullable properties);
`py::object` needing `py::borrow` when returned from a lambda (OSL's `py::object`
returns come from `C_to_val_or_tuple`, which constructs fresh objects); `gil_scoped_release`
(none used); `py::implicitly_convertible` (none used - and OIIO confirms nanobind
supports it anyway, so it was never a blocker).

---

## 5. Decisions

| Decision | Rationale | Alternatives rejected |
|---|---|---|
| Default backend `pybind11` | Zero disruption for existing builders and packagers; nanobind opt-in; mirrors OIIO | `both` - forces a nanobind dependency on everyone and doubles python build time. `nanobind` - makes any nanobind-only bug an immediate build break on day one. |
| Keep `Parameter.type` unchanged in both backends, no `#if` | The failure mode (needs OIIO's Python module, matching backend, compatible internals version) is pre-existing and identical under pybind11 today. Both frameworks resolve unregistered types at runtime, not compile time, so it compiles either way. | Removing it - a gratuitous breaking change. Guarding it pybind11-only - creates a real consumer-visible API difference between backends, violating the invariant this whole design rests on. |
| Skip OIIO's duplicate-then-unify detour | OIIO ended up deleting every duplicated file; on 350 lines the intermediate state has no value | Copying `src/python-nanobind/` wholesale as OIIO first did. |
| Call `add_one_testsuite()` directly for the python test instead of extending `TESTSUITE()` | `TESTSUITE()` has no `ENV` pass-through, and it currently also generates a `python-oslquery.rs_bitcode` variant of a test that executes no shader - pointless work that `both` mode would double | Adding an `ENV` multi-value argument to `TESTSUITE()` and threading it through its ~10 `add_one_testsuite` calls. |
| Set `PYTHONPATH` from CMake | It is set today only in `Makefile:293`, so bare `ctest` cannot run the python test at all, and per-backend selection is impossible without it | Continuing to rely on the Makefile - cannot express two different paths for two test variants. |
| Install `both`-mode nanobind package under a `nanobind/` subdir of the site dir | Avoids the `__init__.py` collision present in OIIO's `both` install | Copying OIIO exactly. |
| Use the existing `BUILD_LOCAL` machinery for nanobind | Already present in `src/cmake/dependency_utils.cmake` (lines 278-473, 605-800) with the same `${PROJECT_NAME}_LOCAL_DEPS_ROOT` variable OIIO uses; `build_nanobind.cmake` drops in unchanged | A `build_nanobind.bash` alongside `build_pybind11.bash` - kept as the fallback if `BUILD_LOCAL`, which OSL has never exercised, misbehaves. |

---

## 6. Open risks

1. **nanobind + `std::vector<Parameter>` through the list caster.** Mitigated by removing
   the `reference_internal` policies first. Fallback: convert to a list explicitly in
   the lambda, or bind the vector as an opaque type.
2. **`make_iterator` scope handle.** Must be a bound type. Fallback: implement `__iter__`
   as iteration over the already-converted list.
3. **First use of `BUILD_LOCAL` in OSL.** Fallback: a `build_nanobind.bash` script.
4. **Re-enabling the `for p in q:` loop** in `test_oslquery.py`, disabled since ~2020 by
   a `FIXME(pybind11)` about a macOS crash with pybind11 2.6 + Python 3.8/3.9. If it
   still crashes, leave it disabled and record it - it is a pre-existing condition, and
   the nanobind variant can still exercise iteration.
5. **Windows `both` mode** - two extension modules plus multi-config output directories.
