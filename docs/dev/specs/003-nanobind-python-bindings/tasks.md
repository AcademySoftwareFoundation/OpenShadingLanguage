# Tasks: nanobind Python bindings (dual-backend)

**Input**: Design documents from `docs/dev/specs/003-nanobind-python-bindings/`

**Branch**: `lg-nanobind`

## Format: `[ID] [P?] [Story?] Description with file path`

- **[P]**: Parallelizable - touches different files or non-overlapping sections
- **[Story]**: Which user story (US1-US5) from spec.md
- Each task is one logical change, reviewable as a single small diff

**Testing strategy**: There is exactly one Python test, `testsuite/python-oslquery`, and
it is shared verbatim by every backend. It is not modified after Phase 1. Every
subsequent phase is validated by that same test producing byte-identical output against
the same unmodified `ref/out.txt`. That identity *is* the API-equivalence proof
(SC-002, SC-003). Phases 1-4 keep the default backend at pybind11, so nothing can break
for anyone until it is explicitly asked for.

**Reference**: `~/code/oiio/oiio.lg` is the working implementation. When a task says
"mirror OIIO", read the cited file there first.

---

## Phase 1: Prep under pybind11 (no behavior change)

**Purpose**: Delete dead code, remove constructs that have no nanobind equivalent, and
turn on the iteration coverage - all while pybind11 is still the only backend, so each
step is independently verifiable.

**Rationale for each deletion**: research.md §1.

- [X] T001 [P] Delete the five `py::return_value_policy::reference_internal` arguments in `src/liboslquery/py_osl.cpp` (`Parameter.metadata` line 86, `OSLQuery.parameters` line 132, `OSLQuery.metadata` line 136, both `__getitem__` overloads lines 147 and 157). Every decorated lambda returns by value, so all five are already no-ops. Commit message must say so.
- [X] T002 Replace the value-returning factory constructor at `src/liboslquery/py_osl.cpp:98-102` with `.def(py::init<const std::string&, const std::string&>(), "shadername"_a, "searchpath"_a = "")`. `OSLQuery(string_view, string_view = {})` exists at `src/include/OSL/oslquery.h:118`; `std::string` converts implicitly. nanobind has no equivalent of pybind11's value-returning `py::init(lambda)`.
- [X] T003 [P] Delete the unused `#include <pybind11/embed.h>` at `src/liboslquery/py_osl.cpp:7`
- [X] T004 [P] Delete dead declarations `python_array_code()` and `typedesc_from_python_array_code()` (`src/liboslquery/py_osl.h:54-55`) - declared, defined nowhere in the repo, called nowhere
- [X] T005 [P] Delete unused `object_classname()` (`src/liboslquery/py_osl.h:58-62`) - also uses the `.cast<py::str>()` member syntax nanobind lacks
- [X] T006 [P] Delete the never-instantiated `C_to_tuple<TypeDesc>` specialization (`src/liboslquery/py_osl.h:107-116`) and the unused `<pybind11/numpy.h>` / `<pybind11/operators.h>` includes (`src/liboslquery/py_osl.h:32-33`)
- [X] T007 [US1] Fix the stale comment in `testsuite/python-oslquery/src/test_oslquery.py` that describes the printed type as "an OpenImageIO::TypeDesc but it can print like a string" - since 642ab36f the test prints `type_name`, a `str`
- [X] T008 [US1] Re-enable the `for p in q:` loop in `testsuite/python-oslquery/src/test_oslquery.py` (commented out at ~lines 65-72 with a 2020-era `FIXME(pybind11)` about a macOS crash under pybind11 2.6) and delete the `for i in range(len(q))` workaround. This is the only coverage of `__iter__` / `make_iterator` / `keep_alive<0,1>` (FR-024). `ref/out.txt` must not change. If the macOS crash recurs, revert to the disabled form and record it in plan.md's risk table - do not paper over it.

**Checkpoint**: `make test TEST=python-oslquery` passes against an unmodified
`testsuite/python-oslquery/ref/out.txt`. `git diff` shows no change to `ref/`.

---

## Phase 2: The compatibility shim (still pybind11-only)

**Purpose**: Introduce `py_backend.h` and route the binding source through it, while
`OSL_PY_BACKEND_NANOBIND` is never defined. Nothing changes behaviorally; this proves
the shim's pybind11 arm is correct before the nanobind arm is ever compiled.

- [X] T009 Create `src/liboslquery/py_backend.h` with **both** arms, modeled on `~/code/oiio/oiio.lg/src/python/py_backend.h`: `namespace py` alias, `py_module` typedef, `OSL_PY_RW` / `OSL_PY_PROP_RO` / `OSL_PY_PROP_RW`, `PY_STR`, and `namespace osl_py` with `make_tuple`, `make_iterator`, `throw_key_error`. Full mapping table in plan.md Phase 1. The nanobind arm is written now but not yet compiled.
- [X] T010 In `src/liboslquery/py_backend.h`, get the nanobind include set right: `<nanobind/nanobind.h>`, `<nanobind/make_iterator.h>`, `<nanobind/stl/string.h>`, `<nanobind/stl/vector.h>`. `stl/vector.h` is load-bearing - it is what converts `std::vector<Parameter>` for `.parameters` and `.metadata`, and omitting it fails at runtime, not compile time. Comment it as such.
- [X] T011 In `src/liboslquery/py_backend.h`, make `osl_py::make_iterator` take the scope as a template parameter and pass `py::type<Scope>()` under nanobind. It must be a **bound** type; `py::type<std::vector<Parameter>>()` would be null. Comment why.
- [X] T012 Rewrite `src/liboslquery/py_osl.h` to `#include "py_backend.h"` in place of the pybind11 include block, the `namespace py` alias, and the `PY_STR` define. Switch the two `C_to_tuple` bodies to `osl_py::make_tuple`. Change `declare_oslquery()` to take `py_module&`.
- [X] T013 Apply the macro substitution in `src/liboslquery/py_osl.cpp`: `def_readwrite`→`OSL_PY_RW`, `def_property_readonly`→`OSL_PY_PROP_RO`, `def_property`→`OSL_PY_PROP_RW`; `declare_oslqueryparam` / `declare_oslquery` take `py_module&`; `py::make_iterator(...)` → `osl_py::make_iterator<OSLQuery>(...)`; `throw py::key_error(...)` → `osl_py::throw_key_error(...)`
- [X] T014 Factor the eight `m.attr(...)` assignments at `src/liboslquery/py_osl.cpp:179-186` into `void declare_module_attributes(py_module& m)`, so each arm of the forthcoming module-entry `#if` is three lines
- [X] T015 Add the dual module entry point at the end of `src/liboslquery/py_osl.cpp` per plan.md Phase 1: `NB_MODULE` at global scope (namespace closed first - unlike `PYBIND11_MODULE`), `_oslquery` vs `oslquery` on `OSL_PY_NANOBIND_ISOLATED_PACKAGE`, and `py::set_leak_warnings(false)` guarded on `PY_VERSION_HEX < 0x030a0000` citing wjakob/nanobind#1405
- [X] T016 Verify the `#if` budget: **one** `OSL_PY_BACKEND_NANOBIND` site in `py_osl.cpp` (the module macro; the Python-version guard nested inside it is not backend-conditional) and zero in `py_osl.h` (FR-018, SC-005) - better than the budgeted two. The six conditionals in `py_backend.h` each carry a comment naming the framework difference that forces them. Equivalence checked by diffing a full public-surface + values + exception-type dump before and after: zero differences.

**Checkpoint**: default build unchanged; `make test TEST=python-oslquery` still passes
against the same `ref/out.txt`. `nm` or equivalent shows the same exported symbols.

---

## Phase 3: Build system - backend selection

**Purpose**: Add the option and dependency plumbing. Default stays `pybind11`, so a
build with no new flags is untouched.

- [X] T017 [US1] Add `OSL_PYTHON_BINDINGS_BACKEND` to `src/cmake/pythonutils.cmake`: `set_cache` (which routes through `set_utils.cmake:80 super_set` → `set_from_env`, making the same-named environment variable work - CI depends on that), `set_property(CACHE ... STRINGS pybind11 nanobind both)`, `string(TOLOWER ...)`, and a `FATAL_ERROR` naming the three accepted values (FR-001, FR-004)
- [X] T018 [US1] Derive `OSL_BUILD_PYTHON_PYBIND11` and `OSL_BUILD_PYTHON_NANOBIND` in `src/cmake/pythonutils.cmake`; these are what the rest of the build tests. Confirm `CMakeLists.txt:209 include (pythonutils)` still precedes `include (externalpackages)`.
- [X] T019 [US2] Add `discover_nanobind_cmake_dir()` to `src/cmake/pythonutils.cmake`, copied from `~/code/oiio/oiio.lg/src/cmake/pythonutils.cmake:112-145`. It runs `python -m nanobind --cmake_dir` to find nanobind installed as a pip/brew Python package (FR-006). Keep it a `function`, not a `macro` - carry over OIIO's comment explaining that a macro's `return()` would abort the including file.
- [X] T020 [US2] In `find_python()` (`src/cmake/pythonutils.cmake:17-55`), add a second `find_package (Python <maj>.<min> EXACT REQUIRED COMPONENTS ...)` when `OSL_BUILD_PYTHON_NANOBIND` - nanobind's CMake package wants the unversioned `Python::Module` target, not OSL's `Python3::*`
- [X] T021 [US1] Guard the existing pybind11 find on `OSL_BUILD_PYTHON_PYBIND11` and add the nanobind find in `src/cmake/externalpackages.cmake:89-93`: `checked_find_package (nanobind CONFIG REQUIRED VERSION_MIN 2.8.0 BUILD_LOCAL missing)` (FR-005)
- [X] T022 [US2] Create `src/cmake/build_nanobind.cmake`, copied from OIIO. Preserve both workarounds *with their comments*: the pre-clone plus `git submodule update --init --depth 1 -- ext/robin_map` (nanobind's CMakeLists hard-errors without it, and `build_dependency_with_cmake()`'s plain clone does not init submodules), and the `nanobind_DIR ... FORCE` pointing at `<prefix>/nanobind/cmake`. This is OSL's first use of `BUILD_LOCAL`. **Verified working** with `-DOSL_BUILD_LOCAL_DEPS=nanobind`: clone + submodule + configure + install + refind, about 1 second. The `build_nanobind.bash` fallback was not needed.

**Checkpoint**: `-DOSL_PYTHON_BINDINGS_BACKEND=garbage` fails at configure with a message
naming the three values (SC-007). Default configure/build/test is unchanged. Configuring
with `nanobind` finds or builds nanobind and reports which (FR-006).

---

## Phase 4: Build system - module targets

**Purpose**: Actually build the nanobind module.

- [X] T023 [US2] Add `setup_python_module_nanobind()` to `src/cmake/pythonutils.cmake`, beside the existing `setup_python_module()`. Differences: `nanobind_add_module()` takes no `${PYLIB_LIB_TYPE}` argument, and `target_compile_options (nanobind-static PUBLIC -Wno-format-nonliteral)` is needed for Clang/AppleClang/IntelLLVM because nanobind's own sources warn.
- [X] T024 [US2] Implement the single-backend output and install layout in `setup_python_module_nanobind()`: build tree `${CMAKE_BINARY_DIR}/lib/python/site-packages`, install `${PYTHON_SITE_DIR}/oslquery/` - byte-identical to what pybind11 produces today, so nanobind is a drop-in (FR-008, SC-001)
- [X] T025 [US3] Implement the `both`-mode layout in `setup_python_module_nanobind()`: build tree `${CMAKE_BINARY_DIR}/lib/python/nanobind/oslquery/`, install `${PYTHON_SITE_DIR}/nanobind/oslquery/`. **This deliberately deviates from OIIO**, which installs its `both`-mode `__init__.py` into the same site dir as pybind11's, where the two collide (FR-009, SC-010). Note the deviation in a comment.
- [X] T026 [US2] Add the nanobind block to `src/liboslquery/CMakeLists.txt`, over the *same* `file(GLOB py_*.cpp)` list as the pybind11 block. (The `if (OSL_BUILD_PYTHON_PYBIND11)` guard on the existing block landed in Phase 3, since searching for pybind11 conditionally while building against it unconditionally made `nanobind`-only fail at `pybind11_add_module`.) The nanobind block adds `-DOSL_PY_BACKEND_NANOBIND`, and `-DOSL_PY_NANOBIND_ISOLATED_PACKAGE` when the backend is `both`. One source list, two targets (`pyoslquery`, `pyoslquery_nanobind`) - FR-016.
- [X] T027 [US3] Create `src/liboslquery/nanobind/__init__.py`: `from ._oslquery import *`, the same Windows `add_dll_directory` block as `src/liboslquery/__init__.py`, and OIIO's trick of appending `Release`/`Debug`/`RelWithDebInfo`/`MinSizeRel` to `__path__` so the per-config `.pyd` resolves on MSVC multi-config builds
- [X] T028 [US3] `configure_file(nanobind/__init__.py ... COPYONLY)` into `${CMAKE_BINARY_DIR}/lib/python/nanobind/oslquery/` from `src/liboslquery/CMakeLists.txt`, and install it, in `both` mode only

**Checkpoint**: all three backend settings configure and build. Inspect the build tree
against the layout table in plan.md Phase 2. Tests are not yet wired up for nanobind -
that is Phase 5.

---

## Phase 5: Test registration

**Purpose**: Run the one existing test against every selected backend. This is where
US2 and US3 become verifiable.

- [X] T029 Add `osl_tests_pythonpath_env_entry(out_var prefix_dir)` to `src/cmake/testing.cmake`, mirroring `~/code/oiio/oiio.lg/src/cmake/testing.cmake:28`. Emits one `PYTHONPATH=<dir>:$ENV{PYTHONPATH}` string; on Windows uses `<dir>` alone, because semicolons get split by CMake list processing in a CTest `ENVIRONMENT` entry. Comment that.
- [X] T030 [US1] Replace the registration at `src/cmake/testing.cmake:474` with a direct `add_one_testsuite()` call for the pybind11 variant carrying `ENV TESTSHADE_OPT=0 "${_pybind_pypath}"`. This drops the pointless `python-oslquery.rs_bitcode` variant that `TESTSUITE()` was generating for a test that executes no shader, and fixes FR-023 (bare `ctest` previously could not run this test at all, since `PYTHONPATH` was set only in `Makefile:293`).
- [X] T031 [P] [US1] Remove the now-vestigial `NOOPTIMIZE` and `NOOPTIX` marker files from `testsuite/python-oslquery/`, and say in the commit that the `.rs_bitcode` variant is gone deliberately. (An existing build tree keeps a stale `build/testsuite/python-oslquery.rs_bitcode/` directory until it is cleaned; harmless, no longer a registered test.)
- [X] T032 [US2] [US3] Add the nanobind variant registration in `src/cmake/testing.cmake` with the empty-suffix trick from plan.md Phase 3: suffix is `.nanobind` only when the pybind11 variant was also registered, otherwise empty so the test keeps its historical name (FR-021, FR-022). Distinct names give distinct sandbox dirs automatically via `add_one_testsuite`'s `testdir` (`testing.cmake:37`), while `OSL_TESTSUITE_SRC` still points at the one source dir.
- [X] T033 [US1] Note in `Makefile` near line 293 that the `PYTHONPATH` prefix is retained for interactive use but that the CTest `ENVIRONMENT` property now takes precedence for the python tests

**Added in this phase (not originally planned)**: `testsuite/python-oslquery/src/test_oslquery.py`
gained an "Empty string properties" section, and `ref/out.txt` was regenerated. `printparam`
only ever reaches `structname` for parameters that *are* structs, so the empty-`ustring`
path had no coverage at all - which is why the `ustring::c_str()` bug in Phase 2 passed
the whole suite. Confirmed the new coverage fails before that fix and passes after, by
temporarily reintroducing the bug.

**Checkpoint**: this is the feature's main gate.

```bash
make OSL_PYTHON_BINDINGS_BACKEND=pybind11 && make test TEST=python-oslquery  # 1 test
make OSL_PYTHON_BINDINGS_BACKEND=nanobind && make test TEST=python-oslquery  # 1 test, same name
make OSL_PYTHON_BINDINGS_BACKEND=both     && make test TEST=python-oslquery  # 2 tests
```

All must diff clean against the single unmodified `ref/out.txt` (SC-002, SC-006).

---

## Phase 6: Equivalence verification

**Purpose**: Prove the claims the spec makes, rather than assuming them. Findings here
are bugs to fix, not results to record.

- [X] T034 [US2] Against a `both` build, compare `sorted(dir(oslquery.OSLQuery))`, `sorted(dir(oslquery.Parameter))`, and the sorted module attribute list between the two `PYTHONPATH`s. Zero differences required (FR-011, SC-003). If any appear, fix the binding - do not document the difference away. **Done via a 1279-observation dump** (surface + every property's type *and* value, via iteration, integer indexing, name indexing and the `parameters` property, plus metadata recursion, property setters and exception types). Zero differences, build tree and installed. Excluded: `_pybind11_conduit_v1_` and package-vs-bare-module artifacts, both documented in the harness.
- [X] T035 [US2] Verify every bound attribute returns the same Python type and value under both backends - spot-check `value` for the int, float, string, tuple, and `None` cases the test shader already covers (FR-012). Types are compared, not just values, so a `str`-vs-`bytes` style divergence would show. Found and fixed a real gap here: `__version__` was lost from *installed* packages (see research.md #16).
- [X] T036 [US2] Verify `__getitem__` raises `IndexError` for an out-of-range index and `KeyError` for an unknown name under both backends (FR-014)
- [X] T037 [US5] Verify `p.type` fails the same way under both backends when `OpenImageIO` has not been imported: a Python-level error, not a crash (FR-015)
- [X] T038 [US3] `cmake --build build --target install` for each backend; confirm the installed layout matches plan.md Phase 2's table and that the `both` install leaves both `__init__.py` files intact (SC-010)
- [X] T039 [P] Confirm a pybind11-backend build succeeds on a machine with no nanobind installed, and a nanobind-backend build succeeds with no pybind11 installed (SC-008, SC-009). Verified by poisoning the other package's `_DIR`/`_ROOT`: both configure and build their module, and the only log mentions of the absent package are CMake's unused-variable warnings.

---

## Phase 7: CI

- [X] T040 [P] Add a `python_bindings_backend` input (default `''`) to `.github/workflows/build-steps.yml`, exported as env `OSL_PYTHON_BINDINGS_BACKEND` next to `PYBIND11_VERSION` / `PYTHON_VERSION` at lines 135-136
- [X] T041 Turn that env var into a `-D` flag in `src/build-scripts/ci-build.bash`, after the `USE_SIMD` block
- [X] T042 Forward the input in `.github/workflows/ci.yml` at the ~5 job-level sites that already forward `pybind11_ver` (lines 66, 150, 435, 515, 585). Do not touch the ~18 matrix entries' `pybind11_ver` values.
- [X] T043 Set `python_bindings_backend: both` on exactly three `.github/workflows/ci.yml` matrix entries - one recent Linux, one macOS-arm, one Windows (SC-011, Constitution IV)
- [X] T044 [P] Add a conditional `brew install nanobind` to `src/build-scripts/install_homebrew_deps.bash` when the backend is `nanobind` or `both`
- [X] T045 [P] Add a pinned nanobind `pip install` to `src/build-scripts/gh-installdeps.bash` and `gh-win-installdeps.bash`, following OIIO's `ci-requirements-nanobind.txt` sha256-pinning pattern. Pinned to match `nanobind_BUILD_VERSION` (2.13.0 originally; bumped to 3.0.1 in T052); the hash was taken from PyPI and the pin verified by installing it into a throwaway venv and running `python -m nanobind --cmake_dir`.
  - **Caveat**: `gh-win-installdeps.bash` has a pre-existing bash syntax error at the `elif [[ "$LLVM_GOOGLE_DRIVE_ID" != "" ]] then` line (missing `;`), introduced by #2011 on 2025-07-28. `build-steps.yml:229` *executes* rather than sources that script, so bash fails to parse it and the whole step dies -- meaning the line added here can never run until that is fixed. Not fixed as part of this feature: it is unrelated, and unbreaking it may surface other Windows CI failures that would muddy this PR. Windows `both` still works regardless, because `BUILD_LOCAL missing` builds nanobind from source when the pip install hasn't happened.
- [X] T046 Separate commit: verify that `python-oslquery` now passes in the two ASWF-container jobs and remove the stale `CTEST_EXCLUSIONS` entries at `.github/workflows/ci.yml:199-201, 212-214`. Their comment says the exclusion is needed "until the ASWF container properly includes OIIO's python bindings"; since 642ab36f the test no longer imports OpenImageIO. This is an independent fix - if it turns out the exclusion is still needed for another reason, drop the task and say why. **Removed on the strength of reading, not running**: the ASWF containers can't be exercised locally, so the first CI run on this branch is what actually confirms it. Left alone: the optix-gpu job also excludes `python-oslquery`, but bundled with GPU-specific exclusions and without the OIIO rationale, so it is a separate question.

---

## Phase 8: Documentation

- [X] T047 [P] [US2] Add nanobind to `INSTALL.md:69-73` as a conditional dependency with its minimum version (Constitution IV requires documented dependency minimums), and document `OSL_PYTHON_BINDINGS_BACKEND` (FR-025). INSTALL.md turned out to have no build-options section, so the option is documented in a new "Python binding backends" section instead, which also carries the `Parameter.type` caveat (T048).
- [X] T048 [P] [US5] Document the `Parameter.type` interoperability constraint (FR-026): it returns an OIIO `TypeDesc`, so it needs OIIO's Python module - built with the same binding backend and a compatible internals/ABI version - to have been imported; `type_name` is the coupling-free alternative. Put it in `INSTALL.md` and as a comment at the binding site in `src/liboslquery/py_osl.cpp`.
- [~] T049 [P] ~~Add a `CHANGES.md` entry~~ **Deferred to release preparation** at the maintainer's direction -- CHANGES.md is written up as part of the release process, not per-PR.
- [X] T050 [P] Create `src/liboslquery/MIGRATION_STATUS.md` (FR-028), a short version of `~/code/oiio/oiio.lg/src/python/MIGRATION_STATUS.md`: which macros to use, `py_module&` rather than `py::module&`, keep `#if` sites minimal and commented, and the invariant - "consumer-visible differences: none intended; if you find one, it is a bug, add a regression test"

---

## Phase 9: Conditional default backend (2026-09-09)

Soak-time follow-on. The dual-backend machinery has been in place and green; make the
default select nanobind only when the build can actually use it, and refresh the
auto-built nanobind version.

- [X] T051 Make the `OSL_PYTHON_BINDINGS_BACKEND` default conditional. Cache default is now `""`; the normalize/validate/derive logic moved from file scope in `src/cmake/pythonutils.cmake` into a new `osl_resolve_python_bindings_backend()` macro. `find_python()` calls it right after `Python3` is found (and after OIIO, found earlier in `externalpackages.cmake`), before the nanobind-specific `find_package(Python)` and the pybind11/nanobind package finds. An empty selector resolves to `nanobind` when `OpenImageIO_VERSION >= 3.2` (OIIO defaults its own bindings to nanobind there) **and** `Python3_VERSION >= 3.10` (nanobind's minimum), else `pybind11`. It does not check whether nanobind is installed - the real `checked_find_package(nanobind ... BUILD_LOCAL missing)` builds it locally when absent (it is a ~1s header/CMake copy). Explicit `pybind11`/`nanobind`/`both` and the env var still honored. FR-002 updated in spec.md.
  - Depends on the `NO_FP_RANGE_CHECK` on the real `checked_find_package(nanobind ...)`: nanobind's config-version file is `SameMajorVersion`, so `VERSION_MIN 2.8.0` alone rejects an installed nanobind 3.x and forces a local build. `NO_FP_RANGE_CHECK` lets the real find accept the installed 3.x.
- [X] T052 Bump `nanobind_BUILD_VERSION` in `src/cmake/build_nanobind.cmake` from 2.13.0 to 3.0.1 (latest), with `nanobind_GIT_COMMIT` = `db4827f06f6f1680e5d4004c95fc8d69299dba8b` (peeled `v3.0.1` tag). Sync `src/build-scripts/ci-requirements-nanobind.txt` to `nanobind==3.0.1` with the PyPI wheel sha256. Verified locally: `BUILD_LOCAL` clone + configure + install of 3.0.1 succeeds, `pyoslquery_nanobind` builds clean, module imports. Homebrew nanobind is already 3.0.1.
- [X] T053 [P] Docs: INSTALL.md "tested through" bump to 3.0, the two-condition auto-select rule (OIIO >= 3.2 and Python >= 3.10) in the dependency list and the `Python binding backends` section, plan.md/spec.md consistency.
- [X] T054 CI: `src/build-scripts/ci-startup.bash` unconditionally exported `OSL_PYTHON_BINDINGS_BACKEND=both`, which forced nanobind (and its Python >= 3.10 requirement) onto the Python 3.9 jobs - `pip install nanobind==3.0.1` fails Requires-Python and the `BUILD_LOCAL` fallback then hits `nanobind-config.cmake` "requires Python 3.10 or newer". Fix: just stop setting it. ci.yml no longer forces a value either, so an unset backend lets `pythonutils.cmake` auto-select per job (pybind11 for old Python/OIIO, nanobind for modern). A ci.yml matrix entry can still set `python_bindings_backend` to force `both` (or either single backend) for a specific job.

---

## Dependencies

```
Phase 1 (T001-T008)  ── independent, lands first, verifiable with today's build
      ↓
Phase 2 (T009-T016)  ── needs T002 (no factory init) and T001 (no rv policies)
      ↓
Phase 3 (T017-T022)  ── option + deps; T021 needs T018's derived booleans
      ↓
Phase 4 (T023-T028)  ── needs Phase 2 (compilable nanobind arm) and Phase 3 (nanobind found)
      ↓
Phase 5 (T029-T033)  ── needs Phase 4 (modules exist to point PYTHONPATH at)
      ↓
Phase 6 (T034-T039)  ── verification; needs Phase 5
      ↓
Phase 7 (T040-T046)  ── CI; needs Phase 5 green locally.  T046 is independent of everything else.
Phase 8 (T047-T050)  ── docs; can start any time after Phase 3 fixes the option's name
```

Phases 7 and 8 are mutually independent and can proceed in parallel.

## Incremental delivery

- **After Phase 1**: dead code gone, iteration covered. Shippable on its own.
- **After Phase 2**: bindings are backend-neutral source; still pybind11-only in
  practice. Shippable on its own.
- **After Phase 5**: US1, US2, and US3 all satisfied - this is the feature's MVP.
- **After Phase 8**: complete.

## Out of scope (restated from spec.md)

Removing pybind11 support; any Python API change; binding any other OSL class; type
stubs; wheels. (The default backend became conditional - nanobind when the build can
use it, else pybind11 - on 2026-09-09; see Phase 9.)
