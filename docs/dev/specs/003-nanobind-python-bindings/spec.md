# Feature Specification: nanobind Python bindings (dual-backend)

**Feature Branch**: `lg-nanobind`

**Created**: 2026-08-02

**Status**: Draft

**Input**: User description: "For OSL's python bindings, I want to switch from pybind11
to nanobind. Look at OpenImageIO (~/code/oiio/oiio.lg), which recently added nanobind
bindings. Salient features: (1) for now both bindings are built, controlled by a
build-time switch that selects pybind11, nanobind, or both; (2) all tests run for either
binding (including both, when 'both' is selected); (3) minimize the amount of code
duplicated for the two bindings (sometimes by using macros that are defined differently
for both). Devise a plan for a similar binding conversion for OSL. It should be a much
smaller task (since it's only the OSLQuery class that we make python bindings for), but
I want it in an analogous style and using a similar approach."

## Overview

OSL exposes exactly one class family to Python: `OSLQuery` and `OSLQuery::Parameter`,
bound with pybind11 in `src/liboslquery/py_osl.{h,cpp}` (~350 lines, ~120 of them live
code). OpenImageIO has migrated to nanobind while keeping pybind11 available behind a
build-time selector, so that both modules can be built from a single set of sources and
both can be tested. OSL should adopt the same structure.

This matters beyond OSL's own modernization: OSL's `Parameter.type` returns an OIIO
`TypeDesc`, whose Python binding is registered by *OIIO's* module in a process-wide
type registry that is shared only among extension modules using the same binding
framework and a compatible internals/ABI version. Once OIIO's Python module is built
with nanobind, an OSL module built with pybind11 can no longer see that registration.
Supporting both backends in OSL lets packagers match whichever OIIO they ship.

Commit 642ab36f already removed the hard dependency: `Parameter.type_name` returns a
plain string covering every use `type` was needed for, and the forced
`import OpenImageIO` at module init is gone.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build OSL exactly as before (Priority: P1)

An existing OSL builder, packager, or CI job configures and builds OSL with no new
flags. Everything about the Python module - its name, contents, behavior, build-tree
location, install location, and test name - is unchanged from today.

**Why this priority**: Non-negotiable. This change must be invisible by default. If the
default path regresses, nothing else in the feature matters.

**Independent Test**: Configure and build with no new options; run the `python-oslquery`
test. It must pass against the unchanged reference output, and the built module must
land in the same paths with the same file names as before the change.

**Acceptance Scenarios**:

1. **Given** a checkout with no new CMake variables set, **When** OSL is configured and
   built, **Then** exactly one Python module is produced, at the same build-tree and
   install paths as before, and no new dependency is required.
2. **Given** that default build, **When** the test suite runs, **Then** a single test
   named `python-oslquery` runs and diffs clean against the existing reference output.
3. **Given** that default build, **When** a user imports the module and exercises
   `OSLQuery`, **Then** every attribute and method available before the change is still
   available with identical behavior.

---

### User Story 2 - Build the nanobind module instead (Priority: P1)

A builder who has moved to a nanobind-based OpenImageIO selects the nanobind backend.
They get a drop-in replacement module: same import name, same public API, same install
location, same test name and reference output.

**Why this priority**: This is the actual goal of the feature. Without it the work
delivers nothing.

**Independent Test**: Configure with the backend set to nanobind, build, and run the
test suite. The same single test name appears and passes against the same reference
output; pybind11 is not required for the build.

**Acceptance Scenarios**:

1. **Given** the nanobind backend is selected, **When** OSL is configured, **Then**
   pybind11 is not searched for and is not required.
2. **Given** the nanobind backend is selected, **When** OSL is built and installed,
   **Then** the Python module has the same import name and occupies the same install
   location as the pybind11 module would have.
3. **Given** a nanobind build, **When** the same test script that runs against the
   pybind11 module is run, **Then** its output is identical.
4. **Given** a nanobind build, **When** a user inspects the module's public surface,
   **Then** it matches the pybind11 module's public surface exactly.

---

### User Story 3 - Build and test both backends at once (Priority: P2)

A maintainer or CI job selects "both". Two Python modules are produced, they do not
collide in the build tree or on install, and the full Python test suite runs twice -
once against each - from a single copy of the test sources and a single reference
output.

**Why this priority**: This is how the equivalence claimed by Stories 1 and 2 is
actually enforced. It is P2 only because Stories 1 and 2 are individually shippable
without it.

**Independent Test**: Configure with "both", build, and list the registered tests. Two
Python test entries appear, distinguished by a suffix, and both pass.

**Acceptance Scenarios**:

1. **Given** the "both" backend selection, **When** OSL is built, **Then** two Python
   extension modules are produced in locations that do not overwrite each other.
2. **Given** the "both" backend selection, **When** the test suite runs, **Then** the
   Python test appears twice - once under its original name and once under a
   nanobind-marked name - and both diff clean against the same single reference output.
3. **Given** the "both" backend selection, **When** each module is imported in
   isolation, **Then** each is importable under the project's normal Python import name
   without the other being present.
4. **Given** the "both" backend selection, **When** OSL is installed, **Then** neither
   backend's package files overwrite the other's.

---

### User Story 4 - Maintain the bindings without writing them twice (Priority: P2)

A maintainer adds or changes a binding. They edit one place, and both backends pick up
the change.

**Why this priority**: This is the sustainability requirement that makes carrying two
backends acceptable at all. Without it, the dual-backend period becomes a permanent
tax.

**Independent Test**: Count the binding source files and the backend-conditional
compilation sites. There must be exactly one set of binding sources, and the number of
backend-conditional sites must be countable on one hand and each justified by a genuine
framework difference.

**Acceptance Scenarios**:

1. **Given** the completed feature, **When** the binding sources are inspected, **Then**
   there is exactly one copy of the binding code, compiled once per selected backend.
2. **Given** a new attribute added to the bound class in the single source, **When**
   "both" is built, **Then** the attribute appears in both modules with no
   backend-specific code.
3. **Given** the completed feature, **When** backend-conditional regions are counted,
   **Then** there are at most three, each with a comment naming the framework
   difference that forces it.

---

### User Story 5 - Understand the OIIO interoperability constraint (Priority: P3)

A user hits an error accessing `Parameter.type` and finds documentation explaining why,
and what to use instead.

**Why this priority**: Documentation-only, and the failure mode is pre-existing rather
than introduced here - but the feature adds one more way to trigger it, so it should be
written down now.

**Independent Test**: Read the documentation and confirm it states the constraint and
names the alternative.

**Acceptance Scenarios**:

1. **Given** the documentation, **When** a user searches for the type-related attribute,
   **Then** they find a statement that it requires OIIO's Python module, built with the
   same binding backend, to have been imported first.
2. **Given** that same documentation, **When** the user looks for an alternative,
   **Then** the string-valued attribute is named as the coupling-free option.

---

### Edge Cases

- **Invalid backend selection**: an unrecognized value must fail at configure time with
  a message listing the accepted values, not fail later at compile or link time.
- **Case variation**: `Both`, `NANOBIND`, etc. must be accepted, matching how the rest
  of the project's string-valued options behave.
- **nanobind not installed**: when the nanobind backend is requested and nanobind is not
  present, the build must either locate it automatically (including when it was
  installed as a Python package rather than a system package) or build it locally, and
  must say clearly which it did.
- **Python bindings disabled entirely**: with Python support turned off, the backend
  selection must have no effect and must not cause a dependency search.
- **Sanitizer builds**: the Python test is already skipped under sanitizers; that must
  remain true for every backend selection.
- **Both backends installed into the same prefix**: package initialization files for the
  two backends must not overwrite one another.
- **Accessing the OIIO-typed attribute with no OIIO Python module imported**: must
  behave the same way under both backends - a Python-level error, not a crash.
- **Multi-configuration builds on Windows**: the "both" mode's second module must remain
  importable even though its binary lands in a per-configuration subdirectory.

## Requirements *(mandatory)*

### Functional Requirements

**Backend selection**

- **FR-001**: The build MUST provide a single user-facing option that selects the Python
  binding backend, accepting exactly three values: pybind11, nanobind, or both.
- **FR-002**: When the option is unset, the build MUST auto-select nanobind when
  OpenImageIO is >= 3.2 (which defaults its own Python bindings to nanobind - matching
  it keeps `Parameter.type` working) and Python is >= 3.10 (nanobind's minimum), and
  pybind11 otherwise. A missing nanobind is built locally by the normal dependency
  machinery, so the auto-selection does not depend on nanobind being pre-installed.
  Any backend produces a Python module with the same public surface, return values,
  and exception types; only the binding framework differs. (Updated 2026-09-09: the
  feature originally shipped with a fixed pybind11 default; after soak time this
  became the conditional auto-select.)
- **FR-003**: The option MUST be settable from the environment as well as from the
  command line, matching the convention used by the project's other cached string
  options, so continuous integration can drive it without editing build scripts.
- **FR-004**: The option MUST be case-insensitive and MUST reject any other value at
  configure time with an error naming the accepted values.
- **FR-005**: The build MUST search for pybind11 only when pybind11 is among the
  selected backends, and for nanobind only when nanobind is among the selected backends.
- **FR-006**: When nanobind is selected and is not already discoverable, the build MUST
  attempt discovery via the active Python environment before falling back to building
  it locally, and MUST report which path it took.
- **FR-007**: When Python bindings are disabled, the backend option MUST have no effect
  and MUST trigger no dependency search.

**Module identity and layout**

- **FR-008**: When exactly one backend is selected, the resulting module MUST use the
  project's established Python import name and MUST occupy the same build-tree and
  install locations that the pybind11 module occupies today.
- **FR-009**: When both backends are selected, the two modules MUST be placed so that
  neither overwrites the other in the build tree or on install, including their package
  initialization files.
- **FR-010**: When both backends are selected, each module MUST still be importable
  under the project's established import name, by putting the appropriate directory on
  the Python module search path.

**API equivalence**

- **FR-011**: The nanobind module MUST expose the identical public surface as the
  pybind11 module: the same module-level attributes, the same classes, and the same
  attributes and methods on each class, with the same names. Members injected by the
  binding framework itself, which are not part of OSL's API, are excluded.
- **FR-012**: Every bound attribute and method MUST return values of the same Python
  types, with the same values, under both backends.
- **FR-013**: Both modules MUST support iteration over a query object's parameters,
  index-based and name-based item lookup, and length.
- **FR-014**: Out-of-range index lookup and unknown-name lookup MUST raise the same
  Python exception types under both backends.
- **FR-015**: The attribute that exposes an OpenImageIO type object MUST be present in
  both backends' modules and MUST fail in the same manner - a Python-level error, not a
  crash - when OIIO's Python module of the matching backend has not been imported.

**Single-source maintenance**

- **FR-016**: There MUST be exactly one copy of the binding source; the same sources MUST
  be compiled once per selected backend rather than duplicated per backend.
- **FR-017**: Framework differences MUST be absorbed by a single compatibility header
  that defines the same names differently per backend.
- **FR-018**: Backend-conditional compilation outside that compatibility header MUST be
  limited to at most three sites, each accompanied by a comment naming the framework
  difference that requires it.

**Testing**

- **FR-019**: The Python test MUST run against every selected backend, using one shared
  copy of the test script and one shared reference output.
- **FR-020**: The test script MUST contain no backend-specific logic; backend selection
  MUST be entirely a matter of which directory is on the Python module search path.
- **FR-021**: When both backends are selected, the two test runs MUST be registered
  under distinct names and MUST execute in distinct working directories.
- **FR-022**: When only one backend is selected, its test MUST be registered under the
  test's original name, regardless of which backend it is.
- **FR-023**: The build MUST configure the Python module search path for these tests
  itself, so the tests pass when run directly by the test driver rather than only
  through the project's convenience wrapper.
- **FR-024**: Iteration over a query object's parameters MUST be exercised by the test
  suite under every selected backend.

**Documentation and continuous integration**

- **FR-025**: Installation documentation MUST list nanobind as a conditional dependency
  and MUST document the backend option and its accepted values.
- **FR-026**: Documentation MUST state the OpenImageIO type-object interoperability
  constraint and name the string-valued alternative.
- **FR-027**: Continuous integration MUST exercise the both-backends configuration on at
  least one job per major platform, while the remaining jobs continue to exercise the
  default.
- **FR-028**: A maintainer-facing note MUST record the conventions for keeping the two
  backends in sync, including the principle that any consumer-visible difference between
  the backends is a bug requiring a regression test.

### Key Entities

- **Binding backend selection**: a build-time choice among pybind11, nanobind, and both;
  drives dependency discovery, which module targets are built, where they are placed,
  and which test variants are registered.
- **Compatibility layer**: the single header that maps one set of binding spellings onto
  either framework, so the binding source itself is backend-neutral.
- **Python module**: the extension module exposing the shader-query API; identified by
  its import name, its location on the module search path, and its public surface.
- **Test variant**: one registered execution of the shared Python test script against
  one backend, distinguished by name, working directory, and module search path.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: With the default configuration, the Python module's public surface, its
  file names, and its build-tree and install locations are identical to those produced
  before this change - verified by direct comparison, with zero differences.
- **SC-002**: The Python test's reference output is unchanged, and all backend
  configurations diff clean against that same single reference file.
- **SC-003**: The full public surface of the two modules - module attributes, class
  names, and per-class attributes and methods - compares equal, with zero differences,
  excluding members the binding framework injects on its own behalf (pybind11 3.x adds
  `_pybind11_conduit_v1_` to every class it binds; nanobind adds nothing comparable).
- **SC-004**: There is exactly one copy of the binding source, and it is compiled once
  per selected backend.
- **SC-005**: Backend-conditional compilation appears at no more than three sites
  outside the compatibility header.
- **SC-006**: Selecting a single backend yields exactly one Python test entry; selecting
  both yields exactly two; all of them pass.
- **SC-007**: Configuring with an unrecognized backend value fails immediately with a
  message naming the three accepted values.
- **SC-008**: A build with the nanobind backend selected completes on a machine with no
  pybind11 installed.
- **SC-009**: A build with the pybind11 backend selected completes on a machine with no
  nanobind installed.
- **SC-010**: Installing a both-backends build leaves both backends' package files
  intact, with neither having overwritten the other.
- **SC-011**: Continuous integration runs the both-backends configuration on at least
  one Linux, one macOS, and one Windows job, and those jobs pass.
- **SC-012**: Iteration over a query object's parameters is covered by the test suite
  under every selected backend.

## Assumptions

- The public Python API is exactly what `OSLQuery` and `OSLQuery::Parameter` expose
  today. No new API is added, and nothing is removed, by this feature.
- `Parameter.type` is retained unchanged in both backends. Its dependence on OIIO's
  Python module having been imported, with a matching binding framework, is a
  pre-existing condition (it predates this feature and predates commit 642ab36f's
  removal of the forced OIIO import) and is addressed by documentation rather than by
  code. `Parameter.type_name` is the supported coupling-free alternative.
- The default backend is auto-selected (as of 2026-09-09): nanobind when OIIO >= 3.2
  and Python >= 3.10, pybind11 otherwise. Removing pybind11 support entirely is a
  separate future change, still gated on soak time.
- OpenImageIO's dual-backend implementation is the reference for structure, naming, and
  known framework differences. Deviating from it requires a stated reason.
- The project's existing local-dependency-build machinery is available for nanobind; if
  it proves unsuitable, a build script in the style of the existing pybind11 one is an
  acceptable fallback.
- Python and C++ language-level minimums (Python 3.9, C++17) already satisfy nanobind's
  requirements, so no minimum-version changes are needed.
- Type stub files are out of scope; OSL ships none today.
- Python packaging (wheels) is out of scope; OSL has no wheel build.
- Bindings for `ShadingSystem` or any other OSL class remain out of scope, as they are
  today.

## Out of Scope

- Removing pybind11 support.
- Adding, removing, or changing any Python-visible API.
- Binding any OSL class other than `OSLQuery` and `OSLQuery::Parameter`.
- Generating type stubs or building Python wheels.
- Changing how OpenImageIO builds its own bindings.
