# OSL Agent Guide

This file provides guidance to coding assistants when working with code in
this repository.

## Project Overview

Open Shading Language (OSL) is a production shading language for VFX/animation
renderers, maintained by the Academy Software Foundation. It compiles `.osl`
shader source to `.oso` bytecode, then JIT-compiles via LLVM to native code at
render time. C++17 codebase.


## Repo map

**Core libraries**

- `src/liboslcomp/` — Shader compiler. Flex/Bison lexer+parser → AST → `.oso`
  bytecode. Entry point: `oslcomp.h`
- `src/liboslexec/` — Shader execution engine. Loads `.oso`, optimizes shader
  groups, JIT-compiles to native via LLVM. Key files: `backendllvm.cpp`,
  `llvm_gen.cpp`, `llvm_ops.cpp`, `instance.cpp`. Contains `wide/`
  subdirectory for SIMD batched execution (SSE2/AVX/AVX-512)
- `src/liboslquery/` — Query compiled shader metadata and parameters
- `src/liboslnoise/` — Noise function implementations
- `src/libbsdl/` — BSDF/closure library

**Command line tools**

- `src/oslc/` — Compiler CLI (`oslc`)
- `src/oslinfo/` — Shader info query tool (`oslinfo`)
- `src/testshade/` — Test harness that executes shaders on rectangular arrays
  of points, saves output as images.
- `src/testrender/` — Minimal path-tracing renderer demonstrating OSL
  integration.
- `src/osltoy/` — Qt-based GUI shader editor (optional)

**Public APIs/headers**

Headers in `src/include/OSL/`: `oslexec.h`, `oslcomp.h`, `oslquery.h`, `rendererservices.h` (abstract interface renderers must implement).

Namespaces:
- `OSL` (outer), `OSL::pvt` (private implementation of internal-only utilities)
- Versioned inner namespace: `v{MAJOR}_{MINOR}`


## Build and verification

Common build commands via the Makefile convenience wrapper:
```bash
make                        # configure, build, install (Release)
make debug                  # debug build
make clean                  # wipe build dir (needed when switching branches/modes)
make clang-format           # format all source files
make test                   # full testsuite
make test TEST=<pattern>    # subset matching regex
```

Or directly with cmake:
```bash
cmake -B build -S .
cmake --build build --target install
ctest --test-dir build -R <pattern> --output-on-failure
```

By default, builds into `./build` and installs into `./dist`.

## Testsuite notes

- Test output lands in `build/testsuite/<testname>/`; references in
  `testsuite/<testname>/ref/`
- Read `testsuite/TESTSUITE-README.md` before updating references or
  diagnosing failures
- For platform-specific diffs, add a variant ref (e.g. `out-win.txt`) rather
  than overwriting
- Be conservative loosening image diff thresholds — use the minimum needed
- Check uploaded CI artifacts before changing references when local
  reproduction is unclear

## Code formatting and file conventions

- `clang-format` enforced (`.clang-format`); CI rejects non-conforming code —
  run `make clang-format` before committing
- Lines ~80 cols; ASCII only in code and comments; `#pragma once` for headers
- New files: standard copyright + SPDX notice
- `CamelCase` classes, `snake_case` locals, `ALL_CAPS` macros, `m_foo` private
  members
- 3 blank lines between free functions/classes; 1 between class methods; max 1
  blank line inside a function body
- `//` for regular comments; `///` Doxygen for public API
- If a file has a strong local style, imitate that.


## C++ guidelines

- Error handling: prefer `bool` returns, explicit status propagation, and
  `errorfmt()`-style reporting — not exceptions — consistent with the
  surrounding subsystem
- Preserve API, ABI, and behavior compatibility unless the task explicitly
  requires a break
- For hot paths: no hidden allocation in inner loops or parallel regions;
  precompute outside hot paths


## OIIO utility types (prefer over raw C++ equivalents)

The [OpenImageIO](http://github.com/AcademySoftwareFoundation/OpenImageIO) (OIIO) project is a key dependency of OSL and the two projects are co-developed. Many utilities shared by both projects live in OpenImageIO.

Prefer C++17 `std` and Imath types except as noted below. Avoid introducing
new third-party dependencies without a strong reason. Rely liberally on these
classes and headers from OIIO:

- `OIIO::TypeDesc` - light-weight description of simple types (aliased
  as `OSL::TypeDesc`)
- `OIIO::ustring` and `OIIO::ustringhash` - light-weight unique string heavily
  used in OSL. (Aliased as `OSL::ustring` and `OSL::ustringhash`).
- `OIIO::string_view` — non-owning string/`char*` (like `std::string_view`).
  Aliased as `OSL::string_view`.
- `OIIO::span` / `OIIO::cspan` — non-owning contiguous data (like
  `std::span`); use instead of pointer+length pairs. Aliased as `OSL::span`
  and `OSL::cspan`.
- `OpenImageIO/strutil.h` : string processing utilities
- `OIIO::Strutil::print()` , also aliased as `OSL::print` — string
  formatting/output; **never** use `printf` or `<<` streams
- `OpenImageIO/fmath.h` — fast/safe math, avoids NaN/Inf
- `OIIO::Filesystem::*` in `OpenImageIO/filesystem.h` — file/directory
  utilities
- `OpenImageIO/simd.h` — SIMD helpers
- `OpenImageIO/parallel.h` — parallel loops, etc.
- `OpenImageIO/unittest.h` — unit test macros


## Safe programming in C++

- Try to avoid passing raw pointers as function arguments.
- Use `std::unique_ptr` and `std::shared_ptr` rather than raw pointers when
  ownership must be expressed.
- Prefer `OSL::string_view` when passing non-mutable strings or C-style
  `char*` strings.
- Prefer `OSL::span` rather than passing raw pointers + a separate length, or
  passing a raw pointer with an implied (but not explicitly passed) length.
  `OSL::cspan` is a synonmym when the underlying data is const/non-mutable.
  `OSL::span<std::byte>` or `OSL::cspan<std::byte>` can be used to represent
  contiguous untyped data. These are our equivalent of C++ `std::span`.
- Use these guidelines always for new code, but do not churn existing code
  just to "modernize" it.

## Architecture

### Compilation Pipeline

OSL source → (Flex/Bison) → AST → (liboslcomp) → `.oso` bytecode → (liboslexec) → LLVM IR → JIT native code


## Code Style

- clang-format config in `.clang-format` (WebKit-based, 80-char line limit, 4-space indent)
- Run `make clang-format` before submitting changes

## Key Dependencies

- **LLVM 14+** (JIT compilation), **OpenImageIO 2.5+** (textures, image I/O, utilities), **Imath 3.1+** (math types), **Flex/Bison** (parser generation), **pybind11** (Python bindings, optional)

## Commits and PRs

- Keep PRs narrow and easy to review.
- Prefix format: `type(subsystem): message` — subsystem is optional but helpful.
- Valid types: `fix:`, `feat:`, `perf:`, `api:`, `int:`, `build:`, `test:`,
  `ci:`, `docs:`, `refactor:`, `style:`, `admin:`, `revert:`.
- Add a subsystem tag when it helps, e.g. `fix(exr):` or `perf(IBA):`.
- Write commit messages and PR descriptions that explain why the change is
  needed, what behavior changes, and any non-obvious implementation choices.

## Spec-driven design

Feature specifications live in `docs/dev/specs/` (not at the project root).
When using speckit commands (`/speckit-specify`, `/speckit-plan`, etc.), always
use `docs/dev/specs/` as the base directory for all spec paths, overriding any
default of `specs/`. When creating a new spec, set `SPECIFY_FEATURE_DIRECTORY`
to `docs/dev/specs/<NNN-feature-name>` and write that path to
`.specify/feature.json`.

The speckit bash scripts expect a `specs/` directory at the project root. A
symlink satisfies this without committing speckit infrastructure to the repo.
This symlink is set up by the setup-agents script, and is not committed to
the repo. All saved specs live in `docs/dev/specs`.

## AI policy

Refer to `docs/dev/AI_Policy.md`.

See `docs/dev/AI_Policy.md`. Key rule: if AI assistance contributed materially
to a patch, the commit must include `Assisted-by: <TOOL> / <MODEL>`. The human
author is responsible for understanding, testing, and defending all changes.

## References

- `CONTRIBUTING.md` : general contribution guidelines and recap of coding
  conventions.
