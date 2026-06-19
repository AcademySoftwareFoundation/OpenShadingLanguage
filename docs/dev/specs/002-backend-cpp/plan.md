# Implementation Plan: BackendCpp — C++ Source Code Generation Backend

**Branch**: `002-backend-cpp` | **Date**: 2026-05-26 | **Spec**: [spec.md](spec.md)

**Input**: [docs/dev/specs/002-backend-cpp/spec.md](spec.md)

## Summary

Complete the `BackendCpp` path that generates human-readable, compilable C++ from post-optimized OSL shader groups. A partial skeleton (`backendcpp.h`, `backendcpp.cpp`) already exists; this plan fills in the generated file structure, wires attribute plumbing, adds compile/load/execute stages, and extends op coverage — all in small, independently reviewable increments.

## Technical Context

**Language/Version**: C++17 (OSL minimum), targeting C++17 ABI-stable generated code

**Primary Dependencies**: OSL internals — `OSOProcessorBase`, `ShaderGroup`, `ShadingSystemImpl`, `BackendLLVM` (layout pass), `OIIO::Filesystem`, `OIIO::Strutil::fmt`, `OIIO::Plugin` (platform-independent DSO load/unload)

**Storage**: Generated `.cpp` and `.so`/`.dylib`/`.dll` files in `cpp_output_dir`

**Testing**: Existing `testsuite/` + `testshade`; new `testsuite/backend-cpp/` entry; `ctest` with `OSL_DEBUG_OUTPUT_CPP=3`

**Target Platform**: Linux (GCC/Clang), macOS (Clang), Windows (MSVC) — all three CI platforms

**Project Type**: Compiler backend / debug/alternate-execution path within a C++ library

**Performance Goals**: Generation step (`debug_output_cpp=1`) adds no perceptible latency to shader compilation. Compile/load steps are debug-path only and have no hot-path budget.

**Constraints**: Generated code must be ABI-stable relative to the `RunLLVMGroupFunc` calling convention and `ShaderGlobals` layout. No public API changes. No changes to JIT path behavior.

## Constitution Check

| Gate | Status | Notes |
|------|--------|-------|
| **I. Backward Compatibility** | PASS | No public header changes. New ShadingSystem attributes are additive. `debug_output_cpp` attribute already exists (bool→int is a compatible widening via ATTR_SET). |
| **II. Physical Accuracy** | PASS | C++ path must match JIT output within FP tolerance (SC-003). Discrepancies are bugs, not accepted divergence. |
| **III. Test-Driven Quality** | PASS | New `testsuite/backend-cpp/` entry required. Testsuite-wide `OSL_DEBUG_OUTPUT_CPP=3` run validates parity. |
| **IV. Cross-Platform Portability** | PASS (with caveat) | DSO loading uses `OIIO::Plugin` (platform-independent `dlopen`/`LoadLibrary`). Compiler/flags baked in at CMake configure time. **Caveat:** the C++-backend *test path* (`OSL_TEST_CPP_BACKEND`) runs in CI only on the Linux and macOS variants; the generated-code runtime is made MSVC-compile-clean (T055) but is **not executed in Windows CI**. The normal (JIT) build/test still covers Windows on every PR, so the constitution's all-platform CI requirement holds for the project; only the opt-in cpp debug path is Windows-unverified. |
| **V. Performance** | PASS | Entire feature is gated behind `debug_output_cpp != 0`; zero overhead when disabled. |

## Project Structure

### Documentation (this feature)

```text
docs/dev/specs/002-backend-cpp/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit-tasks)
```

### Source Code (relevant files)

```text
src/liboslexec/
├── backendcpp.h          # BackendCpp class — extend with new members
├── backendcpp.cpp        # BackendCpp::run(), op generators — primary work file
├── backendllvm.h         # Read-only reference: layout pass, groupdata size
├── llvm_instance.cpp     # Read-only reference: llvm_type_groupdata() layout logic
├── llvm_ops.cpp          # Symbol visibility: osl_* must be exported (not DLL_LOCAL)
├── oslexec_pvt.h         # Add int attrs, OSL_CPP_ABI_VERSION, DSO handle on ShaderGroup
├── shadingsys.cpp        # Add env var, attr registration, BackendCpp invocation, JIT-skip branch
├── instance.cpp          # ShaderGroup DSO-handle lifecycle
├── context.cpp           # execute via the compiled group entry when present
└── osl_cpp_runtime.h     # NEW: OSL_CPP_ABI_VERSION, extern "C" osl_* declarations

src/build-scripts/
└── hidesymbols.map       # export osl_* on Linux (global:) for generated DSOs

src/include/OSL/
└── (no public API changes)

testsuite/
└── backend-cpp/          # NEW: reference test for C++ path correctness
```

> **Note:** the JIT-skip (FR-016c) is in `shadingsys.cpp`, not a separate
> `shadergroupopt.cpp`. File lists here are indicative of the design; see the
> branch diff for the authoritative set of touched files.

## Design Constraint: Subclassability for Future Language Backends

BackendCpp must be structured so that future backends targeting similar-to-C++ languages can subclass it and override only the language-specific leaf methods. The traversal logic and op dispatch are shared; only the emission seam points are virtual:

- **Type name mapping** — `lang_type_name(TypeDesc)`, `lang_sym_type_name(Symbol&)` (virtual; C++ backend provides concrete implementations)
- **Language preamble** — `lang_preamble()` (virtual; emits `#include` directives and any language-specific header boilerplate)
- **Function qualifier** — `lang_function_qualifier()` (virtual; empty string for C++)
- **Linkage prefix** — `lang_linkage_prefix()` (virtual; `extern "C"` for C++)
- **File extension** — `lang_file_extension()` (virtual; `".cpp"` for C++)
- **Pointer syntax** — `lang_ptr_syntax()` (virtual; `"*"` for C++)

The existing `cpp_typedesc_name()` and `cpp_sym_type_name()` methods ARE these virtual methods — they must be renamed to `lang_*` and made `virtual` in Phase 4. The `cpp_` prefix would be misleading in subclasses.

No subclasses are implemented in this feature. The sole obligation is that Phase 4 establishes the virtual interface correctly.

## Implementation Phases

> **Ordering principle**: Each phase produces a reviewable diff. Phases 1–4 build toward a compilable generated file. Phases 5–6 add compile and load. Phase 7 fixes symbol visibility. Phases 8–9 extend op coverage incrementally. Phase 10 adds test infrastructure.

---

### Phase 1 — Attribute Plumbing

**Goal**: Wire the four new ShadingSystem attributes, change `debug_output_cpp` from `bool` to `int`, add `OSL_DEBUG_OUTPUT_CPP` env var, and write the `.cpp` file to `cpp_output_dir` instead of CWD.

**Files changed**:
- `src/liboslexec/oslexec_pvt.h` — change `m_debug_output_cpp` from `bool` to `int`; add `m_cpp_output_dir`, `m_cpp_compiler`, `m_cpp_compiler_flags` string members
- `src/liboslexec/shadingsys.cpp` — update `ATTR_SET`/`ATTR_DECODE` for int type; register three new string attributes; add `OSL_DEBUG_OUTPUT_CPP` env var read in constructor (pattern: same as `OSL_LLVM_DEBUG` at line 1243); update BackendCpp invocation to write to `cpp_output_dir`

**Acceptance**: `testshade --options debug_output_cpp=1,cpp_output_dir=/tmp/osl-cpp` writes `group-cpp-<name>.cpp` to `/tmp/osl-cpp/`. Existing bool behavior (0=off, non-zero=on) unchanged for `debug_output_cpp=1`.

---

### Phase 2 — Runtime Header and ABI Version Constant

**Goal**: Create `osl_cpp_runtime.h` and define `OSL_CPP_ABI_VERSION`.

**Files changed**:
- `src/liboslexec/osl_cpp_runtime.h` — NEW file: `#pragma once`, `OSL_CPP_ABI_VERSION` (see below), `extern "C"` forward declarations for `osl_*` runtime functions referenced by currently-implemented op generators (at minimum: all ops in `op_gen_init()` that call `osl_*`)

> **ABI version (as shipped):** `OSL_CPP_ABI_VERSION` is computed as
> `10000 * OSL_VERSION_MAJOR + 100 * OSL_VERSION_MINOR + revision` (currently
> `revision = 1`), not a bare `1`. Folding in the OSL major/minor version makes
> minor releases link-incompatible automatically; the manual `revision` digit
> covers an incompatible change within a single minor cycle. Because the
> generated DSOs are ephemeral (built against the same library they load into),
> this is a misuse guard, not a durable-compatibility contract. The constant is
> defined identically in `osl_cpp_runtime.h` and `oslexec_pvt.h`; a mismatch is
> caught loudly since every generated DSO fails the load-time ABI check. (See
> tasks.md Phase 13 / T054.)
- `src/liboslexec/backendcpp.cpp` — emit `#include "osl_cpp_runtime.h"` at top of every generated `.cpp`

**Acceptance**: Generated `.cpp` compiles with `#include "osl_cpp_runtime.h"` present and all referenced `osl_*` functions declared.

---

### Phase 3 — GroupData Struct Generation

**Goal**: Emit a typed `GroupData` struct whose layout exactly mirrors BackendLLVM's layout pass output. This makes the generated file reflect real memory layout and enables the `GroupData*` parameter type in layer functions.

**Key insight from code**: `BackendLLVM::llvm_type_groupdata()` (in `llvm_instance.cpp:289`) builds the struct field-by-field, storing byte offsets in `sym.dataoffset()` and the total size in `group().llvm_groupdata_size()`. After the layout pass runs, `BackendCpp` can read `sym.dataoffset()` directly to emit matching C++ field declarations.

**Struct field ordering** (must match LLVM):
1. `bool layer_runflags[N]` (rounded up to 32-bit boundary)
2. Userdata init flags array (if any userdata)
3. Userdata value fields (if any)
4. Per-layer, per-param fields (connected/interpolated/output params, with derivatives)

**Files changed**:
- `src/liboslexec/backendcpp.cpp` — add `generate_groupdata_struct()` method; call from `run()` before layer functions
- `src/liboslexec/backendcpp.h` — declare `generate_groupdata_struct()`

**Acceptance**: Generated file starts with a `struct GroupData { ... };` whose `sizeof(GroupData)` (when compiled) equals `group().llvm_groupdata_size()`.

---

### Phase 4 — Layer Function Signatures, Group Entry Function, and Virtual Interface

**Goal**: Rewrite `BackendCpp::run()` to emit proper layer function signatures (matching `RunLLVMGroupFunc` ABI), a group entry function that orchestrates layer dispatch, and `osl_cpp_abi_version()`. Also establish the virtual interface that future language backends will override.

**Virtual interface** (rename existing `cpp_*` methods → `lang_*`, mark `virtual`):
- `virtual std::string lang_type_name(TypeDesc)` — replaces `cpp_typedesc_name()`
- `virtual std::string lang_sym_type_name(const Symbol&)` — replaces `cpp_sym_type_name()`
- `virtual std::string lang_preamble()` — emits file header / includes
- `virtual std::string lang_function_qualifier()` — per-function qualifier; `""` for C++
- `virtual std::string lang_linkage_prefix()` — linkage specifier; `extern "C"` for C++
- `virtual std::string lang_file_extension()` — output file extension; `".cpp"` for C++
- `virtual std::string lang_ptr_syntax()` — pointer declarator; `"*"` for C++

All existing callers of `cpp_typedesc_name()` / `cpp_sym_type_name()` updated to call `lang_*`.

**Layer function signature** (per FR-003):
```cpp
void layer_N_name(ShaderGlobals* sg, GroupData* gd,
                  void* userdata_base, void* output_base,
                  int shadeindex, void* interactive_params);
```

**Group entry function** (per FR-004): matches `RunLLVMGroupFunc` exactly — checks `layer_runflags` before dispatching each layer.

**ABI version export** (per FR-009):
```cpp
extern "C" int osl_cpp_abi_version() { return OSL_CPP_ABI_VERSION; }
```

**Files changed**:
- `src/liboslexec/backendcpp.h` — rename `cpp_*` to `lang_*`, mark virtual, add new `lang_*` declarations
- `src/liboslexec/backendcpp.cpp` — rename callers, rewrite `run()`, add `generate_layer_func()`, `generate_group_entry()`
- `src/liboslcomp/symtab.cpp` — `cpp_safe_name()` gains reserved-word suffix guard (stays non-virtual; identifier safety is language-independent)

**Acceptance**: Generated `.cpp` compiles cleanly (no linker step yet) against OSL headers on Linux/macOS. Class hierarchy is correct: `BackendCpp` is subclassable with no further changes needed for a language backend to override type names and preamble.

---

### Phase 5 — DSO Compilation (level 2)

**Goal**: When `debug_output_cpp=2`, invoke `cpp_compiler` with `cpp_compiler_flags` via `popen`, capture stderr, forward through `ShadingSystem::errorfmt()` on failure.

**CMake work**: Bake in default compiler (`CMAKE_CXX_COMPILER`) and flags (`-shared -fPIC -O2` + include path to OSL headers) via configure-time substitution into `shadingsys.cpp`.

**Compiler invocation** (per FR-016b): use `popen` to capture stderr; read output into string; if exit status ≠ 0, call `shadingsys().errorfmt("BackendCpp: compilation failed:\n{}", captured_output)` and mark group failed.

**Files changed**:
- `src/liboslexec/backendcpp.cpp` — add `compile_to_dso()` method
- `src/liboslexec/backendcpp.h` — declare `compile_to_dso()`
- `src/liboslexec/shadingsys.cpp` — call `compile_to_dso()` when level ≥ 2
- `src/liboslexec/CMakeLists.txt` — define `OSL_CPP_COMPILER_DEFAULT` and `OSL_CPP_COMPILER_FLAGS_DEFAULT` (per-platform base flags + include paths) as compile definitions (as shipped — not a separate `configure.cmake`)

**Acceptance**: `debug_output_cpp=2` produces a `.so`/`.dylib` alongside the `.cpp`; bad generated code produces a legible OSL error with compiler diagnostics.

---

### Phase 6 — DSO Load, ABI Check, JIT Skip (level 3)

**Goal**: When `debug_output_cpp=3`, load the DSO, verify ABI, store entry point on `ShaderGroup`, skip full JIT. Unload in `ShaderGroup` destructor.

**JIT skip** (per FR-016c): as shipped, the layout-only-vs-full-JIT branch lives in `shadingsys.cpp` (guarded on `debug_output_cpp()`); `shadergroupopt.cpp` was not touched. The layout pass still runs so `sym.dataoffset()` is populated for GroupData generation.

**DSO load** (per FR-015): use `OIIO::Plugin::open()` / `OIIO::Plugin::getsym()` / `OIIO::Plugin::close()` (`OpenImageIO/plugin.h`) — platform-independent wrappers over `dlopen`/`LoadLibrary`. Resolve `osl_cpp_abi_version` symbol; compare to `OSL_CPP_ABI_VERSION`; on mismatch, call `errorfmt` and return error (no JIT fallback per FR-016).

**ShaderGroup storage**: add `OIIO::Plugin::Handle m_cpp_dso_handle` and a `RunLLVMGroupFunc m_cpp_compiled_version` to `ShaderGroup`. Destructor calls `OIIO::Plugin::close(m_cpp_dso_handle)` if non-null.

**Files changed**:
- `src/liboslexec/oslexec_pvt.h` — add DSO handle + function pointer fields to ShaderGroup; add destructor cleanup
- `src/liboslexec/backendcpp.cpp` — add `load_dso()` method
- `src/liboslexec/backendcpp.h` — declare `load_dso()`
- `src/liboslexec/shadingsys.cpp` / `shadergroupopt.cpp` — integrate layout-only + JIT skip path

**Acceptance**: `debug_output_cpp=3` with a simple shader runs without crashing; output matches JIT output for covered ops. ABI mismatch produces a clear error.

---

### Phase 7 — Symbol Visibility for `osl_*` Functions

**Goal**: Ensure `osl_*` functions in `llvm_ops.cpp` are exported from `liboslexec` so compiled shader DSOs can resolve them at load time.

**Implemented approach** (during T023/T033): `llvm_ops.cpp` is compiled twice — once to LLVM bitcode (existing, for JIT inlining) and once as a native object (new, linked into liboslexec). The `OSL_SHADEOP` macro's native-compilation branch was changed from `OSL_LLVM_EXPORT` (hidden) to `OSL_DLL_EXPORT`, exporting all `osl_*` functions globally. On macOS, generated DSOs are compiled with `-undefined dynamic_lookup` so the static linker does not require an explicit `-loslexec`; the dynamic linker resolves the symbols from the already-loaded liboslexec at `dlopen` time.

> **Linux requires more than the source attribute (added in Phase 13 / T053):**
> the `OSL_DLL_EXPORT` attribute alone is *not* sufficient on Linux, because the
> linker version script `src/build-scripts/hidesymbols.map` listed `osl_*` under
> `local:` and so stripped them from liboslexec's dynamic symbol table —
> overriding the attribute. The fix moves `osl_*` to the script's `global:`
> clause; generated DSOs then resolve the shadeops from the already-loaded
> liboslexec at `dlopen` time, with no link flag or RTLD promotion. The symbols
> are documented in the map as INTERNAL/UNSTABLE (exported solely for generated
> DSOs, not a public-API contract). macOS has no version script, which is why it
> passed on the attribute alone; this gap only surfaced once the backend ran in
> Linux CI (Phase 12).

**Files changed**:
- `src/liboslexec/llvm_ops.cpp` — `OSL_SHADEOP` and `OSL_SHADEOP_NOINLINE` native branch: `OSL_DLL_EXPORT` instead of `OSL_LLVM_EXPORT`
- `src/liboslexec/CMakeLists.txt` — added `llvm_ops.cpp` to `lib_src` for native compilation alongside bitcode; added `-undefined dynamic_lookup` to macOS generated-DSO compile flags
- `src/build-scripts/hidesymbols.map` — move `osl_*` from `local:` to `global:` (Phase 13 / T053) so the shadeops are dynamically exported on Linux

**Acceptance**: Compiled shader DSO loads without undefined symbol errors for `osl_*` functions on Linux and macOS.

---

### Phase 8 — Control Flow Op Generators

**Goal**: Implement generators for loop ops (`for`, `while`, `dowhile`) and `return`, `break`, `continue`. These are currently commented out in `op_gen_init()`.

**Files changed**:
- `src/liboslexec/backendcpp.cpp` — add `cpp_gen_loop_op()`, `cpp_gen_return()`, `cpp_gen_loopmod_op()` (break/continue); register in `op_gen_init()`

**Acceptance**: Shaders using `for`/`while` loops generate structurally correct C++ blocks. `return`/`break`/`continue` appear at correct nesting levels.

---

### Phase 9 — Remaining Op Generator Families

**Goal**: Implement generators for the remaining commented-out op families. Each sub-phase is independently reviewable.

**9a — `sincos`**: two output args from one call.

**9b — `printf`/`format`/`fprintf`/`warning`/`error`**: variadic, need format string handling.

**9c — Array ops** (`aref`, `aassign`, `compref`, `compassign`, `mxcompref`, `mxcompassign`, `arraylength`, `arraycopy`): index expressions and lvalue emission.

**9d — `getattribute` / `getmatrix` / `gettextureinfo`**: runtime service calls with out-param write-back.

**9e — Remaining runtime ops** (`raytype`, `backfacing`, `surfacearea`, `regex_match`, `regex_search`, `setmessage`, `getmessage`, `pointcloud_*`, `trace`, `transform` family, `blackbody`, `wavelength_color`, `spline`, `isconstant`, `functioncall`): each a `cpp_gen_*` stub calling the appropriate `osl_*` runtime function.

**Files changed**: `src/liboslexec/backendcpp.cpp` only for each sub-phase.

**Acceptance per sub-phase**: Shaders exercising the new ops generate compilable C++ with correct output under `debug_output_cpp=3`.

---

### Phase 10 — Test Infrastructure

**Goal**: Add a dedicated testsuite entry that exercises the C++ path and a reference test that confirms output parity with JIT.

**Files changed**:
- `testsuite/backend-cpp/` — NEW directory: simple OSL shader, `run.py` that sets `debug_output_cpp=1` and verifies `.cpp` file is created; second `run.py` variant with `debug_output_cpp=3` comparing output to JIT baseline

**Acceptance**: `ctest -R backend-cpp` passes. `OSL_DEBUG_OUTPUT_CPP=3 ctest` has no regressions vs JIT for covered ops.

---

## Complexity Tracking

No constitution violations. All additions are gated behind existing debug-path attribute; no hot-path or public API impact.
