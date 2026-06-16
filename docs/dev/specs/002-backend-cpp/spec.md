# Feature Specification: BackendCpp — C++ Source Code Generation Backend

**Feature Branch**: `002-backend-cpp`

**Created**: 2026-05-26

**Status**: Implemented — all four user stories complete; full testsuite passes
under the C++ backend (`OSL_TEST_CPP_BACKEND=1`) at both opt levels on macOS and
the Linux + macOS CI variants. Remaining items are deferred and out of scope:
`--entry` per-layer execution (not planned) and wiring the `example-*` programs.

## Overview

OSL's runtime currently goes: shader graph → RuntimeOptimizer → LLVM IR (BackendLLVM) → JIT → machine code. This feature adds a `BackendCpp` path that generates human-readable C++ source code from the post-optimized shader graph. The generated C++ is compilable to a DSO that is functionally equivalent to the JIT path, serving two use cases:

1. **Inspection**: human-readable output for debugging the runtime optimizer — developers can see exactly what C++ a shader group is equivalent to after optimization.
2. **Alternate execution**: within the same render, compile the generated C++ to a DSO and load it as a drop-in replacement for the JIT-compiled code, enabling the full test suite to be run via the C++ path to verify correctness parity.

A partial implementation already exists (files `backendcpp.h`, `backendcpp.cpp`). This spec covers completing and wiring that work into a usable, testable path.

The `debug_output_cpp` ShadingSystem attribute is an escalating integer that controls the pipeline stages:
- `1` — generate `.cpp` file only
- `2` — generate `.cpp` and shell out to compile it to a DSO
- `3` — generate `.cpp`, compile to DSO, load DSO, and execute it instead of JIT

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Inspect generated C++ for a shader (Priority: P1)

A developer sets `debug_output_cpp=1` before a shader group is compiled. OSL generates a self-contained `.cpp` file in the configured output directory. The developer opens that file and reads recognizable C++: a typed GroupData struct capturing the group's shared state, one function per layer that manipulates that struct, and a group entry function that orchestrates layer dispatch.

**Why this priority**: Foundational — all subsequent stories depend on generating valid, readable C++. Immediately useful for debugging optimizer behavior.

**Independent Test**: Set `debug_output_cpp=1`, run any OSL test shader through `testshade`, open the resulting `.cpp`, and manually verify structure and correctness against the known shader behavior.

**Acceptance Scenarios**:

1. **Given** `debug_output_cpp=1` and a shader group is compiled, **When** OSL completes, **Then** a `group-cpp-<groupname>.cpp` file exists in `cpp_output_dir`.
2. **Given** the generated file, **When** inspected, **Then** a typed `GroupData` struct appears at the top, each active layer maps to one C++ function, and one C++ statement appears per op.
3. **Given** an op with no implemented code generator, **When** the layer is traversed, **Then** a `// NO CPP GENERATOR FOR <opname>` comment appears — no crash, no silent omission.
4. **Given** a shader using `if`/`else` control flow, **When** generated, **Then** the output contains structurally correct nested `if`/`else` blocks.

---

### User Story 2 — Compile generated C++ to a loadable DSO (Priority: P2)

A developer sets `debug_output_cpp=2`. OSL generates the `.cpp` and shells out to compile it using the configured C++ compiler and flags. The compilation succeeds: all types resolve, all `osl_*` runtime function calls have declarations, and the resulting DSO exports the group entry point.

**Why this priority**: Validates that the generated output is syntactically and semantically correct C++.

**Independent Test**: Run `testshade --options debug_output_cpp=2,cpp_output_dir=/tmp/osl-cpp` on a representative set of shaders. All generated files must compile cleanly.

**Acceptance Scenarios**:

1. **Given** `debug_output_cpp=2`, **When** a shader group is compiled, **Then** OSL shells out to `cpp_compiler` with `cpp_compiler_flags` and the compilation succeeds with zero errors.
2. **Given** compilation succeeds, **When** the DSO is inspected, **Then** the group entry function symbol is exported with a predictable name.
3. **Given** a shader using `Dual2<T>` derivative types, **When** compiled, **Then** those types resolve correctly via the OSL include tree.
4. **Given** the DSO is loaded before any layer functions are called, **When** OSL checks the ABI version, **Then** `osl_cpp_abi_version()` returns a value matching the runtime's `OSL_CPP_ABI_VERSION` constant.

---

### User Story 3 — Load a DSO and execute it instead of JIT (Priority: P3)

A developer sets `debug_output_cpp=3`. OSL generates the `.cpp`, compiles it, loads the DSO, and routes all shader execution through the loaded functions instead of performing LLVM JIT compilation. Rendered output is identical (within floating-point tolerance) to the JIT baseline.

**Why this priority**: Closes the loop — the C++ path becomes a real alternative execution route.

**Independent Test**: Run `testshade --options debug_output_cpp=3,cpp_output_dir=/tmp/osl-cpp` on test shaders and compare output images/values against the JIT baseline.

**Acceptance Scenarios**:

1. **Given** `debug_output_cpp=3`, **When** a shader group is compiled, **Then** JIT compilation is skipped and the DSO's group entry function is stored in place of the JIT-compiled function pointer.
2. **Given** the DSO is executing, **When** the shader group runs, **Then** output values match the JIT path within acceptable floating-point tolerance.
3. **Given** the ABI version check fails on DSO load, **Then** OSL reports a clear error and aborts rather than calling into a potentially incompatible DSO.

---

### User Story 4 — Run the full test suite via the C++ path (Priority: P4)

The existing OSL testsuite can be run with `debug_output_cpp=3` to route all shader execution through generate→compile→load. Tests that pass under JIT also pass under the C++ path once op coverage is complete.

**Why this priority**: This is the definition of feature parity — an automated, ongoing verification target.

**Independent Test**: Set `OSL_OPTIONS=debug_output_cpp=3,cpp_output_dir=/tmp/osl-cpp` in the environment and run `ctest` with no test file modifications. Track pass rate as a fraction rising toward 100%.

**Implemented mechanism** (see tasks.md Phase 12): the testsuite auto-creates `.cpp`/`.cpp.opt` variants of every shader-running test (those whose `run.py` invokes `testshade`/`testrender`), gated by the `OSL_TEST_CPP_BACKEND` cmake option (default OFF; enabled in CI variants). Excluded: `optix`, `BATCHED_REGRESSION`, and tests with a `NOCPP` marker. As of Phase 12 all eligible tests pass at both opt levels; `layers-entry` (needs `--entry`) and `example-*` (run shaders via their own binaries) are the only deferred gaps.

**Acceptance Scenarios**:

1. **Given** `OSL_OPTIONS` sets `debug_output_cpp=3` globally, **When** the testsuite runs, **Then** every test shader generates C++ without BackendCpp crashing.
2. **Given** full op coverage is achieved, **When** the testsuite runs under the C++ path, **Then** all tests that pass under JIT also pass under the C++ path.
3. **Given** a discrepancy between JIT and C++ output, **When** investigated, **Then** the difference is traceable to a specific op generator or type-handling gap.

---

### Edge Cases

- Shader groups with zero active layers — generated file must be a valid empty translation unit.
- Layers with unsized array parameters or closure-typed parameters — use correct type names or emit `// UNIMPLEMENTED` and continue rather than crash.
- Symbol names that are C++ reserved words or contain characters illegal in C++ identifiers — `cpp_safe_name()` must handle all such cases.
- Shaders using derivatives (`Dx`, `Dy`, `Dz`, `filterwidth`) — `Dual2<T>` types must propagate correctly through the generated GroupData struct and layer functions.
- Multi-layer shader groups where an upstream layer feeds multiple downstream layers — the `layer_run` flag array in the generated GroupData struct ensures each layer executes at most once per shade call, exactly as in the JIT path.
- The same shader group name used across renders representing different computation — DSOs are ephemeral to the render that generates them and are not reused across renders; stale DSOs are not a concern for the initial use cases.
- Ops were filled in incrementally, each tracked with a `// NO CPP GENERATOR` stub as a safe fallback until implemented. *(All op families are now implemented — loops, closures, array/component ops, `printf`, `getattribute`/`getmatrix`/`gettextureinfo`, texture, noise, transform, pointcloud, splines, derivatives, etc. No op generators remain commented out; an unrecognized op still degrades gracefully to the stub comment rather than crashing.)*

**Known limitation — explicit per-layer entry points (`--entry`)**: executing a chosen subset of layers as entry points (as `testshade -entry`/`--entryoutput` and `ShaderGroupBegin` entry layers do) is **not supported in the C++ path**. The generated DSO exports a single group-entry function; per-entry-layer entry functions (the JIT's `build_llvm_instance` single-entry mode) are not generated or resolved. The `testsuite/layers-entry` test is therefore excluded from C++ testing (`NOCPP` marker). This feature is a candidate for removal; if it is kept and C++-mode parity becomes desired, generating/loading per-entry-layer functions is the work required.

---

## Clarifications

### Session 2026-05-26

- Q: When compilation fails at level 2 or 3, what should OSL do? → A: Report OSL error, mark shader group failed — no automatic JIT fallback (consistent with FR-016).
- Q: At `debug_output_cpp=3`, does JIT still run (output discarded) or is it skipped entirely? → A: Layout pass only; full JIT codegen is skipped entirely.
- Q: When should a loaded DSO be unloaded? → A: `dlclose` in ShaderGroup destructor — DSO lifetime tied to the group object.
- Q: Where does compiler error output go when compilation fails? → A: Capture via `popen`, forward through `ShadingSystem::errorfmt()`.

---

## Requirements *(mandatory)*

### Functional Requirements

**Code Generation**

- **FR-001**: BackendCpp MUST generate a single self-contained `.cpp` file per shader group. The file must include all necessary headers and be compilable against the OSL include tree without additional generated files.
- **FR-002**: The generated file MUST contain a typed `GroupData` struct whose memory layout exactly matches the layout computed by the BackendLLVM layout pass for the same shader group. BackendLLVM's layout pass MUST run before BackendCpp to provide the authoritative layout; BackendCpp reads and reflects it.
- **FR-003**: Each active shader layer MUST map to one C++ function with a signature analogous to the JIT internal layer signature: `void(ShaderGlobals*, GroupData*, void* userdata_base, void* output_base, int shadeindex, void* interactive_params)`, using the typed `GroupData*` instead of `void*`.
- **FR-004**: The group entry function MUST have exactly the `RunLLVMGroupFunc` signature so it can be stored as a drop-in replacement for the JIT-compiled group function pointer.
- **FR-005**: The generated `GroupData` struct MUST include the `layer_run` flags array, ensuring each layer executes at most once per shade call (same run-once semantics as the JIT path).
- **FR-006**: All ops currently listed (not commented out) in `op_gen_init()` MUST have correct code generators. For ops expressible as plain C++ (arithmetic, comparisons, math), generators MUST emit direct C++ expressions rather than `osl_*` calls. Ops requiring runtime services (noise, texture, getattribute, closures) MAY call `osl_*` functions.
- **FR-007**: Control flow ops (`if`, loops, `return`, `break`, `continue`) MUST generate structurally correct nested C++ blocks.
- **FR-008**: The generated file MUST `#include` a new internal header `osl_cpp_runtime.h` that provides: the `OSL_CPP_ABI_VERSION` constant, `extern "C"` declarations for all `osl_*` runtime functions, and any other generated-code-facing declarations.
- **FR-009**: The generated file MUST export `extern "C" int osl_cpp_abi_version()` returning `OSL_CPP_ABI_VERSION`. OSL MUST call this function immediately after loading the DSO, before resolving or calling any layer functions, and reject the DSO if the version does not match.

**Runtime Symbol Visibility**

- **FR-010**: `osl_*` functions defined in `llvm_ops.cpp` that are called by generated C++ MUST be exported from `liboslexec` (not `OSL_DLL_LOCAL`). These are not part of the public API and carry no public header; they are exported solely for use by compiled shader DSOs. *(Implemented: the `OSL_SHADEOP` macro marks them `OSL_DLL_EXPORT` in the native build, and — critically on Linux — `src/build-scripts/hidesymbols.map` lists `osl_*` under `global:` so the linker version script does not localize them out of the dynamic symbol table. On macOS the generated DSOs resolve them at dlopen time from the already-loaded liboslexec; the symbols are marked INTERNAL/UNSTABLE in the map comment.)*

**ShadingSystem Attributes**

- **FR-011**: The `debug_output_cpp` integer attribute MUST control the C++ pipeline as an escalating integer: `1` = generate `.cpp`; `2` = generate and compile to DSO; `3` = generate, compile, load DSO, and execute instead of JIT.
- **FR-012**: A `cpp_output_dir` string attribute MUST control where generated `.cpp` and DSO files are written. Default: working directory.
- **FR-013**: A `cpp_compiler` string attribute MUST specify the C++ compiler executable. Default: the compiler used to build OSL, baked in at CMake configure time.
- **FR-014**: A `cpp_compiler_flags` string attribute MUST specify the compiler flags for DSO compilation. Default: flags appropriate for the build platform, baked in at CMake configure time.

**DSO Loading and Execution**

- **FR-015**: When `debug_output_cpp=3`, OSL MUST load the compiled DSO, verify the ABI version, resolve the group entry function by its exported symbol name, and store it in the ShaderGroup in place of the JIT-compiled function pointer. The DSO handle MUST be stored in the ShaderGroup and unloaded (`dlclose`/`FreeLibrary`) in the ShaderGroup destructor.
- **FR-016**: If DSO loading fails or the ABI version check fails, OSL MUST report a clear error. Fallback to JIT is not automatic.
- **FR-016b**: If compilation fails at `debug_output_cpp=2` or `=3`, OSL MUST report a clear error and mark the shader group failed. Fallback to JIT is not automatic. The compiler MUST be invoked via `popen` (or platform equivalent) so its stderr output is captured and forwarded verbatim through `ShadingSystem::errorfmt()`; compiler exit status MUST be included in the error message.
- **FR-016c**: When `debug_output_cpp=3`, OSL MUST run BackendLLVM's layout pass only and skip full JIT codegen. The DSO is the sole execution path; running the full JIT compile would waste time and is unnecessary.

**Testing Infrastructure**

- **FR-017**: A dedicated `OSL_DEBUG_OUTPUT_CPP` environment variable MUST default the `debug_output_cpp` attribute at ShadingSystem construction, analogous to `OSL_LLVM_DEBUG`. This allows the entire testsuite to be run via the C++ path by setting one env var (`OSL_DEBUG_OUTPUT_CPP=3`) before invoking `ctest`, with no test file modifications. `OSL_OPTIONS` remains an alternative for setting multiple attributes together.

### Key Entities

- **BackendCpp**: `OSOProcessorBase` subclass that traverses the post-optimized shader group and emits C++ source. Lives in `src/liboslexec/backendcpp.{h,cpp}`. Runs after BackendLLVM's layout pass.
- **OpCppGen**: Function-pointer type `bool (*)(BackendCpp&, int opnum)` stored in `OpDescriptor::cppgen`. `nullptr` → `// NO CPP GENERATOR` stub in output.
- **GroupData struct**: Typed C++ struct generated at the top of each `.cpp` file. Mirrors the exact memory layout computed by BackendLLVM. Contains all shader parameter storage and the `layer_run` flags array.
- **osl_cpp_runtime.h**: New internal header. Declares `OSL_CPP_ABI_VERSION`, forward-declares `osl_*` runtime functions, and provides any other types needed by generated code.
- **OSL_CPP_ABI_VERSION**: Integer constant in `oslexec_pvt.h` (alongside `RunLLVMGroupFunc`). Bumped whenever the generated-code ABI changes (ShaderGlobals layout, osl_* signatures, calling convention, etc.).
- **ShaderGroup**: Unit of compilation. One `.cpp` and one DSO per group. Files named `group-cpp-<sanitized-name>.cpp` / `.so` / `.dylib` / `.dll` in `cpp_output_dir`.
- **cpp_safe_name()**: Symbol method mapping OSL symbol names to valid C++ identifiers. Must handle all reserved words and illegal characters.

---

## Success Criteria *(mandatory)*

- **SC-001**: All shaders in the existing testsuite generate `.cpp` output with `debug_output_cpp=1` without BackendCpp crashing or producing empty output.
- **SC-002**: Generated `.cpp` files compile with zero errors using the `cpp_compiler` / `cpp_compiler_flags` attributes on at least Linux (GCC/Clang) and macOS (Clang). *(Verified on the Linux + macOS CI variants. The shadeop runtime is also MSVC-clean — `OSL::popcount` replaced a GCC/Clang-only builtin.)*
- **SC-003**: Shaders covering all currently-implemented op generators produce execution output via the DSO load path (`debug_output_cpp=3`) that matches the JIT path output within floating-point tolerance.
- **SC-004**: The full testsuite pass rate in C++ DSO mode equals the JIT pass rate once all ops are implemented. *(Achieved: every cpp-eligible test passes under `OSL_TEST_CPP_BACKEND=1` at both opt levels — 454/454 in the local macOS sweep — excluding only the documented opt-outs: `--entry`/`layers-entry`, the `backend-cpp` fixture itself, OptiX/GPU, and batched-regression harnesses.)*
- **SC-005**: The C++ generation step (`debug_output_cpp=1`) adds no perceptible overhead to shader compilation time — it is a debug path, not on the hot render path. *(Verified, T041: +2.1ms / +0.9% (≈1σ, within noise) on a 600-op stress shader; unmeasurable on a typical shader.)*

---

## Assumptions

- BackendLLVM's GroupData layout computation runs before BackendCpp in all modes. When `debug_output_cpp=3`, OSL runs the layout pass only — full JIT codegen is skipped (see FR-016c).
- The `RunLLVMGroupFunc` calling convention and `ShaderGlobals` struct layout are stable. Changes to either require bumping `OSL_CPP_ABI_VERSION`.
- "Feature parity" means passing the same testsuite tests as JIT — not bit-for-bit identical floating-point output (compiler optimizations may produce slightly different FP results).
- DSOs are ephemeral to the render run that generates them. Persistent DSO caching across renders (with hash-based invalidation) is explicitly deferred; the ABI version check and `cpp_output_dir` attribute are the breadcrumbs enabling it in the future.
- Loop ops (`for`, `while`, `dowhile`) and remaining commented-out ops are implemented incrementally with no mandated order.
- Batched/SIMD execution (BatchedBackendLLVM) is out of scope. BackendCpp targets single-point execution only.
- PTX/OptiX output is out of scope.
- **Extensibility**: BackendCpp is designed to be subclassable so that future backends targeting similar-to-C++ languages can override only the language-specific pieces. The parts that vary across languages — type name mapping, language preamble, linkage specifiers, function qualifiers, file extension — MUST be implemented as `virtual` methods in BackendCpp so subclasses can override them without duplicating the traversal logic. No such subclasses are implemented in this feature; the design constraint is that the C++ backend's structure does not foreclose them.
