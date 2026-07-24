# Tasks: BackendCpp — C++ Source Code Generation Backend

**Input**: Design documents from `docs/dev/specs/002-backend-cpp/`

**Branch**: `002-backend-cpp`

## Format: `[ID] [P?] [Story?] Description with file path`

- **[P]**: Parallelizable — touches different files or non-overlapping sections
- **[Story]**: Which user story (US1–US4)
- Each task is one logical change, reviewable as a single small diff

**Testing strategy**: A single `testsuite/backend-cpp/` entry is created in Phase 1 and evolves with every phase. Each phase adds to the test shader and/or `run.py` to cover the new capability and guard against regressions in what came before. By the time Phase 9 completes, the test exercises the full C++ path end-to-end and serves as the permanent CI fixture.

---

## Phase 1: Foundational — Attribute Plumbing

**Purpose**: Wire the four ShadingSystem attributes, change `debug_output_cpp` from `bool` to `int`, add env var, write to `cpp_output_dir`. Creates the testsuite scaffold.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T001 Change `m_debug_output_cpp` from `bool` to `int` and update its accessor in `src/liboslexec/oslexec_pvt.h`
- [X] T002 Update `ATTR_SET` / `ATTR_DECODE` for `debug_output_cpp` (int) in `src/liboslexec/shadingsys.cpp`
- [X] T003 Add `m_cpp_output_dir` member, accessor, `ATTR_SET`/`ATTR_DECODE` for `"cpp_output_dir"` in `src/liboslexec/oslexec_pvt.h` and `src/liboslexec/shadingsys.cpp`
- [X] T004 [P] Add `m_cpp_compiler` and `m_cpp_compiler_flags` members, accessors, `ATTR_SET`/`ATTR_DECODE` in `src/liboslexec/oslexec_pvt.h` and `src/liboslexec/shadingsys.cpp`
- [X] T005 [P] Bake default compiler and flags at configure time via `OSL_CPP_COMPILER_DEFAULT` / `OSL_CPP_COMPILER_FLAGS_DEFAULT`; initialize members from these in `src/liboslexec/shadingsys.cpp` and `src/liboslexec/CMakeLists.txt`
- [X] T006 Add `OSL_DEBUG_OUTPUT_CPP` env-var read in `ShadingSystemImpl` constructor (mirrors `OSL_LLVM_DEBUG` at line 1243) in `src/liboslexec/shadingsys.cpp`
- [X] T007 Update BackendCpp invocation in `src/liboslexec/shadingsys.cpp` to write generated `.cpp` to `cpp_output_dir()` with filename `group-cpp-<name>.cpp`; gate on `debug_output_cpp() >= 1`
- [X] T008 Create `testsuite/backend-cpp/` with a minimal two-param OSL shader (`shader backend_cpp_test`) and `run.py`: set `debug_output_cpp=1`, run `testshade`, assert the `.cpp` output file exists and is non-empty — this is the scaffold all later phases extend

**Checkpoint**: `ctest -R backend-cpp` passes. Generated file is written to the configured output dir.

---

## Phase 2: Foundational — Runtime Header and ABI Version

**Purpose**: Create `osl_cpp_runtime.h`; every generated file includes it.

- [X] T009 Create `src/liboslexec/osl_cpp_runtime.h`: `#pragma once`, `constexpr int OSL_CPP_ABI_VERSION = 1`, `extern "C"` forward declarations for all `osl_*` functions referenced by currently-registered generators in `op_gen_init()`
- [X] T010 Update `BackendCpp::run()` in `src/liboslexec/backendcpp.cpp` to emit `#include "osl_cpp_runtime.h"` as the first line of every generated file; make `OSL_CPP_ABI_VERSION` accessible in `src/liboslexec/oslexec_pvt.h` for later ABI comparison
- [X] T011 Extend `testsuite/backend-cpp/run.py`: parse the generated `.cpp` and assert `#include "osl_cpp_runtime.h"` appears on the first non-blank line

**Checkpoint**: `ctest -R backend-cpp` passes; generated file has the correct include.

---

## Phase 3: User Story 1 — GroupData Struct Generation

**Goal**: Generated file opens with a typed `struct GroupData` mirroring BackendLLVM's layout.

**Independent Test**: Open generated `.cpp`; `struct GroupData {` appears before any function; fields reflect the shader's parameters and `layer_run` flags.

- [X] T012 [US1] Declare `generate_groupdata_struct()` in `src/liboslexec/backendcpp.h`; implement in `src/liboslexec/backendcpp.cpp` — emit `layer_runflags` bool array (field 0, rounded to 32-bit boundary), userdata init flags, userdata value fields, and per-layer connected/output param fields by reading `sym.dataoffset()` and `group().llvm_groupdata_size()`
- [X] T013 [US1] Call `generate_groupdata_struct()` from `BackendCpp::run()` before any layer function emission in `src/liboslexec/backendcpp.cpp`
- [X] T014 [US1] Update `testsuite/backend-cpp/` shader to have a connected output parameter (so GroupData is non-trivial); extend `run.py` to assert `struct GroupData {` appears in the generated file before any function definition

**Checkpoint**: `ctest -R backend-cpp` passes; generated file has a non-empty `struct GroupData`.

---

## Phase 4: User Story 1 — Virtual Language Interface and Layer Function Scaffolding

**Goal**: Establish the `lang_*` virtual interface; generate correct layer function signatures and group entry function. Generated file is now compilable C++.

**Independent Test**: Generated `.cpp` compiles with zero errors (`clang++ -shared -fPIC <file>`); contains one `void layer_N_name(ShaderGlobals*, GroupData*, ...)` per active layer, a group entry matching `RunLLVMGroupFunc`, and `extern "C" int osl_cpp_abi_version()`.

- [X] T015 [US1] Rename `cpp_typedesc_name()` → `lang_type_name()` and `cpp_sym_type_name()` → `lang_sym_type_name()` in `src/liboslexec/backendcpp.h`; mark both `virtual`; update all callers in `src/liboslexec/backendcpp.cpp`
- [X] T016 [P] [US1] Add remaining virtual `lang_*` methods to `src/liboslexec/backendcpp.h` with C++ defaults in `src/liboslexec/backendcpp.cpp`: `lang_preamble()`, `lang_function_qualifier()`, `lang_linkage_prefix()`, `lang_file_extension()`, `lang_ptr_syntax()`
- [X] T017 [US1] Rewrite `BackendCpp::run()` in `src/liboslexec/backendcpp.cpp` to emit per-layer functions with correct signature `void layer_N(ShaderGlobals*, GroupData*, void*, void*, int, void*)` using `lang_*` methods for all language tokens; replace old bare-prototype output
- [X] T018 [US1] Add `generate_group_entry()` in `src/liboslexec/backendcpp.h` and `src/liboslexec/backendcpp.cpp`: emit `RunLLVMGroupFunc`-compatible entry checking `layer_runflags` before dispatching; emit `osl_cpp_abi_version()` export; call from `run()`
- [X] T019 [P] [US1] Extend `Symbol::cpp_safe_name()` in `src/liboslcomp/symtab.cpp` to append `_osl` suffix for C++ reserved words
- [X] T020 [US1] Extend `testsuite/backend-cpp/` shader to use two connected layers so group entry dispatch is exercised; extend `run.py` to: (a) assert correct layer function signatures appear, (b) assert `osl_cpp_abi_version` is present, (c) invoke the compiler directly on the generated file (`cpp_compiler + cpp_compiler_flags`) and assert zero exit status — the test now verifies the file compiles

**Checkpoint**: `ctest -R backend-cpp` passes; generated file compiles cleanly. `BackendCpp` is subclassable via `lang_*` overrides.

---

## Phase 5: User Story 2 — DSO Compilation

**Goal**: `debug_output_cpp=2` invokes the compiler automatically and produces a loadable DSO.

**Independent Test**: `debug_output_cpp=2` produces a `.so`/`.dylib`; a bad compiler path produces a legible OSL error containing compiler diagnostics.

- [X] T021 [US2] Implement `BackendCpp::compile_to_dso()` in `src/liboslexec/backendcpp.cpp`: build compiler command from `cpp_compiler()` + `cpp_compiler_flags()` + input path + output DSO path; invoke via `popen("cmd 2>&1", "r")`; capture output; on nonzero exit status call `shadingsys().errorfmt()` with captured text and mark group failed (no JIT fallback per FR-016b)
- [X] T022 [US2] Declare `compile_to_dso()` in `src/liboslexec/backendcpp.h`; wire call in `src/liboslexec/shadingsys.cpp` when `debug_output_cpp() >= 2`
- [X] T023 [P] [US2] Audit `osl_*` functions in `src/liboslexec/llvm_ops.cpp` called by current generators; add `OSL_DLL_EXPORT` (or visibility attribute) to each; verify `src/liboslexec/CMakeLists.txt` does not hide them. *(Implemented during T033: changed `OSL_SHADEOP` macro's native-compilation branch from `OSL_LLVM_EXPORT`/hidden to `OSL_DLL_EXPORT`, and added a second native-object compilation of `llvm_ops.cpp` in `CMakeLists.txt` alongside the existing bitcode compilation. Added `-undefined dynamic_lookup` on macOS so generated DSOs resolve `osl_*` symbols from the already-loaded liboslexec at dlopen time. All `osl_*` functions are now globally exported; no per-function auditing required.)*
- [X] T024 [US2] Extend `testsuite/backend-cpp/run.py`: add a `debug_output_cpp=2` run; assert the DSO file (`.so`/`.dylib`/`.dll`) is created alongside the `.cpp`; assert `testshade` reports no errors — the test now verifies automatic compilation succeeds

**Checkpoint**: `ctest -R backend-cpp` passes; test verifies both file generation and automatic DSO compilation.

---

## Phase 6: User Story 3 — DSO Load, ABI Check, JIT Skip

**Goal**: `debug_output_cpp=3` skips JIT, loads the DSO, and routes shader execution through the compiled functions.

**Independent Test**: A shader the optimizer folds to constant output runs at `debug_output_cpp=3` and produces output bit-matching the JIT baseline; ABI mismatch produces a clear error. (Value-correctness for shaders with default-valued or connected params is completed in Phase 7.)

- [X] T025 [US3] Add `OIIO::Plugin::Handle m_cpp_dso_handle` and `RunLLVMGroupFunc m_cpp_compiled_version` to `ShaderGroup` in `src/liboslexec/oslexec_pvt.h`; initialize to `nullptr`; add `OIIO::Plugin::close(m_cpp_dso_handle)` to `ShaderGroup` destructor
- [X] T026 [P] [US3] Separate the BackendLLVM layout pass from full JIT codegen in `src/liboslexec/shadingsys.cpp` or `src/liboslexec/shadergroupopt.cpp`: when `debug_output_cpp() == 3`, run layout pass only (so `sym.dataoffset()` is populated) and skip the remainder of JIT compilation
- [X] T027 [US3] Implement `BackendCpp::load_dso()` in `src/liboslexec/backendcpp.cpp`: call `OIIO::Plugin::open()`; resolve `osl_cpp_abi_version` via `OIIO::Plugin::getsym()`; compare to `OSL_CPP_ABI_VERSION`; on mismatch call `errorfmt()` and fail; resolve group entry symbol; store handle and function pointer in `ShaderGroup`
- [X] T028 [US3] Declare `load_dso()` in `src/liboslexec/backendcpp.h`; wire in `src/liboslexec/shadingsys.cpp` when `debug_output_cpp() == 3`; route shader execution through `ShaderGroup::m_cpp_compiled_version`
- [X] T028a [US3] Generate the renderer-output write-back in `BackendCpp::generate_layer_func()`: for each `renderer_output()` param with an `Outputs`-arena symloc, emit `std::memcpy` into `output_base` at `offset + stride*shadeindex` (mirrors the JIT "copy results to renderer outputs" pass at `llvm_instance.cpp:1805`). Add `#include <cstring>` to `osl_cpp_runtime.h`. *(Added during implementation — the level-3 path produced zeros without it; the host reads outputs from `output_base`, not GroupData.)*

**Checkpoint**: `ctest -R backend-cpp` passes. The level-3 path (layout-only pass → DSO load → ABI check → JIT skip → execute via `cpp_compiled_version` → renderer-output write-back) is exercised, and an optimizer-folded shader's level-3 output matches JIT.

---

## Phase 7: User Story 3 — End-to-End Execution Correctness

**Goal**: Generated C++ produces output *values* matching JIT for shaders with default-valued and connected params — not just optimizer-folded constants. Completes the value-correctness gaps in layer-function generation, then adds the level-3 parity test (the original T029).

**Independent Test**: The 2-layer connected `testsuite/backend-cpp/` shader produces level-3 output matching the JIT baseline in *both* the optimized and non-optimized variants.

**Context**: Phase 6 proved the level-3 execution *wiring* is correct — an optimizer-folded shader matches JIT bit-for-bit. But the existing Phase 4 layer scaffolding does not yet emit correct values for shaders whose params have defaults or incoming connections: the non-opt variant of the test shader yields `(0,1,0)` instead of `(1,0,2)`. These tasks close that gap.

- [X] T029a [US3] Implement parameter initialization in `BackendCpp::generate_layer_func()`. Generated layers currently read params straight from zeroed GroupData (literal `/* = init TBD */` marker). Emit default-value / init-op assignment for each param before the main code, mirroring the JIT's `BackendLLVM::llvm_assign_initial_value` / parameter-init pass. After this, an unconnected default-valued param (e.g. `layer_a.in_val = 0.5`) computes correctly.
- [X] T029b [US3] Implement multi-layer execution so connected upstream layers run and propagate their outputs. **Approach TBD — decide before starting**: (a) *eager* — `generate_group_entry()` dispatches all used layers in dependency order with run-flag guards (with `lazylayers=0` the generated entry already does this; make it the generated default), or (b) *lazy* — emit "run connected layer on demand" calls before reading a connected input, mirroring `BackendLLVM::llvm_run_connected_layers`. Eager is simpler and correct for output values; lazy is the more faithful match to JIT's conditional execution.
- [X] T029 [US3] Extend the `testsuite/backend-cpp/` shader to compute a specific numeric output (color from arithmetic on connected + default-valued inputs); extend `run.py` to add a `debug_output_cpp=3` run and compare its image output to a JIT reference (`ref/out.tif`). The test now verifies end-to-end execution correctness in both the opt and non-opt variants.

**Checkpoint**: `ctest -R backend-cpp` passes; level-3 output of the connected, default-valued test shader matches JIT in all variants.

**Dependencies**: T029a → T029 and T029b → T029 (both prerequisites before the parity test). T029a and T029b are largely independent in concept (`generate_layer_func()` vs `generate_group_entry()`) but both edit `backendcpp.cpp`, so land them sequentially.

---

## Phase 8: User Story 1 / US4 — Control Flow Op Generators

**Goal**: Shaders using loops and early-exit statements generate correct C++ control flow and execute correctly.

**Independent Test**: Shader with a `for` loop and a conditional `return` produces correct output at `debug_output_cpp=3`.

- [X] T030 [US1] Implement `cpp_gen_loop_op()` in `src/liboslexec/backendcpp.cpp` for `for`, `while`, `dowhile`; register all three in `op_gen_init()`
- [X] T031 [P] [US1] Implement `cpp_gen_return()` and `cpp_gen_loopmod_op()` (break, continue) in `src/liboslexec/backendcpp.cpp`; register `return`, `break`, `continue`, `exit` in `op_gen_init()`
- [X] T032 [US1] Add a loop and a conditional early-return to the `testsuite/backend-cpp/` shader; extend `run.py` to verify the `debug_output_cpp=3` run produces the same output as JIT for this shader — the test now covers control flow

**Checkpoint**: `ctest -R backend-cpp` passes including the loop/early-return cases.

---

## Phase 9: User Story 1 / US4 — Remaining Op Generator Families

**Goal**: Fill in the commented-out op families incrementally. Each sub-phase lands with the test extended to cover the new ops.

**Testing approach**: Each sub-phase adds new ops to the test shader and extends `run.py` to verify their output at `debug_output_cpp=3` matches JIT. Regressions in previously-added ops are caught by the same run.

- [X] T033 [P] [US1] Implement `cpp_gen_sincos()` in `src/liboslexec/backendcpp.cpp` (two output args); register `sincos` in `op_gen_init()`; add `sincos` usage to the test shader; extend `run.py` to verify output matches JIT
- [X] T034 [P] [US1] Implement `cpp_gen_printf()` for `printf`, `format`, `fprintf`, `warning`, `error` in `src/liboslexec/backendcpp.cpp`; register all five; add a `printf` call to the test shader; extend `run.py` to verify the printed output appears correctly
- [X] T035 [P] [US1] Implement array and component access generators in `src/liboslexec/backendcpp.cpp`: `aref`, `aassign`, `compref`, `compassign`, `mxcompref`, `mxcompassign`, `arraylength`, `arraycopy`; register all; add array indexing to the test shader; verify output. *(`compref`/`aref` and `compassign`/`aassign` kept as separate generators despite identical C++ bodies so `compref` can later emit `.x`/`.y`/`.z` for constant indices. Also fixed `cpp_var_declaration()` to emit the array bound in the declarator (`float arr[4]`) — `lang_sym_type_name()` drops it for plain arrays, so array locals/temps previously declared as scalars and failed to compile.)*
- [X] T036 [P] [US1] Implement `getattribute`, `getmatrix`, `gettextureinfo` generators in `src/liboslexec/backendcpp.cpp` (runtime service calls with out-param write-back); register all; add a `getattribute` call to the test shader; verify output. **`getmatrix` DONE**: `cpp_gen_getmatrix` → `osl_get_from_to_matrix(oec,&M,from,to)` status into Result; plus a `cpp_gen_assign` fix for `matrix = scalar` (set the diagonal, not Imath's all-elements `Matrix44(T)`) factored into `cpp_emit_matrix_diagonal` (shared with `cpp_gen_matrix`). **`getattribute` DONE**: `cpp_gen_getattribute` (eight flavors — optional object name, optional array index), mirroring the non-spec branch of `llvm_gen_getattribute`. **`gettextureinfo` DONE** (landed with the texture family, T046): `cpp_gen_gettextureinfo` for both the 3-arg and 5-arg (s,t) forms. All three registered. **Verified green** in the full opt-out cpp sweep (Phase 12) at both opt levels.
- [X] T037 [US1] Implement remaining runtime op generators in `src/liboslexec/backendcpp.cpp`: `raytype`, `backfacing`, `surfacearea`, `regex_match`, `regex_search`, `setmessage`, `getmessage`, `pointcloud_search`, `pointcloud_get`, `pointcloud_write`, `trace`, `transform`, `transformc`, `transformn`, `transformv`, `blackbody`, `wavelength_color`, `spline`, `splineinverse`, `isconstant`, `functioncall`, `functioncall_nr`; register each; extend test shader to cover a representative subset; verify output matches JIT. **`transform`/`transformv`/`transformn`/`transformc` DONE (value-complete)**: `cpp_gen_transform` (matrix form → generic; named-space form → `osl_transform_triple[_nonlinear]` with identity short-circuit, reusing the T043 infra) + `cpp_gen_transformc` → `osl_transformc`; declared `osl_transformc`. **Verified green: `transform`** — byte-identical to JIT at O0+O2 including the `Dx/Dy` derivative lines (once T048 landed). Zero regressions. **`backfacing`/`surfacearea` DONE**: `cpp_gen_get_simple_SG_field` emits a direct `Result = sg->{field}` read (op name == ShaderGlobals field), mirroring `llvm_gen_get_simple_SG_field`. **`isconstant` DONE**: `cpp_gen_isconstant` folds to a compile-time 0/1 from the post-optimization `is_constant()` state (mirrors `llvm_gen_isconstant`); test is OPTIMIZEONLY so only the `.cpp.opt` variant runs. **`calculatenormal` DONE** (was mis-routed through `generic`, which dropped the exec-context arg → undeclared 2-arg `osl_calculatenormal_vv`; same class as the luminance/T052 bug): dedicated `cpp_gen_calculatenormal` passing `(void*)sg`, zero result when `P` has no derivs, zero result partials otherwise; removed the dead `osl_calculatenormal_vvv` generic alias. **Verified green: `shaderglobals` (covers backfacing+surfacearea+calculatenormal), `isconstant`** — full cpp sweep 359/359, zero regressions. **`pointcloud_search`/`pointcloud_get`/`pointcloud_write` DONE**: `cpp_gen_pointcloud_search`/`_get`/`_write` registered and verified green via the `pointcloud` test in the Phase 12 sweep. No op generators remain commented out.
- [X] T038 [P] [US1] For each op family added in T033–T037: add any new `osl_*` declarations to `src/liboslexec/osl_cpp_runtime.h`. *(The `OSL_DLL_EXPORT` part is already done globally by T023/T033 — all `osl_*` functions in `llvm_ops.cpp` are exported from the native-compiled copy in liboslexec. T038 now only needs to ensure the forward declarations in `osl_cpp_runtime.h` cover each new function called by the generated code.)* **Scalar (`Dual2<float>`) deriv variants added** for the per-component math ops the `derivs` test exercises: `sin cos tan asin acos atan atan2 sinh cosh tanh exp exp2 expm1 log log2 log10 sqrt cbrt inversesqrt pow erf erfc abs fabs` (all native — forward-declared only). `min`/`max` have NO native deriv function (inline-only helpers), so `osl_min_dfdfdf`/`osl_max_dfdfdf` added as `Dual2` inline helpers (`<=`/`>` tie-break matching `llvm_gen_minmax`). Also fixed **deriv-aware safe-divide** (`osl_div_dual` — `llvm_gen_div`'s `binv` formula with `osl_safe_div_fff`, replacing raw `Dual2 operator/` that NaN/Inf'd on zero divisors), **triple-global deriv loading** (`P`/`I` now `Dual2<Vec3>(sg->P, sg->dPdx, sg->dPdy)` instead of deriv-zeroing `= sg->P`), and **space-construct of a deriv-triple** (`point("shader",u,v,0)` now builds via `cpp_triple_ctor` + transforms with real deriv flags, instead of the invalid `R[c]`). **Verified green: `derivs`** at O0+O2; gate 56; zero regressions. **Triple (`Dual2<Vec3>`) deriv variants completed by T048** (deriv-carrying triples promoted to `Dual2<Vec3>`/`Dual2<Color3>`, deriv-aware `cpp_gen_generic`); **noise deriv decls completed by T045**. All required `osl_*` forward declarations are present; the full cpp sweep compiles and links with no missing symbols.

**Checkpoint**: `ctest -R backend-cpp` passes and covers all op families added in this phase. Pass rate under `OSL_DEBUG_OUTPUT_CPP=3 ctest` is monotonically rising.

---

## Phase 10: Polish & Cross-Cutting Concerns

- [X] T039 [P] Verify edge cases in `testsuite/backend-cpp/run.py`: zero-layer shader group produces a valid empty translation unit; closure-typed and unsized-array parameters emit `// UNIMPLEMENTED` stubs without crashing; add these as separate shader variants in `testsuite/backend-cpp/`
- [X] T040 [P] Add an ABI-mismatch test variant to `testsuite/backend-cpp/run.py`: temporarily build a DSO with a wrong `OSL_CPP_ABI_VERSION` and verify OSL reports a clear error rather than crashing (can be simulated by passing a pre-built stub DSO)
- [X] T041 Verify `debug_output_cpp=1` adds no measurable overhead to shader compile time (SC-005): run compile-time benchmark with attribute on vs. off and confirm result is within noise. **DONE.** Benchmarked via `testshade --runstats` (reports "Runtime optimization cost", which is where BackendCpp generation runs) and via whole-process wall-clock over many trials. (a) Typical 22-op shader: optimization cost reported identically `0.09s` across 25 trials at `=0` and `=1` (generation is below the 10ms reporting granularity). (b) Deliberate 600-op stress shader, 40 full-process trials each (warmed): `=0` mean 229.7ms (σ=2.0), `=1` mean 231.8ms (σ=2.2) → **delta +2.1ms (+0.9%), ≈1σ — within noise**. Generation (~2ms even for 600 ops) is dwarfed by the ~160ms optimize/JIT it sits inside. Confirmed the `.cpp` was actually emitted in the `=1` runs (75-line / large TU). SC-005 satisfied: the C++ generation step adds no perceptible compile-time overhead.

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (T001–T008) → Phase 2 (T009–T011) → all user story phases
Phase 3 (T012–T014) → Phase 4 (T015–T020)    [US1: GroupData before layer functions]
Phase 4              → Phase 8 (T030–T032)    [control flow needs correct layer scaffold]
Phase 4              → Phase 9 (T033–T038)    [remaining ops need correct layer scaffold]
Phase 5 (T021–T024)  [US2: compile — can start after Phase 4]
Phase 6 (T025–T028a) [US3: load/execute — T025/T026 after Phase 1; T027/T028 after Phase 5]
Phase 6              → Phase 7 (T029a–T029)   [US3: execution correctness — values match JIT]
Phase 9              → Phase 10               [polish and edge cases after main op coverage]
```

### User Story Dependencies

- **US1 (P1)**: Phases 1–4 → immediately testable; Phases 8–9 extend op coverage
- **US2 (P2)**: Phase 4 (compilable output) + T023 (symbol visibility) → US2 testable
- **US3 (P3)**: Phase 5 (DSO exists) + Phase 6 (load/execute wiring) → execution routed; Phase 7 (param init + multi-layer) → output values match JIT
- **US4 (P4)**: Phases 8–9 substantially complete → full testsuite coverage

### Parallel Opportunities (within phases)

- T004 ∥ T005 (different members, non-overlapping)
- T015 → T016 ∥ T019 (T016 adds new methods; T019 is in a different file)
- T021 → T022 ∥ T023 (T023 is in `llvm_ops.cpp`, independent)
- T025 ∥ T026 (different files: `oslexec_pvt.h` vs `shadingsys.cpp`)
- T030 ∥ T031 (different op families, non-conflicting additions)
- T033 ∥ T034 ∥ T035 ∥ T036 (each adds distinct functions; test shader extensions cover different ops)
- T038 ∥ T039 ∥ T040 (different files, different concerns)

---

## Implementation Strategy

### MVP: US1 Only

1. Phases 1–2 (T001–T011) → attribute plumbing + runtime header
2. Phase 3 (T012–T014) → GroupData struct
3. Phase 4 (T015–T020) → virtual interface + layer functions
4. **STOP and VALIDATE**: `ctest -R backend-cpp` passes; generated `.cpp` compiles

### Incremental Delivery

- After Phase 4: inspectable, compilable `.cpp` for any OSL shader (US1 core)
- After Phase 5: automated compilation validation (US2)
- After Phase 6: alternate execution route wired; optimizer-folded shaders match JIT (US3 wiring)
- After Phase 7: output parity verified for connected / default-valued shaders (US3 complete)
- After Phases 8–9: broad op coverage; each sub-phase raises testsuite pass rate
- After Phase 10: all edge cases covered; C++ path is a permanent CI quality gate

### Suggested PR Breakdown

Each task is a PR candidate. Natural PR groupings:
- T001–T007 as one PR (attribute plumbing, all tightly related)
- T008 alone (testsuite scaffold)
- T009–T011 as one PR (runtime header + test extension)
- T012–T014 as one PR (GroupData + test)
- T015–T020 as one PR (virtual interface + layer functions + test)
- T021–T024 as one PR (compile + test)
- T025–T028a as one PR (load/execute + renderer-output write-back) [Phase 6]
- T029a, T029b, T029 as one PR (execution correctness + parity test) [Phase 7]
- T030–T032 as one PR (control flow + test)
- T033, T034, T035, T036, T037 each as separate PRs (one op family + test extension per PR)
- T038–T041 as final cleanup PR

---

## Notes

- No standalone test tasks — testing is woven into each phase; `testsuite/backend-cpp/` grows incrementally
- `[P]` means truly non-overlapping at the source level; safe to parallelize without conflicts
- `osl_cpp_runtime.h` is seeded in T009 and extended by T038 as op families are added
- `op_gen_init()` mutex pattern is already correct — only the first `BackendCpp` instance initializes; subsequent constructors return early

## Phase 11: Testsuite Parity — Deferred Codegen Fixes

**Goal**: Close the remaining feature/codegen gaps found by sweeping the full
testsuite under `OSL_DEBUG_OUTPUT_CPP=3`. 50 tests currently pass at both opt
levels and carry an empty `RUNCPP` marker; each task below unblocks a cluster of
failures — add the `RUNCPP` marker to each newly-passing test as it lands.

> **NOTE (superseded by Phase 12):** the per-test `RUNCPP` opt-in marker
> described here was removed once most tests passed. Coverage is now opt-*out*
> and automatic — see Phase 12. The task notes below are kept for provenance.

**Testing approach**: For each task, sweep the affected tests at *both* opt
levels (`-O0` executes every runtime op, `-O2` may fold it away), precision-mask
any last-bit divergence per the research.md playbook, then add `RUNCPP`. No
regressions in the existing 50.

Ordered roughly by ratio of (tests likely fixed / effort) — small codegen fixes
first, large feature buckets last.

- [X] T042 [P] **printf/format of aggregate components beyond xyz** — `printf_arg_expr` in `src/liboslexec/backendcpp.cpp` indexed a 3-char `"xyz"` member table for every aggregate component, so a `Matrix44` (16 comps) and any `aggregate > 3` type walked off the end into bogus member names (`.n`, `.u`, `.l`, NUL). Now emits `[c]` for vec/color and `[c/4][c%4]` for `Matrix44`. Verified: matrix-printf codegen compiles/runs; zero regressions in the 50 marked tests. Prerequisite for the `matrix` family, which still needs T043 (named-space construction) to fully green.
- [X] T043 [P] **color/vector/matrix construction with a named space** — `cpp_gen_construct` blindly emitted `Type(arg…)`, so a leading string space arg became `Color3("rgb",…)` / `Matrix44("space",…)` (uncompilable) and `matrix(f)` used Imath's all-elements `Matrix44(T)` instead of a diagonal. Added dedicated `cpp_gen_construct_color`, `cpp_gen_construct_triple`, `cpp_gen_matrix` mirroring the JIT: fill components then `osl_prepend_color_from` / `osl_transform_triple[_nonlinear]` (with constant-common short-circuit + renderer nonlinear probe) / `osl_get_from_to_matrix` / `osl_prepend_matrix_from`; matrix scalar form spelled as a 16-float diagonal. Declarations added to `osl_cpp_runtime.h`. **Verified green: `vecctr`** (point/vector/normal in "shader"/"object"/"myspace" + transforms, byte-identical to JIT), zero regressions in the 50. Remaining space-using tests stay blocked on *separate* gaps: `matrix` needs `getmatrix` (T036); `color` segfaults in `osl_luminance_fv` (generic generator doesn't pass the `OpaqueExecContextPtr` to ctx-taking osl_* funcs — see T052); `vector` has a bad `rotate` line.
- [X] T044 **Structs (incl color2/color4/vector2/vector4)** — struct fields flatten into symbols whose names embed a `.` separator (`p.c`, `a.f`), illegal in a C++ identifier and emitted both as local declarations and as GroupData fields (`lay0param_aparam.f`). Fixed by translating `.`→`__` in `Symbol::cpp_safe_name()` (`src/liboslcomp/symtab.cpp`) and switching the GroupData field-name sites in `generate_groupdata_struct()` / load / store / copy-down from raw `sym.name()` to `sym.cpp_safe_name()` so declaration and access stay consistent. Three supporting codegen fixes the struct cluster exposed (each a general bug, not struct-specific): (a) **string assignment** — string vars are `ustringhash` but string constants are raw `uint64_t` hashes (osl_* take string args by-value as `ustringhash_pod`); `cpp_gen_assign` now wraps a string-const source via `OSL::ustringhash::from_hash(...)`, and the `format()` result (also a `ustringhash_pod`) is wrapped the same way; (b) **`%i`** — has no fmtlib presentation type, now spelled `{:d}`; (c) **float-literal precision** — `float_lit` emitted only 6 sig-figs (`{:g}`), so `M_PI` became the lossy `3.14159f` (a *different* float than the JIT's `3.14159274f`); now `{:.9g}` (round-trips any IEEE single). **Verified green (O0+O2, byte-identical to JIT): `struct`, `struct-array`, `struct-array-mixture`, `struct-init-copy`, `struct-isomorphic-overload`, `struct-nested`, `struct-nested-assign`, `struct-nested-deep`, `struct-operator-overload`, `struct-return`, `struct-with-array`, `color2`, `color4`, `vector2`, `vector4`** (15 tests); zero regressions in the prior 58. `struct-layers` stays blocked on closures (T047).
- [X] T045 **Noise variants** — the generic generator mistranslated `noise("cell", …)` into `osl_noise_fsf` (string name as an `s`-typecode arg) instead of canonicalizing the type name into the function symbol. Added a dedicated `cpp_gen_noise` mirroring `llvm_gen_noise`: constant names ("perlin"→`snoise`, "cell"→`cellnoise`, "hash", "simplex"/"usimplex", + periodic) fold into the osl_* base name (no name/sg/options args); **gabor** and **generic** (non-constant name) take the leading `ustringhash_pod` name, the `ShaderGlobals*`, and a `NoiseParams` options struct built from the trailing (token,value) pairs via `osl_init_noise_options` + `osl_noiseparams_set_*`. A value-only variant feeding a deriv-carrying result writes into a `Dual2` temp and copies `.val()` back. Added a layout-mirror `OSL::pvt::NoiseParams` to `osl_cpp_runtime.h` and X-macro declarations for every noise/pnoise/generic/gabor variant (mirroring the `opnoise.cpp` IMPL macros). **Also fixed a triple-deriv multiply bug** surfaced by the filtered-gabor tests: `cpp_gen_binary_op` stripped a scalar operand's derivs (`b.val()`) whenever the result was a *triple*, dropping the product-rule term in `Dual2<Vec3> * Dual2<float>`; now a deriv-carrying result (scalar **or** triple) keeps operand derivs via the `Dual2` chain-rule `operator*` (the add/sub scalar-broadcast path still strips, since the triple ctor has no `Dual2` overload). **Verified green (O0+O2): `cellnoise`, `hashnoise`, `noise-cell`, `noise-gabor`, `noise-gabor2d-filter`, `noise-gabor3d-filter`, `noise-generic`, `noise-perlin`, `noise-simplex`, `pnoise`, `pnoise-cell`, `pnoise-gabor`, `pnoise-generic`, `pnoise-perlin`** (14 tests); zero regressions in the prior 73. The `noise-reg`/`pnoise-reg`/`noise-gabor-reg` regression suites are excluded from the cpp path by their `BATCHED_REGRESSION` marker (harness, not a codegen gap).
- [X] T046 **Textures** — implemented `cpp_gen_texture`, `cpp_gen_texture3d`, `cpp_gen_environment`, `cpp_gen_gettextureinfo`, `cpp_gen_texture_options`; forward-declared all `osl_texture*`/`osl_environment`/`osl_get_textureinfo*`/`osl_init_texture_options`/`osl_texture_set_*` in `osl_cpp_runtime.h`; bonus fixes: string eq/neq via `ustringhash::from_hash()` in `cpp_gen_binary_op`, `Dual2<triple>` strip `.val()` in scalar_str. RUNCPP added to 21 passing texture tests (4 BATCHED_REGRESSION skipped).
- [X] T047 **Closures** — implemented across 5 reviewable sub-commits. **Key de-risking finding**: testshade registers every closure with `prepare=setup=nullptr` (`simplerend.cpp:229`), so the JIT's prepare/setup function-pointer baking is unnecessary; construction reduces to allocate → memset → memcpy formals+keywords → store, and `ClosureComponent::data()` gives the param memory. **T047a** type plumbing: `closure_color_t = const void*` typedef (+`#include <OSL/oslclosure.h>`) in `osl_cpp_runtime.h`; `lang_sym_type_name` emits it for closure scalars (was EMPTY) and arrays; real `closure_color_t layNparam_*` GroupData fields; and a **global write-back** pass (`generate_layer_func` only loaded globals, never stored them — Ci is now copied back to `sg->Ci`). **T047b** `cpp_gen_closure` + keyword fill (registry queried via `find_closure` at codegen time; scalar-const args materialized to temps for address-taking). **T047c** closure add/mul/assign (`osl_add_closure_closure` / `osl_mul_closure_{float,color}` in `cpp_gen_binary_op`; pointer-copy/null in `cpp_gen_assign`). **T047d** printf `%s` via `osl_closure_to_ustringhash` encoded as a `kUstringHash` arg. **T047e** greening: closure connection copy-down (was skipped) + a general `cpp_gen_aassign` string-const wrap (`ustringhash::from_hash`, exposed by `closure-parameters`). **Verified green (O0+O2): `closure`, `closure-conditional`, `closure-layered`, `closure-parameters`, `closure-zero`** (10 cpp tests); 236 total, zero regressions. `closure-array` left unmarked: its per-layer printfs expose the eager-vs-lazy layer-execution order difference (the C++ backend runs layers eagerly per T029b; closure *values* match) — orthogonal to closure codegen.
- [X] T048 **Vec3 (triple) derivatives** — deriv-carrying triples are now promoted to `OSL::Dual2<OSL::Vec3>`/`Dual2<Color3>` (36 bytes val/dx/dy, matching the `osl_*_dv...` ABI). Implemented: deriv-aware `cpp_gen_generic` (mirrors `llvm_gen_generic`: `any_deriv_args`, `arg_typecode(derivs)`, void* for Dual2 args, result-deriv zeroing on the non-deriv path); deriv-aware triple construct/assign-broadcast (`cpp_triple_ctor`), `compref`/`compassign`, printf (`.val()[c]`); real `Dx/Dy/Dz` extracts (`cpp_gen_DxDy`); transform/transformc pass real deriv flags. Added mix deriv inline helpers + fmod deriv declarations to `osl_cpp_runtime.h`. **Verified green: `miscmath`** (the long-standing fmod-deriv divergence) **and `transform`**, both O0+O2; zero regressions. Gate 55. Follow-up: comprehensive deriv-variant declarations (T038) as more deriv tests are added; space-construct of a deriv-triple still fills via `R[c]` (latent).
- [X] T049 [P] **printf of a whole array** — already working: codegen expands whole-array printf into per-element memcpy calls correctly. Added RUNCPP to `printf-whole-array`.
- [X] T050 [P] **reparam of arrays** — interactively-adjusted params were baked into the layer function as constant defaults (`float scale[2] = {5,2}`), so `reparam` never took effect. The JIT reads them from the *interactive arena* (`group().interactive_param_offset(layer, name)` → `interactive_params_ptr + offset`, see `BackendLLVM::getLLVMSymbolBase`). Added the matching branch to `generate_layer_func()`'s param-load loop: for `s.interactive() && !s.connected()` with a valid offset, declare the local and `memcpy` its value from `(char*)interactive_params + offset` (the arena stores `[val][dx][dy]` contiguous, matching the local's layout incl. `Dual2`, so one `sizeof`-byte copy is correct — arrays copy by name, scalars by `&name`). The branch sits after the `connected()` case so connected wins (mirrors the JIT's `interactive && !connected` guard). **Verified green (O0+O2, byte-identical to JIT): `reparam-arrays` (float[2] + color[2]), and the same fix unblocked scalar `reparam` and `reparam-string`** — RUNCPP added to all three; zero regressions across the 224 marked cpp tests.
- [X] T051 **aastep** — did *not* already pass: `filterwidth` was routed through the `generic` generator, which strips the input's derivatives (`osl_filterwidth_ff(x.val())` → the nominal-1.0 stub) — but `filterwidth`'s *input* carries the derivs that define the width while its *result* carries none, so generic's deriv mangling can't express it (exactly why the JIT has a dedicated `llvm_gen_filterwidth`). Added `cpp_gen_filterwidth` mirroring it: deriv-carrying float → `osl_filterwidth_fdf((void*)&x)` (returns the width; assigning to a `Dual2` result zeroes its derivs), deriv-carrying triple → `osl_filterwidth_vdv(&result, &x)` (+ zero result partials if dual), no-deriv input → zero. Also fixed the wrong `osl_filterwidth_fdf` forward decl in `osl_cpp_runtime.h` (was `void (void*,void*)`; real signature is `float (void*)`) and registered `OP(filterwidth, filterwidth)`. **Verified green (O0+O2): `aastep`**; zero regressions across the 226 marked cpp tests.
- [X] T052 [P] **luminance dedicated generator (exec-context + out-ptr)** — `luminance` was mis-registered as `generic`, emitting `result = osl_luminance_fv(&color)`, but the real signature is `osl_luminance_fv(oec, &out, &color)` (returns void, needs the colorsystem from the exec context) → segfault. Added `cpp_gen_luminance` mirroring `llvm_gen_luminance`; declared it (and `_dfdv`) in `osl_cpp_runtime.h` and removed the stale wrong `float osl_luminance_fv(void*)` decl. **Verified green: `color`** (incl. `color("hsv"/"hsl"/"YIQ"/"XYZ",…)` colorspace conversions via T043) at both opt levels; zero regressions. Gate now 52. NOTE the broader pattern persists for other ctx-taking ops still routed to `generic` (dict ops) or commented out (`blackbody`, `wavelength_color`); `luminance-reg` additionally needs T048 (deriv variant `osl_luminance_dfdv`).

### Already resolved (kept for provenance)

- **Matrix component access** (`.y`/`.z` on a `Matrix44`) — fixed by `mxcompref`/`mxcompassign` emitting `[r][c]` (T035). The remaining matrix failures are the *printf* path (T042), not element access.
- **Array codegen** — array-const self-init + whole-array assign (`cbfd6598`); array params declare with `cpp_var_declaration` `[N]` bound, defaults via `cpp_const_literal_str`, memcpy to/from GroupData (`d35e801e`).
- **Precision divergence** — geomath/trig masked with `%.4g`/`%.3g` + pole-nudge; playbook in `research.md`.

---

## Phase 12: Test-scheme inversion — cpp coverage by default for all shader tests

**Goal**: Replace the per-test `RUNCPP` opt-in allowlist with automatic,
opt-*out* coverage gated by a build option. The allowlist was right when only a
few tests passed; once most did, the *unmarked* tests were the interesting ones
(the `render-*` sweep that motivated this found two real bugs — a per-group DSO
filename collision and a missing displacement global write-back).

**Mechanism**:
- New cmake option **`OSL_TEST_CPP_BACKEND`** (default `OFF`; CI variants turn it
  `ON`). Compiling a DSO per group more than doubles testing time, so it's
  opt-in. The `TESTSUITE_CPP` env var still works for ad-hoc local sweeps.
- A test is cpp-eligible iff its `run.py` invokes `testshade`/`testrender`
  (i.e. it actually executes a shader). `src/cmake/testing.cmake` reads each
  `run.py` at configure time (`file(STRINGS ... REGEX)`). This auto-skips the
  `oslc`/`oslinfo`/`python-oslquery` compile- or query-only tests (no cpp
  aspect) with no per-test marker.
- Excluded: `optix` (GPU), `BATCHED_REGRESSION` (batched harness), and any test
  with an explicit **`NOCPP`** marker.
- Removed all `RUNCPP` files. Added `NOCPP` (with explanatory comments) to
  `layers-entry` (needs `--entry` support) and `backend-cpp` (the cpp fixture
  itself — a cpp variant would be circular).
- Found + fixed `dict_find`/`dict_value`/`dict_next` (ctx-op generic-misroute)
  while greening the newly-included `xml` test.

**Result**: with the option `ON`, all eligible tests pass under the C++ backend
at both opt levels.

**Remaining to-dos (deferred, tracked here)**:
- `layers-entry` / explicit per-layer entry points (`--entry`) — **not supported
  in the C++ path and NOT planned**. This feature is a candidate for removal; the
  test carries a `NOCPP` marker and the limitation is documented in spec.md
  (Edge Cases). Only revisit if the feature is kept and C++ parity is wanted —
  it would need per-entry-layer entry functions generated and resolved.
- `example-*` — these execute shaders via their own binaries (not testshade/
  testrender), so the `run.py` grep doesn't include them; wire them for cpp for
  completeness (low priority; some are optix/batched).
- ~~CI~~ DONE (commit 058bb821): `OSL_TEST_CPP_BACKEND=1` set in the `setenvs`
  of the `linux-vfx2026` and `macos26-arm` ci.yml variants.

---

## Phase 13: Cross-platform CI hardening

**Goal**: With the C++ backend enabled in CI on Linux and macOS (Phase 12),
the broader matrix surfaced platform-specific failures the macOS-only local
development never hit. Each was a real portability bug, fixed and verified.

- [X] T053 **Linux: export `osl_*` shadeops to generated DSOs.** Generated DSOs
  failed at load with `undefined symbol: osl_sincos_fff` (and similar). Root
  cause: `src/build-scripts/hidesymbols.map` listed `osl_*` under `local:`,
  stripping the shadeops from liboslexec's dynamic symbol table on Linux —
  overriding the `OSL_DLL_EXPORT` source attribute. (macOS has no version
  script, so it passed.) Fix: move `osl_*` to `global:`; generated DSOs resolve
  them from the already-loaded liboslexec at dlopen time. Added a comment
  marking these symbols INTERNAL/UNSTABLE (exported only for the C++ backend,
  not a public-API contract). Dead-ends ruled out first: runtime
  `dlopen(RTLD_GLOBAL)` promotion and a DT_NEEDED link both fail while the
  symbols are localized — the version script is the only fix.
- [X] T054 **ABI version folds in OSL major/minor.** `OSL_CPP_ABI_VERSION` is now
  `10000*MAJOR + 100*MINOR + revision`, guaranteeing minor releases are
  link-incompatible automatically (the DSOs are ephemeral, so the check only
  guards against misuse, not durable compatibility). The manual `revision`
  digit covers an incompatible change within a single minor cycle. Defined
  identically in `osl_cpp_runtime.h` and `oslexec_pvt.h`; a mismatch is caught
  loudly (every generated DSO fails the load-time ABI check). The
  ABI-mismatch test reference (T040) was made version- and OS-independent by
  redacting the DSO suffix and the ABI number before comparison.
- [X] T055 **MSVC: `__builtin_popcount` → `OSL::popcount`.** The connection
  copy-down used the GCC/Clang-only builtin, which fails to compile under MSVC
  (C3861). `OSL::popcount` (`OSL/mask.h`) dispatches to `__popcnt` on MSVC.
- [X] T056 **Array copy uses `min(dst,src)` length, not `sizeof(dest)`.**
  `cpp_array_copy` emitted `memcpy(R, A, sizeof(R))`; when the source array is
  shorter (OSL allows assigning a shorter array into a longer one), this
  over-read the source and clobbered the destination's retained trailing
  elements with garbage — nondeterministic across platforms (`array-copy.cpp.opt`
  failed on Linux, passed on macOS). Fix: copy `min(dst,src)` elements with
  per-element size spelled `sizeof(R[0])` (covers the `Dual2` deriv layout),
  mirroring `BackendLLVM::llvm_assign_impl`'s `std::min(Result.size(), Src.size())`.

**Checkpoint**: full `OSL_TEST_CPP_BACKEND=1` sweep green on macOS (454/454, both
opt levels) and on the Linux + macOS CI variants.
