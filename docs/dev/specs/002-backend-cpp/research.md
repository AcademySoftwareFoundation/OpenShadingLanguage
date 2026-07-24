# Research: BackendCpp Implementation

**Branch**: `002-backend-cpp` | **Date**: 2026-05-26

## Current State of the Codebase

### What already exists

| Component | File | Status |
|-----------|------|--------|
| `BackendCpp` class | `src/liboslexec/backendcpp.h` | Exists — 66 lines |
| Op generators (nop, assign, binary, unary, construct, if, generic) | `src/liboslexec/backendcpp.cpp` | Exists — 564 lines |
| `debug_output_cpp` attribute | `src/liboslexec/oslexec_pvt.h:962` | Exists as `bool` — needs int |
| `cppgen` field in `OpDescriptor` | `src/liboslexec/oslexec_pvt.h:138` | Exists |
| `OpCppGen` function type | `src/liboslexec/oslexec_pvt.h:129` | Exists |
| `cpp_safe_name()` | `src/liboslcomp/symtab.cpp:45` | Exists — handles `$` prefix only |
| BackendCpp invocation in shadingsys | `src/liboslexec/shadingsys.cpp:3826` | Exists — writes to CWD |

### Critical gap: `m_debug_output_cpp` is `bool`, not `int`

The spec requires an escalating integer (1/2/3). The existing code uses `bool` and the accessor returns `bool`. Changing the member type to `int` and updating `ATTR_SET`/`ATTR_DECODE` is the first required change. The bool-to-int widening is backward-compatible for existing users who set it to `0` or `1`.

### GroupData layout: authoritative source is BackendLLVM

Decision: **BackendCpp reads `sym.dataoffset()` set by the layout pass**

`BackendLLVM::llvm_type_groupdata()` (in `src/liboslexec/llvm_instance.cpp:289`) constructs the struct layout field-by-field:
1. `layer_runflags` bool array (field 0, offset 0, size rounded up to 32-bit boundary)
2. Userdata init flags (int8 array, if nuserdata > 0)
3. Userdata value fields (with float-derivative expansion)
4. Per-layer param fields (connected/interpolated/output params, with derivs × 3)

After this function runs, `sym.dataoffset()` holds each symbol's byte offset within GroupData, and `group().llvm_groupdata_size()` holds the total struct size. BackendCpp reads these directly — no re-computation needed.

### Symbol visibility for `osl_*`

Decision: **Audit `llvm_ops.cpp` and add `OSL_DLL_EXPORT` to `osl_*` functions used by generated code**

These functions are compiled into `liboslexec`. On Linux/macOS with `-fvisibility=hidden`, they won't be exported unless explicitly marked. The set of functions to export is the union of all `osl_*` calls emitted by `cpp_gen_generic` and other generators.

### DSO loading: platform abstraction

Decision: **Use `OIIO::Plugin` (`OpenImageIO/plugin.h`) — platform-independent wrappers over `dlopen`/`LoadLibrary`**

OIIO provides `OIIO::Plugin::open()`, `OIIO::Plugin::getsym()`, and `OIIO::Plugin::close()` which abstract POSIX `dlopen`/`dlsym`/`dlclose` and Windows `LoadLibrary`/`GetProcAddress`/`FreeLibrary`. OSL already depends on OIIO, so no new dependency is added. The DSO handle type is `OIIO::Plugin::Handle` (a `void*` alias).

### Compiler invocation: `popen` on POSIX, `_popen` on Windows

Decision: **`popen` (POSIX) / `_popen` (Windows) to capture compiler stderr**

`std::system()` discards stderr. `popen("cmd 2>&1", "r")` captures both stdout and stderr. Redirect stderr to stdout with `2>&1` appended to the command string.

### Default compiler and flags at CMake configure time

Decision: **Use `CMAKE_CXX_COMPILER` and a platform-appropriate flag set baked in at configure time**

Minimum flags for a compilable shader DSO:
- Linux/macOS: `-shared -fPIC -O1 -I<OSL_INCLUDE_DIR>`
- Windows: `/LD /O1 /I<OSL_INCLUDE_DIR>`

These are baked as string constants in `shadingsys.cpp` via `#define` from CMake.

### JIT skip at level 3

Decision: **Guard `BackendLLVM lljitter(...)` in `shadergroupopt.cpp` with `if (shadingsys().debug_output_cpp() < 3)`**

The layout pass (`lljitter.run()` up through `llvm_type_groupdata()`) must still run because BackendCpp reads `sym.dataoffset()`. Only the remainder of JIT codegen is skipped. In practice, the current code structure in `shadingsys.cpp` calls BackendCpp *after* BackendLLVM — the layout pass is an implicit side effect of the JIT path. For level 3, we need to ensure the layout pass runs before BackendCpp and then JIT codegen is aborted.

**Alternative considered**: Run the layout pass unconditionally and gate only JIT emit. **Chosen**: same, but requires separating the layout pass call from the full JIT run. The `BackendLLVM::run()` function currently does both; may need a `run_layout_only()` variant or early-exit flag.

### `cpp_safe_name()` completeness

Current implementation handles `$` prefix → `___` prefix. C++ reserved words (`int`, `float`, `if`, `while`, etc.) and characters illegal in identifiers are not handled. Since OSL symbol names are constrained by the OSL language grammar (alphanumeric + `_` + `$`), the only illegal-character case is `$`. C++ reserved words could theoretically collide but are unlikely in practice (OSL `int` type is not a symbol name). **Decision**: extend `cpp_safe_name()` with a reserved-word suffix (`_osl`) as a safety measure; file as a separate small change within Phase 4.

## Extensibility: Virtual Interface for Future Language Backends

Decision: **BackendCpp is the base class; language-specific emission points are `virtual`**

The traversal logic (`run()`, `build_cpp_code()`, op dispatch, GroupData struct layout) is language-independent — it reflects the OSL IR structure. The leaf emission points that vary across C++ and similar target languages are:

| What varies | Virtual method | C++ default |
|-------------|----------------|-------------|
| Scalar/aggregate type names | `lang_type_name(TypeDesc)` | `"float"`, `"int"`, `"ustringhash"`, … |
| Symbol type (with derivs) | `lang_sym_type_name(Symbol&)` | `"Dual2<float>"`, `"Vec3"`, … |
| File preamble / includes | `lang_preamble()` | `#include "osl_cpp_runtime.h"` |
| Per-function qualifier | `lang_function_qualifier()` | `""` |
| Linkage specifier | `lang_linkage_prefix()` | `extern "C"` |
| Output file extension | `lang_file_extension()` | `".cpp"` |
| Pointer syntax | `lang_ptr_syntax()` | `"*"` |

The existing `cpp_typedesc_name()` and `cpp_sym_type_name()` ARE the first two rows above — rename and virtualize in Phase 4. No new method slots needed beyond what's already called.

**Not virtual**: `cpp_safe_name()` (identifier escaping is language-independent), `build_cpp_code()` (op dispatch structure is shared), GroupData struct layout (mirrors LLVM, same for all targets sharing the ABI).

No subclasses are implemented in this feature. The design obligation is that Phase 4 establishes the virtual interface so a future language backend subclass needs only to override the `lang_*` methods.

## Alternatives Considered

| Decision | Alternative | Rejected Because |
|----------|-------------|-----------------|
| `popen` for compiler invocation | `std::system()` | Discards stderr; user can't diagnose compile failures |
| `popen` for compiler invocation | Platform subprocess API | More complex, overkill for a debug path |
| `OIIO::Plugin` for DSO load/unload | Raw `dlopen`/`LoadLibrary` | OIIO::Plugin already abstracts platform differences; OSL already depends on OIIO; no new dependency needed |
| `OIIO::Plugin::close()` in ShaderGroup destructor | Close at ShadingSystem teardown | Destructor is more precise; avoids accumulating handles across many shader compilations in a long-running renderer |
| Layout pass before BackendCpp | Independent layout in BackendCpp | Would duplicate complex logic from BackendLLVM; single source of truth is LLVM layout |
| `OSL_CPP_ABI_VERSION` in `oslexec_pvt.h` | Separate header | Keeps it adjacent to `RunLLVMGroupFunc` which it tracks |

## Testsuite: how cpp coverage is gated (Phase 12)

Coverage is opt-*out* and automatic (the old per-test `RUNCPP` opt-in marker was
removed). `src/cmake/testing.cmake` creates `.cpp`/`.cpp.opt` variants for every
test whose `run.py` invokes `testshade`/`testrender`, gated by the
`OSL_TEST_CPP_BACKEND` option (default OFF — cpp more than doubles test time; CI
variants turn it ON) or the `TESTSUITE_CPP` env var (ad-hoc). The option is
declared with `set_option` (set_utils.cmake), so it's also settable from the
environment or the makefile wrapper, e.g. `OSL_TEST_CPP_BACKEND=1`. Excluded:
`optix`, `BATCHED_REGRESSION`, and tests carrying a `NOCPP` marker (currently
`layers-entry` and `backend-cpp`). To sweep locally: configure with the option
on (`cmake -DOSL_TEST_CPP_BACKEND=ON .` or `OSL_TEST_CPP_BACKEND=1 cmake .`) then
`ctest -j8 --timeout 300 -R '\.cpp(\.opt)?$'`.

## Testsuite: small precision differences vs the JIT

A test may pass under the C++ backend (`OSL_DEBUG_OUTPUT_CPP=3`) yet differ from
the JIT reference only in the last printed digit(s). This is **expected FP
drift, not a codegen bug**.

**Cause:** both paths run the same ops on the same inputs, but the JIT *inlines*
the `osl_*` builtins (from `llvm_ops.cpp`) and fuses/reassociates them (FMA)
with surrounding ops, while the C++ DSO *calls* them across a non-inlinable
boundary into pre-compiled `liboslexec`. Float math is non-associative, so the
last bits differ — amplified by long chains (e.g. `fresnel`:
`normalize -> dot -> sqrt -> divide -> asin`) into the 5th-6th printed digit.
Same class of drift that produced the per-platform `ref/out-*.txt` variants.

**Does NOT help:** `pretty()` (only flushes `|x|<5e-6` to zero, not precision);
`-ffp-contract=off` on the DSO (drift is inside the `osl_*` calls, not the
shader's inline math). The `.txt` compare (`runtest.py text_diff`) is exact —
masking must happen in the shader's print precision.

**Fix:** reduce the drifting `printf` args from `%g` to **`%.4g`** (echoed
constants can stay). Mind rounding boundaries: `%.5g` can still split a value
(`0.862064`/`0.862068` -> `.86206`/`.86207`); if one value sits exactly on the
`%.4g` boundary (`0.779349`/`0.77935`), drop *just that arg* to `%.3g` with a
comment. Then regenerate `ref/out.txt` from the JIT path and confirm the C++
backend is byte-identical. This usually also collapses the per-platform variants
— if so, delete them for a single `ref/out.txt`.

**Example:** `testsuite/geomath` (commit `85367ad0`) — `%.4g` on
`fresnel`/`refract`, `%.3g` on the one boundary component; four variants dropped.

**Singularities (a sharper variant):** at a pole, the same compilation
difference explodes from last-digit into different *magnitudes*. `testsuite/trig`
(commit `54e54e98`) evaluated `tan(M_PI/2)` exactly on the pole, where `fast_tan`
gave `2.28773e+07` (JIT) vs `276244` (C++). Reduced precision cannot mask
different magnitudes — instead nudge the input off the singularity (`M_PI/2 *
0.99`) so the function is finite and stable, then apply the precision playbook to
the residual last-digit drift. `0.999` was too close (still 0.3% apart); `0.99`
agreed at `%.4g`.

## Derivatives on triples (Vec3) — implemented (T048)

Derivative-carrying triples are now promoted to `OSL::Dual2<OSL::Vec3>` /
`OSL::Dual2<OSL::Color3>` — 36 contiguous bytes (`val,dx,dy`) matching the
`osl_*_dv...` deriv-triple void* ABI. The implementation mirrors the scalar
`Dual2<float>` path:

- **Declaration** (`lang_sym_type_name`): a deriv-carrying float scalar *or*
  triple is wrapped in `OSL::Dual2<...>`.
- **`cpp_gen_generic`** is deriv-aware (mirrors `llvm_gen_generic`): it computes
  `any_deriv_args`, mangles with `arg_typecode(derivs)`, passes a `void*` to the
  `Dual2` storage for deriv args, and on the non-deriv path zeroes a
  deriv-carrying triple result's partials (the runtime variant writes only the
  value).
- **construct / assign-broadcast** build a `Dual2<Vec3>` from per-component
  `val/dx/dy` (`cpp_triple_ctor`); **`compref`/`compassign`** index
  `.val()[c]` / `.dx()[c]` / `.dy()[c]`; **printf** reads `.val()[c]`.
- **`Dx`/`Dy`** are real `.dx()`/`.dy()` extracts (`cpp_gen_DxDy`); `Dz` is 0
  (`Dual2<…,2>` stores only two partials).
- **transform/transformc** pass the real `Pin_derivs`/`Pout_derivs` flags.

Some ops have no native deriv `osl_*` function (e.g. `mix`, an inline helper) —
their deriv variants are added as `Dual2` inline helpers in
`osl_cpp_runtime.h`; ops with native deriv variants (e.g. `fmod`) just need the
forward declaration. Verified: `testsuite/miscmath` (the former fmod-deriv
divergence) and `testsuite/transform` are byte-identical to the JIT at both opt
levels.

Latent edges: the *space*-construct path of `point/vector/normal(space,…)` still
fills components via `R[c]`, which won't compile for a deriv-triple result (no
current test hits it); `Dz` of `P` is 0 rather than the true `dPdz`.
