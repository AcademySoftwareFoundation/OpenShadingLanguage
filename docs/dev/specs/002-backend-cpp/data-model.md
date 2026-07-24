# Data Model: BackendCpp

**Branch**: `002-backend-cpp` | **Date**: 2026-05-26

## Key Entities

### BackendCpp (existing, extended)

`OSOProcessorBase` subclass. Lives in `src/liboslexec/backendcpp.{h,cpp}`.

**New members added in this feature**:

| Member | Type | Purpose |
|--------|------|---------|
| `m_out` | `std::ostringstream` | Accumulates generated C++ source (already exists) |
| `m_indentlevel` / `m_indentview` | `int` / `string_view` | Indentation tracking (already exists) |
| *(no new members needed — generation state is transient)* | | |

**New methods**:

| Method | Virtual? | Phase | Purpose |
|--------|----------|-------|---------|
| `generate_groupdata_struct()` | no | 3 | Emit typed `struct GroupData { ... };` |
| `generate_layer_func(int layer)` | no | 4 | Emit one layer function |
| `generate_group_entry()` | no | 4 | Emit `RunLLVMGroupFunc`-compatible entry |
| `compile_to_dso()` | no | 5 | Shell out to compiler via `popen`, capture errors |
| `load_dso()` | no | 6 | `OIIO::Plugin::open()`, ABI check, store handle |

**Virtual interface** — language-specific seam points that subclasses override:

| Method | Default (C++) | Purpose |
|--------|--------------|---------|
| `lang_type_name(TypeDesc)` | `"float"`, `"int"`, `"ustringhash"`, … | Scalar/aggregate type name in target language |
| `lang_sym_type_name(Symbol&)` | `"Dual2<float>"`, `"Vec3"`, … | Full symbol type including derivative wrapper |
| `lang_preamble()` | `#include "osl_cpp_runtime.h"` | File header and include directives |
| `lang_function_qualifier()` | `""` | Per-function qualifier (empty for C++) |
| `lang_linkage_prefix()` | `extern "C"` | Linkage specifier for exported symbols |
| `lang_file_extension()` | `".cpp"` | Output file extension |
| `lang_ptr_syntax()` | `"*"` | Pointer declarator token |

The traversal logic (`run()`, `build_cpp_code()`, `generate_groupdata_struct()`, op dispatching) is non-virtual — shared across all language backends. Only the leaf emission points are virtual.

`cpp_typedesc_name()` and `cpp_sym_type_name()` (already in `backendcpp.h`) are renamed to `lang_type_name()` / `lang_sym_type_name()` and made virtual in Phase 4. The `cpp_` prefix would be misleading in subclasses.

---

### ShadingSystemImpl (existing, extended)

**New/changed members** (`src/liboslexec/oslexec_pvt.h`):

| Member | Old Type | New Type | Default | Purpose |
|--------|----------|----------|---------|---------|
| `m_debug_output_cpp` | `bool` | `int` | `0` | Escalating level: 1/2/3 |
| `m_cpp_output_dir` | — | `std::string` | `"."` | Where `.cpp` and DSO files are written |
| `m_cpp_compiler` | — | `std::string` | *(CMake-baked)* | Compiler executable path |
| `m_cpp_compiler_flags` | — | `std::string` | *(CMake-baked)* | Compilation flags |

**New accessor**:
```cpp
int  debug_output_cpp() const { return m_debug_output_cpp; }  // was bool
std::string_view cpp_output_dir() const { return m_cpp_output_dir; }
std::string_view cpp_compiler() const { return m_cpp_compiler; }
std::string_view cpp_compiler_flags() const { return m_cpp_compiler_flags; }
```

**Attribute names** (ShadingSystem::attribute):
- `"debug_output_cpp"` → `m_debug_output_cpp` (int)
- `"cpp_output_dir"` → `m_cpp_output_dir` (string)
- `"cpp_compiler"` → `m_cpp_compiler` (string)
- `"cpp_compiler_flags"` → `m_cpp_compiler_flags` (string)

**Env var**: `OSL_DEBUG_OUTPUT_CPP` → `atoi()` → `m_debug_output_cpp` (read in ShadingSystemImpl constructor, same pattern as `OSL_LLVM_DEBUG` at `shadingsys.cpp:1243`)

---

### ShaderGroup (existing, extended)

**New members** (`src/liboslexec/oslexec_pvt.h`):

| Member | Type | Purpose |
|--------|------|---------|
| `m_cpp_dso_handle` | `OIIO::Plugin::Handle` | DSO handle from `OIIO::Plugin::open()`; `nullptr` when not loaded |
| `m_cpp_compiled_version` | `RunLLVMGroupFunc` | Entry point resolved from DSO |

**Lifecycle**: `m_cpp_dso_handle` initialized to `nullptr`. Set by `BackendCpp::load_dso()` via `OIIO::Plugin::open()`. Closed via `OIIO::Plugin::close(m_cpp_dso_handle)` in `ShaderGroup` destructor if non-null.

---

### OpDescriptor (existing, unchanged)

```cpp
struct OpDescriptor {
    // ...existing fields...
    OpCppGen cppgen { nullptr };  // already exists — null → NO CPP GENERATOR stub
};
```

`OpCppGen` = `bool (*)(BackendCpp&, int opnum)` — already defined in `oslexec_pvt.h:129`.

---

### Generated File Structure

One `.cpp` file per shader group, written to `cpp_output_dir/group-cpp-<name>.cpp`.

```
#include "osl_cpp_runtime.h"

// --- GroupData ---
struct GroupData {
    bool layer_runflags[N];          // field 0: rounded up to 32-bit boundary
    // ... userdata flags and values if any ...
    // ... per-layer connected/output param fields ...
};

// --- ABI version export ---
extern "C" int osl_cpp_abi_version() { return OSL_CPP_ABI_VERSION; }

// --- Layer functions ---
static void layer_0_name(ShaderGlobals* sg, GroupData* gd,
                         void* userdata_base, void* output_base,
                         int shadeindex, void* interactive_params) {
    // local decls
    // op statements
}
// ... one per active layer ...

// --- Group entry function (matches RunLLVMGroupFunc) ---
// As shipped, the exported entry symbol is osl_init_group_<name> (resolved by
// BackendCpp::load_dso); the generated DSO exports exactly this one entry.
extern "C" void osl_init_group_<name>(void* sg_, void* gd_,
                             void* userdata_base, void* output_base,
                             int shadeindex, void* interactive_params) {
    ShaderGlobals* sg = (ShaderGlobals*)sg_;
    GroupData* gd = (GroupData*)gd_;
    if (!gd->layer_runflags[N-1])
        layer_N_name(sg, gd, userdata_base, output_base,
                     shadeindex, interactive_params);
}
```

---

### `osl_cpp_runtime.h` (new)

Internal header included by every generated `.cpp` file. Not installed or part of the public API.

**Contents**:
- `#pragma once`
- Forward-include of OSL types needed by generated code (`ShaderGlobals`, `Dual2<T>`, etc.)
- `constexpr int OSL_CPP_ABI_VERSION = 10000 * OSL_VERSION_MAJOR + 100 * OSL_VERSION_MINOR + revision;` (as shipped; `revision = 1`) — folds in the OSL major/minor version so minor releases are link-incompatible automatically. Defined identically in `oslexec_pvt.h`; a mismatch fails the load-time ABI check loudly. (See tasks.md Phase 13 / T054.)
- `extern "C"` declarations for all `osl_*` functions referenced by generated code

---

### Generated File Naming

| Artifact | Name pattern | Location |
|----------|-------------|----------|
| C++ source | `group-cpp-<name>.cpp` | `cpp_output_dir/` |
| Shared library (Linux) | `group-cpp-<name>.so` | `cpp_output_dir/` |
| Shared library (macOS) | `group-cpp-<name>.dylib` | `cpp_output_dir/` |
| Shared library (Windows) | `group-cpp-<name>.dll` | `cpp_output_dir/` |

`<name>` = `group.name()` passed through `cpp_safe_name()` to ensure valid filesystem characters.

---

## State Transitions

```
debug_output_cpp value → pipeline stages executed
──────────────────────────────────────────────────
0  →  nothing (existing JIT path unchanged)
1  →  generate .cpp → write to cpp_output_dir
2  →  generate .cpp → write → compile via popen → .so/.dylib/.dll
3  →  generate .cpp → write → compile → load DSO → ABI check →
       store entry point in ShaderGroup → skip JIT → execute via DSO
```

Error handling at each transition (no automatic fallback to JIT):
- Write failure → `ShadingSystem::errorfmt()`
- Compile failure → capture compiler stderr via `popen` → `errorfmt()`, group failed
- Load failure → `errorfmt()`, group failed
- ABI mismatch → `errorfmt()`, group failed
