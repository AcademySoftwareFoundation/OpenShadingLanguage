// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/export.h>
#include <OSL/oslversion.h>
#include <OSL/oslconfig.h>

#include <vector>
#include <unordered_set>

#ifdef LLVM_NAMESPACE
namespace llvm = LLVM_NAMESPACE;
#endif

namespace llvm {
  class BasicBlock;
  class ConstantFolder;
  class DIBuilder;
  class DICompileUnit;
  class DIFile;
  class DILocation;
  class DIScope;
  class DISubprogram;
  class DISubroutineType;
  class ExecutionEngine;
  class Function;
  class FunctionType;
  class SectionMemoryManager;
  class JITEventListener;
  class Linker;
  class LLVMContext;
  class Module;
  class PointerType;
  class Type;
  class Value;
  class VectorType;

  namespace legacy {
    class FunctionPassManager;
    class PassManager;
  }
}



OSL_NAMESPACE_ENTER

namespace pvt {   // OSL::pvt


enum class TargetISA
{
    UNKNOWN,
    NONE,
    x64,
    SSE4_2,
    AVX,
    AVX2,
    AVX2_noFMA,
    AVX512,
    AVX512_noFMA,
    HOST,
    COUNT
};



/// Wrapper class around LLVM functionality.  This handles all the
/// gory details of actually dealing with LLVM.  It should be sufficiently
/// generic that it would be useful for any LLVM-JITing app, and is not
/// tied to OSL internals at all.
class OSLEXECPUBLIC LLVM_Util {
public:
    struct OSLEXECPUBLIC PerThreadInfo {
        PerThreadInfo() {}
        ~PerThreadInfo();
    private:
        friend class LLVM_Util;
        struct Impl;
        mutable Impl* m_thread_info = nullptr;
        Impl* get() const;
    };

    LLVM_Util (const PerThreadInfo &per_thread_info,
               int debuglevel = 0, int vector_width = 4);
    ~LLVM_Util ();

    // JIT'd code needs to exist with a longer lifetime than the LLVM_Util object.
    // To enable better cleanup at shutdown, the lifetime of all JIT'd code is
    // controlled by the the existence of ScopedJitMemoryUser objects.
    // When the last ScopedJitMemoryUser goes out of scope or is deleted,
    // then the underlying memory managers will be deleted
    struct OSLEXECPUBLIC ScopedJitMemoryUser {
        ScopedJitMemoryUser();
        ~ScopedJitMemoryUser();
    };

    /// Set debug level
    void debug (int d) { m_debug = d; }
    int debug () const { return m_debug; }
    void dumpasm(bool val) { m_dumpasm = val; }
    bool dumpasm() const { return m_dumpasm; }
    void jit_fma(bool val) { m_jit_fma = val; }
    bool jit_fma() const { return m_jit_fma; }
    void jit_aggressive(bool val) { m_jit_aggressive = val; }
    bool jit_aggressive() const { return m_jit_aggressive; }

    /// Return a reference to the current context.
    llvm::LLVMContext &context () const { return *m_llvm_context; }

    /// Return a pointer to the current module.  Make a new one if
    /// necessary.
    llvm::Module *module () {
        if (! m_llvm_module)
            m_llvm_module = new_module();
        return m_llvm_module;
    }

    /// Set the current module to m.
    void module (llvm::Module *m) {
        m_llvm_module = m;
        m_ModuleIsFinalized = false;
        m_ModuleIsPruned = false;
    }

    /// Create a new empty module.
    llvm::Module *new_module (const char *id = "default");

    /// Create a new module, populated with functions from the buffer
    /// bitcode[0..size-1].  The name identifies the buffer.  If err is not
    /// NULL, error messages will be stored there.
    llvm::Module *module_from_bitcode (const char *bitcode, size_t size,
                                       const std::string &name=std::string(),
                                       std::string *err=NULL);

    bool debug_is_enabled() const;
    void debug_setup_compilation_unit(const char * compile_unit_name);
    void debug_push_function(const std::string & function_name,
                             OIIO::ustring sourcefile, int sourceline);
    void debug_pop_function();
    void debug_push_inlined_function(OIIO::ustring function_name,
                             OIIO::ustring sourcefile, int sourceline);
    void debug_pop_inlined_function();
    void debug_set_location(OIIO::ustring sourcefile, int sourceline);

    /// Create a new function (that will later be populated with
    /// instructions) with up to 4 args.
    llvm::Function *make_function (const std::string &name, bool fastcall,
                                   llvm::Type *rettype,
                                   llvm::Type *arg1=NULL,
                                   llvm::Type *arg2=NULL,
                                   llvm::Type *arg3=NULL,
                                   llvm::Type *arg4=NULL);

    /// Create a new function (that will later be populated with
    /// instructions) with a vector of args.
    llvm::Function *make_function (const std::string &name, bool fastcall,
                                   llvm::Type *rettype,
                                   const std::vector<llvm::Type*> &paramtypes,
                                   bool varargs=false);

    /// Add a global mapping of a function to its callable address
    /// explicitly instead of relying on dlsym.
    void add_function_mapping (llvm::Function *func, void *addr);

    /// Set up a new current function that subsequent basic blocks will
    /// be added to.
    void current_function (llvm::Function *func) { m_current_function = func; }

    /// Return a ptr to the current function we're generating.
    llvm::Function *current_function () const { return m_current_function; }

    /// Return the value ptr for the a-th argument of the current function.
    llvm::Value *current_function_arg (int a);


    /// Create a new IR builder with the given block as entry point. If
    /// block is NULL, a new basic block for the current function will be
    /// created.
    void new_builder (llvm::BasicBlock *block=NULL);

    /// End the current builder
    void end_builder ();

    /// Create a new JITing ExecutionEngine and make it the current one.
    /// Return a pointer to the new engine.  If err is not NULL, put any
    /// errors there.
    /// Optionally request a specific ISA for JIT on the host
    ///     ["x64", "SSE4.2", "AVX", "AVX2", "AVX512"]
    ///     (ignored if requested ISA not valid for host)
    /// Optionally enable debugging symbols (source file & line number)
    /// Optionally enable profiling events
    llvm::ExecutionEngine* make_jit_execengine (std::string *err = nullptr,
                         TargetISA requestedISA = TargetISA::NONE,
                         bool debugging_symbols = false,
                         bool profiling_events = false);

    /// Report the host's TargetISA as chosen by the last call to
    /// make_jit_execengine() or to detect_cpu_features(). Don't call
    /// target_isa() unless one of those has previously been called.
    TargetISA target_isa() const { return m_target_isa; }

    // Check support for certain CPU ISA features. These are only valid
    // after detect_cpu_features() (or make_jit_execengine()) has been
    // called.
    bool supports_avx() const { return m_supports_avx; }
    bool supports_avx2() const { return m_supports_avx2; }
    bool supports_avx512f() const { return m_supports_avx512f; }
    bool supports_llvm_bit_masks_natively() const { return m_supports_llvm_bit_masks_natively; }
    bool supports_masked_stores() const { return m_supports_masked_stores; }

    // Does this host support the requested target ISA?
    static bool supports_isa(TargetISA target);

    // Look up the TargetISA enum by name. Special names: "host" means
    // figure out what this host is, "none" or "" mean to use the baseline
    // or no special ops not availabe on all flavors of this family of CPUs.
    static TargetISA lookup_isa_by_name(string_view target_name = "");

    // Name for this TargetISA enum.
    static const char* target_isa_name(TargetISA isa);

    // For CPU compilation, inventory the host CPU capabilities. You can
    // optionally request a specific ISA by name. If `no_fma` is true,
    // specifically pretend there is no FMA capability, even if the hardware
    // supports it. Return true if ok, false if it couldn't figure it out.
    bool detect_cpu_features(TargetISA requestedISA = TargetISA::UNKNOWN,
                             bool no_fma = false);

    /// Return a pointer to the current ExecutionEngine.  Create a JITing
    /// ExecutionEngine if one isn't already set up.
    llvm::ExecutionEngine *execengine () {
        if (! m_llvm_exec)
            make_jit_execengine();
        return m_llvm_exec;
    }

    void dump_struct_data_layout(llvm::Type *Ty);
    void validate_struct_data_layout(llvm::Type *Ty,
              const std::vector<unsigned int>& expected_offset_by_index);

    /// Replace the ExecutionEngine (pass NULL to simply delete the
    /// current one).
    void execengine (llvm::ExecutionEngine *exec);

    enum class Linkage {
        External, // Externally visible
        LinkOnceODR, // One Definition Rule:  Inline version, but allow replacement by equivalent.
        Internal, // Treat as static functions.
        Private // Treat as static functions, but omit from symbol table.
    };

    /// Identify exactly which functions need to exist in the module based
    /// on actual usage from a set of external_functions. All unneeded
    /// functions are removed before we move to optimization which is much
    /// faster.  The external_functions will be set to have external linkage
    /// and the remaining functions set to internal linkage.
    void prune_and_internalize_module (
        std::unordered_set<llvm::Function*> external_functions,
        Linkage default_linkage = Linkage::Internal,
        std::string *out_err = nullptr);

    // OLD, might deprecate later
    /// Change symbols in the module that are marked as having external
    /// linkage to an alternate linkage that allows them to be discarded if
    /// not used within the module. Only do this for functions that start
    /// with prefix, and that DON'T match anything in the two exceptions
    /// lists.
    void internalize_module_functions (const std::string &prefix,
                                       const std::vector<std::string> &exceptions,
                                       const std::vector<std::string> &moreexceptions);

    /// Setup LLVM optimization passes.
    /// if targetHost is true, passes to target the host will be added
    void setup_optimization_passes (int optlevel, bool target_host=true);

    /// Run the optimization passes.
    void do_optimize (std::string *err = NULL);

    /// Retrieve a callable pointer to the JITed version of a function.
    /// This will JIT the function if it hasn't already done so. Be sure
    /// you have already called do_optimize() if you want optimization.
    void *getPointerToFunction (llvm::Function *func);

    /// Wrap ExecutionEngine::InstallLazyFunctionCreator.
    void InstallLazyFunctionCreator (void* (*P)(const std::string &));


    /// Create a new LLVM basic block (for the current function) and return
    /// its handle.
    llvm::BasicBlock *new_basic_block (const std::string &name=std::string());

    /// Save the return block pointer when entering a function. If
    /// after==NULL, generate a new basic block for where to go after the
    /// function return.  Return the after BB.
    llvm::BasicBlock *push_function (llvm::BasicBlock *after=NULL);

    /// Pop basic return destination when exiting a function.  This includes
    /// resetting the IR insertion point to the block following the
    /// corresponding function call.
    void pop_function ();

    /// Are we inside a function?
    bool inside_function() const;

    /// Return the basic block where we go after returning from the current
    /// function.
    llvm::BasicBlock *return_block () const;

    /// Save the basic block pointers when entering a loop.
    void push_loop (llvm::BasicBlock *step, llvm::BasicBlock *after);

    /// Pop basic block pointers when exiting a loop.
    void pop_loop ();

    /// Return the basic block of the current loop's 'step' instructions.
    llvm::BasicBlock *loop_step_block () const;

    /// Return the basic block of the current loop's exit point.
    llvm::BasicBlock *loop_after_block () const;


    llvm::Type *type_float() const { return m_llvm_type_float; }
    llvm::Type *type_double() const { return m_llvm_type_double; }
    llvm::Type *type_int() const { return m_llvm_type_int; }
    llvm::Type *type_int8() const { return m_llvm_type_int8; }
    llvm::Type *type_int16() const { return m_llvm_type_int16; }
    llvm::Type *type_addrint() const { return m_llvm_type_addrint; }
    llvm::Type *type_bool() const { return m_llvm_type_bool; }
    llvm::Type *type_char() const { return m_llvm_type_char; }
    llvm::Type *type_longlong() const { return m_llvm_type_longlong; }
    llvm::Type *type_void() const { return m_llvm_type_void; }
    llvm::Type *type_triple() const { return m_llvm_type_triple; }
    llvm::Type *type_matrix() const { return m_llvm_type_matrix; }
    llvm::Type *type_typedesc() const { return m_llvm_type_longlong; }
    llvm::PointerType *type_void_ptr() const { return m_llvm_type_void_ptr; }
    llvm::PointerType *type_string() { return m_llvm_type_char_ptr; }
    llvm::PointerType *type_ustring_ptr() const { return m_llvm_type_ustring_ptr; }
    llvm::PointerType *type_char_ptr() const { return m_llvm_type_char_ptr; }
    llvm::PointerType *type_bool_ptr() const { return m_llvm_type_bool_ptr; }
    llvm::PointerType *type_int_ptr() const { return m_llvm_type_int_ptr; }
    llvm::PointerType *type_float_ptr() const { return m_llvm_type_float_ptr; }
    llvm::PointerType *type_longlong_ptr() const { return m_llvm_type_longlong_ptr; }
    llvm::PointerType *type_triple_ptr() const { return m_llvm_type_triple_ptr; }
    llvm::PointerType *type_matrix_ptr() const { return m_llvm_type_matrix_ptr; }
    llvm::PointerType *type_double_ptr() const { return m_llvm_type_double_ptr; }

    llvm::Type *type_wide_float() const { return m_llvm_type_wide_float; }
    llvm::Type *type_wide_double() const { return m_llvm_type_wide_double; }
    llvm::Type *type_wide_int() const { return m_llvm_type_wide_int; }
    llvm::Type *type_wide_bool() const { return m_llvm_type_wide_bool; }
    llvm::Type *type_wide_char() const { return m_llvm_type_wide_char; }
    llvm::Type *type_wide_longlong() const { return m_llvm_type_wide_longlong; }
    llvm::Type *type_wide_triple() const { return m_llvm_type_wide_triple; }
    llvm::Type *type_wide_matrix() const { return m_llvm_type_wide_matrix; }
    llvm::Type *type_wide_void_ptr() const { return m_llvm_type_wide_void_ptr; }
    llvm::Type *type_wide_string() const { return m_llvm_type_wide_ustring_ptr; }
    llvm::PointerType *type_wide_char_ptr() const { return m_llvm_type_wide_char_ptr; }
    llvm::PointerType *type_wide_bool_ptr() const { return m_llvm_type_wide_bool_ptr; }
    llvm::PointerType *type_wide_int_ptr() const { return m_llvm_type_wide_int_ptr; }
    llvm::PointerType *type_wide_float_ptr() const { return m_llvm_type_wide_float_ptr; }

    // Different ISA's may have different representations of a mask from
    // llvm's vector of bits that comparison operations emit. And we need
    // varying boolean symbols to have the correct data type (size) on the
    // stack and in data structures.
    llvm::Type *type_native_mask() const { return m_llvm_type_native_mask; }

    /// Generate the appropriate llvm type definition for a TypeDesc
    /// (this is the actual type, for example when we allocate it).
    llvm::Type *llvm_type (const OIIO::TypeDesc &typedesc);

    /// Generate the appropriate llvm vector type definition for a TypeDesc
    /// (this is the actual type, for example when we allocate it).
    llvm::Type *llvm_vector_type (const OIIO::TypeDesc &typedesc);

    /// This will return a llvm::Type that is the same as a C union of
    /// the given types[].
    llvm::Type *type_union (const std::vector<llvm::Type *> &types);

    /// This will return a llvm::Type that is the same as a C struct
    /// comprised fields of the given types[], in order.
    llvm::Type *type_struct (const std::vector<llvm::Type *> &types,
                             const std::string &name="", bool is_packed=false);


    /// Return the llvm::Type that is a pointer to the given llvm type.
    llvm::Type *type_ptr (llvm::Type *type);

    /// Return the llvm::Type that is an array of n elements of the given
    /// llvm type.
    llvm::Type *type_array (llvm::Type *type, int n);

    /// Return an llvm::FunctionType that describes a function with the
    /// given return types, parameter types (in a vector), and whether it
    /// uses varargs conventions.
    llvm::FunctionType *type_function (llvm::Type *rettype,
                                       const std::vector<llvm::Type*> &params,
                                       bool varargs=false);

    /// Return a llvm::PointerType that's a pointer to the described
    /// kind of function.
    llvm::PointerType *type_function_ptr (llvm::Type *rettype,
                                          const std::vector<llvm::Type*> &params,
                                          bool varargs=false);

    /// Return the human-readable name of the type of the llvm type.
    std::string llvm_typename (llvm::Type *type) const;

    /// Return the llvm::Type of the llvm value.
    llvm::Type *llvm_typeof (llvm::Value *val) const;

    /// Return the human-readable name of the type of the llvm value.
    std::string llvm_typenameof (llvm::Value *val) const;

    /// Return an llvm::Value holding the given floating point constant.
    llvm::Value *constant (float f);

    /// Return an llvm::Value holding the given integer constant.
    llvm::Value *constant (int i);

    /// Return an llvm::Value holding the given integer constant.
    llvm::Value *constant8 (int i);
    llvm::Value *constant16 (uint16_t i);
    llvm::Value *constant64 (uint64_t i);
    llvm::Value *constant128 (uint64_t i);
    llvm::Value *constant128 (uint64_t left, uint64_t right);

    /// Return an llvm::Value holding the given size_t constant.
    llvm::Value *constant (size_t i);

    /// Return an llvm::Value holding the given bool constant.
    /// Change the name so it doesn't get mixed up with int.
    llvm::Value *constant_bool (bool b);

    /// Return a constant void pointer to the given constant address.
    /// If the type specified is NULL, it will make a 'void *'.
    llvm::Value *constant_ptr (void *p, llvm::PointerType *type=NULL);

    /// Return an llvm::Value holding the given string constant.
    llvm::Value *constant (ustring s);
    llvm::Value *constant (string_view s) {
        return constant(ustring(s));
    }

    /// Return an llvm::Value for a long long that is a packed
    /// representation of a TypeDesc.
    llvm::Value *constant (const OIIO::TypeDesc &type);

    // Return "wide" (SIMD vector) constants of various types.
    llvm::Value *wide_constant (float f);
    llvm::Value *wide_constant (int i);
    llvm::Value *wide_constant (size_t i);
    llvm::Value *wide_constant_bool (bool b);
    llvm::Value *wide_constant (ustring s);
    llvm::Value *wide_constant (string_view s) {
        return wide_constant(ustring(s));
    }

    /// Return an llvm::Value holding wide version of the given scalar
    /// constant.
    llvm::Value *wide_constant (llvm::Value *constant_val);

    /// Return an llvm::Value for a void* variable with value NULL.
    llvm::Value *void_ptr_null ();

    /// Cast the pointer variable specified by val to the kind of pointer
    /// described by type (as an llvm pointer type).
    llvm::Value *ptr_cast (llvm::Value* val, llvm::Type *type);
    llvm::Value *ptr_cast (llvm::Value* val, llvm::PointerType *type) {
        return ptr_cast (val, (llvm::Type *)type);
    }

    /// Cast the pointer variable specified by val to a pointer to the type
    /// described by type (as an llvm data type).
    llvm::Value *ptr_to_cast (llvm::Value* val, llvm::Type *type);

    /// Cast the pointer variable specified by val to a pointer to the given
    /// data type, return the llvm::Value of the new pointer.
    llvm::Value *ptr_cast (llvm::Value* val, const OIIO::TypeDesc &type);

    llvm::Value *wide_ptr_cast (llvm::Value* val, const OIIO::TypeDesc &type);

    /// Cast the variable specified by val to a pointer of type void*,
    /// return the llvm::Value of the new pointer.
    llvm::Value *int_to_ptr_cast (llvm::Value* val);

    /// Cast the pointer variable specified by val to a pointer of type
    /// void* return the llvm::Value of the new pointer.
    llvm::Value *void_ptr (llvm::Value* val);

    /// Generate a pointer that is (ptrtype)((char *)ptr + offset).
    /// If ptrtype is NULL, just return a void*.
    llvm::Value *offset_ptr (llvm::Value *ptr, int offset,
                             llvm::Type *ptrtype=NULL);

    /// Generate an alloca instruction to allocate space for n copies of the
    /// given llvm type, and return its pointer.
    llvm::Value *op_alloca (llvm::Type *llvmtype, int n=1,
                            const std::string &name=std::string(), int align=0);
    llvm::Value *op_alloca (llvm::PointerType *llvmtype, int n=1,
                            const std::string &name=std::string(), int align=0) {
        return op_alloca ((llvm::Type *)llvmtype, n, name, align);
    }

    /// Generate an alloca instruction to allocate space for n copies of the
    /// given type, and return its pointer.
    llvm::Value *op_alloca (const OIIO::TypeDesc &type, int n=1,
                            const std::string &name=std::string(), int align=0);

    /// Generate an alloca instruction to allocate space for n copies of the
    /// given type, and return its pointer.
    llvm::Value *wide_op_alloca (const OIIO::TypeDesc &type, int n=1,
                                 const std::string &name=std::string(), int align=0);

    /// Generate code for a call to the function pointer, with the given
    /// arg list.  Return an llvm::Value* corresponding to the return
    /// value of the function, if any.
    llvm::Value *call_function (llvm::Value *func, cspan<llvm::Value *> args);
    /// Generate code for a call to the named function with the given arg
    /// list.  Return an llvm::Value* corresponding to the return value of
    /// the function, if any.
    llvm::Value *call_function (const char *name, cspan<llvm::Value *> args);

    llvm::Value *call_function (const char *name, llvm::Value *arg0) {
        return call_function (name, cspan<llvm::Value*>(&arg0, 1));
    }
    llvm::Value *call_function (const char *name, llvm::Value *arg0,
                                llvm::Value *arg1) {
        return call_function (name, { arg0, arg1 });
    }
    llvm::Value *call_function (const char *name, llvm::Value *arg0,
                                llvm::Value *arg1, llvm::Value *arg2) {
        return call_function (name, { arg0, arg1, arg2 });
    }
    llvm::Value *call_function (const char *name, llvm::Value *arg0,
                                llvm::Value *arg1, llvm::Value *arg2,
                                llvm::Value *arg3) {
        return call_function (name, { arg0, arg1, arg2, arg3 });
    }

    /// Mark the function call (which MUST be the value returned by a
    /// call_function()) as using the 'fast' calling convention.
    void mark_fast_func_call (llvm::Value *funccall);

    /// Set the code insertion point for subsequent ops to block.
    void set_insert_point (llvm::BasicBlock *block);

    /// Return op from a void function.  If retval is NULL, we are returning
    /// from a void function.
    void op_return (llvm::Value *retval=NULL);

    /// Create a branch instruction to block and establish that as the as
    /// the new code insertion point.
    void op_branch (llvm::BasicBlock *block);

    /// Create a conditional branch instruction to trueblock if cond is
    /// true, to falseblock if cond is false, and establish trueblock as the
    /// new insertion point).
    void op_branch (llvm::Value *cond, llvm::BasicBlock *trueblock,
                    llvm::BasicBlock *falseblock);

    /// Generate code for a memset.
    void op_memset (llvm::Value *ptr, int val, int len, int align=1);

    /// Generate code for variable size memset
    void op_memset (llvm::Value *ptr, int val, llvm::Value *len, int align=1);

    /// Generate code for a memcpy.
    void op_memcpy (llvm::Value *dst, llvm::Value *src, int len, int align=1);

    /// Generate code for a memcpy.
    void op_memcpy (llvm::Value *dst, int dstalign,
                    llvm::Value *src, int srcalign, int len);

    /// Dereference a pointer:  return *ptr
    llvm::Value *op_load (llvm::Value *ptr);

    /// Store to a dereferenced pointer:   *ptr = val
    void op_store (llvm::Value *val, llvm::Value *ptr);

    // N.B. "GEP" -- GetElementPointer -- is a particular LLVM-ism that is
    // the means for retrieving elements from some kind of aggregate: the
    // i-th field in a struct, the i-th element of an array.  They can be
    // chained together, to get at items in a recursive hierarchy.

    /// Generate a GEP (get element pointer) where the element index is an
    /// llvm::Value, which can be generated from either a constant or a
    /// runtime-computed integer element index.
    llvm::Value *GEP (llvm::Value *ptr, llvm::Value *elem);

    /// Generate a GEP (get element pointer) with an integer element
    /// offset.
    llvm::Value *GEP (llvm::Value *ptr, int elem);

    /// Generate a GEP (get element pointer) with two integer element
    /// offsets.  This is just a special (and common) case of GEP where
    /// we have a 2-level hierarchy and we have fixed element indices
    /// that are known at compile time.
    llvm::Value *GEP (llvm::Value *ptr, int elem1, int elem2);

    // Arithmetic ops.  It auto-detects the type (int vs float).
    // ...
    llvm::Value *op_add (llvm::Value *a, llvm::Value *b);
    llvm::Value *op_sub (llvm::Value *a, llvm::Value *b);
    llvm::Value *op_neg (llvm::Value *a);
    llvm::Value *op_mul (llvm::Value *a, llvm::Value *b);
    llvm::Value *op_div (llvm::Value *a, llvm::Value *b);
    llvm::Value *op_mod (llvm::Value *a, llvm::Value *b);
    llvm::Value *op_float_to_int (llvm::Value *a);
    llvm::Value *op_int_to_float (llvm::Value *a);
    llvm::Value *op_bool_to_int (llvm::Value *a);
    llvm::Value *op_bool_to_float (llvm::Value *a);
    llvm::Value *op_int_to_bool (llvm::Value *a);
    llvm::Value *op_float_to_double (llvm::Value *a);
    llvm::Value *op_int_to_longlong (llvm::Value *a);

    llvm::Value *op_and (llvm::Value *a, llvm::Value *b);
    llvm::Value *op_or (llvm::Value *a, llvm::Value *b);
    llvm::Value *op_xor (llvm::Value *a, llvm::Value *b);
    llvm::Value *op_shl (llvm::Value *a, llvm::Value *b);
    llvm::Value *op_shr (llvm::Value *a, llvm::Value *b);
    llvm::Value *op_not (llvm::Value *a);

    /// Generate IR for (cond ? a : b).  Cond should be a bool.
    llvm::Value *op_select (llvm::Value *cond, llvm::Value *a, llvm::Value *b);

    /// Extracts a scalar value from a vector type
    llvm::Value *op_extract(llvm::Value *a, int index);
    llvm::Value *op_extract(llvm::Value *a, llvm::Value * index);
    llvm::Value *op_insert(llvm::Value *v, llvm::Value *a, int index);

    // Comparison ops.  It auto-detects the type (int vs float).
    // ordered only applies to float comparisons -- ordered means the
    // comparison will succeed only if neither arg is NaN.
    // ...
    llvm::Value *op_eq (llvm::Value *a, llvm::Value *b, bool ordered=false);
    llvm::Value *op_ne (llvm::Value *a, llvm::Value *b, bool ordered=false);
    llvm::Value *op_gt (llvm::Value *a, llvm::Value *b, bool ordered=false);
    llvm::Value *op_lt (llvm::Value *a, llvm::Value *b, bool ordered=false);
    llvm::Value *op_ge (llvm::Value *a, llvm::Value *b, bool ordered=false);
    llvm::Value *op_le (llvm::Value *a, llvm::Value *b, bool ordered=false);

    llvm::Value *op_fabs(llvm::Value *v);
    llvm::Value *op_is_not_finite(llvm::Value *v);

    /// Write the module's bitcode (after compilation/optimization) to a
    /// file.  If err is not NULL, errors will be deposited there.
    void write_bitcode_file (const char *filename, std::string *err=NULL);

    /// Generate PTX for the current Module and return it as a string
    bool ptx_compile_group (llvm::Module* lib_module, const std::string& name,
                            std::string& out);

    /// Convert a whole module's bitcode to a string.
    std::string bitcode_string (llvm::Module *module);

    /// Convert one function's bitcode to a string.
    std::string bitcode_string (llvm::Function *func);

    /// Delete the IR for the body of the given function to reclaim its
    /// memory (only helpful if we know we won't use it again).
    void delete_func_body (llvm::Function *func);

    /// Is the function empty, except for simply a ret statement?
    bool func_is_empty (llvm::Function *func);

    std::string func_name (llvm::Function *f);

    static size_t total_jit_memory_held ();

private:
    class MemoryManager;
    class IRBuilder;

    void SetupLLVM ();
    IRBuilder& builder();

    int m_debug;
    bool m_dumpasm = false;
    bool m_jit_fma = false;
    bool m_jit_aggressive = false;
    PerThreadInfo::Impl *m_thread;
    llvm::LLVMContext *m_llvm_context;
    llvm::Module *m_llvm_module;
    IRBuilder *m_builder;
    llvm::SectionMemoryManager *m_llvm_jitmm;
    llvm::Function *m_current_function;
    llvm::legacy::PassManager *m_llvm_module_passes;
    llvm::legacy::FunctionPassManager *m_llvm_func_passes;
    llvm::ExecutionEngine *m_llvm_exec;
    TargetISA m_target_isa = TargetISA::UNKNOWN;
    std::vector<llvm::BasicBlock *> m_return_block;     // stack for func call
    std::vector<llvm::BasicBlock *> m_loop_after_block; // stack for break
    std::vector<llvm::BasicBlock *> m_loop_step_block;  // stack for continue

    llvm::Type *m_llvm_type_float;
    llvm::Type *m_llvm_type_double;
    llvm::Type *m_llvm_type_int;
    llvm::Type *m_llvm_type_int8;
    llvm::Type *m_llvm_type_int16;
    llvm::Type *m_llvm_type_addrint;
    llvm::Type *m_llvm_type_bool;
    llvm::Type *m_llvm_type_char;
    llvm::Type *m_llvm_type_longlong;
    llvm::Type *m_llvm_type_void;
    llvm::Type *m_llvm_type_triple;
    llvm::Type *m_llvm_type_matrix;
    llvm::PointerType *m_llvm_type_void_ptr;
    llvm::PointerType *m_llvm_type_ustring_ptr;
    llvm::PointerType *m_llvm_type_char_ptr;
    llvm::PointerType *m_llvm_type_bool_ptr;
    llvm::PointerType *m_llvm_type_int_ptr;
    llvm::PointerType *m_llvm_type_float_ptr;
    llvm::PointerType *m_llvm_type_longlong_ptr;
    llvm::PointerType *m_llvm_type_triple_ptr;
    llvm::PointerType *m_llvm_type_matrix_ptr;
    llvm::PointerType *m_llvm_type_double_ptr;

    int m_vector_width;
    llvm::Type * m_llvm_type_wide_float;
    llvm::Type * m_llvm_type_wide_double;
    llvm::Type * m_llvm_type_wide_int;
    llvm::Type * m_llvm_type_wide_bool;
    llvm::Type * m_llvm_type_wide_char;
    llvm::Type * m_llvm_type_wide_longlong;
    llvm::Type * m_llvm_type_wide_triple;
    llvm::Type * m_llvm_type_wide_matrix;
    llvm::Type * m_llvm_type_wide_void_ptr;
    llvm::Type * m_llvm_type_wide_ustring_ptr;
    llvm::PointerType * m_llvm_type_wide_char_ptr;
    llvm::PointerType * m_llvm_type_wide_int_ptr;
    llvm::PointerType * m_llvm_type_wide_bool_ptr;
    llvm::PointerType * m_llvm_type_wide_float_ptr;
    llvm::Type * m_llvm_type_native_mask;

    bool m_supports_masked_stores = false;
    bool m_supports_llvm_bit_masks_natively = false;
    bool m_supports_avx512f = false;
    bool m_supports_avx2 = false;
    bool m_supports_avx = false;

    // Profiling Info
    llvm::JITEventListener* mVTuneNotifier;

    // Debug Info
    llvm::DIFile * getOrCreateDebugFileFor(const std::string &file_name);
    llvm::DIScope * getCurrentDebugScope() const;
    llvm::DILocation *getCurrentInliningSite() const;

    llvm::DIBuilder* m_llvm_debug_builder;
    llvm::DICompileUnit *mDebugCU;
    std::vector<llvm::DIScope *> mLexicalBlocks;

    typedef std::unordered_map<std::string, llvm::DIFile *> FileByNameType;
    FileByNameType mDebugFileByName;
    std::vector<llvm::DILocation *> mInliningSites;
    llvm::DISubroutineType * mSubTypeForInlinedFunction;
    bool m_ModuleIsFinalized;
    bool m_ModuleIsPruned;
};



}; // namespace pvt
OSL_NAMESPACE_EXIT
