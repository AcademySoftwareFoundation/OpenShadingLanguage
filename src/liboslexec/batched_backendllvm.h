// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage


#pragma once

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "oslexec_pvt.h"

using namespace OSL;
using namespace OSL::pvt;

#include "OSL/llvm_util.h"
#include "runtimeoptimize.h"

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>

namespace llvm {
class raw_svector_ostream;
}


OSL_NAMESPACE_ENTER

namespace pvt {  // OSL::pvt



/// OSOProcessor that generates LLVM IR and JITs it to give machine
/// code to implement a shader group.
class BatchedBackendLLVM : public OSOProcessorBase {
public:
    BatchedBackendLLVM(ShadingSystemImpl& shadingsys, ShaderGroup& group,
                       ShadingContext* context, int width);

    // Ensure destructor is in the cpp
    // to allow smart pointers of incomplete types
    virtual ~BatchedBackendLLVM();

    virtual void set_inst(int layer);

    /// Create an llvm function for the whole shader group, JIT it,
    /// and store the llvm::Function* handle to it with the ShaderGroup.
    virtual void run();


    /// What LLVM debug level are we at?
    int llvm_debug() const;

    /// Set up a bunch of static things we'll need for the whole group.
    ///
    void initialize_llvm_group();

    int layer_remap(int origlayer) const { return m_layer_remap[origlayer]; }

    /// Create an llvm function for the current shader instance.
    /// This will end up being the group entry if 'groupentry' is true.
    llvm::Function* build_llvm_instance(bool groupentry);

    /// Create an llvm function for group initialization code.
    llvm::Function* build_llvm_init();

    /// Build up LLVM IR code for the given range [begin,end) or
    /// opcodes, putting them (initially) into basic block bb (or the
    /// current basic block if bb==NULL).
    bool build_llvm_code(int beginop, int endop, llvm::BasicBlock* bb = NULL);

    typedef std::map<std::string, llvm::Value*> AllocationMap;

    void llvm_assign_initial_value(const Symbol& sym,
                                   llvm::Value* llvm_initial_shader_mask_value,
                                   bool force = false);
    llvm::LLVMContext& llvm_context() const { return ll.context(); }
    AllocationMap& named_values() { return m_named_values; }

    /// Return an llvm::Value* corresponding to the address of the given
    /// symbol element, with derivative (0=value, 1=dx, 2=dy) and array
    /// index (NULL if it's not an array).
    llvm::Value* llvm_get_pointer(const Symbol& sym, int deriv = 0,
                                  llvm::Value* arrayindex = NULL);

    /// Allocate a new memory location to store a wide copy of the value
    /// in sym. Optionally pass in the deriv to create wide copy of the deriv.
    llvm::Value* llvm_widen_value_into_temp(const Symbol& sym, int deriv = 0);

    /// Return the llvm::Value* corresponding to the given element
    /// value, with derivative (0=value, 1=dx, 2=dy), array index (NULL
    /// if it's not an array), and component (x=0 or scalar, y=1, z=2).
    /// If deriv >0 and the symbol doesn't have derivatives, return 0
    /// for the derivative.  If the component >0 and it's a scalar,
    /// return the scalar -- this allows automatic casting to triples.
    /// Finally, auto-cast int<->float if requested (no conversion is
    /// performed if cast is the default of UNKNOWN).
    llvm::Value* llvm_load_value(const Symbol& sym, int deriv,
                                 llvm::Value* arrayindex, int component,
                                 TypeDesc cast         = TypeDesc::UNKNOWN,
                                 bool op_is_uniform    = true,
                                 bool index_is_uniform = true);


    /// Given an llvm::Value* of a pointer (and the type of the data
    /// that it points to), Return the llvm::Value* corresponding to the
    /// given element value, with derivative (0=value, 1=dx, 2=dy),
    /// array index (NULL if it's not an array), and component (x=0 or
    /// scalar, y=1, z=2).  If deriv >0 and the symbol doesn't have
    /// derivatives, return 0 for the derivative.  If the component >0
    /// and it's a scalar, return the scalar -- this allows automatic
    /// casting to triples.  Finally, auto-cast int<->float if requested
    /// (no conversion is performed if cast is the default of UNKNOWN).
    llvm::Value* llvm_load_value(llvm::Value* ptr, const TypeSpec& type,
                                 int deriv, llvm::Value* arrayindex,
                                 int component,
                                 TypeDesc cast              = TypeDesc::UNKNOWN,
                                 bool op_is_uniform         = true,
                                 bool index_is_uniform      = true,
                                 bool symbol_forced_boolean = false);

    /// Just like llvm_load_value, but when both the symbol and the
    /// array index are known to be constants.  This can even handle
    /// pulling constant-indexed elements out of constant arrays.  Use
    /// arrayindex==-1 to indicate that it's not an array dereference.
    llvm::Value* llvm_load_constant_value(const Symbol& sym, int arrayindex,
                                          int component,
                                          TypeDesc cast = TypeDesc::UNKNOWN,
                                          bool op_is_uniform = true);

    /// llvm_load_value with non-constant component designation.  Does
    /// not work with arrays or do type casts!
    llvm::Value* llvm_load_component_value(const Symbol& sym, int deriv,
                                           llvm::Value* component,
                                           bool op_is_uniform        = true,
                                           bool component_is_uniform = true);

    /// Non-array version of llvm_load_value, with default deriv &
    /// component.
    llvm::Value* llvm_load_value(const Symbol& sym, int deriv = 0,
                                 int component      = 0,
                                 TypeDesc cast      = TypeDesc::UNKNOWN,
                                 bool op_is_uniform = true)
    {
        return llvm_load_value(sym, deriv, NULL, component, cast,
                               op_is_uniform);
    }

    /// Legacy version
    ///
    llvm::Value* loadLLVMValue(const Symbol& sym, int component = 0,
                               int deriv = 0, TypeDesc cast = TypeDesc::UNKNOWN,
                               bool op_is_uniform = true)
    {
        return llvm_load_value(sym, deriv, NULL, component, cast,
                               op_is_uniform);
    }

    /// Version to handle converting from native mask representation
    /// to LLVM's required vector of bits
    llvm::Value* llvm_load_mask(const Symbol& cond);

    /// Return an llvm::Value* that is either a scalar and derivs is
    /// false, or a pointer to sym's values (if sym is an aggreate or
    /// derivs == true).  Furthermore, if deriv == true and sym doesn't
    /// have derivs, coerce it into a variable with zero derivs.
    llvm::Value* llvm_load_arg(const Symbol& sym, bool derivs,
                               bool is_uniform = true);

    /// Just like llvm_load_arg(sym,deriv), except use use sym's derivs
    /// as-is, no coercion.
    llvm::Value* llvm_load_arg(const Symbol& sym)
    {
        return llvm_load_arg(sym, sym.has_derivs());
    }

    /// Store new_val into the given symbol, given the derivative
    /// (0=value, 1=dx, 2=dy), array index (NULL if it's not an array),
    /// and component (x=0 or scalar, y=1, z=2).  If deriv>0 and the
    /// symbol doesn't have a deriv, it's a nop.  If the component >0
    /// and it's a scalar, set the scalar.  Returns true if ok, false
    /// upon failure.
    bool llvm_store_value(llvm::Value* new_val, const Symbol& sym, int deriv,
                          llvm::Value* arrayindex, int component,
                          bool index_is_uniform = true);

    /// Store new_val into the memory pointed to by dst_ptr, given the
    /// derivative (0=value, 1=dx, 2=dy), array index (NULL if it's not
    /// an array), and component (x=0 or scalar, y=1, z=2).  If deriv>0
    /// and the symbol doesn't have a deriv, it's a nop.  If the
    /// component >0 and it's a scalar, set the scalar.  Returns true if
    /// ok, false upon failure.
    bool llvm_store_value(llvm::Value* new_val, llvm::Value* dst_ptr,
                          const TypeSpec& type, int deriv,
                          llvm::Value* arrayindex, int component,
                          bool index_is_uniform = true);

    /// Non-array version of llvm_store_value, with default deriv &
    /// component.
    bool llvm_store_value(llvm::Value* new_val, const Symbol& sym,
                          int deriv = 0, int component = 0)
    {
        return llvm_store_value(new_val, sym, deriv, NULL, component);
    }

    /// Version to handle converting to native mask representation
    /// from LLVM's required vector of bits
    bool llvm_store_mask(llvm::Value* new_mask, const Symbol& cond);

    /// llvm_store_value with non-constant component designation.  Does
    /// not work with arrays or do type casts!
    bool llvm_store_component_value(llvm::Value* new_val, const Symbol& sym,
                                    int deriv, llvm::Value* component,
                                    bool component_is_uniform = true);

    /// Legacy version
    ///
    bool storeLLVMValue(llvm::Value* new_val, const Symbol& sym,
                        int component = 0, int deriv = 0)
    {
        return llvm_store_value(new_val, sym, deriv, component);
    }

    void llvm_conversion_store_masked_status(llvm::Value* val,
                                             const Symbol& Status);
    void llvm_conversion_store_uniform_status(llvm::Value* val,
                                              const Symbol& Status);

    void llvm_broadcast_uniform_value(llvm::Value* tempUniform,
                                      const Symbol& Destination, int derivs = 0,
                                      int component = 0);
    void llvm_broadcast_uniform_value_from_mem(llvm::Value* pointerTotempUniform,
                                               const Symbol& Destination,
                                               bool ignore_derivs = false);

    /// Generate an alloca instruction to allocate space for the given
    /// type, with derivs if derivs==true, and return the its pointer.
    llvm::Value* llvm_alloca(const TypeSpec& type, bool derivs, bool is_uniform,
                             bool forceBool          = false,
                             const std::string& name = "");

private:
    // We have need to allocate temporaries for function calls that
    // take varying arguments when the symbol's passed ar uniform.
    // We need to broadcast the uniform value to a temporary wide block
    // to pass into the function.
    // We also might need a temporary to hold a uniform result from a function
    // that will then need to be broadcast to a varying symbol afterwards.
    // Rather than having a ton of allocs out there, we will keep track of
    // allocs and reuse them once the go out of scope (usually right after
    // a function call finishes).
    // As llvm is type safe, we track the attributes that affect the underlying
    // type for reuse vs. trying to reuse/cast bytes for different types
    struct TempAlloc {
        bool in_use;
        bool derivs;
        bool is_uniform;
        bool forceBool;
        TypeSpec type;
        llvm::Value* llvm_value;
    };
    std::vector<TempAlloc> m_temp_allocs;

public:
    // Any calls to getOrAllocateTemp during the lifetime of a TempScope
    // will be associated with the latest TempScope on the stack and
    // when that TempScope's lifetime ends, any temp allocations will
    // be marked unused and will be available for reuse by the next
    // call to getOrAllocateTemp
    class TempScope {
        friend class BatchedBackendLLVM;
        BatchedBackendLLVM& m_backend;
        // Avoid dynamic allocations if 14 temps or less
        llvm::SmallVector<int, 14> m_in_use_indices;

    public:
        TempScope(BatchedBackendLLVM& backend);
        TempScope(const TempScope& other)  = delete;
        TempScope(const TempScope&& other) = delete;
        TempScope& operator=(const TempScope& other) = delete;
        TempScope& operator=(const TempScope&& other) = delete;
        ~TempScope();
    };

private:
    friend class TempScope;
    std::vector<TempScope*> m_temp_scopes;

public:
    /// Generate an alloca instruction to allocate space for the given
    /// type, with derivs if derivs==true, and return the its pointer.
    llvm::Value* getOrAllocateTemp(const TypeSpec& type, bool derivs,
                                   bool is_uniform, bool forceBool = false,
                                   const std::string& name = "");

    inline llvm::Value* getTempMask(const std::string& name = "")
    {
        ASSERT(
            !m_temp_scopes.empty()
            && "An instance of BatchedBackendLLVM::TempScope must exist higher up in the call stack");
        return getOrAllocateTemp(TypeSpec(TypeDesc::INT), false /*derivs*/,
                                 false /*is_uniform*/, true /*forceBool*/,
                                 name);
    }


    /// Given the OSL symbol, return the llvm::Value* corresponding to the
    /// address of the start of that symbol (first element, first component,
    /// and just the plain value if it has derivatives).  This is retrieved
    /// from the allocation map if already there; and if not yet in the
    /// map, the symbol is alloca'd and placed in the map.
    llvm::Value* getOrAllocateLLVMSymbol(const Symbol& sym);

    /// Retrieve an llvm::Value that is a pointer holding the start address
    /// of the specified symbol. This always works for globals and params;
    /// for stack variables (locals/temps) is succeeds only if the symbol is
    /// already in the allocation table (will fail otherwise). This method
    /// is not designed to retrieve constants.
    llvm::Value* getLLVMSymbolBase(const Symbol& sym);

    /// Retrieve the named global ("P", "N", etc.).
    /// is_uniform is output parameter
    llvm::Value* llvm_global_symbol_ptr(ustring name, bool& is_uniform);

    /// Test whether val is nonzero, return the llvm::Value* that's the
    /// result of a CreateICmpNE or CreateFCmpUNE (depending on the
    /// type).  If test_derivs is true, it it also tests whether the
    /// derivs are zero.
    llvm::Value* llvm_test_nonzero(const Symbol& val, bool test_derivs = false);

    /// Implementaiton of Simple assignment.  If arrayindex >= 0, in
    /// designates a particular array index to assign.
    bool llvm_assign_impl(const Symbol& Result, const Symbol& Src,
                          int arrayindex = -1, int srccomp = -1,
                          int dstcomp = -1);


    /// Convert the name of a global (and its derivative index) into the
    /// field number of the ShaderGlobals struct.
    int ShaderGlobalNameToIndex(ustring name, bool& is_uniform);

    /// Return the LLVM type handle for the BatchedShaderGlobals struct.
    ///
    llvm::Type* llvm_type_sg();

    /// Return the LLVM type handle for a pointer to a
    /// BatchedShaderGlobals struct.
    llvm::Type* llvm_type_sg_ptr();

    /// Return the LLVM type handle for the BatchedTextureOptions struct.
    ///
    llvm::Type* llvm_type_batched_texture_options();

    /// Return the LLVM type handle for the BatchedTraceOptions struct.
    ///
    llvm::Type* llvm_type_batched_trace_options();

    /// Return the ShaderGlobals pointer.
    ///
    llvm::Value* sg_ptr() const { return m_llvm_shaderglobals_ptr; }

    llvm::Type* llvm_type_closure_component();
    llvm::Type* llvm_type_closure_component_ptr();

    /// Return the ShaderGlobals pointer cast as a void*.
    ///
    llvm::Value* sg_void_ptr() { return ll.void_ptr(m_llvm_shaderglobals_ptr); }

    llvm::Value* llvm_ptr_cast(llvm::Value* val, const TypeSpec& type)
    {
        return ll.ptr_cast(val, type.simpletype());
    }

    llvm::Value* llvm_wide_ptr_cast(llvm::Value* val, const TypeSpec& type)
    {
        return ll.wide_ptr_cast(val, type.simpletype());
    }


    llvm::Value* llvm_void_ptr(const Symbol& sym, int deriv = 0)
    {
        return ll.void_ptr(llvm_get_pointer(sym, deriv));
    }

    /// Return the LLVM type handle for a structure of the common group
    /// data that holds all the shader params.
    llvm::Type* llvm_type_groupdata();

    /// Return the LLVM type handle for a pointer to the common group
    /// data that holds all the shader params.
    llvm::Type* llvm_type_groupdata_ptr();

    /// Return the group data pointer.
    ///
    llvm::Value* groupdata_ptr() const { return m_llvm_groupdata_ptr; }

    /// Return the group data pointer cast as a void*.
    ///
    llvm::Value* groupdata_void_ptr()
    {
        return ll.void_ptr(m_llvm_groupdata_ptr);
    }

    /// Return a reference to the specified field within the group data.
    llvm::Value* groupdata_field_ref(int fieldnum);

    /// Return a pointer to the specified field within the group data,
    /// optionally cast to pointer to a particular data type.
    llvm::Value* groupdata_field_ptr(int fieldnum,
                                     TypeDesc type   = TypeDesc::UNKNOWN,
                                     bool is_uniform = true);


    /// Return a pointer to an WideMatrix that was previously alloca
    /// on the stack, meant for generator to reuse as a temporary
    llvm::Value* temp_wide_matrix_ptr();

    /// Return a pointer to an BatchedTextureOptions that was previously alloca
    /// on the stack, meant for generator to reuse as a temporary
    llvm::Value* temp_batched_texture_options_ptr();

    /// Return a pointer to an BatchedTraceOptions that was previously alloca
    /// on the stack, meant for generator to reuse as a temporary
    llvm::Value* temp_batched_trace_options_ptr();


    /// Return a ref to the bool where the "layer_run" flag is stored for
    /// the specified layer.
    llvm::Value* layer_run_ref(int layer);

    /// Return a ref to the int where the "userdata_initialized" Mask is
    /// stored for the specified userdata index.
    llvm::Value* userdata_initialized_ref(int userdata_index = 0);

    /// Generate LLVM code to zero out the variable (including derivs)
    ///
    void llvm_assign_zero(const Symbol& sym);

    /// Generate LLVM code to zero out the derivatives of sym.
    ///
    void llvm_zero_derivs(const Symbol& sym);

    /// Generate LLVM code to zero out the derivatives of an array
    /// only for the first count elements of it.
    ///
    void llvm_zero_derivs(const Symbol& sym, llvm::Value* count);

    /// Generate a debugging printf at shader execution time.
    void llvm_gen_debug_printf(string_view message);

    /// Generate a warning message at shader execution time.
    void llvm_gen_warning(string_view message);

    /// Generate an error message at shader execution time.
    void llvm_gen_error(string_view message);

    /// Generate code to call the given layer.  If 'unconditional' is
    /// true, call it without even testing if the layer has already been
    /// called.
    void llvm_call_layer(int layer, bool unconditional = false);

    /// Execute the upstream connection (if any, and if not yet run) that
    /// establishes the value of symbol sym, which has index 'symindex'
    /// within the current layer rop.inst().  If already_run is not NULL,
    /// it points to a vector of layer indices that are known to have been
    /// run -- those can be skipped without dynamically checking their
    /// execution status.
    void llvm_run_connected_layers(const Symbol& sym, int symindex,
                                   int opnum                  = -1,
                                   std::set<int>* already_run = NULL);



    // Encapsulate creation of function names that encode parameter types,
    // including if each is varying or uniform, and if a mask is required.
    // Utilize llvm::Twine to efficently combine multiple strings
    // Usage is to start with the function name then append arguments,
    // and at any point masking can be turned on/off as well as batching.
    // The effect of disabling batching would be the function name is
    // not mangled to a target ISA specific library.  IE:
    //        FuncSpec func_spec("foo");
    //        func_spec.arg(resultSym, resultSym.has_derivs(), op_is_uniform);
    //        func_spec.arg(op1Sym, op1Sym.has_derivs(), op_is_uniform);
    //        func_spec.arg(op2Sym, op2Sym.has_derivs(), op_is_uniform);
    //        if (is_masking_required) func_spec.mask();
    //        if (op_is_uniform) func_spec.unbatch();
    //        rop.ll.call_function (rop.build_name(func_spec), ...);
    //
    // NOTE:  build_name will add the "osl_" prefix or "osl_TARGETISA_" prefix
    // not need to include it in the FuncSpec.
    // NOTE: FuncSpec is never meant to be stored, only exist on the stack
    class FuncSpec {
    public:
        class Arg {
            const TypeDesc m_type;
            bool m_derivs;
            bool m_is_uniform;

        public:
            Arg(const TypeDesc& type, bool derivs, bool is_uniform)
                : m_type(type), m_derivs(derivs), m_is_uniform(is_uniform)
            {
            }

            const TypeDesc& type() const { return m_type; }
            bool has_derivs() const { return m_derivs; }
            bool is_uniform() const { return m_is_uniform; }
            bool is_varying() const { return !m_is_uniform; }
        };

    private:
        const llvm::Twine m_name;
        bool m_batched;
        bool m_masked;

        typedef llvm::SmallVector<Arg, 16> ArgVector;
        ArgVector m_args;

    public:
        FuncSpec(const FuncSpec&) = delete;
        FuncSpec& operator=(const FuncSpec&) = delete;

        FuncSpec(const llvm::Twine& name)
            : m_name(name), m_batched(true), m_masked(false)
        {
        }

        FuncSpec(const char* name)
            : m_name(name), m_batched(true), m_masked(false)
        {
        }

        FuncSpec& mask()
        {
            m_masked = true;
            return *this;
        }
        FuncSpec& unmask()
        {
            m_masked = false;
            return *this;
        }

        FuncSpec& batch()
        {
            m_batched = true;
            return *this;
        }

        FuncSpec& unbatch()
        {
            m_batched = false;
            unmask();
            return *this;
        }
        bool is_masked() const { return m_masked; }
        bool is_batched() const { return m_batched; }

        const llvm::Twine& name() const { return m_name; }

        FuncSpec& arg(const TypeDesc& type_desc, bool derivs, bool is_uniform)
        {
            m_args.emplace_back(type_desc, derivs, is_uniform);
            return *this;
        }
        FuncSpec& arg(const Symbol& sym, bool derivs, bool is_uniform)
        {
            OSL_DASSERT(sym.typespec().is_closure() == false);
            OSL_DASSERT(sym.typespec().is_structure() == false);
            return arg(sym.typespec().simpletype(), derivs, is_uniform);
        }
        FuncSpec& arg(const Symbol& sym, bool is_uniform)
        {
            return arg(sym, false /*derivs*/, is_uniform);
        }
        FuncSpec& arg(const TypeDesc& type_desc, bool is_uniform)
        {
            return arg(type_desc, false /*derivs*/, is_uniform);
        }

        FuncSpec& arg_uniform(const TypeDesc& type_desc)
        {
            return arg(type_desc, false /*derivs*/, true /*is_uniform*/);
        }
        FuncSpec& arg_varying(const TypeDesc& type_desc)
        {
            return arg(type_desc, false /*derivs*/, false /*is_uniform*/);
        }
        FuncSpec& arg_varying(const Symbol& sym)
        {
            OSL_DASSERT(sym.typespec().is_closure() == false);
            OSL_DASSERT(sym.typespec().is_structure() == false);
            return arg_varying(sym.typespec().simpletype());
        }

        typedef typename ArgVector::const_iterator const_iterator;
        const_iterator begin() const { return m_args.begin(); }
        const_iterator end() const { return m_args.end(); }
        bool empty() const { return begin() == end(); }
    };

    /// Generate code for a call to the named function with the given
    /// arg list as symbols -- float & ints will be passed by value,
    /// triples and matrices will be passed by address.  If deriv_ptrs
    /// is true, pass pointers even for floats if they have derivs.
    /// Return an llvm::Value* corresponding to the return value of the
    /// function, if any.
    llvm::Value* llvm_call_function(const FuncSpec& name, const Symbol** args,
                                    int nargs, bool deriv_ptrs = false,
                                    bool function_is_uniform       = true,
                                    bool functionIsLlvmInlined     = false,
                                    bool ptrToReturnStructIs1stArg = false);
    llvm::Value* llvm_call_function(const FuncSpec& name, const Symbol& A,
                                    bool deriv_ptrs = false);
    llvm::Value* llvm_call_function(const FuncSpec& name, const Symbol& A,
                                    const Symbol& B, bool deriv_ptrs = false);
    llvm::Value* llvm_call_function(const FuncSpec& name, const Symbol& A,
                                    const Symbol& B, const Symbol& C,
                                    bool deriv_ptrs                = false,
                                    bool function_is_uniform       = true,
                                    bool functionIsLlvmInlined     = false,
                                    bool ptrToReturnStructIs1stArg = false);

    TypeDesc llvm_typedesc(const TypeSpec& typespec)
    {
        return typespec.is_closure_based()
                   ? TypeDesc(TypeDesc::PTR, typespec.arraylength())
                   : typespec.simpletype();
    }

    /// Generate the appropriate llvm type definition for a TypeSpec
    /// (this is the actual type, for example when we allocate it).
    /// Allocates ptrs for closures.
    llvm::Type* llvm_type(const TypeSpec& typespec)
    {
        return ll.llvm_type(llvm_typedesc(typespec));
    }

    llvm::Type* llvm_wide_type(const TypeSpec& typespec)
    {
        // We are the "wide" backend, so all types will be vector types
        return ll.llvm_vector_type(llvm_typedesc(typespec));
    }

    /// Generate the parameter-passing llvm type definition for an OSL
    /// TypeSpec.
    llvm::Type* llvm_pass_type(const TypeSpec& typespec);
    llvm::Type* llvm_pass_wide_type(const TypeSpec& typespec);

    llvm::PointerType* llvm_type_prepare_closure_func()
    {
        return m_llvm_type_prepare_closure_func;
    }
    llvm::PointerType* llvm_type_setup_closure_func()
    {
        return m_llvm_type_setup_closure_func;
    }

    /// Return the basic block of the exit for the whole instance.
    ///
    bool llvm_has_exit_instance_block() const { return m_exit_instance_block; }

    /// Return the basic block of the exit for the whole instance.
    ///
    llvm::BasicBlock* llvm_exit_instance_block()
    {
        if (!m_exit_instance_block) {
            std::string name      = Strutil::sprintf("%s_%d_exit_",
                                                inst()->layername(),
                                                inst()->id());
            m_exit_instance_block = ll.new_basic_block(name);
        }
        return m_exit_instance_block;
    }

    /// Check for inf/nan in all written-to arguments of the op
    void llvm_generate_debugnan(const Opcode& op);
    /// Check for uninitialized values in all read-from arguments to the op
    void llvm_generate_debug_uninit(const Opcode& op);
    /// Print debugging line for the op
    void llvm_generate_debug_op_printf(const Opcode& op);

    llvm::Function* layer_func() const { return ll.current_function(); }

    /// Call this when JITing a texture-like call, to track how many.
    void generated_texture_call(bool handle)
    {
        shadingsys().m_stat_tex_calls_codegened += 1;
        if (handle)
            shadingsys().m_stat_tex_calls_as_handles += 1;
    }

    void llvm_print_mask(const char* title, llvm::Value* mask = nullptr);

    /// Return the userdata index for the given Symbol.  Return -1 if the Symbol
    /// is not an input parameter or is constant and therefore doesn't have an
    /// entry in the groupdata struct.
    int find_userdata_index(const Symbol& sym);

    LLVM_Util ll;


    int vector_width() const { return m_width; }
    int true_mask_value() const { return m_true_mask_value; }

private:
    void append_arg_to(llvm::raw_svector_ostream& OS, const FuncSpec::Arg& arg);

public:
    // Uses internal buffer to store concatenated result,
    // Assuming the backend is only used single threaded,
    // the returned string pointer is only valid until the
    // next call to build_name.
    // The "osl_" prefix or library selector prefix
    // is prepended to the function name during build_name
    const char* build_name(const FuncSpec& func_spec);

private:
    // Helpers to export the actual data member offsets from LLVM's point of view
    // of data structures that exist in C++ so we can validate the offsets match
    template<int WidthT>
    void build_offsets_of_BatchedShaderGlobals(
        std::vector<unsigned int>& offset_by_index);
    template<int WidthT>
    void build_offsets_of_BatchedTextureOptions(
        std::vector<unsigned int>& offset_by_index);

    int m_width;
    int m_true_mask_value;

    // Interface and Factory method to construct a Concrete TargetLibraryHelper
    // that provides a prefix string that all function calls will start with
    // and correctly initialize a function map for the shading system with
    // the functions from the target ISA library.
    class TargetLibraryHelper {
    public:
        virtual ~TargetLibraryHelper() {}
        virtual const char* library_selector() const                        = 0;
        virtual void init_function_map(ShadingSystemImpl& shadingsys) const = 0;

        static std::unique_ptr<TargetLibraryHelper> build(ShadingContext* context,
                                                          int vector_width,
                                                          TargetISA target_isa);
    };
    // TargetLibraryHelper is private, so need to be friend with Concrete
    template <int WidthT, TargetISA IsaT>
    friend class ConcreteTargetLibraryHelper;

    std::unique_ptr<TargetLibraryHelper> m_target_lib_helper;
    const char* m_library_selector;
    const char* m_wide_arg_prefix;
    llvm::SmallString<512> m_built_op_name;


    std::vector<int> m_layer_remap;      ///< Remapping of layer ordering
    std::set<int> m_layers_already_run;  ///< List of layers run
    int m_num_used_layers;               ///< Number of layers actually used

    double m_stat_total_llvm_time;  ///<   total time spent on LLVM
    double m_stat_llvm_setup_time;  ///<     llvm setup time
    double m_stat_llvm_irgen_time;  ///<     llvm IR generation time
    double m_stat_llvm_opt_time;    ///<     llvm IR optimization time
    double m_stat_llvm_jit_time;    ///<     llvm JIT time

    // LLVM stuff
    AllocationMap m_named_values;
    std::map<const Symbol*, int> m_param_order_map;
    llvm::Value* m_llvm_shaderglobals_ptr;
    llvm::Value* m_llvm_groupdata_ptr;

    // Reused allocas for temps used to pass options or intermediates
    llvm::Value* m_llvm_temp_wide_matrix_ptr;  // for gen_tranform
    llvm::Value* m_llvm_temp_batched_texture_options_ptr;
    llvm::Value* m_llvm_temp_batched_trace_options_ptr;

    llvm::BasicBlock* m_exit_instance_block;  // exit point for the instance
    llvm::Type* m_llvm_type_sg;         // LLVM type of ShaderGlobals struct
    llvm::Type* m_llvm_type_groupdata;  // LLVM type of group data
    llvm::Type* m_llvm_type_closure_component;
    llvm::Type* m_llvm_type_batched_texture_options;
    llvm::Type* m_llvm_type_batched_trace_options;
    llvm::PointerType* m_llvm_type_prepare_closure_func;
    llvm::PointerType* m_llvm_type_setup_closure_func;
    int m_llvm_local_mem;  // Amount of memory we use for locals

    friend class ShadingSystemImpl;
};


};  // namespace pvt
OSL_NAMESPACE_EXIT
