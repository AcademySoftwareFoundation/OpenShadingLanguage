// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <map>
#include <vector>

#include "oslexec_pvt.h"
using namespace OSL;
using namespace OSL::pvt;

#include <OSL/llvm_util.h>
#include "runtimeoptimize.h"

// additional includes for creating global OptiX variables
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"


OSL_NAMESPACE_ENTER

namespace pvt {  // OSL::pvt



/// OSOProcessor that generates LLVM IR and JITs it to give machine
/// code to implement a shader group.
class BackendLLVM final : public OSOProcessorBase {
public:
    BackendLLVM(ShadingSystemImpl& shadingsys, ShaderGroup& group,
                ShadingContext* context);

    virtual ~BackendLLVM();

    virtual void set_inst(int layer);

    /// Create an llvm function for the whole shader group, JIT it,
    /// and store the llvm::Function* handle to it with the ShaderGroup.
    virtual void run();

    /// Set additional Module/Function options for the CUDA/OptiX target.
    void prepare_module_for_cuda_jit();



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

    // Create llvm functions for OptiX callables
    std::vector<llvm::Function*> build_llvm_optix_callables();
    llvm::Function* build_llvm_fused_callable();

    /// Build up LLVM IR code for the given range [begin,end) or
    /// opcodes, putting them (initially) into basic block bb (or the
    /// current basic block if bb==NULL).
    bool build_llvm_code(int beginop, int endop, llvm::BasicBlock* bb = NULL);

    typedef std::map<std::string, llvm::Value*> AllocationMap;

    void llvm_assign_initial_value(const Symbol& sym, bool force = false);
    llvm::LLVMContext& llvm_context() const { return ll.context(); }
    AllocationMap& named_values() { return m_named_values; }

    /// Return an llvm::Value* corresponding to the address of the given
    /// symbol element, with derivative (0=value, 1=dx, 2=dy) and array
    /// index (NULL if it's not an array).
    llvm::Value* llvm_get_pointer(const Symbol& sym, int deriv = 0,
                                  llvm::Value* arrayindex = NULL);

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
                                 TypeDesc cast = TypeDesc::UNKNOWN);


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
                                 TypeDesc cast             = TypeDesc::UNKNOWN,
                                 const std::string& llname = {});

    /// Just like llvm_load_value, but when both the symbol and the
    /// array index are known to be constants.  This can even handle
    /// pulling constant-indexed elements out of constant arrays.  Use
    /// arrayindex==-1 to indicate that it's not an array dereference.
    llvm::Value* llvm_load_constant_value(const Symbol& sym, int arrayindex,
                                          int component,
                                          TypeDesc cast = TypeDesc::UNKNOWN);

    /// llvm_load_value with non-constant component designation.  Does
    /// not work with arrays or do type casts!
    llvm::Value* llvm_load_component_value(const Symbol& sym, int deriv,
                                           llvm::Value* component);

    /// Non-array version of llvm_load_value, with default deriv &
    /// component.
    llvm::Value* llvm_load_value(const Symbol& sym, int deriv = 0,
                                 int component = 0,
                                 TypeDesc cast = TypeDesc::UNKNOWN)
    {
        return llvm_load_value(sym, deriv, NULL, component, cast);
    }

    /// Convenience function to load a string for CPU or GPU device
    llvm::Value* llvm_load_string(const Symbol& sym)
    {
        OSL_DASSERT(sym.typespec().is_string());
        return llvm_load_value(sym);
    }

    /// Convenience function to load a constant string for CPU or GPU device.
    /// On the GPU, we use the ustring hash, not the character pointer.
    llvm::Value* llvm_load_string(ustring str) { return ll.constant(str); }

    llvm::Value* llvm_load_string(string_view str)
    {
        return llvm_load_string(ustring(str));
    }

    llvm::Value* llvm_load_stringhash(string_view str)
    {
        return llvm_load_stringhash(ustring(str));
    }

    llvm::Value* llvm_load_stringhash(ustring str)
    {
        return ll.constant64((uint64_t)str.hash());
    }

    /// Legacy version
    ///
    llvm::Value* loadLLVMValue(const Symbol& sym, int component = 0,
                               int deriv = 0, TypeDesc cast = TypeDesc::UNKNOWN)
    {
        return llvm_load_value(sym, deriv, NULL, component, cast);
    }

    /// Return an llvm::Value* in the form that we will pass a float-based
    /// symbol as a function argument to any "built-in" OSL function -- as a
    /// simple value if the symbol is a scalar and no derivs are needed, or as
    /// a pointer to the data in all other cases (aggregates, arrays, or
    /// derivs needed). If deriv == true and sym doesn't have derivs, coerce
    /// it into a variable having derivatives set to 0.0.
    llvm::Value* llvm_load_arg(const Symbol& sym, bool derivs);

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
                          llvm::Value* arrayindex, int component);

    /// Store new_val into the memory pointed to by dst_ptr, given the
    /// derivative (0=value, 1=dx, 2=dy), array index (NULL if it's not
    /// an array), and component (x=0 or scalar, y=1, z=2).  If deriv>0
    /// and the symbol doesn't have a deriv, it's a nop.  If the
    /// component >0 and it's a scalar, set the scalar.  Returns true if
    /// ok, false upon failure.
    bool llvm_store_value(llvm::Value* new_val, llvm::Value* dst_ptr,
                          const TypeSpec& type, int deriv,
                          llvm::Value* arrayindex, int component);

    /// Non-array version of llvm_store_value, with default deriv &
    /// component.
    bool llvm_store_value(llvm::Value* new_val, const Symbol& sym,
                          int deriv = 0, int component = 0)
    {
        return llvm_store_value(new_val, sym, deriv, NULL, component);
    }

    /// llvm_store_value with non-constant component designation.  Does
    /// not work with arrays or do type casts!
    bool llvm_store_component_value(llvm::Value* new_val, const Symbol& sym,
                                    int deriv, llvm::Value* component);

    /// Legacy version
    ///
    bool storeLLVMValue(llvm::Value* new_val, const Symbol& sym,
                        int component = 0, int deriv = 0)
    {
        return llvm_store_value(new_val, sym, deriv, component);
    }

    /// Generate an alloca instruction to allocate space for the given
    /// type, with derivs if derivs==true, and return the its pointer.
    llvm::Value* llvm_alloca(const TypeSpec& type, bool derivs,
                             const std::string& name = "", int align = 0);

    /// Checks if a symbol represents a parameter that can be stored on the
    /// stack instead of in GroupData
    bool can_treat_param_as_local(const Symbol& sym);

    /// Given the OSL symbol, return the llvm::Value* corresponding to the
    /// address of the start of that symbol (first element, first component,
    /// and just the plain value if it has derivatives).  This is retrieved
    /// from the allocation map if already there; and if not yet in the
    /// map, the symbol is alloca'd and placed in the map.
    llvm::Value* getOrAllocateLLVMSymbol(const Symbol& sym);

#if OSL_USE_OPTIX
    /// Return a globally unique (to the JIT module) name for symbol `sym`,
    /// assuming it's part of the currently examined layer of the group.
    std::string global_unique_symname(const Symbol& sym)
    {
        // We need to sanitize the symbol name for PTX compatibility. Also, if
        // the sym name starts with a dollar sign, which are not allowed in
        // PTX variable names, then prepend another underscore.
        auto sym_name = Strutil::replace(sym.name(), ".", "_", true);
        int layer     = sym.layer();
        const ShaderInstance* inst_ = group()[layer];
        return fmtformat("{}{}_{}_{}_{}", sym_name.front() == '$' ? "_" : "",
                         sym_name, group().name(), inst_->layername(), layer);
    }

    /// Allocate a CUDA variable for the given OSL symbol and return a pointer
    /// to the corresponding LLVM GlobalVariable, or return the pointer if it
    /// has already been allocated.
    llvm::Value* getOrAllocateCUDAVariable(const Symbol& sym);

    /// Create a named CUDA global variable with the given type, size, and
    /// alignment, and add it to the current Module. It will be initialized
    /// with data pointed to by init_data. A record will be also added to
    /// m_const_map, and will be retrieved by subsequent calls to
    /// getOrAllocateCUDAVariable().
    llvm::Value* addCUDAGlobalVariable(const std::string& name, int size,
                                       int alignment, const void* init_data,
                                       TypeDesc type = TypeDesc::UNKNOWN);
#endif

    /// Retrieve an llvm::Value that is a pointer holding the start address
    /// of the specified symbol. This always works for globals and params;
    /// for stack variables (locals/temps) is succeeds only if the symbol is
    /// already in the allocation table (will fail otherwise). This method
    /// is not designed to retrieve constants.
    llvm::Value* getLLVMSymbolBase(const Symbol& sym);

    /// Retrieve the named global ("P", "N", etc.).
    llvm::Value* llvm_global_symbol_ptr(ustring name);

    /// Test whether val is nonzero, return the llvm::Value* that's the
    /// result of a CreateICmpNE or CreateFCmpUNE (depending on the
    /// type).  If test_derivs is true, it it also tests whether the
    /// derivs are zero.
    llvm::Value* llvm_test_nonzero(Symbol& val, bool test_derivs = false);

    /// Implementation of Simple assignment.  If arrayindex >= 0, in
    /// designates a particular array index to assign.
    bool llvm_assign_impl(Symbol& Result, Symbol& Src, int arrayindex = -1,
                          int srccomp = -1, int dstcomp = -1);


    /// Convert the name of a global (and its derivative index) into the
    /// field number of the ShaderGlobals struct.
    int ShaderGlobalNameToIndex(ustring name);

    /// Return the LLVM type handle for the ShaderGlobals struct.
    ///
    llvm::Type* llvm_type_sg();

    /// Return the LLVM type handle for a pointer to a
    /// ShaderGlobals struct.
    llvm::Type* llvm_type_sg_ptr();

    /// Return the ShaderGlobals pointer.
    ///
    llvm::Value* sg_ptr() const { return m_llvm_shaderglobals_ptr; }

    llvm::Type* llvm_type_closure_component();
    llvm::Type* llvm_type_closure_component_ptr();

    /// Return the ShaderGlobals pointer cast as a void*.
    ///
    llvm::Value* sg_void_ptr() { return ll.void_ptr(m_llvm_shaderglobals_ptr); }

    /// Cast the pointer variable specified by val to a pointer to the
    /// basic type comprising `type`.
    llvm::Value* llvm_ptr_cast(llvm::Value* val, const TypeSpec& type,
                               const std::string& llname = {})
    {
        return ll.ptr_cast(val, type.simpletype(), llname);
    }

    llvm::Value* llvm_void_ptr(const Symbol& sym, int deriv = 0)
    {
        return ll.void_ptr(llvm_get_pointer(sym, deriv),
                           llnamefmt("{}_voidptr", sym.mangled()));
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
                                     TypeDesc type = TypeDesc::UNKNOWN);

    /// Return the userdata base pointer.
    llvm::Value* userdata_base_ptr() const { return m_llvm_userdata_base_ptr; }

    /// Return the output base pointer.
    llvm::Value* output_base_ptr() const { return m_llvm_output_base_ptr; }

    /// Return the shade index
    llvm::Value* shadeindex() const { return m_llvm_shadeindex; }

    // For a symloc, compute the llvm::Value of the pointer to its true,
    // offset location from the base pointer for shade index `sindex`
    // (which should already be a i64, or if nullptr, then use
    // m_llvm_shadeindex and convert it to i64).
    llvm::Value* symloc_ptr(const SymLocationDesc* symloc,
                            llvm::Value* base_ptr,
                            llvm::Value* sindex = nullptr)
    {
        llvm::Value* offset = ll.constanti64(symloc->offset);
        llvm::Value* stride = ll.constanti64(symloc->stride);
        if (!sindex)
            sindex = ll.op_int_to_longlong(m_llvm_shadeindex);
        llvm::Value* fulloffset = ll.op_add(offset, ll.op_mul(stride, sindex));
        return ll.offset_ptr(base_ptr, fulloffset);
    }

    /// Return a ref to the bool where the "layer_run" flag is stored for
    /// the specified layer.
    llvm::Value* layer_run_ref(int layer);

    /// Return a ref to the bool where the "userdata_initialized" flag is
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
    void llvm_run_connected_layers(Symbol& sym, int symindex, int opnum = -1,
                                   std::set<int>* already_run = NULL);

    /// Generate code for a call to the named function with the given
    /// arg list as symbols -- float & ints will be passed by value,
    /// triples and matrices will be passed by address.  If deriv_ptrs
    /// is true, pass pointers even for floats if they have derivs.
    /// Return an llvm::Value* corresponding to the return value of the
    /// function, if any.
    llvm::Value* llvm_call_function(const char* name, cspan<const Symbol*> args,
                                    bool deriv_ptrs = false);
    llvm::Value* llvm_call_function(const char* name, const Symbol& A,
                                    bool deriv_ptrs = false)
    {
        return llvm_call_function(name, { &A }, deriv_ptrs);
    }
    llvm::Value* llvm_call_function(const char* name, const Symbol& A,
                                    const Symbol& B, bool deriv_ptrs = false)
    {
        return llvm_call_function(name, { &A, &B }, deriv_ptrs);
    }
    llvm::Value* llvm_call_function(const char* name, const Symbol& A,
                                    const Symbol& B, const Symbol& C,
                                    bool deriv_ptrs = false)
    {
        return llvm_call_function(name, { &A, &B, &C }, deriv_ptrs);
    }

    TypeDesc llvm_typedesc(const TypeSpec& typespec)
    {
        if (typespec.is_closure_based())
            return TypeDesc(TypeDesc::PTR, typespec.arraylength());
        else if (use_optix() && typespec.is_string_based()) {
            // On the OptiX side, we use the uint64 hash to represent a string
            return TypeDesc(TypeDesc::UINT64, typespec.arraylength());
        } else
            return typespec.simpletype();
    }

    /// Generate the appropriate llvm type definition for a TypeSpec
    /// (this is the actual type, for example when we allocate it).
    /// Allocates ptrs for closures.
    llvm::Type* llvm_type(const TypeSpec& typespec)
    {
        return ll.llvm_type(llvm_typedesc(typespec));
    }

    /// Generate the appropriate llvm type definition for a pointer to
    /// the type specified by the TypeSpec.
    llvm::Type* llvm_ptr_type(const TypeSpec& typespec)
    {
        return ll.type_ptr(ll.llvm_type(llvm_typedesc(typespec)));
    }

    /// Generate the parameter-passing llvm type definition for an OSL
    /// TypeSpec.
    llvm::Type* llvm_pass_type(const TypeSpec& typespec);

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
            std::string name = llnamefmt("{}_{}_exit_", inst()->layername(),
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

    void increment_useparam_ops() { shadingsys().m_stat_useparam_ops++; }

    /// Return the mapping from symbol names to GlobalVariables.
    std::map<std::string, llvm::GlobalVariable*>& get_const_map()
    {
        return m_const_map;
    }

    /// Return whether or not we are compiling for an OptiX-based renderer.
    bool use_optix() { return m_use_optix; }

    /// Return if we should compile against free function versions of Renderer Service.
    bool use_rs_bitcode() { return m_use_rs_bitcode; }

    /// Return the userdata index for the given Symbol.  Return -1 if the Symbol
    /// is not an input parameter or is constant and therefore doesn't have an
    /// entry in the groupdata struct.
    int find_userdata_index(const Symbol& sym);

    // Helpers to export the actual data member offsets from LLVM's point of view
    // of data structures that exist in C++ so we can validate the offsets match
    void
    build_offsets_of_ShaderGlobals(std::vector<unsigned int>& offset_by_index);

    LLVM_Util ll;

    // Utility for constructing names for llvm symbols. It creates a formatted
    // string if the shading system's "llvm_output_bitcode" option is set,
    // otherwise it takes a shortcut and returns an empty string (since nobody
    // is going to see the pretty bitcode anyway).
    template<typename Str, typename... Args>
    OSL_NODISCARD inline std::string llnamefmt(const Str& fmt,
                                               Args&&... args) const
    {
        return m_name_llvm_syms ? fmtformat(fmt, std::forward<Args>(args)...)
                                : std::string();
    }

private:
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
    llvm::Value* m_llvm_interactive_params_ptr;
    llvm::Value* m_llvm_userdata_base_ptr;
    llvm::Value* m_llvm_output_base_ptr;
    llvm::Value* m_llvm_shadeindex;
    llvm::BasicBlock* m_exit_instance_block;  // exit point for the instance
    llvm::Type* m_llvm_type_sg;         // LLVM type of ShaderGlobals struct
    llvm::Type* m_llvm_type_groupdata;  // LLVM type of group data
    llvm::Type* m_llvm_type_closure_component;  // LLVM type for ClosureComponent
    llvm::PointerType* m_llvm_type_prepare_closure_func;
    llvm::PointerType* m_llvm_type_setup_closure_func;
    int m_llvm_local_mem;   // Amount of memory we use for locals
    bool m_name_llvm_syms;  // Whether to name LLVM symbols

    // A mapping from symbol names to llvm::GlobalVariables
    std::map<std::string, llvm::GlobalVariable*> m_const_map;

    // Name of each indexed field in the groupdata, mostly for debugging.
    std::vector<std::string> m_groupdata_field_names;

    bool m_use_optix;  ///< Compile for OptiX?
    bool m_use_rs_bitcode;  /// To use free function versions of Renderer Service functions.

    friend class ShadingSystemImpl;
};


};  // namespace pvt
OSL_NAMESPACE_EXIT
