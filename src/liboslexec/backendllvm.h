/*
Copyright (c) 2009-2013 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <vector>
#include <map>

#include "oslexec_pvt.h"
using namespace OSL;
using namespace OSL::pvt;

#include "runtimeoptimize.h"
#include <OSL/llvm_util.h>

// additional includes for creating global OptiX variables
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"


OSL_NAMESPACE_ENTER

namespace pvt {   // OSL::pvt



/// OSOProcessor that generates LLVM IR and JITs it to give machine
/// code to implement a shader group.
class BackendLLVM : public OSOProcessorBase {
public:
    BackendLLVM (ShadingSystemImpl &shadingsys, ShaderGroup &group,
                ShadingContext *context);

    virtual ~BackendLLVM ();

    virtual void set_inst (int layer);

    /// Create an llvm function for the whole shader group, JIT it,
    /// and store the llvm::Function* handle to it with the ShaderGroup.
    virtual void run ();


    /// What LLVM debug level are we at?
    int llvm_debug() const;

    /// Set up a bunch of static things we'll need for the whole group.
    ///
    void initialize_llvm_group ();

    int layer_remap (int origlayer) const { return m_layer_remap[origlayer]; }

    /// Create an llvm function for the current shader instance.
    /// This will end up being the group entry if 'groupentry' is true.
    llvm::Function* build_llvm_instance (bool groupentry);

    /// Create an llvm function for group initialization code.
    llvm::Function* build_llvm_init ();

    /// Build up LLVM IR code for the given range [begin,end) or
    /// opcodes, putting them (initially) into basic block bb (or the
    /// current basic block if bb==NULL).
    bool build_llvm_code (int beginop, int endop, llvm::BasicBlock *bb=NULL);

    typedef std::map<std::string, llvm::Value*> AllocationMap;

    void llvm_assign_initial_value (const Symbol& sym, bool force = false);
    llvm::LLVMContext &llvm_context () const { return ll.context(); }
    AllocationMap &named_values () { return m_named_values; }

    /// Return an llvm::Value* corresponding to the address of the given
    /// symbol element, with derivative (0=value, 1=dx, 2=dy) and array
    /// index (NULL if it's not an array).
    llvm::Value *llvm_get_pointer (const Symbol& sym, int deriv=0,
                                   llvm::Value *arrayindex=NULL);

    /// Return the llvm::Value* corresponding to the given element
    /// value, with derivative (0=value, 1=dx, 2=dy), array index (NULL
    /// if it's not an array), and component (x=0 or scalar, y=1, z=2).
    /// If deriv >0 and the symbol doesn't have derivatives, return 0
    /// for the derivative.  If the component >0 and it's a scalar,
    /// return the scalar -- this allows automatic casting to triples.
    /// Finally, auto-cast int<->float if requested (no conversion is
    /// performed if cast is the default of UNKNOWN).
    llvm::Value *llvm_load_value (const Symbol& sym, int deriv,
                                  llvm::Value *arrayindex, int component,
                                  TypeDesc cast=TypeDesc::UNKNOWN);


    /// Given an llvm::Value* of a pointer (and the type of the data
    /// that it points to), Return the llvm::Value* corresponding to the
    /// given element value, with derivative (0=value, 1=dx, 2=dy),
    /// array index (NULL if it's not an array), and component (x=0 or
    /// scalar, y=1, z=2).  If deriv >0 and the symbol doesn't have
    /// derivatives, return 0 for the derivative.  If the component >0
    /// and it's a scalar, return the scalar -- this allows automatic
    /// casting to triples.  Finally, auto-cast int<->float if requested
    /// (no conversion is performed if cast is the default of UNKNOWN).
    llvm::Value *llvm_load_value (llvm::Value *ptr, const TypeSpec &type,
                              int deriv, llvm::Value *arrayindex,
                              int component, TypeDesc cast=TypeDesc::UNKNOWN);

    /// Just like llvm_load_value, but when both the symbol and the
    /// array index are known to be constants.  This can even handle
    /// pulling constant-indexed elements out of constant arrays.  Use
    /// arrayindex==-1 to indicate that it's not an array dereference.
    llvm::Value *llvm_load_constant_value (const Symbol& sym,
                                           int arrayindex, int component,
                                           TypeDesc cast=TypeDesc::UNKNOWN);

    /// llvm_load_value with non-constant component designation.  Does
    /// not work with arrays or do type casts!
    llvm::Value *llvm_load_component_value (const Symbol& sym, int deriv,
                                            llvm::Value *component);

    /// Non-array version of llvm_load_value, with default deriv &
    /// component.
    llvm::Value *llvm_load_value (const Symbol& sym, int deriv = 0,
                                  int component = 0,
                                  TypeDesc cast=TypeDesc::UNKNOWN) {
        return llvm_load_value (sym, deriv, NULL, component, cast);
    }

    /// Legacy version
    ///
    llvm::Value *loadLLVMValue (const Symbol& sym, int component=0,
                                int deriv=0, TypeDesc cast=TypeDesc::UNKNOWN) {
        return llvm_load_value (sym, deriv, NULL, component, cast);
    }

    /// Return an llvm::Value* that is either a scalar and derivs is
    /// false, or a pointer to sym's values (if sym is an aggreate or
    /// derivs == true).  Furthermore, if deriv == true and sym doesn't
    /// have derivs, coerce it into a variable with zero derivs.
    llvm::Value *llvm_load_arg (const Symbol& sym, bool derivs);

    /// Just like llvm_load_arg(sym,deriv), except use use sym's derivs
    /// as-is, no coercion.
    llvm::Value *llvm_load_arg (const Symbol& sym) {
        return llvm_load_arg (sym, sym.has_derivs());
    }

    /// Store new_val into the given symbol, given the derivative
    /// (0=value, 1=dx, 2=dy), array index (NULL if it's not an array),
    /// and component (x=0 or scalar, y=1, z=2).  If deriv>0 and the
    /// symbol doesn't have a deriv, it's a nop.  If the component >0
    /// and it's a scalar, set the scalar.  Returns true if ok, false
    /// upon failure.
    bool llvm_store_value (llvm::Value *new_val, const Symbol& sym, int deriv,
                           llvm::Value *arrayindex, int component);

    /// Store new_val into the memory pointed to by dst_ptr, given the
    /// derivative (0=value, 1=dx, 2=dy), array index (NULL if it's not
    /// an array), and component (x=0 or scalar, y=1, z=2).  If deriv>0
    /// and the symbol doesn't have a deriv, it's a nop.  If the
    /// component >0 and it's a scalar, set the scalar.  Returns true if
    /// ok, false upon failure.
    bool llvm_store_value (llvm::Value* new_val, llvm::Value* dst_ptr,
                           const TypeSpec &type, int deriv,
                           llvm::Value* arrayindex, int component);

    /// Non-array version of llvm_store_value, with default deriv &
    /// component.
    bool llvm_store_value (llvm::Value *new_val, const Symbol& sym,
                           int deriv=0, int component=0) {
        return llvm_store_value (new_val, sym, deriv, NULL, component);
    }

    /// llvm_store_value with non-constant component designation.  Does
    /// not work with arrays or do type casts!
    bool llvm_store_component_value (llvm::Value *new_val, const Symbol& sym,
                                     int deriv, llvm::Value *component);

    /// Legacy version
    ///
    bool storeLLVMValue (llvm::Value* new_val, const Symbol& sym,
                         int component=0, int deriv=0) {
        return llvm_store_value (new_val, sym, deriv, component);
    }

    /// Generate an alloca instruction to allocate space for the given
    /// type, with derivs if derivs==true, and return the its pointer.
    llvm::Value *llvm_alloca (const TypeSpec &type, bool derivs,
                              const std::string &name="");

    /// Given the OSL symbol, return the llvm::Value* corresponding to the
    /// address of the start of that symbol (first element, first component,
    /// and just the plain value if it has derivatives).  This is retrieved
    /// from the allocation map if already there; and if not yet in the
    /// map, the symbol is alloca'd and placed in the map.
    llvm::Value *getOrAllocateLLVMSymbol (const Symbol& sym);

    /// Allocate a GlobalVariable for the given OSL symbol and return a pointer
    /// to it, or return the pointer if it has already been allocated
    llvm::Value *getOrAllocateLLVMGlobal (const Symbol& sym);

    /// Create a GlobalVariable and add it to the current Module
    llvm::Value *addGlobalVariable (const std::string& name, int size,
                                    int alignment, void* data,
                                    const std::string& type="");

    /// Create a GlobalVariable with the extra semantic information needed
    /// by OptiX
    llvm::Value *createOptixVariable (const std::string& name,
                                      const std::string& type,
                                      int size, void* data );

    /// Retrieve an llvm::Value that is a pointer holding the start address
    /// of the specified symbol. This always works for globals and params;
    /// for stack variables (locals/temps) is succeeds only if the symbol is
    /// already in the allocation table (will fail otherwise). This method
    /// is not designed to retrieve constants.
    llvm::Value *getLLVMSymbolBase (const Symbol &sym);

    /// Retrieve the named global ("P", "N", etc.).
    llvm::Value *llvm_global_symbol_ptr (ustring name);

    /// Test whether val is nonzero, return the llvm::Value* that's the
    /// result of a CreateICmpNE or CreateFCmpUNE (depending on the
    /// type).  If test_derivs is true, it it also tests whether the
    /// derivs are zero.
    llvm::Value *llvm_test_nonzero (Symbol &val, bool test_derivs = false);

    /// Implementaiton of Simple assignment.  If arrayindex >= 0, in
    /// designates a particular array index to assign.
    bool llvm_assign_impl (Symbol &Result, Symbol &Src, int arrayindex = -1,
                           int srcomp = -1, int dstcomp = -1);


    /// Convert the name of a global (and its derivative index) into the
    /// field number of the ShaderGlobals struct.
    int ShaderGlobalNameToIndex (ustring name);

    /// Return the LLVM type handle for the ShaderGlobals struct.
    ///
    llvm::Type *llvm_type_sg ();

    /// Return the LLVM type handle for a pointer to a
    /// ShaderGlobals struct.
    llvm::Type *llvm_type_sg_ptr ();

    /// Return the ShaderGlobals pointer.
    ///
    llvm::Value *sg_ptr () const { return m_llvm_shaderglobals_ptr; }

    llvm::Type *llvm_type_closure_component ();
    llvm::Type *llvm_type_closure_component_ptr ();

    /// Return the ShaderGlobals pointer cast as a void*.
    ///
    llvm::Value *sg_void_ptr () {
        return ll.void_ptr (m_llvm_shaderglobals_ptr);
    }

    llvm::Value *llvm_ptr_cast (llvm::Value* val, const TypeSpec &type) {
        return ll.ptr_cast (val, type.simpletype());
    }

    llvm::Value *llvm_void_ptr (const Symbol &sym, int deriv=0) {
        return ll.void_ptr (llvm_get_pointer(sym, deriv));
    }

    /// Return the LLVM type handle for a structure of the common group
    /// data that holds all the shader params.
    llvm::Type *llvm_type_groupdata ();

    /// Return the LLVM type handle for a pointer to the common group
    /// data that holds all the shader params.
    llvm::Type *llvm_type_groupdata_ptr ();

    /// Return the group data pointer.
    ///
    llvm::Value *groupdata_ptr () const { return m_llvm_groupdata_ptr; }

    /// Return the group data pointer cast as a void*.
    ///
    llvm::Value *groupdata_void_ptr () {
        return ll.void_ptr (m_llvm_groupdata_ptr);
    }

    /// Return a reference to the specified field within the group data.
    llvm::Value *groupdata_field_ref (int fieldnum);

    /// Return a pointer to the specified field within the group data,
    /// optionally cast to pointer to a particular data type.
    llvm::Value *groupdata_field_ptr (int fieldnum,
                                      TypeDesc type = TypeDesc::UNKNOWN);

    /// Return a ref to the bool where the "layer_run" flag is stored for
    /// the specified layer.
    llvm::Value *layer_run_ref (int layer);

    /// Return a ref to the bool where the "userdata_initialized" flag is
    /// stored for the specified userdata index.
    llvm::Value *userdata_initialized_ref (int userdata_index=0);

    /// Generate LLVM code to zero out the variable (including derivs)
    ///
    void llvm_assign_zero (const Symbol &sym);

    /// Generate LLVM code to zero out the derivatives of sym.
    ///
    void llvm_zero_derivs (const Symbol &sym);

    /// Generate LLVM code to zero out the derivatives of an array
    /// only for the first count elements of it.
    ///
    void llvm_zero_derivs (const Symbol &sym, llvm::Value *count);

    /// Generate a debugging printf at shader execution time.
    void llvm_gen_debug_printf (string_view message);

    /// Generate a warning message at shader execution time.
    void llvm_gen_warning (string_view message);

    /// Generate an error message at shader execution time.
    void llvm_gen_error (string_view message);

    /// Generate code to call the given layer.  If 'unconditional' is
    /// true, call it without even testing if the layer has already been
    /// called.
    void llvm_call_layer (int layer, bool unconditional = false);

    /// Execute the upstream connection (if any, and if not yet run) that
    /// establishes the value of symbol sym, which has index 'symindex'
    /// within the current layer rop.inst().  If already_run is not NULL,
    /// it points to a vector of layer indices that are known to have been 
    /// run -- those can be skipped without dynamically checking their
    /// execution status.
    void llvm_run_connected_layers (Symbol &sym, int symindex, int opnum = -1,
                                    std::set<int> *already_run = NULL);

    /// Generate code for a call to the named function with the given
    /// arg list as symbols -- float & ints will be passed by value,
    /// triples and matrices will be passed by address.  If deriv_ptrs
    /// is true, pass pointers even for floats if they have derivs.
    /// Return an llvm::Value* corresponding to the return value of the
    /// function, if any.
    llvm::Value *llvm_call_function (const char *name,  const Symbol **args,
                                     int nargs, bool deriv_ptrs=false);
    llvm::Value *llvm_call_function (const char *name, const Symbol &A,
                                     bool deriv_ptrs=false);
    llvm::Value *llvm_call_function (const char *name, const Symbol &A,
                                     const Symbol &B, bool deriv_ptrs=false);
    llvm::Value *llvm_call_function (const char *name, const Symbol &A,
                                     const Symbol &B, const Symbol &C,
                                     bool deriv_ptrs=false);

    TypeDesc llvm_typedesc (const TypeSpec &typespec) {
        return typespec.is_closure_based()
           ? TypeDesc(TypeDesc::PTR, typespec.arraylength())
           : typespec.simpletype();
    }

    /// Generate the appropriate llvm type definition for a TypeSpec
    /// (this is the actual type, for example when we allocate it).
    /// Allocates ptrs for closures.
    llvm::Type *llvm_type (const TypeSpec &typespec) {
        return ll.llvm_type (llvm_typedesc(typespec));
    }

    /// Generate the parameter-passing llvm type definition for an OSL
    /// TypeSpec.
    llvm::Type *llvm_pass_type (const TypeSpec &typespec);

    llvm::PointerType *llvm_type_prepare_closure_func() { return m_llvm_type_prepare_closure_func; }
    llvm::PointerType *llvm_type_setup_closure_func() { return m_llvm_type_setup_closure_func; }

    /// Return the basic block of the exit for the whole instance.
    ///
    bool llvm_has_exit_instance_block () const {
        return m_exit_instance_block;
    }

    /// Return the basic block of the exit for the whole instance.
    ///
    llvm::BasicBlock *llvm_exit_instance_block () {
        if (! m_exit_instance_block) {
            std::string name = Strutil::format ("%s_%d_exit_", inst()->layername(), inst()->id());
            m_exit_instance_block = ll.new_basic_block (name);
        }
        return m_exit_instance_block;
    }

    /// Check for inf/nan in all written-to arguments of the op
    void llvm_generate_debugnan (const Opcode &op);
    /// Check for uninitialized values in all read-from arguments to the op
    void llvm_generate_debug_uninit (const Opcode &op);
    /// Print debugging line for the op
    void llvm_generate_debug_op_printf (const Opcode &op);

    llvm::Function *layer_func () const { return ll.current_function(); }

    /// Call this when JITing a texture-like call, to track how many.
    void generated_texture_call (bool handle) {
        shadingsys().m_stat_tex_calls_codegened += 1;
        if (handle)
            shadingsys().m_stat_tex_calls_as_handles += 1;
    }

    /// Return the mapping from symbol names to GlobalVariables.
    std::map<std::string,llvm::GlobalVariable*>& get_const_map() { return m_const_map; }

    /// Return whether or not we are compiling for an OptiX-based renderer.
    bool use_optix() { return m_use_optix; }

    LLVM_Util ll;

private:
    std::vector<int> m_layer_remap;     ///< Remapping of layer ordering
    std::set<int> m_layers_already_run; ///< List of layers run
    int m_num_used_layers;              ///< Number of layers actually used

    double m_stat_total_llvm_time;        ///<   total time spent on LLVM
    double m_stat_llvm_setup_time;        ///<     llvm setup time
    double m_stat_llvm_irgen_time;        ///<     llvm IR generation time
    double m_stat_llvm_opt_time;          ///<     llvm IR optimization time
    double m_stat_llvm_jit_time;          ///<     llvm JIT time

    // LLVM stuff
    AllocationMap m_named_values;
    std::map<const Symbol*,int> m_param_order_map;
    llvm::Value *m_llvm_shaderglobals_ptr;
    llvm::Value *m_llvm_groupdata_ptr;
    llvm::BasicBlock * m_exit_instance_block;  // exit point for the instance
    llvm::Type *m_llvm_type_sg;  // LLVM type of ShaderGlobals struct
    llvm::Type *m_llvm_type_groupdata;  // LLVM type of group data
    llvm::Type *m_llvm_type_closure_component; // LLVM type for ClosureComponent
    llvm::PointerType *m_llvm_type_prepare_closure_func;
    llvm::PointerType *m_llvm_type_setup_closure_func;
    int m_llvm_local_mem;             // Amount of memory we use for locals

    // A mapping from symbol names to llvm::GlobalVariables
    std::map<std::string,llvm::GlobalVariable*> m_const_map;

    bool m_use_optix;                   ///< Compile for OptiX?

    friend class ShadingSystemImpl;
};


}; // namespace pvt
OSL_NAMESPACE_EXIT
