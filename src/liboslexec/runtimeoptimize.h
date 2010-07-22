/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
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

#include <vector>
#include <map>

#include "oslexec_pvt.h"
#include "oslops.h"
using namespace OSL;
using namespace OSL::pvt;

#include "llvm_headers.h"


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {
namespace pvt {   // OSL::pvt



/// Container for state that needs to be passed around
class RuntimeOptimizer {
public:
    RuntimeOptimizer (ShadingSystemImpl &shadingsys, ShaderGroup &group)
        : m_shadingsys(shadingsys), m_group(group),
          m_inst(NULL), m_next_newconst(0)
#if USE_LLVM
        , m_llvm_context(NULL), m_llvm_module(NULL), m_builder(NULL),
          m_llvm_passes(NULL), m_llvm_func_passes(NULL)
#endif
    {
    }

    ~RuntimeOptimizer () {
#if USE_LLVM
        delete m_builder;
        delete m_llvm_passes;
        delete m_llvm_func_passes;
#endif
    }

    void optimize_group ();

    /// Optimize one layer of a group, given what we know about its
    /// instance variables and connections.
    void optimize_instance ();

    /// Post-optimization cleanup of a layer: add 'useparam' instructions,
    /// track variable lifetimes, coalesce temporaries.
    void post_optimize_instance ();

    /// Set which instance we are currently optimizing.
    ///
    void set_inst (int layer);

    ShaderInstance *inst () const { return m_inst; }

    ShaderGroup &group () const { return m_group; }

    ShadingSystemImpl &shadingsys () const { return m_shadingsys; }

    TextureSystem *texturesys () const { return shadingsys().texturesys(); }

    /// Search the instance for a constant whose type and value match
    /// type and data[...].  Return -1 if no matching const is found.
    int find_constant (const TypeSpec &type, const void *data);

    /// Search for a constant whose type and value match type and data[...],
    /// returning its index if one exists, or else creating a new constant
    /// and returning its index.  If copy is true, allocate new space and
    /// copy the data if no matching constant was found.
    int add_constant (const TypeSpec &type, const void *data);

    /// Turn the op into a simple assignment of the new symbol index to the
    /// previous first argument of the op.  That is, changes "OP arg0 arg1..."
    /// into "assign arg0 newarg".
    void turn_into_assign (Opcode &op, int newarg);

    /// Turn the op into a simple assignment of zero to the previous
    /// first argument of the op.  That is, changes "OP arg0 arg1 ..."
    /// into "assign arg0 zero".
    void turn_into_assign_zero (Opcode &op);

    /// Turn the op into a simple assignment of one to the previous
    /// first argument of the op.  That is, changes "OP arg0 arg1 ..."
    /// into "assign arg0 one".
    void turn_into_assign_one (Opcode &op);

    /// Turn the op into a no-op.
    ///
    void turn_into_nop (Opcode &op);

    void find_constant_params (ShaderGroup &group);

    void find_conditionals ();

    void find_basic_blocks (bool do_llvm = false);

    bool coerce_assigned_constant (Opcode &op);

    void make_param_use_instanceval (Symbol *R);

    /// Return the index of the symbol ultimately de-aliases to (it may be
    /// itself, if it doesn't alias to anything else).  Local block aliases
    /// are considered higher precedent than global aliases.
    int dealias_symbol (int symindex);

    /// Return the index of the symbol that 'symindex' aliases to, locally,
    /// or -1 if it has no block-local alias.
    int block_alias (int symindex) const { return m_block_aliases[symindex]; }

    /// Set the new block-local alias of 'symindex' to 'alias'.
    ///
    void block_alias (int symindex, int alias) {
        m_block_aliases[symindex] = alias;
    }

    /// Reset the block-local alias of 'symindex' so it doesn't alias to
    /// anything.
    void block_unalias (int symindex) {
        m_block_aliases[symindex] = -1;
    }

    /// Reset all block-local aliases (done when we enter a new basic
    /// block).
    void clear_block_aliases () {
        m_block_aliases.clear ();
        m_block_aliases.resize (inst()->symbols().size(), -1);
    }

    /// Set the new global alias of 'symindex' to 'alias'.
    ///
    void global_alias (int symindex, int alias) {
        m_symbol_aliases[symindex] = alias;
    }

    /// Replace R's instance value with new data.
    ///
    void replace_param_value (Symbol *R, const void *newdata);

    bool outparam_assign_elision (int opnum, Opcode &op);

    bool useless_op_elision (Opcode &op);

    void make_symbol_room (int howmany=1);

    void insert_code (int opnum, ustring opname, OpImpl impl,
                      const std::vector<int> &args_to_add);

    void insert_useparam (size_t opnum, std::vector<int> &params_to_use);

    /// Add a 'useparam' before any op that reads parameters.  This is what
    /// tells the runtime that it needs to run the layer it came from, if
    /// not already done.
    void add_useparam (SymbolPtrVec &allsyms);

    void coalesce_temporaries ();

    /// Track variable lifetimes for all the symbols of the instance.
    ///
    void track_variable_lifetimes ();
    void track_variable_lifetimes (const SymbolPtrVec &allsymptrs);

    /// For each symbol, have a list of the symbols it depends on (or that
    /// depends on it).
    typedef std::map<int, std::set<int> > SymDependency;

    void syms_used_in_op (Opcode &op,
                          std::vector<int> &rsyms, std::vector<int> &wsyms);

    void track_variable_dependencies ();

    void add_dependency (SymDependency &dmap, int A, int B);

    /// Squeeze out unused symbols from an instance that has been
    /// optimized.
    void collapse_syms ();

    /// Squeeze out nop instructions from an instance that has been
    /// optimized.
    void collapse_ops ();

    /// Let the optimizer know that this (known) message was set.
    ///
    void register_message (ustring name);

    /// Let the optimizer know that an unknown message was set.
    ///
    void register_unknown_message ();

    /// Is it possible that the message with the given name was set?
    ///
    bool message_possibly_set (ustring name) const;

    /// Return the index of the next instruction within the same basic
    /// block that isn't a NOP.  If there are no more non-NOP
    /// instructions in the same basic block as opnum, return 0.
    int next_block_instruction (int opnum);

    /// Search for pairs of ops to perform peephole optimization on.
    /// 
    int peephole2 (int opnum);

    /// Helper: return the ptr to the symbol that is the argnum-th
    /// argument to the given op.
    Symbol *opargsym (const Opcode &op, int argnum) {
        return inst()->argsymbol (op.firstarg()+argnum);
    }

    /// Create an llvm function for the whole shader group, JIT it,
    /// and store the llvm::Function* handle to it with the ShaderGroup.
    void build_llvm_group ();

#if USE_LLVM
    /// Set up a bunch of static things we'll need for the whole group.
    ///
    void initialize_llvm_group ();

    /// Create an llvm function for the current shader instance.
    /// This will end up being the group entry if 'groupentry' is true.
    llvm::Function* build_llvm_instance (bool groupentry);

    /// Build up LLVM IR code for the given range [begin,end) or
    /// opcodes, putting them (initially) into basic block bb (or the
    /// current basic block if bb==NULL).
    bool build_llvm_code (int beginop, int endop, llvm::BasicBlock *bb=NULL);

    typedef std::map<std::string, llvm::AllocaInst*> AllocationMap;

    void llvm_assign_initial_value (const Symbol& sym);
    llvm::LLVMContext &llvm_context () const { return *m_llvm_context; }
    llvm::Module *llvm_module () const { return m_llvm_module; }
    AllocationMap &named_values () { return m_named_values; }
    llvm::IRBuilder<> &builder () { return *m_builder; }

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

    /// Store new_val into the given symbol, given the derivative
    /// (0=value, 1=dx, 2=dy), array index (NULL if it's not an array),
    /// and component (x=0 or scalar, y=1, z=2).  If deriv>0 and the
    /// symbol doesn't have a deriv, it's a nop.  If the component >0
    /// and it's a scalar, set the scalar.  Returns true if ok, false
    /// upon failure.
    bool llvm_store_value (llvm::Value *new_val, const Symbol& sym, int deriv,
                           llvm::Value *arrayindex, int component);

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

    llvm::Value *getOrAllocateLLVMSymbol (const Symbol& sym);
    llvm::Value *getLLVMSymbolBase (const Symbol &sym);

    /// Generate the LLVM IR code to convert fval from a float to
    /// an integer and return the new value.
    llvm::Value *llvm_float_to_int (llvm::Value *fval);

    /// Generate the LLVM IR code to convert ival from an int to a float
    /// and return the new value.
    llvm::Value *llvm_int_to_float (llvm::Value *ival);

    /// Return the LLVM type handle for the SingleShaderGlobals struct.
    ///
    const llvm::Type *llvm_type_sg ();

    /// Return the LLVM type handle for a pointer to a
    /// SingleShaderGlobals struct.
    const llvm::Type *llvm_type_sg_ptr ();

    /// Return the SingleShaderGlobals pointer.
    ///
    llvm::Value *sg_ptr () const { return m_llvm_shaderglobals_ptr; }

    /// Return the SingleShaderGlobals pointer cast as a void*.
    ///
    llvm::Value *sg_void_ptr () {
        return llvm_void_ptr (m_llvm_shaderglobals_ptr);
    }

    llvm::Value *llvm_ptr_cast (llvm::Value* val, const llvm::Type *type) {
        return builder().CreatePointerCast(val,type);
    }

    llvm::Value *llvm_void_ptr (llvm::Value* val) {
        return builder().CreatePointerCast(val,llvm_type_void_ptr());
    }

    llvm::Value *llvm_void_ptr (const Symbol &sym, int deriv=0) {
        return llvm_void_ptr (llvm_get_pointer(sym, deriv));
    }

    llvm::Value *llvm_void_ptr_null () {
        return llvm::ConstantPointerNull::get (llvm_type_void_ptr());
    }

    /// Return the LLVM type handle for a structure of the common group
    /// data that holds all the shader params.
    const llvm::Type *llvm_type_groupdata ();

    /// Return the LLVM type handle for a pointer to the common group
    /// data that holds all the shader params.
    const llvm::Type *llvm_type_groupdata_ptr ();

    /// Return the SingleShaderGlobals pointer.
    ///
    llvm::Value *groupdata_ptr () const { return m_llvm_groupdata_ptr; }

    /// Return the group data pointer cast as a void*.
    ///
    llvm::Value *groupdata_void_ptr () {
        return llvm_void_ptr (m_llvm_groupdata_ptr);
    }

    /// Return a ref to where the "layer_run" flag is stored for the
    /// named layer.
    llvm::Value *layer_run_ptr (int layer);

    /// Return an llvm::Value holding the given floating point constant.
    ///
    llvm::Value *llvm_constant (float f);

    /// Return an llvm::Value holding the given integer constant.
    ///
    llvm::Value *llvm_constant (int i);

    /// Return an llvm::Value holding the given size_t constant.
    ///
    llvm::Value *llvm_constant (size_t i);

    /// Return a constant void pointer to the given address
    ///
    llvm::Value *llvm_constant_ptr (void *p, const llvm::PointerType *type)
    {
        return builder().CreateIntToPtr (llvm_constant (size_t (p)), type, "const pointer");
    }

    /// Return an llvm::Value holding the given integer constant.
    ///
    llvm::Value *llvm_constant (ustring s);
    llvm::Value *llvm_constant (const char *s) {
        return llvm_constant(ustring(s));
    }

    /// Return an llvm::Value for a long long that is a packed
    /// representation of a TypeDesc.
    llvm::Value *llvm_constant (const TypeDesc &type);

    /// Generate LLVM code to zero out the derivatives of sym.
    ///
    void llvm_zero_derivs (const Symbol &sym);

    /// Generate a pointer that is (ptrtype)((char *)ptr + offset).
    /// If ptrtype is NULL, just return a void*.
    llvm::Value *llvm_offset_ptr (llvm::Value *ptr, int offset,
                                  const llvm::Type *ptrtype=NULL);

    /// Generate code for a call to the named function with the given arg
    /// list.  Return an llvm::Value* corresponding to the return value of
    /// the function, if any.
    llvm::Value *llvm_call_function (const char *name,
                                     llvm::Value **args, int nargs);

    llvm::Value *llvm_call_function (const char *name, llvm::Value *arg0) {
        return llvm_call_function (name, &arg0, 1);
    }
    llvm::Value *llvm_call_function (const char *name, llvm::Value *arg0,
                                     llvm::Value *arg1) {
        llvm::Value *args[2];
        args[0] = arg0;  args[1] = arg1;
        return llvm_call_function (name, args, 2);
    }

    void llvm_gen_debug_printf (const std::string &message);

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

    /// Generate the appropriate llvm type definition for an OSL TypeSpec
    /// (this is the actual type, for example when we allocate it).
    const llvm::Type *llvm_type (const TypeSpec &typespec);

    /// Generate the parameter-passing llvm type definition for an OSL
    /// TypeSpec.
    const llvm::Type *llvm_pass_type (const TypeSpec &typespec);

    const llvm::Type *llvm_type_float() { return m_llvm_type_float; }
    const llvm::Type *llvm_type_triple() { return m_llvm_type_triple; }
    const llvm::Type *llvm_type_matrix() { return m_llvm_type_matrix; }
    const llvm::Type *llvm_type_int() { return m_llvm_type_int; }
    const llvm::Type *llvm_type_addrint() { return m_llvm_type_addrint; }
    const llvm::Type *llvm_type_bool() { return m_llvm_type_bool; }
    const llvm::Type *llvm_type_void() { return m_llvm_type_void; }
    const llvm::PointerType *llvm_type_prepare_closure_func() { return m_llvm_type_prepare_closure_func; }
    const llvm::PointerType *llvm_type_setup_closure_func() { return m_llvm_type_setup_closure_func; }
    const llvm::PointerType *llvm_type_int_ptr() { return m_llvm_type_int_ptr; }
    const llvm::PointerType *llvm_type_void_ptr() { return m_llvm_type_char_ptr; }
    const llvm::PointerType *llvm_type_string() { return m_llvm_type_char_ptr; }
    const llvm::PointerType *llvm_type_ustring_ptr() { return m_llvm_type_ustring_ptr; }
    const llvm::PointerType *llvm_type_float_ptr() { return m_llvm_type_float_ptr; }
    const llvm::PointerType *llvm_type_triple_ptr() { return m_llvm_type_triple_ptr; }
    const llvm::PointerType *llvm_type_matrix_ptr() { return m_llvm_type_matrix_ptr; }

    /// Shorthand to create a new LLVM basic block and return its handle.
    ///
    llvm::BasicBlock *llvm_new_basic_block (const std::string &name) {
        return llvm::BasicBlock::Create (llvm_context(), name, m_layer_func);
    }

    llvm::Function *layer_func () const { return m_layer_func; }

    void llvm_setup_optimization_passes ();

    /// Do LLVM optimization on the partcular function func.  If
    /// interproc is true, also do full interprocedural optimization.
    void llvm_do_optimization (llvm::Function *func, bool interproc=false);
#endif

private:
    ShadingSystemImpl &m_shadingsys;
    ShaderGroup &m_group;             ///< Group we're optimizing
    int m_layer;                      ///< Layer we're optimizing
    ShaderInstance *m_inst;           ///< Instance we're optimizing

    // All below is just for the one inst we're optimizing:
    std::vector<int> m_all_consts;    ///< All const symbol indices for inst
    int m_next_newconst;              ///< Unique ID for next new const we add
    std::map<int,int> m_symbol_aliases; ///< Global symbol aliases
    std::vector<int> m_block_aliases;   ///< Local block aliases
    int m_local_unknown_message_sent;   ///< Non-const setmessage in this inst
    std::vector<ustring> m_local_messages_sent; ///< Messages set in this inst
    std::vector<int> m_bblockids;       ///< Basic block IDs for each op
    std::vector<bool> m_in_conditional; ///< Whether each op is in a cond

#if USE_LLVM
    // LLVM stuff
    llvm::LLVMContext *m_llvm_context;
    llvm::Module *m_llvm_module;
    AllocationMap m_named_values;
    std::map<const Symbol*,int> m_param_order_map;
    llvm::IRBuilder<> *m_builder;
    llvm::Value *m_llvm_shaderglobals_ptr;
    llvm::Value *m_llvm_groupdata_ptr;
    llvm::Function *m_layer_func;     ///< Current layer func we're building
    const llvm::Type *m_llvm_type_float;
    const llvm::Type *m_llvm_type_int;
    const llvm::Type *m_llvm_type_addrint;
    const llvm::Type *m_llvm_type_bool;
    const llvm::Type *m_llvm_type_void;
    const llvm::Type *m_llvm_type_triple;
    const llvm::Type *m_llvm_type_matrix;
    const llvm::PointerType *m_llvm_type_ustring_ptr;
    const llvm::PointerType *m_llvm_type_char_ptr;
    const llvm::PointerType *m_llvm_type_int_ptr;
    const llvm::PointerType *m_llvm_type_float_ptr;
    const llvm::PointerType *m_llvm_type_triple_ptr;
    const llvm::PointerType *m_llvm_type_matrix_ptr;
    const llvm::Type *m_llvm_type_sg;  // LLVM type of SingleShaderGlobal struct
    const llvm::Type *m_llvm_type_groupdata;  // LLVM type of group data
    const llvm::PointerType *m_llvm_type_prepare_closure_func;
    const llvm::PointerType *m_llvm_type_setup_closure_func;
    llvm::PassManager *m_llvm_passes;
    llvm::FunctionPassManager *m_llvm_func_passes;
#endif

    // Persistant data shared between layers
    bool m_unknown_message_sent;      ///< Somebody did a non-const setmessage
    std::vector<ustring> m_messages_sent;  ///< Names of messages set
};




/// Macro that defines the arguments to constant-folding routines
///
#define FOLDARGSDECL     RuntimeOptimizer &rop, int opnum

/// Function pointer to a constant-folding routine
///
typedef int (*OpFolder) (FOLDARGSDECL);

/// Macro that defines the full declaration of a shadeop constant-folder.
/// 
#define DECLFOLDER(name)  int name (FOLDARGSDECL)




}; // namespace pvt
}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
#endif
