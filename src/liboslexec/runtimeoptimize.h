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
using namespace OSL;
using namespace OSL::pvt;

#include "llvm_headers.h"


OSL_NAMESPACE_ENTER

namespace pvt {   // OSL::pvt



/// Container for state that needs to be passed around
class RuntimeOptimizer {
public:
    RuntimeOptimizer (ShadingSystemImpl &shadingsys, ShaderGroup &group);

    ~RuntimeOptimizer ();

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

    /// Re-check what debugging level we ought to be at.
    void set_debug ();

    /// Return the layer number of the currently-optimizing instance
    /// within the group.
    int layer () const { return m_layer; }

    /// Return a pointer to the currently-optimizing instance within the
    /// group.
    ShaderInstance *inst () const { return m_inst; }

    /// Return a reference to the shader group being optimized.
    ///
    ShaderGroup &group () const { return m_group; }

    ShadingSystemImpl &shadingsys () const { return m_shadingsys; }

    TextureSystem *texturesys () const { return shadingsys().texturesys(); }

    RendererServices *renderer () const { return shadingsys().renderer(); }

    /// Are we in debugging mode?
    int debug() const { return m_debug; }

    /// What's our current optimization level?
    int optimize() const { return m_optimize; }

    /// Search the instance for a constant whose type and value match
    /// type and data[...].  Return -1 if no matching const is found.
    int find_constant (const TypeSpec &type, const void *data);

    /// Search for a constant whose type and value match type and data[...],
    /// returning its index if one exists, or else creating a new constant
    /// and returning its index.
    int add_constant (const TypeSpec &type, const void *data);

    /// Turn the op into a simple assignment of the new symbol index to the
    /// previous first argument of the op.  That is, changes "OP arg0 arg1..."
    /// into "assign arg0 newarg".
    void turn_into_assign (Opcode &op, int newarg, const char *why=NULL);

    /// Turn the op into a simple assignment of zero to the previous
    /// first argument of the op.  That is, changes "OP arg0 arg1 ..."
    /// into "assign arg0 zero".
    void turn_into_assign_zero (Opcode &op, const char *why=NULL);

    /// Turn the op into a simple assignment of one to the previous
    /// first argument of the op.  That is, changes "OP arg0 arg1 ..."
    /// into "assign arg0 one".
    void turn_into_assign_one (Opcode &op, const char *why=NULL);

    /// Turn the op into a no-op.  Return 1 if it changed, 0 if it was
    /// already a nop.
    int turn_into_nop (Opcode &op, const char *why=NULL);

    /// Turn the whole range [begin,end) into no-ops.  Return the number
    /// of instructions that were altered.
    int turn_into_nop (int begin, int end, const char *why=NULL);

    void find_constant_params (ShaderGroup &group);

    void find_conditionals ();

    void find_loops ();

    void find_basic_blocks (bool do_llvm = false);

    bool coerce_assigned_constant (Opcode &op);

    void make_param_use_instanceval (Symbol *R, const char *why=NULL);

    /// Return the index of the symbol ultimately de-aliases to (it may be
    /// itself, if it doesn't alias to anything else).  Local block aliases
    /// are considered higher precedent than global aliases.
    int dealias_symbol (int symindex, int opnum=-1);

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

    /// Is the given symbol stale?  A "stale" symbol is one that, within
    /// the current basic block, has been assigned in a simple manner
    /// (by a single op with no other side effects), but not yet used.
    /// The point is that if they are simply assigned again before being
    /// used, that first assignment can be turned into a no-op.
    bool sym_is_stale (int sym) {
        return m_stale_syms.find(sym) != m_stale_syms.end();
    }

    /// Clear the stale symbol list -- we do this when entering a new
    /// basic block.
    void clear_stale_syms ();

    /// Take a symbol out of the stale list -- we do this when a symbol
    /// is used in any way.
    void use_stale_sym (int sym);

    /// Is the op a "simple" assignment (arg 0 completely overwritten,
    /// no side effects or funny business)?
    bool is_simple_assign (Opcode &op);

    /// Called when symbol sym is "simply" assigned at the given op.  An
    /// assignment is considered simple if it completely overwrites the
    /// symbol in a single op and has no side effects.  When this
    /// happens, we mark the symbol as "stale", meaning it's got a value
    /// that hasn't been read yet.  If it's wholy assigned again before
    /// it's read, we can go back and remove the earlier assignment.
    void simple_sym_assign (int sym, int op);

    /// Return true if assignments to A on this op have no effect because
    /// they will not be subsequently used.
    bool unread_after (const Symbol *A, int opnum);

    /// Replace R's instance value with new data.
    ///
    void replace_param_value (Symbol *R, const void *newdata);

    bool outparam_assign_elision (int opnum, Opcode &op);

    bool useless_op_elision (Opcode &op, int opnum);

    void make_symbol_room (int howmany=1);

    void insert_code (int opnum, ustring opname,
                      const std::vector<int> &args_to_add,
                      bool recompute_rw_ranges=false);

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

    void mark_outgoing_connections ();

    int remove_unused_params ();

    /// Squeeze out unused symbols from an instance that has been
    /// optimized.
    void collapse_syms ();

    /// Squeeze out nop instructions from an instance that has been
    /// optimized.
    void collapse_ops ();

    /// Let the optimizer know that this (known, constant) message was
    /// set by the current instance.
    void register_message (ustring name);

    /// Let the optimizer know that an unknown message (i.e., we
    /// couldn't reduce the message name to a constant) was set by the
    /// current instance.
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

    /// Helper: return the symbol index of the symbol that is the argnum-th
    /// argument to the given op.
    int oparg (const Opcode &op, int argnum) {
        return inst()->arg (op.firstarg()+argnum);
    }

    /// Helper: return the ptr to the symbol that is the argnum-th
    /// argument to the given op.
    Symbol *opargsym (const Opcode &op, int argnum) {
        return inst()->argsymbol (op.firstarg()+argnum);
    }

    /// Create an llvm function for the whole shader group, JIT it,
    /// and store the llvm::Function* handle to it with the ShaderGroup.
    void build_llvm_group ();

    int layer_remap (int origlayer) const { return m_layer_remap[origlayer]; }

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
    /// type, with derivs if derivs==true, and return the AllocaInst of
    /// its pointer.
    llvm::AllocaInst *llvm_alloca (const TypeSpec &type, bool derivs,
                                   const std::string &name="");

    /// Given the OSL symbol, return the llvm::Value* corresponding to the
    /// start of that symbol (first element, first component, and just the
    /// plain value if it has derivatives).
    llvm::Value *getOrAllocateLLVMSymbol (const Symbol& sym);

    llvm::Value *getLLVMSymbolBase (const Symbol &sym);

    /// Generate the LLVM IR code to convert fval from a float to
    /// an integer and return the new value.
    llvm::Value *llvm_float_to_int (llvm::Value *fval);

    /// Generate the LLVM IR code to convert ival from an int to a float
    /// and return the new value.
    llvm::Value *llvm_int_to_float (llvm::Value *ival);

    /// Generate IR code for simple a/b, but considering OSL's semantics
    /// that x/0 = 0, not inf.
    llvm::Value *llvm_make_safe_div (TypeDesc type,
                                     llvm::Value *a, llvm::Value *b);

    /// Generate IR code for simple a mod b, but considering OSL's
    /// semantics that x mod 0 = 0, not inf.
    llvm::Value *llvm_make_safe_mod (TypeDesc type,
                                     llvm::Value *a, llvm::Value *b);

    /// Implementaiton of Simple assignment.  If arrayindex >= 0, in
    /// designates a particular array index to assign.
    bool llvm_assign_impl (Symbol &Result, Symbol &Src, int arrayindex = -1);


    /// This will return a llvm::Type that is the same as a C union of
    /// the given types[].
    llvm::Type *llvm_type_union(const std::vector<llvm::Type *> &types);

    /// This will return a llvm::Type that is the same as a C struct
    /// comprised fields of the given types[], in order.
    llvm::Type *llvm_type_struct(const std::vector<llvm::Type *> &types,
                                 const std::string &name="");

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
    llvm::Type *llvm_type_closure_component_attr ();
    llvm::Type *llvm_type_closure_component_attr_ptr ();

    /// Return the ShaderGlobals pointer cast as a void*.
    ///
    llvm::Value *sg_void_ptr () {
        return llvm_void_ptr (m_llvm_shaderglobals_ptr);
    }

    llvm::Value *llvm_ptr_cast (llvm::Value* val, llvm::Type *type) {
        return builder().CreatePointerCast(val,type);
    }

    llvm::Value *llvm_ptr_cast (llvm::Value* val, const TypeSpec &type) {
        return llvm_ptr_cast (val, llvm::PointerType::get (llvm_type(type), 0));
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
    llvm::Type *llvm_type_groupdata ();

    /// Return the LLVM type handle for a pointer to the common group
    /// data that holds all the shader params.
    llvm::Type *llvm_type_groupdata_ptr ();

    /// Return the ShaderGlobals pointer.
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

    /// Return an llvm::Value holding the given bool constant.
    /// Change the name so it doesn't get mixed up with int.
    llvm::Value *llvm_constant_bool (bool b);

    /// Return a constant void pointer to the given address
    ///
    llvm::Value *llvm_constant_ptr (void *p, llvm::PointerType *type)
    {
        return builder().CreateIntToPtr (llvm_constant (size_t (p)), type, "const pointer");
    }

    /// Return an llvm::Value holding the given string constant.
    ///
    llvm::Value *llvm_constant (ustring s);
    llvm::Value *llvm_constant (const char *s) {
        return llvm_constant(ustring(s));
    }
    /// Return an llvm::Value holding the given pointer constant.
    ///
    llvm::Value *llvm_constant_ptr (void *p);

    /// Return an llvm::Value for a long long that is a packed
    /// representation of a TypeDesc.
    llvm::Value *llvm_constant (const TypeDesc &type);

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

    /// Generate a pointer that is (ptrtype)((char *)ptr + offset).
    /// If ptrtype is NULL, just return a void*.
    llvm::Value *llvm_offset_ptr (llvm::Value *ptr, int offset,
                                  llvm::Type *ptrtype=NULL);

    /// Generate code for a call to the function pointer, with the given
    /// arg list.  Return an llvm::Value* corresponding to the return
    /// value of the function, if any.
    llvm::Value *llvm_call_function (llvm::Value *func,
                                     llvm::Value **args, int nargs);
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
        llvm::Value *args[2] = { arg0, arg1 };
        return llvm_call_function (name, args, 2);
    }
    llvm::Value *llvm_call_function (const char *name, llvm::Value *arg0,
                                     llvm::Value *arg1, llvm::Value *arg2) {
        llvm::Value *args[3] = { arg0, arg1, arg2 };
        return llvm_call_function (name, args, 3);
    }
    llvm::Value *llvm_call_function (const char *name, llvm::Value *arg0,
                                     llvm::Value *arg1, llvm::Value *arg2,
                                     llvm::Value *arg3) {
        llvm::Value *args[4] = { arg0, arg1, arg2, arg3 };
        return llvm_call_function (name, args, 4);
    }

    void llvm_gen_debug_printf (const std::string &message);

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

    /// Generate code for a memset.
    ///
    void llvm_memset (llvm::Value *ptr, int val, int len, int align=1);

    /// Generate code for variable size memset
    ///
    void llvm_memset (llvm::Value *ptr, int val, llvm::Value *len, int align=1);

    /// Generate code for a memcpy.
    ///
    void llvm_memcpy (llvm::Value *dst, llvm::Value *src,
                      int len, int align=1);

    /// Generate the appropriate llvm type definition for an OSL TypeSpec
    /// (this is the actual type, for example when we allocate it).
    llvm::Type *llvm_type (const TypeSpec &typespec);

    /// Generate the parameter-passing llvm type definition for an OSL
    /// TypeSpec.
    llvm::Type *llvm_pass_type (const TypeSpec &typespec);

    llvm::Type *llvm_type_float() { return m_llvm_type_float; }
    llvm::Type *llvm_type_triple() { return m_llvm_type_triple; }
    llvm::Type *llvm_type_matrix() { return m_llvm_type_matrix; }
    llvm::Type *llvm_type_int() { return m_llvm_type_int; }
    llvm::Type *llvm_type_addrint() { return m_llvm_type_addrint; }
    llvm::Type *llvm_type_bool() { return m_llvm_type_bool; }
    llvm::Type *llvm_type_longlong() { return m_llvm_type_longlong; }
    llvm::Type *llvm_type_void() { return m_llvm_type_void; }
    llvm::PointerType *llvm_type_prepare_closure_func() { return m_llvm_type_prepare_closure_func; }
    llvm::PointerType *llvm_type_setup_closure_func() { return m_llvm_type_setup_closure_func; }
    llvm::PointerType *llvm_type_int_ptr() { return m_llvm_type_int_ptr; }
    llvm::PointerType *llvm_type_void_ptr() { return m_llvm_type_char_ptr; }
    llvm::PointerType *llvm_type_string() { return m_llvm_type_char_ptr; }
    llvm::PointerType *llvm_type_ustring_ptr() { return m_llvm_type_ustring_ptr; }
    llvm::PointerType *llvm_type_float_ptr() { return m_llvm_type_float_ptr; }
    llvm::PointerType *llvm_type_triple_ptr() { return m_llvm_type_triple_ptr; }
    llvm::PointerType *llvm_type_matrix_ptr() { return m_llvm_type_matrix_ptr; }

    /// Shorthand to create a new LLVM basic block and return its handle.
    ///
    llvm::BasicBlock *llvm_new_basic_block (const std::string &name) {
        return llvm::BasicBlock::Create (llvm_context(), name, m_layer_func);
    }

    /// Save the basic block pointers when entering a loop.
    ///
    void llvm_push_loop (llvm::BasicBlock *step, llvm::BasicBlock *after) {
        m_loop_step_block.push_back (step);
        m_loop_after_block.push_back (after);
    }

    /// Pop basic block pointers when exiting a loop.
    ///
    void llvm_pop_loop () {
        ASSERT (! m_loop_step_block.empty() && ! m_loop_after_block.empty());
        m_loop_step_block.pop_back ();
        m_loop_after_block.pop_back ();
    }

    /// Return the basic block of the current loop's 'step' instructions.
    llvm::BasicBlock *llvm_loop_step_block () const {
        ASSERT (! m_loop_step_block.empty());
        return m_loop_step_block.back();
    }

    /// Return the basic block of the current loop's exit point.
    llvm::BasicBlock *llvm_loop_after_block () const {
        ASSERT (! m_loop_after_block.empty());
        return m_loop_after_block.back();
    }

    /// Save the return block pointer when entering a function.
    ///
    void llvm_push_function (llvm::BasicBlock *after) {
        m_return_block.push_back (after);
    }

    /// Pop basic return destination when exiting a function.
    ///
    void llvm_pop_function () {
        ASSERT (! m_return_block.empty());
        m_return_block.pop_back ();
    }

    /// Return the basic block of the current loop's 'step' instructions.
    ///
    llvm::BasicBlock *llvm_return_block () const {
        ASSERT (! m_return_block.empty());
        return m_return_block.back();
    }

    /// Check for inf/nan in all written-to arguments of the op
    void llvm_generate_debugnan (const Opcode &op);

    llvm::Function *layer_func () const { return m_layer_func; }

    void llvm_setup_optimization_passes ();

    bool opt_elide_unconnected_outputs () const {
        return m_opt_elide_unconnected_outputs;
    }

private:
    ShadingSystemImpl &m_shadingsys;
    PerThreadInfo *m_thread;
    ShaderGroup &m_group;             ///< Group we're optimizing
    int m_layer;                      ///< Layer we're optimizing
    ShaderInstance *m_inst;           ///< Instance we're optimizing
    int m_debug;                      ///< Current debug level
    int m_optimize;                   ///< Current optimization level
    bool m_opt_constant_param;            ///< Turn instance params into const?
    bool m_opt_constant_fold;             ///< Allow constant folding?
    bool m_opt_stale_assign;              ///< Optimize stale assignments?
    bool m_opt_elide_useless_ops;         ///< Optimize away useless ops?
    bool m_opt_elide_unconnected_outputs; ///< Optimize unconnected outputs?
    bool m_opt_peephole;                  ///< Do some peephole optimizations?
    bool m_opt_coalesce_temps;            ///< Coalesce temporary variables?
    bool m_opt_assign;                    ///< Do various assign optimizations?

    // All below is just for the one inst we're optimizing:
    std::vector<int> m_all_consts;    ///< All const symbol indices for inst
    int m_next_newconst;              ///< Unique ID for next new const we add
    std::map<int,int> m_symbol_aliases; ///< Global symbol aliases
    std::vector<int> m_block_aliases;   ///< Local block aliases
    std::map<int,int> m_param_aliases;  ///< Params aliasing to params/globals
    std::map<int,int> m_stale_syms;     ///< Stale symbols for this block
    int m_local_unknown_message_sent;   ///< Non-const setmessage in this inst
    std::vector<ustring> m_local_messages_sent; ///< Messages set in this inst
    std::vector<int> m_bblockids;       ///< Basic block IDs for each op
    std::vector<bool> m_in_conditional; ///< Whether each op is in a cond
    std::vector<bool> m_in_loop;        ///< Whether each op is in a loop
    std::vector<int> m_layer_remap;     ///< Remapping of layer ordering
    std::set<int> m_layers_already_run; ///< List of layers run
    int m_num_used_layers;              ///< Number of layers actually used
    double m_stat_opt_locking_time;       ///<   locking time
    double m_stat_specialization_time;    ///<   specialization time
    double m_stat_total_llvm_time;        ///<   total time spent on LLVM
    double m_stat_llvm_setup_time;        ///<     llvm setup time
    double m_stat_llvm_irgen_time;        ///<     llvm IR generation time
    double m_stat_llvm_opt_time;          ///<     llvm IR optimization time
    double m_stat_llvm_jit_time;          ///<     llvm JIT time

    // LLVM stuff
    llvm::LLVMContext *m_llvm_context;
    llvm::Module *m_llvm_module;
    llvm::ExecutionEngine *m_llvm_exec;
    AllocationMap m_named_values;
    std::map<const Symbol*,int> m_param_order_map;
    llvm::IRBuilder<> *m_builder;
    llvm::Value *m_llvm_shaderglobals_ptr;
    llvm::Value *m_llvm_groupdata_ptr;
    llvm::Function *m_layer_func;     ///< Current layer func we're building
    std::vector<llvm::BasicBlock *> m_loop_after_block; // stack for break
    std::vector<llvm::BasicBlock *> m_loop_step_block;  // stack for continue
    std::vector<llvm::BasicBlock *> m_return_block;     // stack for func call
    llvm::Type *m_llvm_type_float;
    llvm::Type *m_llvm_type_int;
    llvm::Type *m_llvm_type_addrint;
    llvm::Type *m_llvm_type_bool;
    llvm::Type *m_llvm_type_longlong;
    llvm::Type *m_llvm_type_void;
    llvm::Type *m_llvm_type_triple;
    llvm::Type *m_llvm_type_matrix;
    llvm::PointerType *m_llvm_type_ustring_ptr;
    llvm::PointerType *m_llvm_type_char_ptr;
    llvm::PointerType *m_llvm_type_int_ptr;
    llvm::PointerType *m_llvm_type_float_ptr;
    llvm::PointerType *m_llvm_type_triple_ptr;
    llvm::PointerType *m_llvm_type_matrix_ptr;
    llvm::Type *m_llvm_type_sg;  // LLVM type of ShaderGlobals struct
    llvm::Type *m_llvm_type_groupdata;  // LLVM type of group data
    llvm::Type *m_llvm_type_closure_component; // LLVM type for ClosureComponent
    llvm::Type *m_llvm_type_closure_component_attr; // LLVM type for ClosureMeta::Attr
    llvm::PointerType *m_llvm_type_prepare_closure_func;
    llvm::PointerType *m_llvm_type_setup_closure_func;
    llvm::PassManager *m_llvm_passes;
    llvm::FunctionPassManager *m_llvm_func_passes;
    int m_llvm_local_mem;             // Amount of memory we use for locals

    // Persistant data shared between layers
    bool m_unknown_message_sent;      ///< Somebody did a non-const setmessage
    std::vector<ustring> m_messages_sent;  ///< Names of messages set

    friend class ShadingSystemImpl;
};




/// Macro that defines the arguments to constant-folding routines
///
#define FOLDARGSDECL     RuntimeOptimizer &rop, int opnum

/// Macro that defines the full declaration of a shadeop constant-folder.
/// 
#define DECLFOLDER(name)  int name (FOLDARGSDECL)



}; // namespace pvt
OSL_NAMESPACE_EXIT
