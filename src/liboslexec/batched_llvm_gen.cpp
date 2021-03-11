// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/imageworks/OpenShadingLanguage

#include <set>

#include <llvm/IR/Constant.h>

#include "batched_backendllvm.h"



using namespace OSL;
using namespace OSL::pvt;

OSL_NAMESPACE_ENTER

namespace pvt {

static ustring op_ceil("ceil");
static ustring op_eq("eq");
static ustring op_floor("floor");
static ustring op_ge("ge");
static ustring op_gt("gt");
static ustring op_logb("logb");
static ustring op_le("le");
static ustring op_lt("lt");
static ustring op_neq("neq");
static ustring op_round("round");
static ustring op_sign("sign");
static ustring op_step("step");
static ustring op_trunc("trunc");

/// Macro that defines the arguments to LLVM IR generating routines
///
#define LLVMGEN_ARGS BatchedBackendLLVM &rop, int opnum

/// Macro that defines the full declaration of an LLVM generator.
///
#define LLVMGEN(name) bool name(LLVMGEN_ARGS)

// Forward decl
LLVMGEN (llvm_gen_generic);


typedef typename BatchedBackendLLVM::FuncSpec FuncSpec;



void
BatchedBackendLLVM::llvm_gen_debug_printf(string_view message)
{
    ustring s = ustring::format("(%s %s) %s", inst()->shadername(),
                                inst()->layername(), message);
    ll.call_function(build_name("printf"), sg_void_ptr(), ll.constant("%s\n"),
                     ll.constant(s));
}



void
BatchedBackendLLVM::llvm_call_layer(int layer, bool unconditional)
{
    OSL_DEV_ONLY(std::cout << "llvm_call_layer layer=" << layer
                           << " unconditional=" << unconditional << std::endl);
    // Make code that looks like:
    //     if (! groupdata->run[parentlayer])
    //         parent_layer (sg, groupdata);
    // if it's a conditional call, or
    //     parent_layer (sg, groupdata);
    // if it's run unconditionally.
    // The code in the parent layer itself will set its 'executed' flag.

    llvm::Value* args[3];
    args[0] = sg_ptr();
    args[1] = groupdata_ptr();

    ShaderInstance* parent       = group()[layer];
    llvm::Value* layerfield      = layer_run_ref(layer_remap(layer));
    llvm::BasicBlock *then_block = NULL, *after_block = NULL;
    llvm::Value* lanes_requiring_execution_value = nullptr;
    if (!unconditional) {
        llvm::Value* previously_executed = ll.int_as_mask(
            ll.op_load(layerfield));
        llvm::Value* lanes_requiring_execution
            = ll.op_select(previously_executed, ll.wide_constant_bool(false),
                           ll.current_mask());
        lanes_requiring_execution_value = ll.mask_as_int(
            lanes_requiring_execution);
        llvm::Value* execution_required
            = ll.op_ne(lanes_requiring_execution_value, ll.constant(0));
        then_block = ll.new_basic_block(
            llvm_debug()
                ? std::string("then layer ").append(std::to_string(layer))
                : std::string());
        after_block = ll.new_basic_block(
            llvm_debug()
                ? std::string("after layer ").append(std::to_string(layer))
                : std::string());
        ll.op_branch(execution_required, then_block, after_block);
        // insert point is now then_block
    } else {
        lanes_requiring_execution_value = ll.mask_as_int(ll.shader_mask());
    }

    args[2] = lanes_requiring_execution_value;

    // Before the merge, keeping in case we broke it
    //std::string name = Strutil::format ("%s_%s_%d", m_library_selector,  parent->layername().c_str(),
    //                                  parent->id());
    std::string name
        = Strutil::format("%s_%s", m_library_selector,
                          layer_function_name(group(), *parent).c_str());

    // Mark the call as a fast call
    llvm::Value* funccall = ll.call_function(name.c_str(), args);
    if (!parent->entry_layer())
        ll.mark_fast_func_call(funccall);

    if (!unconditional)
        ll.op_branch(after_block);  // also moves insert point
}



void
BatchedBackendLLVM::llvm_run_connected_layers(const Symbol& sym, int symindex,
                                              int opnum,
                                              std::set<int>* already_run)
{
    if (sym.valuesource() != Symbol::ConnectedVal)
        return;  // Nothing to do

    OSL_DEV_ONLY(std::cout << "BatchedBackendLLVM::llvm_run_connected_layers "
                           << sym.name().c_str() << " opnum " << opnum
                           << std::endl);
    bool inmain = (opnum >= inst()->maincodebegin()
                   && opnum < inst()->maincodeend());

    for (int c = 0; c < inst()->nconnections(); ++c) {
        const Connection& con(inst()->connection(c));
        // If the connection gives a value to this param
        if (con.dst.param == symindex) {
            // already_run is a set of layers run for this particular op.
            // Just so we don't stupidly do several consecutive checks on
            // whether we ran this same layer. It's JUST for this op.
            if (already_run) {
                if (already_run->count(con.srclayer))
                    continue;  // already ran that one on this op
                else
                    already_run->insert(con.srclayer);  // mark it
            }

            if (inmain) {
                // There is an instance-wide m_layers_already_run that tries
                // to remember which earlier layers have unconditionally
                // been run at any point in the execution of this layer. But
                // only honor (and modify) that when in the main code
                // section, not when in init ops, which are inherently
                // conditional.
                if (m_layers_already_run.count(con.srclayer)) {
                    continue;  // already unconditionally ran the layer
                }
                if (!m_in_conditional[opnum]) {
                    // Unconditionally running -- mark so we don't do it
                    // again. If we're inside a conditional, don't mark
                    // because it may not execute the conditional body.
                    m_layers_already_run.insert(con.srclayer);
                }
            }

            // If the earlier layer it comes from has not yet been
            // executed, do so now.
            llvm_call_layer(con.srclayer);
        }
    }
}


LLVMGEN (llvm_gen_useparam)
{
    OSL_ASSERT (! rop.inst()->unused() &&
            "oops, thought this layer was unused, why do we call it?");
    OSL_DEV_ONLY(std::cout << ">>>>>>>>>>>>>>>>>>>>>llvm_gen_useparam <<<<<<<<<<<<<<<<<<<" << std::endl);

    // If we have multiple params needed on this statement, don't waste
    // time checking the same upstream layer more than once.
    std::set<int> already_run;

    Opcode &op (rop.inst()->ops()[opnum]);
    for (int i = 0;  i < op.nargs();  ++i) {
        Symbol& sym = *rop.opargsym (op, i);
        int symindex = rop.inst()->arg (op.firstarg()+i);
        rop.llvm_run_connected_layers (sym, symindex, opnum, &already_run);
        // If it's an interpolated (userdata) parameter and we're
        // initializing them lazily, now we have to do it.
        if (sym.symtype() == SymTypeParam
                && ! sym.lockgeom() && ! sym.typespec().is_closure()
                && ! sym.connected() && ! sym.connected_down()
                && rop.shadingsys().lazy_userdata()) {
            rop.llvm_assign_initial_value (sym, rop.ll.mask_as_int(rop.ll.current_mask()));
        }
    }
    return true;
}


// Generic llvm code generation.  See the comments in llvm_ops.cpp for
// the full list of assumptions and conventions.  But in short:
//   1. All polymorphic and derivative cases implemented as functions in
//      llvm_ops.cpp -- no custom IR is needed.
//   2. Naming conention is: osl_NAME_{args}, where args is the
//      concatenation of type codes for all args including return value --
//      f/i/v/m/s for float/int/triple/matrix/string, and df/dv/dm for
//      duals.
//   3. The function returns scalars as an actual return value (that
//      must be stored), but "returns" aggregates or duals in the first
//      argument.
//   4. Duals and aggregates are passed as void*'s, float/int/string
//      passed by value.
//   5. Note that this only works if triples are all treated identically,
//      this routine can't be used if it must be polymorphic based on
//      color, point, vector, normal differences.
//
LLVMGEN (llvm_gen_generic)
{
    Opcode &op (rop.inst()->ops()[opnum]);

    bool uniformFormOfFunction = true;
    for (int i = 0;  i < op.nargs();  ++i) {
        Symbol *s (rop.opargsym (op, i));
        if(s->is_uniform() == false) {
            uniformFormOfFunction = false;
        }
    }

    Symbol& Result  = *rop.opargsym (op, 0);

    std::vector<const Symbol *> args;
    bool any_deriv_args = false;
    for (int i = 0;  i < op.nargs();  ++i) {
        Symbol *s (rop.opargsym (op, i));
        args.push_back (s);
        any_deriv_args |= (i > 0 && s->has_derivs() && !s->typespec().is_matrix());
    }

    // Special cases: functions that have no derivs -- suppress them
    if (any_deriv_args)
        if (op.opname() == op_logb  ||
            op.opname() == op_floor || op.opname() == op_ceil ||
            op.opname() == op_round || op.opname() == op_step ||
            op.opname() == op_trunc ||
            op.opname() == op_sign)
            any_deriv_args = false;

    FuncSpec func_spec(op.opname().c_str());
    if (uniformFormOfFunction) {
        func_spec.unbatch();
    }

    for (int i = 0;  i < op.nargs();  ++i) {
        Symbol *s (rop.opargsym (op, i));
        bool has_derivs = any_deriv_args && Result.has_derivs() && s->has_derivs() && !s->typespec().is_matrix();
        func_spec.arg(*s,has_derivs,uniformFormOfFunction);
    }

    OSL_DEV_ONLY(std::cout << "llvm_gen_generic " << rop.build_name(func_spec) << std::endl);

    if (! Result.has_derivs() || ! any_deriv_args) {
        // Right now all library calls are not LLVM IR, so can't be inlined
        // In future perhaps we can detect if function exists in module
        // and choose to inline.
        // Controls if parameters are passed by value or pointer
        // and if the mask is passed as llvm type or integer
        constexpr bool functionIsLlvmInlined = false;

        // This can get a bit confusing here,
        // basically in the uniform version, scalar values can be returned by value
        // by functions.  However, if varying, those scalar's are really wide
        // and we can't return by value.  Except if the function in question
        // is llvm source marked as always inline.  In that case we can return
        // wide types.  For all other cases we need to pass a pointer to the
        // where the return value needs to go.

        // Don't compute derivs -- either not needed or not provided in args
        if (Result.typespec().aggregate() == TypeDesc::SCALAR &&
            (uniformFormOfFunction || functionIsLlvmInlined)) {
            OSL_DEV_ONLY(std::cout << ">>stores return value " << rop.build_name(func_spec) << std::endl);
            llvm::Value *r = rop.llvm_call_function (func_spec,
                                                     &(args[1]), op.nargs()-1,
                                                     /*deriv_ptrs*/ false,
                                                     uniformFormOfFunction,
                                                     functionIsLlvmInlined,
                                                     false /*ptrToReturnStructIs1stArg*/);
            // The store will deal with masking
            rop.llvm_store_value (r, Result);
        } else {
            OSL_DEV_ONLY(std::cout << ">>return value is pointer " << rop.build_name(func_spec) << std::endl);

            rop.llvm_call_function (func_spec,
                                    (args.size())? &(args[0]): NULL, op.nargs(),
                                    /*deriv_ptrs*/ false,
                                    uniformFormOfFunction,
                                    functionIsLlvmInlined,
                                    true /*ptrToReturnStructIs1stArg*/);
        }
        rop.llvm_zero_derivs (Result);
    } else {
        // Cases with derivs
        OSL_DEV_ONLY(std::cout << " Cases with derivs");
        OSL_ASSERT (Result.has_derivs() && any_deriv_args);
        rop.llvm_call_function (func_spec,
                                (args.size())? &(args[0]): NULL, op.nargs(),
                                /*deriv_ptrs*/ true, uniformFormOfFunction, false /*functionIsLlvmInlined*/,
                                true /*ptrToReturnStructIs1stArg*/);
    }

    OSL_DEV_ONLY(std::cout << std::endl);

    return true;
}


LLVMGEN (llvm_gen_add)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);

    bool op_is_uniform = A.is_uniform() && B.is_uniform();
    bool result_is_uniform = Result.is_uniform();
    OSL_ASSERT(op_is_uniform || !result_is_uniform);

    OSL_ASSERT (! A.typespec().is_array() && ! B.typespec().is_array());
    if (Result.typespec().is_closure()) {
        OSL_ASSERT(0 && "incomplete");
        OSL_ASSERT (A.typespec().is_closure() && B.typespec().is_closure());
        llvm::Value *valargs[] = {
            rop.sg_void_ptr(),
            rop.llvm_load_value (A),
            rop.llvm_load_value (B)};
        OSL_ASSERT(0 && "incomplete");
        llvm::Value *res = rop.ll.call_function ("osl_add_closure_closure", valargs);
        rop.llvm_store_value (res, Result, 0, NULL, 0);
        return true;
    }

    TypeDesc type = Result.typespec().simpletype();
    int num_components = type.aggregate;

    // The following should handle f+f, v+v, v+f, f+v, i+i
    // That's all that should be allowed by oslc.
    for (int i = 0; i < num_components; i++) {
        OSL_DEV_ONLY(std::cout << "llvm_gen_add component(" << i << ") of " << A.name() << " " << B.name() << std::endl);
        llvm::Value *a = rop.loadLLVMValue (A, i, 0, type, op_is_uniform);
        llvm::Value *b = rop.loadLLVMValue (B, i, 0, type, op_is_uniform);
        if (!a || !b)
            return false;
        llvm::Value *r = rop.ll.op_add (a, b);
        if (op_is_uniform && !result_is_uniform)
        {
            r = rop.ll.widen_value(r);
        }
        rop.storeLLVMValue (r, Result, i, 0);
    }

    if (Result.has_derivs()) {
        if (A.has_derivs() || B.has_derivs()) {
            for (int d = 1;  d <= 2;  ++d) {  // dx, dy
                for (int i = 0; i < num_components; i++) {
                    llvm::Value *a = rop.loadLLVMValue (A, i, d, type, op_is_uniform);
                    llvm::Value *b = rop.loadLLVMValue (B, i, d, type, op_is_uniform);
                    llvm::Value *r = rop.ll.op_add (a, b);
                    if (op_is_uniform && !result_is_uniform)
                    {
                        r = rop.ll.widen_value(r);
                    }
                    rop.storeLLVMValue (r, Result, i, d);
                }
            }
        } else {
            // Result has derivs, operands do not
            rop.llvm_zero_derivs (Result);
        }
    }
    return true;
}


LLVMGEN (llvm_gen_sub)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);

    bool op_is_uniform = A.is_uniform() && B.is_uniform();
    bool result_is_uniform = Result.is_uniform();
    OSL_ASSERT(op_is_uniform || !result_is_uniform);

    TypeDesc type = Result.typespec().simpletype();
    int num_components = type.aggregate;

    OSL_ASSERT (! Result.typespec().is_closure_based() &&
            "subtraction of closures not supported");

    // The following should handle f-f, v-v, v-f, f-v, i-i
    // That's all that should be allowed by oslc.
    for (int i = 0; i < num_components; i++) {
        OSL_DEV_ONLY(std::cout << "llvm_gen_sub component(" << i << ") of " << A.name() << " " << B.name() << std::endl);
        llvm::Value *a = rop.loadLLVMValue (A, i, 0, type, op_is_uniform);
        llvm::Value *b = rop.loadLLVMValue (B, i, 0, type, op_is_uniform);
        if (!a || !b)
            return false;
        llvm::Value *r = rop.ll.op_sub (a, b);
        if (op_is_uniform && !result_is_uniform)
        {
            r = rop.ll.widen_value(r);
        }
        rop.storeLLVMValue (r, Result, i, 0);
    }

    if (Result.has_derivs()) {
        if (A.has_derivs() || B.has_derivs()) {
            for (int d = 1;  d <= 2;  ++d) {  // dx, dy
                for (int i = 0; i < num_components; i++) {
                    llvm::Value *a = rop.loadLLVMValue (A, i, d, type, op_is_uniform);
                    llvm::Value *b = rop.loadLLVMValue (B, i, d, type, op_is_uniform);
                    llvm::Value *r = rop.ll.op_sub (a, b);
                    if (op_is_uniform && !result_is_uniform)
                    {
                        r = rop.ll.widen_value(r);
                    }
                    rop.storeLLVMValue (r, Result, i, d);
                }
            }
        } else {
            // Result has derivs, operands do not
            rop.llvm_zero_derivs (Result);
        }
    }
    return true;
}


LLVMGEN (llvm_gen_mul)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);

    bool op_is_uniform = A.is_uniform() && B.is_uniform();

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = !Result.typespec().is_closure_based() && Result.typespec().is_float_based();
    int num_components = type.aggregate;

    bool resultIsUniform = Result.is_uniform();
    OSL_ASSERT(op_is_uniform || !resultIsUniform);

    // multiplication involving closures
    if (Result.typespec().is_closure()) {
        OSL_ASSERT(0 && "incomplete");
        llvm::Value *valargs[3];
        valargs[0] = rop.sg_void_ptr();
        bool tfloat;
        if (A.typespec().is_closure()) {
            tfloat = B.typespec().is_float();
            valargs[1] = rop.llvm_load_value (A);
            valargs[2] = tfloat ? rop.llvm_load_value (B) : rop.llvm_void_ptr(B);
        } else {
            tfloat = A.typespec().is_float();
            valargs[1] = rop.llvm_load_value (B);
            valargs[2] = tfloat ? rop.llvm_load_value (A) : rop.llvm_void_ptr(A);
        }
        OSL_ASSERT(0 && "incomplete");
        llvm::Value *res = tfloat ? rop.ll.call_function ("osl_mul_closure_float", valargs)
                                  : rop.ll.call_function ("osl_mul_closure_color", valargs);
        rop.llvm_store_value (res, Result, 0, NULL, 0);
        return true;
    }

    // multiplication involving matrices
    if (Result.typespec().is_matrix()) {
        FuncSpec func_spec("mul");
        func_spec.arg(Result,false,op_is_uniform);
        Symbol* A_prime = &A;
        Symbol* B_prime = &B;
        if (B.typespec().is_matrix()) {
            // Always pass the matrix as the 1st operand
            std::swap(A_prime,B_prime);
        }
        func_spec.arg(*A_prime,false,op_is_uniform);
        func_spec.arg(*B_prime,false,op_is_uniform);

        if (op_is_uniform)
            func_spec.unbatch();
        rop.llvm_call_function (func_spec, Result, *A_prime, *B_prime, false /*deriv_ptrs*/, op_is_uniform, false /*functionIsLlvmInlined*/,  true /*ptrToReturnStructIs1stArg*/);

        if (Result.has_derivs())
            rop.llvm_zero_derivs (Result);
        return true;
    }

    // The following should handle f*f, v*v, v*f, f*v, i*i
    // That's all that should be allowed by oslc.
    for (int i = 0; i < num_components; i++) {
        OSL_DEV_ONLY(std::cout << "llvm_gen_mul component(" << i << ") of " << A.name() << " " << B.name() << std::endl);

        llvm::Value *a = rop.llvm_load_value (A, 0, i, type, op_is_uniform);
        llvm::Value *b = rop.llvm_load_value (B, 0, i, type, op_is_uniform);
        if (!a || !b)
            return false;
        llvm::Value *r = rop.ll.op_mul (a, b);

        if (op_is_uniform && !resultIsUniform) {
            r = rop.ll.widen_value(r);
        }

        rop.llvm_store_value (r, Result, 0, i);

        if (Result.has_derivs() && (A.has_derivs() || B.has_derivs())) {
            // Multiplication of duals: (a*b, a*b.dx + a.dx*b, a*b.dy + a.dy*b)
            OSL_ASSERT (is_float);
            llvm::Value *ax = rop.llvm_load_value (A, 1, i, type, op_is_uniform);
            llvm::Value *bx = rop.llvm_load_value (B, 1, i, type, op_is_uniform);
            llvm::Value *abx = rop.ll.op_mul (a, bx);
            llvm::Value *axb = rop.ll.op_mul (ax, b);
            llvm::Value *rx = rop.ll.op_add (abx, axb);
            llvm::Value *ay = rop.llvm_load_value (A, 2, i, type, op_is_uniform);
            llvm::Value *by = rop.llvm_load_value (B, 2, i, type, op_is_uniform);
            llvm::Value *aby = rop.ll.op_mul (a, by);
            llvm::Value *ayb = rop.ll.op_mul (ay, b);
            llvm::Value *ry = rop.ll.op_add (aby, ayb);

            if (op_is_uniform && !resultIsUniform) {
                rx = rop.ll.widen_value(rx);
                ry = rop.ll.widen_value(ry);
            }

            rop.llvm_store_value (rx, Result, 1, i);
            rop.llvm_store_value (ry, Result, 2, i);
        }
    }

    if (Result.has_derivs() &&  ! (A.has_derivs() || B.has_derivs())) {
        // Result has derivs, operands do not
        rop.llvm_zero_derivs (Result);
    }

    return true;
}


LLVMGEN (llvm_gen_div)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);

    bool op_is_uniform = A.is_uniform() && B.is_uniform();
    bool resultIsUniform = Result.is_uniform();
    OSL_ASSERT(op_is_uniform || !resultIsUniform);

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = Result.typespec().is_float_based();
    int num_components = type.aggregate;
    int B_num_components = B.typespec().simpletype().aggregate;

    OSL_ASSERT (! Result.typespec().is_closure_based());

    // division involving matrices
    if (Result.typespec().is_matrix()) {
        FuncSpec func_spec("div");
        if (op_is_uniform)
            func_spec.unbatch();
        func_spec.arg(Result,false,op_is_uniform);
        func_spec.arg(A,false,op_is_uniform);
        func_spec.arg(B,false,op_is_uniform);
        {
            LLVM_Util::ScopedMasking require_mask_be_passed;
            if (!op_is_uniform && B.typespec().is_matrix()) {
                // We choose to only support masked version of these functions:
                // osl_div_w16mw16fw16m
                // osl_div_w16mw16mw16m
                OSL_ASSERT(A.typespec().is_matrix() || A.typespec().is_float());
                OSL_ASSERT(Result.typespec().is_matrix() && !resultIsUniform);
                // Because then check the matrices to see if they are affine
                // and take a slow path if not.  Unmasked lanes wold most
                // likely take the slow path, which could have been avoided
                // if we passed the mask in.
                require_mask_be_passed = rop.ll.create_masking_scope(/*enabled=*/true);
            }
            rop.llvm_call_function (func_spec, Result, A, B, false /*deriv_ptrs*/, op_is_uniform, false /*functionIsLlvmInlined*/,  true /*ptrToReturnStructIs1stArg*/);
        }

        if (Result.has_derivs())
            rop.llvm_zero_derivs (Result);
        return true;
    }

    // The following should handle f/f, v/v, v/f, f/v, i/i
    // That's all that should be allowed by oslc.
    llvm::Value * c_zero = (op_is_uniform)?
                            (is_float) ? rop.ll.constant(0.0f) : rop.ll.constant(static_cast<int>(0))
                        :   (is_float) ? rop.ll.wide_constant(0.0f) : rop.ll.wide_constant(static_cast<int>(0));

    bool deriv = (Result.has_derivs() && (A.has_derivs() || B.has_derivs()));
    llvm::Value * c_one = nullptr;
    if (deriv || !is_float ) {
        c_one = (op_is_uniform)?
                                (is_float) ? rop.ll.constant(1.0f) : rop.ll.constant(static_cast<int>(1))
                            :   (is_float) ? rop.ll.wide_constant(1.0f) : rop.ll.wide_constant(static_cast<int>(1));
    }


    llvm::Value *b = nullptr;
    for (int i = 0; i < num_components; i++) {
        llvm::Value *a = rop.llvm_load_value (A, 0, i, type, op_is_uniform);
        // Don't reload the same value multiple times
        if (i < B_num_components) {
            b = rop.llvm_load_value (B, 0, i, type, op_is_uniform);
        }
        if (!a || !b)
            return false;

        llvm::Value *a_div_b;
        if (B.is_constant() && ! rop.is_zero(B) && !is_float) {
            a_div_b = rop.ll.op_div (a, b);
        } else {
            // safe_div, implement here vs. calling a function
            if (is_float) {
                a_div_b = rop.ll.op_div (a, b);
                llvm::Value * b_notFiniteResult = rop.ll.op_is_not_finite(a_div_b);
                a_div_b = rop.ll.op_zero_if (b_notFiniteResult, a_div_b);
            } else {
                llvm::Value * b_not_zero = rop.ll.op_ne(b, c_zero);
                // NOTE:  Not sure why, but llvm " sdiv <16 x i32>" is not generating SIMD but
                // instead reverting to regular scalar divisions
                // This means it will execute an IDIV potentially with a 0 causing and exception
                // because we use the "not equal 0" mask to select a 0 vs. the expected NAN from the vectorized division
                // An alternative to the selecting the replacing the results
                // is to selectively change the divisor to a non zero
                llvm::Value * divisor = rop.ll.op_select (b_not_zero, b, c_one);
                a_div_b = rop.ll.op_select (b_not_zero, rop.ll.op_div (a, divisor), c_zero);
                // Alternatively we could call a library function
                // Alternatively we could could emit SIMD intrinsics directly
            }
        }

        llvm::Value *rx = NULL, *ry = NULL;

        if (deriv) {
            // Division of duals: (a/b, 1/b*(ax-a/b*bx), 1/b*(ay-a/b*by))
            OSL_ASSERT (is_float);
            llvm::Value *binv = rop.ll.op_div (c_one, b);
            llvm::Value * binv_notFiniteResult = rop.ll.op_is_not_finite(binv);
            binv = rop.ll.op_zero_if (binv_notFiniteResult, binv);
            llvm::Value *ax = rop.llvm_load_value (A, 1, i, type, op_is_uniform);
            llvm::Value *bx = rop.llvm_load_value (B, 1, i, type, op_is_uniform);
            llvm::Value *a_div_b_mul_bx = rop.ll.op_mul (a_div_b, bx);
            llvm::Value *ax_minus_a_div_b_mul_bx = rop.ll.op_sub (ax, a_div_b_mul_bx);
            rx = rop.ll.op_mul (binv, ax_minus_a_div_b_mul_bx);
            llvm::Value *ay = rop.llvm_load_value (A, 2, i, type, op_is_uniform);
            llvm::Value *by = rop.llvm_load_value (B, 2, i, type, op_is_uniform);
            llvm::Value *a_div_b_mul_by = rop.ll.op_mul (a_div_b, by);
            llvm::Value *ay_minus_a_div_b_mul_by = rop.ll.op_sub (ay, a_div_b_mul_by);
            ry = rop.ll.op_mul (binv, ay_minus_a_div_b_mul_by);
        }

        if (op_is_uniform && !resultIsUniform) {
            a_div_b = rop.ll.widen_value(a_div_b);
            if (deriv) {
                rx = rop.ll.widen_value(rx);
                ry = rop.ll.widen_value(ry);
            }
        }
        rop.llvm_store_value (a_div_b, Result, 0, i);
        if (deriv) {
            rop.llvm_store_value (rx, Result, 1, i);
            rop.llvm_store_value (ry, Result, 2, i);
        }

    }

    if (Result.has_derivs() &&  ! (A.has_derivs() || B.has_derivs())) {
        // Result has derivs, operands do not
        rop.llvm_zero_derivs (Result);
    }

    return true;
}


LLVMGEN (llvm_gen_modulus)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& A = *rop.opargsym (op, 1);
    Symbol& B = *rop.opargsym (op, 2);

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = Result.typespec().is_float_based();

    bool op_is_uniform = A.is_uniform() && B.is_uniform();
    bool result_is_uniform = Result.is_uniform();
    OSL_ASSERT(op_is_uniform || !result_is_uniform);

    int num_components = type.aggregate;

    if (is_float && !op_is_uniform) {
        // llvm 5.0.1 did not do a good job with op_mod when its
        // parameters were <16xf32>.  So we will go ahead
        // and call an optimized library version.
        // Future versions of llvm might do better and this
        // could be removed
        BatchedBackendLLVM::TempScope temp_scope(rop);

        std::vector<llvm::Value *> call_args;
        call_args.push_back(rop.llvm_void_ptr(Result));
        call_args.push_back(rop.llvm_load_arg (A, false/*derivs*/, false/*is_uniform*/));
        call_args.push_back(rop.llvm_load_arg (B, false/*derivs*/, false/*is_uniform*/));

        FuncSpec func_spec("fmod");
        func_spec.arg(Result,false/*derivs*/, false/*is_uniform*/);
        func_spec.arg(A,false/*derivs*/, false/*is_uniform*/);
        func_spec.arg(B,false/*derivs*/, false/*is_uniform*/);

        if (rop.ll.is_masking_required() ) {
            func_spec.mask();
            call_args.push_back(rop.ll.mask_as_int(rop.ll.current_mask()));
        }

        rop.ll.call_function (rop.build_name(func_spec), call_args);
    } else {
        for (int i = 0; i < num_components; i++) {

            llvm::Value *a = rop.loadLLVMValue (A, i, 0, type, op_is_uniform);
            llvm::Value *b = rop.loadLLVMValue (B, i, 0, type, op_is_uniform);
            if (!a || !b)
                return false;
            llvm::Value *zeroConstant;
            if (is_float) {
                zeroConstant = op_is_uniform ? rop.ll.constant(0.0f) : rop.ll.wide_constant(0.0f);
            } else {
                // Integer versions of safe mod handled in stdosl.h
                // We will leave the code to handle ints here as well
                zeroConstant = op_is_uniform ? rop.ll.constant(0) : rop.ll.wide_constant(0);
            }

            llvm::Value *is_zero_mask = rop.ll.op_eq(b, zeroConstant);
            llvm::Value *mod_result = rop.ll.op_mod (a, b);
            llvm::Value * r = rop.ll.op_select(is_zero_mask, zeroConstant, mod_result);
            if (op_is_uniform && !result_is_uniform)
            {
                r = rop.ll.widen_value(r);
            }
            rop.storeLLVMValue (r, Result, i, 0);
        }
    }

    if (Result.has_derivs()) {
        OSL_ASSERT (is_float);
        if (A.has_derivs()) {
            // Modulus of duals: (a mod b, ax, ay)
            for (int d = 1;  d <= 2;  ++d) {
                for (int i = 0; i < num_components; i++) {
                    llvm::Value *deriv = rop.loadLLVMValue (A, i, d, type, op_is_uniform);
                    rop.storeLLVMValue (deriv, Result, i, d);
                }
            }
        } else {
            // Result has derivs, operands do not
            rop.llvm_zero_derivs (Result);
        }
    }
    return true;
}


// Simple assignment
LLVMGEN (llvm_gen_assign)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result (*rop.opargsym (op, 0));
    Symbol& Src (*rop.opargsym (op, 1));

    return rop.llvm_assign_impl (Result, Src);
}


// Vector component reference
LLVMGEN (llvm_gen_compref)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    Symbol& Val = *rop.opargsym (op, 1);
    Symbol& Index = *rop.opargsym (op, 2);


    bool op_is_uniform = Result.is_uniform();

    llvm::Value *c = rop.llvm_load_value(Index);

    if (Index.is_uniform()) {

        if (rop.inst()->master()->range_checking()) {
           if (! (Index.is_constant() &&  *(int *)Index.data() >= 0 &&
                  *(int *)Index.data() < 3)) {
               llvm::Value *args[] = { c, rop.ll.constant(3),
                                       rop.ll.constant(Val.name()),
                                       rop.sg_void_ptr(),
                                       rop.ll.constant(op.sourcefile()),
                                       rop.ll.constant(op.sourceline()),
                                       rop.ll.constant(rop.group().name()),
                                       rop.ll.constant(rop.layer()),
                                       rop.ll.constant(rop.inst()->layername()),
                                       rop.ll.constant(rop.inst()->shadername()) };
               c = rop.ll.call_function (rop.build_name("range_check"), args);
               OSL_ASSERT (c);
           }
       }

        for (int d = 0;  d < 3;  ++d) {  // deriv
            llvm::Value *val = NULL;
            if (Index.is_constant()) {
                int i = *(int*)Index.data();
                i = Imath::clamp (i, 0, 2);
                val = rop.llvm_load_value (Val, d, i, TypeDesc::UNKNOWN, op_is_uniform);
            } else {
                // TODO: handle non constant index
                val = rop.llvm_load_component_value (Val, d, c, op_is_uniform);
            }
            rop.llvm_store_value (val, Result, d);
            if (! Result.has_derivs())  // skip the derivs if we don't need them
                break;
        }
    } else {
        OSL_ASSERT(Index.is_constant() == false);
        OSL_ASSERT(op_is_uniform == false);

        if (rop.inst()->master()->range_checking()) {
            BatchedBackendLLVM::TempScope temp_scope(rop);

            // We need a copy of the indices incase the range check clamps them
            llvm::Value * loc_clamped_wide_index = rop.getOrAllocateTemp (TypeSpec(TypeDesc::INT), false /*derivs*/, false /*is_uniform*/, false /*forceBool*/, std::string("range clamped index:") + Val.name().c_str());
            // copy the indices into our temporary
            rop.ll.op_unmasked_store(c, loc_clamped_wide_index);
            llvm::Value *args[] = { rop.ll.void_ptr(loc_clamped_wide_index),
                                   rop.ll.mask_as_int(rop.ll.current_mask()),
                                   rop.ll.constant(3),
                                   rop.ll.constant(Val.name()),
                                   rop.sg_void_ptr(),
                                   rop.ll.constant(op.sourcefile()),
                                   rop.ll.constant(op.sourceline()),
                                   rop.ll.constant(rop.group().name()),
                                   rop.ll.constant(rop.layer()),
                                   rop.ll.constant(rop.inst()->layername()),
                                   rop.ll.constant(rop.inst()->shadername()) };
            rop.ll.call_function (rop.build_name(FuncSpec("range_check").mask()), args);

            // Use the range check indices
            // Although as our implementation below doesn't use any
            // out of range values, clamping the indices here
            // is of questionable value
            c = rop.ll.op_load(loc_clamped_wide_index);
       }

        // As the index is logically bound to 0, 1, or 2
        // instead of doing a gather (which we will assume to cost 16 loads)
        // We can just load all 3 components and blend them based on the index == 0, index == 1, index == 2
        llvm::Value *comp0Mask = rop.ll.op_eq(c, rop.ll.wide_constant(0));
        llvm::Value *comp1Mask = rop.ll.op_eq(c, rop.ll.wide_constant(1));
        // If index != 0 && index != 1, assume index == 2
        // Essentially free clamping

        for (int d = 0;  d < 3;  ++d) {  // deriv
            llvm::Value *valc0 = rop.llvm_load_value (Val, d, 0, TypeDesc::UNKNOWN, op_is_uniform);
            llvm::Value *valc1 = rop.llvm_load_value (Val, d, 1, TypeDesc::UNKNOWN, op_is_uniform);
            llvm::Value *valc2 = rop.llvm_load_value (Val, d, 2, TypeDesc::UNKNOWN, op_is_uniform);
            llvm::Value *valc0_c2 = rop.ll.op_select(comp0Mask,valc0,valc2);
            llvm::Value *valc0_c1_c2 = rop.ll.op_select(comp1Mask,valc1,valc0_c2);

            rop.llvm_store_value (valc0_c1_c2, Result, d);
            if (! Result.has_derivs())  // skip the derivs if we don't need them
                break;
        }
    }
    return true;
}


// Construct color, optionally with a color transformation from a named
// color space.
LLVMGEN (llvm_gen_construct_color)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    bool using_space = (op.nargs() == 5);
    Symbol& Space = *rop.opargsym (op, 1);
    Symbol& X = *rop.opargsym (op, 1+using_space);
    Symbol& Y = *rop.opargsym (op, 2+using_space);
    Symbol& Z = *rop.opargsym (op, 3+using_space);
    OSL_ASSERT (Result.typespec().is_triple() && X.typespec().is_float() &&
            Y.typespec().is_float() && Z.typespec().is_float() &&
            (using_space == false || Space.typespec().is_string()));

#if 0 && defined(OSL_DEV)
    bool resultIsUniform = Result.is_uniform();
    bool spaceIsUniform = Space.is_uniform();
    bool xIsUniform = X.is_uniform();
    bool yIsUniform = Y.is_uniform();
    bool zIsUniform = Z.is_uniform();
    std::cout << "llvm_gen_construct_color Result=" << Result.name().c_str() << ((resultIsUniform) ? "(uniform)" : "(varying)");
    if (using_space) {
            std::cout << " Space=" << Space.name().c_str() << ((spaceIsUniform) ? "(uniform)" : "(varying)");
    }
    std::cout << " X=" << X.name().c_str() << ((xIsUniform) ? "(uniform)" : "(varying)")
              << " Y=" << Y.name().c_str()<< ((yIsUniform) ? "(uniform)" : "(varying)")
              << " Z=" << Z.name().c_str()<< ((zIsUniform) ? "(uniform)" : "(varying)")
              << std::endl;
#endif
    bool result_is_uniform = Result.is_uniform();

    // First, copy the floats into the vector
    int dmax = Result.has_derivs() ? 3 : 1;
    for (int d = 0;  d < dmax;  ++d) {  // loop over derivs
        for (int c = 0;  c < 3;  ++c) {  // loop over components
            const Symbol& comp = *rop.opargsym (op, c+1+using_space);
            llvm::Value* val = rop.llvm_load_value (comp, d, NULL, 0, TypeDesc::TypeFloat, result_is_uniform);
            rop.llvm_store_value (val, Result, d, NULL, c);
        }
    }

    // Do the color space conversion in-place, if called for
    if (using_space) {
        // TODO: detect if space is constant, then call space specific conversion
        // functions to avoid doing runtime detection of space.

        bool space_is_uniform = Space.is_uniform();
        FuncSpec func_spec("prepend_color_from");

        // Ignoring derivs to match existing behavior, see comment below where
        // any derivs on the result are 0'd out
        func_spec.arg(Result, false /*derivs*/, result_is_uniform);
        func_spec.arg(Space, false /*derivs*/, space_is_uniform);

        llvm::Value *args[4];
        // NOTE:  Shader Globals is only passed to provide access to report an error to the context
        // no implicit dependency on any Shader Globals is necessary
        args[0] = rop.sg_void_ptr ();  // shader globals
        args[1] = rop.llvm_void_ptr (Result, 0);  // color
        args[2] = space_is_uniform ? rop.llvm_load_value (Space) : rop.llvm_void_ptr(Space); // from
        int arg_count = 3;
        // Until we avoid calling back into the shading system,
        // always call the masked version if we are not uniform
        // to allow skipping callbacks for masked off lanes
        if(!result_is_uniform /*&& rop.ll.is_masking_required()*/) {
            args[arg_count++] = rop.ll.mask_as_int(rop.ll.current_mask());
            func_spec.mask();
        }

        rop.ll.call_function (rop.build_name(func_spec), cspan<llvm::Value *>(args, arg_count));
        // FIXME(deriv): Punt on derivs for color ctrs with space names.
        // We should try to do this right, but we never had it right for
        // the interpreter, to it's probably not an emergency.
        if (Result.has_derivs())
            rop.llvm_zero_derivs (Result);
    }

    return true;
}

// Derivs
LLVMGEN (llvm_gen_DxDy)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result (*rop.opargsym (op, 0));
    Symbol& Src (*rop.opargsym (op, 1));
    int deriv = (op.opname() == "Dx") ? 1 : 2;

    bool result_is_uniform = Result.is_uniform();

    for (int i = 0; i < Result.typespec().aggregate(); ++i) {
        llvm::Value* src_val = rop.llvm_load_value (Src, deriv, i, TypeDesc::UNKNOWN, result_is_uniform);
        rop.storeLLVMValue (src_val, Result, i, 0);
    }

    // Don't have 2nd order derivs
    rop.llvm_zero_derivs (Result);
    return true;
}

// Dz
LLVMGEN (llvm_gen_Dz)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result (*rop.opargsym (op, 0));
    Symbol& Src (*rop.opargsym (op, 1));

    bool result_is_uniform = Result.is_uniform();

    if (&Src == rop.inst()->symbol(rop.inst()->Psym())) {
        // dPdz -- the only Dz we know how to take
        int deriv = 3;
        for (int i = 0; i < Result.typespec().aggregate(); ++i) {
            llvm::Value* src_val = rop.llvm_load_value (Src, deriv, i, TypeDesc::UNKNOWN, result_is_uniform);
            rop.storeLLVMValue (src_val, Result, i, 0);
        }
        // Don't have 2nd order derivs
        rop.llvm_zero_derivs (Result);
    } else {
        // Punt, everything else for now returns 0 for Dz
        // FIXME?
        rop.llvm_assign_zero (Result);
    }
    return true;
}


// Comparison ops
LLVMGEN (llvm_gen_compare_op)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol &Result (*rop.opargsym (op, 0));
    Symbol &A (*rop.opargsym (op, 1));
    Symbol &B (*rop.opargsym (op, 2));
    OSL_ASSERT (Result.typespec().is_int() && ! Result.has_derivs());

    bool op_is_uniform = A.is_uniform() && B.is_uniform();
    bool result_is_uniform = Result.is_uniform();

    if (A.typespec().is_closure()) {
        OSL_ASSERT(0 && "incomplete");
        OSL_ASSERT (B.typespec().is_int() &&
                "Only closure==0 and closure!=0 allowed");
        llvm::Value *a = rop.llvm_load_value (A);
        llvm::Value *b = rop.ll.void_ptr_null ();
        llvm::Value *r = (op.opname()==op_eq) ? rop.ll.op_eq(a,b)
                                              : rop.ll.op_ne(a,b);
        // TODO: handle convert the single bit bool into an int, if necessary
        rop.llvm_store_value (r, Result);
        return true;
    }

    int num_components = std::max (A.typespec().aggregate(), B.typespec().aggregate());
    bool float_based = A.typespec().is_float_based() || B.typespec().is_float_based();
    TypeDesc cast (float_based ? TypeDesc::FLOAT : TypeDesc::UNKNOWN);

    llvm::Value* final_result = 0;
    ustring opname = op.opname();

    for (int i = 0; i < num_components; i++) {
        // Get A&B component i -- note that these correctly handle mixed
        // scalar/triple comparisons as well as int->float casts as needed.
        llvm::Value* a = rop.loadLLVMValue (A, i, 0, cast, op_is_uniform);
        llvm::Value* b = rop.loadLLVMValue (B, i, 0, cast, op_is_uniform);

        llvm::Type * typeOfA = rop.ll.llvm_typeof(a);
        llvm::Type * typeOfB = rop.ll.llvm_typeof(b);

        if (typeOfA != typeOfB) {
            if ((typeOfA == rop.ll.type_bool() && typeOfB == rop.ll.type_int()) ||
                (typeOfA == rop.ll.type_wide_bool() && typeOfB == rop.ll.type_wide_int())) {

                // TODO: could optimize for contant 0 and 1 and skip the comparison
                a = rop.ll.op_bool_to_int(a);
            }
            if ((typeOfB == rop.ll.type_bool() && typeOfA == rop.ll.type_int()) ||
                (typeOfB == rop.ll.type_wide_bool() && typeOfA == rop.ll.type_wide_int())) {
                b = rop.ll.op_bool_to_int(b);
            }
        }

        // Trickery for mixed matrix/scalar comparisons -- compare
        // on-diagonal to the scalar, off-diagonal to zero
        if (A.typespec().is_matrix() && !B.typespec().is_matrix()) {
            if ((i/4) != (i%4)) {
                if (op_is_uniform)
                    b = rop.ll.constant (0.0f);
                else
                    b = rop.ll.wide_constant (0.0f);
            }
        }
        if (! A.typespec().is_matrix() && B.typespec().is_matrix()) {
            if ((i/4) != (i%4)) {
                if (op_is_uniform)
                    a = rop.ll.constant (0.0f);
                else
                    a = rop.ll.wide_constant (0.0f);
            }
        }

        // Perform the op
        llvm::Value* result = 0;
        if (opname == op_lt) {
            result = rop.ll.op_lt (a, b);
        } else if (opname == op_le) {
            result = rop.ll.op_le (a, b);
        } else if (opname == op_eq) {
            result = rop.ll.op_eq (a, b);
        } else if (opname == op_ge) {
            result = rop.ll.op_ge (a, b);
        } else if (opname == op_gt) {
            result = rop.ll.op_gt (a, b);
        } else if (opname == op_neq) {
            result = rop.ll.op_ne (a, b);
        } else {
            // Don't know how to handle this.
            OSL_ASSERT (0 && "Comparison error");
        }
        OSL_ASSERT (result);

        if (final_result) {
            // Combine the component bool based on the op
            if (opname != op_neq)        // final_result &= result
                final_result = rop.ll.op_and (final_result, result);
            else                         // final_result |= result
                final_result = rop.ll.op_or (final_result, result);
        } else {
            final_result = result;
        }
    }
    OSL_ASSERT (final_result);

    // Lets not convert comparisons from bool to int

    OSL_DEV_ONLY(std::cout << "About to rop.storeLLVMValue (final_result, Result, 0, 0); op_is_uniform=" << op_is_uniform  << std::endl);

    OSL_ASSERT(op_is_uniform || !result_is_uniform);

    if (op_is_uniform && !result_is_uniform)
    {
        final_result = rop.ll.widen_value(final_result);
    }


    // Although we try to use llvm bool (i1) for comparison results
    // sometimes we could not force the data type to be an bool and it remains
    // an int, for those cases we will need to convert the boolean to int
    if (Result.forced_llvm_bool()) {
        if (!result_is_uniform) {
            final_result = rop.ll.llvm_mask_to_native(final_result);
        }
    } else {
        llvm::Type * resultType = rop.ll.llvm_typeof(rop.llvm_get_pointer(Result));
        OSL_ASSERT((resultType == reinterpret_cast<llvm::Type *>(rop.ll.type_wide_int_ptr())) ||
               (resultType == reinterpret_cast<llvm::Type *>(rop.ll.type_int_ptr())));
        final_result = rop.ll.op_bool_to_int (final_result);
    }


    rop.storeLLVMValue (final_result, Result, 0, 0);
    OSL_DEV_ONLY(std::cout << "AFTER to rop.storeLLVMValue (final_result, Result, 0, 0);" << std::endl);

    return true;
}


// Construct spatial triple (point, vector, normal), optionally with a
// transformation from a named coordinate system.
LLVMGEN (llvm_gen_construct_triple)
{
    Opcode &op (rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym (op, 0);
    bool using_space = (op.nargs() == 5);
    Symbol& Space = *rop.opargsym (op, 1);
    Symbol& X = *rop.opargsym (op, 1+using_space);
    Symbol& Y = *rop.opargsym (op, 2+using_space);
    Symbol& Z = *rop.opargsym (op, 3+using_space);
    OSL_ASSERT (Result.typespec().is_triple() && X.typespec().is_float() &&
            Y.typespec().is_float() && Z.typespec().is_float() &&
            (using_space == false || Space.typespec().is_string()));

#if 0 && defined(OSL_DEV)
    bool spaceIsUniform = Space.is_uniform();
    bool xIsUniform = X.is_uniform(X);
    bool yIsUniform = Y.is_uniform(Y);
    bool zIsUniform = Z.is_uniform(Z);
    std::cout << "llvm_gen_construct_triple Result=" << Result.name().c_str();
    if (using_space) {
            std::cout << " Space=" << Space.name().c_str() << ((spaceIsUniform) ? "(uniform)" : "(varying)");
    }
    std::cout << " X=" << X.name().c_str() << ((xIsUniform) ? "(uniform)" : "(varying)")
              << " Y=" << Y.name().c_str()<< ((yIsUniform) ? "(uniform)" : "(varying)")
              << " Z=" << Z.name().c_str()<< ((zIsUniform) ? "(uniform)" : "(varying)")
              << std::endl;
#endif


    bool space_is_uniform = Space.is_uniform();
    bool op_is_uniform = X.is_uniform() && Y.is_uniform() && Z.is_uniform() && space_is_uniform;

    bool resultIsUniform = Result.is_uniform();
    OSL_ASSERT(op_is_uniform || !resultIsUniform);



    // First, copy the floats into the vector
    int dmax = Result.has_derivs() ? 3 : 1;
    for (int d = 0;  d < dmax;  ++d) {  // loop over derivs
        for (int c = 0;  c < 3;  ++c) {  // loop over components
            const Symbol& comp = *rop.opargsym (op, c+1+using_space);
            llvm::Value* val = rop.llvm_load_value (comp, d, NULL, 0, TypeDesc::TypeFloat, op_is_uniform);

            if (op_is_uniform && !resultIsUniform) {
                rop.llvm_broadcast_uniform_value(val, Result, d, c);
            } else {
                rop.llvm_store_value (val, Result, d, NULL, c);
            }

        }
    }

    // Do the transformation in-place, if called for
    if (using_space) {
        ustring from, to;  // N.B. initialize to empty strings
        if (Space.is_constant()) {
            from = *(ustring *)Space.data();
            if (from == Strings::common ||
                from == rop.shadingsys().commonspace_synonym())
                return true;  // no transformation necessary
        }
        TypeDesc::VECSEMANTICS vectype = TypeDesc::POINT;
        ustring triple_type("point");
        if (op.opname() == "vector") {
            vectype = TypeDesc::VECTOR;
            triple_type = ustring("vector");
        } else if (op.opname() == "normal") {
            vectype = TypeDesc::NORMAL;
            triple_type = ustring("normal");
        }

        OSL_DEV_ONLY(std::cout << "llvm_gen_construct_triple Result.has_derivs()=" << Result.has_derivs() << std::endl);


        RendererServices *rend (rop.shadingsys().renderer());

        OSL_ASSERT((false == rend->transform_points (NULL, Strings::_emptystring_, Strings::_emptystring_, 0.0f, NULL, NULL, 0, vectype)) && "incomplete");
        // Didn't want to make RenderServices have to deal will all variants of from/to
        // unless it is going to be used, yes it will have to be done though
//        if (rend->transform_points (NULL, from, to, 0.0f, NULL, NULL, 0, vectype)) {
//            // TODO: Handle non-uniform case below minding mask values
//            OSL_ASSERT(0 && "incomplete"); // needs uniform version accepting BatchedShaderGlobals
//
//            // renderer potentially knows about a nonlinear transformation.
//            // Note that for the case of non-constant strings, passing empty
//            // from & to will make transform_points just tell us if ANY
//            // nonlinear transformations potentially are supported.
//            rop.ll.call_function ("osl_transform_triple_nonlinear", args, 8);
//        } else
        llvm::Value * transform = rop.temp_wide_matrix_ptr();
        llvm::Value *succeeded_as_int = nullptr;
        {
            llvm::Value *args[] = { rop.sg_void_ptr(),
                rop.ll.void_ptr(transform),
                space_is_uniform ? rop.llvm_load_value(Space) : rop.llvm_void_ptr(Space),
                rop.ll.constant(Strings::common),
                rop.ll.mask_as_int(rop.ll.current_mask())};

            // Dynamically build function name
            FuncSpec func_spec("build_transform_matrix");
            func_spec.arg_varying(TypeDesc::TypeMatrix);
            func_spec.arg(Space,false/*derivs*/, space_is_uniform);
            func_spec.arg_uniform(TypeDesc::TypeString);
            func_spec.mask();

            succeeded_as_int = rop.ll.call_function (rop.build_name(func_spec), args);
        }
        {
            llvm::Value *args[] = {
                rop.llvm_void_ptr(Result /* src */),
                rop.llvm_void_ptr(Result /* dest */),
                rop.ll.void_ptr(transform),
                succeeded_as_int,
                rop.ll.mask_as_int(rop.ll.current_mask())};

            OSL_ASSERT(Result.is_uniform() == false && "unreachable case");
            // definitely not a nonlinear transformation

            // Dynamically build function name
            auto transform_name = llvm::Twine("transform_") + triple_type.c_str();
            FuncSpec func_spec(transform_name);
            func_spec.arg(Result, Result.has_derivs(), resultIsUniform);
            func_spec.arg(Result, Result.has_derivs(), resultIsUniform);
            func_spec.arg_varying(TypeDesc::TypeMatrix44);
            func_spec.mask();

            rop.ll.call_function (rop.build_name(func_spec), args);
        }
    }
    return true;
}


LLVMGEN (llvm_gen_end)
{
    // Dummy routine needed only for the op_descriptor table
    return false;
}


// batched code gen left to be implemented
#define TBD_LLVMGEN(NAME) \
LLVMGEN(NAME) \
{ \
    OSL_ASSERT(0 && #NAME && " To Be Implemented"); \
    return false; \
} \

TBD_LLVMGEN(llvm_gen_getattribute)
TBD_LLVMGEN(llvm_gen_calculatenormal)
TBD_LLVMGEN(llvm_gen_compassign)
TBD_LLVMGEN(llvm_gen_sincos)
TBD_LLVMGEN(llvm_gen_andor)
TBD_LLVMGEN(llvm_gen_filterwidth)
TBD_LLVMGEN(llvm_gen_arraylength)
TBD_LLVMGEN(llvm_gen_arraycopy)
TBD_LLVMGEN(llvm_gen_neg)
TBD_LLVMGEN(llvm_gen_texture)
TBD_LLVMGEN(llvm_gen_printf)
TBD_LLVMGEN(llvm_gen_area)
TBD_LLVMGEN(llvm_gen_getmessage)
TBD_LLVMGEN(llvm_gen_bitwise_binary_op)
TBD_LLVMGEN(llvm_gen_if)
TBD_LLVMGEN(llvm_gen_noise)
TBD_LLVMGEN(llvm_gen_transformc)
TBD_LLVMGEN(llvm_gen_pointcloud_search)
TBD_LLVMGEN(llvm_gen_mxcompref)
TBD_LLVMGEN(llvm_gen_dict_find)
TBD_LLVMGEN(llvm_gen_functioncall)
TBD_LLVMGEN(llvm_gen_functioncall_nr)
TBD_LLVMGEN(llvm_gen_clamp)
TBD_LLVMGEN(llvm_gen_aassign)
TBD_LLVMGEN(llvm_gen_mxcompassign)
TBD_LLVMGEN(llvm_gen_get_simple_SG_field)
TBD_LLVMGEN(llvm_gen_raytype)
TBD_LLVMGEN(llvm_gen_trace)
TBD_LLVMGEN(llvm_gen_pointcloud_get)
TBD_LLVMGEN(llvm_gen_return)
TBD_LLVMGEN(llvm_gen_regex)
TBD_LLVMGEN(llvm_gen_pointcloud_write)
TBD_LLVMGEN(llvm_gen_isconstant)
TBD_LLVMGEN(llvm_gen_loop_op)
TBD_LLVMGEN(llvm_gen_matrix)
TBD_LLVMGEN(llvm_gen_select)
TBD_LLVMGEN(llvm_gen_split)
TBD_LLVMGEN(llvm_gen_unary_op)
TBD_LLVMGEN(llvm_gen_aref)
TBD_LLVMGEN(llvm_gen_luminance)
TBD_LLVMGEN(llvm_gen_dict_value)
TBD_LLVMGEN(llvm_gen_loopmod_op)
TBD_LLVMGEN(llvm_gen_transform)
TBD_LLVMGEN(llvm_gen_closure)
TBD_LLVMGEN(llvm_gen_gettextureinfo)
TBD_LLVMGEN(llvm_gen_blackbody)
TBD_LLVMGEN(llvm_gen_spline)
TBD_LLVMGEN(llvm_gen_dict_next)
TBD_LLVMGEN(llvm_gen_texture3d)
TBD_LLVMGEN(llvm_gen_nop)
TBD_LLVMGEN(llvm_gen_minmax)
TBD_LLVMGEN(llvm_gen_getmatrix)
TBD_LLVMGEN(llvm_gen_environment)
TBD_LLVMGEN(llvm_gen_mix)
TBD_LLVMGEN(llvm_gen_setmessage)


// TODO: rest of gen functions to be added in separate PR

};  // namespace pvt
OSL_NAMESPACE_EXIT
