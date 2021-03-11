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


/// Macro that defines the arguments to LLVM IR generating routines
///
#define LLVMGEN_ARGS BatchedBackendLLVM &rop, int opnum

/// Macro that defines the full declaration of an LLVM generator.
///
#define LLVMGEN(name) bool name(LLVMGEN_ARGS)



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
        = Strutil::fmt::format("{}_{}", m_library_selector,
                               layer_function_name(group(), *parent));

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



// Comparison ops
LLVMGEN(llvm_gen_compare_op)
{
    OSL_ASSERT(0 && "To Be Implemented");
    return false;
}



LLVMGEN(llvm_gen_getattribute)
{
    OSL_ASSERT(0 && "To Be Implemented");
    return false;
}



// TODO: rest of gen functions to be added in separate PR

};  // namespace pvt
OSL_NAMESPACE_EXIT
