// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <set>

#include <llvm/IR/Constant.h>

#include <OSL/oslconfig.h>

#include <OSL/batched_rendererservices.h>

#ifdef OSL_DEV
#    include <llvm/Support/raw_os_ostream.h>
#endif

#include "batched_backendllvm.h"



using namespace OSL;
using namespace OSL::pvt;

OSL_NAMESPACE_ENTER

namespace pvt {

static ustring op_and("and");
static ustring op_bitand("bitand");
static ustring op_bitor("bitor");
static ustring op_break("break");
static ustring op_ceil("ceil");
static ustring op_compl("compl");
static ustring op_continue("continue");
static ustring op_dowhile("dowhile");
static ustring op_eq("eq");
static ustring op_error("error");
static ustring op_floor("floor");
static ustring op_format("format");
static ustring op_fprintf("fprintf");
static ustring op_ge("ge");
static ustring op_gt("gt");
static ustring op_logb("logb");
static ustring op_le("le");
static ustring op_lt("lt");
static ustring op_min("min");
static ustring op_neq("neq");
static ustring op_printf("printf");
static ustring op_round("round");
static ustring op_shl("shl");
static ustring op_shr("shr");
static ustring op_sign("sign");
static ustring op_step("step");
static ustring op_trunc("trunc");
static ustring op_warning("warning");
static ustring op_xor("xor");

static ustring u_distance("distance");
static ustring u_index("index");

/// Macro that defines the arguments to LLVM IR generating routines
///
#define LLVMGEN_ARGS BatchedBackendLLVM &rop, int opnum

/// Macro that defines the full declaration of an LLVM generator.
///
#define LLVMGEN(name) bool name(LLVMGEN_ARGS)

// Forward decl
LLVMGEN(llvm_gen_generic);


typedef typename BatchedBackendLLVM::FuncSpec FuncSpec;



void
BatchedBackendLLVM::llvm_gen_debug_printf(string_view message)
{
    ustring s = ustring::fmtformat("({} {}) {}", inst()->shadername(),
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
    //         parent_layer (sg, groupdata, wide_shadeindex_ptr,
    //                       userdata_base_ptr, output_base_ptr,
    //                       execution_mask, interactive_params_ptr);
    // if it's a conditional call, or
    //     parent_layer (sg, groupdata, wide_shadeindex_ptr,
    //                   userdata_base_ptr, output_base_ptr,
    //                   execution_mask, interactive_params_ptr);
    // if it's run unconditionally.
    // The code in the parent layer itself will set its 'executed' flag.

    llvm::Value* args[7];
    args[0] = sg_ptr();
    args[1] = groupdata_ptr();
    args[2] = ll.void_ptr(wide_shadeindex_ptr());
    args[3] = userdata_base_ptr();
    args[4] = output_base_ptr();

    ShaderInstance* parent       = group()[layer];
    llvm::Value* layerfield      = layer_run_ref(layer_remap(layer));
    llvm::BasicBlock *then_block = NULL, *after_block = NULL;
    llvm::Value* lanes_requiring_execution_value = nullptr;
    if (!unconditional) {
        llvm::Value* previously_executed = ll.int_as_mask(
            ll.op_load(ll.type_int(), layerfield));
        llvm::Value* lanes_requiring_execution
            = ll.op_select(previously_executed, ll.wide_constant_bool(false),
                           ll.current_mask());
        lanes_requiring_execution_value = ll.mask_as_int(
            lanes_requiring_execution);
        llvm::Value* execution_required
            = ll.op_ne(lanes_requiring_execution_value, ll.constant(0));
        then_block = ll.new_basic_block(
            llvm_debug() ? Strutil::fmt::format("then layer {}", layer)
                         : std::string());
        after_block = ll.new_basic_block(
            llvm_debug() ? Strutil::fmt::format("after layer {}", layer)
                         : std::string());
        ll.op_branch(execution_required, then_block, after_block);
        // insert point is now then_block
    } else {
        lanes_requiring_execution_value = ll.mask_as_int(ll.shader_mask());
    }

    args[5] = lanes_requiring_execution_value;
    args[6] = m_llvm_interactive_params_ptr;

    // Before the merge, keeping in case we broke it
    //std::string name = fmtformat("{}_{}_{}", m_library_selector,  parent->layername().c_str(),
    //                             parent->id());
    std::string name
        = Strutil::fmt::format("{}_{}", m_library_selector,
                               layer_function_name(group(), *parent));

    // Mark the call as a fast call
    llvm::Value* funccall = ll.call_function(name.c_str(), args);
    if (!parent->entry_layer())
        ll.mark_fast_func_call(funccall);

    if (!unconditional)
        ll.op_branch(after_block);  // also moves insert point

    shadingsys().m_stat_call_layers_inserted++;
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

            if (shadingsys().m_opt_useparam && inmain) {
                // m_call_layers_inserted tracks if we've already run this layer inside the current basic block
                const CallLayerKey key = { m_bblockids[opnum], con.srclayer };
                if (m_call_layers_inserted.count(key)) {
                    continue;
                }
                m_call_layers_inserted.insert(key);
            }

            // If the earlier layer it comes from has not yet been
            // executed, do so now.
            llvm_call_layer(con.srclayer);
        }
    }
}



LLVMGEN(llvm_gen_nop)
{
    return true;
}



LLVMGEN(llvm_gen_useparam)
{
    OSL_ASSERT(!rop.inst()->unused()
               && "oops, thought this layer was unused, why do we call it?");
    OSL_DEV_ONLY(
        std::cout << ">>>>>>>>>>>>>>>>>>>>>llvm_gen_useparam <<<<<<<<<<<<<<<<<<<"
                  << std::endl);

    // If we have multiple params needed on this statement, don't waste
    // time checking the same upstream layer more than once.
    std::set<int> already_run;

    Opcode& op(rop.inst()->ops()[opnum]);
    for (int i = 0; i < op.nargs(); ++i) {
        Symbol& sym  = *rop.opargsym(op, i);
        int symindex = rop.inst()->arg(op.firstarg() + i);
        rop.llvm_run_connected_layers(sym, symindex, opnum, &already_run);
        // If it's an interpolated (userdata) parameter and we're
        // initializing them lazily, now we have to do it.
        if ((sym.symtype() == SymTypeParam
             || sym.symtype() == SymTypeOutputParam)
            && sym.interpolated() && !sym.typespec().is_closure()
            && !sym.connected() && !sym.connected_down()
            && rop.shadingsys().lazy_userdata()) {
            rop.llvm_assign_initial_value(sym, rop.ll.mask_as_int(
                                                   rop.ll.current_mask()));
        }
    }

    rop.increment_useparam_ops();

    return true;
}



// Used for printf, error, warning, format, fprintf
LLVMGEN(llvm_gen_printf)
{
    Opcode& op(rop.inst()->ops()[opnum]);

    // Prepare the args for the call

    // Which argument is the format string?  Usually 0, but for op
    // format() and fprintf(), the formatting string is argument #1.
    int format_arg = (op.opname() == "format" || op.opname() == "fprintf") ? 1
                                                                           : 0;
    Symbol& format_sym = *rop.opargsym(op, format_arg);

    OSL_ASSERT(format_sym.is_uniform());

    // For WIDE parameters we want to test the lane first to see
    // if we need to extract values or not
    struct DelayedExtraction {
        int argument_slot;
        bool is_float;
        bool is_closure;
        llvm::Value* loaded_value;
    };

    std::vector<DelayedExtraction> delay_extraction_args;
    std::vector<llvm::Value*> call_args;
    if (!format_sym.is_constant()) {
        rop.shadingcontext()->warningfmt(
            "{} must currently have constant format\n", op.opname());
        return false;
    }

    ustring format_ustring = format_sym.get_string();
    const char* format     = format_ustring.c_str();
    std::string s;
    int arg = format_arg + 1;

    // Check all arguments to see if we will need to generate
    // a separate printf call for each data lane or not
    // consider the op to be uniform until we find an argument that isn't
    bool op_is_uniform = true;
    for (int a = arg; a < op.nargs(); ++a) {
        Symbol& sym(*rop.opargsym(op, a));
        bool arg_is_uniform = sym.is_uniform();
        if (arg_is_uniform == false) {
            op_is_uniform = false;
        }
    }

    int mask_slot = -1;
    // For some ops, we push the shader globals pointer
    if (op.opname() == op_printf || op.opname() == op_error
        || op.opname() == op_warning || op.opname() == op_fprintf) {
        auto sg = rop.sg_void_ptr();
        call_args.push_back(sg);
        // Add mask or placeholder
        mask_slot = call_args.size();
        call_args.push_back(op_is_uniform
                                ? rop.ll.mask_as_int(rop.ll.current_mask())
                                : nullptr);
    }

    if (op.opname() == op_fprintf) {
        Symbol& Filename = *rop.opargsym(op, 0);
        llvm::Value* fn  = rop.llvm_load_value(Filename);
        call_args.push_back(fn);
    }

    // For some ops, we push the output symbol & mask
    if ((op.opname() == op_format) && (false == op_is_uniform)) {
        Symbol& outSymbol = *rop.opargsym(op, 0);

        llvm::Value* outPtr = rop.llvm_void_ptr(outSymbol);
        call_args.push_back(outPtr);
        // Add placeholder for mask
        mask_slot = call_args.size();
        call_args.push_back(nullptr);
    }

    // We're going to need to adjust the format string as we go, but I'd
    // like to reserve a spot for the char*.
    size_t new_format_slot = call_args.size();
    call_args.push_back(NULL);

    while (*format != '\0') {
        if (*format == '%') {
            if (format[1] == '%') {
                // '%%' is a literal '%'
                s += "%%";
                format += 2;  // skip both percentages
                continue;
            }
            const char* oldfmt = format;  // mark beginning of format
            while (*format && *format != 'c' && *format != 'd' && *format != 'e'
                   && *format != 'f' && *format != 'g' && *format != 'i'
                   && *format != 'm' && *format != 'n' && *format != 'o'
                   && *format != 'p' && *format != 's' && *format != 'u'
                   && *format != 'v' && *format != 'x' && *format != 'X')
                ++format;
            char formatchar = *format++;  // Also eat the format char
            if (arg >= op.nargs()) {
                rop.shadingcontext()->errorfmt(
                    "Mismatch between format string and arguments ({}:{})",
                    op.sourcefile(), op.sourceline());
                return false;
            }

            std::string ourformat(oldfmt, format);  // straddle the format
            // Doctor it to fix mismatches between format and data
            Symbol& sym(*rop.opargsym(op, arg));
            OSL_ASSERT(!sym.typespec().is_structure_based());

            bool arg_is_uniform = sym.is_uniform();

            TypeDesc simpletype(sym.typespec().simpletype());
            int num_elements   = simpletype.numelements();
            int num_components = simpletype.aggregate;
            if ((sym.typespec().is_closure_based()
                 || simpletype.basetype == TypeDesc::STRING)
                && formatchar != 's') {
                ourformat[ourformat.length() - 1] = 's';
            }
            if (simpletype.basetype == TypeDesc::INT && formatchar != 'd'
                && formatchar != 'i' && formatchar != 'o' && formatchar != 'u'
                && formatchar != 'x' && formatchar != 'X') {
                ourformat[ourformat.length() - 1] = 'd';
            }
            if (simpletype.basetype == TypeDesc::FLOAT && formatchar != 'f'
                && formatchar != 'g' && formatchar != 'c' && formatchar != 'e'
                && formatchar != 'm' && formatchar != 'n' && formatchar != 'p'
                && formatchar != 'v') {
                ourformat[ourformat.length() - 1] = 'f';
            }
            // NOTE(boulos): Only for debug mode do the derivatives get printed...
            for (int a = 0; a < num_elements; ++a) {
                llvm::Value* arrind = simpletype.arraylen ? rop.ll.constant(a)
                                                          : NULL;

                for (int c = 0; c < num_components; c++) {
                    if (c != 0 || a != 0)
                        s += " ";
                    s += ourformat;

                    // As the final printf library call does not handle wide
                    // data types, we will load the wide data type here and
                    // in a loop extract scalar values for the current data
                    // lane before making the scalar printf call.
                    // NOTE:  We don't want any uniform arguments to be
                    // widened, so our typical op_is_uniform doesn't do what we
                    // want for this when loading.  So just pass arg_is_uniform
                    // which will avoid widening any uniform arguments.
                    llvm::Value* loaded
                        = rop.llvm_load_value(sym, 0, arrind, c,
                                              TypeDesc::UNKNOWN,
                                              /*op_is_uniform*/ arg_is_uniform,
                                              /*index_is_uniform*/ true);

                    // Always expand llvm booleans to integers
                    if (sym.forced_llvm_bool()) {
                        loaded = rop.ll.op_bool_to_int(loaded);
                    }

                    if (arg_is_uniform) {
                        if (simpletype.basetype == TypeDesc::FLOAT) {
                            // C varargs convention upconverts float->double.
                            loaded = rop.ll.op_float_to_double(loaded);
                        } else if (sym.typespec().is_closure_based()) {
                            // we need to convert this closure to a string
                            FuncSpec to_string("closure_to_string");
                            loaded = rop.ll.call_function(rop.build_name(
                                                              to_string),
                                                          rop.sg_void_ptr(),
                                                          loaded);
                        }
                        call_args.push_back(loaded);
                    } else {
                        OSL_ASSERT(false == op_is_uniform);
                        delay_extraction_args.push_back(DelayedExtraction {
                            static_cast<int>(call_args.size()),
                            simpletype.basetype == TypeDesc::FLOAT,
                            sym.typespec().is_closure_based(), loaded });
                        // Need to populate call arguments with a place holder
                        // that we can fill in later from a loop that loads values
                        // for each lane
                        call_args.push_back(nullptr);
                    }
                }
            }
            ++arg;
        } else {
            // Everything else -- just copy the character and advance
            s += *format++;
        }
    }

    // Some ops prepend things
    if (op.opname() == op_error || op.opname() == op_warning) {
        s = fmtformat("Shader {} [{}]: {}", op.opname(),
                      rop.inst()->shadername(), s);
    }

    // Now go back and put the new format string in its place
    auto llvm_new_format_string = rop.ll.constant(s.c_str());
    call_args[new_format_slot]  = llvm_new_format_string;

    // Construct the function name and call it.
    FuncSpec func_spec(op.opname().c_str());

    if ((op.opname() == op_format) && op_is_uniform) {
        func_spec.unbatch();
    }

    if (op_is_uniform) {
        llvm::Value* ret = rop.ll.call_function(rop.build_name(func_spec),
                                                call_args);

        // The format op returns a string value, put in in the right spot
        if (op.opname() == op_format)
            rop.llvm_store_value(ret, *rop.opargsym(op, 0));
    } else {
        // Loop over each lane, if mask is active for the lane,
        // extract values and call printf
        llvm::Value* loc_of_lane_index
            = rop.ll.op_alloca(rop.ll.type_int(), 1,
                               rop.llvm_debug() ? std::string("printf index")
                                                : std::string());
        rop.ll.op_unmasked_store(rop.ll.constant(0), loc_of_lane_index);
        llvm::Value* mask = rop.ll.current_mask();
#ifdef __OSL_TRACE_MASKS
        rop.llvm_print_mask("printf loop over:", mask);
#endif

        llvm::BasicBlock* cond_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("printf_cond") : std::string());
        llvm::BasicBlock* step_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("printf_step") : std::string());
        llvm::BasicBlock* body_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("printf_body") : std::string());
        llvm::BasicBlock* nested_body_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("printf_nested_body")
                             : std::string());
        llvm::BasicBlock* after_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("after_printf") : std::string());
        rop.ll.op_branch(cond_block);
        {
            // Condition
            llvm::Value* lane_index = rop.ll.op_load(rop.ll.type_int(),
                                                     loc_of_lane_index);
            llvm::Value* more_lanes_to_process
                = rop.ll.op_lt(lane_index, rop.ll.constant(rop.vector_width()));

            rop.ll.op_branch(more_lanes_to_process, body_block, after_block);

            // body_block
            // do printf of a single lane
            llvm::Value* lane_active = rop.ll.test_mask_lane(mask, lane_index);
            rop.ll.op_branch(lane_active, nested_body_block, step_block);

            // nested_body_block
            llvm::Value* int_value_lane_mask = rop.ll.op_shl(rop.ll.constant(1),
                                                             lane_index);
            call_args[mask_slot]             = int_value_lane_mask;
            for (const DelayedExtraction& de : delay_extraction_args) {
                llvm::Value* scalar_val = rop.ll.op_extract(de.loaded_value,
                                                            lane_index);

                if (de.is_float) {
                    // C varargs convention upconverts float->double.
                    scalar_val = rop.ll.op_float_to_double(scalar_val);
                } else if (de.is_closure) {
                    // we need to convert this closure to a string
                    FuncSpec to_string("closure_to_string");
                    scalar_val = rop.ll.call_function(rop.build_name(to_string),
                                                      rop.sg_void_ptr(),
                                                      scalar_val);
                }
                call_args[de.argument_slot] = scalar_val;
            }
            rop.ll.call_function(rop.build_name(func_spec), call_args);

            rop.ll.op_branch(step_block);

            // Step
            //lane_index = rop.ll.op_load(rop.ll.type_int(), loc_of_lane_index);
            llvm::Value* next_lane_index = rop.ll.op_add(lane_index,
                                                         rop.ll.constant(1));
            rop.ll.op_unmasked_store(next_lane_index, loc_of_lane_index);
            rop.ll.op_branch(cond_block);

            // Continue on with the previous flow
            rop.ll.set_insert_point(after_block);
        }
    }

    return true;
}



// Array length
LLVMGEN(llvm_gen_arraylength)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& A      = *rop.opargsym(op, 1);
    OSL_DASSERT(Result.typespec().is_int() && A.typespec().is_array());

    int len = A.typespec().is_unsized_array() ? A.initializers()
                                              : A.typespec().arraylength();

    // Even thought the array's size should be uniform across all lanes,
    // the symbol its being stored in maybe varying
    llvm::Value* const_length = Result.is_uniform() ? rop.ll.constant(len)
                                                    : rop.ll.wide_constant(len);
    rop.llvm_store_value(const_length, Result);
    return true;
}



// Array reference
LLVMGEN(llvm_gen_aref)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& Src    = *rop.opargsym(op, 1);
    Symbol& Index  = *rop.opargsym(op, 2);

    bool op_is_uniform    = Result.is_uniform();
    bool index_is_uniform = Index.is_uniform();

    // Get array index we're interested in
    llvm::Value* index = rop.loadLLVMValue(Index, 0, 0, TypeInt,
                                           index_is_uniform);
    if (!index)
        return false;

    if (rop.inst()->master()->range_checking()) {
        if (index_is_uniform) {
            if (!(Index.is_constant() && Index.get_int() >= 0
                  && Index.get_int() < Src.typespec().arraylength())) {
                llvm::Value* args[]
                    = { index,
                        rop.ll.constant(Src.typespec().arraylength()),
                        rop.ll.constant(Src.unmangled()),
                        rop.sg_void_ptr(),
                        rop.ll.constant(op.sourcefile()),
                        rop.ll.constant(op.sourceline()),
                        rop.ll.constant(rop.group().name()),
                        rop.ll.constant(rop.layer()),
                        rop.ll.constant(rop.inst()->layername()),
                        rop.ll.constant(rop.inst()->shadername()) };
                index = rop.ll.call_function(rop.build_name("range_check"),
                                             args);
            }
        } else {
            BatchedBackendLLVM::TempScope temp_scope(rop);

            // We need a copy of the indices in case the range check clamps them
            llvm::Value* loc_clamped_wide_index = rop.getOrAllocateTemp(
                TypeSpec(TypeDesc::INT), false /*derivs*/, false /*is_uniform*/,
                false /*forceBool*/,
                std::string("range clamped index:") + Src.name().c_str());
            // copy the indices into our temporary
            rop.ll.op_unmasked_store(index, loc_clamped_wide_index);

            llvm::Value* args[]
                = { rop.ll.void_ptr(loc_clamped_wide_index),
                    rop.ll.mask_as_int(rop.ll.current_mask()),
                    rop.ll.constant(Src.typespec().arraylength()),
                    rop.ll.constant(Src.unmangled()),
                    rop.sg_void_ptr(),
                    rop.ll.constant(op.sourcefile()),
                    rop.ll.constant(op.sourceline()),
                    rop.ll.constant(rop.group().name()),
                    rop.ll.constant(rop.layer()),
                    rop.ll.constant(rop.inst()->layername()),
                    rop.ll.constant(rop.inst()->shadername()) };
            rop.ll.call_function(rop.build_name(FuncSpec("range_check").mask()),
                                 args);
            // Use the range check indices
            index = rop.ll.op_load(rop.ll.type_wide_int(),
                                   loc_clamped_wide_index);
        }
    }

    int num_components = Src.typespec().simpletype().aggregate;
    for (int d = 0; d <= 2; ++d) {
        for (int c = 0; c < num_components; ++c) {
            llvm::Value* val
                = rop.llvm_load_value(Src, d, index, c, TypeDesc::UNKNOWN,
                                      op_is_uniform, index_is_uniform);
            rop.storeLLVMValue(val, Result, c, d);
        }
        if (!Result.has_derivs())
            break;
    }

    return true;
}


// Array assignment
LLVMGEN(llvm_gen_aassign)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& Index  = *rop.opargsym(op, 1);
    Symbol& Src    = *rop.opargsym(op, 2);

    bool resultIsUniform  = Result.is_uniform();
    bool index_is_uniform = Index.is_uniform();
    OSL_ASSERT(index_is_uniform || !resultIsUniform);

    // Get array index we're interested in
    llvm::Value* index = rop.loadLLVMValue(Index, 0, 0, TypeInt,
                                           index_is_uniform);

    if (!index)
        return false;

    if (rop.inst()->master()->range_checking()) {
        if (index_is_uniform) {
            if (!(Index.is_constant() && Index.get_int() >= 0
                  && Index.get_int() < Result.typespec().arraylength())) {
                llvm::Value* args[]
                    = { index,
                        rop.ll.constant(Result.typespec().arraylength()),
                        rop.ll.constant(Result.unmangled()),
                        rop.sg_void_ptr(),
                        rop.ll.constant(op.sourcefile()),
                        rop.ll.constant(op.sourceline()),
                        rop.ll.constant(rop.group().name()),
                        rop.ll.constant(rop.layer()),
                        rop.ll.constant(rop.inst()->layername()),
                        rop.ll.constant(rop.inst()->shadername()) };
                index = rop.ll.call_function(rop.build_name("range_check"),
                                             args);
            }
        } else {
            BatchedBackendLLVM::TempScope temp_scope(rop);
            // We need a copy of the indices in case the range check clamps them
            llvm::Value* loc_clamped_wide_index = rop.getOrAllocateTemp(
                TypeSpec(TypeDesc::INT), false /*derivs*/, false /*is_uniform*/,
                false /*forceBool*/,
                std::string("range clamped index:") + Result.name().c_str());
            // copy the indices into our temporary
            rop.ll.op_unmasked_store(index, loc_clamped_wide_index);

            llvm::Value* args[]
                = { rop.ll.void_ptr(loc_clamped_wide_index),
                    rop.ll.mask_as_int(rop.ll.current_mask()),
                    rop.ll.constant(Result.typespec().arraylength()),
                    rop.ll.constant(Result.unmangled()),
                    rop.sg_void_ptr(),
                    rop.ll.constant(op.sourcefile()),
                    rop.ll.constant(op.sourceline()),
                    rop.ll.constant(rop.group().name()),
                    rop.ll.constant(rop.layer()),
                    rop.ll.constant(rop.inst()->layername()),
                    rop.ll.constant(rop.inst()->shadername()) };
            rop.ll.call_function(rop.build_name(FuncSpec("range_check").mask()),
                                 args);
            // Use the range check indices
            index = rop.ll.op_load(rop.ll.type_wide_int(),
                                   loc_clamped_wide_index);
        }
    }

    int num_components = Result.typespec().simpletype().aggregate;

    // Allow float <=> int casting
    TypeDesc cast;  // defaults to TypeDesc::UNKNOWN
    if (num_components == 1 && !Result.typespec().is_closure()
        && !Src.typespec().is_closure()
        && (Result.typespec().is_int_based()
            || Result.typespec().is_float_based())
        && (Src.typespec().is_int_based() || Src.typespec().is_float_based())) {
        cast          = Result.typespec().simpletype();
        cast.arraylen = 0;
    } else {
        // Try to warn before llvm_fatal_error is called which provides little
        // context as to what went wrong.
        OSL_ASSERT(Result.typespec().simpletype().basetype
                   == Src.typespec().simpletype().basetype);
    }

    for (int d = 0; d <= 2; ++d) {
        for (int c = 0; c < num_components; ++c) {
            llvm::Value* val = rop.loadLLVMValue(Src, c, d, cast,
                                                 resultIsUniform);

            // Bool is not a supported OSL type, so if we find one it needs
            // to be promoted to an int
            llvm::Type* typeOfVal = rop.ll.llvm_typeof(val);
            if (typeOfVal == rop.ll.type_bool()
                || typeOfVal == rop.ll.type_wide_bool()) {
                val = rop.ll.op_bool_to_int(val);
            }

            rop.llvm_store_value(val, Result, d, index, c, index_is_uniform);
        }
        if (!Result.has_derivs())
            break;
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
LLVMGEN(llvm_gen_generic)
{
    Opcode& op(rop.inst()->ops()[opnum]);

    bool uniformFormOfFunction = true;
    for (int i = 0; i < op.nargs(); ++i) {
        Symbol* s(rop.opargsym(op, i));
        if (s->is_uniform() == false) {
            uniformFormOfFunction = false;
        }
    }

    Symbol& Result = *rop.opargsym(op, 0);

    std::vector<const Symbol*> args;
    bool any_deriv_args = false;
    for (int i = 0; i < op.nargs(); ++i) {
        Symbol* s(rop.opargsym(op, i));
        args.push_back(s);
        any_deriv_args |= (i > 0 && s->has_derivs()
                           && !s->typespec().is_matrix());
    }

    // Special cases: functions that have no derivs -- suppress them
    if (any_deriv_args)
        if (op.opname() == op_logb || op.opname() == op_floor
            || op.opname() == op_ceil || op.opname() == op_round
            || op.opname() == op_step || op.opname() == op_trunc
            || op.opname() == op_sign)
            any_deriv_args = false;

    FuncSpec func_spec(op.opname().c_str());
    if (uniformFormOfFunction) {
        func_spec.unbatch();
    }

    for (int i = 0; i < op.nargs(); ++i) {
        Symbol* s(rop.opargsym(op, i));
        bool has_derivs = any_deriv_args && Result.has_derivs()
                          && s->has_derivs() && !s->typespec().is_matrix();
        func_spec.arg(*s, has_derivs, uniformFormOfFunction);
    }

    OSL_DEV_ONLY(std::cout << "llvm_gen_generic " << rop.build_name(func_spec)
                           << std::endl);

    if (!Result.has_derivs() || !any_deriv_args) {
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
        if (Result.typespec().aggregate() == TypeDesc::SCALAR
            && (uniformFormOfFunction || functionIsLlvmInlined)) {
            OSL_DEV_ONLY(std::cout << ">>stores return value "
                                   << rop.build_name(func_spec) << std::endl);
            llvm::Value* r = rop.llvm_call_function(
                func_spec, &(args[1]), op.nargs() - 1,
                /*deriv_ptrs*/ false, uniformFormOfFunction,
                functionIsLlvmInlined, false /*ptrToReturnStructIs1stArg*/);
            // The store will deal with masking
            rop.llvm_store_value(r, Result);
        } else {
            OSL_DEV_ONLY(std::cout << ">>return value is pointer "
                                   << rop.build_name(func_spec) << std::endl);

            rop.llvm_call_function(func_spec, (args.size()) ? &(args[0]) : NULL,
                                   op.nargs(),
                                   /*deriv_ptrs*/ false, uniformFormOfFunction,
                                   functionIsLlvmInlined,
                                   true /*ptrToReturnStructIs1stArg*/);
        }
        rop.llvm_zero_derivs(Result);
    } else {
        // Cases with derivs
        OSL_DEV_ONLY(std::cout << " Cases with derivs");
        OSL_ASSERT(Result.has_derivs() && any_deriv_args);
        rop.llvm_call_function(func_spec, (args.size()) ? &(args[0]) : NULL,
                               op.nargs(),
                               /*deriv_ptrs*/ true, uniformFormOfFunction,
                               false /*functionIsLlvmInlined*/,
                               true /*ptrToReturnStructIs1stArg*/);
    }

    OSL_DEV_ONLY(std::cout << std::endl);

    return true;
}


LLVMGEN(llvm_gen_sincos)
{
    Opcode& op(rop.inst()->ops()[opnum]);

    Symbol& Theta   = *rop.opargsym(op, 0);  //Input
    Symbol& Sin_out = *rop.opargsym(op, 1);  //Output
    Symbol& Cos_out = *rop.opargsym(op, 2);  //Output

    bool theta_deriv   = Theta.has_derivs();
    bool result_derivs = (Sin_out.has_derivs() || Cos_out.has_derivs());

    bool op_is_uniform = Theta.is_uniform();

    OSL_ASSERT(op_is_uniform
               || (!Sin_out.is_uniform() && !Cos_out.is_uniform()));

    // Handle broadcasting results to wide results
    BatchedBackendLLVM::TempScope temp_scope(rop);

    llvm::Value* theta_param = nullptr;
    // Need 2 pointers, because the parameter must be void *
    // but we need a typed * for the broadcast later
    llvm::Value* sin_out_typed_temp = nullptr;
    llvm::Value* sin_out_param      = nullptr;

    llvm::Value* cos_out_typed_temp = nullptr;
    llvm::Value* cos_out_param      = nullptr;

    if (true
        == ((theta_deriv && result_derivs) || Theta.typespec().is_triple()
            || !op_is_uniform)) {
        theta_param = rop.llvm_void_ptr(Theta, 0);  //If varying
    } else {
        theta_param = rop.llvm_load_value(Theta);
    }

    //rop.llvm_store_value(wideTheta, Theta);
    FuncSpec func_spec("sincos");

    func_spec.arg(Theta, result_derivs && theta_deriv, op_is_uniform);
    func_spec.arg(Sin_out, Sin_out.has_derivs() && result_derivs && theta_deriv,
                  op_is_uniform);
    func_spec.arg(Cos_out, Cos_out.has_derivs() && result_derivs && theta_deriv,
                  op_is_uniform);


    if (op_is_uniform && !Sin_out.is_uniform()) {
        sin_out_typed_temp = rop.getOrAllocateTemp(Sin_out.typespec(),
                                                   Sin_out.has_derivs(),
                                                   true /*is_uniform*/);
        sin_out_param      = rop.ll.void_ptr(sin_out_typed_temp);
    } else {
        sin_out_param = rop.llvm_void_ptr(Sin_out, 0);
    }

    if (op_is_uniform && !Cos_out.is_uniform()) {
        cos_out_typed_temp = rop.getOrAllocateTemp(Cos_out.typespec(),
                                                   Cos_out.has_derivs(),
                                                   true /*is_uniform*/);
        cos_out_param      = rop.ll.void_ptr(cos_out_typed_temp);
    } else {
        cos_out_param = rop.llvm_void_ptr(Cos_out, 0);
    }

    llvm::Value* args[] = { theta_param, sin_out_param, cos_out_param,
                            nullptr };
    int arg_count       = 3;

    if (!op_is_uniform) {
        if (rop.ll.is_masking_required()) {
            func_spec.mask();
            args[arg_count++] = rop.ll.mask_as_int(rop.ll.current_mask());
        }
    } else {
        func_spec.unbatch();
    }

    rop.ll.call_function(rop.build_name(func_spec),
                         cspan<llvm::Value*>(args, arg_count));

    if (op_is_uniform && !Sin_out.is_uniform()) {
        rop.llvm_broadcast_uniform_value_from_mem(sin_out_typed_temp, Sin_out);
    }

    if (op_is_uniform && !Cos_out.is_uniform()) {
        rop.llvm_broadcast_uniform_value_from_mem(cos_out_typed_temp, Cos_out);
    }

    // If the input angle didn't have derivatives, we would not have
    // called the version of sincos with derivs; however in that case we
    // need to clear the derivs of either of the outputs that has them.
    if (Sin_out.has_derivs() && !theta_deriv)
        rop.llvm_zero_derivs(Sin_out);
    if (Cos_out.has_derivs() && !theta_deriv)
        rop.llvm_zero_derivs(Cos_out);

    return true;
}



LLVMGEN(llvm_gen_andor)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& result = *rop.opargsym(op, 0);
    Symbol& a      = *rop.opargsym(op, 1);
    Symbol& b      = *rop.opargsym(op, 2);

    bool op_is_uniform     = a.is_uniform() && b.is_uniform();
    bool result_is_uniform = result.is_uniform();


    llvm::Value* i1_res = NULL;
    llvm::Value* a_val  = rop.llvm_load_value(a, 0, 0, TypeInt, op_is_uniform);
    llvm::Value* b_val  = rop.llvm_load_value(b, 0, 0, TypeInt, op_is_uniform);
    llvm::Value* zero_set = op_is_uniform ? rop.ll.constant(0)
                                          : rop.ll.wide_constant(0);

    if (op.opname() == op_and) {
        // From the old bitcode generated
        // define i32 @osl_and_iii(i32 %a, i32 %b) nounwind readnone ssp {
        //     %1 = icmp ne i32 %b, 0
        //  %not. = icmp ne i32 %a, 0
        //     %2 = and i1 %1, %not.
        //     %3 = zext i1 %2 to i32
        //   ret i32 %3


        llvm::Value* b_ne_0    = rop.ll.op_ne(b_val, zero_set);
        llvm::Value* a_ne_0    = rop.ll.op_ne(a_val, zero_set);
        llvm::Value* both_ne_0 = rop.ll.op_and(b_ne_0, a_ne_0);
        i1_res                 = both_ne_0;
    } else {
        OSL_DASSERT(op.opname() == ustring("or"));
        // Also from the bitcode
        // %1 = or i32 %b, %a
        // %2 = icmp ne i32 %1, 0
        // %3 = zext i1 %2 to i32
        llvm::Value* or_ab      = rop.ll.op_or(a_val, b_val);
        llvm::Value* or_ab_ne_0 = rop.ll.op_ne(or_ab, zero_set);
        i1_res                  = or_ab_ne_0;
    }

    if (op_is_uniform && !result_is_uniform) {
        i1_res = rop.ll.widen_value(i1_res);
    }

    // Although we try to use llvm bool (i1) for comparison results
    // sometimes we could not force the data type to be an bool and it remains
    // an int, for those cases we will need to convert the boolean to int
    if (!result.forced_llvm_bool()) {
#ifndef OSL_LLVM_OPAQUE_POINTERS
        llvm::Type* resultType = rop.ll.llvm_typeof(
            rop.llvm_get_pointer(result));
        OSL_ASSERT(
            (resultType
             == reinterpret_cast<llvm::Type*>(rop.ll.type_wide_int_ptr()))
            || (resultType
                == reinterpret_cast<llvm::Type*>(rop.ll.type_int_ptr())));
        llvm::Type* typeOfR = rop.ll.llvm_typeof(i1_res);
        OSL_ASSERT(typeOfR == rop.ll.type_wide_bool()
                   || typeOfR == rop.ll.type_bool());
#endif
        i1_res = rop.ll.op_bool_to_int(i1_res);
    }

    rop.llvm_store_value(i1_res, result, 0, 0);
    return true;
}



LLVMGEN(llvm_gen_if)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& cond = *rop.opargsym(op, 0);

    const char* cond_name = cond.name().c_str();
    bool op_is_uniform    = cond.is_uniform();

    bool elseBlockRequired = op.jump(0) != op.jump(1);

    int beforeThenElseReturnCount   = rop.ll.masked_return_count();
    int beforeThenElseBreakCount    = rop.ll.masked_break_count();
    int beforeThenElseContinueCount = rop.ll.masked_continue_count();

    if (op_is_uniform) {
        // Load the condition variable and figure out if it's nonzero
        llvm::Value* cond_val = rop.llvm_test_nonzero(cond);

        // Branch on the condition, to our blocks
        llvm::BasicBlock* then_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("then (uniform)") + cond_name
                             : std::string());
        llvm::BasicBlock* else_block
            = elseBlockRequired ? rop.ll.new_basic_block(
                  rop.llvm_debug() ? std::string("else (uniform)") + cond_name
                                   : std::string())
                                : nullptr;
        llvm::BasicBlock* after_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("after_if (uniform)") + cond_name
                             : std::string());
        rop.ll.op_branch(cond_val, then_block,
                         elseBlockRequired ? else_block : after_block);

        // Then block
        rop.build_llvm_code(opnum + 1, op.jump(0), then_block);
        rop.ll.op_branch(after_block);  // insert point is now after_block
        if (elseBlockRequired) {
            // Else block
            rop.build_llvm_code(op.jump(0), op.jump(1), else_block);
            rop.ll.op_branch(after_block);  // insert point is now after_block
        }

        // NOTE: if a return or exit is encounter inside a uniform
        // conditional block, then it will branch to the last
        // rop.ll.push_masked_return_block(...)
        // or if there is none, operate in a scalar fashion
        // branching to the return_block() or exit_instance()
    } else {
        llvm::Value* mask = rop.llvm_load_mask(cond);
        OSL_ASSERT(mask->getType() == rop.ll.type_wide_bool());
#ifdef __OSL_TRACE_MASKS
        rop.llvm_print_mask("if(cond)", mask);
#endif
        rop.ll.push_mask(mask);
#ifdef __OSL_TRACE_MASKS
        rop.llvm_print_mask("if STACK");
#endif

        // TODO:  Add heuristic to control if we can avoid testing
        // for any lanes active and just execute masked.
        // However must make sure the then or else block does not
        // contain a call to a lower level, those must not be executed
        // if the mask is all off
//#define SKIP_TESTING_IF_ANY_LANES_ACTIVE_TO_AVOID_BRANCH 1
// NOTE: if SKIP_TESTING_IF_ANY_LANES_ACTIVE_TO_AVOID_BRANCH is defined then
// library and layer functions may get called with an entirely empty mask
#ifndef SKIP_TESTING_IF_ANY_LANES_ACTIVE_TO_AVOID_BRANCH
        // We use the combined mask stack + the if condition's mask we already pushed
        llvm::Value* anyThenLanesActive = rop.ll.test_if_mask_is_non_zero(
            rop.ll.current_mask());
#endif
        // Branch on the condition, to our blocks
        llvm::BasicBlock* then_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("then (varying)") + cond_name
                             : std::string());

        llvm::BasicBlock* test_else_block
            = elseBlockRequired ? rop.ll.new_basic_block(
                  rop.llvm_debug()
                      ? std::string("test_else (varying)") + cond_name
                      : std::string())
                                : nullptr;
        llvm::BasicBlock* else_block
            = elseBlockRequired ? rop.ll.new_basic_block(
                  rop.llvm_debug() ? std::string("else (varying)") + cond_name
                                   : std::string())
                                : nullptr;

        llvm::BasicBlock* after_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("after_if (varying)") + cond_name
                             : std::string());

        // Then block
        // Perhaps mask should be parameter to build_llvm_code?
#ifndef SKIP_TESTING_IF_ANY_LANES_ACTIVE_TO_AVOID_BRANCH
        rop.ll.op_branch(anyThenLanesActive, then_block,
                         elseBlockRequired ? test_else_block : after_block);
#else
        rop.ll.op_branch(then_block);
#endif

        rop.ll.set_insert_point(then_block);
        //rop.ll.push_mask(mask); // we pushed this mask before the then block so we can test for 0 active lanes
        rop.ll.push_masked_return_block(elseBlockRequired ? test_else_block
                                                          : after_block);
#ifdef __OSL_TRACE_MASKS
        rop.llvm_print_mask("then");
#endif
        rop.build_llvm_code(opnum + 1, op.jump(0), then_block);
        rop.ll.pop_masked_return_block();
        rop.ll.pop_mask();
        // Execute both the "then" and the "else" blocks with masking
        rop.ll.op_branch(elseBlockRequired ? test_else_block : after_block);
        if (elseBlockRequired) {
            // Else block
            // insertion point should be test_else_block
            rop.ll.push_mask(mask, true /* negate */);
            llvm::Value* anyElseLanesActive = rop.ll.test_if_mask_is_non_zero(
                rop.ll.current_mask());

            rop.ll.op_branch(anyElseLanesActive, else_block, after_block);
            rop.ll.set_insert_point(else_block);
            rop.ll.push_masked_return_block(after_block);
#ifdef __OSL_TRACE_MASKS
            rop.llvm_print_mask("else");
#endif
            rop.build_llvm_code(op.jump(0), op.jump(1), else_block);
            rop.ll.pop_masked_return_block();
            rop.ll.pop_mask();
            rop.ll.op_branch(after_block);
        }
    }

    bool requiresTestForActiveLanes = false;
    if (rop.ll.masked_continue_count() > beforeThenElseContinueCount) {
        // Inside the 'then' or 'else' blocks a continue may have been executed
        // we need to update the current mask to reflect the disabled lanes
        // We needed to wait until were were in the after block so the produced
        // mask is available to subsequent instructions
        rop.ll.apply_continue_to_mask_stack();
        requiresTestForActiveLanes = true;
#ifdef __OSL_TRACE_MASKS
        rop.llvm_print_mask("continue applied");
#endif
    }
    if (rop.ll.masked_break_count() > beforeThenElseBreakCount) {
        // Inside the 'then' or 'else' blocks a return may have been executed
        // we need to update the current mask to reflect the disabled lanes
        // We needed to wait until were were in the after block so the produced
        // mask is available to subsequent instructions
        rop.ll.apply_break_to_mask_stack();
        requiresTestForActiveLanes = true;
#ifdef __OSL_TRACE_MASKS
        rop.llvm_print_mask("break applied");
#endif
    }
    if (rop.ll.masked_return_count() > beforeThenElseReturnCount) {
        // Inside the 'then' or 'else' blocks a return may have been executed
        // we need to update the current mask to reflect the disabled lanes
        // We needed to wait until were were in the after block so the produced
        // mask is available to subsequent instructions
        rop.ll.apply_return_to_mask_stack();
        requiresTestForActiveLanes = true;
#ifdef __OSL_TRACE_MASKS
        rop.llvm_print_mask("return applied");
#endif
    }
    if (requiresTestForActiveLanes) {
        // through a combination of the break or return mask and any lanes conditionally
        // masked off, all lanes could be 0 at this point and we wouldn't
        // want to call down to any layers at this point

        // NOTE: testing the return/exit masks themselves is not sufficient
        // as some lanes may be disabled by the conditional mask stack

        // TODO: do we want a test routine that can handle negated masks?
        llvm::Value* anyLanesActive = rop.ll.test_if_mask_is_non_zero(
            rop.ll.current_mask());

        llvm::BasicBlock* nextMaskScope;
        if (rop.ll.has_masked_return_block()) {
            nextMaskScope = rop.ll.masked_return_block();
        } else {
            nextMaskScope = rop.ll.inside_function()
                                ? rop.ll.return_block()
                                : rop.llvm_exit_instance_block();
        }
        llvm::BasicBlock* after_applying_return_block = rop.ll.new_basic_block(
            rop.llvm_debug()
                ? std::string("after_if_applied_return_mask (varying)")
                      + cond_name
                : std::string());

        rop.ll.op_branch(anyLanesActive, after_applying_return_block,
                         nextMaskScope);
    }

    // Continue on with the previous flow
    return true;
}



LLVMGEN(llvm_gen_add)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& A      = *rop.opargsym(op, 1);
    Symbol& B      = *rop.opargsym(op, 2);

    bool op_is_uniform     = A.is_uniform() && B.is_uniform();
    bool result_is_uniform = Result.is_uniform();
    OSL_ASSERT(op_is_uniform || !result_is_uniform);

    OSL_ASSERT(!A.typespec().is_array() && !B.typespec().is_array());
    if (Result.typespec().is_closure()) {
        OSL_ASSERT(A.typespec().is_closure() && B.typespec().is_closure());
        OSL_ASSERT(!A.is_uniform() && !B.is_uniform() && !result_is_uniform);
        llvm::Value* valargs[] = { rop.sg_void_ptr(), rop.llvm_void_ptr(Result),
                                   rop.llvm_void_ptr(A), rop.llvm_void_ptr(B),
                                   rop.ll.mask_as_int(rop.ll.current_mask()) };

        FuncSpec add_closure("add_closure_closure");
        add_closure.mask();
        // we directly write the result in add_closure_closure instead of using a
        // store to write it back to result as in the single-point path
        rop.ll.call_function(rop.build_name(add_closure), valargs);

        return true;
    }

    TypeDesc type      = Result.typespec().simpletype();
    int num_components = type.aggregate;

    // The following should handle f+f, v+v, v+f, f+v, i+i
    // That's all that should be allowed by oslc.
    for (int i = 0; i < num_components; i++) {
        OSL_DEV_ONLY(std::cout << "llvm_gen_add component(" << i << ") of "
                               << A.name() << " " << B.name() << std::endl);
        llvm::Value* a = rop.loadLLVMValue(A, i, 0, type, op_is_uniform);
        llvm::Value* b = rop.loadLLVMValue(B, i, 0, type, op_is_uniform);
        if (!a || !b)
            return false;
        llvm::Value* r = rop.ll.op_add(a, b);
        if (op_is_uniform && !result_is_uniform) {
            r = rop.ll.widen_value(r);
        }
        rop.storeLLVMValue(r, Result, i, 0);
    }

    if (Result.has_derivs()) {
        if (A.has_derivs() || B.has_derivs()) {
            for (int d = 1; d <= 2; ++d) {  // dx, dy
                for (int i = 0; i < num_components; i++) {
                    llvm::Value* a = rop.loadLLVMValue(A, i, d, type,
                                                       op_is_uniform);
                    llvm::Value* b = rop.loadLLVMValue(B, i, d, type,
                                                       op_is_uniform);
                    llvm::Value* r = rop.ll.op_add(a, b);
                    if (op_is_uniform && !result_is_uniform) {
                        r = rop.ll.widen_value(r);
                    }
                    rop.storeLLVMValue(r, Result, i, d);
                }
            }
        } else {
            // Result has derivs, operands do not
            rop.llvm_zero_derivs(Result);
        }
    }
    return true;
}



LLVMGEN(llvm_gen_sub)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& A      = *rop.opargsym(op, 1);
    Symbol& B      = *rop.opargsym(op, 2);

    bool op_is_uniform     = A.is_uniform() && B.is_uniform();
    bool result_is_uniform = Result.is_uniform();
    OSL_ASSERT(op_is_uniform || !result_is_uniform);

    TypeDesc type      = Result.typespec().simpletype();
    int num_components = type.aggregate;

    OSL_ASSERT(!Result.typespec().is_closure_based()
               && "subtraction of closures not supported");

    // The following should handle f-f, v-v, v-f, f-v, i-i
    // That's all that should be allowed by oslc.
    for (int i = 0; i < num_components; i++) {
        OSL_DEV_ONLY(std::cout << "llvm_gen_sub component(" << i << ") of "
                               << A.name() << " " << B.name() << std::endl);
        llvm::Value* a = rop.loadLLVMValue(A, i, 0, type, op_is_uniform);
        llvm::Value* b = rop.loadLLVMValue(B, i, 0, type, op_is_uniform);
        if (!a || !b)
            return false;
        llvm::Value* r = rop.ll.op_sub(a, b);
        if (op_is_uniform && !result_is_uniform) {
            r = rop.ll.widen_value(r);
        }
        rop.storeLLVMValue(r, Result, i, 0);
    }

    if (Result.has_derivs()) {
        if (A.has_derivs() || B.has_derivs()) {
            for (int d = 1; d <= 2; ++d) {  // dx, dy
                for (int i = 0; i < num_components; i++) {
                    llvm::Value* a = rop.loadLLVMValue(A, i, d, type,
                                                       op_is_uniform);
                    llvm::Value* b = rop.loadLLVMValue(B, i, d, type,
                                                       op_is_uniform);
                    llvm::Value* r = rop.ll.op_sub(a, b);
                    if (op_is_uniform && !result_is_uniform) {
                        r = rop.ll.widen_value(r);
                    }
                    rop.storeLLVMValue(r, Result, i, d);
                }
            }
        } else {
            // Result has derivs, operands do not
            rop.llvm_zero_derivs(Result);
        }
    }
    return true;
}



LLVMGEN(llvm_gen_mul)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& A      = *rop.opargsym(op, 1);
    Symbol& B      = *rop.opargsym(op, 2);

    bool op_is_uniform = A.is_uniform() && B.is_uniform();

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = !Result.typespec().is_closure_based()
                    && Result.typespec().is_float_based();
    int num_components = type.aggregate;

    bool resultIsUniform = Result.is_uniform();
    OSL_ASSERT(op_is_uniform || !resultIsUniform);

    // multiplication involving closures
    if (Result.typespec().is_closure()) {
        BatchedBackendLLVM::TempScope temp_scope(rop);

        llvm::Value* valargs[5];
        valargs[0] = rop.sg_void_ptr();
        valargs[1] = rop.llvm_void_ptr(Result);
        bool tfloat;
        bool aIsTheClosure = A.typespec().is_closure();
        Symbol& inClosure  = aIsTheClosure ? A : B;
        Symbol& t          = aIsTheClosure ? B : A;

        valargs[2] = rop.llvm_void_ptr(inClosure);

        tfloat = t.typespec().is_float();
        if (t.is_uniform()) {
            valargs[3] = rop.llvm_widen_value_into_temp(t, 0);
        } else {
            valargs[3] = rop.llvm_void_ptr(t);
        }

        valargs[4] = rop.ll.mask_as_int(rop.ll.current_mask());

        FuncSpec mul_closure(tfloat ? "mul_closure_float"
                                    : "mul_closure_color");
        mul_closure.mask();

        // we directly write the result in add_closure_closure instead of using a
        // store to write it back to result as in the single-point path
        rop.ll.call_function(rop.build_name(mul_closure), valargs);

        return true;
    }

    // multiplication involving matrices
    if (Result.typespec().is_matrix()) {
        FuncSpec func_spec("mul");
        func_spec.arg(Result, false, op_is_uniform);
        Symbol* A_prime = &A;
        Symbol* B_prime = &B;
        if (!A.typespec().is_matrix()) {
            // Always pass the matrix as the 1st operand
            std::swap(A_prime, B_prime);
        }
        func_spec.arg(*A_prime, false, op_is_uniform);
        func_spec.arg(*B_prime, false, op_is_uniform);

        if (op_is_uniform)
            func_spec.unbatch();
        rop.llvm_call_function(func_spec, Result, *A_prime, *B_prime,
                               false /*deriv_ptrs*/, op_is_uniform,
                               false /*functionIsLlvmInlined*/,
                               true /*ptrToReturnStructIs1stArg*/);

        if (Result.has_derivs())
            rop.llvm_zero_derivs(Result);
        return true;
    }

    // The following should handle f*f, v*v, v*f, f*v, i*i
    // That's all that should be allowed by oslc.
    for (int i = 0; i < num_components; i++) {
        OSL_DEV_ONLY(std::cout << "llvm_gen_mul component(" << i << ") of "
                               << A.name() << " " << B.name() << std::endl);

        llvm::Value* a = rop.llvm_load_value(A, 0, i, type, op_is_uniform);
        llvm::Value* b = rop.llvm_load_value(B, 0, i, type, op_is_uniform);
        if (!a || !b)
            return false;
        llvm::Value* r = rop.ll.op_mul(a, b);

        if (op_is_uniform && !resultIsUniform) {
            r = rop.ll.widen_value(r);
        }

        rop.llvm_store_value(r, Result, 0, i);

        if (Result.has_derivs() && (A.has_derivs() || B.has_derivs())) {
            // Multiplication of duals: (a*b, a*b.dx + a.dx*b, a*b.dy + a.dy*b)
            OSL_ASSERT(is_float);
            llvm::Value* ax = rop.llvm_load_value(A, 1, i, type, op_is_uniform);
            llvm::Value* bx = rop.llvm_load_value(B, 1, i, type, op_is_uniform);
            llvm::Value* abx = rop.ll.op_mul(a, bx);
            llvm::Value* axb = rop.ll.op_mul(ax, b);
            llvm::Value* rx  = rop.ll.op_add(abx, axb);
            llvm::Value* ay = rop.llvm_load_value(A, 2, i, type, op_is_uniform);
            llvm::Value* by = rop.llvm_load_value(B, 2, i, type, op_is_uniform);
            llvm::Value* aby = rop.ll.op_mul(a, by);
            llvm::Value* ayb = rop.ll.op_mul(ay, b);
            llvm::Value* ry  = rop.ll.op_add(aby, ayb);

            if (op_is_uniform && !resultIsUniform) {
                rx = rop.ll.widen_value(rx);
                ry = rop.ll.widen_value(ry);
            }

            rop.llvm_store_value(rx, Result, 1, i);
            rop.llvm_store_value(ry, Result, 2, i);
        }
    }

    if (Result.has_derivs() && !(A.has_derivs() || B.has_derivs())) {
        // Result has derivs, operands do not
        rop.llvm_zero_derivs(Result);
    }

    return true;
}



LLVMGEN(llvm_gen_div)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& A      = *rop.opargsym(op, 1);
    Symbol& B      = *rop.opargsym(op, 2);

    bool op_is_uniform   = A.is_uniform() && B.is_uniform();
    bool resultIsUniform = Result.is_uniform();
    OSL_ASSERT(op_is_uniform || !resultIsUniform);

    TypeDesc type        = Result.typespec().simpletype();
    bool is_float        = Result.typespec().is_float_based();
    int num_components   = type.aggregate;
    int B_num_components = B.typespec().simpletype().aggregate;

    OSL_ASSERT(!Result.typespec().is_closure_based());

    // division involving matrices
    if (Result.typespec().is_matrix()) {
        FuncSpec func_spec("div");
        if (op_is_uniform)
            func_spec.unbatch();
        func_spec.arg(Result, false, op_is_uniform);
        func_spec.arg(A, false, op_is_uniform);
        func_spec.arg(B, false, op_is_uniform);
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
                require_mask_be_passed = rop.ll.create_masking_scope(
                    /*enabled=*/true);
            }
            rop.llvm_call_function(func_spec, Result, A, B,
                                   false /*deriv_ptrs*/, op_is_uniform,
                                   false /*functionIsLlvmInlined*/,
                                   true /*ptrToReturnStructIs1stArg*/);
        }

        if (Result.has_derivs())
            rop.llvm_zero_derivs(Result);
        return true;
    }

    // The following should handle f/f, v/v, v/f, f/v, i/i
    // That's all that should be allowed by oslc.
    llvm::Value* c_zero = (op_is_uniform)
                              ? (is_float)
                                    ? rop.ll.constant(0.0f)
                                    : rop.ll.constant(static_cast<int>(0))
                          : (is_float)
                              ? rop.ll.wide_constant(0.0f)
                              : rop.ll.wide_constant(static_cast<int>(0));

    bool deriv = (Result.has_derivs() && (A.has_derivs() || B.has_derivs()));
    llvm::Value* c_one = nullptr;
    if (deriv || !is_float) {
        c_one = (op_is_uniform) ? (is_float)
                                      ? rop.ll.constant(1.0f)
                                      : rop.ll.constant(static_cast<int>(1))
                : (is_float) ? rop.ll.wide_constant(1.0f)
                                : rop.ll.wide_constant(static_cast<int>(1));
    }

    llvm::Value* b = nullptr;
    for (int i = 0; i < num_components; i++) {
        llvm::Value* a = rop.llvm_load_value(A, 0, i, type, op_is_uniform);
        // Don't reload the same value multiple times
        if (i < B_num_components) {
            b = rop.llvm_load_value(B, 0, i, type, op_is_uniform);
        }
        if (!a || !b)
            return false;

        llvm::Value* a_div_b;
        if (B.is_constant() && !rop.is_zero(B) && !is_float) {
            a_div_b = rop.ll.op_div(a, b);
        } else {
            // safe_div, implement here vs. calling a function
            if (is_float) {
                a_div_b                        = rop.ll.op_div(a, b);
                llvm::Value* b_notFiniteResult = rop.ll.op_is_not_finite(
                    a_div_b);
                a_div_b = rop.ll.op_zero_if(b_notFiniteResult, a_div_b);
            } else {
                llvm::Value* b_not_zero = rop.ll.op_ne(b, c_zero);
                // NOTE:  Not sure why, but llvm " sdiv <16 x i32>" is not generating SIMD but
                // instead reverting to regular scalar divisions
                // This means it will execute an IDIV potentially with a 0 causing and exception
                // because we use the "not equal 0" mask to select a 0 vs. the expected NAN from the vectorized division
                // An alternative to the selecting the replacing the results
                // is to selectively change the divisor to a non zero
                llvm::Value* divisor = rop.ll.op_select(b_not_zero, b, c_one);
                a_div_b              = rop.ll.op_select(b_not_zero,
                                                        rop.ll.op_div(a, divisor), c_zero);
                // Alternatively we could call a library function
                // Alternatively we could could emit SIMD intrinsics directly
            }
        }

        llvm::Value *rx = NULL, *ry = NULL;

        if (deriv) {
            // Division of duals: (a/b, 1/b*(ax-a/b*bx), 1/b*(ay-a/b*by))
            OSL_ASSERT(is_float);
            llvm::Value* binv                 = rop.ll.op_div(c_one, b);
            llvm::Value* binv_notFiniteResult = rop.ll.op_is_not_finite(binv);
            binv            = rop.ll.op_zero_if(binv_notFiniteResult, binv);
            llvm::Value* ax = rop.llvm_load_value(A, 1, i, type, op_is_uniform);
            llvm::Value* bx = rop.llvm_load_value(B, 1, i, type, op_is_uniform);
            llvm::Value* a_div_b_mul_bx = rop.ll.op_mul(a_div_b, bx);
            llvm::Value* ax_minus_a_div_b_mul_bx
                = rop.ll.op_sub(ax, a_div_b_mul_bx);
            rx              = rop.ll.op_mul(binv, ax_minus_a_div_b_mul_bx);
            llvm::Value* ay = rop.llvm_load_value(A, 2, i, type, op_is_uniform);
            llvm::Value* by = rop.llvm_load_value(B, 2, i, type, op_is_uniform);
            llvm::Value* a_div_b_mul_by = rop.ll.op_mul(a_div_b, by);
            llvm::Value* ay_minus_a_div_b_mul_by
                = rop.ll.op_sub(ay, a_div_b_mul_by);
            ry = rop.ll.op_mul(binv, ay_minus_a_div_b_mul_by);
        }

        if (op_is_uniform && !resultIsUniform) {
            a_div_b = rop.ll.widen_value(a_div_b);
            if (deriv) {
                rx = rop.ll.widen_value(rx);
                ry = rop.ll.widen_value(ry);
            }
        }
        rop.llvm_store_value(a_div_b, Result, 0, i);
        if (deriv) {
            rop.llvm_store_value(rx, Result, 1, i);
            rop.llvm_store_value(ry, Result, 2, i);
        }
    }

    if (Result.has_derivs() && !(A.has_derivs() || B.has_derivs())) {
        // Result has derivs, operands do not
        rop.llvm_zero_derivs(Result);
    }

    return true;
}



LLVMGEN(llvm_gen_modulus)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& A      = *rop.opargsym(op, 1);
    Symbol& B      = *rop.opargsym(op, 2);

    TypeDesc type = Result.typespec().simpletype();
    bool is_float = Result.typespec().is_float_based();

    bool op_is_uniform     = A.is_uniform() && B.is_uniform();
    bool result_is_uniform = Result.is_uniform();
    OSL_ASSERT(op_is_uniform || !result_is_uniform);

    int num_components = type.aggregate;

#if 0
    if (is_float && !op_is_uniform) {
        // llvm 5.0.1 did not do a good job with op_mod when its
        // parameters were <16xf32>.  So we will go ahead
        // and call an optimized library version.
        // Future versions of llvm might do better and this
        // could be removed
        BatchedBackendLLVM::TempScope temp_scope(rop);

        std::vector<llvm::Value*> call_args;
        call_args.push_back(rop.llvm_void_ptr(Result));
        call_args.push_back(
            rop.llvm_load_arg(A, false /*derivs*/, false /*is_uniform*/));
        call_args.push_back(
            rop.llvm_load_arg(B, false /*derivs*/, false /*is_uniform*/));

        FuncSpec func_spec("fmod");
        func_spec.arg(Result, false /*derivs*/, false /*is_uniform*/);
        func_spec.arg(A, false /*derivs*/, false /*is_uniform*/);
        func_spec.arg(B, false /*derivs*/, false /*is_uniform*/);

        if (rop.ll.is_masking_required()) {
            func_spec.mask();
            call_args.push_back(rop.ll.mask_as_int(rop.ll.current_mask()));
        }

        rop.ll.call_function(rop.build_name(func_spec), call_args);
    } else
#endif
    {
        for (int i = 0; i < num_components; i++) {
            llvm::Value* a = rop.loadLLVMValue(A, i, 0, type, op_is_uniform);
            llvm::Value* b = rop.loadLLVMValue(B, i, 0, type, op_is_uniform);
            if (!a || !b)
                return false;
            llvm::Value* zeroConstant;
            if (is_float) {
                zeroConstant = op_is_uniform ? rop.ll.constant(0.0f)
                                             : rop.ll.wide_constant(0.0f);
            } else {
                // Integer versions of safe mod handled in stdosl.h
                // We will leave the code to handle ints here as well
                zeroConstant = op_is_uniform ? rop.ll.constant(0)
                                             : rop.ll.wide_constant(0);
            }

            llvm::Value* is_zero_mask = rop.ll.op_eq(b, zeroConstant);
            llvm::Value* mod_result   = rop.ll.op_mod(a, b);
            llvm::Value* r = rop.ll.op_select(is_zero_mask, zeroConstant,
                                              mod_result);
            if (op_is_uniform && !result_is_uniform) {
                r = rop.ll.widen_value(r);
            }
            rop.storeLLVMValue(r, Result, i, 0);
        }
    }

    if (Result.has_derivs()) {
        OSL_ASSERT(is_float);
        if (A.has_derivs()) {
            // Modulus of duals: (a mod b, ax, ay)
            for (int d = 1; d <= 2; ++d) {
                for (int i = 0; i < num_components; i++) {
                    llvm::Value* deriv = rop.loadLLVMValue(A, i, d, type,
                                                           result_is_uniform);
                    rop.storeLLVMValue(deriv, Result, i, d);
                }
            }
        } else {
            // Result has derivs, operands do not
            rop.llvm_zero_derivs(Result);
        }
    }
    return true;
}



LLVMGEN(llvm_gen_neg)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& A      = *rop.opargsym(op, 1);

    bool op_is_uniform     = A.is_uniform();
    bool result_is_uniform = Result.is_uniform();
    OSL_ASSERT(op_is_uniform || !result_is_uniform);

    TypeDesc type      = Result.typespec().simpletype();
    int num_components = type.aggregate;
    for (int d = 0; d < 3; ++d) {  // dx, dy
        for (int i = 0; i < num_components; i++) {
            llvm::Value* a = rop.llvm_load_value(A, d, i, type, op_is_uniform);
            llvm::Value* r = rop.ll.op_neg(a);
            if (op_is_uniform && !result_is_uniform) {
                r = rop.ll.widen_value(r);
            }
            rop.llvm_store_value(r, Result, d, i);
        }
        if (!Result.has_derivs())
            break;
    }
    return true;
}



// Implementation for clamp
LLVMGEN(llvm_gen_clamp)
{
    // NOTE: stdosl.h appears to not expose clamp as a builtin
    // so don't expect this to be executed
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& X      = *rop.opargsym(op, 1);
    Symbol& Min    = *rop.opargsym(op, 2);
    Symbol& Max    = *rop.opargsym(op, 3);

    bool op_is_uniform = X.is_uniform() && Min.is_uniform() && Max.is_uniform();
    bool result_is_uniform = Result.is_uniform();
    OSL_ASSERT(op_is_uniform || !result_is_uniform);

    TypeDesc type      = Result.typespec().simpletype();
    int num_components = type.aggregate;
    for (int i = 0; i < num_components; i++) {
        // First do the lower bound
        llvm::Value* val  = rop.llvm_load_value(X, 0, i, type, op_is_uniform);
        llvm::Value* min  = rop.llvm_load_value(Min, 0, i, type, op_is_uniform);
        llvm::Value* cond = rop.ll.op_lt(val, min);
        val               = rop.ll.op_select(cond, min, val);
        llvm::Value *valdx = NULL, *valdy = NULL;
        if (Result.has_derivs()) {
            valdx = rop.llvm_load_value(X, 1, i, type, op_is_uniform);
            valdy = rop.llvm_load_value(X, 2, i, type, op_is_uniform);
            llvm::Value* mindx = rop.llvm_load_value(Min, 1, i, type,
                                                     op_is_uniform);
            llvm::Value* mindy = rop.llvm_load_value(Min, 2, i, type,
                                                     op_is_uniform);
            valdx              = rop.ll.op_select(cond, mindx, valdx);
            valdy              = rop.ll.op_select(cond, mindy, valdy);
        }
        // Now do the upper bound
        llvm::Value* max = rop.llvm_load_value(Max, 0, i, type, op_is_uniform);
        cond             = rop.ll.op_gt(val, max);
        val              = rop.ll.op_select(cond, max, val);
        if (Result.has_derivs()) {
            llvm::Value* maxdx = rop.llvm_load_value(Max, 1, i, type,
                                                     op_is_uniform);
            llvm::Value* maxdy = rop.llvm_load_value(Max, 2, i, type,
                                                     op_is_uniform);
            valdx              = rop.ll.op_select(cond, maxdx, valdx);
            valdy              = rop.ll.op_select(cond, maxdy, valdy);
        }

        if (op_is_uniform && !result_is_uniform) {
            val   = rop.ll.widen_value(val);
            valdx = rop.ll.widen_value(valdx);
            valdy = rop.ll.widen_value(valdy);
        }

        rop.llvm_store_value(val, Result, 0, i);
        rop.llvm_store_value(valdx, Result, 1, i);
        rop.llvm_store_value(valdy, Result, 2, i);
    }
    return true;
}



LLVMGEN(llvm_gen_mix)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& A      = *rop.opargsym(op, 1);
    Symbol& B      = *rop.opargsym(op, 2);
    Symbol& X      = *rop.opargsym(op, 3);

    bool op_is_uniform = Result.is_uniform();

    TypeDesc type = Result.typespec().simpletype();
    OSL_ASSERT(!Result.typespec().is_closure_based()
               && Result.typespec().is_float_based());
    int num_components = type.aggregate;
    int x_components   = X.typespec().aggregate();
    bool derivs        = (Result.has_derivs()
                   && (A.has_derivs() || B.has_derivs() || X.has_derivs()));

    llvm::Value* one;
    if (op_is_uniform)
        one = rop.ll.constant(1.0f);
    else
        one = rop.ll.wide_constant(1.0f);

    llvm::Value* x = rop.llvm_load_value(X, 0, 0, type, op_is_uniform);
    llvm::Value* one_minus_x = rop.ll.op_sub(one, x);
    llvm::Value* xx = derivs ? rop.llvm_load_value(X, 1, 0, type, op_is_uniform)
                             : NULL;
    llvm::Value* xy = derivs ? rop.llvm_load_value(X, 2, 0, type, op_is_uniform)
                             : NULL;
    for (int i = 0; i < num_components; i++) {
        llvm::Value* a = rop.llvm_load_value(A, 0, i, type, op_is_uniform);
        llvm::Value* b = rop.llvm_load_value(B, 0, i, type, op_is_uniform);
        if (!a || !b)
            return false;
        if (i > 0 && x_components > 1) {
            // Only need to recompute x and 1-x if they change
            x           = rop.llvm_load_value(X, 0, i, type, op_is_uniform);
            one_minus_x = rop.ll.op_sub(one, x);
        }
        // r = a*one_minus_x + b*x
        llvm::Value* r1 = rop.ll.op_mul(a, one_minus_x);
        llvm::Value* r2 = rop.ll.op_mul(b, x);
        llvm::Value* r  = rop.ll.op_add(r1, r2);
        rop.llvm_store_value(r, Result, 0, i);

        if (derivs) {
            // mix of duals:
            //   (a*one_minus_x + b*x,
            //    a*one_minus_x.dx + a.dx*one_minus_x + b*x.dx + b.dx*x,
            //    a*one_minus_x.dy + a.dy*one_minus_x + b*x.dy + b.dy*x)
            // and since one_minus_x.dx = -x.dx, one_minus_x.dy = -x.dy,
            //   (a*one_minus_x + b*x,
            //    -a*x.dx + a.dx*one_minus_x + b*x.dx + b.dx*x,
            //    -a*x.dy + a.dy*one_minus_x + b*x.dy + b.dy*x)
            llvm::Value* ax = rop.llvm_load_value(A, 1, i, type, op_is_uniform);
            llvm::Value* bx = rop.llvm_load_value(B, 1, i, type, op_is_uniform);
            if (i > 0 && x_components > 1)
                xx = rop.llvm_load_value(X, 1, i, type, op_is_uniform);
            llvm::Value* rx1 = rop.ll.op_mul(a, xx);
            llvm::Value* rx2 = rop.ll.op_mul(ax, one_minus_x);
            llvm::Value* rx  = rop.ll.op_sub(rx2, rx1);
            llvm::Value* rx3 = rop.ll.op_mul(b, xx);
            rx               = rop.ll.op_add(rx, rx3);
            llvm::Value* rx4 = rop.ll.op_mul(bx, x);
            rx               = rop.ll.op_add(rx, rx4);

            llvm::Value* ay = rop.llvm_load_value(A, 2, i, type, op_is_uniform);
            llvm::Value* by = rop.llvm_load_value(B, 2, i, type, op_is_uniform);
            if (i > 0 && x_components > 1)
                xy = rop.llvm_load_value(X, 2, i, type, op_is_uniform);
            llvm::Value* ry1 = rop.ll.op_mul(a, xy);
            llvm::Value* ry2 = rop.ll.op_mul(ay, one_minus_x);
            llvm::Value* ry  = rop.ll.op_sub(ry2, ry1);
            llvm::Value* ry3 = rop.ll.op_mul(b, xy);
            ry               = rop.ll.op_add(ry, ry3);
            llvm::Value* ry4 = rop.ll.op_mul(by, x);
            ry               = rop.ll.op_add(ry, ry4);

            rop.llvm_store_value(rx, Result, 1, i);
            rop.llvm_store_value(ry, Result, 2, i);
        }
    }

    if (Result.has_derivs() && !derivs) {
        // Result has derivs, operands do not
        rop.llvm_zero_derivs(Result);
    }

    return true;
}



LLVMGEN(llvm_gen_select)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& A      = *rop.opargsym(op, 1);
    Symbol& B      = *rop.opargsym(op, 2);
    Symbol& X      = *rop.opargsym(op, 3);
    TypeDesc type  = Result.typespec().simpletype();
    OSL_ASSERT(!Result.typespec().is_closure_based()
               && Result.typespec().is_float_based());
    int num_components = type.aggregate;
    int x_components   = X.typespec().aggregate();
    bool derivs = (Result.has_derivs() && (A.has_derivs() || B.has_derivs()));


    bool op_is_uniform     = A.is_uniform() && B.is_uniform() && X.is_uniform();
    bool result_is_uniform = Result.is_uniform();
    OSL_ASSERT(op_is_uniform || (op_is_uniform == result_is_uniform));

    llvm::Value* zero;
    if (X.is_uniform()) {
        zero = X.typespec().is_int() ? rop.ll.constant(0)
                                     : rop.ll.constant(0.0f);
    } else {
        zero = X.typespec().is_int() ? rop.ll.wide_constant(0)
                                     : rop.ll.wide_constant(0.0f);
    }
    llvm::Value* cond[3];
    for (int i = 0; i < x_components; ++i) {
        llvm::Value* x_val = rop.llvm_load_value(X, 0, i, TypeDesc::UNKNOWN,
                                                 X.is_uniform());
        if (rop.ll.llvm_typeof(x_val) == rop.ll.type_wide_bool()) {
            // X was already a result of a comparison we can use it directly
            // in the select, no need to promote it to integer and
            // compare to 0
            cond[i] = x_val;
        } else {
            cond[i] = rop.ll.op_ne(x_val, zero);
        }
    }

    for (int i = 0; i < num_components; i++) {
        llvm::Value* a = rop.llvm_load_value(A, 0, i, type, op_is_uniform);
        llvm::Value* b = rop.llvm_load_value(B, 0, i, type, op_is_uniform);
        llvm::Value* c = (i >= x_components) ? cond[0] : cond[i];
        llvm::Value* r = rop.ll.op_select(c, b, a);
        if (op_is_uniform && !result_is_uniform) {
            r = rop.ll.widen_value(r);
        }
        rop.llvm_store_value(r, Result, 0, i);
        if (derivs) {
            for (int d = 1; d < 3; ++d) {
                a = rop.llvm_load_value(A, d, i, type, op_is_uniform);
                b = rop.llvm_load_value(B, d, i, type, op_is_uniform);
                r = rop.ll.op_select(c, b, a);
                if (op_is_uniform && !result_is_uniform) {
                    r = rop.ll.widen_value(r);
                }
                rop.llvm_store_value(r, Result, d, i);
            }
        }
    }

    if (Result.has_derivs() && !derivs) {
        // Result has derivs, operands do not
        rop.llvm_zero_derivs(Result);
    }
    return true;
}



// Implementation for min/max
LLVMGEN(llvm_gen_minmax)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& x      = *rop.opargsym(op, 1);
    Symbol& y      = *rop.opargsym(op, 2);

    bool op_is_uniform     = x.is_uniform() && y.is_uniform();
    bool result_is_uniform = Result.is_uniform();

    TypeDesc type      = Result.typespec().simpletype();
    int num_components = type.aggregate;
    for (int i = 0; i < num_components; i++) {
        // First do the lower bound
        llvm::Value* x_val = rop.llvm_load_value(x, 0, i, type, op_is_uniform);
        llvm::Value* y_val = rop.llvm_load_value(y, 0, i, type, op_is_uniform);

        llvm::Value* cond = NULL;
        // NOTE(boulos): Using <= instead of < to match old behavior
        // (only matters for derivs)
        if (op.opname() == op_min) {
            cond = rop.ll.op_le(x_val, y_val);
        } else {
            cond = rop.ll.op_gt(x_val, y_val);
        }

        llvm::Value* res_val = rop.ll.op_select(cond, x_val, y_val);
        if (op_is_uniform && !result_is_uniform) {
            res_val = rop.ll.widen_value(res_val);
        }
        rop.llvm_store_value(res_val, Result, 0, i);
        if (Result.has_derivs()) {
            llvm::Value* x_dx = rop.llvm_load_value(x, 1, i, type,
                                                    op_is_uniform);
            llvm::Value* x_dy = rop.llvm_load_value(x, 2, i, type,
                                                    op_is_uniform);
            llvm::Value* y_dx = rop.llvm_load_value(y, 1, i, type,
                                                    op_is_uniform);
            llvm::Value* y_dy = rop.llvm_load_value(y, 2, i, type,
                                                    op_is_uniform);

            llvm::Value* res_dx = rop.ll.op_select(cond, x_dx, y_dx);
            llvm::Value* res_dy = rop.ll.op_select(cond, x_dy, y_dy);
            if (op_is_uniform && !result_is_uniform) {
                res_dx = rop.ll.widen_value(res_dx);
                res_dy = rop.ll.widen_value(res_dy);
            }

            rop.llvm_store_value(res_dx, Result, 1, i);
            rop.llvm_store_value(res_dy, Result, 2, i);
        }
    }
    return true;
}



LLVMGEN(llvm_gen_bitwise_binary_op)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& A      = *rop.opargsym(op, 1);
    Symbol& B      = *rop.opargsym(op, 2);
    OSL_ASSERT(Result.typespec().is_int() && A.typespec().is_int()
               && B.typespec().is_int());

    bool op_is_uniform     = A.is_uniform() && B.is_uniform();
    bool result_is_uniform = Result.is_uniform();

    TypeDesc type      = TypeInt;
    bool is_logical_op = op.opname() == op_bitand || op.opname() == op_bitor
                         || op.opname() == op_xor;
    if (is_logical_op) {
        // NOTE: we are loading with TypeDesc::UNKNOWN to avoid automatically
        // promoting bools to ints, so we can perform the logic operation on booleans
        // when possible
        type = TypeDesc::UNKNOWN;
    }

    llvm::Value* a = rop.loadLLVMValue(A, 0, 0, type, op_is_uniform);
    llvm::Value* b = rop.loadLLVMValue(B, 0, 0, type, op_is_uniform);

    if (is_logical_op) {
        llvm::Type* typeOfA = rop.ll.llvm_typeof(a);
        llvm::Type* typeOfB = rop.ll.llvm_typeof(b);
        if (typeOfA != typeOfB) {
            if (typeOfA == rop.ll.type_wide_bool()
                || typeOfA == rop.ll.type_bool()) {
                a = rop.ll.op_bool_to_int(a);
            }
            if (typeOfB == rop.ll.type_wide_bool()
                || typeOfB == rop.ll.type_bool()) {
                b = rop.ll.op_bool_to_int(b);
            }
        }
    }

    if (!a || !b)
        return false;
    llvm::Value* r = NULL;
    if (op.opname() == op_bitand) {
        r = rop.ll.op_and(a, b);
    } else if (op.opname() == op_bitor) {
        r = rop.ll.op_or(a, b);
    } else if (op.opname() == op_xor) {
        r = rop.ll.op_xor(a, b);
    } else if (op.opname() == op_shl) {
        r = rop.ll.op_shl(a, b);
    } else if (op.opname() == op_shr) {
        r = rop.ll.op_shr(a, b);
    } else
        return false;

    if (op_is_uniform && !result_is_uniform) {
        r = rop.ll.widen_value(r);
    }

    // TODO: currently only only the results of a 'compare op' or
    // getattribute success return value are forced to be boolean
    // Would be nice if results of and,or,xor were allowed to be boolean,
    // but that would only be true when both operands are bool as well
    // which would require work in BatchedAnalysis, so leave for later

    // We allowed boolean values to pass through or,and,xor
    // so we might need to promote to integer before storage
    if (!Result.forced_llvm_bool()) {
#ifndef OSL_LLVM_OPAQUE_POINTERS
        llvm::Type* resultType = rop.ll.llvm_typeof(
            rop.llvm_get_pointer(Result));
        OSL_ASSERT(
            (resultType
             == reinterpret_cast<llvm::Type*>(rop.ll.type_wide_int_ptr()))
            || (resultType
                == reinterpret_cast<llvm::Type*>(rop.ll.type_int_ptr())));
#endif
        llvm::Type* typeOfR = rop.ll.llvm_typeof(r);
        if (typeOfR == rop.ll.type_wide_bool()
            || typeOfR == rop.ll.type_bool()) {
            r = rop.ll.op_bool_to_int(r);
        }
    }

    rop.storeLLVMValue(r, Result);
    return true;
}



// Simple (pointwise) unary ops (Abs, ...,
LLVMGEN(llvm_gen_unary_op)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& dst        = *rop.opargsym(op, 0);
    Symbol& src        = *rop.opargsym(op, 1);
    bool dst_derivs    = dst.has_derivs();
    int num_components = dst.typespec().simpletype().aggregate;

    bool op_is_uniform     = src.is_uniform();
    bool result_is_uniform = dst.is_uniform();

    bool dst_float = dst.typespec().is_float_based();
    bool src_float = src.typespec().is_float_based();

    TypeDesc type = dst.typespec().simpletype();

    for (int i = 0; i < num_components; i++) {
        // Get src1/2 component i
        llvm::Value* src_load = rop.loadLLVMValue(src, i, 0, type,
                                                  op_is_uniform);
        if (!src_load)
            return false;

        llvm::Value* src_val = src_load;

        // Perform the op
        llvm::Value* result = 0;
        ustring opname      = op.opname();

        if (opname == op_compl) {
            OSL_ASSERT(dst.typespec().is_int());
            result = rop.ll.op_not(src_val);

            //Uniform/Varying check

            if (op_is_uniform && !result_is_uniform) {
                result = rop.ll.widen_value(result);
            }
        } else {
            // Don't know how to handle this.
            rop.shadingcontext()->errorfmt(
                "Don't know how to handle op '{}', eliding the store\n",
                opname);
        }

        // Store the result
        if (result) {
            // if our op type doesn't match result, convert
            if (dst_float && !src_float) {
                // Op was int, but we need to store float
                result = rop.ll.op_int_to_float(result);
            } else if (!dst_float && src_float) {
                // Op was float, but we need to store int
                result = rop.ll.op_float_to_int(result);
            }  // otherwise just fine
            rop.storeLLVMValue(result, dst, i, 0);
        }

        if (dst_derivs) {
            // mul results in <a * b, a * b_dx + b * a_dx, a * b_dy + b * a_dy>
            rop.shadingcontext()->infofmt("punting on derivatives for now\n");
            // FIXME!!
        }
    }
    return true;
}



// Simple assignment
LLVMGEN(llvm_gen_assign)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result(*rop.opargsym(op, 0));
    Symbol& Src(*rop.opargsym(op, 1));

    return rop.llvm_assign_impl(Result, Src);
}



// Entire array copying
LLVMGEN(llvm_gen_arraycopy)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result(*rop.opargsym(op, 0));
    Symbol& Src(*rop.opargsym(op, 1));

    return rop.llvm_assign_impl(Result, Src);
}



// Vector component reference
LLVMGEN(llvm_gen_compref)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& Val    = *rop.opargsym(op, 1);
    Symbol& Index  = *rop.opargsym(op, 2);

    bool op_is_uniform = Result.is_uniform();

    llvm::Value* c = rop.llvm_load_value(Index, /*deriv=*/0, /*component=*/0,
                                         TypeDesc::UNKNOWN, Index.is_uniform());

    if (Index.is_uniform()) {
        if (rop.inst()->master()->range_checking()) {
            if (!(Index.is_constant() && Index.get_int() >= 0
                  && Index.get_int() < 3)) {
                llvm::Value* args[]
                    = { c,
                        rop.ll.constant(3),
                        rop.ll.constant(Val.unmangled()),
                        rop.sg_void_ptr(),
                        rop.ll.constant(op.sourcefile()),
                        rop.ll.constant(op.sourceline()),
                        rop.ll.constant(rop.group().name()),
                        rop.ll.constant(rop.layer()),
                        rop.ll.constant(rop.inst()->layername()),
                        rop.ll.constant(rop.inst()->shadername()) };
                c = rop.ll.call_function(rop.build_name("range_check"), args);
                OSL_ASSERT(c);
            }
        }
        for (int d = 0; d < 3; ++d) {  // deriv
            llvm::Value* val = NULL;
            if (Index.is_constant()) {
                int i = Index.get_int();
                i     = Imath::clamp(i, 0, 2);
                val   = rop.llvm_load_value(Val, d, i, TypeDesc::UNKNOWN,
                                            op_is_uniform);
            } else {
                // TODO: handle non constant index
                val = rop.llvm_load_component_value(Val, d, c, op_is_uniform);
            }
            rop.llvm_store_value(val, Result, d);
            if (!Result.has_derivs())  // skip the derivs if we don't need them
                break;
        }
    } else {
        OSL_ASSERT(Index.is_constant() == false);
        OSL_ASSERT(op_is_uniform == false);

        if (rop.inst()->master()->range_checking()) {
            BatchedBackendLLVM::TempScope temp_scope(rop);

            // We need a copy of the indices in case the range check clamps them
            llvm::Value* loc_clamped_wide_index = rop.getOrAllocateTemp(
                TypeSpec(TypeDesc::INT), false /*derivs*/, false /*is_uniform*/,
                false /*forceBool*/,
                std::string("range clamped index:") + Val.name().c_str());
            // copy the indices into our temporary
            rop.ll.op_unmasked_store(c, loc_clamped_wide_index);
            llvm::Value* args[] = { rop.ll.void_ptr(loc_clamped_wide_index),
                                    rop.ll.mask_as_int(rop.ll.current_mask()),
                                    rop.ll.constant(3),
                                    rop.ll.constant(Val.unmangled()),
                                    rop.sg_void_ptr(),
                                    rop.ll.constant(op.sourcefile()),
                                    rop.ll.constant(op.sourceline()),
                                    rop.ll.constant(rop.group().name()),
                                    rop.ll.constant(rop.layer()),
                                    rop.ll.constant(rop.inst()->layername()),
                                    rop.ll.constant(rop.inst()->shadername()) };
            rop.ll.call_function(rop.build_name(FuncSpec("range_check").mask()),
                                 args);

            // Use the range check indices
            // Although as our implementation below doesn't use any
            // out of range values, clamping the indices here
            // is of questionable value
            c = rop.ll.op_load(rop.ll.type_wide_int(), loc_clamped_wide_index);
        }

        // As the index is logically bound to 0, 1, or 2
        // instead of doing a gather (which we will assume to cost 16 loads)
        // We can just load all 3 components and blend them based on the index == 0, index == 1, index == 2
        llvm::Value* comp0Mask = rop.ll.op_eq(c, rop.ll.wide_constant(0));
        llvm::Value* comp1Mask = rop.ll.op_eq(c, rop.ll.wide_constant(1));
        // If index != 0 && index != 1, assume index == 2
        // Essentially free clamping

        for (int d = 0; d < 3; ++d) {  // deriv
            llvm::Value* valc0    = rop.llvm_load_value(Val, d, 0,
                                                        TypeDesc::UNKNOWN,
                                                        op_is_uniform);
            llvm::Value* valc1    = rop.llvm_load_value(Val, d, 1,
                                                        TypeDesc::UNKNOWN,
                                                        op_is_uniform);
            llvm::Value* valc2    = rop.llvm_load_value(Val, d, 2,
                                                        TypeDesc::UNKNOWN,
                                                        op_is_uniform);
            llvm::Value* valc0_c2 = rop.ll.op_select(comp0Mask, valc0, valc2);
            llvm::Value* valc0_c1_c2 = rop.ll.op_select(comp1Mask, valc1,
                                                        valc0_c2);

            rop.llvm_store_value(valc0_c1_c2, Result, d);
            if (!Result.has_derivs())  // skip the derivs if we don't need them
                break;
        }
    }
    return true;
}


// Vector component assignment
LLVMGEN(llvm_gen_compassign)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& Index  = *rop.opargsym(op, 1);
    Symbol& Val    = *rop.opargsym(op, 2);

    bool op_is_uniform = Result.is_uniform();

    llvm::Value* c = rop.llvm_load_value(Index,
                                         /*deriv*/ 0,
                                         /*component*/ 0, TypeDesc::UNKNOWN,
                                         Index.is_uniform());

    if (Index.is_uniform()) {
        if (rop.inst()->master()->range_checking()) {
            if (!(Index.is_constant() && Index.get_int() >= 0
                  && Index.get_int() < 3)) {
                llvm::Value* args[]
                    = { c,
                        rop.ll.constant(3),
                        rop.ll.constant(Result.unmangled()),
                        rop.sg_void_ptr(),
                        rop.ll.constant(op.sourcefile()),
                        rop.ll.constant(op.sourceline()),
                        rop.ll.constant(rop.group().name()),
                        rop.ll.constant(rop.layer()),
                        rop.ll.constant(rop.inst()->layername()),
                        rop.ll.constant(rop.inst()->shadername()) };
                c = rop.ll.call_function(rop.build_name("range_check"), args);
                OSL_ASSERT(c);
            }
        }

        for (int d = 0; d < 3; ++d) {  // deriv
            llvm::Value* val = rop.llvm_load_value(Val, d, 0, TypeFloat,
                                                   op_is_uniform);
            if (Index.is_constant()) {
                int i = Index.get_int();
                i     = Imath::clamp(i, 0, 2);
                rop.llvm_store_value(val, Result, d, i);
            } else {
                rop.llvm_store_component_value(val, Result, d, c);
            }
            if (!Result.has_derivs())  // skip the derivs if we don't need them
                break;
        }
    } else {
        OSL_ASSERT(Index.is_constant() == false);
        OSL_ASSERT(op_is_uniform == false);
        OSL_ASSERT(Result.typespec().aggregate() == 3
                   && "Implementation assumes 3 components");
        if (rop.inst()->master()->range_checking()) {
            BatchedBackendLLVM::TempScope temp_scope(rop);

            // We need a copy of the indices in case the range check clamps them
            llvm::Value* loc_clamped_wide_index = rop.getOrAllocateTemp(
                TypeSpec(TypeDesc::INT), false /*derivs*/, false /*is_uniform*/,
                false /*forceBool*/,
                std::string("range clamped index:") + Val.name().c_str());
            // copy the indices into our temporary
            rop.ll.op_unmasked_store(c, loc_clamped_wide_index);
            llvm::Value* args[] = { rop.ll.void_ptr(loc_clamped_wide_index),
                                    rop.ll.mask_as_int(rop.ll.current_mask()),
                                    rop.ll.constant(3),
                                    rop.ll.constant(Result.unmangled()),
                                    rop.sg_void_ptr(),
                                    rop.ll.constant(op.sourcefile()),
                                    rop.ll.constant(op.sourceline()),
                                    rop.ll.constant(rop.group().name()),
                                    rop.ll.constant(rop.layer()),
                                    rop.ll.constant(rop.inst()->layername()),
                                    rop.ll.constant(rop.inst()->shadername()) };
            rop.ll.call_function(rop.build_name(FuncSpec("range_check").mask()),
                                 args);
            // Use the range check indices
            // Although as our implementation below doesn't use any
            // out of range values, clamping the indices here
            // is of questionable value
            c = rop.ll.op_load(rop.ll.type_wide_int(), loc_clamped_wide_index);
        }


        // As the index is logically bound to 0, 1, or 2
        // instead of doing a scatter
        // We can just load all 3 components and blend them based on the index == 0, index == 1, index == 2
        llvm::Value* comp0Mask = rop.ll.op_eq(c, rop.ll.wide_constant(0));
        llvm::Value* comp1Mask = rop.ll.op_eq(c, rop.ll.wide_constant(1));
        llvm::Value* comp2Mask = rop.ll.op_eq(c, rop.ll.wide_constant(2));
        // If index != 0 && index != 1, assume index == 2
        // Essentially free clamping

        for (int d = 0; d < 3; ++d) {  // deriv

            llvm::Value* val = rop.llvm_load_value(Val, d, 0, TypeFloat,
                                                   op_is_uniform);

            llvm::Value* valc0 = rop.llvm_load_value(Result, d, 0,
                                                     TypeDesc::UNKNOWN,
                                                     op_is_uniform);
            llvm::Value* valc1 = rop.llvm_load_value(Result, d, 1,
                                                     TypeDesc::UNKNOWN,
                                                     op_is_uniform);
            llvm::Value* valc2 = rop.llvm_load_value(Result, d, 2,
                                                     TypeDesc::UNKNOWN,
                                                     op_is_uniform);

            llvm::Value* resultc0 = rop.ll.op_select(comp0Mask, val, valc0);
            llvm::Value* resultc1 = rop.ll.op_select(comp1Mask, val, valc1);
            llvm::Value* resultc2 = rop.ll.op_select(comp2Mask, val, valc2);

            rop.llvm_store_value(resultc0, Result, d, 0);
            rop.llvm_store_value(resultc1, Result, d, 1);
            rop.llvm_store_value(resultc2, Result, d, 2);

            if (!Result.has_derivs())  // skip the derivs if we don't need them
                break;
        }
    }
    return true;
}



// Matrix component reference
LLVMGEN(llvm_gen_mxcompref)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& M      = *rop.opargsym(op, 1);
    Symbol& Row    = *rop.opargsym(op, 2);
    Symbol& Col    = *rop.opargsym(op, 3);

    bool op_is_uniform          = Result.is_uniform();
    bool components_are_uniform = Row.is_uniform() && Col.is_uniform();

    llvm::Value* row = rop.llvm_load_value(Row, 0, 0, TypeDesc::UNKNOWN,
                                           components_are_uniform);
    llvm::Value* col = rop.llvm_load_value(Col, 0, 0, TypeDesc::UNKNOWN,
                                           components_are_uniform);

    if (rop.inst()->master()->range_checking()) {
        if (components_are_uniform) {
            if (!(Row.is_constant() && Row.get_int() >= 0 && Row.get_int() < 4
                  && Col.is_constant() && Col.get_int() >= 0
                  && Col.get_int() < 4)) {
                llvm::Value* args[]
                    = { row,
                        rop.ll.constant(4),
                        rop.ll.constant(M.name()),
                        rop.sg_void_ptr(),
                        rop.ll.constant(op.sourcefile()),
                        rop.ll.constant(op.sourceline()),
                        rop.ll.constant(rop.group().name()),
                        rop.ll.constant(rop.layer()),
                        rop.ll.constant(rop.inst()->layername()),
                        rop.ll.constant(rop.inst()->shadername()) };
                const char* func_name = rop.build_name("range_check");
                if (!(Row.is_constant() && Row.get_int() >= 0
                      && Row.get_int() < 4)) {
                    row = rop.ll.call_function(func_name, args);
                }
                if (!(Col.is_constant() && Col.get_int() >= 0
                      && Col.get_int() < 4)) {
                    args[0] = col;
                    col     = rop.ll.call_function(func_name, args);
                }
            }
        } else {
            BatchedBackendLLVM::TempScope temp_scope(rop);
            // We need a copy of the indices in case the range check clamps them
            llvm::Value* loc_clamped_wide_index = rop.getOrAllocateTemp(
                TypeSpec(TypeDesc::INT), false /*derivs*/, false /*is_uniform*/,
                false /*forceBool*/,
                std::string("range clamped row or col:") + M.name().c_str());
            // copy the indices into our temporary
            rop.ll.op_unmasked_store(row, loc_clamped_wide_index);
            llvm::Value* args[]   = { rop.ll.void_ptr(loc_clamped_wide_index),
                                      rop.ll.mask_as_int(rop.ll.current_mask()),
                                      rop.ll.constant(4),
                                      rop.ll.constant(M.name()),
                                      rop.sg_void_ptr(),
                                      rop.ll.constant(op.sourcefile()),
                                      rop.ll.constant(op.sourceline()),
                                      rop.ll.constant(rop.group().name()),
                                      rop.ll.constant(rop.layer()),
                                      rop.ll.constant(rop.inst()->layername()),
                                      rop.ll.constant(rop.inst()->shadername()) };
            const char* func_name = rop.build_name(
                FuncSpec("range_check").mask());
            rop.ll.call_function(func_name, args);
            // Use the range check row
            row = rop.ll.op_load(rop.ll.type_wide_int(),
                                 loc_clamped_wide_index);

            // copy the indices into our temporary
            rop.ll.op_unmasked_store(col, loc_clamped_wide_index);
            rop.ll.call_function(func_name, args);
            // Use the range check col
            col = rop.ll.op_load(rop.ll.type_wide_int(),
                                 loc_clamped_wide_index);
        }
    }

    llvm::Value* val = NULL;
    if (Row.is_constant() && Col.is_constant()) {
        int r    = Imath::clamp(Row.get_int(), 0, 3);
        int c    = Imath::clamp(Col.get_int(), 0, 3);
        int comp = 4 * r + c;
        val      = rop.llvm_load_value(M, 0, comp, TypeFloat, op_is_uniform);
    } else {
        llvm::Value* comp = rop.ll.op_mul(row, components_are_uniform
                                                   ? rop.ll.constant(4)
                                                   : rop.ll.wide_constant(4));
        comp              = rop.ll.op_add(comp, col);
        val = rop.llvm_load_component_value(M, 0, comp, op_is_uniform,
                                            components_are_uniform);
    }
    rop.llvm_store_value(val, Result);
    rop.llvm_zero_derivs(Result);

    return true;
}



// Matrix component assignment
LLVMGEN(llvm_gen_mxcompassign)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& Row    = *rop.opargsym(op, 1);
    Symbol& Col    = *rop.opargsym(op, 2);
    Symbol& Val    = *rop.opargsym(op, 3);

    bool op_is_uniform          = Result.is_uniform();
    bool components_are_uniform = Row.is_uniform() && Col.is_uniform();

    llvm::Value* row = rop.llvm_load_value(Row, 0, 0, TypeDesc::UNKNOWN,
                                           components_are_uniform);
    llvm::Value* col = rop.llvm_load_value(Col, 0, 0, TypeDesc::UNKNOWN,
                                           components_are_uniform);

    if (rop.inst()->master()->range_checking()) {
        if (components_are_uniform) {
            if (!(Row.is_constant() && Row.get_int() >= 0 && Row.get_int() < 4
                  && Col.is_constant() && Col.get_int() >= 0
                  && Col.get_int() < 4)) {
                llvm::Value* args[]
                    = { row,
                        rop.ll.constant(4),
                        rop.ll.constant(Result.name()),
                        rop.sg_void_ptr(),
                        rop.ll.constant(op.sourcefile()),
                        rop.ll.constant(op.sourceline()),
                        rop.ll.constant(rop.group().name()),
                        rop.ll.constant(rop.layer()),
                        rop.ll.constant(rop.inst()->layername()),
                        rop.ll.constant(rop.inst()->shadername()) };
                const char* func_name = rop.build_name("range_check");
                if (!(Row.is_constant() && Row.get_int() >= 0
                      && Row.get_int() < 4)) {
                    row = rop.ll.call_function(func_name, args);
                }

                if (!(Col.is_constant() && Col.get_int() >= 0
                      && Col.get_int() < 4)) {
                    args[0] = col;
                    col     = rop.ll.call_function(func_name, args);
                }
            }
        } else {
            BatchedBackendLLVM::TempScope temp_scope(rop);
            // We need a copy of the indices in case the range check clamps them
            llvm::Value* loc_clamped_wide_index = rop.getOrAllocateTemp(
                TypeSpec(TypeDesc::INT), false /*derivs*/, false /*is_uniform*/,
                false /*forceBool*/,
                std::string("range clamped row:") + Result.name().c_str());
            // copy the indices into our temporary
            rop.ll.op_unmasked_store(row, loc_clamped_wide_index);
            llvm::Value* args[]   = { rop.ll.void_ptr(loc_clamped_wide_index),
                                      rop.ll.mask_as_int(rop.ll.current_mask()),
                                      rop.ll.constant(4),
                                      rop.ll.constant(Result.name()),
                                      rop.sg_void_ptr(),
                                      rop.ll.constant(op.sourcefile()),
                                      rop.ll.constant(op.sourceline()),
                                      rop.ll.constant(rop.group().name()),
                                      rop.ll.constant(rop.layer()),
                                      rop.ll.constant(rop.inst()->layername()),
                                      rop.ll.constant(rop.inst()->shadername()) };
            const char* func_name = rop.build_name(
                FuncSpec("range_check").mask());
            rop.ll.call_function(func_name, args);
            // Use the range check row
            row = rop.ll.op_load(rop.ll.type_wide_int(),
                                 loc_clamped_wide_index);

            // copy the indices into our temporary
            rop.ll.op_unmasked_store(col, loc_clamped_wide_index);
            rop.ll.call_function(func_name, args);
            // Use the range check col
            col = rop.ll.op_load(rop.ll.type_wide_int(),
                                 loc_clamped_wide_index);
        }
    }

    llvm::Value* val = rop.llvm_load_value(Val, 0, 0, TypeFloat, op_is_uniform);

    if (Row.is_constant() && Col.is_constant()) {
        int r    = Imath::clamp(Row.get_int(), 0, 3);
        int c    = Imath::clamp(Col.get_int(), 0, 3);
        int comp = 4 * r + c;
        rop.llvm_store_value(val, Result, 0, comp);
    } else {
        llvm::Value* comp = rop.ll.op_mul(row, components_are_uniform
                                                   ? rop.ll.constant(4)
                                                   : rop.ll.wide_constant(4));
        comp              = rop.ll.op_add(comp, col);
        rop.llvm_store_component_value(val, Result, 0, comp,
                                       components_are_uniform);
    }
    return true;
}



// Construct color, optionally with a color transformation from a named
// color space.
LLVMGEN(llvm_gen_construct_color)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result   = *rop.opargsym(op, 0);
    bool using_space = (op.nargs() == 5);
    Symbol& Space    = *rop.opargsym(op, 1);
    Symbol& X        = *rop.opargsym(op, 1 + using_space);
    Symbol& Y        = *rop.opargsym(op, 2 + using_space);
    Symbol& Z        = *rop.opargsym(op, 3 + using_space);
    OSL_ASSERT(Result.typespec().is_triple() && X.typespec().is_float()
               && Y.typespec().is_float() && Z.typespec().is_float()
               && (using_space == false || Space.typespec().is_string()));

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
    for (int d = 0; d < dmax; ++d) {   // loop over derivs
        for (int c = 0; c < 3; ++c) {  // loop over components
            const Symbol& comp = *rop.opargsym(op, c + 1 + using_space);
            llvm::Value* val = rop.llvm_load_value(comp, d, NULL, 0, TypeFloat,
                                                   result_is_uniform);
            rop.llvm_store_value(val, Result, d, NULL, c);
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

        llvm::Value* args[4];
        // NOTE:  Shader Globals is only passed to provide access to report an error to the context
        // no implicit dependency on any Shader Globals is necessary
        args[0]       = rop.sg_void_ptr();             // shader globals
        args[1]       = rop.llvm_void_ptr(Result, 0);  // color
        args[2]       = space_is_uniform ? rop.llvm_load_value(Space)
                                         : rop.llvm_void_ptr(Space);  // from
        int arg_count = 3;
        // Until we avoid calling back into the shading system,
        // always call the masked version if we are not uniform
        // to allow skipping callbacks for masked off lanes
        if (!result_is_uniform /*&& rop.ll.is_masking_required()*/) {
            args[arg_count++] = rop.ll.mask_as_int(rop.ll.current_mask());
            func_spec.mask();
        }

        rop.ll.call_function(rop.build_name(func_spec),
                             cspan<llvm::Value*>(args, arg_count));
        // FIXME(deriv): Punt on derivs for color ctrs with space names.
        // We should try to do this right, but we never had it right for
        // the interpreter, to it's probably not an emergency.
        if (Result.has_derivs())
            rop.llvm_zero_derivs(Result);
    }

    return true;
}



// Derivs
LLVMGEN(llvm_gen_DxDy)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result(*rop.opargsym(op, 0));
    Symbol& Src(*rop.opargsym(op, 1));
    int deriv              = (op.opname() == "Dx") ? 1 : 2;
    bool result_is_uniform = Result.is_uniform();

    for (int i = 0; i < Result.typespec().aggregate(); ++i) {
        llvm::Value* src_val = rop.llvm_load_value(Src, deriv, i,
                                                   TypeDesc::UNKNOWN,
                                                   result_is_uniform);
        rop.storeLLVMValue(src_val, Result, i, 0);
    }

    // Don't have 2nd order derivs
    rop.llvm_zero_derivs(Result);
    return true;
}



// Dz
LLVMGEN(llvm_gen_Dz)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result(*rop.opargsym(op, 0));
    Symbol& Src(*rop.opargsym(op, 1));
    bool result_is_uniform = Result.is_uniform();

    if (&Src == rop.inst()->symbol(rop.inst()->Psym())) {
        // dPdz -- the only Dz we know how to take
        int deriv = 3;
        for (int i = 0; i < Result.typespec().aggregate(); ++i) {
            llvm::Value* src_val = rop.llvm_load_value(Src, deriv, i,
                                                       TypeDesc::UNKNOWN,
                                                       result_is_uniform);
            rop.storeLLVMValue(src_val, Result, i, 0);
        }
        // Don't have 2nd order derivs
        rop.llvm_zero_derivs(Result);
    } else {
        // Punt, everything else for now returns 0 for Dz
        // FIXME?
        rop.llvm_assign_zero(Result);
    }
    return true;
}



LLVMGEN(llvm_gen_filterwidth)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result(*rop.opargsym(op, 0));
    Symbol& Src(*rop.opargsym(op, 1));
    OSL_ASSERT(Src.typespec().is_float() || Src.typespec().is_triple());

    bool result_is_uniform = Result.is_uniform();
    bool op_is_uniform     = Src.is_uniform();

    if (Src.has_derivs()) {
        if (op_is_uniform) {  // result is Uniform
            if (Src.typespec().is_float()) {
                llvm::Value* r = rop.ll.call_function("osl_filterwidth_fdf",
                                                      rop.llvm_void_ptr(Src));

                if (!result_is_uniform) {
                    r = rop.ll.widen_value(r);
                }
                rop.llvm_store_value(r, Result);
            } else {
                BatchedBackendLLVM::TempScope temp_scope(rop);
                // Need 2 pointers, because the parameter must be void *
                // but we need a typed triple * for the broadcast later
                llvm::Value* result_typed_temp = nullptr;
                llvm::Value* result_param      = nullptr;
                if (!result_is_uniform) {
                    result_typed_temp
                        = rop.getOrAllocateTemp(Result.typespec(),
                                                Result.has_derivs(),
                                                true /*is_uniform*/);
                    result_param = rop.ll.void_ptr(result_typed_temp);
                } else {
                    result_param = rop.llvm_void_ptr(Result);
                }
                rop.ll.call_function("osl_filterwidth_vdv", result_param,
                                     rop.llvm_void_ptr(Src));

                if (!result_is_uniform) {
                    rop.llvm_broadcast_uniform_value_from_mem(result_typed_temp,
                                                              Result);
                }
            }
            // Don't have 2nd order derivs
            rop.llvm_zero_derivs(Result);
        } else {  // op is Varying
            FuncSpec func_spec("filterwidth");
            // The result may have derivatives, but we zero them out after this
            // function call, so just always treat the result as not having derivates
            func_spec.arg(Result, false /*derivs*/, false /*is_uniform*/);
            func_spec.arg(Src, true /*derivs*/, false /*is_uniform*/);

            llvm::Value* args[3];
            args[0]      = rop.llvm_void_ptr(Result);
            args[1]      = rop.llvm_void_ptr(Src);
            int argCount = 2;

            if (rop.ll.is_masking_required()) {
                func_spec.mask();
                args[2]  = rop.ll.mask_as_int(rop.ll.current_mask());
                argCount = 3;
            }

            rop.ll.call_function(rop.build_name(func_spec),
                                 { &args[0],
                                   cspan<llvm::Value*>::size_type(argCount) });
            // Don't have 2nd order derivs
            rop.llvm_zero_derivs(Result);
        }
    }

    else {  //If source has no derivs
        // No derivs to be had
        rop.llvm_assign_zero(Result);
    }

    return true;
}



// Comparison ops
LLVMGEN(llvm_gen_compare_op)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result(*rop.opargsym(op, 0));
    Symbol& A(*rop.opargsym(op, 1));
    Symbol& B(*rop.opargsym(op, 2));
    OSL_ASSERT(Result.typespec().is_int() && !Result.has_derivs());

    bool op_is_uniform     = A.is_uniform() && B.is_uniform();
    bool result_is_uniform = Result.is_uniform();

    if (A.typespec().is_closure()) {
        OSL_ASSERT(0 && "incomplete");
        OSL_ASSERT(B.typespec().is_int()
                   && "Only closure==0 and closure!=0 allowed");
        llvm::Value* a = rop.llvm_load_value(A);
        llvm::Value* b = rop.ll.void_ptr_null();
        llvm::Value* r = (op.opname() == op_eq) ? rop.ll.op_eq(a, b)
                                                : rop.ll.op_ne(a, b);
        // TODO: handle convert the single bit bool into an int, if necessary
        rop.llvm_store_value(r, Result);
        return true;
    }

    int num_components = std::max(A.typespec().aggregate(),
                                  B.typespec().aggregate());
    bool float_based   = A.typespec().is_float_based()
                       || B.typespec().is_float_based();
    TypeDesc cast(float_based ? TypeDesc::FLOAT : TypeDesc::UNKNOWN);

    llvm::Value* final_result = 0;
    ustring opname            = op.opname();

    for (int i = 0; i < num_components; i++) {
        // Get A&B component i -- note that these correctly handle mixed
        // scalar/triple comparisons as well as int->float casts as needed.
        llvm::Value* a = rop.loadLLVMValue(A, i, 0, cast, op_is_uniform);
        llvm::Value* b = rop.loadLLVMValue(B, i, 0, cast, op_is_uniform);

        llvm::Type* typeOfA = rop.ll.llvm_typeof(a);
        llvm::Type* typeOfB = rop.ll.llvm_typeof(b);

        if (typeOfA != typeOfB) {
            if ((typeOfA == rop.ll.type_bool() && typeOfB == rop.ll.type_int())
                || (typeOfA == rop.ll.type_wide_bool()
                    && typeOfB == rop.ll.type_wide_int())) {
                // TODO: could optimize for constant 0 and 1 and skip the comparison
                a = rop.ll.op_bool_to_int(a);
            }
            if ((typeOfB == rop.ll.type_bool() && typeOfA == rop.ll.type_int())
                || (typeOfB == rop.ll.type_wide_bool()
                    && typeOfA == rop.ll.type_wide_int())) {
                b = rop.ll.op_bool_to_int(b);
            }
        }

        // Trickery for mixed matrix/scalar comparisons -- compare
        // on-diagonal to the scalar, off-diagonal to zero
        if (A.typespec().is_matrix() && !B.typespec().is_matrix()) {
            if ((i / 4) != (i % 4)) {
                if (op_is_uniform)
                    b = rop.ll.constant(0.0f);
                else
                    b = rop.ll.wide_constant(0.0f);
            }
        }
        if (!A.typespec().is_matrix() && B.typespec().is_matrix()) {
            if ((i / 4) != (i % 4)) {
                if (op_is_uniform)
                    a = rop.ll.constant(0.0f);
                else
                    a = rop.ll.wide_constant(0.0f);
            }
        }

        // Perform the op
        llvm::Value* result = 0;
        if (opname == op_lt) {
            result = rop.ll.op_lt(a, b);
        } else if (opname == op_le) {
            result = rop.ll.op_le(a, b);
        } else if (opname == op_eq) {
            result = rop.ll.op_eq(a, b);
        } else if (opname == op_ge) {
            result = rop.ll.op_ge(a, b);
        } else if (opname == op_gt) {
            result = rop.ll.op_gt(a, b);
        } else if (opname == op_neq) {
            result = rop.ll.op_ne(a, b);
        } else {
            // Don't know how to handle this.
            OSL_ASSERT(0 && "Comparison error");
        }
        OSL_ASSERT(result);

        if (final_result) {
            // Combine the component bool based on the op
            if (opname != op_neq)  // final_result &= result
                final_result = rop.ll.op_and(final_result, result);
            else  // final_result |= result
                final_result = rop.ll.op_or(final_result, result);
        } else {
            final_result = result;
        }
    }
    OSL_ASSERT(final_result);

    // Lets not convert comparisons from bool to int

    OSL_DEV_ONLY(
        std::cout
        << "About to rop.storeLLVMValue (final_result, Result, 0, 0); op_is_uniform="
        << op_is_uniform << std::endl);

    OSL_ASSERT(op_is_uniform || !result_is_uniform);

    if (op_is_uniform && !result_is_uniform) {
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
#ifndef OSL_LLVM_OPAQUE_POINTERS
        llvm::Type* resultType = rop.ll.llvm_typeof(
            rop.llvm_get_pointer(Result));
        OSL_ASSERT(
            (resultType
             == reinterpret_cast<llvm::Type*>(rop.ll.type_wide_int_ptr()))
            || (resultType
                == reinterpret_cast<llvm::Type*>(rop.ll.type_int_ptr())));
#endif
        final_result = rop.ll.op_bool_to_int(final_result);
    }

    rop.storeLLVMValue(final_result, Result, 0, 0);
    OSL_DEV_ONLY(
        std::cout << "AFTER to rop.storeLLVMValue (final_result, Result, 0, 0);"
                  << std::endl);

    return true;
}



// int regex_search (string subject, string pattern)
// int regex_search (string subject, int results[], string pattern)
// int regex_match (string subject, string pattern)
// int regex_match (string subject, int results[], string pattern)
LLVMGEN(llvm_gen_regex)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    int nargs = op.nargs();
    OSL_ASSERT(nargs == 3 || nargs == 4);
    Symbol& Result(*rop.opargsym(op, 0));
    Symbol& Subject(*rop.opargsym(op, 1));
    bool do_match_results = (nargs == 4);
    bool fullmatch        = (op.opname() == "regex_match");
    Symbol& Match(*rop.opargsym(op, 2));
    Symbol& Pattern(*rop.opargsym(op, 2 + do_match_results));
    OSL_ASSERT(Result.typespec().is_int() && Subject.typespec().is_string()
               && Pattern.typespec().is_string());
    OSL_ASSERT(!do_match_results
               || (Match.typespec().is_array()
                   && Match.typespec().elementtype().is_int()));

    bool op_is_uniform     = Subject.is_uniform() && Pattern.is_uniform();
    bool result_is_uniform = Result.is_uniform();
    bool match_is_uniform  = (do_match_results && Match.is_uniform());

    BatchedBackendLLVM::TempScope temp_scope(rop);

    std::vector<llvm::Value*> call_args;
    // First arg is ShaderGlobals ptr
    call_args.push_back(rop.sg_void_ptr());

    // Next arg is subject string

    if (!op_is_uniform) {
        call_args.push_back(rop.llvm_void_ptr(Result));
        call_args.push_back(rop.llvm_load_arg(Subject, false, op_is_uniform));
    } else {
        call_args.push_back(rop.llvm_load_value(Subject));
    }

    llvm::Value* temp_match_array = nullptr;
    // Pass the results array and length (just pass 0 if no results wanted).
    if (op_is_uniform && !match_is_uniform) {
        // allocate a temporary to hold the uniform match result
        // then afterwards broadcast it out to the varying match
        temp_match_array
            = rop.getOrAllocateTemp(Match.typespec(), false /*derivs*/,
                                    true /*is_uniform*/, false /*forceBool*/,
                                    "uniform match result");
        call_args.push_back(rop.ll.void_ptr(temp_match_array));
    } else {
        call_args.push_back(rop.llvm_void_ptr(Match));
    }
    if (do_match_results)
        call_args.push_back(rop.ll.constant(Match.typespec().arraylength()));
    else
        call_args.push_back(rop.ll.constant(0));
    // Pass the regex match pattern
    // call_args.push_back (rop.llvm_load_value (Pattern));

    if (!op_is_uniform) {
        call_args.push_back(rop.llvm_load_arg(Pattern, false, op_is_uniform));
    } else {
        call_args.push_back(rop.llvm_load_value(Pattern));
    }

    // Pass whether or not to do the full match
    call_args.push_back(rop.ll.constant(fullmatch));

    FuncSpec func_spec("regex_impl");
    if (!op_is_uniform) {
        func_spec.mask();
        call_args.push_back(rop.ll.mask_as_int(rop.ll.current_mask()));
    }

    llvm::Value* ret = rop.ll.call_function(rop.build_name(func_spec),
                                            call_args);

    if (op_is_uniform) {
        if (!result_is_uniform) {
            ret = rop.ll.widen_value(ret);
        }
        rop.llvm_store_value(ret, Result);
        if (!match_is_uniform) {
            OSL_ASSERT(temp_match_array);
            llvm::Type* elem_type = rop.ll.llvm_type(
                Match.typespec().elementtype().simpletype());
            for (int ai = 0; ai < Match.typespec().arraylength(); ++ai) {
                llvm::Value* elem_ptr  = rop.ll.GEP(elem_type, temp_match_array,
                                                    ai);
                llvm::Value* elem      = rop.ll.op_load(elem_type, elem_ptr);
                llvm::Value* wide_elem = rop.ll.widen_value(elem);
                rop.llvm_store_value(wide_elem, Match, 0 /*deriv*/,
                                     rop.ll.constant(ai) /*arrayindex*/,
                                     0 /* component*/);
            }
        }
    }

    return true;
}



// Construct spatial triple (point, vector, normal), optionally with a
// transformation from a named coordinate system.
LLVMGEN(llvm_gen_construct_triple)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result   = *rop.opargsym(op, 0);
    bool using_space = (op.nargs() == 5);
    Symbol& Space    = *rop.opargsym(op, 1);
    Symbol& X        = *rop.opargsym(op, 1 + using_space);
    Symbol& Y        = *rop.opargsym(op, 2 + using_space);
    Symbol& Z        = *rop.opargsym(op, 3 + using_space);
    OSL_ASSERT(Result.typespec().is_triple() && X.typespec().is_float()
               && Y.typespec().is_float() && Z.typespec().is_float()
               && (using_space == false || Space.typespec().is_string()));

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
    bool op_is_uniform    = X.is_uniform() && Y.is_uniform() && Z.is_uniform()
                         && space_is_uniform;
    bool resultIsUniform = Result.is_uniform();
    OSL_ASSERT(op_is_uniform || !resultIsUniform);

    // First, copy the floats into the vector
    int dmax = Result.has_derivs() ? 3 : 1;
    for (int d = 0; d < dmax; ++d) {   // loop over derivs
        for (int c = 0; c < 3; ++c) {  // loop over components
            const Symbol& comp = *rop.opargsym(op, c + 1 + using_space);
            llvm::Value* val = rop.llvm_load_value(comp, d, NULL, 0, TypeFloat,
                                                   op_is_uniform);

            if (op_is_uniform && !resultIsUniform) {
                rop.llvm_broadcast_uniform_value(val, Result, d, c);
            } else {
                rop.llvm_store_value(val, Result, d, NULL, c);
            }
        }
    }

    // Do the transformation in-place, if called for
    if (using_space) {
        if (Space.is_constant()) {
            ustring from = Space.get_string();
            if (from == Strings::common
                || from == rop.shadingsys().commonspace_synonym())
                return true;  // no transformation necessary
        }
        TypeDesc::VECSEMANTICS vectype = TypeDesc::POINT;
        ustring triple_type("point");
        if (op.opname() == "vector") {
            vectype     = TypeDesc::VECTOR;
            triple_type = ustring("vector");
        } else if (op.opname() == "normal") {
            vectype     = TypeDesc::NORMAL;
            triple_type = ustring("normal");
        }

        OSL_DEV_ONLY(std::cout
                     << "llvm_gen_construct_triple Result.has_derivs()="
                     << Result.has_derivs() << std::endl);

        RendererServices* rend(rop.shadingsys().renderer());

        OSL_ASSERT((false
                    == rend->transform_points(NULL, Strings::_emptystring_,
                                              Strings::_emptystring_, 0.0f,
                                              NULL, NULL, 0, vectype))
                   && "incomplete");
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
        llvm::Value* transform        = rop.temp_wide_matrix_ptr();
        llvm::Value* succeeded_as_int = nullptr;
        {
            llvm::Value* args[]
                = { rop.sg_void_ptr(), rop.ll.void_ptr(transform),
                    space_is_uniform ? rop.llvm_load_value(Space)
                                     : rop.llvm_void_ptr(Space),
                    rop.ll.constant(Strings::common),
                    rop.ll.mask_as_int(rop.ll.current_mask()) };

            // Dynamically build function name
            FuncSpec func_spec("build_transform_matrix");
            func_spec.arg_varying(TypeMatrix);
            func_spec.arg(Space, false /*derivs*/, space_is_uniform);
            func_spec.arg_uniform(TypeString);
            func_spec.mask();

            succeeded_as_int = rop.ll.call_function(rop.build_name(func_spec),
                                                    args);
        }
        {
            llvm::Value* args[] = { rop.llvm_void_ptr(Result /* src */),
                                    rop.llvm_void_ptr(Result /* dest */),
                                    rop.ll.void_ptr(transform),
                                    succeeded_as_int,
                                    rop.ll.mask_as_int(rop.ll.current_mask()) };

            OSL_ASSERT(Result.is_uniform() == false && "unreachable case");
            // definitely not a nonlinear transformation

            // Dynamically build function name
            auto transform_name = llvm::Twine("transform_")
                                  + triple_type.c_str();
            FuncSpec func_spec(transform_name);
            func_spec.arg(Result, Result.has_derivs(), resultIsUniform);
            func_spec.arg(Result, Result.has_derivs(), resultIsUniform);
            func_spec.arg_varying(TypeMatrix);
            func_spec.mask();

            rop.ll.call_function(rop.build_name(func_spec), args);
        }
    }
    return true;
}



/// matrix constructor.  Comes in several varieties:
///    matrix (float)
///    matrix (space, float)
///    matrix (...16 floats...)
///    matrix (space, ...16 floats...)
///    matrix (fromspace, tospace)
LLVMGEN(llvm_gen_matrix)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result        = *rop.opargsym(op, 0);
    int nargs             = op.nargs();
    bool using_space      = (nargs == 3 || nargs == 18);
    bool using_two_spaces = (nargs == 3
                             && rop.opargsym(op, 2)->typespec().is_string());
    int nfloats           = nargs - 1 - (int)using_space;
    OSL_ASSERT(nargs == 2 || nargs == 3 || nargs == 17 || nargs == 18);

    bool result_is_uniform = Result.is_uniform();

    if (using_two_spaces) {
        // Implicit dependencies to shader globals
        // could mean the result needs to be varying
        Symbol& From         = *rop.opargsym(op, 1);
        Symbol& To           = *rop.opargsym(op, 2);
        bool from_is_uniform = From.is_uniform();
        bool to_is_uniform   = To.is_uniform();

        llvm::Value* args[] = { rop.sg_void_ptr(),          // shader globals
                                rop.llvm_void_ptr(Result),  // result
                                from_is_uniform ? rop.llvm_load_value(From)
                                                : rop.llvm_void_ptr(From),
                                to_is_uniform ? rop.llvm_load_value(To)
                                              : rop.llvm_void_ptr(To),
                                rop.ll.mask_as_int(rop.ll.current_mask()) };

        // Dynamically build width suffix
        FuncSpec func_spec("get_from_to_matrix");
        func_spec.arg(Result, result_is_uniform);
        func_spec.arg(From, from_is_uniform);
        func_spec.arg(To, to_is_uniform);
        // Because we want to mask off potentially expensive scalar
        // non-affine matrix inversion, we will always call a masked version
        func_spec.mask();

        rop.ll.call_function(rop.build_name(func_spec), args);
    } else {
        if (nfloats == 1) {
            llvm::Value* zero;
            if (result_is_uniform)
                zero = rop.ll.constant(0.0f);
            else
                zero = rop.ll.wide_constant(0.0f);

            for (int i = 0; i < 16; i++) {
                llvm::Value* src_val
                    = ((i % 4) == (i / 4))
                          ? rop.llvm_load_value(*rop.opargsym(op,
                                                              1 + using_space),
                                                0, 0, TypeDesc::UNKNOWN,
                                                result_is_uniform)
                          : zero;
                rop.llvm_store_value(src_val, Result, 0, i);
            }
        } else if (nfloats == 16) {
            for (int i = 0; i < 16; i++) {
                llvm::Value* src_val = rop.llvm_load_value(
                    *rop.opargsym(op, i + 1 + using_space), 0, 0,
                    TypeDesc::UNKNOWN, result_is_uniform);
                rop.llvm_store_value(src_val, Result, 0, i);
            }
        } else {
            OSL_ASSERT(0);
        }
        if (using_space) {
            // Implicit dependencies to shader globals
            // could mean the result needs to be varying
            Symbol& From = *rop.opargsym(op, 1);
            // Avoid the prepend call if the from space is common which
            // would be identity matrix.
            if (!From.is_constant()
                || (From.get_string() != Strings::common
                    && From.get_string()
                           != rop.shadingsys().commonspace_synonym())) {
                bool from_is_uniform = From.is_uniform();
                llvm::Value* args[]
                    = { rop.sg_void_ptr(),          // shader globals
                        rop.llvm_void_ptr(Result),  // result
                        from_is_uniform ? rop.llvm_load_value(From)
                                        : rop.llvm_void_ptr(From),
                        rop.ll.mask_as_int(rop.ll.current_mask()) };

                // Dynamically build width suffix
                FuncSpec func_spec("prepend_matrix_from");
                func_spec.arg(Result, result_is_uniform);
                func_spec.arg(From, from_is_uniform);
                // Because we want to mask off potentially expensive calls to
                // renderer services to lookup matrices,  we will always call a masked version
                func_spec.mask();

                rop.ll.call_function(rop.build_name(func_spec), args);
            }
        }
    }
    if (Result.has_derivs())
        rop.llvm_zero_derivs(Result);
    return true;
}



/// int getmatrix (fromspace, tospace, M)
LLVMGEN(llvm_gen_getmatrix)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    int nargs = op.nargs();
    OSL_ASSERT(nargs == 4);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& From   = *rop.opargsym(op, 1);
    Symbol& To     = *rop.opargsym(op, 2);
    Symbol& M      = *rop.opargsym(op, 3);


    // Implicit dependencies to shader globals
    // could mean the result needs to be varying
    bool result_is_uniform = Result.is_uniform();
    OSL_ASSERT(M.is_uniform() == result_is_uniform);

    bool from_is_uniform = From.is_uniform();
    bool to_is_uniform   = To.is_uniform();

    llvm::Value* args[] = {
        rop.sg_void_ptr(),     // shader globals
        rop.llvm_void_ptr(M),  // matrix result
        from_is_uniform ? rop.llvm_load_value(From) : rop.llvm_void_ptr(From),
        to_is_uniform ? rop.llvm_load_value(To) : rop.llvm_void_ptr(To),
        rop.ll.mask_as_int(rop.ll.current_mask())
    };

    FuncSpec func_spec("get_from_to_matrix");
    func_spec.arg(M, result_is_uniform);
    func_spec.arg(From, from_is_uniform);
    func_spec.arg(To, to_is_uniform);
    // Because we want to mask off potentially expensive scalar
    // non-affine matrix inversion, we will always call a masked version
    func_spec.mask();

    llvm::Value* result = rop.ll.call_function(rop.build_name(func_spec), args);
    rop.llvm_conversion_store_masked_status(result, Result);
    rop.llvm_zero_derivs(M);
    return true;
}



// transform{,v,n} (string tospace, triple p)
// transform{,v,n} (string fromspace, string tospace, triple p)
// transform{,v,n} (matrix, triple p)
LLVMGEN(llvm_gen_transform)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    int nargs      = op.nargs();
    Symbol* Result = rop.opargsym(op, 0);
    Symbol* From   = (nargs == 3) ? NULL : rop.opargsym(op, 1);
    Symbol* To     = rop.opargsym(op, (nargs == 3) ? 1 : 2);
    Symbol* P      = rop.opargsym(op, (nargs == 3) ? 2 : 3);

    bool result_is_uniform = Result->is_uniform();
    bool to_is_uniform     = To->is_uniform();
    bool P_is_uniform      = P->is_uniform();
    bool from_is_uniform   = (From == NULL) ? true : From->is_uniform();

    TypeDesc::VECSEMANTICS vectype = TypeDesc::POINT;
    // TODO: switch statement with static/extern strings to avoid lookup
    ustring triple_type("point");
    if (op.opname() == "transformv") {
        vectype     = TypeDesc::VECTOR;
        triple_type = ustring("vector");
    } else if (op.opname() == "transformn") {
        vectype     = TypeDesc::NORMAL;
        triple_type = ustring("normal");
    }

    llvm::Value* transform        = nullptr;
    llvm::Value* succeeded_as_int = nullptr;
    if (To->typespec().is_matrix()) {
        OSL_ASSERT(From == NULL);
        // llvm_ops has the matrix version already implemented
        //llvm_gen_generic (rop, opnum);
        //return true;
        transform        = rop.llvm_void_ptr(*To);
        succeeded_as_int = rop.ll.mask_as_int(rop.ll.current_mask());
    } else {
        // Named space versions from here on out.
        if ((From == NULL || From->is_constant()) && To->is_constant()) {
            // We can know all the space names at this time
            ustring from = From ? From->get_string() : Strings::common;
            ustring to   = To->get_string();
            ustring syn  = rop.shadingsys().commonspace_synonym();
            if (from == syn)
                from = Strings::common;
            if (to == syn)
                to = Strings::common;
            if (from == to) {
                // An identity transformation, just copy
                if (Result != P)  // don't bother in-place copy
                    rop.llvm_assign_impl(*Result, *P);
                return true;
            }
        }
        //OSL_DEV_ONLY(std::cout << "wide transform 'source space' = " << from << " 'dest space' = " << to << std::endl);

        RendererServices* rend(rop.shadingsys().renderer());

        OSL_ASSERT((false
                    == rend->transform_points(NULL, Strings::_emptystring_,
                                              Strings::_emptystring_, 0.0f,
                                              NULL, NULL, 0, vectype))
                   && "incomplete");
        // TODO:  Didn't want to make RenderServices have to deal will all variants of from/to
        // unless it is going to be used, yes it will have to be done though
        //    if (rend->transform_points (NULL, from, to, 0.0f, NULL, NULL, 0, vectype)) {
        //
        //        // TODO: Handle non-uniform case below minding mask values
        //        OSL_ASSERT(Result->is_uniform());
        //        OSL_ASSERT(0 && "incomplete"); // needs uniform version accepting BatchedShaderGlobals
        //
        //        // renderer potentially knows about a nonlinear transformation.
        //        // Note that for the case of non-constant strings, passing empty
        //        // from & to will make transform_points just tell us if ANY
        //        // nonlinear transformations potentially are supported.
        //        rop.ll.call_function ("osl_transform_triple_nonlinear", args, 8);
        //    } else
        transform = rop.temp_wide_matrix_ptr();
        {
            OSL_ASSERT(
                From != NULL
                && "expect NULL was replaced by constant folding to a common_space");
            llvm::Value* args[] = { rop.sg_void_ptr(),
                                    rop.ll.void_ptr(transform),
                                    from_is_uniform ? rop.llvm_load_value(*From)
                                                    : rop.llvm_void_ptr(*From),
                                    to_is_uniform ? rop.llvm_load_value(*To)
                                                  : rop.llvm_void_ptr(*To),
                                    rop.ll.mask_as_int(rop.ll.current_mask()) };

            FuncSpec func_spec("build_transform_matrix");
            func_spec.arg_varying(TypeMatrix);
            // Ignore derivatives if unneeded or unsupplied
            func_spec.arg(*From, from_is_uniform);
            func_spec.arg(*To, to_is_uniform);
            func_spec.mask();

            succeeded_as_int = rop.ll.call_function(rop.build_name(func_spec),
                                                    args);
        }
        // The results of looking up a transform are always wide
    }
    {
        if (result_is_uniform) {
            OSL_ASSERT(to_is_uniform);
            OSL_ASSERT(P_is_uniform);

            llvm::Value* args[] = { rop.llvm_void_ptr(*Result),
                                    rop.ll.void_ptr(transform),
                                    rop.llvm_void_ptr(*P) };

            // Dynamically build function name
            FuncSpec func_spec(op.opname().c_str());
            func_spec.unbatch();
            //std::string func_name = std::string("osl_") + op.opname().c_str() + "_";
            // Ignore derivatives if unneeded or unsupplied
            bool has_derivs = (Result->has_derivs() && P->has_derivs());
            func_spec.arg(*P, has_derivs, P_is_uniform);
            // The matrix is always varying if we looked it up,
            // if it was passed directly in "To", then we respect to's uniformity
            // otherwise it will be the varying result of the callback to the renderer
            func_spec.arg(TypeMatrix,
                          To->typespec().is_matrix() ? to_is_uniform : false);
            func_spec.arg(*Result, has_derivs, result_is_uniform);

            rop.ll.call_function(rop.build_name(func_spec), args);
        } else {
            llvm::Value* args[]
                = { rop.llvm_void_ptr(*P), rop.llvm_void_ptr(*Result),
                    rop.ll.void_ptr(transform), succeeded_as_int,
                    rop.ll.mask_as_int(rop.ll.current_mask()) };

            // definitely not a nonlinear transformation

            auto func_name = llvm::Twine("transform_") + triple_type.c_str();
            FuncSpec func_spec(func_name);
            // Ignore derivatives if unneeded or not supplied
            // NOTE: odd case where P is uniform but still reported as having
            // derivatives.  Choose to ignore uniform derivatives
            bool has_derivs = (Result->has_derivs()
                               && (P->has_derivs() && !P_is_uniform));
            func_spec.arg(*P, has_derivs, P_is_uniform);
            func_spec.arg(*Result, has_derivs, result_is_uniform);
            // The matrix is always varying if we looked it up,
            // if it was passed directly in "To", then we respect to's uniformity
            // otherwise it will be the varying result of the callback to the renderer
            func_spec.arg(TypeMatrix,
                          To->typespec().is_matrix() ? to_is_uniform : false);
            func_spec.mask();

            rop.ll.call_function(rop.build_name(func_spec), args);
        }

        // To reduce the number of combinations to support
        // we take on the work of zero'ing out the derivatives here
        // versus adding another version of the functions that just
        // zeros them out.
        // NOTE:  the original scalar version 0's out derivatives
        // regardless of the success of the transformation
        // however the operation mask should still be respected
        // NOTE: odd case where P is uniform but still reported as having
        // derivatives.  Choose to ignore uniform derivatives
        if (Result->has_derivs() && (!P->has_derivs() || P_is_uniform)) {
            rop.llvm_zero_derivs(*Result);
        }
    }
    return true;
}



// transformc (string fromspace, string tospace, color p)
LLVMGEN(llvm_gen_transformc)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_ASSERT(op.nargs() == 4);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& From   = *rop.opargsym(op, 1);
    Symbol& To     = *rop.opargsym(op, 2);
    Symbol& C      = *rop.opargsym(op, 3);

    bool op_is_uniform = C.is_uniform() && From.is_uniform() && To.is_uniform();
    bool result_is_uniform = Result.is_uniform();

    BatchedBackendLLVM::TempScope temp_scope(rop);
    FuncSpec func_spec("transform_color");
    func_spec.arg(C, false, op_is_uniform);
    // We will only call a transform library with uniform 'From'
    // and 'To' spaces.  If our 'From' and 'To' are not uniform we will
    // loop identifying which lanes have the same 'From' and 'To' values and
    // call the library function multiple times with different combinations
    // of 'From' and 'To' and different masks
    func_spec.arg(From, false, true /*is_uniform*/);
    func_spec.arg(To, false, true /*is_uniform*/);
    if (!op_is_uniform) {
        OSL_ASSERT(!result_is_uniform);
        func_spec.mask();

        const char* transformColorFuncName = rop.build_name(func_spec);

        if (From.is_uniform() && To.is_uniform()) {
            // When both From and To are uniform, we can just
            // skip the loop and call the library function once
            llvm::Value* args[]
                = { rop.sg_void_ptr(),
                    rop.llvm_load_arg(C, C.has_derivs(), op_is_uniform),
                    rop.ll.constant(C.has_derivs()),
                    rop.llvm_void_ptr(Result),
                    rop.ll.constant(Result.has_derivs()),
                    rop.llvm_load_arg(From, false /*has_derivs*/,
                                      true /*is_uniform*/),
                    rop.llvm_load_arg(To, false /*has_derivs*/,
                                      true /*is_uniform*/),
                    rop.ll.mask_as_int(rop.ll.current_mask()) };
            rop.ll.call_function(transformColorFuncName, args);
        } else {
            // From or To must be varying, setup a loop to call the library
            // function with uniform from and to values adjusting the mask
            // to identify which lanes match the from and to combinations.
            llvm::Value* FromVal
                = rop.llvm_load_value(From, 0 /*deriv*/, 0 /*component*/,
                                      TypeDesc::UNKNOWN, From.is_uniform());
            llvm::Value* ToVal
                = rop.llvm_load_value(To, 0 /*deriv*/, 0 /*component*/,
                                      TypeDesc::UNKNOWN, To.is_uniform());

            llvm::Value* args[] = {
                rop.sg_void_ptr(),
                rop.llvm_load_arg(C, C.has_derivs(), op_is_uniform),
                rop.ll.constant(C.has_derivs()),
                rop.llvm_void_ptr(Result),
                rop.ll.constant(Result.has_derivs()),
                From.is_uniform() ? FromVal : nullptr /* filled in by loop */,
                To.is_uniform() ? ToVal : nullptr /* filled in by loop */,
                nullptr /*mask filled in by loop*/
            };
            constexpr int fromArgumentIndex = 5;
            constexpr int toArgumentIndex   = 6;
            constexpr int maskArgumentIndex = 7;

            // do while(remaining)
            llvm::Value* loc_of_remainingMask = rop.getTempMask(
                "lanes remaining to transformc");
            rop.ll.op_store_mask(rop.ll.current_mask(), loc_of_remainingMask);
            llvm::BasicBlock* bin_block = rop.ll.new_basic_block(
                rop.llvm_debug()
                    ? std::string("bin_transformc (varying spaces)")
                    : std::string());
            llvm::BasicBlock* after_block = rop.ll.new_basic_block(
                rop.llvm_debug()
                    ? std::string("after_bin_transformc (varying spaces)")
                    : std::string());
            rop.ll.op_branch(bin_block);
            {
                llvm::Value* remainingMask = rop.ll.op_load_mask(
                    loc_of_remainingMask);
                llvm::Value* leadLane = rop.ll.op_1st_active_lane_of(
                    remainingMask);
                llvm::Value* lanesMatchingSpaces = remainingMask;

                if (!From.is_uniform()) {
                    llvm::Value* scalar_fromSpace = rop.ll.op_extract(FromVal,
                                                                      leadLane);
                    args[fromArgumentIndex]       = scalar_fromSpace;
                    lanesMatchingSpaces = rop.ll.op_lanes_that_match_masked(
                        scalar_fromSpace, FromVal, remainingMask);
                    OSL_DASSERT(lanesMatchingSpaces);
                }

                if (!To.is_uniform()) {
                    llvm::Value* scalar_toSpace = rop.ll.op_extract(ToVal,
                                                                    leadLane);
                    args[toArgumentIndex]       = scalar_toSpace;
                    lanesMatchingSpaces = rop.ll.op_lanes_that_match_masked(
                        scalar_toSpace, ToVal, lanesMatchingSpaces);
                    OSL_DASSERT(lanesMatchingSpaces);
                }

                //rop.llvm_print_mask("lanesMatchingSpaces", lanesMatchingSpaces);
                args[maskArgumentIndex] = rop.ll.mask_as_int(
                    lanesMatchingSpaces);

                rop.ll.call_function(transformColorFuncName, args);

                remainingMask = rop.ll.op_xor(remainingMask,
                                              lanesMatchingSpaces);
                //rop.llvm_print_mask("xor remainingMask,lanesMatchingSpaces", remainingMask);
                rop.ll.op_store_mask(remainingMask, loc_of_remainingMask);

                llvm::Value* int_remainingMask = rop.ll.mask_as_int(
                    remainingMask);
                //rop.llvm_print_mask("remainingMask", remainingMask);
                llvm::Value* cond_more_lanes_to_bin
                    = rop.ll.op_ne(int_remainingMask, rop.ll.constant(0));
                rop.ll.op_branch(cond_more_lanes_to_bin, bin_block,
                                 after_block);
            }
            // Continue on with the previous flow
            rop.ll.set_insert_point(after_block);
        }
    } else {
        // All inputs (Color, From, To) are uniform, but result might be varying.
        // In that case still call the uniform version of the library function,
        // but allocate a temp for the result and after library call,
        // broadcast the temp result out to the varying result symbol
        llvm::Value* result_param = nullptr;
        if (!result_is_uniform) {
            result_param = rop.getOrAllocateTemp(Result.typespec(),
                                                 Result.has_derivs(),
                                                 true /*is_uniform*/);
        } else {
            result_param = rop.llvm_void_ptr(Result);
        }

        llvm::Value* args[7] = {
            rop.sg_void_ptr(),
            rop.llvm_load_arg(C, C.has_derivs(), true /*op_is_uniform*/),
            rop.ll.constant(C.has_derivs()),
            rop.ll.void_ptr(result_param),
            rop.ll.constant(Result.has_derivs()),
            rop.llvm_load_arg(From, false /*has_derivs*/,
                              true /*op_is_uniform*/),
            rop.llvm_load_arg(To, false /*has_derivs*/, true /*op_is_uniform*/)
        };

        func_spec.unmask();
        rop.ll.call_function(rop.build_name(func_spec), args);

        if (!result_is_uniform) {
            rop.llvm_broadcast_uniform_value_from_mem(result_param, Result);
        }
    }
    return true;
}



LLVMGEN(llvm_gen_loop_op)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& cond          = *rop.opargsym(op, 0);
    bool op_is_uniform    = cond.is_uniform();
    const char* cond_name = cond.name().c_str();

    if (op_is_uniform) {
        OSL_DEV_ONLY(std::cout << "llvm_gen_loop_op UNIFORM based on "
                               << cond.name().c_str() << std::endl);

        // Branch on the condition, to our blocks
        llvm::BasicBlock* cond_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("cond (uniform)") + cond_name
                             : std::string());
        llvm::BasicBlock* body_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("body (uniform)") + cond_name
                             : std::string());
        llvm::BasicBlock* step_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("step (uniform)") + cond_name
                             : std::string());
        llvm::BasicBlock* after_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("after_loop (uniform)") + cond_name
                             : std::string());
        // Save the step and after block pointers for possible break/continue
        rop.ll.push_loop(step_block, after_block);
        // We need to track uniform loops as well
        // to properly handle a uniform loop inside of a varying loop
        // and since the "break" op has no symbol for us to check for
        // uniformity, it can check the current masked loop condition location
        // to see if it is null or not (uniform vs. varying)
        rop.ll.push_masked_loop(nullptr, nullptr);

        // Initialization (will be empty except for "for" loops)
        rop.build_llvm_code(opnum + 1, op.jump(0));

        // For "do-while", we go straight to the body of the loop, but for
        // "for" or "while", we test the condition next.
        rop.ll.op_branch(op.opname() == op_dowhile ? body_block : cond_block);

        // Load the condition variable and figure out if it's nonzero
        rop.build_llvm_code(op.jump(0), op.jump(1), cond_block);
        llvm::Value* cond_val = rop.llvm_test_nonzero(cond);

        // Jump to either LoopBody or AfterLoop
        rop.ll.op_branch(cond_val, body_block, after_block);

        // Body of loop
        rop.build_llvm_code(op.jump(1), op.jump(2), body_block);
        rop.ll.op_branch(step_block);

        // Step
        rop.build_llvm_code(op.jump(2), op.jump(3), step_block);
        rop.ll.op_branch(cond_block);

        // Continue on with the previous flow
        rop.ll.set_insert_point(after_block);
        rop.ll.pop_masked_loop();
        rop.ll.pop_loop();
    } else {
        OSL_DEV_ONLY(std::cout << "llvm_gen_loop_op VARYING based on "
                               << cond.name().c_str() << std::endl);

        // make sure that any temps created for control flow
        // are not released until we are done using them!
        BatchedBackendLLVM::TempScope temp_scope(rop);

        // Branch on the condition, to our blocks
        llvm::BasicBlock* cond_block;
        llvm::BasicBlock* body_block;
        // Improve readability of generated IR by creating basic blocks in the order they
        // will be processed
        if (op.opname() == op_dowhile) {
            body_block = rop.ll.new_basic_block(
                rop.llvm_debug() ? std::string("body (varying):") + cond_name
                                 : std::string());
            cond_block = rop.ll.new_basic_block(
                rop.llvm_debug() ? std::string("cond (varying):") + cond_name
                                 : std::string());
        } else {
            cond_block = rop.ll.new_basic_block(
                rop.llvm_debug() ? std::string("cond (varying):") + cond_name
                                 : std::string());
            body_block = rop.ll.new_basic_block(
                rop.llvm_debug() ? std::string("body (varying):") + cond_name
                                 : std::string());
        }
        llvm::BasicBlock* step_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("step (varying):") + cond_name
                             : std::string());
        llvm::BasicBlock* after_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("after_loop (varying):") + cond_name
                             : std::string());

        int return_count_before_loop = rop.ll.masked_return_count();

        // Save the step and after block pointers for possible break/continue
        rop.ll.push_loop(step_block, after_block);

        // The analysis flag for loop Opcodes
        // indicates if the loop contains a continue.
        // NOTE: BatchedAnalysis populates the analysis_flag.
        bool loopHasContinue = op.analysis_flag();
        // TODO: possible optimization when the condition has been forced to be a bool, that we
        // might be able to use the condition as the control mask, but only if that condition is never
        // used after the loop as the control mask will be all 0's at the end of the loop which might
        // be different value that the condition symbol should be.
        llvm::Value* loc_of_control_mask = rop.getTempMask(
            std::string("control flow mask:") + cond_name);
        llvm::Value* loc_of_continue_mask
            = loopHasContinue
                  ? rop.getTempMask(std::string("continue mask:") + cond_name)
                  : nullptr;

        rop.ll.push_masked_loop(loc_of_control_mask, loc_of_continue_mask);

        // Initialization (will be empty except for "for" loops)
        rop.build_llvm_code(opnum + 1, op.jump(0));

        // Store current top of the mask stack (or all 1's) as the current mask value
        // as we enter the loop
        llvm::Value* initial_mask = rop.ll.current_mask();
        rop.ll.op_store_mask(initial_mask, loc_of_control_mask);

        // If all lanes inside the loop become inactive,
        // jump to the step as it may have been cause by a continue.
        // If no continue is possible, then we can just jump to the
        // after_block when all lanes become inactive
        rop.ll.push_masked_return_block(loopHasContinue ? step_block
                                                        : after_block);

        // For "do-while", we go straight to the body of the loop, but for
        // "for" or "while", we test the condition next.
        if (op.opname() == op_dowhile) {
            rop.ll.op_branch(body_block);

            llvm::Value* pre_condition_mask = rop.ll.op_load_mask(
                loc_of_control_mask);
            OSL_ASSERT(pre_condition_mask->getType()
                       == rop.ll.type_wide_bool());

            rop.ll.push_mask(pre_condition_mask, false /* negate */,
                             true /* absolute */);
#ifdef __OSL_TRACE_MASKS
            rop.llvm_print_mask("pre_condition_mask");
#endif

            // Body of loop
            // We need to zero out the continue mask at the top loop body, as the previous
            // iteration could have set continue.
            // TODO, move allocation of continue mask inside the loop body to minimize its
            // scope, although it is still a loop resource perhaps we can delay
            // setting it until now
            if (loopHasContinue) {
                rop.ll.op_store_mask(rop.ll.wide_constant_bool(false),
                                     loc_of_continue_mask);
            }

            rop.build_llvm_code(op.jump(1), op.jump(2), body_block);
            rop.ll.op_branch(step_block);

            // Step
            // The step shares the same mask as the body, unless a continue was called
            if (rop.ll.masked_continue_count() > 0) {
                //std::cout << "(rop.ll.masked_continue_count() > 0) == true\n";
                // Get rid of any modified mask that had the continue mask applied to it
                rop.ll.pop_mask();
                // Restore the condition mask for the step to execute with
                llvm::Value* pre_step_mask = pre_condition_mask;
                // We are trying to reuse the conditional loaded before the body
                // executes, however a 'break' would have written to that conditional mask
                // In that case, we need to reload the mask
                if (rop.ll.masked_break_count() > 0) {
                    pre_step_mask = rop.ll.op_load_mask(loc_of_control_mask);
                    // The break could have caused all lanes to be 0,
                    // If there was no continue that would have jumped to the after block already.
                    // But we are here because perhaps some lanes were 0 because of the continue.
                    // Reloading the condition variable will not contain any continued lanes.
                    // So we can test it to see if any lanes are active. If not,
                    // we don't want to execute the condition block as it might contain function calls
                    // or use param which calls down to subsequent layers.
                    // So we will test to see if any lanes are active
                    llvm::Value* anyLanesActive
                        = rop.ll.test_if_mask_is_non_zero(pre_step_mask);
                    llvm::BasicBlock* some_lanes_active_after_continue_break
                        = rop.ll.new_basic_block(
                            rop.llvm_debug()
                                ? std::string(
                                      "some_lanes_active_after_continue_break (varying)")
                                      + cond_name
                                : std::string());

                    rop.ll.op_branch(anyLanesActive,
                                     some_lanes_active_after_continue_break,
                                     after_block);
                }
                rop.ll.push_mask(pre_step_mask, false /* negate */,
                                 true /* absolute */);
#ifdef __OSL_TRACE_MASKS
                rop.llvm_print_mask("pre_step_mask");
#endif
            }
            OSL_ASSERT(op.jump(2) == op.jump(3));
            // why bother building empty step
            //rop.build_llvm_code (op.jump(2), op.jump(3), step_block);
            rop.ll.op_branch(cond_block);

            // Load the condition variable and figure out if it's nonzero
            // The step shares the same mask as the step
            rop.build_llvm_code(op.jump(0), op.jump(1), cond_block);
            rop.ll.pop_mask();
            // Here we will look at the actual conditional symbol (vs. the loop control)
            // and store it to the loop control mask, if necessary
            llvm::Value* post_condition_mask = rop.llvm_load_mask(cond);
            post_condition_mask = rop.ll.op_and(post_condition_mask,
                                                pre_condition_mask);

            // if a return could have been
            // executed, we need to mask out those lanes from the conditional symbol
            // because the step function would have executed with those lanes off
            // causing an endless loop
            // No need to handle break here, if encountered, it was immediately applied to the condition mask
            if (rop.ll.masked_return_count() > return_count_before_loop) {
                post_condition_mask = rop.ll.apply_return_to(
                    post_condition_mask);
            }

            // we need to store the masked conditional result to the control flow mask
            rop.ll.op_store_mask(post_condition_mask, loc_of_control_mask);
            llvm::Value* cond_val = rop.ll.test_if_mask_is_non_zero(
                post_condition_mask);

            // Jump to either LoopBody or AfterLoop
            rop.ll.op_branch(cond_val, body_block, after_block);

        } else {
            rop.ll.op_branch(cond_block);

            llvm::Value* pre_condition_mask = rop.ll.op_load_mask(
                loc_of_control_mask);
            OSL_ASSERT(pre_condition_mask->getType()
                       == rop.ll.type_wide_bool());

            rop.ll.push_mask(pre_condition_mask, false /* negate */,
                             true /* absolute */);
            rop.build_llvm_code(op.jump(0), op.jump(1), cond_block);
            rop.ll.pop_mask();
            // Load the condition variable and figure out if it's nonzero
            // Here we will look at the actual conditional symbol (vs. the loop control)
            // and store it to the loop control mask, if necessary
            llvm::Value* post_condition_mask = rop.llvm_load_mask(cond);
            post_condition_mask = rop.ll.op_and(post_condition_mask,
                                                pre_condition_mask);
            // we need to store the masked conditional result to the control flow mask
            rop.ll.op_store_mask(post_condition_mask, loc_of_control_mask);

            // The condition was initialized with the current_mask before the loop
            // and considered an absolute value, therefore should be OK to test directly
            llvm::Value* cond_val = rop.ll.test_if_mask_is_non_zero(
                post_condition_mask);

            // Jump to either LoopBody or AfterLoop
            rop.ll.op_branch(cond_val, body_block, after_block);

            // Body of loop
            rop.ll.push_mask(post_condition_mask, false /* negate */,
                             true /* absolute */);
            // We need to zero out the continue mask at the top loop body, as the previous
            // iteration could have set continue, alternatively we could zero at the end
            // of the loop body so its ready for the next iteration, perhaps as part
            // of the step, but if we know we will need it simplest to do at top of loop body
            // TODO, move allocation of continue mask inside the loop body to minimize its
            // scope, although it is still a loop resource perhaps we can delay
            // setting it until now
            if (loopHasContinue) {
                rop.ll.op_store_mask(rop.ll.wide_constant_bool(false),
                                     loc_of_continue_mask);
            }
            rop.build_llvm_code(op.jump(1), op.jump(2), body_block);

            rop.ll.op_branch(step_block);

            // Step
            // The step shares the same mask as the body, unless a continue was called
            if (rop.ll.masked_continue_count() > 0) {
                // Get rid of any modified mask that had the continue mask applied to it
                rop.ll.pop_mask();
                // Restore the condition mask for the step to execute with
                llvm::Value* pre_step_mask = post_condition_mask;
                // We are trying to reuse the conditional loaded before the body
                // executes, however a 'break' would have written to that conditional mask
                // In that case, we need to reload the mask
                if (rop.ll.masked_break_count() > 0) {
                    pre_step_mask = rop.ll.op_load_mask(loc_of_control_mask);
                }
                rop.ll.push_mask(pre_step_mask, false /* negate */,
                                 true /* absolute */);
#ifdef __OSL_TRACE_MASKS
                rop.llvm_print_mask("pre_step_mask");
#endif
            }
            rop.build_llvm_code(op.jump(2), op.jump(3), step_block);
            rop.ll.pop_mask();

            // before we jump back to the condition block, if a return could have been
            // executed, we need to mask out those lanes from the conditional symbol
            // because the step function would have executed with those lanes off
            // causing an endless loop
            // No need to handle break here, if encountered, it was immediately applied to the condition mask
            if (rop.ll.masked_return_count() > return_count_before_loop) {
                // We are trying to reuse the conditional loaded before the body
                // executes, however a 'break' would have written to that conditional mask
                // In that case, we need to reload the mask
                if (rop.ll.masked_break_count() > 0) {
                    post_condition_mask = rop.ll.op_load_mask(
                        loc_of_control_mask);
                }
                llvm::Value* post_step_mask = rop.ll.apply_return_to(
                    post_condition_mask);
                rop.ll.op_store_mask(post_step_mask, loc_of_control_mask);
            }
            rop.ll.op_branch(cond_block);
        }
        rop.ll.pop_masked_loop();
        rop.ll.pop_loop();

        // Continue on with the previous flow
        rop.ll.set_insert_point(after_block);

        rop.ll.pop_masked_return_block();

        if (rop.ll.masked_return_count() > return_count_before_loop) {
            // Inside the loop a return may have been executed
            // we need to update the current mask to reflect the disabled lanes
            // We needed to wait until were were in the after block so the produced
            // mask is available to subsequent instructions
            rop.ll.apply_return_to_mask_stack();

            // through a combination of the return mask and any lanes conditionally
            // masked off, all lanes could be 0 at this point and we wouldn't
            // want to call down to any layers at this point

            // NOTE: testing the return/exit masks themselves is not sufficient
            // as some lanes may be disabled by the conditional mask stack

            // TODO: do we want a test routine that can handle negated masks?
            llvm::Value* anyLanesActive = rop.ll.test_if_mask_is_non_zero(
                rop.ll.current_mask());

            llvm::BasicBlock* nextMaskScope;
            if (rop.ll.has_masked_return_block()) {
                nextMaskScope = rop.ll.masked_return_block();
            } else {
                nextMaskScope = rop.ll.inside_function()
                                    ? rop.ll.return_block()
                                    : rop.llvm_exit_instance_block();
            }
            llvm::BasicBlock* after_applying_return_block
                = rop.ll.new_basic_block(
                    rop.llvm_debug()
                        ? std::string("after_loop_applied_return_mask (varying)")
                              + cond_name
                        : std::string());
            rop.ll.op_branch(anyLanesActive, after_applying_return_block,
                             nextMaskScope);
        }
    }
    return true;
}



LLVMGEN(llvm_gen_loopmod_op)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 0);

    bool inside_masked_loop = rop.ll.is_innermost_loop_masked();
    if (false == inside_masked_loop) {
        // Inside a uniform loop, can use branching
        if (op.opname() == op_break) {
            rop.ll.op_branch(rop.ll.loop_after_block());
        } else {  // continue
            rop.ll.op_branch(rop.ll.loop_step_block());
        }
        llvm::BasicBlock* next_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? "next_block" : std::string());
        rop.ll.set_insert_point(next_block);
    } else {
        if (op.opname() == op_break) {
            // Inside a varying loop, can not only branch
            // must mask off additional lanes for remainder of loop
            // We can just take the absolute mask that is executing the 'break'
            // instruction and store an absolute modified mask to the
            // condition variable (which the conditional block of the loop
            // will hopefully pickup and use)
            // Trick is we then will need to pop and push a different mask
            // back on the stack for the remainder of the loop body.
            rop.ll.op_masked_break();
            // But there may still be more instructions in the body after the break
            // Rely on front end dead code elimination to remove any instructions
            // after a break.
        } else {
            OSL_ASSERT(op.opname() == op_continue);
            // Inside a varying loop, can not only branch
            // must mask off additional lanes for remainder of loop
            // We can just take the absolute mask that is executing the 'break'
            // instruction and store an absolute modified mask to the
            // condition variable (which the conditional block of the loop
            // will hopefully pickup and use)
            // Trick is we then will need to pop and push a different mask
            // back on the stack for the remainder of the loop body.
            rop.ll.op_masked_continue();
            // But there may still be more instructions in the body after the break
            // Rely on front end dead code elimination to remove any instructions
            // after a break.
        }
    }
    return true;
}



static llvm::Value*
llvm_batched_texture_options(BatchedBackendLLVM& rop, int opnum,
                             int first_optional_arg, bool tex3d, int nchans,
                             llvm::Value*& alpha, bool& alphaHasDerivs,
                             llvm::Value*& errormessage,
                             llvm::Value*& missingcolor_buffer)
{
    llvm::Value* bto     = rop.temp_batched_texture_options_ptr();
    llvm::Type* bto_type = rop.llvm_type_batched_texture_options();

    // The BatchedTextureOptions & missingcolor_buffer are local alloca,
    // so no need to mask off non-active lanes

    // Explicitly assign a default value or an optional parameter value to every data member
    // of BatchedTextureOptions.
    llvm::Value* wide_const_fzero_value = rop.ll.wide_constant(0.0f);
    llvm::Value* wide_const_fone_value  = rop.ll.wide_constant(1.0f);
    llvm::Value* const_zero_value       = rop.ll.constant(0);
    llvm::Value* wrap_default_value     = rop.ll.constant(
        static_cast<int>(Tex::Wrap::Default));

    llvm::Value* sblur  = wide_const_fzero_value;
    llvm::Value* tblur  = wide_const_fzero_value;
    llvm::Value* rblur  = wide_const_fzero_value;
    llvm::Value* swidth = wide_const_fone_value;
    llvm::Value* twidth = wide_const_fone_value;
    llvm::Value* rwidth = wide_const_fone_value;
    // TODO: llvm_gen is not yet populating rnd, so neither will the batched
    //       version.  But below is where we would do so
    llvm::Value* rnd = wide_const_fzero_value;

    llvm::Value* firstchannel = const_zero_value;
    llvm::Value* subimage     = const_zero_value;
    llvm::Value* subimagename = rop.ll.constant_ptr(nullptr);
    llvm::Value* swrap        = wrap_default_value;
    llvm::Value* twrap        = wrap_default_value;
    llvm::Value* rwrap        = wrap_default_value;
    llvm::Value* mipmode      = rop.ll.constant(
        static_cast<int>(Tex::MipMode::Default));
    llvm::Value* interpmode = rop.ll.constant(
        static_cast<int>(Tex::InterpMode::SmartBicubic));
    llvm::Value* anisotropic         = rop.ll.constant(32);
    llvm::Value* conservative_filter = rop.ll.constant(1);
    llvm::Value* fill                = rop.ll.constant(0.0f);

    bool is_swrap_uniform = true;
    bool is_twrap_uniform = true;
    bool is_rwrap_uniform = true;

    bool is_fill_uniform         = true;
    bool is_firstchannel_uniform = true;
    bool is_subimage_uniform     = true;
    bool is_subimagename_uniform = true;
    bool is_interpmode_uniform   = true;

    llvm::Value* missingcolor = rop.ll.constant_ptr(nullptr,
                                                    rop.ll.type_float_ptr());

    Opcode& op(rop.inst()->ops()[opnum]);
    for (int a = first_optional_arg; a < op.nargs(); ++a) {
        Symbol& Name(*rop.opargsym(op, a));
        OSL_ASSERT(Name.typespec().is_string()
                   && "optional texture token must be a string");
        OSL_ASSERT(a + 1 < op.nargs() && "malformed argument list for texture");
        ustring name = Name.get_string();
        ++a;  // advance to next argument (value)

        if (name.empty())  // skip empty string param name
            continue;
        Symbol& Val(*rop.opargsym(op, a));
        TypeDesc valtype = Val.typespec().simpletype();


        bool nameIsVarying = !Name.is_uniform();
        // assuming option names can't be varying
        OSL_ASSERT(!nameIsVarying);
        // data could be varying
        bool valIsVarying = !Val.is_uniform();

#define PARAM_WIDE_FLOAT(paramname)                                            \
    if (name == Strings::paramname                                             \
        && (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) {         \
        llvm::Value* val                                                       \
            = rop.llvm_load_value(Val, /*deriv=*/0, /*component=*/0,           \
                                  TypeDesc::UNKNOWN, /*op_is_uniform=*/false); \
        if (valtype == TypeDesc::INT)                                          \
            val = rop.ll.op_int_to_float(val);                                 \
        paramname = val;                                                       \
        continue;                                                              \
    }

#define PARAM_WIDE_FLOAT_S_T_R(paramname)                                      \
    if (name == Strings::paramname                                             \
        && (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) {         \
        llvm::Value* val                                                       \
            = rop.llvm_load_value(Val, /*deriv=*/0, /*component=*/0,           \
                                  TypeDesc::UNKNOWN, /*op_is_uniform=*/false); \
        if (valtype == TypeDesc::INT) {                                        \
            val = rop.ll.op_int_to_float(val);                                 \
        }                                                                      \
        s##paramname = val;                                                    \
        t##paramname = val;                                                    \
        if (tex3d) {                                                           \
            r##paramname = val;                                                \
        }                                                                      \
        continue;                                                              \
    }

        PARAM_WIDE_FLOAT_S_T_R(width)
        PARAM_WIDE_FLOAT(swidth)
        PARAM_WIDE_FLOAT(twidth)
        if (tex3d) {
            PARAM_WIDE_FLOAT(rwidth)
        }

        PARAM_WIDE_FLOAT_S_T_R(blur)
        PARAM_WIDE_FLOAT(sblur)
        PARAM_WIDE_FLOAT(tblur)
        if (tex3d) {
            PARAM_WIDE_FLOAT(rblur)
        }

#define PARAM_UNIFORM_FLOAT(paramname)                                 \
    if (name == Strings::paramname                                     \
        && (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) { \
        is_##paramname##_uniform = !valIsVarying;                      \
        if (valIsVarying) {                                            \
            continue;                                                  \
        }                                                              \
        llvm::Value* val = rop.llvm_load_value(Val);                   \
        if (valtype == TypeDesc::INT)                                  \
            val = rop.ll.op_int_to_float(val);                         \
        paramname = val;                                               \
        continue;                                                      \
    }

#define PARAM_UNIFORM_INT(paramname)                                \
    if (name == Strings::paramname && (valtype == TypeDesc::INT)) { \
        is_##paramname##_uniform = !valIsVarying;                   \
        if (valIsVarying) {                                         \
            continue;                                               \
        }                                                           \
        llvm::Value* val = rop.llvm_load_value(Val);                \
        paramname        = val;                                     \
        continue;                                                   \
    }

#define PARAM_UNIFORM_STRING_CODE(paramname, decoder, llvm_decoder, fieldname) \
    if (name == Strings::paramname && valtype == TypeDesc::STRING) {           \
        if (valIsVarying) {                                                    \
            is_##fieldname##_uniform = false;                                  \
            continue;                                                          \
        }                                                                      \
        llvm::Value* val = nullptr;                                            \
        if (Val.is_constant()) {                                               \
            int mode = decoder(Val.get_string());                              \
            val      = rop.ll.constant(mode);                                  \
        } else {                                                               \
            val = rop.llvm_load_value(Val);                                    \
            val = rop.ll.call_function(#llvm_decoder, val);                    \
        }                                                                      \
        fieldname = val;                                                       \
        continue;                                                              \
    }

        if (name == Strings::wrap && valtype == TypeDesc::STRING) {
            if (valIsVarying) {
                is_swrap_uniform = false;
                is_twrap_uniform = false;
                if (tex3d) {
                    is_rwrap_uniform = false;
                }
                continue;
            }
            llvm::Value* val = nullptr;
            if (Val.is_constant()) {
                int mode = TextureOpt::decode_wrapmode(Val.get_string());
                val      = rop.ll.constant(mode);
            } else {
                val = rop.llvm_load_value(Val);
                val = rop.ll.call_function("osl_texture_decode_wrapmode", val);
            }
            swrap = val;
            twrap = val;
            if (tex3d) {
                rwrap = val;
            }
            continue;
        }
        PARAM_UNIFORM_STRING_CODE(swrap, OIIO::TextureOpt::decode_wrapmode,
                                  osl_texture_decode_wrapmode, swrap)
        PARAM_UNIFORM_STRING_CODE(twrap, OIIO::TextureOpt::decode_wrapmode,
                                  osl_texture_decode_wrapmode, twrap)
        if (tex3d) {
            PARAM_UNIFORM_STRING_CODE(rwrap, OIIO::TextureOpt::decode_wrapmode,
                                      osl_texture_decode_wrapmode, rwrap)
        }

        PARAM_UNIFORM_FLOAT(fill)
        PARAM_UNIFORM_INT(firstchannel)
        PARAM_UNIFORM_INT(subimage)

        if (name == Strings::subimage && valtype == TypeDesc::STRING) {
            if (valIsVarying) {
                is_subimagename_uniform = false;
                continue;
            }
            if (Val.is_constant()) {
                ustring v = Val.get_string();
                if (v.empty() && (subimage == const_zero_value)) {
                    continue;  // Ignore nulls unless they are overrides
                }
            }
            llvm::Value* val = rop.llvm_load_value(Val);
            subimagename     = val;
            continue;
        }

        PARAM_UNIFORM_STRING_CODE(interp, tex_interp_to_code,
                                  osl_texture_decode_interpmode, interpmode)


        if (name == Strings::alpha && valtype == TypeDesc::FLOAT) {
            OSL_ASSERT(
                valIsVarying
                && "thought to be unreachable as texture is inherently wide, so any alpha variable needs to be as well");
            // We will handle this as part of the uniform section, as we will point to wide variables
            // and technically the alpha is part of the output parameters not a texture option
            alpha          = rop.llvm_get_pointer(Val);
            alphaHasDerivs = Val.has_derivs();
            continue;
        }

        if (name == Strings::errormessage && valtype == TypeDesc::STRING) {
            OSL_ASSERT(
                valIsVarying
                && "thought to be unreachable as texture is inherently wide, so any errormessage variable needs to be as well");
            // We will handle this as part of the uniform section, as we will point to wide variables
            // and technically the alpha is part of the output parameters not a texture option
            errormessage = rop.llvm_get_pointer(Val);
            continue;
        }

        if (name == Strings::missingcolor && equivalent(valtype, TypeColor)) {
            if (!missingcolor_buffer) {
                // If not already done, allocate enough storage for the
                // missingcolor value (4 floats), and call the special
                // function that points the TextureOpt.missingcolor to it.
                missingcolor_buffer = rop.getOrAllocateTemp(
                    TypeSpec(TypeDesc(TypeDesc::FLOAT, 4)), false /*derivs()*/,
                    true /*is_uniform*/, false /*forceBool*/,
                    "float missingcolor[4]");
                missingcolor = missingcolor_buffer;
            }
            if (valIsVarying) {
                // For the varying case, we still wanted to allocate a scalar missingcolor_buffer
                // and point the uniform portion of the BatchedTextureOptions at it
                // So we don't track if its uniform, as we always write it out
                continue;
            }
            OSL_ASSERT(missingcolor_buffer != nullptr);
            rop.ll.op_memcpy(rop.ll.void_ptr(missingcolor_buffer),
                             rop.llvm_void_ptr(Val), (int)sizeof(Color3));
            continue;
        }

        if (name == Strings::missingalpha && valtype == TypeDesc::FLOAT) {
            if (!missingcolor_buffer) {
                // If not already done, allocate enough storage for the
                // missingcolor value (4 floats), and call the special
                // function that points the TextureOpt.missingcolor to it.
                missingcolor_buffer = rop.getOrAllocateTemp(
                    TypeSpec(TypeDesc(TypeDesc::FLOAT, 4)), false /*derivs()*/,
                    true /*is_uniform*/, false /*forceBool*/,
                    "float missingcolor[4]");
                missingcolor = missingcolor_buffer;
            }
            if (valIsVarying) {
                // For the varying case, we still wanted to allocate a scalar missingcolor_buffer
                // and point the uniform portion of the BatchedTextureOptions at it
                // So we don't track if its uniform, as we always write it out
                continue;
            }
            OSL_ASSERT(missingcolor_buffer != nullptr);
            llvm::Value* val = rop.llvm_load_value(Val);
            // Depending on how render services handles channels, might need to be 3 period.
            rop.ll.op_unmasked_store(val,
                                     rop.ll.GEP(rop.ll.type_float(),
                                                missingcolor_buffer, nchans));
            continue;
        }

        if (name == Strings::time
            && (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) {
            // NOTE: currently no supported 3d texture format makes use of time.
            // Therefore we will not represented time in BatchedTextureOptions
            // but will silently accept and ignore the time option
            continue;
        }

        rop.shadingcontext()->errorfmt(
            "Unknown texture{} optional argument: \"{}\", <{}> ({}:{})",
            tex3d ? "3d" : "", name, valtype, op.sourcefile(), op.sourceline());

#undef PARAM_WIDE_FLOAT
#undef PARAM_WIDE_FLOAT_S_T_R
#undef PARAM_UNIFORM_FLOAT
#undef PARAM_UNIFORM_INT
#undef PARAM_UNIFORM_STRING_CODE
    }

    // The LLVMMemberIndex will be the same for any width of BatchedTextureOptions,
    // so we will just use a WidthT=16 to look them up
    typedef OSL::BatchedTextureOptions<16>::LLVMMemberIndex LLVMMemberIndex;

    if (is_firstchannel_uniform)
        rop.ll.op_unmasked_store(
            firstchannel,
            rop.ll.GEP(bto_type, bto, 0,
                       static_cast<int>(LLVMMemberIndex::firstchannel)));
    if (is_subimage_uniform)
        rop.ll.op_unmasked_store(
            subimage, rop.ll.GEP(bto_type, bto, 0,
                                 static_cast<int>(LLVMMemberIndex::subimage)));

    if (is_subimagename_uniform)
        rop.ll.op_unmasked_store(
            subimagename,
            rop.ll.GEP(bto_type, bto, 0,
                       static_cast<int>(LLVMMemberIndex::subimagename)));

    if (is_swrap_uniform)
        rop.ll.op_unmasked_store(
            swrap, rop.ll.GEP(bto_type, bto, 0,
                              static_cast<int>(LLVMMemberIndex::swrap)));
    if (is_twrap_uniform)
        rop.ll.op_unmasked_store(
            twrap, rop.ll.GEP(bto_type, bto, 0,
                              static_cast<int>(LLVMMemberIndex::twrap)));
    if (is_rwrap_uniform)
        rop.ll.op_unmasked_store(
            rwrap, rop.ll.GEP(bto_type, bto, 0,
                              static_cast<int>(LLVMMemberIndex::rwrap)));

    // No way to set mipmode option from OSL
    rop.ll.op_unmasked_store(
        mipmode, rop.ll.GEP(bto_type, bto, 0,
                            static_cast<int>(LLVMMemberIndex::mipmode)));

    if (is_interpmode_uniform)
        rop.ll.op_unmasked_store(interpmode,
                                 rop.ll.GEP(bto_type, bto, 0,
                                            static_cast<int>(
                                                LLVMMemberIndex::interpmode)));

    // No way to set anisotropic option from OSL
    rop.ll.op_unmasked_store(anisotropic,
                             rop.ll.GEP(bto_type, bto, 0,
                                        static_cast<int>(
                                            LLVMMemberIndex::anisotropic)));

    // No way to set conservative_filter option from OSL
    rop.ll.op_unmasked_store(
        conservative_filter,
        rop.ll.GEP(bto_type, bto, 0,
                   static_cast<int>(LLVMMemberIndex::conservative_filter)));

    if (is_fill_uniform)
        rop.ll.op_unmasked_store(fill, rop.ll.GEP(bto_type, bto, 0,
                                                  static_cast<int>(
                                                      LLVMMemberIndex::fill)));

    // For uniform and varying we point the missingcolor to nullptr or the missingcolor_buffer,
    // The varying options will copy the lead lane's missing color value into the missingcolor_buffer
    rop.ll.op_unmasked_store(missingcolor,
                             rop.ll.GEP(bto_type, bto, 0,
                                        static_cast<int>(
                                            LLVMMemberIndex::missingcolor)));

    // blur's and width's are always communicated as wide, we we will handle them here
    rop.ll.op_unmasked_store(sblur, rop.ll.GEP(bto_type, bto, 0,
                                               static_cast<int>(
                                                   LLVMMemberIndex::sblur)));
    rop.ll.op_unmasked_store(tblur, rop.ll.GEP(bto_type, bto, 0,
                                               static_cast<int>(
                                                   LLVMMemberIndex::tblur)));
    rop.ll.op_unmasked_store(swidth, rop.ll.GEP(bto_type, bto, 0,
                                                static_cast<int>(
                                                    LLVMMemberIndex::swidth)));
    rop.ll.op_unmasked_store(twidth, rop.ll.GEP(bto_type, bto, 0,
                                                static_cast<int>(
                                                    LLVMMemberIndex::twidth)));

    if (tex3d) {
        rop.ll.op_unmasked_store(
            rblur, rop.ll.GEP(bto_type, bto, 0,
                              static_cast<int>(LLVMMemberIndex::rblur)));
        rop.ll.op_unmasked_store(
            rwidth, rop.ll.GEP(bto_type, bto, 0,
                               static_cast<int>(LLVMMemberIndex::rwidth)));
    }
    rop.ll.op_unmasked_store(rnd, rop.ll.GEP(bto_type, bto, 0,
                                             static_cast<float>(
                                                 LLVMMemberIndex::rnd)));

    return rop.ll.void_ptr(bto);
}



static llvm::Value*
llvm_batched_texture_varying_options(BatchedBackendLLVM& rop, int opnum,
                                     int first_optional_arg, bool tex3d,
                                     int nchans, llvm::Value* remainingMask,
                                     llvm::Value* leadLane,
                                     llvm::Value* missingcolor_buffer)
{
    llvm::Value* bto     = rop.temp_batched_texture_options_ptr();
    llvm::Type* bto_type = rop.llvm_type_batched_texture_options();

    // The BatchedTextureOptions & missingcolor_buffer are local alloca,
    // so no need to mask off non-active lanes

    // The LLVMMemberIndex will be the same for any width of BatchedTextureOptions,
    // so we will just use a WidthT=16 to look them up
    typedef OSL::BatchedTextureOptions<16>::LLVMMemberIndex LLVMMemberIndex;

    Opcode& op(rop.inst()->ops()[opnum]);
    for (int a = first_optional_arg; a < op.nargs(); ++a) {
        Symbol& Name(*rop.opargsym(op, a));
        OSL_ASSERT(Name.typespec().is_string()
                   && "optional texture token must be a string");
        OSL_ASSERT(a + 1 < op.nargs() && "malformed argument list for texture");
        ustring name = Name.get_string();
        ++a;  // advance to next argument (value)

        if (name.empty())  // skip empty string param name
            continue;
        Symbol& Val(*rop.opargsym(op, a));
        TypeDesc valtype = Val.typespec().simpletype();

        bool nameIsVarying = !Name.is_uniform();
        // assuming option names can't be varying
        OSL_ASSERT(!nameIsVarying);

        // data could be uniform
        bool valIsVarying = !Val.is_uniform();
        if (!valIsVarying)
            continue;

        OSL_ASSERT(!Val.is_constant() && "can't be a varying constant");

#define SKIP_PARAM_WIDE_FLOAT(paramname)                               \
    if (name == Strings::paramname                                     \
        && (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) { \
        continue;                                                      \
    }

#define SKIP_PARAM_WIDE_STRING(paramname)                            \
    if (name == Strings::paramname && valtype == TypeDesc::STRING) { \
        continue;                                                    \
    }

        SKIP_PARAM_WIDE_FLOAT(width)
        SKIP_PARAM_WIDE_FLOAT(swidth)
        SKIP_PARAM_WIDE_FLOAT(twidth)
        SKIP_PARAM_WIDE_FLOAT(rwidth)

        SKIP_PARAM_WIDE_FLOAT(blur)
        SKIP_PARAM_WIDE_FLOAT(sblur)
        SKIP_PARAM_WIDE_FLOAT(tblur)
        SKIP_PARAM_WIDE_FLOAT(rblur)

        SKIP_PARAM_WIDE_FLOAT(alpha)
        SKIP_PARAM_WIDE_STRING(errormessage)

        if (name == Strings::wrap && valtype == TypeDesc::STRING) {
            llvm::Value* wide_wrap
                = rop.llvm_load_value(Val, /*deriv=*/0, /*component=*/0,
                                      TypeDesc::UNKNOWN,
                                      /*op_is_uniform=*/false);
            llvm::Value* scalar_value = rop.ll.op_extract(wide_wrap, leadLane);
            llvm::Value* wrap_code
                = rop.ll.call_function("osl_texture_decode_wrapmode",
                                       scalar_value);
            rop.ll.op_unmasked_store(wrap_code,
                                     rop.ll.GEP(bto_type, bto, 0,
                                                static_cast<int>(
                                                    LLVMMemberIndex::swrap)));
            rop.ll.op_unmasked_store(wrap_code,
                                     rop.ll.GEP(bto_type, bto, 0,
                                                static_cast<int>(
                                                    LLVMMemberIndex::twrap)));

            if (tex3d)
                rop.ll.op_unmasked_store(
                    wrap_code,
                    rop.ll.GEP(bto_type, bto, 0,
                               static_cast<int>(LLVMMemberIndex::rwrap)));

            remainingMask = rop.ll.op_lanes_that_match_masked(scalar_value,
                                                              wide_wrap,
                                                              remainingMask);
            continue;
        }

#define PARAM_VARYING_STRING_CODE(paramname, llvm_decoder, fieldname)          \
    if (name == Strings::paramname && valtype == TypeDesc::STRING) {           \
        llvm::Value* wide_value                                                \
            = rop.llvm_load_value(Val, /*deriv=*/0, /*component=*/0,           \
                                  TypeDesc::UNKNOWN, /*op_is_uniform=*/false); \
        llvm::Value* scalar_value = rop.ll.op_extract(wide_value, leadLane);   \
        llvm::Value* scalar_code  = rop.ll.call_function(#llvm_decoder,        \
                                                         scalar_value);        \
        rop.ll.op_unmasked_store(scalar_code,                                  \
                                 rop.ll.GEP(bto_type, bto, 0,                  \
                                            static_cast<int>(                  \
                                                LLVMMemberIndex::fieldname))); \
        remainingMask = rop.ll.op_lanes_that_match_masked(scalar_value,        \
                                                          wide_value,          \
                                                          remainingMask);      \
        continue;                                                              \
    }

        PARAM_VARYING_STRING_CODE(swrap, osl_texture_decode_wrapmode, swrap)
        PARAM_VARYING_STRING_CODE(twrap, osl_texture_decode_wrapmode, twrap)
        if (tex3d) {
            PARAM_VARYING_STRING_CODE(rwrap, osl_texture_decode_wrapmode, rwrap)
        }


        if (name == Strings::fill
            && (valtype == TypeDesc::FLOAT || valtype == TypeDesc::INT)) {
            llvm::Value* wide_val
                = rop.llvm_load_value(Val, /*deriv=*/0, /*component=*/0,
                                      TypeDesc::UNKNOWN,
                                      /*op_is_uniform=*/false);
            llvm::Value* scalar_value = rop.ll.op_extract(wide_val, leadLane);
            remainingMask = rop.ll.op_lanes_that_match_masked(scalar_value,
                                                              wide_val,
                                                              remainingMask);
            if (valtype == TypeDesc::INT)
                scalar_value = rop.ll.op_int_to_float(scalar_value);
            rop.ll.op_unmasked_store(scalar_value,
                                     rop.ll.GEP(bto_type, bto, 0,
                                                static_cast<int>(
                                                    LLVMMemberIndex::fill)));
            continue;
        }

#define PARAM_VARYING(paramname, paramtype, fieldname)                         \
    if (name == Strings::paramname && valtype == paramtype) {                  \
        llvm::Value* wide_val                                                  \
            = rop.llvm_load_value(Val, /*deriv=*/0, /*component=*/0,           \
                                  TypeDesc::UNKNOWN, /*op_is_uniform=*/false); \
        llvm::Value* scalar_value = rop.ll.op_extract(wide_val, leadLane);     \
        rop.ll.op_unmasked_store(scalar_value,                                 \
                                 rop.ll.GEP(bto_type, bto, 0,                  \
                                            static_cast<int>(                  \
                                                LLVMMemberIndex::fieldname))); \
        remainingMask = rop.ll.op_lanes_that_match_masked(scalar_value,        \
                                                          wide_val,            \
                                                          remainingMask);      \
        continue;                                                              \
    }
        PARAM_VARYING(firstchannel, TypeDesc::INT, firstchannel)
        PARAM_VARYING(subimage, TypeDesc::INT, subimage)
        PARAM_VARYING(subimage, TypeDesc::STRING, subimagename)

        PARAM_VARYING_STRING_CODE(interp, osl_texture_decode_interpmode,
                                  interpmode)

        if (name == Strings::missingcolor && equivalent(valtype, TypeColor)) {
            OSL_ASSERT(missingcolor_buffer != nullptr);

            int num_components = valtype.aggregate;
            for (int i = 0; i < num_components; i++) {
                llvm::Value* wide_component
                    = rop.llvm_load_value(Val, 0, i, valtype,
                                          false /*op_is_uniform*/);
                llvm::Value* scalar_component
                    = rop.ll.op_extract(wide_component, leadLane);
                // The missingcolor_buffer is a local alloca, so no need to mask off non-active lanes
                rop.ll.op_unmasked_store(scalar_component,
                                         rop.ll.GEP(rop.ll.type_float(),
                                                    missingcolor_buffer, i));

                remainingMask = rop.ll.op_lanes_that_match_masked(
                    scalar_component, wide_component, remainingMask);
            }
            continue;
        }

        if (name == Strings::missingalpha && valtype == TypeDesc::FLOAT) {
            OSL_ASSERT(missingcolor_buffer != nullptr);

            llvm::Value* wide_missingalpha
                = rop.llvm_load_value(Val, /*deriv=*/0, /*component=*/0,
                                      TypeDesc::UNKNOWN,
                                      /*op_is_uniform=*/false);
            llvm::Value* scalar_missingalpha
                = rop.ll.op_extract(wide_missingalpha, leadLane);
            // Depending on how render services handles channels, might need to be 3 period.
            rop.ll.op_unmasked_store(scalar_missingalpha,
                                     rop.ll.GEP(rop.ll.type_float(),
                                                missingcolor_buffer, nchans));

            remainingMask = rop.ll.op_lanes_that_match_masked(
                scalar_missingalpha, wide_missingalpha, remainingMask);
            continue;
        }

        SKIP_PARAM_WIDE_FLOAT(time)

        rop.shadingcontext()->errorfmt(
            "Unknown texture{} optional argument: \"{}\", <{}> ({}:{})",
            tex3d ? "3d" : "", name, valtype, op.sourcefile(), op.sourceline());

#undef SKIP_PARAM_WIDE_FLOAT
#undef SKIP_PARAM_WIDE_STRING
#undef PARAM_VARYING_STRING_CODE
#undef PARAM_VARYING
    }
    return remainingMask;
}



LLVMGEN(llvm_gen_texture)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result   = *rop.opargsym(op, 0);
    Symbol& Filename = *rop.opargsym(op, 1);
    Symbol& S        = *rop.opargsym(op, 2);
    Symbol& T        = *rop.opargsym(op, 3);

    // The current texture interface doesn't support uniform results
    // If uniform results become a common use-case, it should probably
    // be a special call added to get_texture_info so it can be
    // constant folded.
    OSL_ASSERT(Result.is_uniform() == false);

    int nchans = Result.typespec().aggregate();

    bool user_derivs       = false;
    int first_optional_arg = 4;
    if (op.nargs() > 4 && rop.opargsym(op, 4)->typespec().is_float()) {
        user_derivs        = true;
        first_optional_arg = 8;
        OSL_DASSERT(rop.opargsym(op, 5)->typespec().is_float());
        OSL_DASSERT(rop.opargsym(op, 6)->typespec().is_float());
        OSL_DASSERT(rop.opargsym(op, 7)->typespec().is_float());
    }

    // make sure that any temps created by llvm_batched_texture_options
    // and llvm_batched_texture_varying_options
    // are not released until we are done using them!
    BatchedBackendLLVM::TempScope temp_scope(rop);

    llvm::Value* opt;  // TextureOpt
    llvm::Value* alpha        = NULL;
    bool alphaHasDerivs       = false;
    llvm::Value* errormessage = NULL;

    llvm::Value* missingcolor_buffer = nullptr;
    opt = llvm_batched_texture_options(rop, opnum, first_optional_arg,
                                       false /*3d*/, nchans, alpha,
                                       alphaHasDerivs, errormessage,
                                       missingcolor_buffer);

    // Now call the osl_texture function, passing the options and all the
    // explicit args like texture coordinates.
    llvm::Value* args[17];
    args[0] = rop.sg_void_ptr();

    const char* texFuncName = rop.build_name(FuncSpec("texture").mask());
    RendererServices::TextureHandle* texture_handle = NULL;
    if (Filename.is_constant() && rop.shadingsys().opt_texture_handle()) {
        OSL_ASSERT(Filename.is_uniform());
        texture_handle
            = rop.renderer()->get_texture_handle(Filename.get_string(),
                                                 rop.shadingcontext());
    }

    // We will just load the filename here if we are uniform, otherwise just remember where the filename
    // is so we can update it later in the loop over varying options
    static constexpr int filenameArgumentIndex = 1;
    llvm::Value* filenameVal
        = rop.llvm_load_value(Filename, /*deriv=*/0, /*component=*/0,
                              TypeDesc::UNKNOWN, Filename.is_uniform());
    args[filenameArgumentIndex] = Filename.is_uniform() ? filenameVal : nullptr;

    args[2] = rop.ll.constant_ptr(texture_handle);
    rop.generated_texture_call(texture_handle != NULL);

    // check S & T are not uniform

    llvm::Value* wideS   = nullptr;
    llvm::Value* wideSD1 = nullptr;
    llvm::Value* wideSD2 = nullptr;
    llvm::Value* wideT   = nullptr;
    llvm::Value* wideTD1 = nullptr;
    llvm::Value* wideTD2 = nullptr;

    if (S.is_uniform()) {
        wideS = rop.llvm_widen_value_into_temp(S, 0);
        if (!user_derivs) {
            wideSD1 = rop.llvm_widen_value_into_temp(S, 1);
            wideSD2 = rop.llvm_widen_value_into_temp(S, 2);
        }
    } else {
        wideS = rop.llvm_void_ptr(S, 0);
        if (!user_derivs) {
            wideSD1 = rop.llvm_void_ptr(S, 1);
            wideSD2 = rop.llvm_void_ptr(S, 2);
        }
    }
    if (T.is_uniform()) {
        wideT = rop.llvm_widen_value_into_temp(T, 0);
        if (!user_derivs) {
            wideTD1 = rop.llvm_widen_value_into_temp(T, 1);
            wideTD2 = rop.llvm_widen_value_into_temp(T, 2);
        }
    } else {
        wideT = rop.llvm_void_ptr(T);
        if (!user_derivs) {
            wideTD1 = rop.llvm_void_ptr(T, 1);
            wideTD2 = rop.llvm_void_ptr(T, 2);
        }
    }
    args[3]               = opt;
    args[4]               = wideS;
    args[5]               = wideT;
    llvm::Value* wideDsDx = nullptr;
    llvm::Value* wideDtDx = nullptr;
    llvm::Value* wideDsDy = nullptr;
    llvm::Value* wideDtDy = nullptr;

    if (user_derivs) {
        Symbol& DsDx = *rop.opargsym(op, 4);
        Symbol& DtDx = *rop.opargsym(op, 5);
        Symbol& DsDy = *rop.opargsym(op, 6);
        Symbol& DtDy = *rop.opargsym(op, 7);
        if (DsDx.is_uniform()) {
            wideDsDx = rop.llvm_widen_value_into_temp(DsDx, 0);
        } else {
            wideDsDx = rop.llvm_void_ptr(DsDx, 0);
        }
        if (DtDx.is_uniform()) {
            wideDtDx = rop.llvm_widen_value_into_temp(DtDx, 0);
        } else {
            wideDtDx = rop.llvm_void_ptr(DtDx, 0);
        }
        if (DsDy.is_uniform()) {
            wideDsDy = rop.llvm_widen_value_into_temp(DsDy, 0);
        } else {
            wideDsDy = rop.llvm_void_ptr(DsDy, 0);
        }
        if (DtDy.is_uniform()) {
            wideDtDy = rop.llvm_widen_value_into_temp(DtDy, 0);
        } else {
            wideDtDy = rop.llvm_void_ptr(DtDy, 0);
        }
    } else {
        // Auto derivs of S and T
        wideDsDx = wideSD1;
        wideDtDx = wideTD1;
        wideDsDy = wideSD2;
        wideDtDy = wideTD2;
    }
    args[6] = wideDsDx;
    args[7] = wideDtDx;
    args[8] = wideDsDy;
    args[9] = wideDtDy;

    OSL_DEV_ONLY(
        std::cout << "texture result type: "
                  << rop.ll.llvm_typenameof(rop.llvm_get_pointer(Result, 1))
                  << std::endl);
    args[10] = rop.ll.constant(nchans);
    args[11] = rop.ll.void_ptr(rop.llvm_get_pointer(Result, 0));
    args[12] = Result.has_derivs() ? rop.ll.constant(1) : rop.ll.constant(0);
    args[13] = rop.ll.void_ptr(alpha ? alpha : rop.ll.void_ptr_null());
    args[14] = alphaHasDerivs ? rop.ll.constant(1) : rop.ll.constant(0);
    args[15] = rop.ll.void_ptr(errormessage ? errormessage
                                            : rop.ll.void_ptr_null());

    // do while(remaining)
    llvm::Value* loc_of_remainingMask = rop.getTempMask(
        "lanes remaining to texture");
    rop.ll.op_store_mask(rop.ll.current_mask(), loc_of_remainingMask);

    llvm::BasicBlock* bin_block = rop.ll.new_basic_block(
        rop.llvm_debug()
            ? std::string("bin_texture_options (varying texture options)")
            : std::string());
    llvm::BasicBlock* after_block = rop.ll.new_basic_block(
        rop.llvm_debug()
            ? std::string("after_bin_texture_options (varying texture options)")
            : std::string());
    rop.ll.op_branch(bin_block);
    {
        llvm::Value* remainingMask = rop.ll.op_load_mask(loc_of_remainingMask);
        llvm::Value* leadLane = rop.ll.op_1st_active_lane_of(remainingMask);
        llvm::Value* lanesMatchingFilename = remainingMask;

        if (false == Filename.is_uniform()) {
            llvm::Value* scalar_filename = rop.ll.op_extract(filenameVal,
                                                             leadLane);
            args[filenameArgumentIndex]  = scalar_filename;
            lanesMatchingFilename
                = rop.ll.op_lanes_that_match_masked(scalar_filename,
                                                    filenameVal, remainingMask);
        }

        //rop.llvm_print_mask("before remainingMask", remainingMask);
        llvm::Value* lanesMatchingOptions = llvm_batched_texture_varying_options(
            rop, opnum, first_optional_arg, false /*3d*/, nchans,
            lanesMatchingFilename, leadLane, missingcolor_buffer);
        OSL_ASSERT(lanesMatchingOptions);
        //rop.llvm_print_mask("lanesMatchingOptions", lanesMatchingOptions);
        args[16] = rop.ll.mask_as_int(lanesMatchingOptions);

        rop.ll.call_function(texFuncName, args);

        remainingMask = rop.ll.op_xor(remainingMask, lanesMatchingOptions);
        //rop.llvm_print_mask("xor remainingMask,lanesMatchingOptions", remainingMask);
        rop.ll.op_store_mask(remainingMask, loc_of_remainingMask);

        llvm::Value* int_remainingMask = rop.ll.mask_as_int(remainingMask);
        //rop.llvm_print_mask("remainingMask", remainingMask);
        llvm::Value* cond_more_lanes_to_bin = rop.ll.op_ne(int_remainingMask,
                                                           rop.ll.constant(0));
        rop.ll.op_branch(cond_more_lanes_to_bin, bin_block, after_block);
    }
    // Continue on with the previous flow
    rop.ll.set_insert_point(after_block);

    return true;
}



LLVMGEN(llvm_gen_texture3d)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result   = *rop.opargsym(op, 0);
    Symbol& Filename = *rop.opargsym(op, 1);
    Symbol& P        = *rop.opargsym(op, 2);

    // The current texture interface doesn't support uniform results
    OSL_ASSERT(Result.is_uniform() == false);

    int nchans = Result.typespec().aggregate();

    bool user_derivs       = false;
    int first_optional_arg = 3;  // Optional arguments after filename and point.

    // derivatives
    if (op.nargs() > 3 && rop.opargsym(op, 3)->typespec().is_triple()) {
        user_derivs        = true;
        first_optional_arg = 6;
        OSL_DASSERT(
            rop.opargsym(op, 3)->typespec().is_triple());  // vector dpdx
        OSL_DASSERT(
            rop.opargsym(op, 4)->typespec().is_triple());  // vector dpdy
        OSL_DASSERT(
            rop.opargsym(op, 5)->typespec().is_triple());  // vector dpdy
    }

    // make sure that any temps created by llvm_batched_texture_options
    // and llvm_batched_texture_varying_options
    // are not released until we are done using them!
    BatchedBackendLLVM::TempScope temp_scope(rop);

    llvm::Value* opt;  // TextureOpt
    llvm::Value* alpha        = NULL;
    bool alphaHasDerivs       = false;
    llvm::Value* errormessage = NULL;

    llvm::Value* missingcolor_buffer = nullptr;
    opt = llvm_batched_texture_options(rop, opnum, first_optional_arg,
                                       true /*enable 3d*/, nchans, alpha,
                                       alphaHasDerivs, errormessage,
                                       missingcolor_buffer);

    // Now call the osl_texture function, passing the options and all the
    // explicit args like texture coordinates.
    llvm::Value* args[15];
    args[0] = rop.sg_void_ptr();

    bool fileNameIsUniform = Filename.is_uniform();

    const char* texFuncName = rop.build_name(FuncSpec("texture3d").mask());
    RendererServices::TextureHandle* texture_handle = NULL;
    if (Filename.is_constant() && rop.shadingsys().opt_texture_handle()) {
        OSL_ASSERT(fileNameIsUniform);
        texture_handle
            = rop.renderer()->get_texture_handle(Filename.get_string(),
                                                 rop.shadingcontext());
    }

    // We will just load the filename here if we are uniform, otherwise just remember where the filename
    // is so we can update it later in the loop over varying options
    int filenameArgumentIndex = 1;
    llvm::Value* filenameVal
        = rop.llvm_load_value(Filename, /*deriv=*/0, /*component=*/0,
                              TypeDesc::UNKNOWN, Filename.is_uniform());
    args[filenameArgumentIndex] = fileNameIsUniform ? filenameVal : nullptr;

    args[2] = rop.ll.constant_ptr(texture_handle);
    rop.generated_texture_call(texture_handle != NULL);

    // check S & T & R are not uniform

    llvm::Value* wideSTR   = nullptr;
    llvm::Value* wideSTRD1 = nullptr;
    llvm::Value* wideSTRD2 = nullptr;

    if (P.is_uniform()) {
        wideSTR = rop.llvm_widen_value_into_temp(P, 0);
        if (!user_derivs) {
            wideSTRD1 = rop.llvm_widen_value_into_temp(P, 1);
            wideSTRD2 = rop.llvm_widen_value_into_temp(P, 2);
        }
    } else {
        wideSTR = rop.llvm_void_ptr(P, 0);
        if (!user_derivs) {
            wideSTRD1 = rop.llvm_void_ptr(P, 1);
            wideSTRD2 = rop.llvm_void_ptr(P, 2);
        }
    }

    args[3] = opt;
    args[4] = wideSTR;

    llvm::Value* wideSTRDx = nullptr;
    llvm::Value* wideSTRDy = nullptr;
    llvm::Value* wideSTRDz = nullptr;

    if (user_derivs) {
        Symbol& PDx = *rop.opargsym(op, 3);
        Symbol& PDy = *rop.opargsym(op, 4);
        Symbol& PDz = *rop.opargsym(op, 5);

        if (PDx.is_uniform()) {
            wideSTRDx = rop.llvm_widen_value_into_temp(PDx, 0);
        } else {
            wideSTRDx = rop.llvm_void_ptr(PDx, 0);
        }


        if (PDy.is_uniform()) {
            wideSTRDy = rop.llvm_widen_value_into_temp(PDy, 0);
        } else {
            wideSTRDy = rop.llvm_void_ptr(PDy, 0);
        }

        if (PDz.is_uniform()) {
            wideSTRDz = rop.llvm_widen_value_into_temp(PDz, 0);
        } else {
            wideSTRDz = rop.llvm_void_ptr(PDz, 0);
        }

    } else {
        // Auto derivs of S and T, leave R as nullptr and
        // ensure implmementation can handle a nullptr
        wideSTRDx = wideSTRD1;
        wideSTRDy = wideSTRD2;
        wideSTRDz = rop.ll.void_ptr_null();
    }

    args[5] = wideSTRDx;
    args[6] = wideSTRDy;
    args[7] = wideSTRDz;

    OSL_DEV_ONLY(
        std::cout << "texture result type: "
                  << rop.ll.llvm_typenameof(rop.llvm_get_pointer(Result, 1))
                  << std::endl);
    args[8]  = rop.ll.constant(nchans);
    args[9]  = rop.ll.void_ptr(rop.llvm_get_pointer(Result, 0));
    args[10] = Result.has_derivs() ? rop.ll.constant(1) : rop.ll.constant(0);
    args[11] = rop.ll.void_ptr(alpha ? alpha : rop.ll.void_ptr_null());
    args[12] = alphaHasDerivs ? rop.ll.constant(1) : rop.ll.constant(0);
    args[13] = rop.ll.void_ptr(errormessage ? errormessage
                                            : rop.ll.void_ptr_null());

    // do while(remaining)
    llvm::Value* loc_of_remainingMask = rop.getTempMask(
        "lanes remaining to texture");
    rop.ll.op_store_mask(rop.ll.current_mask(), loc_of_remainingMask);

    llvm::BasicBlock* bin_block = rop.ll.new_basic_block(
        rop.llvm_debug()
            ? std::string("bin_texture_options (varying texture options)")
            : std::string());
    llvm::BasicBlock* after_block = rop.ll.new_basic_block(
        rop.llvm_debug()
            ? std::string("after_bin_texture_options (varying texture options)")
            : std::string());
    rop.ll.op_branch(bin_block);
    {
        llvm::Value* remainingMask = rop.ll.op_load_mask(loc_of_remainingMask);
        llvm::Value* leadLane = rop.ll.op_1st_active_lane_of(remainingMask);
        llvm::Value* lanesMatchingFilename = remainingMask;

        if (false == fileNameIsUniform) {
            llvm::Value* scalar_filename = rop.ll.op_extract(filenameVal,
                                                             leadLane);
            args[filenameArgumentIndex]  = scalar_filename;
            lanesMatchingFilename
                = rop.ll.op_lanes_that_match_masked(scalar_filename,
                                                    filenameVal, remainingMask);
        }

        //rop.llvm_print_mask("before remainingMask", remainingMask);
        llvm::Value* lanesMatchingOptions = llvm_batched_texture_varying_options(
            rop, opnum, first_optional_arg, true /* enable 3d*/, nchans,
            lanesMatchingFilename, leadLane, missingcolor_buffer);
        OSL_ASSERT(lanesMatchingOptions);
        //rop.llvm_print_mask("lanesMatchingOptions", lanesMatchingOptions);
        args[14] = rop.ll.mask_as_int(lanesMatchingOptions);

        rop.ll.call_function(texFuncName, args);

        remainingMask = rop.ll.op_xor(remainingMask, lanesMatchingOptions);
        //rop.llvm_print_mask("xor remainingMask,lanesMatchingOptions", remainingMask);
        rop.ll.op_store_mask(remainingMask, loc_of_remainingMask);

        llvm::Value* int_remainingMask = rop.ll.mask_as_int(remainingMask);
        //rop.llvm_print_mask("remainingMask", remainingMask);
        llvm::Value* cond_more_lanes_to_bin = rop.ll.op_ne(int_remainingMask,
                                                           rop.ll.constant(0));
        rop.ll.op_branch(cond_more_lanes_to_bin, bin_block, after_block);
    }
    // Continue on with the previous flow
    rop.ll.set_insert_point(after_block);

    return true;
}



LLVMGEN(llvm_gen_environment)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result   = *rop.opargsym(op, 0);
    Symbol& Filename = *rop.opargsym(op, 1);
    Symbol& R        = *rop.opargsym(op, 2);

    // The current texture interface doesn't support uniform results
    OSL_ASSERT(Result.is_uniform() == false);

    int nchans = Result.typespec().aggregate();

    bool user_derivs       = false;
    int first_optional_arg = 3;  // Optional arguments after filename and point.

    // derivatives
    if (op.nargs() > 3 && rop.opargsym(op, 3)->typespec().is_triple()) {
        user_derivs        = true;
        first_optional_arg = 5;
        OSL_DASSERT(
            rop.opargsym(op, 4)->typespec().is_triple());  // vector dpdy
    }

    // make sure that any temps created by llvm_batched_texture_options
    // and llvm_batched_texture_varying_options
    // are not released until we are done using them!
    BatchedBackendLLVM::TempScope temp_scope(rop);

    llvm::Value* opt;  // TextureOpt
    llvm::Value* alpha        = NULL;
    bool alphaHasDerivs       = false;
    llvm::Value* errormessage = NULL;

    llvm::Value* missingcolor_buffer = nullptr;
    opt = llvm_batched_texture_options(rop, opnum, first_optional_arg,
                                       false /*enable 3d*/, nchans, alpha,
                                       alphaHasDerivs, errormessage,
                                       missingcolor_buffer);

    // Now call the osl_texture function, passing the options and all the
    // explicit args like texture coordinates.
    llvm::Value* args[14];
    args[0] = rop.sg_void_ptr();

    bool fileNameIsUniform = Filename.is_uniform();

    const char* texFuncName = rop.build_name(FuncSpec("environment").mask());
    RendererServices::TextureHandle* texture_handle = NULL;
    if (Filename.is_constant() && rop.shadingsys().opt_texture_handle()) {
        OSL_ASSERT(fileNameIsUniform);
        texture_handle
            = rop.renderer()->get_texture_handle(Filename.get_string(),
                                                 rop.shadingcontext());
    }

    // We will just load the filename here if we are uniform, otherwise just remember where the filename
    // is so we can update it later in the loop over varying options
    int filenameArgumentIndex = 1;
    llvm::Value* filenameVal
        = rop.llvm_load_value(Filename, /*deriv=*/0, /*component=*/0,
                              TypeDesc::UNKNOWN, Filename.is_uniform());
    args[filenameArgumentIndex] = fileNameIsUniform ? filenameVal : nullptr;

    args[2] = rop.ll.constant_ptr(texture_handle);
    rop.generated_texture_call(texture_handle != NULL);

    // check S & T & R are not uniform

    llvm::Value* wideR   = nullptr;
    llvm::Value* wideRD1 = nullptr;
    llvm::Value* wideRD2 = nullptr;

    if (R.is_uniform()) {
        wideR = rop.llvm_widen_value_into_temp(R, 0);
        if (!user_derivs) {
            wideRD1 = rop.llvm_widen_value_into_temp(R, 1);
            wideRD2 = rop.llvm_widen_value_into_temp(R, 2);
        }
    } else {
        wideR = rop.llvm_void_ptr(R, 0);
        if (!user_derivs) {
            wideRD1 = rop.llvm_void_ptr(R, 1);
            wideRD2 = rop.llvm_void_ptr(R, 2);
        }
    }

    args[3] = opt;
    args[4] = wideR;

    llvm::Value* wideRDx = nullptr;
    llvm::Value* wideRDy = nullptr;

    if (user_derivs) {
        Symbol& RDx = *rop.opargsym(op, 3);
        Symbol& RDy = *rop.opargsym(op, 4);

        if (RDx.is_uniform()) {
            wideRDx = rop.llvm_widen_value_into_temp(RDx, 0);
        } else {
            wideRDx = rop.llvm_void_ptr(RDx, 0);
        }


        if (RDy.is_uniform()) {
            wideRDy = rop.llvm_widen_value_into_temp(RDy, 0);
        } else {
            wideRDy = rop.llvm_void_ptr(RDy, 0);
        }
    } else {
        // Auto derivs of S and T, leave R as nullptr and
        // ensure implmementation can handle a nullptr
        wideRDx = wideRD1;
        wideRDy = wideRD2;
    }

    args[5] = wideRDx;
    args[6] = wideRDy;

    OSL_DEV_ONLY(
        std::cout << "environment result type: "
                  << rop.ll.llvm_typenameof(rop.llvm_get_pointer(Result, 1))
                  << std::endl);
    args[7]  = rop.ll.constant(nchans);
    args[8]  = rop.ll.void_ptr(rop.llvm_get_pointer(Result, 0));
    args[9]  = Result.has_derivs() ? rop.ll.constant(1) : rop.ll.constant(0);
    args[10] = rop.ll.void_ptr(alpha ? alpha : rop.ll.void_ptr_null());
    args[11] = alphaHasDerivs ? rop.ll.constant(1) : rop.ll.constant(0);
    args[12] = rop.ll.void_ptr(errormessage ? errormessage
                                            : rop.ll.void_ptr_null());

    // do while(remaining)
    llvm::Value* loc_of_remainingMask = rop.getTempMask(
        "lanes remaining to environment");
    rop.ll.op_store_mask(rop.ll.current_mask(), loc_of_remainingMask);

    llvm::BasicBlock* bin_block = rop.ll.new_basic_block(
        rop.llvm_debug() ? std::string(
            "bin_environment_options (varying environment options)")
                         : std::string());
    llvm::BasicBlock* after_block = rop.ll.new_basic_block(
        rop.llvm_debug() ? std::string(
            "after_bin_environment_options (varying environment options)")
                         : std::string());
    rop.ll.op_branch(bin_block);
    {
        llvm::Value* remainingMask = rop.ll.op_load_mask(loc_of_remainingMask);
        llvm::Value* leadLane = rop.ll.op_1st_active_lane_of(remainingMask);
        llvm::Value* lanesMatchingFilename = remainingMask;

        if (false == fileNameIsUniform) {
            llvm::Value* scalar_filename = rop.ll.op_extract(filenameVal,
                                                             leadLane);
            args[filenameArgumentIndex]  = scalar_filename;
            lanesMatchingFilename
                = rop.ll.op_lanes_that_match_masked(scalar_filename,
                                                    filenameVal, remainingMask);
        }

        //rop.llvm_print_mask("before remainingMask", remainingMask);
        llvm::Value* lanesMatchingOptions = llvm_batched_texture_varying_options(
            rop, opnum, first_optional_arg, false /* enable 3d*/, nchans,
            lanesMatchingFilename, leadLane, missingcolor_buffer);
        OSL_ASSERT(lanesMatchingOptions);
        //rop.llvm_print_mask("lanesMatchingOptions", lanesMatchingOptions);
        args[13] = rop.ll.mask_as_int(lanesMatchingOptions);

        rop.ll.call_function(texFuncName, args);

        remainingMask = rop.ll.op_xor(remainingMask, lanesMatchingOptions);
        //rop.llvm_print_mask("xor remainingMask,lanesMatchingOptions", remainingMask);
        rop.ll.op_store_mask(remainingMask, loc_of_remainingMask);

        llvm::Value* int_remainingMask = rop.ll.mask_as_int(remainingMask);
        //rop.llvm_print_mask("remainingMask", remainingMask);
        llvm::Value* cond_more_lanes_to_bin = rop.ll.op_ne(int_remainingMask,
                                                           rop.ll.constant(0));
        rop.ll.op_branch(cond_more_lanes_to_bin, bin_block, after_block);
    }
    // Continue on with the previous flow
    rop.ll.set_insert_point(after_block);

    return true;
}



static llvm::Value*
llvm_batched_trace_options(BatchedBackendLLVM& rop, int opnum,
                           int first_optional_arg)
{
    using TraceOpt       = RendererServices::TraceOpt;
    llvm::Value* bto     = rop.temp_batched_trace_options_ptr();
    llvm::Type* bto_type = rop.llvm_type_batched_trace_options();

    // Explicitly assign a default value or an optional parameter value to every data member
    // of TraceOpt.
    llvm::Value* mindist  = rop.ll.constant(0.0f);
    llvm::Value* maxdist  = rop.ll.constant(1.0e30f);
    llvm::Value* shade    = rop.ll.constant(0);
    llvm::Value* traceset = rop.ll.constant_ptr(nullptr);

    bool is_mindist_uniform  = true;
    bool is_maxdist_uniform  = true;
    bool is_shade_uniform    = true;
    bool is_traceset_uniform = true;

    Opcode& op(rop.inst()->ops()[opnum]);
    for (int a = first_optional_arg; a < op.nargs(); ++a) {
        Symbol& Name(*rop.opargsym(op, a));
        OSL_DASSERT(Name.typespec().is_string()
                    && "optional trace token must be a string");
        OSL_DASSERT(a + 1 < op.nargs() && "malformed argument list for trace");
        ustring name = *(ustring*)Name.data();
        ++a;  // advance to next argument (value)

        if (name.empty())  // skip empty string param name
            continue;
        Symbol& Val(*rop.opargsym(op, a));
        TypeDesc valtype = Val.typespec().simpletype();


        bool nameIsVarying = !Name.is_uniform();
        // assuming option names can't be varying
        OSL_ASSERT(!nameIsVarying);
        // data could be varying
        bool valIsVarying = !Val.is_uniform();

#define PARAM_UNIFORM(paramname, paramtype)                     \
    if (name == Strings::paramname && (valtype == paramtype)) { \
        if (valIsVarying) {                                     \
            is_##paramname##_uniform = false;                   \
            continue;                                           \
        }                                                       \
        llvm::Value* val = rop.llvm_load_value(Val);            \
        paramname        = val;                                 \
        continue;                                               \
    }

        PARAM_UNIFORM(mindist, TypeDesc::FLOAT)
        PARAM_UNIFORM(maxdist, TypeDesc::FLOAT)
        PARAM_UNIFORM(shade, TypeDesc::INT)
        PARAM_UNIFORM(traceset, TypeDesc::STRING)

        rop.shadingcontext()->errorfmt(
            "Unknown trace optional argument: \"{}\", <{}> ({}:{})", name,
            valtype, op.sourcefile(), op.sourceline());

#undef PARAM_UNIFORM
    }

    if (is_mindist_uniform)
        rop.ll.op_unmasked_store(
            mindist,
            rop.ll.GEP(bto_type, bto, 0,
                       static_cast<int>(TraceOpt::LLVMMemberIndex::mindist)));

    if (is_maxdist_uniform)
        rop.ll.op_unmasked_store(
            maxdist,
            rop.ll.GEP(bto_type, bto, 0,
                       static_cast<int>(TraceOpt::LLVMMemberIndex::maxdist)));

    if (is_shade_uniform)
        rop.ll.op_unmasked_store(
            shade,
            rop.ll.GEP(bto_type, bto, 0,
                       static_cast<int>(TraceOpt::LLVMMemberIndex::shade)));

    if (is_traceset_uniform)
        rop.ll.op_unmasked_store(
            traceset,
            rop.ll.GEP(bto_type, bto, 0,
                       static_cast<int>(TraceOpt::LLVMMemberIndex::traceset)));

    return rop.ll.void_ptr(bto);
}



llvm::Value*
llvm_batched_trace_varying_options(BatchedBackendLLVM& rop, int opnum,
                                   int first_optional_arg,
                                   llvm::Value* remainingMask,
                                   llvm::Value* leadLane)
{
    using TraceOpt       = RendererServices::TraceOpt;
    llvm::Value* bto     = rop.temp_batched_trace_options_ptr();
    llvm::Type* bto_type = rop.llvm_type_batched_trace_options();

    // The BatchedTraceOptions is local alloca,
    // so no need to mask off non-active lanes

    Opcode& op(rop.inst()->ops()[opnum]);
    for (int a = first_optional_arg; a < op.nargs(); ++a) {
        Symbol& Name(*rop.opargsym(op, a));
        OSL_DASSERT(Name.typespec().is_string()
                    && "optional trace token must be a string");
        OSL_DASSERT(a + 1 < op.nargs()
                    && "malformed argument list for texture");
        ustring name = *(ustring*)Name.data();
        ++a;  // advance to next argument (value)

        if (name.empty())  // skip empty string param name
            continue;
        Symbol& Val(*rop.opargsym(op, a));
        TypeDesc valtype = Val.typespec().simpletype();


        bool nameIsVarying = !Name.is_uniform();
        // assuming option names can't be varying
        OSL_ASSERT(!nameIsVarying);

        // data could be uniform
        bool valIsVarying = !Val.is_uniform();
        if (!valIsVarying)
            continue;

        OSL_ASSERT(!Val.is_constant() && "can't be a varying constant");

#define PARAM_VARYING(paramname, paramtype)                                    \
    if (name == Strings::paramname && valtype == paramtype) {                  \
        llvm::Value* wide_val                                                  \
            = rop.llvm_load_value(Val, /*deriv=*/0, /*component=*/0,           \
                                  TypeDesc::UNKNOWN, /*op_is_uniform=*/false); \
        llvm::Value* scalar_value = rop.ll.op_extract(wide_val, leadLane);     \
        rop.ll.op_unmasked_store(                                              \
            scalar_value,                                                      \
            rop.ll.GEP(bto_type, bto, 0,                                       \
                       static_cast<int>(                                       \
                           TraceOpt::LLVMMemberIndex::paramname)));            \
        remainingMask = rop.ll.op_lanes_that_match_masked(scalar_value,        \
                                                          wide_val,            \
                                                          remainingMask);      \
        continue;                                                              \
    }
        PARAM_VARYING(mindist, TypeDesc::FLOAT)
        PARAM_VARYING(maxdist, TypeDesc::FLOAT)
        PARAM_VARYING(shade, TypeDesc::INT)
        PARAM_VARYING(traceset, TypeDesc::STRING)

        rop.shadingcontext()->errorfmt(
            "Unknown trace optional argument: \"{}\", <{}> ({}:{})", name,
            valtype, op.sourcefile(), op.sourceline());
#undef PARAM_VARYING
    }
    return remainingMask;
}



LLVMGEN(llvm_gen_trace)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    Symbol& Result         = *rop.opargsym(op, 0);
    Symbol& Pos            = *rop.opargsym(op, 1);
    Symbol& Dir            = *rop.opargsym(op, 2);
    int first_optional_arg = 3;

    OSL_ASSERT(!Result.is_uniform());

    // make sure that any temps created by llvm_batched_trace_options
    // and llvm_batched_trace_varying_options
    // are not released until we are done using them!
    BatchedBackendLLVM::TempScope temp_scope(rop);

    llvm::Value* opt;  // TraceOpt
    opt = llvm_batched_trace_options(rop, opnum,
                                     first_optional_arg);  //These are uniform

    // Now call the osl_trace function, passing the options and all the
    // explicit args like trace coordinates.
    llvm::Value* args[10];
    args[0] = rop.sg_void_ptr();
    args[1] = rop.llvm_void_ptr(Result);

    llvm::Value* widePos   = nullptr;
    llvm::Value* widePosDx = nullptr;
    llvm::Value* widePosDy = nullptr;
    if (Pos.is_uniform()) {
        widePos   = rop.llvm_widen_value_into_temp(Pos, 0);
        widePosDx = rop.llvm_widen_value_into_temp(Pos, 1);
        widePosDy = rop.llvm_widen_value_into_temp(Pos, 2);
    } else {
        widePos   = rop.llvm_void_ptr(Pos, 0);
        widePosDx = rop.llvm_void_ptr(Pos, 1);
        widePosDy = rop.llvm_void_ptr(Pos, 2);
    }

    llvm::Value* wideDir   = nullptr;
    llvm::Value* wideDirDx = nullptr;
    llvm::Value* wideDirDy = nullptr;
    if (Dir.is_uniform()) {
        wideDir   = rop.llvm_widen_value_into_temp(Dir, 0);
        wideDirDx = rop.llvm_widen_value_into_temp(Dir, 1);
        wideDirDy = rop.llvm_widen_value_into_temp(Dir, 2);
    } else {
        wideDir   = rop.llvm_void_ptr(Dir, 0);
        wideDirDx = rop.llvm_void_ptr(Dir, 1);
        wideDirDy = rop.llvm_void_ptr(Dir, 2);
    }

    args[2] = opt;
    args[3] = widePos;
    args[4] = widePosDx;
    args[5] = widePosDy;
    args[6] = wideDir;
    args[7] = wideDirDx;
    args[8] = wideDirDy;

    // do while(remaining)
    llvm::Value* loc_of_remainingMask = rop.getTempMask(
        "lanes remaining to trace");
    rop.ll.op_store_mask(rop.ll.current_mask(), loc_of_remainingMask);

    llvm::BasicBlock* bin_block = rop.ll.new_basic_block(
        rop.llvm_debug()
            ? std::string("bin_trace_options (varying trace options)")
            : std::string());
    llvm::BasicBlock* after_block = rop.ll.new_basic_block(
        rop.llvm_debug()
            ? std::string("after_bin_trace_options (varying trace options)")
            : std::string());
    rop.ll.op_branch(bin_block);
    {
        llvm::Value* remainingMask = rop.ll.op_load_mask(loc_of_remainingMask);
        llvm::Value* leadLane = rop.ll.op_1st_active_lane_of(remainingMask);

        //rop.llvm_print_mask("before remainingMask", remainingMask);
        llvm::Value* lanesMatchingOptions
            = llvm_batched_trace_varying_options(rop, opnum, first_optional_arg,
                                                 remainingMask, leadLane);
        OSL_ASSERT(lanesMatchingOptions);
        //rop.llvm_print_mask("lanesMatchingOptions", lanesMatchingOptions);
        args[9] = rop.ll.mask_as_int(lanesMatchingOptions);

        rop.ll.call_function(rop.build_name(FuncSpec("trace").mask()), args);

        remainingMask = rop.ll.op_xor(remainingMask, lanesMatchingOptions);
        //rop.llvm_print_mask("xor remainingMask,lanesMatchingOptions", remainingMask);
        rop.ll.op_store_mask(remainingMask, loc_of_remainingMask);

        llvm::Value* int_remainingMask = rop.ll.mask_as_int(remainingMask);
        //rop.llvm_print_mask("remainingMask", remainingMask);
        llvm::Value* cond_more_lanes_to_bin = rop.ll.op_ne(int_remainingMask,
                                                           rop.ll.constant(0));
        rop.ll.op_branch(cond_more_lanes_to_bin, bin_block, after_block);
    }
    // Continue on with the previous flow
    rop.ll.set_insert_point(after_block);

    return true;
}



// TODO: future optimization, don't call multiple functions to set noise options.
// Instead modify a NoiseOptions directly from LLVM (similar to batched texturing).
static llvm::Value*
llvm_batched_noise_options(BatchedBackendLLVM& rop, int opnum,
                           int first_optional_arg,
                           llvm::Value*& loc_wide_direction,
                           bool& all_options_are_uniform)
{
    llvm::Value* opt = rop.ll.call_function(rop.build_name("get_noise_options"),
                                            rop.sg_void_ptr());
    bool is_anisotropic_uniform = true;
    bool is_bandwidth_uniform   = true;
    bool is_impulses_uniform    = true;
    bool is_do_filter_uniform   = true;

    OSL_DASSERT(loc_wide_direction == nullptr);

    Opcode& op(rop.inst()->ops()[opnum]);
    for (int a = first_optional_arg; a < op.nargs(); ++a) {
        Symbol& Name(*rop.opargsym(op, a));
        OSL_ASSERT(Name.typespec().is_string()
                   && "optional noise token must be a string");
        OSL_ASSERT(a + 1 < op.nargs() && "malformed argument list for noise");
        ustring name = Name.get_string();

        ++a;  // advance to next argument
        Symbol& Val(*rop.opargsym(op, a));
        TypeDesc valtype = Val.typespec().simpletype();

        if (name.empty())  // skip empty string param name
            continue;

        bool nameIsVarying = !Name.is_uniform();
        // assuming option names can't be varying
        OSL_ASSERT(!nameIsVarying);

        // Make sure to skip varying values, but track
        // if option was specified
        if (name == Strings::anisotropic && Val.typespec().is_int()) {
            if (!Val.is_uniform()) {
                is_anisotropic_uniform = false;
                continue;  // We are only setting uniform options here
            }
            rop.ll.call_function("osl_noiseparams_set_anisotropic", opt,
                                 rop.llvm_load_value(Val));
        } else if (name == Strings::do_filter && Val.typespec().is_int()) {
            if (!Val.is_uniform()) {
                is_do_filter_uniform = false;
                continue;  // We are only setting uniform options here
            }
            rop.ll.call_function("osl_noiseparams_set_do_filter", opt,
                                 rop.llvm_load_value(Val));
        } else if (name == Strings::direction && Val.typespec().is_triple()) {
            // We are not going to bin by varying direction
            // instead we will pass a pointer to its wide value
            // as an extra parameter.
            // If it is null, then the uniform value from noise options
            // should be used.
            llvm::Value* loc_of_val = rop.llvm_void_ptr(Val);
            if (!Val.is_uniform()) {
                loc_wide_direction = loc_of_val;
            } else {
                rop.ll.call_function("osl_noiseparams_set_direction", opt,
                                     loc_of_val);
            }
        } else if (name == Strings::bandwidth
                   && (Val.typespec().is_float() || Val.typespec().is_int())) {
            if (!Val.is_uniform()) {
                is_bandwidth_uniform = false;
                continue;  // We are only setting uniform options here
            }
            rop.ll.call_function("osl_noiseparams_set_bandwidth", opt,
                                 rop.llvm_load_value(Val, 0, NULL, 0,
                                                     TypeFloat));
        } else if (name == Strings::impulses
                   && (Val.typespec().is_float() || Val.typespec().is_int())) {
            if (!Val.is_uniform()) {
                is_impulses_uniform = false;
                continue;  // We are only setting uniform options here
            }
            rop.ll.call_function("osl_noiseparams_set_impulses", opt,
                                 rop.llvm_load_value(Val, 0, NULL, 0,
                                                     TypeFloat));
        } else {
            rop.shadingcontext()->errorfmt(
                "Unknown {} optional argument: \"{}\", <{}> ({}:{})",
                op.opname(), name, valtype, op.sourcefile(), op.sourceline());
        }
    }

    // NOTE: may have been previously set to false if name wasn't uniform
    all_options_are_uniform &= is_anisotropic_uniform && is_bandwidth_uniform
                               && is_impulses_uniform && is_do_filter_uniform;

    return opt;
}



static llvm::Value*
llvm_batched_noise_varying_options(BatchedBackendLLVM& rop, int opnum,
                                   int first_optional_arg, llvm::Value* opt,
                                   llvm::Value* remainingMask,
                                   llvm::Value* leadLane)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    for (int a = first_optional_arg; a < op.nargs(); ++a) {
        Symbol& Name(*rop.opargsym(op, a));
        OSL_ASSERT(Name.typespec().is_string()
                   && "optional noise token must be a string");
        OSL_ASSERT(a + 1 < op.nargs() && "malformed argument list for noise");
        ustring name = Name.get_string();

        ++a;  // advance to next argument
        Symbol& Val(*rop.opargsym(op, a));
        TypeDesc valtype = Val.typespec().simpletype();

        if (name.empty())  // skip empty string param name
            continue;

        bool nameIsVarying = !Name.is_uniform();
        // assuming option names can't be varying
        OSL_ASSERT(!nameIsVarying);
        // data could be uniform
        if (Val.is_uniform())
            continue;

        OSL_ASSERT(!Val.is_constant() && "can't be a varying constant");

        if (name == Strings::anisotropic && Val.typespec().is_int()) {
            OSL_DEV_ONLY(std::cout << "Varying anisotropic" << std::endl);
            llvm::Value* wide_anisotropic
                = rop.llvm_load_value(Val,
                                      /*deriv=*/0, /*component=*/0,
                                      /*cast=*/TypeDesc::UNKNOWN,
                                      /*op_is_uniform=*/false);
            llvm::Value* scalar_anisotropic
                = rop.ll.op_extract(wide_anisotropic, leadLane);
            remainingMask = rop.ll.op_lanes_that_match_masked(
                scalar_anisotropic, wide_anisotropic, remainingMask);
            rop.ll.call_function("osl_noiseparams_set_anisotropic", opt,
                                 scalar_anisotropic);
        } else if (name == Strings::do_filter && Val.typespec().is_int()) {
            OSL_DEV_ONLY(std::cout << "Varying do_filter" << std::endl);
            llvm::Value* wide_do_filter
                = rop.llvm_load_value(Val,
                                      /*deriv=*/0, /*component=*/0,
                                      /*cast=*/TypeDesc::UNKNOWN,
                                      /*op_is_uniform=*/false);
            llvm::Value* scalar_do_filter = rop.ll.op_extract(wide_do_filter,
                                                              leadLane);
            remainingMask = rop.ll.op_lanes_that_match_masked(scalar_do_filter,
                                                              wide_do_filter,
                                                              remainingMask);
            rop.ll.call_function("osl_noiseparams_set_do_filter", opt,
                                 scalar_do_filter);
        } else if (name == Strings::bandwidth
                   && (Val.typespec().is_float() || Val.typespec().is_int())) {
            OSL_DEV_ONLY(std::cout << "Varying bandwidth" << std::endl);
            llvm::Value* wide_bandwidth
                = rop.llvm_load_value(Val,
                                      /*deriv=*/0, /*component=*/0,
                                      /*cast=*/TypeFloat,
                                      /*op_is_uniform=*/false);
            llvm::Value* scalar_bandwidth = rop.ll.op_extract(wide_bandwidth,
                                                              leadLane);
            remainingMask = rop.ll.op_lanes_that_match_masked(scalar_bandwidth,
                                                              wide_bandwidth,
                                                              remainingMask);
            rop.ll.call_function("osl_noiseparams_set_bandwidth", opt,
                                 scalar_bandwidth);
        } else if (name == Strings::impulses
                   && (Val.typespec().is_float() || Val.typespec().is_int())) {
            OSL_DEV_ONLY(std::cout << "Varying impulses" << std::endl);
            llvm::Value* wide_impulses
                = rop.llvm_load_value(Val,
                                      /*deriv=*/0, /*component=*/0,
                                      /*cast=*/TypeFloat,
                                      /*op_is_uniform=*/false);
            llvm::Value* scalar_impulses = rop.ll.op_extract(wide_impulses,
                                                             leadLane);
            remainingMask = rop.ll.op_lanes_that_match_masked(scalar_impulses,
                                                              wide_impulses,
                                                              remainingMask);
            rop.ll.call_function("osl_noiseparams_set_impulses", opt,
                                 scalar_impulses);
        } else if (name == Strings::direction && Val.typespec().is_triple()) {
            OSL_DEV_ONLY(std::cout << "Varying direction" << std::endl);
            // As we passed the pointer to the varying direction along
            // with the uniform noise options, there is no need to
            // do any binning for the varying direction.
            continue;
        } else {
            rop.shadingcontext()->errorfmt(
                "Unknown {} optional argument: \"{}\", <{}> ({}:{})",
                op.opname(), name, valtype, op.sourcefile(), op.sourceline());
        }
    }
    return remainingMask;
}



// T noise ([string name,] float s, ...);
// T noise ([string name,] float s, float t, ...);
// T noise ([string name,] point P, ...);
// T noise ([string name,] point P, float t, ...);
// T pnoise ([string name,] float s, float sper, ...);
// T pnoise ([string name,] float s, float t, float sper, float tper, ...);
// T pnoise ([string name,] point P, point Pper, ...);
// T pnoise ([string name,] point P, float t, point Pper, float tper, ...);
LLVMGEN(llvm_gen_noise)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    bool periodic = (op.opname() == Strings::pnoise
                     || op.opname() == Strings::psnoise);

    int arg        = 0;  // Next arg to read
    Symbol& Result = *rop.opargsym(op, arg++);

    // TODO: We could check all parameters to see if we can operate in uniform fashion
    // and just broadcast the result if needed
    bool op_is_uniform = Result.is_uniform();
    OSL_DEV_ONLY(std::cout << "llvm_gen_noise op_is_uniform=" << op_is_uniform
                           << std::endl);

    BatchedBackendLLVM::TempScope temp_scope(rop);

    int outdim   = Result.typespec().is_triple() ? 3 : 1;
    Symbol* Name = rop.opargsym(op, arg++);
    ustring name;
    // NOTE: Name may not be a string, in which case we can treat it as uniform
    bool name_is_uniform = true;
    if (Name->typespec().is_string()) {
        name            = Name->is_constant() ? Name->get_string() : ustring();
        name_is_uniform = Name->is_uniform();
    } else {
        // Not a string, must be the old-style noise/pnoise
        --arg;  // forget that arg
        Name = NULL;
        name = op.opname();
    }

    Symbol *S = rop.opargsym(op, arg++), *T = NULL;
    Symbol *Sper = NULL, *Tper = NULL;
    int indim   = S->typespec().is_triple() ? 3 : 1;
    bool derivs = S->has_derivs();

    if (periodic) {
        if (op.nargs() > (arg + 1)
            && (rop.opargsym(op, arg + 1)->typespec().is_float()
                || rop.opargsym(op, arg + 1)->typespec().is_triple())) {
            // 2D or 4D
            ++indim;
            T = rop.opargsym(op, arg++);
            derivs |= T->has_derivs();
        }
        Sper = rop.opargsym(op, arg++);
        if (indim == 2 || indim == 4)
            Tper = rop.opargsym(op, arg++);
    } else {
        // non-periodic case
        if (op.nargs() > arg && rop.opargsym(op, arg)->typespec().is_float()) {
            // either 2D or 4D, so needs a second index
            ++indim;
            T = rop.opargsym(op, arg++);
            derivs |= T->has_derivs();
        }
    }
    derivs &= Result.has_derivs();  // ignore derivs if result doesn't need

    bool pass_name = false, pass_sg = false, pass_options = false;
    bool all_options_are_uniform = true;
    if (name.empty()) {
        // name is not a constant
        name      = periodic ? Strings::genericpnoise : Strings::genericnoise;
        pass_name = true;
        pass_sg   = true;
        pass_options = true;
        derivs       = true;  // always take derivs if we don't know noise type
        all_options_are_uniform &= name_is_uniform;
    } else if (name == Strings::perlin || name == Strings::snoise
               || name == Strings::psnoise) {
        name = periodic ? Strings::psnoise : Strings::snoise;
        // derivs = false;
    } else if (name == Strings::uperlin || name == Strings::noise
               || name == Strings::pnoise) {
        name = periodic ? Strings::pnoise : Strings::noise;
        // derivs = false;
    } else if (name == Strings::cell || name == Strings::cellnoise) {
        name   = periodic ? Strings::pcellnoise : Strings::cellnoise;
        derivs = false;  // cell noise derivs are always zero
    } else if (name == Strings::hash || name == Strings::hashnoise) {
        name   = periodic ? Strings::phashnoise : Strings::hashnoise;
        derivs = false;  // hash noise derivs are always zero
    } else if (name == Strings::simplex && !periodic) {
        name = Strings::simplexnoise;
    } else if (name == Strings::usimplex && !periodic) {
        name = Strings::usimplexnoise;
    } else if (name == Strings::gabor) {
        // already named
        pass_name    = true;
        pass_sg      = true;
        pass_options = true;
        derivs       = true;
        name         = periodic ? Strings::gaborpnoise : Strings::gabornoise;
    } else {
        rop.shadingcontext()->errorfmt(
            "{}noise type \"{}\" is unknown, called from ({}:{})",
            (periodic ? "periodic " : ""), name, op.sourcefile(),
            op.sourceline());
        return false;
    }

    if (rop.shadingsys().no_noise()) {
        // renderer option to replace noise with constant value. This can be
        // useful as a profiling aid, to see how much it speeds up to have
        // trivial expense for noise calls.
        if (name == Strings::uperlin || name == Strings::noise
            || name == Strings::usimplexnoise || name == Strings::usimplex
            || name == Strings::cell || name == Strings::cellnoise
            || name == Strings::hash || name == Strings::hashnoise
            || name == Strings::pcellnoise || name == Strings::pnoise)
            name = ustring("unullnoise");
        else
            name = ustring("nullnoise");
        pass_name    = false;
        periodic     = false;
        pass_sg      = false;
        pass_options = false;
    }

    llvm::Value* opt                = NULL;
    llvm::Value* loc_wide_direction = nullptr;
    if (pass_options) {
        opt = llvm_batched_noise_options(rop, opnum, arg, loc_wide_direction,
                                         all_options_are_uniform);
    }

    OSL_DEV_ONLY(std::cout << "llvm_gen_noise function name=" << name
                           << std::endl);


    FuncSpec func_spec(name.c_str());
    func_spec.arg(Result, derivs, op_is_uniform);
    std::vector<llvm::Value*> args;

    llvm::Value* nameVal  = nullptr;
    int nameArgumentIndex = -1;
    if (pass_name) {
        nameArgumentIndex = args.size();
        nameVal = rop.llvm_load_value(*Name, /*deriv=*/0, /*component=*/0,
                                      /*cast=*/TypeDesc::UNKNOWN,
                                      name_is_uniform);
        // If we are binning the name, we will replace this
        // argument later in the binning code;
        args.push_back(nameVal);
    }
    llvm::Value* tmpresult = NULL;

    // triple return, or float return with derivs, passes result pointer
    // Always pass result as we can't return a wide type through C ABI
    if (outdim == 3 || derivs || !op_is_uniform) {
        if (derivs && !Result.has_derivs()) {
            tmpresult = rop.llvm_load_arg(Result, true, op_is_uniform);
            args.push_back(tmpresult);
        } else
            args.push_back(rop.llvm_void_ptr(Result));
    }
    func_spec.arg(*S, derivs, op_is_uniform);
    args.push_back(rop.llvm_load_arg(*S, derivs, op_is_uniform));
    if (T) {
        func_spec.arg(*T, derivs, op_is_uniform);
        args.push_back(rop.llvm_load_arg(*T, derivs, op_is_uniform));
    }

    if (periodic) {
        func_spec.arg(*Sper, false /* no derivs */, op_is_uniform);
        args.push_back(rop.llvm_load_arg(*Sper, false, op_is_uniform));
        if (Tper) {
            func_spec.arg(*Tper, false /* no derivs */, op_is_uniform);
            args.push_back(rop.llvm_load_arg(*Tper, false, op_is_uniform));
        }
    }

    if (pass_sg)
        args.push_back(rop.sg_void_ptr());
    if (pass_options) {
        args.push_back(opt);
        // The non wide versions don't take a varying direction
        // so don't push it on the argument list
        if (!op_is_uniform) {
            if (loc_wide_direction == nullptr) {
                loc_wide_direction = rop.ll.void_ptr_null();
            }
            args.push_back(loc_wide_direction);
        }
    }

#ifdef OSL_DEV
    std::cout << "About to push " << rop.build_name(func_spec) << "\n";
    for (size_t i = 0; i < args.size(); ++i) {
        {
            llvm::raw_os_ostream os_cout(std::cout);
            args[i]->print(os_cout);
        }
        std::cout << "\n";
    }
#endif

    if (pass_options && !all_options_are_uniform) {
        OSL_DASSERT(!op_is_uniform);
        func_spec.mask();

        // do while(remaining)
        llvm::Value* loc_of_remainingMask = rop.getTempMask(
            "lanes remaining to gen noise");
        rop.ll.op_store_mask(rop.ll.current_mask(), loc_of_remainingMask);

        llvm::BasicBlock* bin_block = rop.ll.new_basic_block(
            rop.llvm_debug()
                ? std::string("bin_noise_options (varying noise options)")
                : std::string());
        llvm::BasicBlock* after_block = rop.ll.new_basic_block(
            rop.llvm_debug()
                ? std::string("after_bin_noise_options (varying noise options)")
                : std::string());
        rop.ll.op_branch(bin_block);
        {
            llvm::Value* remainingMask = rop.ll.op_load_mask(
                loc_of_remainingMask);
            llvm::Value* leadLane = rop.ll.op_1st_active_lane_of(remainingMask);
            llvm::Value* lanesMatchingName = remainingMask;
#ifdef __OSL_TRACE_MASKS
            rop.llvm_print_mask("before remainingMask", remainingMask);
#endif

            if (false == name_is_uniform) {
                llvm::Value* scalar_name = rop.ll.op_extract(nameVal, leadLane);
                args[nameArgumentIndex]  = scalar_name;
                lanesMatchingName
                    = rop.ll.op_lanes_that_match_masked(scalar_name, nameVal,
                                                        lanesMatchingName);
            }

            llvm::Value* lanesMatchingOptions
                = llvm_batched_noise_varying_options(rop, opnum, arg, opt,
                                                     lanesMatchingName,
                                                     leadLane);

            OSL_ASSERT(lanesMatchingOptions);
#ifdef __OSL_TRACE_MASKS
            rop.llvm_print_mask("lanesMatchingOptions", lanesMatchingOptions);
#endif
            args.push_back(rop.ll.mask_as_int(lanesMatchingOptions));

            rop.ll.call_function(rop.build_name(func_spec), args);

            remainingMask = rop.ll.op_xor(remainingMask, lanesMatchingOptions);
#ifdef __OSL_TRACE_MASKS
            rop.llvm_print_mask("xor remainingMask,lanesMatchingOptions",
                                remainingMask);
#endif
            rop.ll.op_store_mask(remainingMask, loc_of_remainingMask);

            llvm::Value* int_remainingMask = rop.ll.mask_as_int(remainingMask);
#ifdef __OSL_TRACE_MASKS
            rop.llvm_print_mask("remainingMask", remainingMask);
#endif
            llvm::Value* cond_more_lanes_to_bin
                = rop.ll.op_ne(int_remainingMask, rop.ll.constant(0));
            rop.ll.op_branch(cond_more_lanes_to_bin, bin_block, after_block);
        }
        // Continue on with the previous flow
        rop.ll.set_insert_point(after_block);
    } else {
        if (!op_is_uniform) {
            // force masking, but wait push it on as we might be binning for options
            args.push_back(rop.ll.mask_as_int(rop.ll.current_mask()));
            func_spec.mask();
        } else {
            func_spec.unbatch();
        }

        llvm::Value* r = rop.ll.call_function(rop.build_name(func_spec), args);
        if (op_is_uniform && outdim == 1 && !derivs) {
            // Just plain float (no derivs) returns its value
            rop.llvm_store_value(r, Result);
        }
    }
    if (derivs && !Result.has_derivs()) {
        // Function needed to take derivs, but our result doesn't have them.
        // We created a temp, now we need to copy to the real result.

        //tmpresult = rop.llvm_ptr_cast (tmpresult, Result.typespec());
        if (op_is_uniform)
            tmpresult = rop.llvm_ptr_cast(tmpresult, Result.typespec());
        else
            tmpresult = rop.llvm_wide_ptr_cast(tmpresult, Result.typespec());

        for (int c = 0; c < Result.typespec().aggregate(); ++c) {
            llvm::Value* v = rop.llvm_load_value(tmpresult, Result.typespec(),
                                                 0, NULL, c, op_is_uniform,
                                                 TypeDesc::UNKNOWN,
                                                 op_is_uniform);
            rop.llvm_store_value(v, Result, 0, c);
        }
    }  // N.B. other cases already stored their result in the right place

    // Clear derivs if result has them but we couldn't compute them
    if (Result.has_derivs() && !derivs)
        rop.llvm_zero_derivs(Result);

    if (rop.shadingsys().profile() >= 1)
        rop.ll.call_function(rop.build_name(FuncSpec("count_noise").mask()),
                             { rop.sg_void_ptr(),
                               rop.ll.mask_as_int(rop.ll.current_mask()) });

    return true;
}



LLVMGEN(llvm_gen_getattribute)
{
    // getattribute() has eight "flavors":
    //   * getattribute (attribute_name, value)
    //   * getattribute (attribute_name, value[])
    //   * getattribute (attribute_name, index, value)
    //   * getattribute (attribute_name, index, value[])
    //   * getattribute (object, attribute_name, value)
    //   * getattribute (object, attribute_name, value[])
    //   * getattribute (object, attribute_name, index, value)
    //   * getattribute (object, attribute_name, index, value[])
    Opcode& op(rop.inst()->ops()[opnum]);
    int nargs = op.nargs();
    OSL_DASSERT(nargs >= 3 && nargs <= 5);

    bool array_lookup  = rop.opargsym(op, nargs - 2)->typespec().is_int();
    bool object_lookup = rop.opargsym(op, 2)->typespec().is_string()
                         && nargs >= 4;
    int object_slot = (int)object_lookup;
    int attrib_slot = object_slot + 1;
    int index_slot  = array_lookup ? nargs - 2 : 0;

    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& ObjectName
        = *rop.opargsym(op, object_slot);  // only valid if object_slot is true
    Symbol& Attribute = *rop.opargsym(op, attrib_slot);
    Symbol& Index
        = *rop.opargsym(op, index_slot);  // only valid if array_lookup is true
    Symbol& Destination = *rop.opargsym(op, nargs - 1);
    OSL_DASSERT(!Result.typespec().is_closure_based()
                && !ObjectName.typespec().is_closure_based()
                && !Attribute.typespec().is_closure_based()
                && !Index.typespec().is_closure_based()
                && !Destination.typespec().is_closure_based());

    // Special case for get attributes where the result uniformity can differ
    // from the callback
    bool result_is_uniform      = Result.is_uniform();
    bool destination_is_uniform = Destination.is_uniform();
    bool attribute_is_uniform   = Attribute.is_uniform();

    OSL_ASSERT((!array_lookup || Index.is_uniform()) && "incomplete");
    OSL_ASSERT((!object_lookup || ObjectName.is_uniform()) && "incomplete");

    // The analysis flag was populated by BatchedAnalysis and
    // indicates if the render will provide a uniform result
    bool op_is_uniform = op.analysis_flag();

    // We'll pass the destination's attribute type directly to the
    // RenderServices callback so that the renderer can perform any
    // necessary conversions from its internal format to OSL's.
    const TypeDesc* dest_type = &Destination.typespec().simpletype();

    if (false == op_is_uniform) {
        OSL_ASSERT((!result_is_uniform) && (!destination_is_uniform));

        llvm::Value* args[]
            = { rop.sg_void_ptr(),
                rop.ll.constant((int)Destination.has_derivs()),
                object_lookup ? rop.llvm_load_value(ObjectName)
                              : rop.ll.constant(ustring()),
                attribute_is_uniform ? rop.llvm_load_value(Attribute)
                                     : rop.llvm_void_ptr(Attribute),
                rop.ll.constant((int)array_lookup),
                array_lookup ? rop.llvm_load_value(Index)
                             : rop.ll.constant((
                                 int)0),  // Never load a symbol that is invalid
                rop.ll.constant_ptr((void*)dest_type),
                rop.llvm_void_ptr(Destination),
                rop.ll.mask_as_int(rop.ll.current_mask()) };

        FuncSpec func_spec("get_attribute");
        func_spec.arg(Attribute, attribute_is_uniform);
        if (!attribute_is_uniform) {
            func_spec.mask();
        }

        llvm::Value* r = rop.ll.call_function(rop.build_name(func_spec), args);
        rop.llvm_conversion_store_masked_status(r, Result);
    } else {
        OSL_ASSERT((!object_lookup || ObjectName.is_uniform())
                   && Attribute.is_uniform());

        BatchedBackendLLVM::TempScope temp_scope(rop);
        llvm::Value* tempUniformDestination = nullptr;
        llvm::Value* uniformDestination     = nullptr;
        if (destination_is_uniform) {
            uniformDestination = rop.llvm_void_ptr(Destination);
        } else {
            tempUniformDestination
                = rop.getOrAllocateTemp(Destination.typespec(),
                                        Destination.has_derivs(),
                                        true /*is_uniform*/);
            uniformDestination = rop.ll.void_ptr(tempUniformDestination);
        }

        llvm::Value* args[]
            = { rop.sg_void_ptr(),
                rop.ll.constant((int)Destination.has_derivs()),
                object_lookup ? rop.llvm_load_value(ObjectName)
                              : rop.ll.constant(ustring()),
                rop.llvm_load_value(Attribute),
                rop.ll.constant((int)array_lookup),
                array_lookup ? rop.llvm_load_value(Index)
                             : rop.ll.constant((
                                 int)0),  // Never load a symbol that is invalid
                rop.ll.constant_ptr((void*)dest_type),
                uniformDestination };

        llvm::Value* r = rop.ll.call_function(
            rop.build_name(FuncSpec("get_attribute_uniform")), args);

        if (!destination_is_uniform) {
            // Only broadcast our result if the value lookup succeeded
            // Branch on the condition, to our blocks
            llvm::Value* cond_val             = rop.ll.op_int_to_bool(r);
            llvm::BasicBlock* broadcast_block = rop.ll.new_basic_block(
                std::string("uniform getattribute result broadcast"));
            llvm::BasicBlock* after_block = rop.ll.new_basic_block(
                std::string("after uniform getattribute result broadcast"));
            rop.ll.op_branch(cond_val, broadcast_block, after_block);

            rop.ll.set_insert_point(broadcast_block);
            rop.llvm_broadcast_uniform_value_from_mem(tempUniformDestination,
                                                      Destination);
            rop.ll.op_branch(after_block);

            rop.ll.set_insert_point(after_block);
        }
        rop.llvm_conversion_store_uniform_status(r, Result);
    }

    return true;
}



LLVMGEN(llvm_gen_gettextureinfo)
{
    OSL_DEV_ONLY(std::cout << "llvm_gen_gettextureinfo" << std::endl);
    Opcode& op(rop.inst()->ops()[opnum]);

    OSL_DASSERT(op.nargs() == 4 || op.nargs() == 6);
    bool use_coords  = (op.nargs() == 6);
    Symbol& Result   = *rop.opargsym(op, 0);
    Symbol& Filename = *rop.opargsym(op, 1);
    Symbol& Dataname = *rop.opargsym(op, use_coords ? 4 : 2);
    Symbol& Data     = *rop.opargsym(op, use_coords ? 5 : 3);
    Symbol* S        = use_coords ? rop.opargsym(op, 2) : nullptr;
    Symbol* T        = use_coords ? rop.opargsym(op, 3) : nullptr;

    OSL_DASSERT(
        !Result.typespec().is_closure_based() && Filename.typespec().is_string()
        && (S == nullptr || S->typespec().is_float())
        && (T == nullptr || T->typespec().is_float())
        && Dataname.typespec().is_string()
        && !Data.typespec().is_closure_based() && Result.typespec().is_int());

    const TypeDesc* dest_type = &Data.typespec().simpletype();

    std::vector<llvm::Value*> args;
    args.push_back(rop.sg_void_ptr());

    // We will just load the spline name here if we are uniform,
    // otherwise just remember where the spline name
    // is so we can update it later in the loop over varying spline names
    int fileNameArgumentIndex = args.size();
    llvm::Value* fileNameVal
        = rop.llvm_load_value(Filename, /*deriv=*/0, /*component=*/0,
                              TypeDesc::UNKNOWN, Filename.is_uniform());
    args.push_back(Filename.is_uniform() ? fileNameVal : nullptr);

    RendererServices::TextureHandle* texture_handle = NULL;
    if (Filename.is_constant() && rop.shadingsys().opt_texture_handle()) {
        texture_handle
            = rop.renderer()->get_texture_handle(Filename.get_string(),
                                                 rop.shadingcontext());
        if (!rop.renderer()->is_udim(texture_handle))
            use_coords = false;
    }

    BatchedBackendLLVM::TempScope temp_scope(rop);
    llvm::Value* loc_texture_handles  = nullptr;
    llvm::Value* udim_texture_handles = nullptr;
    llvm::Value* resolve_udim_args[7];
    constexpr int udim_fileName_arg_index = 1;
    constexpr int udim_mask_arg_index     = 6;
    std::string resolve_udim_name;
    // BatchedAnalysis should have made Data varying if any of the inputs (include S and T)
    // were varying, so we can just check it to see if we can resolve uniformly or not
    bool resolve_is_uniform = Data.is_uniform();
    if (use_coords) {
        FuncSpec resolve_udim_spec(resolve_is_uniform ? "resolve_udim_uniform"
                                                      : "resolve_udim");
        if (!resolve_is_uniform) {
            loc_texture_handles = rop.getOrAllocateTemp(TypeSpec(TypeDesc::PTR),
                                                        false /*has_derivs*/,
                                                        false /*is_uniform*/);
            resolve_udim_spec.mask();
        }
        // NOTE: we make a copy so the build name internal buffer can be reused.
        resolve_udim_name    = rop.build_name(resolve_udim_spec);
        resolve_udim_args[0] = rop.sg_void_ptr();
        resolve_udim_args[1] = Filename.is_uniform() ? fileNameVal : nullptr;
        resolve_udim_args[2] = rop.ll.constant_ptr(texture_handle);
        resolve_udim_args[3] = rop.llvm_load_arg(*S, /*has_derivs*/ false,
                                                 resolve_is_uniform);
        resolve_udim_args[4] = rop.llvm_load_arg(*T, /*has_derivs*/ false,
                                                 resolve_is_uniform);

        if (!resolve_is_uniform) {
            resolve_udim_args[5] = rop.ll.void_ptr(loc_texture_handles);
            resolve_udim_args[6] = nullptr;  // mask_as_int
            if (Filename.is_uniform()) {
                resolve_udim_args[udim_mask_arg_index] = rop.ll.mask_as_int(
                    rop.ll.current_mask());
                rop.ll.call_function(resolve_udim_name.c_str(),
                                     resolve_udim_args);
                udim_texture_handles
                    = rop.ll.op_load(rop.ll.type_wide_void_ptr(),
                                     loc_texture_handles);
            }
        }
    }

    int textureHandleArgumentIndex    = args.size();
    llvm::Value* texture_handle_value = nullptr;
    if (use_coords && resolve_is_uniform) {
        texture_handle_value
            = rop.ll.call_function(resolve_udim_name.c_str(),
                                   cspan<llvm::Value*>(resolve_udim_args, 5));
        // because the resolve was uniform and we just resolved the texture handle
        // no need to do binning because of coords
        use_coords = false;
    } else {
        texture_handle_value = rop.ll.constant_ptr(texture_handle);
    }
    args.push_back(texture_handle_value);

    int dataNameArgumentIndex = args.size();
    llvm::Value* dataNameVal
        = rop.llvm_load_value(Dataname, /*deriv=*/0, /*component=*/0,
                              TypeDesc::UNKNOWN, Dataname.is_uniform());
    args.push_back(Dataname.is_uniform() ? dataNameVal : nullptr);

    // this is passes a TypeDesc to an LLVM op-code
    // NOTE: not GPU friendly to pass pointer to host memory,
    //       but this is CPU only right now
    args.push_back(rop.ll.constant_ptr((void*)dest_type));
    // destination
    llvm::Value* tempUniformData = nullptr;
    if (Data.is_uniform()) {
        args.push_back(rop.llvm_void_ptr(Data));
    } else {
        tempUniformData = rop.getOrAllocateTemp(Data.typespec(),
                                                Data.has_derivs(),
                                                true /*is_uniform*/);
        args.push_back(rop.ll.void_ptr(tempUniformData));
    }

    FuncSpec func_spec("get_textureinfo_uniform");
    const char* funcName = rop.build_name(func_spec);

    bool requires_binning = (!Filename.is_uniform()) || (!Dataname.is_uniform())
                            || use_coords;
    if (requires_binning) {
        // Result mask starts at 0 and success is bitwise OR'd in
        llvm::Value* loc_of_successMask = rop.getTempMask(
            "lanes succeeded to gettextureinfo");
        rop.ll.op_store_mask(rop.ll.int_as_mask(rop.ll.constant(0)),
                             loc_of_successMask);
        // do while(remaining)
        llvm::Value* loc_of_remainingMask = rop.getTempMask(
            "lanes remaining to gettextureinfo");
        rop.ll.op_store_mask(rop.ll.current_mask(), loc_of_remainingMask);

        llvm::BasicBlock* bin_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("bin_gettextureinfo")
                             : std::string());
        llvm::BasicBlock* after_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("after_bin_gettextureinfo")
                             : std::string());
        rop.ll.op_branch(bin_block);
        {
            llvm::Value* remainingMask = rop.ll.op_load_mask(
                loc_of_remainingMask);
            llvm::Value* leadLane = rop.ll.op_1st_active_lane_of(remainingMask);
            llvm::Value* lanesMatching = remainingMask;

            if (!Filename.is_uniform()) {
                llvm::Value* scalar_fileName = rop.ll.op_extract(fileNameVal,
                                                                 leadLane);
                args[fileNameArgumentIndex]  = scalar_fileName;
                lanesMatching = rop.ll.op_lanes_that_match_masked(
                    scalar_fileName, fileNameVal, remainingMask);

                if (use_coords) {
                    resolve_udim_args[udim_fileName_arg_index] = scalar_fileName;
                    resolve_udim_args[udim_mask_arg_index] = rop.ll.mask_as_int(
                        lanesMatching);
                    rop.ll.call_function(resolve_udim_name.c_str(),
                                         resolve_udim_args);
                    udim_texture_handles
                        = rop.ll.op_load(rop.ll.type_wide_void_ptr(),
                                         loc_texture_handles);
                }
            }
            OSL_ASSERT(lanesMatching);

            if (use_coords) {
                OSL_ASSERT(udim_texture_handles != nullptr);
                llvm::Value* scalar_textureHandle
                    = rop.ll.op_extract(udim_texture_handles, leadLane);
                args[textureHandleArgumentIndex] = scalar_textureHandle;
                lanesMatching = rop.ll.op_lanes_that_match_masked(
                    scalar_textureHandle, udim_texture_handles, lanesMatching);
            }

            if (!Dataname.is_uniform()) {
                llvm::Value* scalar_dataName = rop.ll.op_extract(dataNameVal,
                                                                 leadLane);
                args[dataNameArgumentIndex]  = scalar_dataName;
                lanesMatching = rop.ll.op_lanes_that_match_masked(
                    scalar_dataName, dataNameVal, lanesMatching);
            }
            //rop.llvm_print_mask("lanesMatching", lanesMatching);
            rop.ll.push_mask(lanesMatching);

            llvm::Value* success = rop.ll.call_function(funcName, args);

            // Only broadcast our result if the value lookup succeeded
            // Branch on the condition, to our blocks
            llvm::Value* broadcast_cond_val   = rop.ll.op_int_to_bool(success);
            llvm::BasicBlock* broadcast_block = rop.ll.new_basic_block(
                std::string("uniform gettextureinfo broadcast"));
            llvm::BasicBlock* after_broadcast_block = rop.ll.new_basic_block(
                std::string("after uniform gettextureinfo broadcast"));
            rop.ll.op_branch(broadcast_cond_val, broadcast_block,
                             after_broadcast_block);

            rop.ll.set_insert_point(broadcast_block);
            llvm::Value* success_mask = rop.ll.op_load_mask(loc_of_successMask);
            success_mask = rop.ll.op_or(success_mask, lanesMatching);
            rop.ll.op_store_mask(success_mask, loc_of_successMask);
            // this broadcast should respect current mask stack
            OSL_ASSERT(false == Data.is_uniform());
            rop.llvm_broadcast_uniform_value_from_mem(tempUniformData, Data);
            rop.ll.op_branch(after_broadcast_block);
            rop.ll.set_insert_point(after_broadcast_block);

            rop.ll.pop_mask();  // lanes matching

            remainingMask = rop.ll.op_xor(remainingMask, lanesMatching);
            //rop.llvm_print_mask("xor remainingMask,lanesMatchingSplineName", remainingMask);
            rop.ll.op_store_mask(remainingMask, loc_of_remainingMask);

            llvm::Value* int_remainingMask = rop.ll.mask_as_int(remainingMask);
            //rop.llvm_print_mask("remainingMask", remainingMask);
            llvm::Value* cond_more_lanes_to_bin
                = rop.ll.op_ne(int_remainingMask, rop.ll.constant(0));
            rop.ll.op_branch(cond_more_lanes_to_bin, bin_block, after_block);
        }
        // Continue on with the previous flow
        rop.ll.set_insert_point(after_block);
        OSL_ASSERT(false == Result.is_uniform());
        llvm::Value* success_mask = rop.ll.op_load_mask(loc_of_successMask);
        rop.llvm_conversion_store_masked_status(rop.ll.mask_as_int(success_mask),
                                                Result);
    } else {
        llvm::Value* success = rop.ll.call_function(funcName, args);

        if (!Data.is_uniform()) {
            // Only broadcast our result if the value lookup succeeded
            // Branch on the condition, to our blocks
            llvm::Value* broadcast_cond_val   = rop.ll.op_int_to_bool(success);
            llvm::BasicBlock* broadcast_block = rop.ll.new_basic_block(
                std::string("uniform gettextureinfo broadcast"));
            llvm::BasicBlock* after_broadcast_block = rop.ll.new_basic_block(
                std::string("after uniform gettextureinfo broadcast"));
            rop.ll.op_branch(broadcast_cond_val, broadcast_block,
                             after_broadcast_block);

            rop.ll.set_insert_point(broadcast_block);
            // this broadcast should respect current mask stack
            rop.llvm_broadcast_uniform_value_from_mem(tempUniformData, Data);
            rop.ll.op_branch(after_broadcast_block);
            rop.ll.set_insert_point(after_broadcast_block);
        }

        if (!Result.is_uniform()) {
            rop.llvm_broadcast_uniform_value(success, Result);
        } else {
            rop.llvm_store_value(success, Result);
        }
    }

    /* Do not leave derivs uninitialized */
    if (Data.has_derivs())
        rop.llvm_zero_derivs(Data);

    return true;
}



LLVMGEN(llvm_gen_getmessage)
{
    // getmessage() has four "flavors":
    //   * getmessage (attribute_name, value)
    //   * getmessage (attribute_name, value[])
    //   * getmessage (source, attribute_name, value)
    //   * getmessage (source, attribute_name, value[])
    Opcode& op(rop.inst()->ops()[opnum]);

    OSL_DASSERT(op.nargs() == 3 || op.nargs() == 4);
    int has_source = (op.nargs() == 4);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& Source = *rop.opargsym(op, 1);
    Symbol& Name   = *rop.opargsym(op, 1 + has_source);
    Symbol& Data   = *rop.opargsym(op, 2 + has_source);
    OSL_DASSERT(Result.typespec().is_int() && Name.typespec().is_string());
    OSL_DASSERT(has_source == 0 || Source.typespec().is_string());

    // BatchedAnalysis should guarantee varying results
    OSL_ASSERT(Result.is_uniform() == false);
    OSL_ASSERT(Data.is_uniform() == false);

    BatchedBackendLLVM::TempScope temp_scope(rop);
    std::vector<llvm::Value*> args;

    //arg[0]: Push shader global
    args.push_back(rop.sg_void_ptr());
    args.push_back(rop.llvm_void_ptr(Result));

    //arg[1]: source (deferred)
    int sourceArgumentIndex = args.size();
    args.push_back(nullptr);

    //arg[2]: name (deferred)
    int nameArgumentIndex = args.size();
    args.push_back(nullptr);

    //Data
    //args[3]
    args.push_back(rop.ll.constant(Data.typespec().simpletype()));

    //Check for closures, and skip it
    if (Data.typespec().is_closure_based()) {
        // Needed for closure support
        OSL_ASSERT(0 && "incomplete");
    } else {
        args.push_back(rop.llvm_void_ptr(Data));
    }

    //Do not leave derivs uninitialized or bad things may happen

    if (Data.has_derivs()) {
        rop.llvm_zero_derivs(Data);
    }

    //args[5]: Data.has_derivs
    args.push_back(rop.ll.constant((int)Data.has_derivs()));

    //args[6]: ID
    args.push_back(rop.ll.constant(rop.inst()->id()));

    //args[7]: sourcefile
    args.push_back(rop.ll.constant(op.sourcefile()));

    //args[8]: sourceline
    args.push_back(rop.ll.constant(op.sourceline()));

    bool sourceVal_is_uniform = !has_source || Source.is_uniform();
    bool nameVal_is_uniform   = Name.is_uniform();
    if (nameVal_is_uniform && sourceVal_is_uniform) {  //&& has_source
        args[nameArgumentIndex]   = rop.llvm_load_value(Name);
        args[sourceArgumentIndex] = has_source ? rop.llvm_load_value(Source)
                                               : rop.ll.constant(ustring());
        args.push_back(rop.ll.mask_as_int(rop.ll.current_mask()));
        rop.ll.call_function(rop.build_name(FuncSpec("getmessage").mask()),
                             args);
    } else {
        llvm::Value* nameVal
            = rop.llvm_load_value(Name, /*deriv=*/0, /*component=*/0,
                                  TypeDesc::UNKNOWN, nameVal_is_uniform);
        if (nameVal_is_uniform) {
            args[nameArgumentIndex] = nameVal;
        }

        llvm::Value* sourceVal
            = has_source
                  ? rop.llvm_load_value(Source, /*deriv=*/0, /*component=*/0,
                                        TypeDesc::UNKNOWN, sourceVal_is_uniform)
                  : rop.ll.constant(ustring());
        if (sourceVal_is_uniform) {
            args[sourceArgumentIndex] = sourceVal;
        }

        llvm::Value* loc_of_remainingMask = rop.getTempMask(
            "lanes remaining to getmessage");
        rop.ll.op_store_mask(rop.ll.current_mask(), loc_of_remainingMask);

        llvm::BasicBlock* bin_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("bin_getmessage") : std::string());
        llvm::BasicBlock* after_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("after_bin_getmessage")
                             : std::string());

        rop.ll.op_branch(bin_block);
        {
            llvm::Value* remainingMask = rop.ll.op_load_mask(
                loc_of_remainingMask);
            llvm::Value* leadLane = rop.ll.op_1st_active_lane_of(remainingMask);

            llvm::Value* lanesMatching = remainingMask;
            if (!nameVal_is_uniform) {
                llvm::Value* scalar_name = rop.ll.op_extract(nameVal, leadLane);
                args[nameArgumentIndex]  = scalar_name;

                lanesMatching
                    = rop.ll.op_lanes_that_match_masked(scalar_name, nameVal,
                                                        lanesMatching);
            }

            if (!sourceVal_is_uniform) {
                llvm::Value* scalar_source = rop.ll.op_extract(sourceVal,
                                                               leadLane);
                args[sourceArgumentIndex]  = scalar_source;

                lanesMatching = rop.ll.op_lanes_that_match_masked(
                    scalar_source, sourceVal, lanesMatching);
            }

            //Check matched masks, and push. Get matching lane
            args.push_back(rop.ll.mask_as_int(lanesMatching));

            rop.ll.call_function(rop.build_name(FuncSpec("getmessage").mask()),
                                 args);

            remainingMask = rop.ll.op_xor(remainingMask, lanesMatching);

            //rop.llvm_print_mask("xor remainingMask,lanesMatchingOptions", remainingMask);
            rop.ll.op_store_mask(remainingMask, loc_of_remainingMask);

            llvm::Value* int_remainingMask = rop.ll.mask_as_int(remainingMask);
            //rop.llvm_print_mask("remainingMask", remainingMask);
            llvm::Value* cond_more_lanes_to_bin
                = rop.ll.op_ne(int_remainingMask, rop.ll.constant(0));
            rop.ll.op_branch(cond_more_lanes_to_bin, bin_block, after_block);
        }
        // Continue on with the previous flow
        rop.ll.set_insert_point(after_block);
    }
    return true;
}



LLVMGEN(llvm_gen_setmessage)
{
    Opcode& op(rop.inst()->ops()[opnum]);

    OSL_DASSERT(op.nargs() == 2);
    Symbol& Name = *rop.opargsym(op, 0);
    Symbol& Data = *rop.opargsym(op, 1);
    OSL_DASSERT(Name.typespec().is_string());

    BatchedBackendLLVM::TempScope temp_scope(rop);

    llvm::Value* args[8];
    args[0] = rop.sg_void_ptr();
    args[1] = rop.llvm_load_arg(Name, false /*derivs*/, Name.is_uniform());

    if (Data.typespec().is_closure_based()) {
        // Still needed for complete batched closure support
        OSL_ASSERT(0 && "incomplete");
        // FIXME: secret handshake for closures ...
        args[2] = rop.ll.constant(
            TypeDesc(TypeDesc::UNKNOWN, Data.typespec().arraylength()));
        // We need a void ** here so the function can modify the closure
        args[3] = rop.llvm_void_ptr(Data);
    } else {
        args[2] = rop.ll.constant(Data.typespec().simpletype());

        // Implementation chooses to only accept wide Data
        // broadcast uniform data values if necessary
        args[3] = rop.llvm_load_arg(Data, Data.has_derivs(),
                                    false /*data is always wide*/);
    }
    args[4] = rop.ll.constant(rop.inst()->id());
    args[5] = rop.ll.constant(op.sourcefile());
    args[6] = rop.ll.constant(op.sourceline());
    args[7] = rop.ll.mask_as_int(rop.ll.current_mask());

    FuncSpec func("setmessage");
    func.arg(Name, Name.is_uniform());
    func.arg_varying(TypeDesc { TypeDesc::PTR });
    func.mask();
    rop.ll.call_function(rop.build_name(func), args);

    return true;
}



LLVMGEN(llvm_gen_get_simple_SG_field)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 1);

    Symbol& Result = *rop.opargsym(op, 0);
    bool is_uniform;
    int sg_index = rop.ShaderGlobalNameToIndex(op.opname(), is_uniform);
    OSL_ASSERT(sg_index >= 0);
    llvm::Value* sg_field = rop.ll.GEP(rop.llvm_type_sg(), rop.sg_ptr(), 0,
                                       sg_index);
    llvm::Type* sg_field_type
        = rop.ll.type_struct_field_at_index(rop.llvm_type_sg(), sg_index);
    llvm::Value* r = rop.ll.op_load(sg_field_type, sg_field);
    rop.llvm_store_value(r, Result);

    return true;
}



LLVMGEN(llvm_gen_calculatenormal)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 2);

    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& P      = *rop.opargsym(op, 1);

    // NOTE: because calculatenormal implicitly uses the flip-handedness
    // of the BatchedShaderGlobals, all of its results must be varying
    OSL_ASSERT(false == Result.is_uniform());

    OSL_DASSERT(Result.typespec().is_triple() && P.typespec().is_triple());
    if (!P.has_derivs()) {
        rop.llvm_assign_zero(Result);
        return true;
    }

    BatchedBackendLLVM::TempScope temp_scope(rop);

    llvm::Value* args[] = {
        rop.llvm_void_ptr(Result), rop.sg_void_ptr(),
        rop.llvm_load_arg(P, true /*derivs*/, false /*op_is_uniform*/), nullptr
    };
    int arg_count = 3;

    FuncSpec func_spec("calculatenormal");
    if (rop.ll.is_masking_required()) {
        args[arg_count++] = rop.ll.mask_as_int(rop.ll.current_mask());
        func_spec.mask();
    }
    rop.ll.call_function(rop.build_name(func_spec),
                         { &args[0],
                           cspan<llvm::Value*>::size_type(arg_count) });

    if (Result.has_derivs())
        rop.llvm_zero_derivs(Result);
    return true;
}



LLVMGEN(llvm_gen_area)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 2);

    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& P      = *rop.opargsym(op, 1);

    OSL_DASSERT(Result.typespec().is_float() && P.typespec().is_triple());
    if (!P.has_derivs()) {
        rop.llvm_assign_zero(Result);
        return true;
    }

    bool op_is_uniform = Result.is_uniform();

    FuncSpec func_spec("area");
    if (op_is_uniform) {
        func_spec.unbatch();

        llvm::Value* r = rop.ll.call_function(rop.build_name(func_spec),
                                              rop.llvm_void_ptr(P));
        rop.llvm_store_value(r, Result);

    } else {
        func_spec.arg_varying(Result);
        func_spec.arg(P, true /*derivs*/, false /*uniform*/);


        const Symbol* args[] = { &Result, &P };

        rop.llvm_call_function(func_spec, &args[0], 2, true /*deriv_ptrs*/,
                               false /*function_is_uniform*/,
                               false /*functionIsLlvmInlined*/,
                               true /*ptrToReturnStructIs1stArg*/);
    }

    if (Result.has_derivs())
        rop.llvm_zero_derivs(Result);

    return true;
}



LLVMGEN(llvm_gen_spline)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() >= 4 && op.nargs() <= 5);

    bool has_knot_count = (op.nargs() == 5);
    Symbol& Result      = *rop.opargsym(op, 0);
    Symbol& Spline      = *rop.opargsym(op, 1);
    Symbol& Value       = *rop.opargsym(op, 2);
    Symbol& Knot_count  = *rop.opargsym(op, 3);  // might alias Knots
    Symbol& Knots       = has_knot_count ? *rop.opargsym(op, 4)
                                         : *rop.opargsym(op, 3);

    OSL_DASSERT(!Result.typespec().is_closure_based()
                && Spline.typespec().is_string() && Value.typespec().is_float()
                && !Knots.typespec().is_closure_based()
                && Knots.typespec().is_array()
                && (!has_knot_count
                    || (has_knot_count && Knot_count.typespec().is_int())));

    if (has_knot_count && !Knot_count.is_uniform()) {
        // TODO: varying knot count support could be added below as we already
        // have a loop to handle varying spline basis.  Just need to add tests
        // to exercise and verify functions properly, until then...
        rop.shadingcontext()->errorfmt(
            "spline nknots parameter is varying, batched code gen requires a constant or uniform nknots, called from ({}:{})",
            op.sourcefile(), op.sourceline());
        return false;
    }

    bool op_is_uniform = Spline.is_uniform() && Value.is_uniform()
                         && Knots.is_uniform();

    FuncSpec func_spec(op.opname().c_str());

    //std::string name = fmtfrmat("osl_{}_", op.opname());
    std::vector<llvm::Value*> args;
    // only use derivatives for result if:
    //   result has derivs and (value || knots) have derivs
    bool op_derivs = Result.has_derivs()
                     && (Value.has_derivs() || Knots.has_derivs());

    func_spec.arg(Result, op_derivs, op_is_uniform);
    func_spec.arg(Value, op_derivs && Value.has_derivs(), Value.is_uniform());
    func_spec.arg(Knots.typespec().simpletype().elementtype(),
                  op_derivs && Knots.has_derivs(), Knots.is_uniform());

    if (op_is_uniform) {
        func_spec.unbatch();
    } else {
        // for simplicity, always call the masked version
        func_spec.mask();
    }
    const char* splineFuncName = rop.build_name(func_spec);

    BatchedBackendLLVM::TempScope temp_scope(rop);

    llvm::Value* temp_uniform_results = nullptr;
    if (op_is_uniform && !Result.is_uniform()) {
        temp_uniform_results
            = rop.getOrAllocateTemp(Result.typespec(), Result.has_derivs(),
                                    true /*is_uniform*/, false /*forceBool*/,
                                    "uniform spline result");
        args.push_back(rop.ll.void_ptr(temp_uniform_results));

    } else {
        args.push_back(rop.llvm_void_ptr(Result));
    }
    // We will just load the spline name here if we are uniform,
    // otherwise just remember where the spline name
    // is so we can update it later in the loop over varying spline names
    int splineNameArgumentIndex = args.size();
    llvm::Value* splineNameVal
        = rop.llvm_load_value(Spline, /*deriv=*/0, /*component=*/0,
                              TypeDesc::UNKNOWN, Spline.is_uniform());
    args.push_back(Spline.is_uniform() ? splineNameVal : nullptr);

    args.push_back(rop.llvm_void_ptr(Value));  // make things easy
    args.push_back(rop.llvm_void_ptr(Knots));
    if (has_knot_count)
        args.push_back(rop.llvm_load_value(
            Knot_count));  // TODO: add support for varying Knot_count
    else
        args.push_back(rop.ll.constant((int)Knots.typespec().arraylength()));
    args.push_back(rop.ll.constant((int)Knots.typespec().arraylength()));

    if (false == op_is_uniform) {
        // We always call the masked version, need to pass the mask value
        args.push_back(Spline.is_uniform()
                           ? rop.ll.mask_as_int(rop.ll.current_mask())
                           : nullptr);
    }

    if (Spline.is_uniform()) {
        rop.ll.call_function(splineFuncName, args);
        if (op_is_uniform && !Result.is_uniform()) {
            rop.llvm_broadcast_uniform_value_from_mem(temp_uniform_results,
                                                      Result);
        }
    } else {
        // do while(remaining)
        llvm::Value* loc_of_remainingMask = rop.getTempMask(
            "lanes remaining to spline");
        rop.ll.op_store_mask(rop.ll.current_mask(), loc_of_remainingMask);

        llvm::BasicBlock* bin_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("bin_spline (varying spline name)")
                             : std::string());
        llvm::BasicBlock* after_block = rop.ll.new_basic_block(
            rop.llvm_debug()
                ? std::string("after_bin_spline (varying spline name)")
                : std::string());
        rop.ll.op_branch(bin_block);
        {
            llvm::Value* remainingMask = rop.ll.op_load_mask(
                loc_of_remainingMask);
            llvm::Value* leadLane = rop.ll.op_1st_active_lane_of(remainingMask);
            llvm::Value* lanesMatchingSplineName = remainingMask;

            llvm::Value* scalar_splineName = rop.ll.op_extract(splineNameVal,
                                                               leadLane);
            args[splineNameArgumentIndex]  = scalar_splineName;
            lanesMatchingSplineName        = rop.ll.op_lanes_that_match_masked(
                scalar_splineName, splineNameVal, remainingMask);

            OSL_ASSERT(lanesMatchingSplineName);
            //rop.llvm_print_mask("lanesMatchingSplineName", lanesMatchingSplineName);
            args.back() = rop.ll.mask_as_int(lanesMatchingSplineName);

            rop.ll.call_function(splineFuncName, args);

            remainingMask = rop.ll.op_xor(remainingMask,
                                          lanesMatchingSplineName);
            //rop.llvm_print_mask("xor remainingMask,lanesMatchingSplineName", remainingMask);
            rop.ll.op_store_mask(remainingMask, loc_of_remainingMask);

            llvm::Value* int_remainingMask = rop.ll.mask_as_int(remainingMask);
            //rop.llvm_print_mask("remainingMask", remainingMask);
            llvm::Value* cond_more_lanes_to_bin
                = rop.ll.op_ne(int_remainingMask, rop.ll.constant(0));
            rop.ll.op_branch(cond_more_lanes_to_bin, bin_block, after_block);
        }
        // Continue on with the previous flow
        rop.ll.set_insert_point(after_block);
    }

    if (Result.has_derivs() && !op_derivs)
        rop.llvm_zero_derivs(Result);

    return true;
}



static void
llvm_gen_keyword_fill(BatchedBackendLLVM& rop, Opcode& op,
                      const ClosureRegistry::ClosureEntry* clentry,
                      ustring clname, llvm::Value* mem_void_ptr, int argsoffset,
                      int lane_index)
{
    OSL_DASSERT(((op.nargs() - argsoffset) % 2) == 0);

    int Nattrs = (op.nargs() - argsoffset) / 2;

    for (int attr_i = 0; attr_i < Nattrs; ++attr_i) {
        int argno     = attr_i * 2 + argsoffset;
        Symbol& Key   = *rop.opargsym(op, argno);
        Symbol& Value = *rop.opargsym(op, argno + 1);
        OSL_ASSERT(Key.typespec().is_string());
        OSL_ASSERT(Key.is_constant());
        ustring* key       = (ustring*)Key.data();
        TypeDesc ValueType = Value.typespec().simpletype();

        bool legal = false;
        // Make sure there is some keyword arg that has the name and the type
        for (int t = 0; t < clentry->nkeyword; ++t) {
            const ClosureParam& p = clentry->params[clentry->nformal + t];
            // strcmp might be too much, we could precompute the ustring for the param,
            // but in this part of the code is not a big deal
            if (equivalent(p.type, ValueType) && !strcmp(key->c_str(), p.key)) {
                // store data
                OSL_DASSERT(p.offset + p.field_size <= clentry->struct_size);

                bool arg_is_uniform = Value.is_uniform();

                TypeDesc simpletype(Value.typespec().simpletype());

                // We don't currently offer a helper macro in genclosure.h for creating
                // keyword array parameters, so perhaps we can assume we don't need to
                // support them
                OSL_DASSERT(simpletype.arraylen == 0);

                int num_components = simpletype.aggregate;

                llvm::Value* dest_base = rop.ll.offset_ptr(mem_void_ptr,
                                                           p.offset);
                dest_base              = rop.llvm_ptr_cast(dest_base, p.type);
                llvm::Type* dest_type  = rop.ll.llvm_type(p.type);

                for (int c = 0; c < num_components; c++) {
                    llvm::Value* dest_elem;
                    if (num_components > 1)
                        dest_elem = rop.ll.GEP(dest_type, dest_base, 0, c);
                    else
                        dest_elem = dest_base;

                    // NOTE:  We don't want any uniform arguments to be
                    // widened, so our typical op_is_uniform doesn't do what we
                    // want for this when loading.  So just pass arg_is_uniform
                    // which will avoid widening any uniform arguments.
                    llvm::Value* loaded
                        = rop.llvm_load_value(Value, 0, NULL, c,
                                              TypeDesc::UNKNOWN,
                                              /*op_is_uniform*/ arg_is_uniform,
                                              /*index_is_uniform*/ true);

                    if (!arg_is_uniform) {
                        loaded = rop.ll.op_extract(loaded, lane_index);
                    }
                    rop.ll.op_unmasked_store(loaded, dest_elem);
                }

                legal = true;
                break;
            }
        }
        if (!legal) {
            rop.shadingcontext()->warningfmt(
                "Unsupported closure keyword arg \"{}\" for {} ({}:{})", *key,
                clname, op.sourcefile(), op.sourceline());
        }
    }
}



LLVMGEN(llvm_gen_closure)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() >= 2);  // at least the result and the ID
    Symbol& Result = *rop.opargsym(op, 0);
    int weighted   = rop.opargsym(op, 1)->typespec().is_string() ? 0 : 1;
    Symbol* weight = weighted ? rop.opargsym(op, 1) : NULL;
    Symbol& Id     = *rop.opargsym(op, 1 + weighted);
    OSL_DASSERT(Result.typespec().is_closure());
    OSL_DASSERT(Id.typespec().is_string());
    OSL_DASSERT(
        Id.is_uniform());  // we have to know which closure at compile time
    ustring closure_name = Id.get_string();

    const ClosureRegistry::ClosureEntry* clentry
        = rop.shadingsys().find_closure(closure_name);
    if (!clentry) {
        rop.shadingcontext()->errorfmt(
            "Closure '{}' is not supported by the current renderer, called from {}:{} in shader \"{}\", layer {} \"{}\", group \"{}\"",
            closure_name, op.sourcefile(), op.sourceline(),
            rop.inst()->shadername(), rop.layer(), rop.inst()->layername(),
            rop.group().name());
        return false;
    }

    OSL_DASSERT(op.nargs() >= (2 + weighted + clentry->nformal));
    OSL_DASSERT(
        !Result.is_uniform());  // we don't optimize for when closures happen to be uniform

    bool weight_is_uniform = weighted ? weight->is_uniform()
                                      : false /*don't care*/;

    // Call osl_allocate_{weighted_}closure_component(globals, result, id, size,{ weight,} mask).
    // It writes WidthT pointers into result that map to the memory for the closure parameter data.
    // this is just a batch of closures instead of a closure for a batch
    llvm::Value* render_ptr = rop.ll.constant_ptr(rop.shadingsys().renderer(),
                                                  rop.ll.type_void_ptr());
    llvm::Value* sg_ptr     = rop.sg_void_ptr();
    llvm::Value* id_int     = rop.ll.constant(clentry->id);
    llvm::Value* size_int   = rop.ll.constant(clentry->struct_size);
    FuncSpec func_spec(weighted ? "allocate_weighted_closure_component"
                                : "allocate_closure_component");
    func_spec.mask();

    // make sure that any temps created to widen uniform values
    // are not released until we are done using them!
    BatchedBackendLLVM::TempScope temp_scope(rop);

    llvm::Value* args[6] = { sg_ptr,  rop.llvm_void_ptr(Result),
                             id_int,  size_int,
                             nullptr, nullptr };
    if (weighted) {
        if (weighted && weight_is_uniform) {
            args[4] = rop.llvm_widen_value_into_temp(*weight, 0);
        } else {
            args[4] = rop.llvm_void_ptr(*weight);
        }

        args[5] = rop.ll.mask_as_int(rop.ll.current_mask());
        rop.ll.call_function(rop.build_name(func_spec),
                             cspan<llvm::Value*>(args, 6));
    } else {
        args[4] = rop.ll.mask_as_int(rop.ll.current_mask());
        rop.ll.call_function(rop.build_name(func_spec),
                             cspan<llvm::Value*>(args, 5));
    }

    llvm::Value* comp_void_ptr = rop.llvm_get_pointer(Result);

    llvm::Value* comp_wide_ptr_ptr = rop.ll.ptr_cast(
        comp_void_ptr,
        rop.ll.type_ptr(rop.llvm_type_closure_component_wide_ptr()));
    llvm::Value* comp_wide_ptr
        = rop.ll.op_load(rop.llvm_type_closure_component_wide_ptr(),
                         comp_wide_ptr_ptr);

    llvm::Value* mask = rop.ll.current_mask();
    // This is an unrolled vector-width loop.  It was unrolled because it's
    // was more expedient to write the for in C++ than IR, but if compilation
    // of the extra ops becomes problematic, this choice can be revisited
    for (int lane_index = 0; lane_index < rop.vector_width(); ++lane_index) {
        llvm::Value* comp_ptr = rop.ll.op_extract(comp_wide_ptr, lane_index);
        // Get the address of the primitive buffer, which is the 2nd field
        llvm::Value* mem_void_ptr
            = rop.ll.GEP(rop.llvm_type_closure_component(), comp_ptr, 0, 2);
        mem_void_ptr = rop.ll.ptr_cast(mem_void_ptr, rop.ll.type_void_ptr());

        // For the weighted closures, we need a surrounding "if" so that it's safe
        // for osl_allocate_weighted_closure_component to return NULL (unless we
        // know for sure that it's constant weighted and that the weight is
        // not zero).  We also need the surrounding "if" to reject writing
        // to closures that aren't masked.
        llvm::BasicBlock* next_block    = NULL;
        llvm::BasicBlock* notnull_block = rop.ll.new_basic_block(
            "non_null_and_masked_closure");
        next_block        = rop.ll.new_basic_block("");
        llvm::Value* cond = rop.ll.op_and(
            rop.ll.op_ne(rop.ll.ptr_cast(comp_ptr, rop.ll.type_void_ptr()),
                         rop.ll.void_ptr_null()),
            rop.ll.test_mask_lane(mask, lane_index));
        rop.ll.op_branch(cond, notnull_block, next_block);

        // If the closure has a "prepare" method, call
        // prepare(renderer, id, memptr).  If there is no prepare method, just
        // zero out the closure parameter memory.
        if (clentry->prepare) {
            // Call clentry->prepare(renderservices *, int id, void *mem)
            llvm::Value* funct_ptr
                = rop.ll.constant_ptr((void*)clentry->prepare,
                                      rop.llvm_type_prepare_closure_func());
            llvm::Value* args[] = { render_ptr, id_int, mem_void_ptr };
            rop.ll.call_function(funct_ptr, args);
        } else {
            rop.ll.op_memset(mem_void_ptr, 0, clentry->struct_size,
                             4 /*align*/);
        }

        for (int carg = 0; carg < clentry->nformal; ++carg) {
            const ClosureParam& p = clentry->params[carg];

            if (p.key != NULL)
                break;
            OSL_DASSERT(p.offset + p.field_size <= clentry->struct_size);
            Symbol& sym = *rop.opargsym(op, carg + 2 + weighted);

            bool arg_is_uniform = sym.is_uniform();

            TypeDesc simpletype(sym.typespec().simpletype());
            int num_elements   = simpletype.numelements();
            int num_components = simpletype.aggregate;

            llvm::Value* dest_base = rop.ll.offset_ptr(mem_void_ptr, p.offset);
            llvm::Type* dest_type  = rop.ll.llvm_type(
                static_cast<const TypeSpec&>(p.type).simpletype());
            dest_base = rop.ll.ptr_to_cast(dest_base, dest_type);

            if (num_elements > 1) {
                OSL_DASSERT(rop.ll.is_type_array(dest_type));

                // The type is an array type,
                // which means our pointer is to an array type,
                // not to &[0] (the address of the 1st element).
                // IE:  typedef int Int5Array[5];
                //      Int5Array *dest_base;
                // in contrast to what most of use might consider an array pointer
                //      int *dest_base;
                // So if we dereference Int5Array* to with [4], the 5th element,
                // we would be really be pointing to ((int *)dest_base)[5*4] which is
                // out of bounds of our 1 instance of Int5Array.
                // Thus when dealing with arrays we will just dereference it and get the address to get
                // the address of the 1st element.

                dest_base = rop.ll.GEP(dest_type, dest_base, 0, 0);
                llvm::Type* element_type = rop.ll.element_type_of(dest_type);
                dest_type                = element_type;
            }

            for (int a = 0; a < num_elements; ++a) {
                llvm::Value* arrind = simpletype.arraylen ? rop.ll.constant(a)
                                                          : NULL;
                for (int c = 0; c < num_components; c++) {
                    llvm::Value* dest_elem;
                    if (num_components > 1)
                        dest_elem = rop.ll.GEP(dest_type, dest_base, a, c);
                    else if (num_elements > 1)
                        dest_elem = rop.ll.GEP(dest_type, dest_base, a);
                    else
                        dest_elem = dest_base;

                    // NOTE:  We don't want any uniform arguments to be
                    // widened, so our typical op_is_uniform doesn't do what we
                    // want for this when loading.  So just pass arg_is_uniform
                    // which will avoid widening any uniform arguments.
                    llvm::Value* loaded
                        = rop.llvm_load_value(sym, 0, arrind, c,
                                              TypeDesc::UNKNOWN,
                                              /*op_is_uniform*/ arg_is_uniform,
                                              /*index_is_uniform*/ true);

                    if (!arg_is_uniform) {
                        loaded = rop.ll.op_extract(loaded, lane_index);
                    }
                    rop.ll.op_unmasked_store(loaded, dest_elem);
                }
            }
        }

        // If the closure has a "setup" method, call
        // setup(render_services, id, mem_ptr).
        if (clentry->setup) {
            // Call clentry->setup(renderservices *, int id, void *mem)
            llvm::Value* funct_ptr
                = rop.ll.constant_ptr((void*)clentry->setup,
                                      rop.llvm_type_setup_closure_func());
            llvm::Value* args[] = { render_ptr, id_int, mem_void_ptr };
            rop.ll.call_function(funct_ptr, args);
        }

        llvm_gen_keyword_fill(rop, op, clentry, closure_name, mem_void_ptr,
                              2 + weighted + clentry->nformal, lane_index);

        if (next_block)
            rop.ll.op_branch(next_block);
    }

    return true;
}



LLVMGEN(llvm_gen_pointcloud_search)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() >= 5);
    Symbol& Result     = *rop.opargsym(op, 0);
    Symbol& Filename   = *rop.opargsym(op, 1);
    Symbol& Center     = *rop.opargsym(op, 2);
    Symbol& Radius     = *rop.opargsym(op, 3);
    Symbol& Max_points = *rop.opargsym(op, 4);

    OSL_ASSERT(Result.is_varying());
    OSL_DASSERT(Result.typespec().is_int() && Filename.typespec().is_string()
                && Center.typespec().is_triple() && Radius.typespec().is_float()
                && Max_points.typespec().is_int());

    // NOTE:  It would be trivial to pass a wide max_points through RendererServices.
    //        But as we don't really expect them to ever be, we can just pay the
    //        binning overhead should it accidentally happen.  If Max_points ends
    //        up being varying in practice, suggest changing what is passed to
    //        RendererServices to avoid binning.
    bool requires_binning = Filename.is_varying() || Max_points.is_varying();

    std::vector<Symbol*>
        clear_derivs_of;  // arguments whose derivs we need to zero at the end
    int attr_arg_offset  = 5;  // where the opt attrs begin
    Symbol* Sort         = NULL;
    llvm::Value* sortVal = nullptr;
    if (op.nargs() > 5 && rop.opargsym(op, 5)->typespec().is_int()) {
        Sort             = rop.opargsym(op, 5);
        requires_binning = requires_binning || Sort->is_varying();
        sortVal = rop.llvm_load_value(*Sort, /*deriv=*/0, /*component=*/0,
                                      TypeDesc::UNKNOWN, Sort->is_uniform());
        ++attr_arg_offset;
    }
    int nattrs = (op.nargs() - attr_arg_offset) / 2;

    BatchedBackendLLVM::TempScope temp_scope(rop);

    bool pass_center_derivs = false;
    std::vector<llvm::Value*> args;
    args.push_back(rop.sg_void_ptr());          // 0 sg
    args.push_back(rop.llvm_void_ptr(Result));  // 1 result

    constexpr int fileNameArgumentIndex = 2;  // 2 filename
    OSL_ASSERT(args.size() == fileNameArgumentIndex);
    llvm::Value* fileNameVal
        = rop.llvm_load_value(Filename, /*deriv=*/0, /*component=*/0,
                              TypeDesc::UNKNOWN, Filename.is_uniform());
    args.push_back(Filename.is_uniform() ? fileNameVal : nullptr);

    args.push_back(nullptr /*deferred*/);  // 3 center
    args.push_back(rop.llvm_load_arg(Radius, false /*derivs*/,
                                     false /*is_uniform*/));  // 4 radius
    constexpr int maxPointsArgumentIndex = 5;                 // 5 max_points
    OSL_ASSERT(args.size() == maxPointsArgumentIndex);
    llvm::Value* maxPointsVal
        = rop.llvm_load_value(Max_points, /*deriv=*/0, /*component=*/0,
                              TypeDesc::UNKNOWN, Max_points.is_uniform());
    args.push_back(Max_points.is_uniform() ? maxPointsVal : nullptr);

    constexpr int sortArgumentIndex = 6;  // 6 sort
    OSL_ASSERT(args.size() == sortArgumentIndex);
    args.push_back((Sort && Sort->is_uniform()) ? sortVal : rop.ll.constant(0));
    args.push_back(rop.ll.constant_ptr(nullptr));  // 7 indices
    args.push_back(rop.ll.constant(0));            // 8 indices array length
    args.push_back(rop.ll.constant_ptr(nullptr));  // 9 distances
    args.push_back(rop.ll.constant(1));            // 10 distances array length
    args.push_back(rop.ll.constant(0));            // 11 distances_has_derivs

    constexpr int maskArgumentIndex = 12;  // 12 mask_value
    OSL_ASSERT(args.size() == maskArgumentIndex);
    args.push_back(nullptr /*deferred*/);

    args.push_back(nullptr /*deferred*/);  // 13 nattrs
    int capacity    = 0x7FFFFFFF;          // Lets put a 32 bit limit
    int extra_attrs = 0;                   // Extra query attrs to search
    // This loop does three things. 1) Look for the special attributes
    // "distance", "index" and grab the pointer. 2) Compute the minimmum
    // size of the provided output arrays to check against max_points
    // 3) push optional args to the arg list
    for (int i = 0; i < nattrs; ++i) {
        Symbol& Name  = *rop.opargsym(op, attr_arg_offset + i * 2);
        Symbol& Value = *rop.opargsym(op, attr_arg_offset + i * 2 + 1);

        OSL_DASSERT(Name.typespec().is_string());
        TypeDesc simpletype = Value.typespec().simpletype();
        if (Name.is_constant() && Name.get_string() == u_index
            && simpletype.elementtype() == TypeDesc::INT) {
            args[7] = rop.llvm_void_ptr(Value);
            args[8] = rop.ll.constant(Value.typespec().arraylength());
        } else if (Name.is_constant() && Name.get_string() == u_distance
                   && simpletype.elementtype() == TypeDesc::FLOAT) {
            args[9]  = rop.llvm_void_ptr(Value);
            args[10] = rop.ll.constant(Value.typespec().arraylength());
            if (Value.has_derivs()) {
                if (Center.has_derivs()) {
                    pass_center_derivs = true;
                    args[11]           = rop.ll.constant(1);
                } else
                    clear_derivs_of.push_back(&Value);
            }
        } else {
            // It is a regular attribute, push it to the arg list
            if (!Name.is_uniform()) {
                rop.shadingcontext()->errorfmt(
                    "pointcloud_search named attribute parameter is varying, batched code gen requires a constant or uniform attribute name, called from ({}:{})",
                    op.sourcefile(), op.sourceline());
                return false;
            }

            OSL_ASSERT(Name.is_uniform());
            OSL_ASSERT(Value.is_varying());
            args.push_back(rop.llvm_load_value(Name));
            args.push_back(rop.ll.constant(simpletype));
            args.push_back(rop.llvm_void_ptr(Value));
            if (Value.has_derivs())
                clear_derivs_of.push_back(&Value);
            extra_attrs++;
        }
        // minimum capacity of the output arrays
        capacity = std::min(static_cast<int>(simpletype.numelements()),
                            capacity);
    }
    args[3] = rop.llvm_load_arg(Center, pass_center_derivs,
                                false /*is_uniform*/);  // 3 center

    args[13] = rop.ll.constant(extra_attrs);

    if (Max_points.is_constant()) {
        // Compare capacity to the requested number of points.
        // Choose not to do a runtime check because generated code will still work,
        // arrays will only be filled up to the capacity,
        // per the OSL language specification
        const int const_max_points = Max_points.get_int();
        if (capacity < const_max_points) {
            rop.shadingcontext()->warningfmt(
                "Arrays too small for pointcloud lookup at ({}:{}) ({}:{})",
                op.sourcefile(), op.sourceline());
            args[maxPointsArgumentIndex] = rop.ll.constant(capacity);
        }
    } else {
        // Clamp max points to the capacity of the arrays
        if (Max_points.is_uniform()) {
            llvm::Value* capacity_val = rop.ll.constant(capacity);
            llvm::Value* cond = rop.ll.op_le(capacity_val, maxPointsVal);
            llvm::Value* clampedMaxPoints = rop.ll.op_select(cond, capacity_val,
                                                             maxPointsVal);
            args[maxPointsArgumentIndex]  = clampedMaxPoints;
        } else {
            llvm::Value* wcapacity_val = rop.ll.wide_constant(capacity);
            llvm::Value* cond = rop.ll.op_le(wcapacity_val, maxPointsVal);
            maxPointsVal = rop.ll.op_select(cond, wcapacity_val, maxPointsVal);
        }
    }

    // non-error code case
    FuncSpec func_spec("pointcloud_search");
    func_spec.mask();
    const char* funcName = rop.build_name(func_spec);

    if (requires_binning) {
        // do while(remaining)
        llvm::Value* loc_of_remainingMask = rop.getTempMask(
            "lanes remaining to pointcloud_search");
        rop.ll.op_store_mask(rop.ll.current_mask(), loc_of_remainingMask);

        llvm::BasicBlock* bin_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("bin_pointcloud_search")
                             : std::string());
        llvm::BasicBlock* after_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("after_bin_pointcloud_search")
                             : std::string());
        rop.ll.op_branch(bin_block);
        {
            llvm::Value* remainingMask = rop.ll.op_load_mask(
                loc_of_remainingMask);
            llvm::Value* leadLane = rop.ll.op_1st_active_lane_of(remainingMask);
            llvm::Value* lanesMatching = remainingMask;

            if (!Filename.is_uniform()) {
                llvm::Value* scalar_fileName = rop.ll.op_extract(fileNameVal,
                                                                 leadLane);
                args[fileNameArgumentIndex]  = scalar_fileName;
                lanesMatching = rop.ll.op_lanes_that_match_masked(
                    scalar_fileName, fileNameVal, remainingMask);
            }
            OSL_ASSERT(lanesMatching);

            if (!Max_points.is_uniform()) {
                llvm::Value* scalar_maxPoints = rop.ll.op_extract(maxPointsVal,
                                                                  leadLane);
                args[maxPointsArgumentIndex]  = scalar_maxPoints;
                lanesMatching = rop.ll.op_lanes_that_match_masked(
                    scalar_maxPoints, maxPointsVal, lanesMatching);
            }

            if (Sort && (!Sort->is_uniform())) {
                OSL_ASSERT(sortVal);
                llvm::Value* scalar_sort = rop.ll.op_extract(sortVal, leadLane);
                args[sortArgumentIndex]  = scalar_sort;
                lanesMatching
                    = rop.ll.op_lanes_that_match_masked(scalar_sort, sortVal,
                                                        lanesMatching);
            }

            //rop.llvm_print_mask("lanesMatching", lanesMatching);

            args[maskArgumentIndex] = rop.ll.mask_as_int(lanesMatching);

            rop.ll.call_function(funcName, args);

            remainingMask = rop.ll.op_xor(remainingMask, lanesMatching);
            //rop.llvm_print_mask("xor remainingMask,lanesMatchingSplineName", remainingMask);
            rop.ll.op_store_mask(remainingMask, loc_of_remainingMask);

            llvm::Value* int_remainingMask = rop.ll.mask_as_int(remainingMask);
            //rop.llvm_print_mask("remainingMask", remainingMask);
            llvm::Value* cond_more_lanes_to_bin
                = rop.ll.op_ne(int_remainingMask, rop.ll.constant(0));
            rop.ll.op_branch(cond_more_lanes_to_bin, bin_block, after_block);
        }
        // Continue on with the previous flow
        rop.ll.set_insert_point(after_block);

    } else {
        args[maskArgumentIndex] = rop.ll.mask_as_int(rop.ll.current_mask());

        rop.ll.call_function(funcName, args);
    }
    // Clear derivs if necessary
    if (!clear_derivs_of.empty()) {
        llvm::Value* count
            = rop.llvm_load_value(Result, /*deriv=*/0, /*component=*/0,
                                  TypeDesc::UNKNOWN, /*is_uniform*/ false);
        // NOTE: normally BatchedAnalysis would have handled requiring masking, however
        // the clearing of these derivs is conditional based on the count stored in Result.
        // We will always need masking when clearing the results, so choose to just do
        // that here versus complicating BatchedAnalysis
        auto zero_deriv_masking_scope = rop.ll.create_masking_scope(true);
        for (size_t i = 0; i < clear_derivs_of.size(); ++i)
            rop.llvm_zero_derivs(*clear_derivs_of[i], count);
    }
    return true;
}



LLVMGEN(llvm_gen_pointcloud_get)
{
    Opcode& op(rop.inst()->ops()[opnum]);

    OSL_DASSERT(op.nargs() >= 6);

    Symbol& Result    = *rop.opargsym(op, 0);
    Symbol& Filename  = *rop.opargsym(op, 1);
    Symbol& Indices   = *rop.opargsym(op, 2);
    Symbol& Count     = *rop.opargsym(op, 3);
    Symbol& Attr_name = *rop.opargsym(op, 4);
    Symbol& Data      = *rop.opargsym(op, 5);

    OSL_ASSERT(Data.is_varying());
    if (!Attr_name.is_uniform()) {
        rop.shadingcontext()->errorfmt(
            "pointcloud_get named attribute parameter is varying, batched code gen requires a constant or uniform attribute name, called from ({}:{})",
            op.sourcefile(), op.sourceline());
        return false;
    }
    bool requires_binning = Filename.is_varying();

    BatchedBackendLLVM::TempScope temp_scope(rop);

    llvm::Value* count_value
        = rop.llvm_load_value(Count, /*deriv*/ 0, /*component*/ 0,
                              TypeDesc::UNKNOWN, Count.is_uniform());

    constexpr int fileNameArgumentIndex = 1;
    llvm::Value* fileNameVal
        = rop.llvm_load_value(Filename, /*deriv=*/0, /*component=*/0,
                              TypeDesc::UNKNOWN, Filename.is_uniform());

    constexpr int maskArgumentIndex = 8;

    llvm::Value* wcount_val
        = rop.llvm_load_value(Count, /*deriv=*/0, /*component=*/0,
                              TypeDesc::UNKNOWN, /*is_uniform*/ false);
    int element_count             = std::min(Data.typespec().arraylength(),
                                             Indices.typespec().arraylength());
    llvm::Value* welem_count_val  = rop.ll.wide_constant(element_count);
    llvm::Value* wcond            = rop.ll.op_le(welem_count_val, wcount_val);
    llvm::Value* wclampedCountVal = rop.ll.op_select(wcond, welem_count_val,
                                                     wcount_val);
    llvm::Value* loc_clampedCount = rop.getOrAllocateTemp(Count.typespec(),
                                                          /*derivs*/ false,
                                                          /*is_uniform*/ false);
    rop.ll.op_unmasked_store(wclampedCountVal, loc_clampedCount);

    llvm::Value* args[] = {
        rop.sg_void_ptr(),
        Filename.is_uniform() ? fileNameVal : nullptr,
        rop.llvm_load_arg(Indices, false /*derivs*/, false /*is_uniform*/),
        rop.ll.constant(Indices.typespec().arraylength()),
        rop.ll.void_ptr(loc_clampedCount),
        rop.llvm_load_value(Attr_name),
        rop.ll.constant(Data.typespec().simpletype()),
        rop.llvm_void_ptr(Data),
        nullptr /*deferred*/  // mask value
    };
    FuncSpec func_spec("pointcloud_get");
    func_spec.mask();
    const char* funcName = rop.build_name(func_spec);

    llvm::Value* found_mask_value = nullptr;
    if (requires_binning) {
        // do while(remaining)
        llvm::Value* loc_of_remainingMask = rop.getTempMask(
            "lanes remaining to pointcloud_get");
        rop.ll.op_store_mask(rop.ll.current_mask(), loc_of_remainingMask);

        llvm::Value* loc_of_foundMaskVal
            = rop.getOrAllocateTemp(TypeSpec(TypeDesc::INT), false /*derivs*/,
                                    true /*is_uniform*/, false /*forceBool*/,
                                    "found mask value");
        rop.ll.op_store(rop.ll.constant(0), loc_of_foundMaskVal);

        llvm::BasicBlock* bin_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("bin_pointcloud_get")
                             : std::string());
        llvm::BasicBlock* after_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("after_bin_pointcloud_get")
                             : std::string());
        rop.ll.op_branch(bin_block);
        {
            llvm::Value* remainingMask = rop.ll.op_load_mask(
                loc_of_remainingMask);
            llvm::Value* leadLane = rop.ll.op_1st_active_lane_of(remainingMask);
            llvm::Value* lanesMatching = remainingMask;

            if (!Filename.is_uniform()) {
                llvm::Value* scalar_fileName = rop.ll.op_extract(fileNameVal,
                                                                 leadLane);
                args[fileNameArgumentIndex]  = scalar_fileName;
                lanesMatching = rop.ll.op_lanes_that_match_masked(
                    scalar_fileName, fileNameVal, remainingMask);
            }
            OSL_ASSERT(lanesMatching);
            //rop.llvm_print_mask("lanesMatching", lanesMatching);

            args[maskArgumentIndex] = rop.ll.mask_as_int(lanesMatching);

            llvm::Value* binned_found_mask_value
                = rop.ll.call_function(funcName, args);
            found_mask_value = rop.ll.op_load(rop.ll.type_int(),
                                              loc_of_foundMaskVal);
            found_mask_value = rop.ll.op_or(found_mask_value,
                                            binned_found_mask_value);
            rop.ll.op_store(found_mask_value, loc_of_foundMaskVal);


            remainingMask = rop.ll.op_xor(remainingMask, lanesMatching);
            //rop.llvm_print_mask("xor remainingMask,lanesMatchingSplineName", remainingMask);
            rop.ll.op_store_mask(remainingMask, loc_of_remainingMask);

            llvm::Value* int_remainingMask = rop.ll.mask_as_int(remainingMask);
            //rop.llvm_print_mask("remainingMask", remainingMask);
            llvm::Value* cond_more_lanes_to_bin
                = rop.ll.op_ne(int_remainingMask, rop.ll.constant(0));
            rop.ll.op_branch(cond_more_lanes_to_bin, bin_block, after_block);
        }
        // Continue on with the previous flow
        rop.ll.set_insert_point(after_block);
        found_mask_value = rop.ll.op_load(rop.ll.type_int(),
                                          loc_of_foundMaskVal);
    } else {
        args[maskArgumentIndex] = rop.ll.mask_as_int(rop.ll.current_mask());
        found_mask_value        = rop.ll.call_function(funcName, args);
    }


    llvm::Value* found_mask = rop.ll.int_as_mask(found_mask_value);
    llvm::Value* found      = rop.ll.op_bool_to_int(found_mask);
    rop.llvm_store_value(found, Result);
    if (Data.has_derivs()) {
        // NOTE: normally BatchedAnalysis would have handled requiring masking, however
        // the clearing of these derivs is conditional based on the count_value.
        // We will always need masking when clearing the results, so choose to just do
        // that here versus complicating BatchedAnalysis
        auto zero_deriv_masking_scope = rop.ll.create_masking_scope(true);
        rop.llvm_zero_derivs(Data, count_value);
    }

    return true;
}



LLVMGEN(llvm_gen_pointcloud_write)
{
    Opcode& op(rop.inst()->ops()[opnum]);

    OSL_DASSERT(op.nargs() >= 3);
    Symbol& Result   = *rop.opargsym(op, 0);
    Symbol& Filename = *rop.opargsym(op, 1);
    Symbol& Pos      = *rop.opargsym(op, 2);
    OSL_DASSERT(Result.typespec().is_int() && Filename.typespec().is_string()
                && Pos.typespec().is_triple());
    OSL_DASSERT((op.nargs() & 1) && "must have an even number of attribs");

    OSL_ASSERT(Result.is_varying());

    bool requires_binning = Filename.is_varying();

    int nattrs = (op.nargs() - 3) / 2;

    BatchedBackendLLVM::TempScope temp_scope(rop);

    // Generate local space for the names/types/values arrays
    TypeSpec namesArrayType { TypeDesc { TypeDesc::STRING, TypeDesc::SCALAR,
                                         TypeDesc::NOSEMANTICS, nattrs } };
    TypeSpec typesArrayType { TypeDesc { TypeDesc::LONGLONG, TypeDesc::SCALAR,
                                         TypeDesc::NOSEMANTICS, nattrs } };
    TypeSpec valuesArrayType { TypeDesc { TypeDesc::PTR, TypeDesc::SCALAR,
                                          TypeDesc::NOSEMANTICS, nattrs } };
    llvm::Value* names  = rop.ll.op_alloca(rop.ll.type_ustring(), nattrs);
    llvm::Value* types  = rop.ll.op_alloca(rop.ll.type_typedesc(), nattrs);
    llvm::Value* values = rop.ll.op_alloca(rop.ll.type_void_ptr(), nattrs);

    // Fill in the arrays with the params, use helper function because
    // it's a pain to offset things into the array ourselves.
    for (int i = 0; i < nattrs; ++i) {
        Symbol* namesym = rop.opargsym(op, 3 + 2 * i);
        if (!namesym->is_uniform()) {
            rop.shadingcontext()->errorfmt(
                "pointcloud_write named attribute parameter is varying, batched code gen requires a constant or uniform attribute name, called from ({}:{})",
                op.sourcefile(), op.sourceline());
            return false;
        }

        llvm::Value* arrayindex = rop.ll.constant(i);

        // name[i]
        llvm::Value* name_value = rop.llvm_load_value(*namesym);
        rop.llvm_store_value(name_value, names, namesArrayType, /*deriv*/ 0,
                             arrayindex, /*component*/ 0,
                             namesym->is_uniform());

        // type[i]
        Symbol* valsym          = rop.opargsym(op, 3 + 2 * i + 1);
        llvm::Value* type_value = rop.ll.constant(
            valsym->typespec().simpletype());
        rop.llvm_store_value(type_value, types, typesArrayType, /*deriv*/ 0,
                             arrayindex, /*component*/ 0,
                             /*dst_is_uniform*/ true);

        // value[i]
        llvm::Value* attr_value_ptr = rop.llvm_load_arg(*valsym,
                                                        false /*derivs*/,
                                                        false /*is_uniform*/);
        rop.llvm_store_value(attr_value_ptr, values, valuesArrayType,
                             /*deriv*/ 0, arrayindex, /*component*/ 0,
                             /*dst_is_uniform*/ true);
    }

    constexpr int fileNameArgumentIndex = 1;
    llvm::Value* fileNameVal
        = rop.llvm_load_value(Filename, /*deriv=*/0, /*component=*/0,
                              TypeDesc::UNKNOWN, Filename.is_uniform());

    constexpr int maskArgumentIndex = 7;

    llvm::Value* args[] = {
        rop.sg_void_ptr(),                              // shaderglobals pointer
        Filename.is_uniform() ? fileNameVal : nullptr,  // name
        rop.llvm_load_arg(Pos, false /*derivs*/, false /*is_uniform*/),
        rop.ll.constant(nattrs),  // number of attributes
        rop.ll.void_ptr(names),   // attribute names array
        rop.ll.void_ptr(types),   // attribute types array
        rop.ll.void_ptr(values),  // attribute values array
        nullptr /*deferred*/      // mask value
    };

    FuncSpec func_spec("pointcloud_write");
    func_spec.mask();
    const char* funcName = rop.build_name(func_spec);

    llvm::Value* success_mask_value = nullptr;
    if (requires_binning) {
        // do while(remaining)
        llvm::Value* loc_of_remainingMask = rop.getTempMask(
            "lanes remaining to pointcloud_write");
        rop.ll.op_store_mask(rop.ll.current_mask(), loc_of_remainingMask);

        llvm::Value* loc_of_successesMaskVal
            = rop.getOrAllocateTemp(TypeSpec(TypeDesc::INT), false /*derivs*/,
                                    true /*is_uniform*/, false /*forceBool*/,
                                    "found mask value");
        rop.ll.op_store(rop.ll.constant(0), loc_of_successesMaskVal);

        llvm::BasicBlock* bin_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("bin_pointcloud_write")
                             : std::string());
        llvm::BasicBlock* after_block = rop.ll.new_basic_block(
            rop.llvm_debug() ? std::string("after_bin_pointcloud_write")
                             : std::string());
        rop.ll.op_branch(bin_block);
        {
            llvm::Value* remainingMask = rop.ll.op_load_mask(
                loc_of_remainingMask);
            llvm::Value* leadLane = rop.ll.op_1st_active_lane_of(remainingMask);
            llvm::Value* lanesMatching = remainingMask;

            OSL_ASSERT(!Filename.is_uniform());
            llvm::Value* scalar_fileName = rop.ll.op_extract(fileNameVal,
                                                             leadLane);
            args[fileNameArgumentIndex]  = scalar_fileName;
            lanesMatching = rop.ll.op_lanes_that_match_masked(scalar_fileName,
                                                              fileNameVal,
                                                              remainingMask);
            OSL_ASSERT(lanesMatching);
            //rop.llvm_print_mask("lanesMatching", lanesMatching);

            args[maskArgumentIndex] = rop.ll.mask_as_int(lanesMatching);

            llvm::Value* binned_success_mask_value
                = rop.ll.call_function(funcName, args);
            success_mask_value = rop.ll.op_load(rop.ll.type_int(),
                                                loc_of_successesMaskVal);
            success_mask_value = rop.ll.op_or(success_mask_value,
                                              binned_success_mask_value);
            rop.ll.op_store(success_mask_value, loc_of_successesMaskVal);


            remainingMask = rop.ll.op_xor(remainingMask, lanesMatching);
            //rop.llvm_print_mask("xor remainingMask,lanesMatchingSplineName", remainingMask);
            rop.ll.op_store_mask(remainingMask, loc_of_remainingMask);

            llvm::Value* int_remainingMask = rop.ll.mask_as_int(remainingMask);
            //rop.llvm_print_mask("remainingMask", remainingMask);
            llvm::Value* cond_more_lanes_to_bin
                = rop.ll.op_ne(int_remainingMask, rop.ll.constant(0));
            rop.ll.op_branch(cond_more_lanes_to_bin, bin_block, after_block);
        }
        // Continue on with the previous flow
        rop.ll.set_insert_point(after_block);
        success_mask_value = rop.ll.op_load(rop.ll.type_int(),
                                            loc_of_successesMaskVal);
    } else {
        args[maskArgumentIndex] = rop.ll.mask_as_int(rop.ll.current_mask());
        success_mask_value      = rop.ll.call_function(funcName, args);
    }

    llvm::Value* success_mask = rop.ll.int_as_mask(success_mask_value);
    llvm::Value* success      = rop.ll.op_bool_to_int(success_mask);
    rop.llvm_store_value(success, Result);

    return true;
}



LLVMGEN(llvm_gen_isconstant)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 2);
    Symbol& Result(*rop.opargsym(op, 0));
    OSL_DASSERT(Result.typespec().is_int());
    Symbol& A(*rop.opargsym(op, 1));
    llvm::Value* result = rop.ll.constant(A.is_constant() ? 1 : 0);
    if (!Result.is_uniform()) {
        result = rop.ll.widen_value(result);
    }
    rop.llvm_store_value(result, Result);
    return true;
}



LLVMGEN(llvm_gen_functioncall)
{
    //std::cout << "llvm_gen_functioncall" << std::endl;
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_ASSERT(op.nargs() == 1);

    Symbol& functionNameSymbol(*rop.opargsym(op, 0));
    OSL_ASSERT(functionNameSymbol.is_constant());
    OSL_ASSERT(functionNameSymbol.typespec().is_string());
    ustring functionName = functionNameSymbol.get_string();

    int exit_count_before_functioncall = rop.ll.masked_exit_count();
#ifdef __OSL_TRACE_MASKS
    rop.llvm_print_mask("function_call", rop.ll.current_mask());
#endif

    rop.ll.push_function_mask(rop.ll.current_mask());
    llvm::BasicBlock* after_block          = rop.ll.push_function();
    unsigned int op_num_function_starts_at = opnum + 1;
    unsigned int op_num_function_ends_at   = op.jump(0);
    if (rop.ll.debug_is_enabled()) {
        ustring file_name
            = rop.inst()->op(op_num_function_starts_at).sourcefile();
        unsigned int method_line
            = rop.inst()->op(op_num_function_starts_at).sourceline();
        rop.ll.debug_push_inlined_function(functionName, file_name,
                                           method_line);
    }

    // Generate the code for the body of the function
    rop.build_llvm_code(op_num_function_starts_at, op_num_function_ends_at);
    rop.ll.op_branch(after_block);

    // Continue on with the previous flow
    if (rop.ll.debug_is_enabled()) {
        rop.ll.debug_pop_inlined_function();
    }
    rop.ll.pop_function();
    rop.ll.pop_function_mask();

    if (rop.ll.masked_exit_count() > exit_count_before_functioncall) {
        // At some point one or more calls to exit have been made
        // we need to apply that exit mask the the current function scope's return mask
        rop.ll.apply_exit_to_mask_stack();
    }

    return true;
}



LLVMGEN(llvm_gen_functioncall_nr)
{
    OSL_DEV_ONLY(std::cout << "llvm_gen_functioncall_nr" << std::endl);
    OSL_ASSERT(rop.ll.debug_is_enabled()
               && "no return version should only exist when debug is enabled");
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_ASSERT(op.nargs() == 1);

    Symbol& functionNameSymbol(*rop.opargsym(op, 0));
    OSL_ASSERT(functionNameSymbol.is_constant());
    OSL_ASSERT(functionNameSymbol.typespec().is_string());
    ustring functionName = functionNameSymbol.get_string();

    int op_num_function_starts_at = opnum + 1;
    int op_num_function_ends_at   = op.jump(0);
    OSL_ASSERT(
        op.farthest_jump() == op_num_function_ends_at
        && "As we are not doing any branching, we should ensure that the inlined function truly ends at the farthest jump");
    {
        ustring file_name
            = rop.inst()->op(op_num_function_starts_at).sourcefile();
        unsigned int method_line
            = rop.inst()->op(op_num_function_starts_at).sourceline();
        rop.ll.debug_push_inlined_function(functionName, file_name,
                                           method_line);
    }

    // Generate the code for the body of the function
    rop.build_llvm_code(op_num_function_starts_at, op_num_function_ends_at);

    // Continue on with the previous flow
    rop.ll.debug_pop_inlined_function();

    return true;
}



LLVMGEN(llvm_gen_dict_find)
{
    // OSL has two variants of this function:
    //     dict_find (string dict, string query)
    //     dict_find (int nodeID, string query)
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 3);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& Source = *rop.opargsym(
        op, 1);  //Can be nodeID or string; can be uniform or varying
    Symbol& Query = *rop.opargsym(op, 2);  //can be uniform or varying
    OSL_DASSERT(
        Result.typespec().is_int() && Query.typespec().is_string()
        && (Source.typespec().is_int() || Source.typespec().is_string()));

    BatchedBackendLLVM::TempScope temp_scope(rop);
    std::vector<llvm::Value*> args;
    bool op_is_uniform = Source.is_uniform() && Query.is_uniform();

    //Load shader global
    args.push_back(rop.sg_void_ptr());

    if (!op_is_uniform) {
        args.push_back(rop.llvm_void_ptr(Result));
        args.push_back(
            rop.llvm_load_arg(Source, false, false /*op_is_uniform*/));
        args.push_back(
            rop.llvm_load_arg(Query, false, false /*op_is_uniform*/));
    } else {
        args.push_back(rop.llvm_load_value(Source));
        args.push_back(rop.llvm_load_value(Query));
    }

    FuncSpec func_spec("dict_find");
    func_spec.arg(Result, op_is_uniform);
    func_spec.arg(Source, op_is_uniform);
    func_spec.arg(Query, op_is_uniform);

    if (!op_is_uniform) {
        func_spec.mask();
        args.push_back(rop.ll.mask_as_int(rop.ll.current_mask()));
    }

    llvm::Value* ret = rop.ll.call_function(rop.build_name(func_spec), args);

    if (op_is_uniform) {
        if (!Result.is_uniform()) {
            ret = rop.ll.widen_value(ret);
        }
        rop.llvm_store_value(ret, Result);
    }

    return true;
}



LLVMGEN(llvm_gen_dict_next)
{
    // dict_net is very straightforward -- just insert sg ptr as first arg
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 2);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& NodeID = *rop.opargsym(op, 1);
    OSL_DASSERT(Result.typespec().is_int() && NodeID.typespec().is_int());

    bool op_is_uniform = NodeID.is_uniform();

    BatchedBackendLLVM::TempScope temp_scope(rop);
    std::vector<llvm::Value*> args;

    FuncSpec func_spec("dict_next");

    args.push_back(rop.sg_void_ptr());

    if (!op_is_uniform) {
        args.push_back(rop.llvm_void_ptr(Result));
        args.push_back(rop.llvm_load_arg(NodeID, /*derivs=*/false,
                                         false /*op_is_uniform*/));
    } else {
        args.push_back(rop.llvm_load_value(NodeID));
    }

    if (!op_is_uniform) {
        func_spec.mask();
        args.push_back(rop.ll.mask_as_int(rop.ll.current_mask()));
    }
    llvm::Value* ret = rop.ll.call_function(rop.build_name(func_spec), args);

    if (op_is_uniform) {
        if (!Result.is_uniform()) {
            ret = rop.ll.widen_value(ret);
        }
        rop.llvm_store_value(ret, Result);
    }

    return true;
}



LLVMGEN(llvm_gen_dict_value)
{
    // int dict_value (int nodeID, string attribname, output TYPE value)
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 4);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& NodeID = *rop.opargsym(op, 1);
    Symbol& Name   = *rop.opargsym(op, 2);
    Symbol& Value  = *rop.opargsym(op, 3);
    OSL_DASSERT(Result.typespec().is_int() && NodeID.typespec().is_int()
                && Name.typespec().is_string());

    bool op_is_uniform = NodeID.is_uniform() && Name.is_uniform();

    BatchedBackendLLVM::TempScope temp_scope(rop);

    FuncSpec func_spec("dict_value");
    if (op_is_uniform) {
        OSL_ASSERT(!Value.has_derivs());
        BatchedBackendLLVM::TempScope temp_scope(rop);
        llvm::Value* temp_uniform_value = nullptr;
        llvm::Value* uniform_value      = nullptr;
        if (!Value.is_uniform()) {
            temp_uniform_value = rop.getOrAllocateTemp(
                Value.typespec(), Value.has_derivs(), true /*is_uniform*/,
                false /*forceBool*/, "uniform dictionary value");
            uniform_value = rop.ll.void_ptr(temp_uniform_value);
        } else {
            uniform_value = rop.llvm_void_ptr(Value);
        }

        llvm::Value* args[] = { // arg 0: shaderglobals ptr
                                rop.sg_void_ptr(),
                                // arg 1: nodeID
                                rop.llvm_load_value(NodeID),
                                // arg 2: attribute name
                                rop.llvm_load_value(Name),
                                // arg 3: encoded type of Value
                                rop.ll.constant(Value.typespec().simpletype()),
                                // arg 4: pointer to Value
                                uniform_value
        };

        llvm::Value* ret = rop.ll.call_function(rop.build_name(func_spec),
                                                args);

        if (!Value.is_uniform()) {
            // Only broadcast our result if the value lookup succeeded
            // Branch on the condition, to our blocks
            llvm::Value* cond_val             = rop.ll.op_int_to_bool(ret);
            llvm::BasicBlock* broadcast_block = rop.ll.new_basic_block(
                std::string("uniform dict_value broadcast"));
            llvm::BasicBlock* after_block = rop.ll.new_basic_block(
                std::string("after uniform dict_value broadcast"));
            rop.ll.op_branch(cond_val, broadcast_block, after_block);

            rop.ll.set_insert_point(broadcast_block);
            rop.llvm_broadcast_uniform_value_from_mem(temp_uniform_value,
                                                      Value);
            rop.ll.op_branch(after_block);

            rop.ll.set_insert_point(after_block);
        }

        if (!Result.is_uniform()) {
            ret = rop.ll.widen_value(ret);
        }
        rop.llvm_store_value(ret, Result);
    } else {
        func_spec.mask();
        OSL_DASSERT(!Value.is_uniform());
        llvm::Value* args[] = { rop.sg_void_ptr(),
                                rop.llvm_void_ptr(Result),
                                rop.llvm_load_arg(NodeID, false /*derivs*/,
                                                  false /*op_is_uniform*/),
                                rop.llvm_load_arg(Name, false /*derivs*/,
                                                  false /*op_is_uniform*/),
                                rop.ll.constant(Value.typespec().simpletype()),
                                rop.llvm_void_ptr(Value),
                                rop.ll.mask_as_int(rop.ll.current_mask()) };

        rop.ll.call_function(rop.build_name(func_spec), args);
    }
    return true;
}



LLVMGEN(llvm_gen_split)
{
    // int split (string str, output string result[], string sep, int maxsplit)
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() >= 3 && op.nargs() <= 5);
    Symbol& R       = *rop.opargsym(op, 0);
    Symbol& Str     = *rop.opargsym(op, 1);
    Symbol& Results = *rop.opargsym(op, 2);
    OSL_DASSERT(R.typespec().is_int() && Str.typespec().is_string()
                && Results.typespec().is_array()
                && Results.typespec().is_string_based());

    Symbol* optSep      = (op.nargs() >= 4) ? rop.opargsym(op, 3) : nullptr;
    Symbol* optMaxsplit = (op.nargs() >= 5) ? rop.opargsym(op, 4) : nullptr;


    OSL_ASSERT(R.is_uniform() == Results.is_uniform());

    bool op_is_uniform = Str.is_uniform()
                         && ((!optSep) || (*optSep).is_uniform())
                         && ((!optMaxsplit) || (*optMaxsplit).is_uniform());
    bool result_is_uniform = Results.is_uniform();
    OSL_ASSERT(op_is_uniform || (op_is_uniform == result_is_uniform));


    FuncSpec func_spec("split");

    // llvm::Value *args[5];
    std::vector<llvm::Value*> args;
    BatchedBackendLLVM::TempScope temp_scope(rop);

    if (!op_is_uniform) {
        args.push_back(rop.llvm_void_ptr(R));
    }

    args.push_back(rop.llvm_load_arg(Str, false /*derivs*/, op_is_uniform));

    llvm::Value* temp_results_array = nullptr;
    if (op_is_uniform && !result_is_uniform) {
        temp_results_array
            = rop.getOrAllocateTemp(Results.typespec(), false /*derivs*/,
                                    true /*is_uniform*/, false /*forceBool*/,
                                    "uniform split result");
        args.push_back(rop.ll.void_ptr(temp_results_array));
    } else {
        args.push_back(rop.llvm_void_ptr(Results));
    }

    if (optSep) {
        args.push_back(
            rop.llvm_load_arg(*optSep, false /*derivs*/, op_is_uniform));
    } else {
        if (op_is_uniform) {
            args.push_back(rop.ll.constant(""));
        } else {
            llvm::Value* wide_sep      = rop.ll.wide_constant("");
            llvm::Value* temp_wide_sep = rop.getOrAllocateTemp(
                TypeSpec(TypeDesc::STRING), false /*derivs*/,
                false /*op_is_uniform*/, false /*forceBool*/, "wide separator");
            rop.ll.op_unmasked_store(wide_sep, temp_wide_sep);
            args.push_back(rop.ll.void_ptr(temp_wide_sep));
        }
    }

    if (optMaxsplit) {
        OSL_DASSERT(optMaxsplit->typespec().is_int());
        args.push_back(
            rop.llvm_load_arg(*optMaxsplit, false /*derivs*/, op_is_uniform));
    } else {
        if (op_is_uniform) {
            args.push_back(rop.ll.constant(Results.typespec().arraylength()));
        } else {
            llvm::Value* wide_max_split = rop.ll.wide_constant(
                Results.typespec().arraylength());
            llvm::Value* temp_wide_max_split = rop.getOrAllocateTemp(
                TypeSpec(TypeDesc::INT), false /*derivs*/,
                false /*op_is_uniform*/, false /*forceBool*/,
                "wide wide max split");
            rop.ll.op_unmasked_store(wide_max_split, temp_wide_max_split);
            args.push_back(rop.ll.void_ptr(temp_wide_max_split));
        }
    }

    args.push_back(rop.ll.constant(Results.typespec().arraylength()));


    if (!op_is_uniform) {
        func_spec.mask();
        args.push_back(rop.ll.mask_as_int(rop.ll.current_mask()));
        // std::cout<<"LLVM GEN: added on mask"<<std::endl;
    } else {
        func_spec.unbatch();
    }
    llvm::Value* ret = rop.ll.call_function(rop.build_name(func_spec), args);
    if (op_is_uniform && !result_is_uniform) {
        ret = rop.ll.widen_value(ret);

        OSL_ASSERT(temp_results_array);
        llvm::Type* elem_type = rop.ll.llvm_type(
            Results.typespec().elementtype().simpletype());
        for (int ai = 0; ai < Results.typespec().arraylength(); ++ai) {
            llvm::Value* elem_ptr  = rop.ll.GEP(elem_type, temp_results_array,
                                                ai);
            llvm::Value* elem      = rop.ll.op_load(elem_type, elem_ptr);
            llvm::Value* wide_elem = rop.ll.widen_value(elem);
            rop.llvm_store_value(wide_elem, Results, 0 /*deriv*/,
                                 rop.ll.constant(ai) /*arrayindex*/,
                                 0 /* component*/);
        }
    }
    if (op_is_uniform) {
        rop.llvm_store_value(ret, R);
    }
    return true;
}



LLVMGEN(llvm_gen_raytype)
{
    // int raytype (string name)
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_DASSERT(op.nargs() == 2);
    Symbol& Result = *rop.opargsym(op, 0);
    Symbol& Name   = *rop.opargsym(op, 1);

    bool result_is_uniform = Result.is_uniform();
    bool op_is_uniform     = Name.is_uniform();

    llvm::Value* sg = rop.sg_void_ptr();
    if (Name.is_constant()) {
        // We can statically determine the bit pattern
        ustring name = Name.get_string();
        llvm::Value* args[]
            = { sg, rop.ll.constant(rop.shadingsys().raytype_bit(name)) };

        llvm::Value* ret = rop.ll.call_function(rop.build_name("raytype_bit"),
                                                args);

        if (!result_is_uniform) {
            ret = rop.ll.widen_value(ret);
        }
        rop.llvm_store_value(ret, Result);
    } else {
        FuncSpec func_spec("raytype_name");
        // No way to know which name is being asked for
        if (op_is_uniform) {
            llvm::Value* args[] = { sg, rop.llvm_get_pointer(Name) };
            llvm::Value* ret = rop.ll.call_function(rop.build_name(func_spec),
                                                    args);
            if (!result_is_uniform) {
                ret = rop.ll.widen_value(ret);
            }
            rop.llvm_store_value(ret, Result);
        } else {
            func_spec.mask();
            OSL_ASSERT(!result_is_uniform);
            llvm::Value* args[] = { sg, rop.llvm_void_ptr(Result),
                                    rop.llvm_void_ptr(Name),
                                    rop.ll.mask_as_int(rop.ll.current_mask()) };
            rop.ll.call_function(rop.build_name(func_spec), args);
        }
    }
    return true;
}



// color blackbody (float temperatureK)
// color wavelength_color (float wavelength_nm)  // same function signature
LLVMGEN(llvm_gen_blackbody)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_ASSERT(op.nargs() == 2);
    Symbol& Result(*rop.opargsym(op, 0));
    Symbol& Temperature(*rop.opargsym(op, 1));
    OSL_ASSERT(Result.typespec().is_triple()
               && Temperature.typespec().is_float());

    bool result_is_uniform = Result.is_uniform();
    bool op_is_uniform     = Temperature.is_uniform();

    std::vector<llvm::Value*> args;
    args.push_back(rop.sg_void_ptr());

    FuncSpec func_spec(op.opname().c_str());

    BatchedBackendLLVM::TempScope temp_scope(rop);
    llvm::Value* u_result = nullptr;

    if (op_is_uniform && !result_is_uniform) {
        u_result = rop.getOrAllocateTemp(Result.typespec(), false /*derivs()*/,
                                         true /*is_uniform*/,
                                         false /*forceBool*/,
                                         "blackbody temp result");
        args.push_back(rop.ll.void_ptr(u_result));
    } else {
        args.push_back(rop.llvm_void_ptr(Result));
    }
    func_spec.arg(Result, false /* no derivs */, op_is_uniform);
    func_spec.arg(Temperature, false /* no derivs */, op_is_uniform);
    args.push_back(op_is_uniform ? rop.llvm_load_value(Temperature)
                                 : rop.llvm_void_ptr(Temperature));

    if (!op_is_uniform) {
        func_spec.mask();
        args.push_back(rop.ll.mask_as_int(rop.ll.current_mask()));
        // tack on additional mask argument
    }

    rop.ll.call_function(rop.build_name(func_spec), args);

    if (op_is_uniform && !result_is_uniform) {
        rop.llvm_broadcast_uniform_value_from_mem(u_result, Result,
                                                  true /* ignore_derivs */);
    }

    // Punt, zero out derivs.
    // FIXME -- only of some day, someone truly needs blackbody() to
    // correctly return derivs with spatially-varying temperature.
    if (Result.has_derivs())
        rop.llvm_zero_derivs(Result);

    return true;
}



// float luminance (color c)
LLVMGEN(llvm_gen_luminance)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_ASSERT(op.nargs() == 2);
    Symbol& Result(*rop.opargsym(op, 0));
    Symbol& C(*rop.opargsym(op, 1));
    OSL_ASSERT(Result.typespec().is_float() && C.typespec().is_triple());

    // luminance = red * luminance_scale.red + green * luminance_scale.green + blue * luminance_scale.blue;

    // Although color systems can be changed via a ShadingSystem attribute,
    // any change of attributes should cause/require a rebuild of the shaders
    // So we will emit the luminance scale as comple time constants
    // and emit the simple math, vs. incur the overhead of a function call
    // TODO: same logic could be applied to the non batched gen

    Color3 luminance_scale = rop.shadingsys().colorsystem().luminance_scale();
    //
    bool result_is_uniform = Result.is_uniform();
    bool op_is_uniform     = C.is_uniform();

    llvm::Value* red_scale   = op_is_uniform
                                   ? rop.ll.constant(luminance_scale.x)
                                   : rop.ll.wide_constant(luminance_scale.x);
    llvm::Value* green_scale = op_is_uniform
                                   ? rop.ll.constant(luminance_scale.y)
                                   : rop.ll.wide_constant(luminance_scale.y);
    llvm::Value* blue_scale  = op_is_uniform
                                   ? rop.ll.constant(luminance_scale.z)
                                   : rop.ll.wide_constant(luminance_scale.z);

    for (int d = 0; d < 3; ++d) {  // deriv
        llvm::Value* red   = rop.llvm_load_value(C, d, 0, TypeDesc::UNKNOWN,
                                                 op_is_uniform);
        llvm::Value* green = rop.llvm_load_value(C, d, 1, TypeDesc::UNKNOWN,
                                                 op_is_uniform);
        llvm::Value* blue  = rop.llvm_load_value(C, d, 2, TypeDesc::UNKNOWN,
                                                 op_is_uniform);

        llvm::Value* scaled_red   = rop.ll.op_mul(red_scale, red);
        llvm::Value* scaled_green = rop.ll.op_mul(green_scale, green);
        llvm::Value* scaled_blue  = rop.ll.op_mul(blue_scale, blue);

        llvm::Value* result
            = rop.ll.op_add(rop.ll.op_add(scaled_red, scaled_green),
                            scaled_blue);

        OSL_ASSERT(op_is_uniform || !result_is_uniform);
        if (op_is_uniform && !result_is_uniform) {
            result = rop.ll.widen_value(result);
        }

        rop.llvm_store_value(result, Result, d);
        if (!Result.has_derivs())  // skip the derivs if we don't need them
            break;
    }
    return true;
}



LLVMGEN(llvm_gen_return)
{
    Opcode& op(rop.inst()->ops()[opnum]);
    OSL_ASSERT(op.nargs() == 0);

    // mask stack is never empty as we keep one around to handle early returns
    if (rop.ll.has_masked_return_block()) {
        // Rely on front end dead code elimination to ensure no instructions
        // exist in the same scope after a return/exit.
        // Do not bother updating the mask stack for the current scope
        if (op.opname() == Strings::op_exit) {
            rop.ll.op_masked_exit();
        } else {
            rop.ll.op_masked_return();
        }
        OSL_DEV_ONLY(std::cout << " branching to rop.ll.masked_return_block()");
        rop.ll.op_branch(rop.ll.masked_return_block());
    } else {
        if (op.opname() == Strings::op_exit) {
            OSL_DEV_ONLY(std::cout
                         << " branching to rop.llvm_exit_instance_block()");
            // If it's a real "exit", totally jump out of the shader instance.
            // The exit instance block will be created if it doesn't yet exist.
            rop.ll.op_branch(rop.llvm_exit_instance_block());
        } else {
            OSL_DEV_ONLY(std::cout << " branching to rop.ll.return_block()");
            // If it's a "return", jump to the exit point of the function.
            rop.ll.op_branch(rop.ll.return_block());
        }
    }
    // Need an unreachable block for any instuctions after the return
    // or exit
    llvm::BasicBlock* next_block = rop.ll.new_basic_block(
        rop.llvm_debug() ? std::string("after ") + op.opname().c_str()
                         : std::string());
    rop.ll.set_insert_point(next_block);

    return true;
}



LLVMGEN(llvm_gen_end)
{
    // Dummy routine needed only for the op_descriptor table
    return false;
}


};  // namespace pvt
OSL_NAMESPACE_EXIT
